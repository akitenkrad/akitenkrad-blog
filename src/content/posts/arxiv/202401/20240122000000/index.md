---
draft: false
title: "arXiv @ 2024.01.22"
date: 2024-01-22
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2024"]
menu:
  sidebar:
    name: "arXiv @ 2024.01.22"
    identifier: arxiv_20240122
    parent: 202401_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CV (15)](#cscv-15)
- [cs.CY (2)](#cscy-2)
- [cs.LG (7)](#cslg-7)
- [cs.CL (11)](#cscl-11)
- [cs.HC (3)](#cshc-3)
- [cs.MA (1)](#csma-1)
- [cs.IR (2)](#csir-2)
- [eess.IV (1)](#eessiv-1)
- [cs.IT (1)](#csit-1)
- [cs.RO (1)](#csro-1)
- [cs.DC (1)](#csdc-1)
- [cs.DB (1)](#csdb-1)
- [cs.SE (1)](#csse-1)
- [cs.CR (5)](#cscr-5)
- [eess.SY (1)](#eesssy-1)
- [cs.DL (1)](#csdl-1)
- [cs.SD (1)](#cssd-1)
- [cs.AI (1)](#csai-1)

## cs.CV (15)



### (1/56) Prompting Large Vision-Language Models for Compositional Reasoning (Timothy Ossowski et al., 2024)

{{<citation>}}

Timothy Ossowski, Ming Jiang, Junjie Hu. (2024)  
**Prompting Large Vision-Language Models for Compositional Reasoning**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: GPT, GPT-4, Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2401.11337v1)  

---


**ABSTRACT**  
Vision-language models such as CLIP have shown impressive capabilities in encoding texts and images into aligned embeddings, enabling the retrieval of multimodal data in a shared embedding space. However, these embedding-based models still face challenges in effectively matching images and texts with similar visio-linguistic compositionality, as evidenced by their performance on the recent Winoground dataset. In this paper, we argue that this limitation stems from two factors: the use of single vector representations for complex multimodal data, and the absence of step-by-step reasoning in these embedding-based methods. To address this issue, we make an exploratory step using a novel generative method that prompts large vision-language models (e.g., GPT-4) to depict images and perform compositional reasoning. Our method outperforms other embedding-based methods on the Winoground dataset, and obtains further improvement of up to 10% accuracy when enhanced with the optimal description.

{{</citation>}}


### (2/56) Weakly-Supervised Semantic Segmentation of Circular-Scan, Synthetic-Aperture-Sonar Imagery (Isaac J. Sledge et al., 2024)

{{<citation>}}

Isaac J. Sledge, Dominic M. Byrne, Jonathan L. King, Steven H. Ostertag, Denton L. Woods, James L. Prater, Jermaine L. Kennedy, Timothy M. Marston, Jose C. Principe. (2024)  
**Weakly-Supervised Semantic Segmentation of Circular-Scan, Synthetic-Aperture-Sonar Imagery**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV, eess-IV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2401.11313v1)  

---


**ABSTRACT**  
We propose a weakly-supervised framework for the semantic segmentation of circular-scan synthetic-aperture-sonar (CSAS) imagery. The first part of our framework is trained in a supervised manner, on image-level labels, to uncover a set of semi-sparse, spatially-discriminative regions in each image. The classification uncertainty of each region is then evaluated. Those areas with the lowest uncertainties are then chosen to be weakly labeled segmentation seeds, at the pixel level, for the second part of the framework. Each of the seed extents are progressively resized according to an unsupervised, information-theoretic loss with structured-prediction regularizers. This reshaping process uses multi-scale, adaptively-weighted features to delineate class-specific transitions in local image content. Content-addressable memories are inserted at various parts of our framework so that it can leverage features from previously seen images to improve segmentation performance for related images.   We evaluate our weakly-supervised framework using real-world CSAS imagery that contains over ten seafloor classes and ten target classes. We show that our framework performs comparably to nine fully-supervised deep networks. Our framework also outperforms eleven of the best weakly-supervised deep networks. We achieve state-of-the-art performance when pre-training on natural imagery. The average absolute performance gap to the next-best weakly-supervised network is well over ten percent for both natural imagery and sonar imagery. This gap is found to be statistically significant.

{{</citation>}}


### (3/56) A Novel Benchmark for Few-Shot Semantic Segmentation in the Era of Foundation Models (Reda Bensaid et al., 2024)

{{<citation>}}

Reda Bensaid, Vincent Gripon, François Leduc-Primeau, Lukas Mauch, Ghouthi Boukli Hacene, Fabien Cardinaux. (2024)  
**A Novel Benchmark for Few-Shot Semantic Segmentation in the Era of Foundation Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Few-Shot, Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2401.11311v1)  

---


**ABSTRACT**  
In recent years, the rapid evolution of computer vision has seen the emergence of various vision foundation models, each tailored to specific data types and tasks. While large language models often share a common pretext task, the diversity in vision foundation models arises from their varying training objectives. In this study, we delve into the quest for identifying the most effective vision foundation models for few-shot semantic segmentation, a critical task in computer vision. Specifically, we conduct a comprehensive comparative analysis of four prominent foundation models: DINO V2, Segment Anything, CLIP, Masked AutoEncoders, and a straightforward ResNet50 pre-trained on the COCO dataset. Our investigation focuses on their adaptability to new semantic segmentation tasks, leveraging only a limited number of segmented images. Our experimental findings reveal that DINO V2 consistently outperforms the other considered foundation models across a diverse range of datasets and adaptation methods. This outcome underscores DINO V2's superior capability to adapt to semantic segmentation tasks compared to its counterparts. Furthermore, our observations indicate that various adapter methods exhibit similar performance, emphasizing the paramount importance of selecting a robust feature extractor over the intricacies of the adaptation technique itself. This insight sheds light on the critical role of feature extraction in the context of few-shot semantic segmentation. This research not only contributes valuable insights into the comparative performance of vision foundation models in the realm of few-shot semantic segmentation but also highlights the significance of a robust feature extractor in this domain.

{{</citation>}}


### (4/56) Evaluating Driver Readiness in Conditionally Automated Vehicles from Eye-Tracking Data and Head Pose (Mostafa Kazemi et al., 2024)

{{<citation>}}

Mostafa Kazemi, Mahdi Rezaei, Mohsen Azarmi. (2024)  
**Evaluating Driver Readiness in Conditionally Automated Vehicles from Eye-Tracking Data and Head Pose**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-NE, cs.CV  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2401.11284v1)  

---


**ABSTRACT**  
As automated driving technology advances, the role of the driver to resume control of the vehicle in conditionally automated vehicles becomes increasingly critical. In the SAE Level 3 or partly automated vehicles, the driver needs to be available and ready to intervene when necessary. This makes it essential to evaluate their readiness accurately. This article presents a comprehensive analysis of driver readiness assessment by combining head pose features and eye-tracking data. The study explores the effectiveness of predictive models in evaluating driver readiness, addressing the challenges of dataset limitations and limited ground truth labels. Machine learning techniques, including LSTM architectures, are utilised to model driver readiness based on the Spatio-temporal status of the driver's head pose and eye gaze. The experiments in this article revealed that a Bidirectional LSTM architecture, combining both feature sets, achieves a mean absolute error of 0.363 on the DMD dataset, demonstrating superior performance in assessing driver readiness. The modular architecture of the proposed model also allows the integration of additional driver-specific features, such as steering wheel activity, enhancing its adaptability and real-world applicability.

{{</citation>}}


### (5/56) LRP-QViT: Mixed-Precision Vision Transformer Quantization via Layer-wise Relevance Propagation (Navin Ranjan et al., 2024)

{{<citation>}}

Navin Ranjan, Andreas Savakis. (2024)  
**LRP-QViT: Mixed-Precision Vision Transformer Quantization via Layer-wise Relevance Propagation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Quantization, Transformer  
[Paper Link](http://arxiv.org/abs/2401.11243v1)  

---


**ABSTRACT**  
Vision transformers (ViTs) have demonstrated remarkable performance across various visual tasks. However, ViT models suffer from substantial computational and memory requirements, making it challenging to deploy them on resource-constrained platforms. Quantization is a popular approach for reducing model size, but most studies mainly focus on equal bit-width quantization for the entire network, resulting in sub-optimal solutions. While there are few works on mixed precision quantization (MPQ) for ViTs, they typically rely on search space-based methods or employ mixed precision arbitrarily. In this paper, we introduce LRP-QViT, an explainability-based method for assigning mixed-precision bit allocations to different layers based on their importance during classification. Specifically, to measure the contribution score of each layer in predicting the target class, we employ the Layer-wise Relevance Propagation (LRP) method. LRP assigns local relevance at the output layer and propagates it through all layers, distributing the relevance until it reaches the input layers. These relevance scores serve as indicators for computing the layer contribution score. Additionally, we have introduced a clipped channel-wise quantization aimed at eliminating outliers from post-LayerNorm activations to alleviate severe inter-channel variations. To validate and assess our approach, we employ LRP-QViT across ViT, DeiT, and Swin transformer models on various datasets. Our experimental findings demonstrate that both our fixed-bit and mixed-bit post-training quantization methods surpass existing models in the context of 4-bit and 6-bit quantization.

{{</citation>}}


### (6/56) Unifying Visual and Vision-Language Tracking via Contrastive Learning (Yinchao Ma et al., 2024)

{{<citation>}}

Yinchao Ma, Yuyang Tang, Wenfei Yang, Tianzhu Zhang, Jinpeng Zhang, Mengxue Kang. (2024)  
**Unifying Visual and Vision-Language Tracking via Contrastive Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2401.11228v1)  

---


**ABSTRACT**  
Single object tracking aims to locate the target object in a video sequence according to the state specified by different modal references, including the initial bounding box (BBOX), natural language (NL), or both (NL+BBOX). Due to the gap between different modalities, most existing trackers are designed for single or partial of these reference settings and overspecialize on the specific modality. Differently, we present a unified tracker called UVLTrack, which can simultaneously handle all three reference settings (BBOX, NL, NL+BBOX) with the same parameters. The proposed UVLTrack enjoys several merits. First, we design a modality-unified feature extractor for joint visual and language feature learning and propose a multi-modal contrastive loss to align the visual and language features into a unified semantic space. Second, a modality-adaptive box head is proposed, which makes full use of the target reference to mine ever-changing scenario features dynamically from video contexts and distinguish the target in a contrastive way, enabling robust performance in different reference settings. Extensive experimental results demonstrate that UVLTrack achieves promising performance on seven visual tracking datasets, three vision-language tracking datasets, and three visual grounding datasets. Codes and models will be open-sourced at https://github.com/OpenSpaceAI/UVLTrack.

{{</citation>}}


### (7/56) Pixel-Wise Recognition for Holistic Surgical Scene Understanding (Nicolás Ayobi et al., 2024)

{{<citation>}}

Nicolás Ayobi, Santiago Rodríguez, Alejandra Pérez, Isabela Hernández, Nicolás Aparicio, Eugénie Dessevres, Sebastián Peña, Jessica Santander, Juan Ignacio Caicedo, Nicolás Fernández, Pablo Arbeláez. (2024)  
**Pixel-Wise Recognition for Holistic Surgical Scene Understanding**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.11174v1)  

---


**ABSTRACT**  
This paper presents the Holistic and Multi-Granular Surgical Scene Understanding of Prostatectomies (GraSP) dataset, a curated benchmark that models surgical scene understanding as a hierarchy of complementary tasks with varying levels of granularity. Our approach enables a multi-level comprehension of surgical activities, encompassing long-term tasks such as surgical phases and steps recognition and short-term tasks including surgical instrument segmentation and atomic visual actions detection. To exploit our proposed benchmark, we introduce the Transformers for Actions, Phases, Steps, and Instrument Segmentation (TAPIS) model, a general architecture that combines a global video feature extractor with localized region proposals from an instrument segmentation model to tackle the multi-granularity of our benchmark. Through extensive experimentation, we demonstrate the impact of including segmentation annotations in short-term recognition tasks, highlight the varying granularity requirements of each task, and establish TAPIS's superiority over previously proposed baselines and conventional CNN-based models. Additionally, we validate the robustness of our method across multiple public benchmarks, confirming the reliability and applicability of our dataset. This work represents a significant step forward in Endoscopic Vision, offering a novel and comprehensive framework for future research towards a holistic understanding of surgical procedures.

{{</citation>}}


### (8/56) Inducing High Energy-Latency of Large Vision-Language Models with Verbose Images (Kuofeng Gao et al., 2024)

{{<citation>}}

Kuofeng Gao, Yang Bai, Jindong Gu, Shu-Tao Xia, Philip Torr, Zhifeng Li, Wei Liu. (2024)  
**Inducing High Energy-Latency of Large Vision-Language Models with Verbose Images**  

---
Primary Category: cs.CV  
Categories: cs-CR, cs-CV, cs.CV  
Keywords: GPT, GPT-4, ImageNet, Language Model  
[Paper Link](http://arxiv.org/abs/2401.11170v1)  

---


**ABSTRACT**  
Large vision-language models (VLMs) such as GPT-4 have achieved exceptional performance across various multi-modal tasks. However, the deployment of VLMs necessitates substantial energy consumption and computational resources. Once attackers maliciously induce high energy consumption and latency time (energy-latency cost) during inference of VLMs, it will exhaust computational resources. In this paper, we explore this attack surface about availability of VLMs and aim to induce high energy-latency cost during inference of VLMs. We find that high energy-latency cost during inference of VLMs can be manipulated by maximizing the length of generated sequences. To this end, we propose verbose images, with the goal of crafting an imperceptible perturbation to induce VLMs to generate long sentences during inference. Concretely, we design three loss objectives. First, a loss is proposed to delay the occurrence of end-of-sequence (EOS) token, where EOS token is a signal for VLMs to stop generating further tokens. Moreover, an uncertainty loss and a token diversity loss are proposed to increase the uncertainty over each generated token and the diversity among all tokens of the whole generated sequence, respectively, which can break output dependency at token-level and sequence-level. Furthermore, a temporal weight adjustment algorithm is proposed, which can effectively balance these losses. Extensive experiments demonstrate that our verbose images can increase the length of generated sequences by 7.87 times and 8.56 times compared to original images on MS-COCO and ImageNet datasets, which presents potential challenges for various applications. Our code is available at https://github.com/KuofengGao/Verbose_Images.

{{</citation>}}


### (9/56) Large-scale Reinforcement Learning for Diffusion Models (Yinan Zhang et al., 2024)

{{<citation>}}

Yinan Zhang, Eric Tzeng, Yilun Du, Dmitry Kislyuk. (2024)  
**Large-scale Reinforcement Learning for Diffusion Models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.12244v1)  

---


**ABSTRACT**  
Text-to-image diffusion models are a class of deep generative models that have demonstrated an impressive capacity for high-quality image generation. However, these models are susceptible to implicit biases that arise from web-scale text-image training pairs and may inaccurately model aspects of images we care about. This can result in suboptimal samples, model bias, and images that do not align with human ethics and preferences. In this paper, we present an effective scalable algorithm to improve diffusion models using Reinforcement Learning (RL) across a diverse set of reward functions, such as human preference, compositionality, and fairness over millions of images. We illustrate how our approach substantially outperforms existing methods for aligning diffusion models with human preferences. We further illustrate how this substantially improves pretrained Stable Diffusion (SD) models, generating samples that are preferred by humans 80.3% of the time over those from the base SD model while simultaneously improving both the composition and diversity of generated samples.

{{</citation>}}


### (10/56) Stability Plasticity Decoupled Fine-tuning For Few-shot end-to-end Object Detection (Yuantao Yin et al., 2024)

{{<citation>}}

Yuantao Yin, Ping Yin. (2024)  
**Stability Plasticity Decoupled Fine-tuning For Few-shot end-to-end Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2401.11140v1)  

---


**ABSTRACT**  
Few-shot object detection(FSOD) aims to design methods to adapt object detectors efficiently with only few annotated samples. Fine-tuning has been shown to be an effective and practical approach. However, previous works often take the classical base-novel two stage fine-tuning procedure but ignore the implicit stability-plasticity contradiction among different modules. Specifically, the random re-initialized classifiers need more plasticity to adapt to novel samples. The other modules inheriting pre-trained weights demand more stability to reserve their class-agnostic knowledge. Regular fine-tuning which couples the optimization of these two parts hurts the model generalization in FSOD scenarios. In this paper, we find that this problem is prominent in the end-to-end object detector Sparse R-CNN for its multi-classifier cascaded architecture. We propose to mitigate this contradiction by a new three-stage fine-tuning procedure by introducing an addtional plasticity classifier fine-tuning(PCF) stage. We further design the multi-source ensemble(ME) technique to enhance the generalization of the model in the final fine-tuning stage. Extensive experiments verify that our method is effective in regularizing Sparse R-CNN, outperforming previous methods in the FSOD benchmark.

{{</citation>}}


### (11/56) Uncertainty-aware Bridge based Mobile-Former Network for Event-based Pattern Recognition (Haoxiang Yang et al., 2024)

{{<citation>}}

Haoxiang Yang, Chengguo Yuan, Yabin Zhu, Lan Chen, Xiao Wang, Jin Tang. (2024)  
**Uncertainty-aware Bridge based Mobile-Former Network for Event-based Pattern Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.11123v1)  

---


**ABSTRACT**  
The mainstream human activity recognition (HAR) algorithms are developed based on RGB cameras, which are easily influenced by low-quality images (e.g., low illumination, motion blur). Meanwhile, the privacy protection issue caused by ultra-high definition (HD) RGB cameras aroused more and more people's attention. Inspired by the success of event cameras which perform better on high dynamic range, no motion blur, and low energy consumption, we propose to recognize human actions based on the event stream. We propose a lightweight uncertainty-aware information propagation based Mobile-Former network for efficient pattern recognition, which aggregates the MobileNet and Transformer network effectively. Specifically, we first embed the event images using a stem network into feature representations, then, feed them into uncertainty-aware Mobile-Former blocks for local and global feature learning and fusion. Finally, the features from MobileNet and Transformer branches are concatenated for pattern recognition. Extensive experiments on multiple event-based recognition datasets fully validated the effectiveness of our model. The source code of this work will be released at https://github.com/Event-AHU/Uncertainty_aware_MobileFormer.

{{</citation>}}


### (12/56) Spatial Structure Constraints for Weakly Supervised Semantic Segmentation (Tao Chen et al., 2024)

{{<citation>}}

Tao Chen, Yazhou Yao, Xingguo Huang, Zechao Li, Liqiang Nie, Jinhui Tang. (2024)  
**Spatial Structure Constraints for Weakly Supervised Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2401.11122v1)  

---


**ABSTRACT**  
The image-level label has prevailed in weakly supervised semantic segmentation tasks due to its easy availability. Since image-level labels can only indicate the existence or absence of specific categories of objects, visualization-based techniques have been widely adopted to provide object location clues. Considering class activation maps (CAMs) can only locate the most discriminative part of objects, recent approaches usually adopt an expansion strategy to enlarge the activation area for more integral object localization. However, without proper constraints, the expanded activation will easily intrude into the background region. In this paper, we propose spatial structure constraints (SSC) for weakly supervised semantic segmentation to alleviate the unwanted object over-activation of attention expansion. Specifically, we propose a CAM-driven reconstruction module to directly reconstruct the input image from deep CAM features, which constrains the diffusion of last-layer object attention by preserving the coarse spatial structure of the image content. Moreover, we propose an activation self-modulation module to refine CAMs with finer spatial structure details by enhancing regional consistency. Without external saliency models to provide background clues, our approach achieves 72.7\% and 47.0\% mIoU on the PASCAL VOC 2012 and COCO datasets, respectively, demonstrating the superiority of our proposed approach.

{{</citation>}}


### (13/56) DengueNet: Dengue Prediction using Spatiotemporal Satellite Imagery for Resource-Limited Countries (Kuan-Ting Kuo et al., 2024)

{{<citation>}}

Kuan-Ting Kuo, Dana Moukheiber, Sebastian Cajas Ordonez, David Restrepo, Atika Rahman Paddo, Tsung-Yu Chen, Lama Moukheiber, Mira Moukheiber, Sulaiman Moukheiber, Saptarshi Purkayastha, Po-Chih Kuo, Leo Anthony Celi. (2024)  
**DengueNet: Dengue Prediction using Spatiotemporal Satellite Imagery for Resource-Limited Countries**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.11114v2)  

---


**ABSTRACT**  
Dengue fever presents a substantial challenge in developing countries where sanitation infrastructure is inadequate. The absence of comprehensive healthcare systems exacerbates the severity of dengue infections, potentially leading to life-threatening circumstances. Rapid response to dengue outbreaks is also challenging due to limited information exchange and integration. While timely dengue outbreak forecasts have the potential to prevent such outbreaks, the majority of dengue prediction studies have predominantly relied on data that impose significant burdens on individual countries for collection. In this study, our aim is to improve health equity in resource-constrained countries by exploring the effectiveness of high-resolution satellite imagery as a nontraditional and readily accessible data source. By leveraging the wealth of publicly available and easily obtainable satellite imagery, we present a scalable satellite extraction framework based on Sentinel Hub, a cloud-based computing platform. Furthermore, we introduce DengueNet, an innovative architecture that combines Vision Transformer, Radiomics, and Long Short-term Memory to extract and integrate spatiotemporal features from satellite images. This enables dengue predictions on an epi-week basis. To evaluate the effectiveness of our proposed method, we conducted experiments on five municipalities in Colombia. We utilized a dataset comprising 780 high-resolution Sentinel-2 satellite images for training and evaluation. The performance of DengueNet was assessed using the mean absolute error (MAE) metric. Across the five municipalities, DengueNet achieved an average MAE of 43.92. Our findings strongly support the efficacy of satellite imagery as a valuable resource for dengue prediction, particularly in informing public health policies within countries where manually collected data is scarce and dengue virus prevalence is severe.

{{</citation>}}


### (14/56) VONet: Unsupervised Video Object Learning With Parallel U-Net Attention and Object-wise Sequential VAE (Haonan Yu et al., 2024)

{{<citation>}}

Haonan Yu, Wei Xu. (2024)  
**VONet: Unsupervised Video Object Learning With Parallel U-Net Attention and Object-wise Sequential VAE**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2401.11110v1)  

---


**ABSTRACT**  
Unsupervised video object learning seeks to decompose video scenes into structural object representations without any supervision from depth, optical flow, or segmentation. We present VONet, an innovative approach that is inspired by MONet. While utilizing a U-Net architecture, VONet employs an efficient and effective parallel attention inference process, generating attention masks for all slots simultaneously. Additionally, to enhance the temporal consistency of each mask across consecutive video frames, VONet develops an object-wise sequential VAE framework. The integration of these innovative encoder-side techniques, in conjunction with an expressive transformer-based decoder, establishes VONet as the leading unsupervised method for object learning across five MOVI datasets, encompassing videos of diverse complexities. Code is available at https://github.com/hnyu/vonet.

{{</citation>}}


### (15/56) Adaptive Global-Local Representation Learning and Selection for Cross-Domain Facial Expression Recognition (Yuefang Gao et al., 2024)

{{<citation>}}

Yuefang Gao, Yuhao Xie, Zeke Zexi Hu, Tianshui Chen, Liang Lin. (2024)  
**Adaptive Global-Local Representation Learning and Selection for Cross-Domain Facial Expression Recognition**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2401.11085v1)  

---


**ABSTRACT**  
Domain shift poses a significant challenge in Cross-Domain Facial Expression Recognition (CD-FER) due to the distribution variation across different domains. Current works mainly focus on learning domain-invariant features through global feature adaptation, while neglecting the transferability of local features. Additionally, these methods lack discriminative supervision during training on target datasets, resulting in deteriorated feature representation in target domain. To address these limitations, we propose an Adaptive Global-Local Representation Learning and Selection (AGLRLS) framework. The framework incorporates global-local adversarial adaptation and semantic-aware pseudo label generation to enhance the learning of domain-invariant and discriminative feature during training. Meanwhile, a global-local prediction consistency learning is introduced to improve classification results during inference. Specifically, the framework consists of separate global-local adversarial learning modules that learn domain-invariant global and local features independently. We also design a semantic-aware pseudo label generation module, which computes semantic labels based on global and local features. Moreover, a novel dynamic threshold strategy is employed to learn the optimal thresholds by leveraging independent prediction of global and local features, ensuring filtering out the unreliable pseudo labels while retaining reliable ones. These labels are utilized for model optimization through the adversarial learning process in an end-to-end manner. During inference, a global-local prediction consistency module is developed to automatically learn an optimal result from multiple predictions. We conduct comprehensive experiments and analysis based on a fair evaluation benchmark. The results demonstrate that the proposed framework outperforms the current competing methods by a substantial margin.

{{</citation>}}


## cs.CY (2)



### (16/56) Deception and Manipulation in Generative AI (Christian Tarsney, 2024)

{{<citation>}}

Christian Tarsney. (2024)  
**Deception and Manipulation in Generative AI**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2401.11335v1)  

---


**ABSTRACT**  
Large language models now possess human-level linguistic abilities in many contexts. This raises the concern that they can be used to deceive and manipulate on unprecedented scales, for instance spreading political misinformation on social media. In future, agentic AI systems might also deceive and manipulate humans for their own ends. In this paper, first, I argue that AI-generated content should be subject to stricter standards against deception and manipulation than we ordinarily apply to humans. Second, I offer new characterizations of AI deception and manipulation meant to support such standards, according to which a statement is deceptive (manipulative) if it leads human addressees away from the beliefs (choices) they would endorse under ``semi-ideal'' conditions. Third, I propose two measures to guard against AI deception and manipulation, inspired by this characterization: "extreme transparency" requirements for AI-generated content and defensive systems that, among other things, annotate AI-generated statements with contextualizing information. Finally, I consider to what extent these measures can protect against deceptive behavior in future, agentic AIs, and argue that non-agentic defensive systems can provide an important layer of defense even against more powerful agentic systems.

{{</citation>}}


### (17/56) Evaluating if trust and personal information privacy concerns are barriers to using health insurance that explicitly utilizes AI (Alex Zarifis et al., 2024)

{{<citation>}}

Alex Zarifis, Peter Kawalek, Aida Azadegan. (2024)  
**Evaluating if trust and personal information privacy concerns are barriers to using health insurance that explicitly utilizes AI**  

---
Primary Category: cs.CY  
Categories: H-0; A-0; K-4; K-6, cs-AI, cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.11249v1)  

---


**ABSTRACT**  
Trust and privacy have emerged as significant concerns in online transactions. Sharing information on health is especially sensitive but it is necessary for purchasing and utilizing health insurance. Evidence shows that consumers are increasingly comfortable with technology in place of humans, but the expanding use of AI potentially changes this. This research explores whether trust and privacy concern are barriers to the adoption of AI in health insurance. Two scenarios are compared: The first scenario has limited AI that is not in the interface and its presence is not explicitly revealed to the consumer. In the second scenario there is an AI interface and AI evaluation, and this is explicitly revealed to the consumer. The two scenarios were modeled and compared using SEM PLS-MGA. The findings show that trust is significantly lower in the second scenario where AI is visible. Privacy concerns are higher with AI but the difference is not statistically significant within the model.

{{</citation>}}


## cs.LG (7)



### (18/56) Detecting Hidden Triggers: Mapping Non-Markov Reward Functions to Markov (Gregory Hyde et al., 2024)

{{<citation>}}

Gregory Hyde, Eugene Santos Jr. (2024)  
**Detecting Hidden Triggers: Mapping Non-Markov Reward Functions to Markov**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.11325v1)  

---


**ABSTRACT**  
Many Reinforcement Learning algorithms assume a Markov reward function to guarantee optimality. However, not all reward functions are known to be Markov. In this paper, we propose a framework for mapping non-Markov reward functions into equivalent Markov ones by learning a Reward Machine - a specialized reward automaton. Unlike the general practice of learning Reward Machines, we do not require a set of high-level propositional symbols from which to learn. Rather, we learn \emph{hidden triggers} directly from data that encode them. We demonstrate the importance of learning Reward Machines versus their Deterministic Finite-State Automata counterparts, for this task, given their ability to model reward dependencies in a single automaton. We formalize this distinction in our learning objective. Our mapping process is constructed as an Integer Linear Programming problem. We prove that our mappings provide consistent expectations for the underlying process. We empirically validate our approach by learning black-box non-Markov Reward functions in the Officeworld Domain. Additionally, we demonstrate the effectiveness of learning dependencies between rewards in a new domain, Breakfastworld.

{{</citation>}}


### (19/56) DACR: Distribution-Augmented Contrastive Reconstruction for Time-Series Anomaly Detection (Lixu Wang et al., 2024)

{{<citation>}}

Lixu Wang, Shichao Xu, Xinyu Du, Qi Zhu. (2024)  
**DACR: Distribution-Augmented Contrastive Reconstruction for Time-Series Anomaly Detection**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2401.11271v1)  

---


**ABSTRACT**  
Anomaly detection in time-series data is crucial for identifying faults, failures, threats, and outliers across a range of applications. Recently, deep learning techniques have been applied to this topic, but they often struggle in real-world scenarios that are complex and highly dynamic, e.g., the normal data may consist of multiple distributions, and various types of anomalies may differ from the normal data to different degrees. In this work, to tackle these challenges, we propose Distribution-Augmented Contrastive Reconstruction (DACR). DACR generates extra data disjoint from the normal data distribution to compress the normal data's representation space, and enhances the feature extractor through contrastive learning to better capture the intrinsic semantics from time-series data. Furthermore, DACR employs an attention mechanism to model the semantic dependencies among multivariate time-series features, thereby achieving more robust reconstruction for anomaly detection. Extensive experiments conducted on nine benchmark datasets in various anomaly detection scenarios demonstrate the effectiveness of DACR in achieving new state-of-the-art time-series anomaly detection.

{{</citation>}}


### (20/56) TreeMIL: A Multi-instance Learning Framework for Time Series Anomaly Detection with Inexact Supervision (Chen Liu et al., 2024)

{{<citation>}}

Chen Liu, Shibo He, Haoyu Liu, Shizhong Li. (2024)  
**TreeMIL: A Multi-instance Learning Framework for Time Series Anomaly Detection with Inexact Supervision**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Anomaly Detection, Time Series  
[Paper Link](http://arxiv.org/abs/2401.11235v1)  

---


**ABSTRACT**  
Time series anomaly detection (TSAD) plays a vital role in various domains such as healthcare, networks, and industry. Considering labels are crucial for detection but difficult to obtain, we turn to TSAD with inexact supervision: only series-level labels are provided during the training phase, while point-level anomalies are predicted during the testing phase. Previous works follow a traditional multi-instance learning (MIL) approach, which focuses on encouraging high anomaly scores at individual time steps. However, time series anomalies are not only limited to individual point anomalies, they can also be collective anomalies, typically exhibiting abnormal patterns over subsequences. To address the challenge of collective anomalies, in this paper, we propose a tree-based MIL framework (TreeMIL). We first adopt an N-ary tree structure to divide the entire series into multiple nodes, where nodes at different levels represent subsequences with different lengths. Then, the subsequence features are extracted to determine the presence of collective anomalies. Finally, we calculate point-level anomaly scores by aggregating features from nodes at different levels. Experiments conducted on seven public datasets and eight baselines demonstrate that TreeMIL achieves an average 32.3% improvement in F1- score compared to previous state-of-the-art methods. The code is available at https://github.com/fly-orange/TreeMIL.

{{</citation>}}


### (21/56) Selecting Walk Schemes for Database Embedding (Yuval Lev Lubarsky et al., 2024)

{{<citation>}}

Yuval Lev Lubarsky, Jan Tönshoff, Martin Grohe, Benny Kimelfeld. (2024)  
**Selecting Walk Schemes for Database Embedding**  

---
Primary Category: cs.LG  
Categories: cs-DB, cs-LG, cs.LG  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2401.11215v1)  

---


**ABSTRACT**  
Machinery for data analysis often requires a numeric representation of the input. Towards that, a common practice is to embed components of structured data into a high-dimensional vector space. We study the embedding of the tuples of a relational database, where existing techniques are often based on optimization tasks over a collection of random walks from the database. The focus of this paper is on the recent FoRWaRD algorithm that is designed for dynamic databases, where walks are sampled by following foreign keys between tuples. Importantly, different walks have different schemas, or "walk schemes", that are derived by listing the relations and attributes along the walk. Also importantly, different walk schemes describe relationships of different natures in the database. We show that by focusing on a few informative walk schemes, we can obtain tuple embedding significantly faster, while retaining the quality. We define the problem of scheme selection for tuple embedding, devise several approaches and strategies for scheme selection, and conduct a thorough empirical study of the performance over a collection of downstream tasks. Our results confirm that with effective strategies for scheme selection, we can obtain high-quality embeddings considerably (e.g., three times) faster, preserve the extensibility to newly inserted tuples, and even achieve an increase in the precision of some tasks.

{{</citation>}}


### (22/56) Gaussian Adaptive Attention is All You Need: Robust Contextual Representations Across Multiple Modalities (Georgios Ioannides et al., 2024)

{{<citation>}}

Georgios Ioannides, Aman Chadha, Aaron Elkins. (2024)  
**Gaussian Adaptive Attention is All You Need: Robust Contextual Representations Across Multiple Modalities**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs-SD, cs.LG, eess-AS, eess-SP  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2401.11143v1)  

---


**ABSTRACT**  
We propose the Multi-Head Gaussian Adaptive Attention Mechanism (GAAM), a novel probabilistic attention framework, and the Gaussian Adaptive Transformer (GAT), designed to enhance information aggregation across multiple modalities, including Speech, Text and Vision. GAAM integrates learnable mean and variance into its attention mechanism, implemented in a Multi-Headed framework enabling it to collectively model any Probability Distribution for dynamic recalibration of feature significance. This method demonstrates significant improvements, especially with highly non-stationary data, surpassing the state-of-the-art attention techniques in model performance (up to approximately +20% in accuracy) by identifying key elements within the feature space. GAAM's compatibility with dot-product-based attention models and relatively low number of parameters showcases its adaptability and potential to boost existing attention frameworks. Empirically, GAAM exhibits superior adaptability and efficacy across a diverse range of tasks, including emotion recognition in speech, image classification, and text classification, thereby establishing its robustness and versatility in handling multi-modal data. Furthermore, we introduce the Importance Factor (IF), a new learning-based metric that enhances the explainability of models trained with GAAM-based methods. Overall, GAAM represents an advancement towards development of better performing and more explainable attention models across multiple modalities.

{{</citation>}}


### (23/56) Meta Reinforcement Learning for Strategic IoT Deployments Coverage in Disaster-Response UAV Swarms (Marwan Dhuheir et al., 2024)

{{<citation>}}

Marwan Dhuheir, Aiman Erbad, Ala Al-Fuqaha. (2024)  
**Meta Reinforcement Learning for Strategic IoT Deployments Coverage in Disaster-Response UAV Swarms**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-RO, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.11118v1)  

---


**ABSTRACT**  
In the past decade, Unmanned Aerial Vehicles (UAVs) have grabbed the attention of researchers in academia and industry for their potential use in critical emergency applications, such as providing wireless services to ground users and collecting data from areas affected by disasters, due to their advantages in terms of maneuverability and movement flexibility. The UAVs' limited resources, energy budget, and strict mission completion time have posed challenges in adopting UAVs for these applications. Our system model considers a UAV swarm that navigates an area collecting data from ground IoT devices focusing on providing better service for strategic locations and allowing UAVs to join and leave the swarm (e.g., for recharging) in a dynamic way. In this work, we introduce an optimization model with the aim of minimizing the total energy consumption and provide the optimal path planning of UAVs under the constraints of minimum completion time and transmit power. The formulated optimization is NP-hard making it not applicable for real-time decision making. Therefore, we introduce a light-weight meta-reinforcement learning solution that can also cope with sudden changes in the environment through fast convergence. We conduct extensive simulations and compare our approach to three state-of-the-art learning models. Our simulation results prove that our introduced approach is better than the three state-of-the-art algorithms in providing coverage to strategic locations with fast convergence.

{{</citation>}}


### (24/56) On The Temporal Domain of Differential Equation Inspired Graph Neural Networks (Moshe Eliasof et al., 2024)

{{<citation>}}

Moshe Eliasof, Eldad Haber, Eran Treister, Carola-Bibiane Schönlieb. (2024)  
**On The Temporal Domain of Differential Equation Inspired Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2401.11074v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) have demonstrated remarkable success in modeling complex relationships in graph-structured data. A recent innovation in this field is the family of Differential Equation-Inspired Graph Neural Networks (DE-GNNs), which leverage principles from continuous dynamical systems to model information flow on graphs with built-in properties such as feature smoothing or preservation. However, existing DE-GNNs rely on first or second-order temporal dependencies. In this paper, we propose a neural extension to those pre-defined temporal dependencies. We show that our model, called TDE-GNN, can capture a wide range of temporal dynamics that go beyond typical first or second-order methods, and provide use cases where existing temporal models are challenged. We demonstrate the benefit of learning the temporal dependencies using our method rather than using pre-defined temporal dynamics on several graph benchmarks.

{{</citation>}}


## cs.CL (11)



### (25/56) Analyzing Task-Encoding Tokens in Large Language Models (Yu Bai et al., 2024)

{{<citation>}}

Yu Bai, Heyan Huang, Cesare Spinoso-Di Piano, Marc-Antoine Rondeau, Sanxing Chen, Yang Gao, Jackie Chi Kit Cheung. (2024)  
**Analyzing Task-Encoding Tokens in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.11323v1)  

---


**ABSTRACT**  
In-context learning (ICL) has become an effective solution for few-shot learning in natural language processing. Past work has found that, during this process, representations of the last prompt token are utilized to store task reasoning procedures, thereby explaining the working mechanism of in-context learning. In this paper, we seek to locate and analyze other task-encoding tokens whose representations store task reasoning procedures. Supported by experiments that ablate the representations of different token types, we find that template and stopword tokens are the most prone to be task-encoding tokens. In addition, we demonstrate experimentally that lexical cues, repetition, and text formats are the main distinguishing characteristics of these tokens. Our work provides additional insights into how large language models (LLMs) leverage task reasoning procedures in ICL and suggests that future work may involve using task-encoding tokens to improve the computational efficiency of LLMs at inference time and their ability to handle long sequences.

{{</citation>}}


### (26/56) PRILoRA: Pruned and Rank-Increasing Low-Rank Adaptation (Nadav Benedek et al., 2024)

{{<citation>}}

Nadav Benedek, Lior Wolf. (2024)  
**PRILoRA: Pruned and Rank-Increasing Low-Rank Adaptation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GLUE  
[Paper Link](http://arxiv.org/abs/2401.11316v1)  

---


**ABSTRACT**  
With the proliferation of large pre-trained language models (PLMs), fine-tuning all model parameters becomes increasingly inefficient, particularly when dealing with numerous downstream tasks that entail substantial training and storage costs. Several approaches aimed at achieving parameter-efficient fine-tuning (PEFT) have been proposed. Among them, Low-Rank Adaptation (LoRA) stands out as an archetypal method, incorporating trainable rank decomposition matrices into each target module. Nevertheless, LoRA does not consider the varying importance of each layer. To address these challenges, we introduce PRILoRA, which linearly allocates a different rank for each layer, in an increasing manner, and performs pruning throughout the training process, considering both the temporary magnitude of weights and the accumulated statistics of the input to any given layer. We validate the effectiveness of PRILoRA through extensive experiments on eight GLUE benchmarks, setting a new state of the art.

{{</citation>}}


### (27/56) Word-Level ASR Quality Estimation for Efficient Corpus Sampling and Post-Editing through Analyzing Attentions of a Reference-Free Metric (Golara Javadi et al., 2024)

{{<citation>}}

Golara Javadi, Kamer Ali Yuksel, Yunsu Kim, Thiago Castro Ferreira, Mohamed Al-Badrashiny. (2024)  
**Word-Level ASR Quality Estimation for Efficient Corpus Sampling and Post-Editing through Analyzing Attentions of a Reference-Free Metric**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: AI, Attention  
[Paper Link](http://arxiv.org/abs/2401.11268v1)  

---


**ABSTRACT**  
In the realm of automatic speech recognition (ASR), the quest for models that not only perform with high accuracy but also offer transparency in their decision-making processes is crucial. The potential of quality estimation (QE) metrics is introduced and evaluated as a novel tool to enhance explainable artificial intelligence (XAI) in ASR systems. Through experiments and analyses, the capabilities of the NoRefER (No Reference Error Rate) metric are explored in identifying word-level errors to aid post-editors in refining ASR hypotheses. The investigation also extends to the utility of NoRefER in the corpus-building process, demonstrating its effectiveness in augmenting datasets with insightful annotations. The diagnostic aspects of NoRefER are examined, revealing its ability to provide valuable insights into model behaviors and decision patterns. This has proven beneficial for prioritizing hypotheses in post-editing workflows and fine-tuning ASR models. The findings suggest that NoRefER is not merely a tool for error detection but also a comprehensive framework for enhancing ASR systems' transparency, efficiency, and effectiveness. To ensure the reproducibility of the results, all source codes of this study are made publicly available.

{{</citation>}}


### (28/56) Prompt-RAG: Pioneering Vector Embedding-Free Retrieval-Augmented Generation in Niche Domains, Exemplified by Korean Medicine (Bongsu Kang et al., 2024)

{{<citation>}}

Bongsu Kang, Jundong Kim, Tae-Rim Yun, Chang-Eop Kim. (2024)  
**Prompt-RAG: Pioneering Vector Embedding-Free Retrieval-Augmented Generation in Niche Domains, Exemplified by Korean Medicine**  

---
Primary Category: cs.CL  
Categories: I-2-7; H-3-3; J-3, cs-CL, cs-IR, cs.CL  
Keywords: ChatGPT, Embedding, GPT, QA  
[Paper Link](http://arxiv.org/abs/2401.11246v1)  

---


**ABSTRACT**  
We propose a natural language prompt-based retrieval augmented generation (Prompt-RAG), a novel approach to enhance the performance of generative large language models (LLMs) in niche domains. Conventional RAG methods mostly require vector embeddings, yet the suitability of generic LLM-based embedding representations for specialized domains remains uncertain. To explore and exemplify this point, we compared vector embeddings from Korean Medicine (KM) and Conventional Medicine (CM) documents, finding that KM document embeddings correlated more with token overlaps and less with human-assessed document relatedness, in contrast to CM embeddings. Prompt-RAG, distinct from conventional RAG models, operates without the need for embedding vectors. Its performance was assessed through a Question-Answering (QA) chatbot application, where responses were evaluated for relevance, readability, and informativeness. The results showed that Prompt-RAG outperformed existing models, including ChatGPT and conventional vector embedding-based RAGs, in terms of relevance and informativeness. Despite challenges like content structuring and response latency, the advancements in LLMs are expected to encourage the use of Prompt-RAG, making it a promising tool for other domains in need of RAG methods.

{{</citation>}}


### (29/56) Orion-14B: Open-source Multilingual Large Language Models (Du Chen et al., 2024)

{{<citation>}}

Du Chen, Yi Huang, Xiaopu Li, Yongqiang Li, Yongqiang Liu, Haihui Pan, Leichao Xu, Dacheng Zhang, Zhipeng Zhang, Kun Han. (2024)  
**Orion-14B: Open-source Multilingual Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: AI, Language Model, Multilingual  
[Paper Link](http://arxiv.org/abs/2401.12246v1)  

---


**ABSTRACT**  
In this study, we introduce Orion-14B, a collection of multilingual large language models with 14 billion parameters. We utilize a data scheduling approach to train a foundational model on a diverse corpus of 2.5 trillion tokens, sourced from texts in English, Chinese, Japanese, Korean, and other languages. Additionally, we fine-tuned a series of models tailored for conversational applications and other specific use cases. Our evaluation results demonstrate that Orion-14B achieves state-of-the-art performance across a broad spectrum of tasks. We make the Orion-14B model family and its associated code publicly accessible https://github.com/OrionStarAI/Orion, aiming to inspire future research and practical applications in the field.

{{</citation>}}


### (30/56) Unfair TOS: An Automated Approach using Customized BERT (Bathini Sai Akash et al., 2024)

{{<citation>}}

Bathini Sai Akash, Akshara Kupireddy, Lalita Bhanu Murthy. (2024)  
**Unfair TOS: An Automated Approach using Customized BERT**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs.CL  
Keywords: BERT, Transformer  
[Paper Link](http://arxiv.org/abs/2401.11207v1)  

---


**ABSTRACT**  
Terms of Service (ToS) form an integral part of any agreement as it defines the legal relationship between a service provider and an end-user. Not only do they establish and delineate reciprocal rights and responsibilities, but they also provide users with information on essential aspects of contracts that pertain to the use of digital spaces. These aspects include a wide range of topics, including limitation of liability, data protection, etc. Users tend to accept the ToS without going through it before using any application or service. Such ignorance puts them in a potentially weaker situation in case any action is required. Existing methodologies for the detection or classification of unfair clauses are however obsolete and show modest performance. In this research paper, we present SOTA(State of The Art) results on unfair clause detection from ToS documents based on unprecedented Fine-tuning BERT in integration with SVC(Support Vector Classifier). The study shows proficient performance with a macro F1-score of 0.922 at unfair clause detection, and superior performance is also shown in the classification of unfair clauses by each tag. Further, a comparative analysis is performed by answering research questions on the Transformer models utilized. In order to further research and experimentation the code and results are made available on https://github.com/batking24/Unfair-TOS-An-Automated-Approach-based-on-Fine-tuning-BERT-in-conjunction-with-ML.

{{</citation>}}


### (31/56) InferAligner: Inference-Time Alignment for Harmlessness through Cross-Model Guidance (Pengyu Wang et al., 2024)

{{<citation>}}

Pengyu Wang, Dong Zhang, Linyang Li, Chenkun Tan, Xinghao Wang, Ke Ren, Botian Jiang, Xipeng Qiu. (2024)  
**InferAligner: Inference-Time Alignment for Harmlessness through Cross-Model Guidance**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.11206v1)  

---


**ABSTRACT**  
With the rapid development of large language models (LLMs), they are not only used as general-purpose AI assistants but are also customized through further fine-tuning to meet the requirements of different applications. A pivotal factor in the success of current LLMs is the alignment process. Current alignment methods, such as supervised fine-tuning (SFT) and reinforcement learning from human feedback (RLHF), focus on training-time alignment and are often complex and cumbersome to implement. Therefore, we develop \textbf{InferAligner}, a novel inference-time alignment method that utilizes cross-model guidance for harmlessness alignment. InferAligner utilizes safety steering vectors extracted from safety-aligned model to modify the activations of the target model when responding to harmful inputs, thereby guiding the target model to provide harmless responses. Experimental results show that our method can be very effectively applied to domain-specific models in finance, medicine, and mathematics, as well as to multimodal large language models (MLLMs) such as LLaVA. It significantly diminishes the Attack Success Rate (ASR) of both harmful instructions and jailbreak attacks, while maintaining almost unchanged performance in downstream tasks.

{{</citation>}}


### (32/56) How the Advent of Ubiquitous Large Language Models both Stymie and Turbocharge Dynamic Adversarial Question Generation (Yoo Yeon Sung et al., 2024)

{{<citation>}}

Yoo Yeon Sung, Ishani Mondal, Jordan Boyd-Graber. (2024)  
**How the Advent of Ubiquitous Large Language Models both Stymie and Turbocharge Dynamic Adversarial Question Generation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-HC, cs.CL  
Keywords: Language Model, Question Generation  
[Paper Link](http://arxiv.org/abs/2401.11185v1)  

---


**ABSTRACT**  
Dynamic adversarial question generation, where humans write examples to stump a model, aims to create examples that are realistic and informative. However, the advent of large language models (LLMs) has been a double-edged sword for human authors: more people are interested in seeing and pushing the limits of these models, but because the models are so much stronger an opponent, they are harder to defeat. To understand how these models impact adversarial question writing process, we enrich the writing guidance with LLMs and retrieval models for the authors to reason why their questions are not adversarial. While authors could create interesting, challenging adversarial questions, they sometimes resort to tricks that result in poor questions that are ambiguous, subjective, or confusing not just to a computer but also to humans. To address these issues, we propose new metrics and incentives for eliciting good, challenging questions and present a new dataset of adversarially authored questions.

{{</citation>}}


### (33/56) Enhancing Large Language Models for Clinical Decision Support by Incorporating Clinical Practice Guidelines (David Oniani et al., 2024)

{{<citation>}}

David Oniani, Xizhi Wu, Shyam Visweswaran, Sumit Kapoor, Shravan Kooragayalu, Katelyn Polanska, Yanshan Wang. (2024)  
**Enhancing Large Language Models for Clinical Decision Support by Incorporating Clinical Practice Guidelines**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Clinical, Few-Shot, GPT, GPT-3.5, GPT-4, LLaMA, Language Model, PaLM, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2401.11120v2)  

---


**ABSTRACT**  
Background Large Language Models (LLMs), enhanced with Clinical Practice Guidelines (CPGs), can significantly improve Clinical Decision Support (CDS). However, methods for incorporating CPGs into LLMs are not well studied. Methods We develop three distinct methods for incorporating CPGs into LLMs: Binary Decision Tree (BDT), Program-Aided Graph Construction (PAGC), and Chain-of-Thought-Few-Shot Prompting (CoT-FSP). To evaluate the effectiveness of the proposed methods, we create a set of synthetic patient descriptions and conduct both automatic and human evaluation of the responses generated by four LLMs: GPT-4, GPT-3.5 Turbo, LLaMA, and PaLM 2. Zero-Shot Prompting (ZSP) was used as the baseline method. We focus on CDS for COVID-19 outpatient treatment as the case study. Results All four LLMs exhibit improved performance when enhanced with CPGs compared to the baseline ZSP. BDT outperformed both CoT-FSP and PAGC in automatic evaluation. All of the proposed methods demonstrated high performance in human evaluation. Conclusion LLMs enhanced with CPGs demonstrate superior performance, as compared to plain LLMs with ZSP, in providing accurate recommendations for COVID-19 outpatient treatment, which also highlights the potential for broader applications beyond the case study.

{{</citation>}}


### (34/56) Exploiting Duality in Open Information Extraction with Predicate Prompt (Zhen Chen et al., 2024)

{{<citation>}}

Zhen Chen, Jingping Liu, Deqing Yang, Yanghua Xiao, Huimin Xu, Zongyu Wang, Rui Xie, Yunsen Xian. (2024)  
**Exploiting Duality in Open Information Extraction with Predicate Prompt**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-IR, cs.CL  
Keywords: Information Extraction  
[Paper Link](http://arxiv.org/abs/2401.11107v1)  

---


**ABSTRACT**  
Open information extraction (OpenIE) aims to extract the schema-free triplets in the form of (\emph{subject}, \emph{predicate}, \emph{object}) from a given sentence. Compared with general information extraction (IE), OpenIE poses more challenges for the IE models, {especially when multiple complicated triplets exist in a sentence. To extract these complicated triplets more effectively, in this paper we propose a novel generative OpenIE model, namely \emph{DualOIE}, which achieves a dual task at the same time as extracting some triplets from the sentence, i.e., converting the triplets into the sentence.} Such dual task encourages the model to correctly recognize the structure of the given sentence and thus is helpful to extract all potential triplets from the sentence. Specifically, DualOIE extracts the triplets in two steps: 1) first extracting a sequence of all potential predicates, 2) then using the predicate sequence as a prompt to induce the generation of triplets. Our experiments on two benchmarks and our dataset constructed from Meituan demonstrate that DualOIE achieves the best performance among the state-of-the-art baselines. Furthermore, the online A/B test on Meituan platform shows that 0.93\% improvement of QV-CTR and 0.56\% improvement of UV-CTR have been obtained when the triplets extracted by DualOIE were leveraged in Meituan's search system.

{{</citation>}}


### (35/56) Evaluating and Enhancing Large Language Models Performance in Domain-specific Medicine: Osteoarthritis Management with DocOA (Xi Chen et al., 2024)

{{<citation>}}

Xi Chen, MingKe You, Li Wang, WeiZhi Liu, Yu Fu, Jie Xu, Shaoting Zhang, Gang Chen, Kang Li, Jian Li. (2024)  
**Evaluating and Enhancing Large Language Models Performance in Domain-specific Medicine: Osteoarthritis Management with DocOA**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-3.5, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2401.12998v1)  

---


**ABSTRACT**  
The efficacy of large language models (LLMs) in domain-specific medicine, particularly for managing complex diseases such as osteoarthritis (OA), remains largely unexplored. This study focused on evaluating and enhancing the clinical capabilities of LLMs in specific domains, using osteoarthritis (OA) management as a case study. A domain specific benchmark framework was developed, which evaluate LLMs across a spectrum from domain-specific knowledge to clinical applications in real-world clinical scenarios. DocOA, a specialized LLM tailored for OA management that integrates retrieval-augmented generation (RAG) and instruction prompts, was developed. The study compared the performance of GPT-3.5, GPT-4, and a specialized assistant, DocOA, using objective and human evaluations. Results showed that general LLMs like GPT-3.5 and GPT-4 were less effective in the specialized domain of OA management, particularly in providing personalized treatment recommendations. However, DocOA showed significant improvements. This study introduces a novel benchmark framework which assesses the domain-specific abilities of LLMs in multiple aspects, highlights the limitations of generalized LLMs in clinical contexts, and demonstrates the potential of tailored approaches for developing domain-specific medical LLMs.

{{</citation>}}


## cs.HC (3)



### (36/56) CodeAid: Evaluating a Classroom Deployment of an LLM-based Programming Assistant that Balances Student and Educator Needs (Majeed Kazemitabaar et al., 2024)

{{<citation>}}

Majeed Kazemitabaar, Runlong Ye, Xiaoning Wang, Austin Z. Henley, Paul Denny, Michelle Craig, Tovi Grossman. (2024)  
**CodeAid: Evaluating a Classroom Deployment of an LLM-based Programming Assistant that Balances Student and Educator Needs**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2401.11314v1)  

---


**ABSTRACT**  
Timely, personalized feedback is essential for students learning programming, especially as class sizes expand. LLM-based tools like ChatGPT offer instant support, but reveal direct answers with code, which may hinder deep conceptual engagement. We developed CodeAid, an LLM-based programming assistant delivering helpful, technically correct responses, without revealing code solutions. For example, CodeAid can answer conceptual questions, generate pseudo-code with line-by-line explanations, and annotate student's incorrect code with fix suggestions. We deployed CodeAid in a programming class of 700 students for a 12-week semester. A thematic analysis of 8,000 usages of CodeAid was performed, further enriched by weekly surveys, and 22 student interviews. We then interviewed eight programming educators to gain further insights on CodeAid. Findings revealed students primarily used CodeAid for conceptual understanding and debugging, although a minority tried to obtain direct code. Educators appreciated CodeAid's educational approach, and expressed concerns about occasional incorrect feedback and students defaulting to ChatGPT.

{{</citation>}}


### (37/56) Visualization Generation with Large Language Models: An Evaluation (Guozheng Li et al., 2024)

{{<citation>}}

Guozheng Li, Xinyu Wang, Gerile Aodeng, Shunyuan Zheng, Yu Zhang, Chuangxin Ou, Song Wang, Chi Harold Liu. (2024)  
**Visualization Generation with Large Language Models: An Evaluation**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: GPT, GPT-3.5, Language Model  
[Paper Link](http://arxiv.org/abs/2401.11255v1)  

---


**ABSTRACT**  
Analysts frequently need to create visualizations in the data analysis process to obtain and communicate insights. To reduce the burden of creating visualizations, previous research has developed various approaches for analysts to create visualizations from natural language queries. Recent studies have demonstrated the capabilities of large language models in natural language understanding and code generation tasks. The capabilities imply the potential of using large language models to generate visualization specifications from natural language queries. In this paper, we evaluate the capability of a large language model to generate visualization specifications on the task of natural language to visualization (NL2VIS). More specifically, we have opted for GPT-3.5 and Vega-Lite to represent large language models and visualization specifications, respectively. The evaluation is conducted on the nvBench dataset. In the evaluation, we utilize both zero-shot and few-shot prompt strategies. The results demonstrate that GPT-3.5 surpasses previous NL2VIS approaches. Additionally, the performance of few-shot prompts is higher than that of zero-shot prompts. We discuss the limitations of GPT-3.5 on NL2VIS, such as misunderstanding the data attributes and grammar errors in generated specifications. We also summarized several directions, such as correcting the ground truth and reducing the ambiguities in natural language queries, to improve the NL2VIS benchmark.

{{</citation>}}


### (38/56) ChatGPT in the classroom. Exploring its potential and limitations in a Functional Programming course (Dan-Matei Popovici, 2024)

{{<citation>}}

Dan-Matei Popovici. (2024)  
**ChatGPT in the classroom. Exploring its potential and limitations in a Functional Programming course**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2401.11166v1)  

---


**ABSTRACT**  
In November 2022, OpenAI has introduced ChatGPT, a chatbot based on supervised and reinforcement learning. Not only can it answer questions emulating human-like responses, but it can also generate code from scratch or complete coding templates provided by the user. ChatGPT can generate unique responses which render any traditional anti-plagiarism tool useless. Its release has ignited a heated debate about its usage in academia, especially by students. We have found, to our surprise, that our students at POLITEHNICA University of Bucharest (UPB) have been using generative AI tools (ChatGPT and its predecessors) for solving homework, for at least 6 months. We therefore set out to explore the capabilities of ChatGPT and assess its value for educational purposes. We solved all our coding assignments for the semester from our UPB Functional Programming course. We discovered that, although ChatGPT provides correct answers in 68% of the cases, only around half of those are legible solutions which can benefit students in some form. On the other hand, ChatGPT has a very good ability to perform code review on student programming homework. Based on these findings, we discuss the pros and cons of ChatGPT in education.

{{</citation>}}


## cs.MA (1)



### (39/56) Measuring Policy Distance for Multi-Agent Reinforcement Learning (Tianyi Hu et al., 2024)

{{<citation>}}

Tianyi Hu, Zhiqiang Pu, Xiaolin Ai, Tenghai Qiu, Jianqiang Yi. (2024)  
**Measuring Policy Distance for Multi-Agent Reinforcement Learning**  

---
Primary Category: cs.MA  
Categories: cs-AI, cs-MA, cs.MA  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.11257v1)  

---


**ABSTRACT**  
Diversity plays a crucial role in improving the performance of multi-agent reinforcement learning (MARL). Currently, many diversity-based methods have been developed to overcome the drawbacks of excessive parameter sharing in traditional MARL. However, there remains a lack of a general metric to quantify policy differences among agents. Such a metric would not only facilitate the evaluation of the diversity evolution in multi-agent systems, but also provide guidance for the design of diversity-based MARL algorithms. In this paper, we propose the multi-agent policy distance (MAPD), a general tool for measuring policy differences in MARL. By learning the conditional representations of agents' decisions, MAPD can computes the policy distance between any pair of agents. Furthermore, we extend MAPD to a customizable version, which can quantify differences among agent policies on specified aspects. Based on the online deployment of MAPD, we design a multi-agent dynamic parameter sharing (MADPS) algorithm as an example of the MAPD's applications. Extensive experiments demonstrate that our method is effective in measuring differences in agent policies and specific behavioral tendencies. Moreover, in comparison to other methods of parameter sharing, MADPS exhibits superior performance.

{{</citation>}}


## cs.IR (2)



### (40/56) Drop your Decoder: Pre-training with Bag-of-Word Prediction for Dense Passage Retrieval (Guangyuan Ma et al., 2024)

{{<citation>}}

Guangyuan Ma, Xing Wu, Zijia Lin, Songlin Hu. (2024)  
**Drop your Decoder: Pre-training with Bag-of-Word Prediction for Dense Passage Retrieval**  

---
Primary Category: cs.IR  
Categories: cs-CL, cs-IR, cs.IR  
Keywords: BERT, Transformer  
[Paper Link](http://arxiv.org/abs/2401.11248v1)  

---


**ABSTRACT**  
Masked auto-encoder pre-training has emerged as a prevalent technique for initializing and enhancing dense retrieval systems. It generally utilizes additional Transformer decoder blocks to provide sustainable supervision signals and compress contextual information into dense representations. However, the underlying reasons for the effectiveness of such a pre-training technique remain unclear. The usage of additional Transformer-based decoders also incurs significant computational costs. In this study, we aim to shed light on this issue by revealing that masked auto-encoder (MAE) pre-training with enhanced decoding significantly improves the term coverage of input tokens in dense representations, compared to vanilla BERT checkpoints. Building upon this observation, we propose a modification to the traditional MAE by replacing the decoder of a masked auto-encoder with a completely simplified Bag-of-Word prediction task. This modification enables the efficient compression of lexical signals into dense representations through unsupervised pre-training. Remarkably, our proposed method achieves state-of-the-art retrieval performance on several large-scale retrieval benchmarks without requiring any additional parameters, which provides a 67% training speed-up compared to standard masked auto-encoder pre-training with enhanced decoding.

{{</citation>}}


### (41/56) Navigating the Thin Line: Examining User Behavior in Search to Detect Engagement and Backfire Effects (F. M. Cau et al., 2024)

{{<citation>}}

F. M. Cau, N. Tintarev. (2024)  
**Navigating the Thin Line: Examining User Behavior in Search to Detect Engagement and Backfire Effects**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.11201v1)  

---


**ABSTRACT**  
Opinionated users often seek information that aligns with their preexisting beliefs while dismissing contradictory evidence due to confirmation bias. This conduct hinders their ability to consider alternative stances when searching the web. Despite this, few studies have analyzed how the diversification of search results on disputed topics influences the search behavior of highly opinionated users. To this end, we present a preregistered user study (n = 257) investigating whether different levels (low and high) of bias metrics and search results presentation (with or without AI-predicted stances labels) can affect the stance diversity consumption and search behavior of opinionated users on three debated topics (i.e., atheism, intellectual property rights, and school uniforms). Our results show that exposing participants to (counter-attitudinally) biased search results increases their consumption of attitude-opposing content, but we also found that bias was associated with a trend toward overall fewer interactions within the search page. We also found that 19% of users interacted with queries and search pages but did not select any search results. When we removed these participants in a post-hoc analysis, we found that stance labels increased the diversity of stances consumed by users, particularly when the search results were biased. Our findings highlight the need for future research to explore distinct search scenario settings to gain insight into opinionated users' behavior.

{{</citation>}}


## eess.IV (1)



### (42/56) Susceptibility of Adversarial Attack on Medical Image Segmentation Models (Zhongxuan Wang et al., 2024)

{{<citation>}}

Zhongxuan Wang, Leo Xu. (2024)  
**Susceptibility of Adversarial Attack on Medical Image Segmentation Models**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2401.11224v1)  

---


**ABSTRACT**  
The nature of deep neural networks has given rise to a variety of attacks, but little work has been done to address the effect of adversarial attacks on segmentation models trained on MRI datasets. In light of the grave consequences that such attacks could cause, we explore four models from the U-Net family and examine their responses to the Fast Gradient Sign Method (FGSM) attack. We conduct FGSM attacks on each of them and experiment with various schemes to conduct the attacks. In this paper, we find that medical imaging segmentation models are indeed vulnerable to adversarial attacks and that there is a negligible correlation between parameter size and adversarial attack success. Furthermore, we show that using a different loss function than the one used for training yields higher adversarial attack success, contrary to what the FGSM authors suggested. In future efforts, we will conduct the experiments detailed in this paper with more segmentation models and different attacks. We will also attempt to find ways to counteract the attacks by using model ensembles or special data augmentations. Our code is available at https://github.com/ZhongxuanWang/adv_attk

{{</citation>}}


## cs.IT (1)



### (43/56) On the Information Leakage Performance of Secure Finite Blocklength Transmissions over Rayleigh Fading Channels (Milad Tatar Mamaghani et al., 2024)

{{<citation>}}

Milad Tatar Mamaghani, Xiangyun Zhou, Nan Yang, A. Lee Swindlehurst, H. Vincent Poor. (2024)  
**On the Information Leakage Performance of Secure Finite Blocklength Transmissions over Rayleigh Fading Channels**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, math-IT  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.11219v1)  

---


**ABSTRACT**  
This paper presents a secrecy performance study of a wiretap communication system with finite blocklength (FBL) transmissions over Rayleigh fading channels, based on the definition of an average information leakage (AIL) metric. We evaluate the exact and closed-form approximate AIL performance, assuming that only statistical channel state information (CSI) of the eavesdropping link is available. Then, we reveal an inherent statistical relationship between the AIL metric in the FBL regime and the commonly-used secrecy outage probability in conventional infinite blocklength communications. Aiming to improve the secure communication performance of the considered system, we formulate a blocklength optimization problem and solve it via a low-complexity approach. Next, we present numerical results to verify our analytical findings and provide various important insights into the impacts of system parameters on the AIL. Specifically, our results indicate that i) compromising a small amount of AIL can lead to significant reliability improvements, and ii) the AIL experiences a secrecy floor in the high signal-to-noise ratio regime.

{{</citation>}}


## cs.RO (1)



### (44/56) Obstacle-Aware Navigation of Soft Growing Robots via Deep Reinforcement Learning (Haitham El-Hussieny et al., 2024)

{{<citation>}}

Haitham El-Hussieny, Ibrahim Hameed. (2024)  
**Obstacle-Aware Navigation of Soft Growing Robots via Deep Reinforcement Learning**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.11203v2)  

---


**ABSTRACT**  
Soft growing robots, are a type of robots that are designed to move and adapt to their environment in a similar way to how plants grow and move with potential applications where they could be used to navigate through tight spaces, dangerous terrain, and hard-to-reach areas. This research explores the application of deep reinforcement Q-learning algorithm for facilitating the navigation of the soft growing robots in cluttered environments. The proposed algorithm utilizes the flexibility of the soft robot to adapt and incorporate the interaction between the robot and the environment into the decision-making process. Results from simulations show that the proposed algorithm improves the soft robot's ability to navigate effectively and efficiently in confined spaces. This study presents a promising approach to addressing the challenges faced by growing robots in particular and soft robots general in planning obstacle-aware paths in real-world scenarios.

{{</citation>}}


## cs.DC (1)



### (45/56) Inference without Interference: Disaggregate LLM Inference for Mixed Downstream Workloads (Cunchen Hu et al., 2024)

{{<citation>}}

Cunchen Hu, Heyang Huang, Liangliang Xu, Xusheng Chen, Jiang Xu, Shuang Chen, Hao Feng, Chenxi Wang, Sa Wang, Yungang Bao, Ninghui Sun, Yizhou Shan. (2024)  
**Inference without Interference: Disaggregate LLM Inference for Mixed Downstream Workloads**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.11181v1)  

---


**ABSTRACT**  
Transformer-based large language model (LLM) inference serving is now the backbone of many cloud services. LLM inference consists of a prefill phase and a decode phase. However, existing LLM deployment practices often overlook the distinct characteristics of these phases, leading to significant interference. To mitigate interference, our insight is to carefully schedule and group inference requests based on their characteristics. We realize this idea in TetriInfer through three pillars. First, it partitions prompts into fixed-size chunks so that the accelerator always runs close to its computationsaturated limit. Second, it disaggregates prefill and decode instances so each can run independently. Finally, it uses a smart two-level scheduling algorithm augmented with predicted resource usage to avoid decode scheduling hotspots. Results show that TetriInfer improves time-to-first-token (TTFT), job completion time (JCT), and inference efficiency in turns of performance per dollar by a large margin, e.g., it uses 38% less resources all the while lowering average TTFT and average JCT by 97% and 47%, respectively.

{{</citation>}}


## cs.DB (1)



### (46/56) Extending Polaris to Support Transactions (Josep Aguilar-Saborit et al., 2024)

{{<citation>}}

Josep Aguilar-Saborit, Raghu Ramakrishnan, Kevin Bocksrocker, Alan Halverson, Konstantin Kosinsky, Ryan O'Connor, Nadejda Poliakova, Moe Shafiei, Taewoo Kim, Phil Kon-Kim, Haris Mahmud-Ansari, Blazej Matuszyk, Matt Miles, Sumin Mohanan, Cristian Petculescu, Ishan Rahesh-Madan, Emma Rose-Wirshing, Elias Yousefi. (2024)  
**Extending Polaris to Support Transactions**  

---
Primary Category: cs.DB  
Categories: cs-DB, cs.DB  
Keywords: Microsoft  
[Paper Link](http://arxiv.org/abs/2401.11162v1)  

---


**ABSTRACT**  
In Polaris, we introduced a cloud-native distributed query processor to perform analytics at scale. In this paper, we extend the underlying Polaris distributed computation framework, which can be thought of as a read-only transaction engine, to execute general transactions (including updates, deletes, inserts and bulk loads, in addition to queries) for Tier 1 warehousing workloads in a highly performant and predictable manner. We take advantage of the immutability of data files in log-structured data stores and build on SQL Server transaction management to deliver full transactional support with Snapshot Isolation semantics, including multi-table and multi-statement transactions. With the enhancements described in this paper, Polaris supports both query processing and transactions for T-SQL in Microsoft Fabric.

{{</citation>}}


## cs.SE (1)



### (47/56) BinaryAI: Binary Software Composition Analysis via Intelligent Binary Source Code Matching (Ling Jiang et al., 2024)

{{<citation>}}

Ling Jiang, Junwen An, Huihui Huang, Qiyi Tang, Sen Nie, Shi Wu, Yuqun Zhang. (2024)  
**BinaryAI: Binary Software Composition Analysis via Intelligent Binary Source Code Matching**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.11161v2)  

---


**ABSTRACT**  
While third-party libraries are extensively reused to enhance productivity during software development, they can also introduce potential security risks such as vulnerability propagation. Software composition analysis, proposed to identify reused TPLs for reducing such risks, has become an essential procedure within modern DevSecOps. As one of the mainstream SCA techniques, binary-to-source SCA identifies the third-party source projects contained in binary files via binary source code matching, which is a major challenge in reverse engineering since binary and source code exhibit substantial disparities after compilation. The existing binary-to-source SCA techniques leverage basic syntactic features that suffer from redundancy and lack robustness in the large-scale TPL dataset, leading to inevitable false positives and compromised recall. To mitigate these limitations, we introduce BinaryAI, a novel binary-to-source SCA technique with two-phase binary source code matching to capture both syntactic and semantic code features. First, BinaryAI trains a transformer-based model to produce function-level embeddings and obtain similar source functions for each binary function accordingly. Then by applying the link-time locality to facilitate function matching, BinaryAI detects the reused TPLs based on the ratio of matched source functions. Our experimental results demonstrate the superior performance of BinaryAI in terms of binary source code matching and the downstream SCA task. Specifically, our embedding model outperforms the state-of-the-art model CodeCMR, i.e., achieving 22.54% recall@1 and 0.34 MRR compared with 10.75% and 0.17 respectively. Additionally, BinaryAI outperforms all existing binary-to-source SCA tools in TPL detection, increasing the precision from 73.36% to 85.84% and recall from 59.81% to 64.98% compared with the well-recognized commercial SCA product Black Duck.

{{</citation>}}


## cs.CR (5)



### (48/56) Generalizing Speaker Verification for Spoof Awareness in the Embedding Space (Xuechen Liu et al., 2024)

{{<citation>}}

Xuechen Liu, Md Sahidullah, Kong Aik Lee, Tomi Kinnunen. (2024)  
**Generalizing Speaker Verification for Spoof Awareness in the Embedding Space**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs-SD, cs.CR, eess-AS  
Keywords: Embedding, Speaker Verification  
[Paper Link](http://arxiv.org/abs/2401.11156v1)  

---


**ABSTRACT**  
It is now well-known that automatic speaker verification (ASV) systems can be spoofed using various types of adversaries. The usual approach to counteract ASV systems against such attacks is to develop a separate spoofing countermeasure (CM) module to classify speech input either as a bonafide, or a spoofed utterance. Nevertheless, such a design requires additional computation and utilization efforts at the authentication stage. An alternative strategy involves a single monolithic ASV system designed to handle both zero-effort imposter (non-targets) and spoofing attacks. Such spoof-aware ASV systems have the potential to provide stronger protections and more economic computations. To this end, we propose to generalize the standalone ASV (G-SASV) against spoofing attacks, where we leverage limited training data from CM to enhance a simple backend in the embedding space, without the involvement of a separate CM module during the test (authentication) phase. We propose a novel yet simple backend classifier based on deep neural networks and conduct the study via domain adaptation and multi-task integration of spoof embeddings at the training stage. Experiments are conducted on the ASVspoof 2019 logical access dataset, where we improve the performance of statistical ASV backends on the joint (bonafide and spoofed) and spoofed conditions by a maximum of 36.2% and 49.8% in terms of equal error rates, respectively.

{{</citation>}}


### (49/56) CARE: Ensemble Adversarial Robustness Evaluation Against Adaptive Attackers for Security Applications (Hangsheng Zhang et al., 2024)

{{<citation>}}

Hangsheng Zhang, Jiqiang Liu, Jinsong Dong. (2024)  
**CARE: Ensemble Adversarial Robustness Evaluation Against Adaptive Attackers for Security Applications**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2401.11126v1)  

---


**ABSTRACT**  
Ensemble defenses, are widely employed in various security-related applications to enhance model performance and robustness. The widespread adoption of these techniques also raises many questions: Are general ensembles defenses guaranteed to be more robust than individuals? Will stronger adaptive attacks defeat existing ensemble defense strategies as the cybersecurity arms race progresses? Can ensemble defenses achieve adversarial robustness to different types of attacks simultaneously and resist the continually adjusted adaptive attacks? Unfortunately, these critical questions remain unresolved as there are no platforms for comprehensive evaluation of ensemble adversarial attacks and defenses in the cybersecurity domain. In this paper, we propose a general Cybersecurity Adversarial Robustness Evaluation (CARE) platform aiming to bridge this gap.

{{</citation>}}


### (50/56) BadChain: Backdoor Chain-of-Thought Prompting for Large Language Models (Zhen Xiang et al., 2024)

{{<citation>}}

Zhen Xiang, Fengqing Jiang, Zidi Xiong, Bhaskar Ramasubramanian, Radha Poovendran, Bo Li. (2024)  
**BadChain: Backdoor Chain-of-Thought Prompting for Large Language Models**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs.CR  
Keywords: GPT, GPT-3.5, GPT-4, Language Model, PaLM  
[Paper Link](http://arxiv.org/abs/2401.12242v1)  

---


**ABSTRACT**  
Large language models (LLMs) are shown to benefit from chain-of-thought (COT) prompting, particularly when tackling tasks that require systematic reasoning processes. On the other hand, COT prompting also poses new vulnerabilities in the form of backdoor attacks, wherein the model will output unintended malicious content under specific backdoor-triggered conditions during inference. Traditional methods for launching backdoor attacks involve either contaminating the training dataset with backdoored instances or directly manipulating the model parameters during deployment. However, these approaches are not practical for commercial LLMs that typically operate via API access. In this paper, we propose BadChain, the first backdoor attack against LLMs employing COT prompting, which does not require access to the training dataset or model parameters and imposes low computational overhead. BadChain leverages the inherent reasoning capabilities of LLMs by inserting a backdoor reasoning step into the sequence of reasoning steps of the model output, thereby altering the final response when a backdoor trigger exists in the query prompt. Empirically, we show the effectiveness of BadChain for two COT strategies across four LLMs (Llama2, GPT-3.5, PaLM2, and GPT-4) and six complex benchmark tasks encompassing arithmetic, commonsense, and symbolic reasoning. Moreover, we show that LLMs endowed with stronger reasoning capabilities exhibit higher susceptibility to BadChain, exemplified by a high average attack success rate of 97.0% across the six benchmark tasks on GPT-4. Finally, we propose two defenses based on shuffling and demonstrate their overall ineffectiveness against BadChain. Therefore, BadChain remains a severe threat to LLMs, underscoring the urgency for the development of robust and effective future defenses.

{{</citation>}}


### (51/56) LLM4Fuzz: Guided Fuzzing of Smart Contracts with Large Language Models (Chaofan Shou et al., 2024)

{{<citation>}}

Chaofan Shou, Jing Liu, Doudou Lu, Koushik Sen. (2024)  
**LLM4Fuzz: Guided Fuzzing of Smart Contracts with Large Language Models**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-SE, cs.CR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.11108v1)  

---


**ABSTRACT**  
As blockchain platforms grow exponentially, millions of lines of smart contract code are being deployed to manage extensive digital assets. However, vulnerabilities in this mission-critical code have led to significant exploitations and asset losses. Thorough automated security analysis of smart contracts is thus imperative. This paper introduces LLM4Fuzz to optimize automated smart contract security analysis by leveraging large language models (LLMs) to intelligently guide and prioritize fuzzing campaigns. While traditional fuzzing suffers from low efficiency in exploring the vast state space, LLM4Fuzz employs LLMs to direct fuzzers towards high-value code regions and input sequences more likely to trigger vulnerabilities. Additionally, LLM4Fuzz can leverage LLMs to guide fuzzers based on user-defined invariants, reducing blind exploration overhead. Evaluations of LLM4Fuzz on real-world DeFi projects show substantial gains in efficiency, coverage, and vulnerability detection compared to baseline fuzzing. LLM4Fuzz also uncovered five critical vulnerabilities that can lead to a loss of more than $247k.

{{</citation>}}


### (52/56) FedRKG: A Privacy-preserving Federated Recommendation Framework via Knowledge Graph Enhancement (Dezhong Yao et al., 2024)

{{<citation>}}

Dezhong Yao, Tongtong Liu, Qi Cao, Hai Jin. (2024)  
**FedRKG: A Privacy-preserving Federated Recommendation Framework via Knowledge Graph Enhancement**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs-DC, cs-IR, cs.CR  
Keywords: GNN, Graph Neural Network, Graph Neural Networks, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2401.11089v1)  

---


**ABSTRACT**  
Federated Learning (FL) has emerged as a promising approach for preserving data privacy in recommendation systems by training models locally. Recently, Graph Neural Networks (GNN) have gained popularity in recommendation tasks due to their ability to capture high-order interactions between users and items. However, privacy concerns prevent the global sharing of the entire user-item graph. To address this limitation, some methods create pseudo-interacted items or users in the graph to compensate for missing information for each client. Unfortunately, these methods introduce random noise and raise privacy concerns. In this paper, we propose FedRKG, a novel federated recommendation system, where a global knowledge graph (KG) is constructed and maintained on the server using publicly available item information, enabling higher-order user-item interactions. On the client side, a relation-aware GNN model leverages diverse KG relationships. To protect local interaction items and obscure gradients, we employ pseudo-labeling and Local Differential Privacy (LDP). Extensive experiments conducted on three real-world datasets demonstrate the competitive performance of our approach compared to centralized algorithms while ensuring privacy preservation. Moreover, FedRKG achieves an average accuracy improvement of 4% compared to existing federated learning baselines.

{{</citation>}}


## eess.SY (1)



### (53/56) Enhancing System-Level Safety in Mixed-Autonomy Platoon via Safe Reinforcement Learning (Jingyuan Zhou et al., 2024)

{{<citation>}}

Jingyuan Zhou, Longhao Yan, Kaidi Yang. (2024)  
**Enhancing System-Level Safety in Mixed-Autonomy Platoon via Safe Reinforcement Learning**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.11148v1)  

---


**ABSTRACT**  
Connected and automated vehicles (CAVs) have recently gained prominence in traffic research, thanks to the advancements in communication technology and autonomous driving. A variety of longitudinal control strategies for CAVs have been developed to enhance traffic efficiency, stability, and safety in mixed-autonomy scenarios. Deep reinforcement learning (DRL) is one promising strategy for mixed-autonomy platoon control since it can tackle complex scenarios in real-time. However, there are three research gaps for DRL-based mixed-autonomy platoon control. First, incorporating safety considerations into DRL typically relies on designing collision avoidance-based reward functions, which lack collision-free guarantees. Second, current DRL-based-control approaches for mixed traffic only consider the safety of CAVs, with little attention paid to the surrounding HDVs. To address the research gaps, we introduce a differentiable safety layer that converts DRL actions into safe actions with collision-free guarantees. This process relies on solving a differentiable quadratic programming problem that incorporates control barrier function-based (CBF) safety constraints for both CAV and its following HDVs to achieve system-level safety. Moreover, constructing CBF constraints needs system dynamics for the following HDVs, and thus we employ an online system identification module to estimate the car-following dynamics of the surrounding HDVs. The proposed safe reinforcement learning approach explicitly integrates system-level safety constraints into the training process and enables our method to adapt to varying safety-critical scenarios. Simulation results demonstrate that our proposed method effectively ensures CAV safety and improves HDV safety in mixed platoon environments while simultaneously enhancing traffic capacity and string stability.

{{</citation>}}


## cs.DL (1)



### (54/56) Promotion of Scientific Publications on ArXiv and X Is on the Rise and Impacts Citations (Chhandak Bagchi et al., 2024)

{{<citation>}}

Chhandak Bagchi, Eric Malmi, Przemyslaw Grabowicz. (2024)  
**Promotion of Scientific Publications on ArXiv and X Is on the Rise and Impacts Citations**  

---
Primary Category: cs.DL  
Categories: cs-CY, cs-DL, cs-SI, cs.DL  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2401.11116v1)  

---


**ABSTRACT**  
In the evolving landscape of scientific publishing, it is important to understand the drivers of high-impact research, to equip scientists with actionable strategies to enhance the reach of their work, and to understand trends in the use modern scientific publishing tools to inform their further development. Here, based on a large dataset of computer science publications, we study trends in the use of early preprint publications and revisions on ArXiv and the use of X (formerly Twitter) for promotion of such papers in the last 10 years. We find that early submission to ArXiv and promotion on X have soared in recent years. Estimating the effect that the use of each of these modern affordances has on the number of citations of scientific publications, we find that in the first 5 years from an initial publication peer-reviewed conference papers submitted early to ArXiv gain on average $21.1 \pm 17.4$ more citations, revised on ArXiv gain $18.4 \pm 17.6$ more citations, and promoted on X gain $44.4 \pm 8$ more citations. Our results show that promoting one's work on ArXiv or X has a large impact on the number of citations, as well as the number of influential citations computed by Semantic Scholar, and thereby on the career of researchers. We discuss the far-reaching implications of these findings for future scientific publishing systems and measures of scientific impact.

{{</citation>}}


## cs.SD (1)



### (55/56) ASM: Audio Spectrogram Mixer (Qingfeng Ji et al., 2024)

{{<citation>}}

Qingfeng Ji, Jicun Zhang, Yuxin Wang. (2024)  
**ASM: Audio Spectrogram Mixer**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.11102v1)  

---


**ABSTRACT**  
Transformer structures have demonstrated outstanding skills in the deep learning space recently, significantly increasing the accuracy of models across a variety of domains. Researchers have started to question whether such a sophisticated network structure is actually necessary and whether equally outstanding results can be reached with reduced inference cost due to its complicated network topology and high inference cost. In order to prove the Mixer's efficacy on three datasets Speech Commands, UrbanSound8k, and CASIA Chinese Sentiment Corpus this paper applies amore condensed version of the Mixer to an audio classification task and conducts comparative experiments with the Transformer-based Audio Spectrogram Transformer (AST)model. In addition, this paper conducts comparative experiments on the application of several activation functions in Mixer, namely GeLU, Mish, Swish and Acon-C. Further-more, the use of various activation functions in Mixer, including GeLU, Mish, Swish, and Acon-C, is compared in this research through comparison experiments. Additionally, some AST model flaws are highlighted, and the model suggested in this study is improved as a result. In conclusion, a model called the Audio Spectrogram Mixer, which is the first model for audio classification with Mixer, is suggested in this study and the model's future directions for improvement are examined.

{{</citation>}}


## cs.AI (1)



### (56/56) TypeDance: Creating Semantic Typographic Logos from Image through Personalized Generation (Shishi Xiao et al., 2024)

{{<citation>}}

Shishi Xiao, Liangwei Wang, Xiaojuan Ma, Wei Zeng. (2024)  
**TypeDance: Creating Semantic Typographic Logos from Image through Personalized Generation**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.11094v1)  

---


**ABSTRACT**  
Semantic typographic logos harmoniously blend typeface and imagery to represent semantic concepts while maintaining legibility. Conventional methods using spatial composition and shape substitution are hindered by the conflicting requirement for achieving seamless spatial fusion between geometrically dissimilar typefaces and semantics. While recent advances made AI generation of semantic typography possible, the end-to-end approaches exclude designer involvement and disregard personalized design. This paper presents TypeDance, an AI-assisted tool incorporating design rationales with the generative model for personalized semantic typographic logo design. It leverages combinable design priors extracted from uploaded image exemplars and supports type-imagery mapping at various structural granularity, achieving diverse aesthetic designs with flexible control. Additionally, we instantiate a comprehensive design workflow in TypeDance, including ideation, selection, generation, evaluation, and iteration. A two-task user evaluation, including imitation and creation, confirmed the usability of TypeDance in design across different usage scenarios

{{</citation>}}
