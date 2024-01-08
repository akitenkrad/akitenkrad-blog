---
draft: false
title: "arXiv @ 2024.01.04"
date: 2024-01-04
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2024"]
menu:
  sidebar:
    name: "arXiv @ 2024.01.04"
    identifier: arxiv_20240104
    parent: 202401_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CV (22)](#cscv-22)
- [cs.IT (3)](#csit-3)
- [cs.AI (3)](#csai-3)
- [cs.CL (18)](#cscl-18)
- [cs.LG (9)](#cslg-9)
- [eess.SP (2)](#eesssp-2)
- [cs.MA (1)](#csma-1)
- [cs.CR (7)](#cscr-7)
- [cs.CY (1)](#cscy-1)
- [cs.SI (1)](#cssi-1)
- [cs.HC (1)](#cshc-1)
- [eess.AS (2)](#eessas-2)
- [astro-ph.SR (1)](#astro-phsr-1)
- [cs.RO (1)](#csro-1)
- [cs.SE (1)](#csse-1)
- [cs.SD (1)](#cssd-1)

## cs.CV (22)



### (1/74) ColorizeDiffusion: Adjustable Sketch Colorization with Reference Image and Text (Dingkun Yan et al., 2024)

{{<citation>}}

Dingkun Yan, Liang Yuan, Yuma Nishioka, Issei Fujishiro, Suguru Saito. (2024)  
**ColorizeDiffusion: Adjustable Sketch Colorization with Reference Image and Text**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Sketch  
[Paper Link](http://arxiv.org/abs/2401.01456v1)  

---


**ABSTRACT**  
Recently, diffusion models have demonstrated their effectiveness in generating extremely high-quality images and have found wide-ranging applications, including automatic sketch colorization. However, most existing models use text to guide the conditional generation, with fewer attempts exploring the potential advantages of using image tokens as conditional inputs for networks. As such, this paper exhaustively investigates image-guided models, specifically targeting reference-based sketch colorization, which aims to colorize sketch images using reference color images. We investigate three critical aspects of reference-based diffusion models: the shortcomings compared to text-based counterparts, the training strategies, and the capability in zero-shot, sequential text-based manipulation. We introduce two variations of an image-guided latent diffusion model using different image tokens from the pre-trained CLIP image encoder, and we propose corresponding manipulation methods to adjust their results sequentially using weighted text inputs. We conduct comprehensive evaluations of our models through qualitative and quantitative experiments, as well as a user study.

{{</citation>}}


### (2/74) ProbMCL: Simple Probabilistic Contrastive Learning for Multi-label Visual Classification (Ahmad Sajedi et al., 2024)

{{<citation>}}

Ahmad Sajedi, Samir Khaki, Yuri A. Lawryshyn, Konstantinos N. Plataniotis. (2024)  
**ProbMCL: Simple Probabilistic Contrastive Learning for Multi-label Visual Classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2401.01448v1)  

---


**ABSTRACT**  
Multi-label image classification presents a challenging task in many domains, including computer vision and medical imaging. Recent advancements have introduced graph-based and transformer-based methods to improve performance and capture label dependencies. However, these methods often include complex modules that entail heavy computation and lack interpretability. In this paper, we propose Probabilistic Multi-label Contrastive Learning (ProbMCL), a novel framework to address these challenges in multi-label image classification tasks. Our simple yet effective approach employs supervised contrastive learning, in which samples that share enough labels with an anchor image based on a decision threshold are introduced as a positive set. This structure captures label dependencies by pulling positive pair embeddings together and pushing away negative samples that fall below the threshold. We enhance representation learning by incorporating a mixture density network into contrastive learning and generating Gaussian mixture distributions to explore the epistemic uncertainty of the feature encoder. We validate the effectiveness of our framework through experimentation with datasets from the computer vision and medical imaging domains. Our method outperforms the existing state-of-the-art methods while achieving a low computational footprint on both datasets. Visualization analyses also demonstrate that ProbMCL-learned classifiers maintain a meaningful semantic topology.

{{</citation>}}


### (3/74) Off-Road LiDAR Intensity Based Semantic Segmentation (Kasi Viswanath et al., 2024)

{{<citation>}}

Kasi Viswanath, Peng Jiang, Sujit PB, Srikanth Saripalli. (2024)  
**Off-Road LiDAR Intensity Based Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2401.01439v1)  

---


**ABSTRACT**  
LiDAR is used in autonomous driving to provide 3D spatial information and enable accurate perception in off-road environments, aiding in obstacle detection, mapping, and path planning. Learning-based LiDAR semantic segmentation utilizes machine learning techniques to automatically classify objects and regions in LiDAR point clouds. Learning-based models struggle in off-road environments due to the presence of diverse objects with varying colors, textures, and undefined boundaries, which can lead to difficulties in accurately classifying and segmenting objects using traditional geometric-based features. In this paper, we address this problem by harnessing the LiDAR intensity parameter to enhance object segmentation in off-road environments. Our approach was evaluated in the RELLIS-3D data set and yielded promising results as a preliminary analysis with improved mIoU for classes "puddle" and "grass" compared to more complex deep learning-based benchmarks. The methodology was evaluated for compatibility across both Velodyne and Ouster LiDAR systems, assuring its cross-platform applicability. This analysis advocates for the incorporation of calibrated intensity as a supplementary input, aiming to enhance the prediction accuracy of learning based semantic segmentation frameworks. https://github.com/MOONLABIISERB/lidar-intensity-predictor/tree/main

{{</citation>}}


### (4/74) MOC-RVQ: Multilevel Codebook-assisted Digital Generative Semantic Communication (Yingbin Zhou et al., 2024)

{{<citation>}}

Yingbin Zhou, Yaping Sun, Guanying Chen, Xiaodong Xu, Hao Chen, Binhong Huang, Shuguang Cui, Ping Zhang. (2024)  
**MOC-RVQ: Multilevel Codebook-assisted Digital Generative Semantic Communication**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.01272v1)  

---


**ABSTRACT**  
Vector quantization-based image semantic communication systems have successfully boosted transmission efficiency, but face a challenge with conflicting requirements between codebook design and digital constellation modulation. Traditional codebooks need a wide index range, while modulation favors few discrete states. To address this, we propose a multilevel generative semantic communication system with a two-stage training framework. In the first stage, we train a high-quality codebook, using a multi-head octonary codebook (MOC) to compress the index range. We also integrate a residual vector quantization (RVQ) mechanism for effective multilevel communication. In the second stage, a noise reduction block (NRB) based on Swin Transformer is introduced, coupled with the multilevel codebook from the first stage, serving as a high-quality semantic knowledge base (SKB) for generative feature restoration. Experimental results highlight MOC-RVQ's superior performance over methods like BPG or JPEG, even without channel error correction coding.

{{</citation>}}


### (5/74) VideoDrafter: Content-Consistent Multi-Scene Video Generation with LLM (Fuchen Long et al., 2024)

{{<citation>}}

Fuchen Long, Zhaofan Qiu, Ting Yao, Tao Mei. (2024)  
**VideoDrafter: Content-Consistent Multi-Scene Video Generation with LLM**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.01256v1)  

---


**ABSTRACT**  
The recent innovations and breakthroughs in diffusion models have significantly expanded the possibilities of generating high-quality videos for the given prompts. Most existing works tackle the single-scene scenario with only one video event occurring in a single background. Extending to generate multi-scene videos nevertheless is not trivial and necessitates to nicely manage the logic in between while preserving the consistent visual appearance of key content across video scenes. In this paper, we propose a novel framework, namely VideoDrafter, for content-consistent multi-scene video generation. Technically, VideoDrafter leverages Large Language Models (LLM) to convert the input prompt into comprehensive multi-scene script that benefits from the logical knowledge learnt by LLM. The script for each scene includes a prompt describing the event, the foreground/background entities, as well as camera movement. VideoDrafter identifies the common entities throughout the script and asks LLM to detail each entity. The resultant entity description is then fed into a text-to-image model to generate a reference image for each entity. Finally, VideoDrafter outputs a multi-scene video by generating each scene video via a diffusion process that takes the reference images, the descriptive prompt of the event and camera movement into account. The diffusion model incorporates the reference images as the condition and alignment to strengthen the content consistency of multi-scene videos. Extensive experiments demonstrate that VideoDrafter outperforms the SOTA video generation models in terms of visual quality, content consistency, and user preference.

{{</citation>}}


### (6/74) Whole-examination AI estimation of fetal biometrics from 20-week ultrasound scans (Lorenzo Venturini et al., 2024)

{{<citation>}}

Lorenzo Venturini, Samuel Budd, Alfonso Farruggia, Robert Wright, Jacqueline Matthew, Thomas G. Day, Bernhard Kainz, Reza Razavi, Jo V. Hajnal. (2024)  
**Whole-examination AI estimation of fetal biometrics from 20-week ultrasound scans**  

---
Primary Category: cs.CV  
Categories: I-4-7; J-3, cs-CV, cs-LG, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.01201v1)  

---


**ABSTRACT**  
The current approach to fetal anomaly screening is based on biometric measurements derived from individually selected ultrasound images. In this paper, we introduce a paradigm shift that attains human-level performance in biometric measurement by aggregating automatically extracted biometrics from every frame across an entire scan, with no need for operator intervention. We use a convolutional neural network to classify each frame of an ultrasound video recording. We then measure fetal biometrics in every frame where appropriate anatomy is visible. We use a Bayesian method to estimate the true value of each biometric from a large number of measurements and probabilistically reject outliers. We performed a retrospective experiment on 1457 recordings (comprising 48 million frames) of 20-week ultrasound scans, estimated fetal biometrics in those scans and compared our estimates to the measurements sonographers took during the scan. Our method achieves human-level performance in estimating fetal biometrics and estimates well-calibrated credible intervals in which the true biometric value is expected to lie.

{{</citation>}}


### (7/74) Freeze the backbones: A Parameter-Efficient Contrastive Approach to Robust Medical Vision-Language Pre-training (Jiuming Qin et al., 2024)

{{<citation>}}

Jiuming Qin, Che Liu, Sibo Cheng, Yike Guo, Rossella Arcucci. (2024)  
**Freeze the backbones: A Parameter-Efficient Contrastive Approach to Robust Medical Vision-Language Pre-training**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Self-Supervised, Transformer  
[Paper Link](http://arxiv.org/abs/2401.01179v1)  

---


**ABSTRACT**  
Modern healthcare often utilises radiographic images alongside textual reports for diagnostics, encouraging the use of Vision-Language Self-Supervised Learning (VL-SSL) with large pre-trained models to learn versatile medical vision representations. However, most existing VL-SSL frameworks are trained end-to-end, which is computation-heavy and can lose vital prior information embedded in pre-trained encoders. To address both issues, we introduce the backbone-agnostic Adaptor framework, which preserves medical knowledge in pre-trained image and text encoders by keeping them frozen, and employs a lightweight Adaptor module for cross-modal learning. Experiments on medical image classification and segmentation tasks across three datasets reveal that our framework delivers competitive performance while cutting trainable parameters by over 90% compared to current pre-training approaches. Notably, when fine-tuned with just 1% of data, Adaptor outperforms several Transformer-based methods trained on full datasets in medical image segmentation.

{{</citation>}}


### (8/74) GBSS:a global building semantic segmentation dataset for large-scale remote sensing building extraction (Yuping Hu et al., 2024)

{{<citation>}}

Yuping Hu, Xin Huang, Jiayi Li, Zhen Zhang. (2024)  
**GBSS:a global building semantic segmentation dataset for large-scale remote sensing building extraction**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2401.01178v1)  

---


**ABSTRACT**  
Semantic segmentation techniques for extracting building footprints from high-resolution remote sensing images have been widely used in many fields such as urban planning. However, large-scale building extraction demands higher diversity in training samples. In this paper, we construct a Global Building Semantic Segmentation (GBSS) dataset (The dataset will be released), which comprises 116.9k pairs of samples (about 742k buildings) from six continents. There are significant variations of building samples in terms of size and style, so the dataset can be a more challenging benchmark for evaluating the generalization and robustness of building semantic segmentation models. We validated through quantitative and qualitative comparisons between different datasets, and further confirmed the potential application in the field of transfer learning by conducting experiments on subsets.

{{</citation>}}


### (9/74) Hybrid Pooling and Convolutional Network for Improving Accuracy and Training Convergence Speed in Object Detection (Shiwen Zhao et al., 2024)

{{<citation>}}

Shiwen Zhao, Wei Wang, Junhui Hou, Hai Wu. (2024)  
**Hybrid Pooling and Convolutional Network for Improving Accuracy and Training Convergence Speed in Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2401.01134v1)  

---


**ABSTRACT**  
This paper introduces HPC-Net, a high-precision and rapidly convergent object detection network.

{{</citation>}}


### (10/74) SSP: A Simple and Safe automatic Prompt engineering method towards realistic image synthesis on LVM (Weijin Cheng et al., 2024)

{{<citation>}}

Weijin Cheng, Jianzhi Liu, Jiawen Deng, Fuji Ren. (2024)  
**SSP: A Simple and Safe automatic Prompt engineering method towards realistic image synthesis on LVM**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.01128v1)  

---


**ABSTRACT**  
Recently, text-to-image (T2I) synthesis has undergone significant advancements, particularly with the emergence of Large Language Models (LLM) and their enhancement in Large Vision Models (LVM), greatly enhancing the instruction-following capabilities of traditional T2I models. Nevertheless, previous methods focus on improving generation quality but introduce unsafe factors into prompts. We explore that appending specific camera descriptions to prompts can enhance safety performance. Consequently, we propose a simple and safe prompt engineering method (SSP) to improve image generation quality by providing optimal camera descriptions. Specifically, we create a dataset from multi-datasets as original prompts. To select the optimal camera, we design an optimal camera matching approach and implement a classifier for original prompts capable of automatically matching. Appending camera descriptions to original prompts generates optimized prompts for further LVM image generation. Experiments demonstrate that SSP improves semantic consistency by an average of 16% compared to others and safety metrics by 48.9%.

{{</citation>}}


### (11/74) Q-Refine: A Perceptual Quality Refiner for AI-Generated Image (Chunyi Li et al., 2024)

{{<citation>}}

Chunyi Li, Haoning Wu, Zicheng Zhang, Hongkun Hao, Kaiwei Zhang, Lei Bai, Xiaohong Liu, Xiongkuo Min, Weisi Lin, Guangtao Zhai. (2024)  
**Q-Refine: A Perceptual Quality Refiner for AI-Generated Image**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: AI, QA  
[Paper Link](http://arxiv.org/abs/2401.01117v1)  

---


**ABSTRACT**  
With the rapid evolution of the Text-to-Image (T2I) model in recent years, their unsatisfactory generation result has become a challenge. However, uniformly refining AI-Generated Images (AIGIs) of different qualities not only limited optimization capabilities for low-quality AIGIs but also brought negative optimization to high-quality AIGIs. To address this issue, a quality-award refiner named Q-Refine is proposed. Based on the preference of the Human Visual System (HVS), Q-Refine uses the Image Quality Assessment (IQA) metric to guide the refining process for the first time, and modify images of different qualities through three adaptive pipelines. Experimental shows that for mainstream T2I models, Q-Refine can perform effective optimization to AIGIs of different qualities. It can be a general refiner to optimize AIGIs from both fidelity and aesthetic quality levels, thus expanding the application of the T2I generation models.

{{</citation>}}


### (12/74) CityPulse: Fine-Grained Assessment of Urban Change with Street View Time Series (Tianyuan Huang et al., 2024)

{{<citation>}}

Tianyuan Huang, Zejia Wu, Jiajun Wu, Jackelyn Hwang, Ram Rajagopal. (2024)  
**CityPulse: Fine-Grained Assessment of Urban Change with Street View Time Series**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2401.01107v2)  

---


**ABSTRACT**  
Urban transformations have profound societal impact on both individuals and communities at large. Accurately assessing these shifts is essential for understanding their underlying causes and ensuring sustainable urban planning. Traditional measurements often encounter constraints in spatial and temporal granularity, failing to capture real-time physical changes. While street view imagery, capturing the heartbeat of urban spaces from a pedestrian point of view, can add as a high-definition, up-to-date, and on-the-ground visual proxy of urban change. We curate the largest street view time series dataset to date, and propose an end-to-end change detection model to effectively capture physical alterations in the built environment at scale. We demonstrate the effectiveness of our proposed method by benchmark comparisons with previous literature and implementing it at the city-wide level. Our approach has the potential to supplement existing dataset and serve as a fine-grained and accurate assessment of urban change.

{{</citation>}}


### (13/74) Dual Teacher Knowledge Distillation with Domain Alignment for Face Anti-spoofing (Zhe Kong et al., 2024)

{{<citation>}}

Zhe Kong, Wentian Zhang, Tao Wang, Kaihao Zhang, Yuexiang Li, Xiaoying Tang, Wenhan Luo. (2024)  
**Dual Teacher Knowledge Distillation with Domain Alignment for Face Anti-spoofing**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2401.01102v1)  

---


**ABSTRACT**  
Face recognition systems have raised concerns due to their vulnerability to different presentation attacks, and system security has become an increasingly critical concern. Although many face anti-spoofing (FAS) methods perform well in intra-dataset scenarios, their generalization remains a challenge. To address this issue, some methods adopt domain adversarial training (DAT) to extract domain-invariant features. However, the competition between the encoder and the domain discriminator can cause the network to be difficult to train and converge. In this paper, we propose a domain adversarial attack (DAA) method to mitigate the training instability problem by adding perturbations to the input images, which makes them indistinguishable across domains and enables domain alignment. Moreover, since models trained on limited data and types of attacks cannot generalize well to unknown attacks, we propose a dual perceptual and generative knowledge distillation framework for face anti-spoofing that utilizes pre-trained face-related models containing rich face priors. Specifically, we adopt two different face-related models as teachers to transfer knowledge to the target student model. The pre-trained teacher models are not from the task of face anti-spoofing but from perceptual and generative tasks, respectively, which implicitly augment the data. By combining both DAA and dual-teacher knowledge distillation, we develop a dual teacher knowledge distillation with domain alignment framework (DTDA) for face anti-spoofing. The advantage of our proposed method has been verified through extensive ablation studies and comparison with state-of-the-art methods on public datasets across multiple protocols.

{{</citation>}}


### (14/74) Exploring Hyperspectral Anomaly Detection with Human Vision: A Small Target Aware Detector (Jitao Ma et al., 2024)

{{<citation>}}

Jitao Ma, Weiying Xie, Yunsong Li. (2024)  
**Exploring Hyperspectral Anomaly Detection with Human Vision: A Small Target Aware Detector**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2401.01093v1)  

---


**ABSTRACT**  
Hyperspectral anomaly detection (HAD) aims to localize pixel points whose spectral features differ from the background. HAD is essential in scenarios of unknown or camouflaged target features, such as water quality monitoring, crop growth monitoring and camouflaged target detection, where prior information of targets is difficult to obtain. Existing HAD methods aim to objectively detect and distinguish background and anomalous spectra, which can be achieved almost effortlessly by human perception. However, the underlying processes of human visual perception are thought to be quite complex. In this paper, we analyze hyperspectral image (HSI) features under human visual perception, and transfer the solution process of HAD to the more robust feature space for the first time. Specifically, we propose a small target aware detector (STAD), which introduces saliency maps to capture HSI features closer to human visual perception. STAD not only extracts more anomalous representations, but also reduces the impact of low-confidence regions through a proposed small target filter (STF). Furthermore, considering the possibility of HAD algorithms being applied to edge devices, we propose a full connected network to convolutional network knowledge distillation strategy. It can learn the spectral and spatial features of the HSI while lightening the network. We train the network on the HAD100 training set and validate the proposed method on the HAD100 test set. Our method provides a new solution space for HAD that is closer to human visual perception with high confidence. Sufficient experiments on real HSI with multiple method comparisons demonstrate the excellent performance and unique potential of the proposed method. The code is available at https://github.com/majitao-xd/STAD-HAD.

{{</citation>}}


### (15/74) Depth-discriminative Metric Learning for Monocular 3D Object Detection (Wonhyeok Choi et al., 2024)

{{<citation>}}

Wonhyeok Choi, Mingyu Shin, Sunghoon Im. (2024)  
**Depth-discriminative Metric Learning for Monocular 3D Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2401.01075v1)  

---


**ABSTRACT**  
Monocular 3D object detection poses a significant challenge due to the lack of depth information in RGB images. Many existing methods strive to enhance the object depth estimation performance by allocating additional parameters for object depth estimation, utilizing extra modules or data. In contrast, we introduce a novel metric learning scheme that encourages the model to extract depth-discriminative features regardless of the visual attributes without increasing inference time and model size. Our method employs the distance-preserving function to organize the feature space manifold in relation to ground-truth object depth. The proposed (K, B, eps)-quasi-isometric loss leverages predetermined pairwise distance restriction as guidance for adjusting the distance among object descriptors without disrupting the non-linearity of the natural feature manifold. Moreover, we introduce an auxiliary head for object-wise depth estimation, which enhances depth quality while maintaining the inference time. The broad applicability of our method is demonstrated through experiments that show improvements in overall performance when integrated into various baselines. The results show that our method consistently improves the performance of various baselines by 23.51% and 5.78% on average across KITTI and Waymo, respectively.

{{</citation>}}


### (16/74) DTBS: Dual-Teacher Bi-directional Self-training for Domain Adaptation in Nighttime Semantic Segmentation (Fanding Huang et al., 2024)

{{<citation>}}

Fanding Huang, Zihao Yao, Wenhui Zhou. (2024)  
**DTBS: Dual-Teacher Bi-directional Self-training for Domain Adaptation in Nighttime Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2401.01066v1)  

---


**ABSTRACT**  
Due to the poor illumination and the difficulty in annotating, nighttime conditions pose a significant challenge for autonomous vehicle perception systems. Unsupervised domain adaptation (UDA) has been widely applied to semantic segmentation on such images to adapt models from normal conditions to target nighttime-condition domains. Self-training (ST) is a paradigm in UDA, where a momentum teacher is utilized for pseudo-label prediction, but a confirmation bias issue exists. Because the one-directional knowledge transfer from a single teacher is insufficient to adapt to a large domain shift. To mitigate this issue, we propose to alleviate domain gap by incrementally considering style influence and illumination change. Therefore, we introduce a one-stage Dual-Teacher Bi-directional Self-training (DTBS) framework for smooth knowledge transfer and feedback. Based on two teacher models, we present a novel pipeline to respectively decouple style and illumination shift. In addition, we propose a new Re-weight exponential moving average (EMA) to merge the knowledge of style and illumination factors, and provide feedback to the student model. In this way, our method can be embedded in other UDA methods to enhance their performance. For example, the Cityscapes to ACDC night task yielded 53.8 mIoU (\%), which corresponds to an improvement of +5\% over the previous state-of-the-art. The code is available at \url{https://github.com/hf618/DTBS}.

{{</citation>}}


### (17/74) Relating Events and Frames Based on Self-Supervised Learning and Uncorrelated Conditioning for Unsupervised Domain Adaptation (Mohammad Rostami et al., 2024)

{{<citation>}}

Mohammad Rostami, Dayuan Jian. (2024)  
**Relating Events and Frames Based on Self-Supervised Learning and Uncorrelated Conditioning for Unsupervised Domain Adaptation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2401.01042v1)  

---


**ABSTRACT**  
Event-based cameras provide accurate and high temporal resolution measurements for performing computer vision tasks in challenging scenarios, such as high-dynamic range environments and fast-motion maneuvers. Despite their advantages, utilizing deep learning for event-based vision encounters a significant obstacle due to the scarcity of annotated data caused by the relatively recent emergence of event-based cameras. To overcome this limitation, leveraging the knowledge available from annotated data obtained with conventional frame-based cameras presents an effective solution based on unsupervised domain adaptation. We propose a new algorithm tailored for adapting a deep neural network trained on annotated frame-based data to generalize well on event-based unannotated data. Our approach incorporates uncorrelated conditioning and self-supervised learning in an adversarial learning scheme to close the gap between the two source and target domains. By applying self-supervised learning, the algorithm learns to align the representations of event-based data with those from frame-based camera data, thereby facilitating knowledge transfer.Furthermore, the inclusion of uncorrelated conditioning ensures that the adapted model effectively distinguishes between event-based and conventional data, enhancing its ability to classify event-based images accurately.Through empirical experimentation and evaluation, we demonstrate that our algorithm surpasses existing approaches designed for the same purpose using two benchmarks. The superior performance of our solution is attributed to its ability to effectively utilize annotated data from frame-based cameras and transfer the acquired knowledge to the event-based vision domain.

{{</citation>}}


### (18/74) Unsupervised Continual Anomaly Detection with Contrastively-learned Prompt (Jiaqi Liu et al., 2024)

{{<citation>}}

Jiaqi Liu, Kai Wu, Qiang Nie, Ying Chen, Bin-Bin Gao, Yong Liu, Jinbao Wang, Chengjie Wang, Feng Zheng. (2024)  
**Unsupervised Continual Anomaly Detection with Contrastively-learned Prompt**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Anomaly Detection, Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2401.01010v1)  

---


**ABSTRACT**  
Unsupervised Anomaly Detection (UAD) with incremental training is crucial in industrial manufacturing, as unpredictable defects make obtaining sufficient labeled data infeasible. However, continual learning methods primarily rely on supervised annotations, while the application in UAD is limited due to the absence of supervision. Current UAD methods train separate models for different classes sequentially, leading to catastrophic forgetting and a heavy computational burden. To address this issue, we introduce a novel Unsupervised Continual Anomaly Detection framework called UCAD, which equips the UAD with continual learning capability through contrastively-learned prompts. In the proposed UCAD, we design a Continual Prompting Module (CPM) by utilizing a concise key-prompt-knowledge memory bank to guide task-invariant `anomaly' model predictions using task-specific `normal' knowledge. Moreover, Structure-based Contrastive Learning (SCL) is designed with the Segment Anything Model (SAM) to improve prompt learning and anomaly segmentation results. Specifically, by treating SAM's masks as structure, we draw features within the same mask closer and push others apart for general feature representations. We conduct comprehensive experiments and set the benchmark on unsupervised continual anomaly detection and segmentation, demonstrating that our method is significantly better than anomaly detection methods, even with rehearsal training. The code will be available at https://github.com/shirowalker/UCAD.

{{</citation>}}


### (19/74) Diversity-aware Buffer for Coping with Temporally Correlated Data Streams in Online Test-time Adaptation (Mario Döbler et al., 2024)

{{<citation>}}

Mario Döbler, Florian Marencke, Robert A. Marsden, Bin Yang. (2024)  
**Diversity-aware Buffer for Coping with Temporally Correlated Data Streams in Online Test-time Adaptation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2401.00989v1)  

---


**ABSTRACT**  
Since distribution shifts are likely to occur after a model's deployment and can drastically decrease the model's performance, online test-time adaptation (TTA) continues to update the model during test-time, leveraging the current test data. In real-world scenarios, test data streams are not always independent and identically distributed (i.i.d.). Instead, they are frequently temporally correlated, making them non-i.i.d. Many existing methods struggle to cope with this scenario. In response, we propose a diversity-aware and category-balanced buffer that can simulate an i.i.d. data stream, even in non-i.i.d. scenarios. Combined with a diversity and entropy-weighted entropy loss, we show that a stable adaptation is possible on a wide range of corruptions and natural domain shifts, based on ImageNet. We achieve state-of-the-art results on most considered benchmarks.

{{</citation>}}


### (20/74) Holistic Autonomous Driving Understanding by Bird's-Eye-View Injected Multi-Modal Large Models (Xinpeng Ding et al., 2024)

{{<citation>}}

Xinpeng Ding, Jinahua Han, Hang Xu, Xiaodan Liang, Wei Zhang, Xiaomeng Li. (2024)  
**Holistic Autonomous Driving Understanding by Bird's-Eye-View Injected Multi-Modal Large Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2401.00988v1)  

---


**ABSTRACT**  
The rise of multimodal large language models (MLLMs) has spurred interest in language-based driving tasks. However, existing research typically focuses on limited tasks and often omits key multi-view and temporal information which is crucial for robust autonomous driving. To bridge these gaps, we introduce NuInstruct, a novel dataset with 91K multi-view video-QA pairs across 17 subtasks, where each task demands holistic information (e.g., temporal, multi-view, and spatial), significantly elevating the challenge level. To obtain NuInstruct, we propose a novel SQL-based method to generate instruction-response pairs automatically, which is inspired by the driving logical progression of humans. We further present BEV-InMLLM, an end-to-end method for efficiently deriving instruction-aware Bird's-Eye-View (BEV) features, language-aligned for large language models. BEV-InMLLM integrates multi-view, spatial awareness, and temporal semantics to enhance MLLMs' capabilities on NuInstruct tasks. Moreover, our proposed BEV injection module is a plug-and-play method for existing MLLMs. Our experiments on NuInstruct demonstrate that BEV-InMLLM significantly outperforms existing MLLMs, e.g. around 9% improvement on various tasks. We plan to release our NuInstruct for future research development.

{{</citation>}}


### (21/74) Real-Time Object Detection in Occluded Environment with Background Cluttering Effects Using Deep Learning (Syed Muhammad Aamir et al., 2024)

{{<citation>}}

Syed Muhammad Aamir, Hongbin Ma, Malak Abid Ali Khan, Muhammad Aaqib. (2024)  
**Real-Time Object Detection in Occluded Environment with Background Cluttering Effects Using Deep Learning**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2401.00986v1)  

---


**ABSTRACT**  
Detection of small, undetermined moving objects or objects in an occluded environment with a cluttered background is the main problem of computer vision. This greatly affects the detection accuracy of deep learning models. To overcome these problems, we concentrate on deep learning models for real-time detection of cars and tanks in an occluded environment with a cluttered background employing SSD and YOLO algorithms and improved precision of detection and reduce problems faced by these models. The developed method makes the custom dataset and employs a preprocessing technique to clean the noisy dataset. For training the developed model we apply the data augmentation technique to balance and diversify the data. We fine-tuned, trained, and evaluated these models on the established dataset by applying these techniques and highlighting the results we got more accurately than without applying these techniques. The accuracy and frame per second of the SSD-Mobilenet v2 model are higher than YOLO V3 and YOLO V4. Furthermore, by employing various techniques like data enhancement, noise reduction, parameter optimization, and model fusion we improve the effectiveness of detection and recognition. We further added a counting algorithm, and target attributes experimental comparison, and made a graphical user interface system for the developed model with features of object counting, alerts, status, resolution, and frame per second. Subsequently, to justify the importance of the developed method analysis of YOLO V3, V4, and SSD were incorporated. Which resulted in the overall completion of the proposed method.

{{</citation>}}


### (22/74) Unsupervised Federated Domain Adaptation for Segmentation of MRI Images (Navapat Nananukul et al., 2024)

{{<citation>}}

Navapat Nananukul, Hamid Soltanian-zadeh, Mohammad Rostami. (2024)  
**Unsupervised Federated Domain Adaptation for Segmentation of MRI Images**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.02941v1)  

---


**ABSTRACT**  
Automatic semantic segmentation of magnetic resonance imaging (MRI) images using deep neural networks greatly assists in evaluating and planning treatments for various clinical applications. However, training these models is conditioned on the availability of abundant annotated data to implement the end-to-end supervised learning procedure. Even if we annotate enough data, MRI images display considerable variability due to factors such as differences in patients, MRI scanners, and imaging protocols. This variability necessitates retraining neural networks for each specific application domain, which, in turn, requires manual annotation by expert radiologists for all new domains. To relax the need for persistent data annotation, we develop a method for unsupervised federated domain adaptation using multiple annotated source domains. Our approach enables the transfer of knowledge from several annotated source domains to adapt a model for effective use in an unannotated target domain. Initially, we ensure that the target domain data shares similar representations with each source domain in a latent embedding space, modeled as the output of a deep encoder, by minimizing the pair-wise distances of the distributions for the target domain and the source domains. We then employ an ensemble approach to leverage the knowledge obtained from all domains. We provide theoretical analysis and perform experiments on the MICCAI 2016 multi-site dataset to demonstrate our method is effective.

{{</citation>}}


## cs.IT (3)



### (23/74) Multiple Access Techniques for Intelligent and Multi-Functional 6G: Tutorial, Survey, and Outlook (Bruno Clerckx et al., 2024)

{{<citation>}}

Bruno Clerckx, Yijie Mao, Zhaohui Yang, Mingzhe Chen, Ahmed Alkhateeb, Liang Liu, Min Qiu, Jinhong Yuan, Vincent W. S. Wong, Juan Montojo. (2024)  
**Multiple Access Techniques for Intelligent and Multi-Functional 6G: Tutorial, Survey, and Outlook**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, eess-SP, math-IT  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.01433v1)  

---


**ABSTRACT**  
Multiple access (MA) is a crucial part of any wireless system and refers to techniques that make use of the resource dimensions to serve multiple users/devices/machines/services, ideally in the most efficient way. Given the needs of multi-functional wireless networks for integrated communications, sensing, localization, computing, coupled with the surge of machine learning / artificial intelligence (AI) in wireless networks, MA techniques are expected to experience a paradigm shift in 6G and beyond. In this paper, we provide a tutorial, survey and outlook of past, emerging and future MA techniques and pay a particular attention to how wireless network intelligence and multi-functionality will lead to a re-thinking of those techniques. The paper starts with an overview of orthogonal, physical layer multicasting, space domain, power domain, ratesplitting, code domain MAs, and other domains, and highlight the importance of researching universal multiple access to shrink instead of grow the knowledge tree of MA schemes by providing a unified understanding of MA schemes across all resource dimensions. It then jumps into rethinking MA schemes in the era of wireless network intelligence, covering AI for MA such as AI-empowered resource allocation, optimization, channel estimation, receiver designs, user behavior predictions, and MA for AI such as federated learning/edge intelligence and over the air computation. We then discuss MA for network multi-functionality and the interplay between MA and integrated sensing, localization, and communications. We finish with studying MA for emerging intelligent applications before presenting a roadmap toward 6G standardization. We also point out numerous directions that are promising for future research.

{{</citation>}}


### (24/74) Joint Offloading and Resource Allocation for Hybrid Cloud and Edge Computing in SAGINs: A Decision Assisted Hybrid Action Space Deep Reinforcement Learning Approach (Chong Huang et al., 2024)

{{<citation>}}

Chong Huang, Gaojie Chen, Pei Xiao, Yue Xiao, Zhu Han, Jonathon A. Chambers. (2024)  
**Joint Offloading and Resource Allocation for Hybrid Cloud and Edge Computing in SAGINs: A Decision Assisted Hybrid Action Space Deep Reinforcement Learning Approach**  

---
Primary Category: cs.IT  
Categories: cs-DC, cs-IT, cs.IT, math-IT  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.01140v1)  

---


**ABSTRACT**  
In recent years, the amalgamation of satellite communications and aerial platforms into space-air-ground integrated network (SAGINs) has emerged as an indispensable area of research for future communications due to the global coverage capacity of low Earth orbit (LEO) satellites and the flexible Deployment of aerial platforms. This paper presents a deep reinforcement learning (DRL)-based approach for the joint optimization of offloading and resource allocation in hybrid cloud and multi-access edge computing (MEC) scenarios within SAGINs. The proposed system considers the presence of multiple satellites, clouds and unmanned aerial vehicles (UAVs). The multiple tasks from ground users are modeled as directed acyclic graphs (DAGs). With the goal of reducing energy consumption and latency in MEC, we propose a novel multi-agent algorithm based on DRL that optimizes both the offloading strategy and the allocation of resources in the MEC infrastructure within SAGIN. A hybrid action algorithm is utilized to address the challenge of hybrid continuous and discrete action space in the proposed problems, and a decision-assisted DRL method is adopted to reduce the impact of unavailable actions in the training process of DRL. Through extensive simulations, the results demonstrate the efficacy of the proposed learning-based scheme, the proposed approach consistently outperforms benchmark schemes, highlighting its superior performance and potential for practical applications.

{{</citation>}}


### (25/74) Wireless 6G Connectivity for Massive Number of Devices and Critical Services (Anders E. Kalør et al., 2024)

{{<citation>}}

Anders E. Kalør, Giuseppe Duris, Sinem Coleri, Stefan Parkvall, Wei Yu, Andreas Mueller, Petar Popovski. (2024)  
**Wireless 6G Connectivity for Massive Number of Devices and Critical Services**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, math-IT  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.01127v1)  

---


**ABSTRACT**  
Compared to the generations up to 4G, whose main focus was on broadband and coverage aspects, 5G has expanded the scope of wireless cellular systems towards embracing two new types of connectivity: massive machine-type communication (mMTC) and ultra-reliable low-latency communications (URLLC). This paper will discuss the possible evolution of these two types of connectivity within the umbrella of 6G wireless systems. The paper consists of three parts. The first part deals with the connectivity for a massive number of devices. While mMTC research in 5G was predominantly focused on the problem of uncoordinated access in the uplink for a large number of devices, the traffic patterns in 6G may become more symmetric, leading to closed-loop massive connectivity. One of the drivers for this is distributed learning/inference. The second part of the paper will discuss the evolution of wireless connectivity for critical services. While latency and reliability are tightly coupled in 5G, 6G will support a variety of safety critical control applications with different types of timing requirements, as evidenced by the emergence of metrics related to information freshness and information value. Additionally, ensuring ultra-high reliability for safety critical control applications requires modeling and estimation of the tail statistics of the wireless channel, queue length, and delay. The fulfillment of these stringent requirements calls for the development of novel AI-based techniques, incorporating optimization theory, explainable AI, generative AI and digital twins. The third part will analyze the coexistence of massive connectivity and critical services. We will consider scenarios in which a massive number of devices need to support traffic patterns of mixed criticality. This will be followed by a discussion about the management of wireless resources shared by services with different criticality.

{{</citation>}}


## cs.AI (3)



### (26/74) SwapTransformer: highway overtaking tactical planner model via imitation learning on OSHA dataset (Alireza Shamsoshoara et al., 2024)

{{<citation>}}

Alireza Shamsoshoara, Safin B Salih, Pedram Aghazadeh. (2024)  
**SwapTransformer: highway overtaking tactical planner model via imitation learning on OSHA dataset**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CV, cs-LG, cs-RO, cs.AI  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.01425v1)  

---


**ABSTRACT**  
This paper investigates the high-level decision-making problem in highway scenarios regarding lane changing and over-taking other slower vehicles. In particular, this paper aims to improve the Travel Assist feature for automatic overtaking and lane changes on highways. About 9 million samples including lane images and other dynamic objects are collected in simulation. This data; Overtaking on Simulated HighwAys (OSHA) dataset is released to tackle this challenge. To solve this problem, an architecture called SwapTransformer is designed and implemented as an imitation learning approach on the OSHA dataset. Moreover, auxiliary tasks such as future points and car distance network predictions are proposed to aid the model in better understanding the surrounding environment. The performance of the proposed solution is compared with a multi-layer perceptron (MLP) and multi-head self-attention networks as baselines in a simulation environment. We also demonstrate the performance of the model with and without auxiliary tasks. All models are evaluated based on different metrics such as time to finish each lap, number of overtakes, and speed difference with speed limit. The evaluation shows that the SwapTransformer model outperforms other models in different traffic densities in the inference phase.

{{</citation>}}


### (27/74) Towards Cognitive AI Systems: a Survey and Prospective on Neuro-Symbolic AI (Zishen Wan et al., 2024)

{{<citation>}}

Zishen Wan, Che-Kai Liu, Hanchen Yang, Chaojian Li, Haoran You, Yonggan Fu, Cheng Wan, Tushar Krishna, Yingyan Lin, Arijit Raychowdhury. (2024)  
**Towards Cognitive AI Systems: a Survey and Prospective on Neuro-Symbolic AI**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-AR, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.01040v1)  

---


**ABSTRACT**  
The remarkable advancements in artificial intelligence (AI), primarily driven by deep neural networks, have significantly impacted various aspects of our lives. However, the current challenges surrounding unsustainable computational trajectories, limited robustness, and a lack of explainability call for the development of next-generation AI systems. Neuro-symbolic AI (NSAI) emerges as a promising paradigm, fusing neural, symbolic, and probabilistic approaches to enhance interpretability, robustness, and trustworthiness while facilitating learning from much less data. Recent NSAI systems have demonstrated great potential in collaborative human-AI scenarios with reasoning and cognitive capabilities. In this paper, we provide a systematic review of recent progress in NSAI and analyze the performance characteristics and computational operators of NSAI models. Furthermore, we discuss the challenges and potential future directions of NSAI from both system and architectural perspectives.

{{</citation>}}


### (28/74) Safety and Performance, Why Not Both? Bi-Objective Optimized Model Compression against Heterogeneous Attacks Toward AI Software Deployment (Jie Zhu et al., 2024)

{{<citation>}}

Jie Zhu, Leye Wang, Xiao Han, Anmin Liu, Tao Xie. (2024)  
**Safety and Performance, Why Not Both? Bi-Objective Optimized Model Compression against Heterogeneous Attacks Toward AI Software Deployment**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CR, cs-SE, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.00996v1)  

---


**ABSTRACT**  
The size of deep learning models in artificial intelligence (AI) software is increasing rapidly, hindering the large-scale deployment on resource-restricted devices (e.g., smartphones). To mitigate this issue, AI software compression plays a crucial role, which aims to compress model size while keeping high performance. However, the intrinsic defects in a big model may be inherited by the compressed one. Such defects may be easily leveraged by adversaries, since a compressed model is usually deployed in a large number of devices without adequate protection. In this article, we aim to address the safe model compression problem from the perspective of safety-performance co-optimization. Specifically, inspired by the test-driven development (TDD) paradigm in software engineering, we propose a test-driven sparse training framework called SafeCompress. By simulating the attack mechanism as safety testing, SafeCompress can automatically compress a big model to a small one following the dynamic sparse training paradigm. Then, considering two kinds of representative and heterogeneous attack mechanisms, i.e., black-box membership inference attack and white-box membership inference attack, we develop two concrete instances called BMIA-SafeCompress and WMIA-SafeCompress. Further, we implement another instance called MMIA-SafeCompress by extending SafeCompress to defend against the occasion when adversaries conduct black-box and white-box membership inference attacks simultaneously. We conduct extensive experiments on five datasets for both computer vision and natural language processing tasks. The results show the effectiveness and generalizability of our framework. We also discuss how to adapt SafeCompress to other attacks besides membership inference attack, demonstrating the flexibility of SafeCompress.

{{</citation>}}


## cs.CL (18)



### (29/74) To Diverge or Not to Diverge: A Morphosyntactic Perspective on Machine Translation vs Human Translation (Jiaming Luo et al., 2024)

{{<citation>}}

Jiaming Luo, Colin Cherry, George Foster. (2024)  
**To Diverge or Not to Diverge: A Morphosyntactic Perspective on Machine Translation vs Human Translation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Machine Translation  
[Paper Link](http://arxiv.org/abs/2401.01419v1)  

---


**ABSTRACT**  
We conduct a large-scale fine-grained comparative analysis of machine translations (MT) against human translations (HT) through the lens of morphosyntactic divergence. Across three language pairs and two types of divergence defined as the structural difference between the source and the target, MT is consistently more conservative than HT, with less morphosyntactic diversity, more convergent patterns, and more one-to-one alignments. Through analysis on different decoding algorithms, we attribute this discrepancy to the use of beam search that biases MT towards more convergent patterns. This bias is most amplified when the convergent pattern appears around 50% of the time in training data. Lastly, we show that for a majority of morphosyntactic divergences, their presence in HT is correlated with decreased MT performance, presenting a greater challenge for MT systems.

{{</citation>}}


### (30/74) An Autoregressive Text-to-Graph Framework for Joint Entity and Relation Extraction (Zaratiana Urchade et al., 2024)

{{<citation>}}

Zaratiana Urchade, Nadi Tomeh, Pierre Holat, Thierry Charnois. (2024)  
**An Autoregressive Text-to-Graph Framework for Joint Entity and Relation Extraction**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Relation Extraction  
[Paper Link](http://arxiv.org/abs/2401.01326v1)  

---


**ABSTRACT**  
In this paper, we propose a novel method for joint entity and relation extraction from unstructured text by framing it as a conditional sequence generation problem. In contrast to conventional generative information extraction models that are left-to-right token-level generators, our approach is \textit{span-based}. It generates a linearized graph where nodes represent text spans and edges represent relation triplets. Our method employs a transformer encoder-decoder architecture with pointing mechanism on a dynamic vocabulary of spans and relation types. Our model can capture the structural characteristics and boundaries of entities and relations through span representations while simultaneously grounding the generated output in the original text thanks to the pointing mechanism. Evaluation on benchmark datasets validates the effectiveness of our approach, demonstrating competitive results. Code is available at https://github.com/urchade/ATG.

{{</citation>}}


### (31/74) LLM Maybe LongLM: Self-Extend LLM Context Window Without Tuning (Hongye Jin et al., 2024)

{{<citation>}}

Hongye Jin, Xiaotian Han, Jingfeng Yang, Zhimeng Jiang, Zirui Liu, Chia-Yuan Chang, Huiyuan Chen, Xia Hu. (2024)  
**LLM Maybe LongLM: Self-Extend LLM Context Window Without Tuning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.01325v1)  

---


**ABSTRACT**  
This work elicits LLMs' inherent ability to handle long contexts without fine-tuning. The limited length of the training sequence during training may limit the application of Large Language Models (LLMs) on long input sequences for inference. In this work, we argue that existing LLMs themselves have inherent capabilities for handling long contexts. Based on this argument, we suggest extending LLMs' context window by themselves to fully utilize the inherent ability.We propose Self-Extend to stimulate LLMs' long context handling potential. The basic idea is to construct bi-level attention information: the group level and the neighbor level. The two levels are computed by the original model's self-attention, which means the proposed does not require any training. With only four lines of code modification, the proposed method can effortlessly extend existing LLMs' context window without any fine-tuning. We conduct comprehensive experiments and the results show that the proposed method can effectively extend existing LLMs' context window's length.

{{</citation>}}


### (32/74) A Comprehensive Survey of Hallucination Mitigation Techniques in Large Language Models (S. M Towhidul Islam Tonmoy et al., 2024)

{{<citation>}}

S. M Towhidul Islam Tonmoy, S M Mehedi Zaman, Vinija Jain, Anku Rani, Vipula Rawte, Aman Chadha, Amitava Das. (2024)  
**A Comprehensive Survey of Hallucination Mitigation Techniques in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Language Model, NLI  
[Paper Link](http://arxiv.org/abs/2401.01313v2)  

---


**ABSTRACT**  
As Large Language Models (LLMs) continue to advance in their ability to write human-like text, a key challenge remains around their tendency to hallucinate generating content that appears factual but is ungrounded. This issue of hallucination is arguably the biggest hindrance to safely deploying these powerful LLMs into real-world production systems that impact people's lives. The journey toward widespread adoption of LLMs in practical settings heavily relies on addressing and mitigating hallucinations. Unlike traditional AI systems focused on limited tasks, LLMs have been exposed to vast amounts of online text data during training. While this allows them to display impressive language fluency, it also means they are capable of extrapolating information from the biases in training data, misinterpreting ambiguous prompts, or modifying the information to align superficially with the input. This becomes hugely alarming when we rely on language generation capabilities for sensitive applications, such as summarizing medical records, financial analysis reports, etc. This paper presents a comprehensive survey of over 32 techniques developed to mitigate hallucination in LLMs. Notable among these are Retrieval Augmented Generation (Lewis et al, 2021), Knowledge Retrieval (Varshney et al,2023), CoNLI (Lei et al, 2023), and CoVe (Dhuliawala et al, 2023). Furthermore, we introduce a detailed taxonomy categorizing these methods based on various parameters, such as dataset utilization, common tasks, feedback mechanisms, and retriever types. This classification helps distinguish the diverse approaches specifically designed to tackle hallucination issues in LLMs. Additionally, we analyze the challenges and limitations inherent in these techniques, providing a solid foundation for future research in addressing hallucinations and related phenomena within the realm of LLMs.

{{</citation>}}


### (33/74) Large Legal Fictions: Profiling Legal Hallucinations in Large Language Models (Matthew Dahl et al., 2024)

{{<citation>}}

Matthew Dahl, Varun Magesh, Mirac Suzgun, Daniel E. Ho. (2024)  
**Large Legal Fictions: Profiling Legal Hallucinations in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CY, cs.CL  
Keywords: ChatGPT, GPT, Language Model, Legal  
[Paper Link](http://arxiv.org/abs/2401.01301v1)  

---


**ABSTRACT**  
Large language models (LLMs) have the potential to transform the practice of law, but this potential is threatened by the presence of legal hallucinations -- responses from these models that are not consistent with legal facts. We investigate the extent of these hallucinations using an original suite of legal queries, comparing LLMs' responses to structured legal metadata and examining their consistency. Our work makes four key contributions: (1) We develop a typology of legal hallucinations, providing a conceptual framework for future research in this area. (2) We find that legal hallucinations are alarmingly prevalent, occurring between 69% of the time with ChatGPT 3.5 and 88% with Llama 2, when these models are asked specific, verifiable questions about random federal court cases. (3) We illustrate that LLMs often fail to correct a user's incorrect legal assumptions in a contra-factual question setup. (4) We provide evidence that LLMs cannot always predict, or do not always know, when they are producing legal hallucinations. Taken together, these findings caution against the rapid and unsupervised integration of popular LLMs into legal tasks. Even experienced lawyers must remain wary of legal hallucinations, and the risks are highest for those who stand to benefit from LLMs the most -- pro se litigants or those without access to traditional legal resources.

{{</citation>}}


### (34/74) A Comprehensive Study of Knowledge Editing for Large Language Models (Ningyu Zhang et al., 2024)

{{<citation>}}

Ningyu Zhang, Yunzhi Yao, Bozhong Tian, Peng Wang, Shumin Deng, Mengru Wang, Zekun Xi, Shengyu Mao, Jintian Zhang, Yuansheng Ni, Siyuan Cheng, Ziwen Xu, Xin Xu, Jia-Chen Gu, Yong Jiang, Pengjun Xie, Fei Huang, Lei Liang, Zhiqiang Zhang, Xiaowei Zhu, Jun Zhou, Huajun Chen. (2024)  
**A Comprehensive Study of Knowledge Editing for Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CV, cs-HC, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.01286v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have shown extraordinary capabilities in understanding and generating text that closely mirrors human communication. However, a primary limitation lies in the significant computational demands during training, arising from their extensive parameterization. This challenge is further intensified by the dynamic nature of the world, necessitating frequent updates to LLMs to correct outdated information or integrate new knowledge, thereby ensuring their continued relevance. Note that many applications demand continual model adjustments post-training to address deficiencies or undesirable behaviors. There is an increasing interest in efficient, lightweight methods for on-the-fly model modifications. To this end, recent years have seen a burgeoning in the techniques of knowledge editing for LLMs, which aim to efficiently modify LLMs' behaviors within specific domains while preserving overall performance across various inputs. In this paper, we first define the knowledge editing problem and then provide a comprehensive review of cutting-edge approaches. Drawing inspiration from educational and cognitive research theories, we propose a unified categorization criterion that classifies knowledge editing methods into three groups: resorting to external knowledge, merging knowledge into the model, and editing intrinsic knowledge. Furthermore, we introduce a new benchmark, KnowEdit, for a comprehensive empirical evaluation of representative knowledge editing approaches. Additionally, we provide an in-depth analysis of knowledge location, which can provide a deeper understanding of the knowledge structures inherent within LLMs. Finally, we discuss several potential applications of knowledge editing, outlining its broad and impactful implications.

{{</citation>}}


### (35/74) Quality and Quantity of Machine Translation References for Automated Metrics (Vilém Zouhar et al., 2024)

{{<citation>}}

Vilém Zouhar, Ondřej Bojar. (2024)  
**Quality and Quantity of Machine Translation References for Automated Metrics**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Machine Translation  
[Paper Link](http://arxiv.org/abs/2401.01283v2)  

---


**ABSTRACT**  
Automatic machine translation metrics often use human translations to determine the quality system translations. Common wisdom in the field dictates that the human references should be of very high quality. However, there are no cost-benefit analyses that could be used to guide practitioners who plan to collect references for machine translation evaluation. We find that higher-quality references lead to better metric correlations with humans at the segment-level. Having up to 7 references per segment and taking their average helps all metrics. Interestingly, the references from vendors of different qualities can be mixed together and improve metric success. Higher quality references, however, cost more to create and we frame this as an optimization problem: given a specific budget, what references should be collected to maximize metric success. These findings can be used by evaluators of shared tasks when references need to be created under a certain budget.

{{</citation>}}


### (36/74) CharacterEval: A Chinese Benchmark for Role-Playing Conversational Agent Evaluation (Quan Tu et al., 2024)

{{<citation>}}

Quan Tu, Shilong Fan, Zihang Tian, Rui Yan. (2024)  
**CharacterEval: A Chinese Benchmark for Role-Playing Conversational Agent Evaluation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2401.01275v1)  

---


**ABSTRACT**  
Recently, the advent of large language models (LLMs) has revolutionized generative agents. Among them, Role-Playing Conversational Agents (RPCAs) attract considerable attention due to their ability to emotionally engage users. However, the absence of a comprehensive benchmark impedes progress in this field. To bridge this gap, we introduce CharacterEval, a Chinese benchmark for comprehensive RPCA assessment, complemented by a tailored high-quality dataset. The dataset comprises 1,785 multi-turn role-playing dialogues, encompassing 23,020 examples and featuring 77 characters derived from Chinese novels and scripts. It was carefully constructed, beginning with initial dialogue extraction via GPT-4, followed by rigorous human-led quality control, and enhanced with in-depth character profiles sourced from Baidu Baike. CharacterEval employs a multifaceted evaluation approach, encompassing thirteen targeted metrics on four dimensions. Comprehensive experiments on CharacterEval demonstrate that Chinese LLMs exhibit more promising capabilities than GPT-4 in Chinese role-playing conversation. Source code, data source and reward model will be publicly accessible at https://github.com/morecry/CharacterEval.

{{</citation>}}


### (37/74) Fairness Certification for Natural Language Processing and Large Language Models (Vincent Freiberger et al., 2024)

{{<citation>}}

Vincent Freiberger, Erik Buchmann. (2024)  
**Fairness Certification for Natural Language Processing and Large Language Models**  

---
Primary Category: cs.CL  
Categories: 68T50, I-2-7, cs-AI, cs-CL, cs-CY, cs-LG, cs.CL  
Keywords: Language Model, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2401.01262v2)  

---


**ABSTRACT**  
Natural Language Processing (NLP) plays an important role in our daily lives, particularly due to the enormous progress of Large Language Models (LLM). However, NLP has many fairness-critical use cases, e.g., as an expert system in recruitment or as an LLM-based tutor in education. Since NLP is based on human language, potentially harmful biases can diffuse into NLP systems and produce unfair results, discriminate against minorities or generate legal issues. Hence, it is important to develop a fairness certification for NLP approaches. We follow a qualitative research approach towards a fairness certification for NLP. In particular, we have reviewed a large body of literature on algorithmic fairness, and we have conducted semi-structured expert interviews with a wide range of experts from that area. We have systematically devised six fairness criteria for NLP, which can be further refined into 18 sub-categories. Our criteria offer a foundation for operationalizing and testing processes to certify fairness, both from the perspective of the auditor and the audited organization.

{{</citation>}}


### (38/74) Zero-Shot Position Debiasing for Large Language Models (Zhongkun Liu et al., 2024)

{{<citation>}}

Zhongkun Liu, Zheng Chen, Mengqi Zhang, Zhaochun Ren, Zhumin Chen, Pengjie Ren. (2024)  
**Zero-Shot Position Debiasing for Large Language Models**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2401.01218v1)  

---


**ABSTRACT**  
Fine-tuning has been demonstrated to be an effective method to improve the domain performance of large language models (LLMs). However, LLMs might fit the dataset bias and shortcuts for prediction, leading to poor generation performance. Experimental result shows that LLMs are prone to exhibit position bias, i.e., leveraging information positioned at the beginning or end, or specific positional cues within the input. Existing works on mitigating position bias require external bias knowledge or annotated non-biased samples, which is unpractical in reality. In this work, we propose a zero-shot position debiasing (ZOE) framework to mitigate position bias for LLMs. ZOE leverages unsupervised responses from pre-trained LLMs for debiasing, thus without any external knowledge or datasets. To improve the quality of unsupervised responses, we propose a master-slave alignment (MSA) module to prune these responses. Experiments on eight datasets and five tasks show that ZOE consistently outperforms existing methods in mitigating four types of position biases. Besides, ZOE achieves this by sacrificing only a small performance on biased samples, which is simple and effective.

{{</citation>}}


### (39/74) Uncertainty Resolution in Misinformation Detection (Yury Orlovskiy et al., 2024)

{{<citation>}}

Yury Orlovskiy, Camille Thibault, Anne Imouza, Jean-François Godbout, Reihaneh Rabbany, Kellin Pelrine. (2024)  
**Uncertainty Resolution in Misinformation Detection**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2401.01197v1)  

---


**ABSTRACT**  
Misinformation poses a variety of risks, such as undermining public trust and distorting factual discourse. Large Language Models (LLMs) like GPT-4 have been shown effective in mitigating misinformation, particularly in handling statements where enough context is provided. However, they struggle to assess ambiguous or context-deficient statements accurately. This work introduces a new method to resolve uncertainty in such statements. We propose a framework to categorize missing information and publish category labels for the LIAR-New dataset, which is adaptable to cross-domain content with missing information. We then leverage this framework to generate effective user queries for missing context. Compared to baselines, our method improves the rate at which generated questions are answerable by the user by 38 percentage points and classification performance by over 10 percentage points macro F1. Thus, this approach may provide a valuable component for future misinformation mitigation pipelines.

{{</citation>}}


### (40/74) Unifying Structured Data as Graph for Data-to-Text Pre-Training (Shujie Li et al., 2024)

{{<citation>}}

Shujie Li, Liang Li, Ruiying Geng, Min Yang, Binhua Li, Guanghu Yuan, Wanwei He, Shao Yuan, Can Ma, Fei Huang, Yongbin Li. (2024)  
**Unifying Structured Data as Graph for Data-to-Text Pre-Training**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, Transformer  
[Paper Link](http://arxiv.org/abs/2401.01183v1)  

---


**ABSTRACT**  
Data-to-text (D2T) generation aims to transform structured data into natural language text. Data-to-text pre-training has proved to be powerful in enhancing D2T generation and yields impressive performances. However, previous pre-training methods either oversimplified structured data into a sequence without considering input structures or designed training objectives tailored for a specific data structure (e.g., table or knowledge graph). In this paper, we unify different types of structured data (i.e., table, key-value data, knowledge graph) into the graph format and cast different data-to-text generation tasks as graph-to-text generation. To effectively exploit the structural information of the input graph, we propose a structure-enhanced pre-training method for D2T generation by designing a structure-enhanced Transformer. Concretely, we devise a position matrix for the Transformer, encoding relative positional information of connected nodes in the input graph. In addition, we propose a new attention matrix to incorporate graph structures into the original Transformer by taking the available explicit connectivity structure into account. Extensive experiments on six benchmark datasets show the effectiveness of our model. Our source codes are available at https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/unid2t.

{{</citation>}}


### (41/74) Quokka: An Open-source Large Language Model ChatBot for Material Science (Xianjun Yang et al., 2024)

{{<citation>}}

Xianjun Yang, Stephen D. Wilson, Linda Petzold. (2024)  
**Quokka: An Open-source Large Language Model ChatBot for Material Science**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CE, cs-CL, cs.CL  
Keywords: ChatBot, Language Model  
[Paper Link](http://arxiv.org/abs/2401.01089v1)  

---


**ABSTRACT**  
This paper presents the development of a specialized chatbot for materials science, leveraging the Llama-2 language model, and continuing pre-training on the expansive research articles in the materials science domain from the S2ORC dataset. The methodology involves an initial pretraining phase on over one million domain-specific papers, followed by an instruction-tuning process to refine the chatbot's capabilities. The chatbot is designed to assist researchers, educators, and students by providing instant, context-aware responses to queries in the field of materials science. We make the four trained checkpoints (7B, 13B, with or without chat ability) freely available to the research community at https://github.com/Xianjun-Yang/Quokka.

{{</citation>}}


### (42/74) Vietnamese Poem Generation & The Prospect Of Cross-Language Poem-To-Poem Translation (Triet Minh Huynh et al., 2024)

{{<citation>}}

Triet Minh Huynh, Quan Le Bao. (2024)  
**Vietnamese Poem Generation & The Prospect Of Cross-Language Poem-To-Poem Translation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, Language Model, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2401.01078v3)  

---


**ABSTRACT**  
Poetry generation has been a challenging task in the field of Natural Language Processing, as it requires the model to understand the nuances of language, sentiment, and style. In this paper, we propose using Large Language Models to generate Vietnamese poems of various genres from natural language prompts, thereby facilitating an intuitive process with enhanced content control. Our most efficacious model, the GPT-3 Babbage variant, achieves a custom evaluation score of 0.8, specifically tailored to the "luc bat" genre of Vietnamese poetry. Furthermore, we also explore the idea of paraphrasing poems into normal text prompts and yield a relatively high score of 0.781 in the "luc bat" genre. This experiment presents the potential for cross-Language poem-to-poem translation with translated poems as the inputs while concurrently maintaining complete control over the generated content.

{{</citation>}}


### (43/74) DialCLIP: Empowering CLIP as Multi-Modal Dialog Retriever (Zhichao Yin et al., 2024)

{{<citation>}}

Zhichao Yin, Binyuan Hui, Min Yang, Fei Huang, Yongbin Li. (2024)  
**DialCLIP: Empowering CLIP as Multi-Modal Dialog Retriever**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog  
[Paper Link](http://arxiv.org/abs/2401.01076v2)  

---


**ABSTRACT**  
Recently, substantial advancements in pre-trained vision-language models have greatly enhanced the capabilities of multi-modal dialog systems. These models have demonstrated significant improvements by fine-tuning on downstream tasks. However, the existing pre-trained models primarily focus on effectively capturing the alignment between vision and language modalities, often ignoring the intricate nature of dialog context. In this paper, we propose a parameter-efficient prompt-tuning method named DialCLIP for multi-modal dialog retrieval. Specifically, our approach introduces a multi-modal context prompt generator to learn context features which are subsequently distilled into prompts within the pre-trained vision-language model CLIP. Besides, we introduce domain prompt to mitigate the disc repancy from the downstream dialog data. To facilitate various types of retrieval, we also design multiple experts to learn mappings from CLIP outputs to multi-modal representation space, with each expert being responsible to one specific retrieval type. Extensive experiments show that DialCLIP achieves state-of-the-art performance on two widely recognized benchmark datasets (i.e., PhotoChat and MMDialog) by tuning a mere 0.04% of the total parameters. These results highlight the efficacy and efficiency of our proposed approach, underscoring its potential to advance the field of multi-modal dialog retrieval.

{{</citation>}}


### (44/74) Discovering Significant Topics from Legal Decisions with Selective Inference (Jerrold Soh, 2024)

{{<citation>}}

Jerrold Soh. (2024)  
**Discovering Significant Topics from Legal Decisions with Selective Inference**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Legal  
[Paper Link](http://arxiv.org/abs/2401.01068v1)  

---


**ABSTRACT**  
We propose and evaluate an automated pipeline for discovering significant topics from legal decision texts by passing features synthesized with topic models through penalised regressions and post-selection significance tests. The method identifies case topics significantly correlated with outcomes, topic-word distributions which can be manually-interpreted to gain insights about significant topics, and case-topic weights which can be used to identify representative cases for each topic. We demonstrate the method on a new dataset of domain name disputes and a canonical dataset of European Court of Human Rights violation cases. Topic models based on latent semantic analysis as well as language model embeddings are evaluated. We show that topics derived by the pipeline are consistent with legal doctrines in both areas and can be useful in other related legal analysis tasks.

{{</citation>}}


### (45/74) LLaMA Beyond English: An Empirical Study on Language Capability Transfer (Jun Zhao et al., 2024)

{{<citation>}}

Jun Zhao, Zhihao Zhang, Qi Zhang, Tao Gui, Xuanjing Huang. (2024)  
**LLaMA Beyond English: An Empirical Study on Language Capability Transfer**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT, LLaMA  
[Paper Link](http://arxiv.org/abs/2401.01055v1)  

---


**ABSTRACT**  
In recent times, substantial advancements have been witnessed in large language models (LLMs), exemplified by ChatGPT, showcasing remarkable proficiency across a range of complex tasks. However, many mainstream LLMs (e.g. LLaMA) are pretrained on English-dominant corpus, which limits their performance in other non-English languages. In this paper, we focus on how to effectively transfer the capabilities of language generation and following instructions to a non-English language. To answer this question, we conduct an extensive empirical investigation based on LLaMA, accumulating over 1440 GPU hours. We analyze the impact of key factors such as vocabulary extension, further pretraining, and instruction tuning on transfer. To accurately assess the model's level of knowledge, we employ four widely used standardized testing benchmarks: C-Eval, MMLU, AGI-Eval, and GAOKAO-Bench. Furthermore, a comprehensive evaluation of the model's response quality is conducted, considering aspects such as accuracy, fluency, informativeness, logical coherence, and harmlessness, based on LLM-Eval, a benchmarks consisting instruction tasks from 17 diverse categories. Our evaluation results demonstrate that comparable performance to state-of-the-art transfer models can be achieved with less than 1% of the pretraining data, both in terms of knowledge alignment and response quality. Furthermore, the experimental outcomes across the thirteen low-resource languages also exhibit similar trends. We anticipate that the conclusions revealed by the experiments will aid the community in developing non-English LLMs.

{{</citation>}}


### (46/74) Cheetah: Natural Language Generation for 517 African Languages (Ife Adebara et al., 2024)

{{<citation>}}

Ife Adebara, AbdelRahim Elmadany, Muhammad Abdul-Mageed. (2024)  
**Cheetah: Natural Language Generation for 517 African Languages**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP, Natural Language Generation  
[Paper Link](http://arxiv.org/abs/2401.01053v1)  

---


**ABSTRACT**  
Low-resource African languages pose unique challenges for natural language processing (NLP) tasks, including natural language generation (NLG). In this paper, we develop Cheetah, a massively multilingual NLG language model for African languages. Cheetah supports 517 African languages and language varieties, allowing us to address the scarcity of NLG resources and provide a solution to foster linguistic diversity. We demonstrate the effectiveness of Cheetah through comprehensive evaluations across seven generation downstream tasks. In five of the seven tasks, Cheetah significantly outperforms other models, showcasing its remarkable performance for generating coherent and contextually appropriate text in a wide range of African languages. We additionally conduct a detailed human evaluation to delve deeper into the linguistic capabilities of Cheetah. The introduction of Cheetah has far-reaching benefits for linguistic diversity. By leveraging pretrained models and adapting them to specific languages, our approach facilitates the development of practical NLG applications for African communities. The findings of this study contribute to advancing NLP research in low-resource settings, enabling greater accessibility and inclusion for African languages in a rapidly expanding digital landscape. We will publicly release our models for research.

{{</citation>}}


## cs.LG (9)



### (47/74) Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models (Zixiang Chen et al., 2024)

{{<citation>}}

Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, Quanquan Gu. (2024)  
**Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG, stat-ML  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2401.01335v1)  

---


**ABSTRACT**  
Harnessing the power of human-annotated data through Supervised Fine-Tuning (SFT) is pivotal for advancing Large Language Models (LLMs). In this paper, we delve into the prospect of growing a strong LLM out of a weak one without the need for acquiring additional human-annotated data. We propose a new fine-tuning method called Self-Play fIne-tuNing (SPIN), which starts from a supervised fine-tuned model. At the heart of SPIN lies a self-play mechanism, where the LLM refines its capability by playing against instances of itself. More specifically, the LLM generates its own training data from its previous iterations, refining its policy by discerning these self-generated responses from those obtained from human-annotated data. Our method progressively elevates the LLM from a nascent model to a formidable one, unlocking the full potential of human-annotated demonstration data for SFT. Theoretically, we prove that the global optimum to the training objective function of our method is achieved only when the LLM policy aligns with the target data distribution. Empirically, we evaluate our method on several benchmark datasets including the HuggingFace Open LLM Leaderboard, MT-Bench, and datasets from Big-Bench. Our results show that SPIN can significantly improve the LLM's performance across a variety of benchmarks and even outperform models trained through direct preference optimization (DPO) supplemented with extra GPT-4 preference data. This sheds light on the promise of self-play, enabling the achievement of human-level performance in LLMs without the need for expert opponents.

{{</citation>}}


### (48/74) Learning-based agricultural management in partially observable environments subject to climate variability (Zhaoan Wang et al., 2024)

{{<citation>}}

Zhaoan Wang, Shaoping Xiao, Junchao Li, Jun Wang. (2024)  
**Learning-based agricultural management in partially observable environments subject to climate variability**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.01273v1)  

---


**ABSTRACT**  
Agricultural management, with a particular focus on fertilization strategies, holds a central role in shaping crop yield, economic profitability, and environmental sustainability. While conventional guidelines offer valuable insights, their efficacy diminishes when confronted with extreme weather conditions, such as heatwaves and droughts. In this study, we introduce an innovative framework that integrates Deep Reinforcement Learning (DRL) with Recurrent Neural Networks (RNNs). Leveraging the Gym-DSSAT simulator, we train an intelligent agent to master optimal nitrogen fertilization management. Through a series of simulation experiments conducted on corn crops in Iowa, we compare Partially Observable Markov Decision Process (POMDP) models with Markov Decision Process (MDP) models. Our research underscores the advantages of utilizing sequential observations in developing more efficient nitrogen input policies. Additionally, we explore the impact of climate variability, particularly during extreme weather events, on agricultural outcomes and management. Our findings demonstrate the adaptability of fertilization policies to varying climate conditions. Notably, a fixed policy exhibits resilience in the face of minor climate fluctuations, leading to commendable corn yields, cost-effectiveness, and environmental conservation. However, our study illuminates the need for agent retraining to acquire new optimal policies under extreme weather events. This research charts a promising course toward adaptable fertilization strategies that can seamlessly align with dynamic climate scenarios, ultimately contributing to the optimization of crop management practices.

{{</citation>}}


### (49/74) Encoding Binary Events from Continuous Time Series in Rooted Trees using Contrastive Learning (Tobias Engelhardt Rasmussen et al., 2024)

{{<citation>}}

Tobias Engelhardt Rasmussen, Siv Sørensen. (2024)  
**Encoding Binary Events from Continuous Time Series in Rooted Trees using Contrastive Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-SI, cs.LG, stat-ML  
Keywords: Contrastive Learning, Time Series  
[Paper Link](http://arxiv.org/abs/2401.01242v1)  

---


**ABSTRACT**  
Broadband infrastructure owners do not always know how their customers are connected in the local networks, which are structured as rooted trees. A recent study is able to infer the topology of a local network using discrete time series data from the leaves of the tree (customers). In this study we propose a contrastive approach for learning a binary event encoder from continuous time series data. As a preliminary result, we show that our approach has some potential in learning a valuable encoder.

{{</citation>}}


### (50/74) Graph Elimination Networks (Shuo Wang et al., 2024)

{{<citation>}}

Shuo Wang, Ge Cheng, Yun Zhang. (2024)  
**Graph Elimination Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2401.01233v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) are widely applied across various domains, yet they perform poorly in deep layers. Existing research typically attributes this problem to node over-smoothing, where node representations become indistinguishable after multiple rounds of propagation. In this paper, we delve into the neighborhood propagation mechanism of GNNs and discover that the real root cause of GNNs' performance degradation in deep layers lies in ineffective neighborhood feature propagation. This propagation leads to an exponential growth of a node's current representation at every propagation step, making it extremely challenging to capture valuable dependencies between long-distance nodes. To address this issue, we introduce Graph Elimination Networks (GENs), which employ a specific algorithm to eliminate redundancies during neighborhood propagation. We demonstrate that GENs can enhance nodes' perception of distant neighborhoods and extend the depth of network propagation. Extensive experiments show that GENs outperform the state-of-the-art methods on various graph-level and node-level datasets.

{{</citation>}}


### (51/74) Motif-aware Riemannian Graph Neural Network with Generative-Contrastive Learning (Li Sun et al., 2024)

{{<citation>}}

Li Sun, Zhenhao Huang, Zixi Wang, Feiyang Wang, Hao Peng, Philip Yu. (2024)  
**Motif-aware Riemannian Graph Neural Network with Generative-Contrastive Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Contrastive Learning, Graph Neural Network, Representation Learning  
[Paper Link](http://arxiv.org/abs/2401.01232v1)  

---


**ABSTRACT**  
Graphs are typical non-Euclidean data of complex structures. In recent years, Riemannian graph representation learning has emerged as an exciting alternative to Euclidean ones. However, Riemannian methods are still in an early stage: most of them present a single curvature (radius) regardless of structural complexity, suffer from numerical instability due to the exponential/logarithmic map, and lack the ability to capture motif regularity. In light of the issues above, we propose the problem of \emph{Motif-aware Riemannian Graph Representation Learning}, seeking a numerically stable encoder to capture motif regularity in a diverse-curvature manifold without labels. To this end, we present a novel Motif-aware Riemannian model with Generative-Contrastive learning (MotifRGC), which conducts a minmax game in Riemannian manifold in a self-supervised manner. First, we propose a new type of Riemannian GCN (D-GCN), in which we construct a diverse-curvature manifold by a product layer with the diversified factor, and replace the exponential/logarithmic map by a stable kernel layer. Second, we introduce a motif-aware Riemannian generative-contrastive learning to capture motif regularity in the constructed manifold and learn motif-aware node representation without external labels. Empirical results show the superiority of MofitRGC.

{{</citation>}}


### (52/74) Deep-ELA: Deep Exploratory Landscape Analysis with Self-Supervised Pretrained Transformers for Single- and Multi-Objective Continuous Optimization Problems (Moritz Vinzent Seiler et al., 2024)

{{<citation>}}

Moritz Vinzent Seiler, Pascal Kerschke, Heike Trautmann. (2024)  
**Deep-ELA: Deep Exploratory Landscape Analysis with Self-Supervised Pretrained Transformers for Single- and Multi-Objective Continuous Optimization Problems**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Self-Supervised, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.01192v1)  

---


**ABSTRACT**  
In many recent works, the potential of Exploratory Landscape Analysis (ELA) features to numerically characterize, in particular, single-objective continuous optimization problems has been demonstrated. These numerical features provide the input for all kinds of machine learning tasks on continuous optimization problems, ranging, i.a., from High-level Property Prediction to Automated Algorithm Selection and Automated Algorithm Configuration. Without ELA features, analyzing and understanding the characteristics of single-objective continuous optimization problems would be impossible.   Yet, despite their undisputed usefulness, ELA features suffer from several drawbacks. These include, in particular, (1.) a strong correlation between multiple features, as well as (2.) its very limited applicability to multi-objective continuous optimization problems. As a remedy, recent works proposed deep learning-based approaches as alternatives to ELA. In these works, e.g., point-cloud transformers were used to characterize an optimization problem's fitness landscape. However, these approaches require a large amount of labeled training data.   Within this work, we propose a hybrid approach, Deep-ELA, which combines (the benefits of) deep learning and ELA features. Specifically, we pre-trained four transformers on millions of randomly generated optimization problems to learn deep representations of the landscapes of continuous single- and multi-objective optimization problems. Our proposed framework can either be used out-of-the-box for analyzing single- and multi-objective continuous optimization problems, or subsequently fine-tuned to various tasks focussing on algorithm behavior and problem understanding.

{{</citation>}}


### (53/74) Reinforcement Learning for SAR View Angle Inversion with Differentiable SAR Renderer (Yanni Wang et al., 2024)

{{<citation>}}

Yanni Wang, Hecheng Jia, Shilei Fu, Huiping Lin, Feng Xu. (2024)  
**Reinforcement Learning for SAR View Angle Inversion with Differentiable SAR Renderer**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, eess-SP  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.01165v1)  

---


**ABSTRACT**  
The electromagnetic inverse problem has long been a research hotspot. This study aims to reverse radar view angles in synthetic aperture radar (SAR) images given a target model. Nonetheless, the scarcity of SAR data, combined with the intricate background interference and imaging mechanisms, limit the applications of existing learning-based approaches. To address these challenges, we propose an interactive deep reinforcement learning (DRL) framework, where an electromagnetic simulator named differentiable SAR render (DSR) is embedded to facilitate the interaction between the agent and the environment, simulating a human-like process of angle prediction. Specifically, DSR generates SAR images at arbitrary view angles in real-time. And the differences in sequential and semantic aspects between the view angle-corresponding images are leveraged to construct the state space in DRL, which effectively suppress the complex background interference, enhance the sensitivity to temporal variations, and improve the capability to capture fine-grained information. Additionally, in order to maintain the stability and convergence of our method, a series of reward mechanisms, such as memory difference, smoothing and boundary penalty, are utilized to form the final reward function. Extensive experiments performed on both simulated and real datasets demonstrate the effectiveness and robustness of our proposed method. When utilized in the cross-domain area, the proposed method greatly mitigates inconsistency between simulated and real domains, outperforming reference methods significantly.

{{</citation>}}


### (54/74) Explainable Adaptive Tree-based Model Selection for Time Series Forecasting (Matthias Jakobs et al., 2024)

{{<citation>}}

Matthias Jakobs, Amal Saadallah. (2024)  
**Explainable Adaptive Tree-based Model Selection for Time Series Forecasting**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2401.01124v1)  

---


**ABSTRACT**  
Tree-based models have been successfully applied to a wide variety of tasks, including time series forecasting. They are increasingly in demand and widely accepted because of their comparatively high level of interpretability. However, many of them suffer from the overfitting problem, which limits their application in real-world decision-making. This problem becomes even more severe in online-forecasting settings where time series observations are incrementally acquired, and the distributions from which they are drawn may keep changing over time. In this context, we propose a novel method for the online selection of tree-based models using the TreeSHAP explainability method in the task of time series forecasting. We start with an arbitrary set of different tree-based models. Then, we outline a performance-based ranking with a coherent design to make TreeSHAP able to specialize the tree-based forecasters across different regions in the input time series. In this framework, adequate model selection is performed online, adaptively following drift detection in the time series. In addition, explainability is supported on three levels, namely online input importance, model selection, and model output explanation. An extensive empirical study on various real-world datasets demonstrates that our method achieves excellent or on-par results in comparison to the state-of-the-art approaches as well as several baselines.

{{</citation>}}


### (55/74) Boosting Transformer's Robustness and Efficacy in PPG Signal Artifact Detection with Self-Supervised Learning (Thanh-Dung Le, 2024)

{{<citation>}}

Thanh-Dung Le. (2024)  
**Boosting Transformer's Robustness and Efficacy in PPG Signal Artifact Detection with Self-Supervised Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, eess-SP  
Keywords: Self-Supervised, Transformer  
[Paper Link](http://arxiv.org/abs/2401.01013v1)  

---


**ABSTRACT**  
Recent research at CHU Sainte Justine's Pediatric Critical Care Unit (PICU) has revealed that traditional machine learning methods, such as semi-supervised label propagation and K-nearest neighbors, outperform Transformer-based models in artifact detection from PPG signals, mainly when data is limited. This study addresses the underutilization of abundant unlabeled data by employing self-supervised learning (SSL) to extract latent features from these data, followed by fine-tuning on labeled data. Our experiments demonstrate that SSL significantly enhances the Transformer model's ability to learn representations, improving its robustness in artifact classification tasks. Among various SSL techniques, including masking, contrastive learning, and DINO (self-distillation with no labels)-contrastive learning exhibited the most stable and superior performance in small PPG datasets. Further, we delve into optimizing contrastive loss functions, which are crucial for contrastive SSL. Inspired by InfoNCE, we introduce a novel contrastive loss function that facilitates smoother training and better convergence, thereby enhancing performance in artifact classification. In summary, this study establishes the efficacy of SSL in leveraging unlabeled data, particularly in enhancing the capabilities of the Transformer model. This approach holds promise for broader applications in PICU environments, where annotated data is often limited.

{{</citation>}}


## eess.SP (2)



### (56/74) Self-Supervised Millimeter Wave Indoor Localization using Tiny Neural Networks (Anish Shastri et al., 2024)

{{<citation>}}

Anish Shastri, Steve Blandino, Camillo Gentile, Chiehping Lai, Paolo Casari. (2024)  
**Self-Supervised Millimeter Wave Indoor Localization using Tiny Neural Networks**  

---
Primary Category: eess.SP  
Categories: cs-NI, eess-SP, eess.SP  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2401.01329v1)  

---


**ABSTRACT**  
The quasi-optical propagation of millimeter-wave signals enables high-accuracy localization algorithms that employ geometric approaches or machine learning models. However, most algorithms require information on the indoor environment, may entail the collection of large training datasets, or bear an infeasible computational burden for commercial off-the-shelf (COTS) devices. In this work, we propose to use tiny neural networks (NNs) to learn the relationship between angle difference-of-arrival (ADoA) measurements and locations of a receiver in an indoor environment. To relieve training data collection efforts, we resort to a self-supervised approach by bootstrapping the training of our neural network through location estimates obtained from a state-of-the-art localization algorithm. We evaluate our scheme via mmWave measurements from indoor 60-GHz double-directional channel sounding. We process the measurements to yield dominant multipath components, use the corresponding angles to compute ADoA values, and finally obtain location fixes. Results show that the tiny NN achieves sub-meter errors in 74\% of the cases, thus performing as good as or even better than the state-of-the-art algorithm, with significantly lower computational complexity.

{{</citation>}}


### (57/74) Enhancing Automatic Modulation Recognition through Robust Global Feature Extraction (Yunpeng Qu et al., 2024)

{{<citation>}}

Yunpeng Qu, Zhilin Lu, Rui Zeng, Jintao Wang, Jian Wang. (2024)  
**Enhancing Automatic Modulation Recognition through Robust Global Feature Extraction**  

---
Primary Category: eess.SP  
Categories: cs-AI, cs-LG, eess-SP, eess.SP  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2401.01056v1)  

---


**ABSTRACT**  
Automatic Modulation Recognition (AMR) plays a crucial role in wireless communication systems. Deep learning AMR strategies have achieved tremendous success in recent years. Modulated signals exhibit long temporal dependencies, and extracting global features is crucial in identifying modulation schemes. Traditionally, human experts analyze patterns in constellation diagrams to classify modulation schemes. Classical convolutional-based networks, due to their limited receptive fields, excel at extracting local features but struggle to capture global relationships. To address this limitation, we introduce a novel hybrid deep framework named TLDNN, which incorporates the architectures of the transformer and long short-term memory (LSTM). We utilize the self-attention mechanism of the transformer to model the global correlations in signal sequences while employing LSTM to enhance the capture of temporal dependencies. To mitigate the impact like RF fingerprint features and channel characteristics on model generalization, we propose data augmentation strategies known as segment substitution (SS) to enhance the model's robustness to modulation-related features. Experimental results on widely-used datasets demonstrate that our method achieves state-of-the-art performance and exhibits significant advantages in terms of complexity. Our proposed framework serves as a foundational backbone that can be extended to different datasets. We have verified the effectiveness of our augmentation approach in enhancing the generalization of the models, particularly in few-shot scenarios. Code is available at \url{https://github.com/AMR-Master/TLDNN}.

{{</citation>}}


## cs.MA (1)



### (58/74) LLM Harmony: Multi-Agent Communication for Problem Solving (Sumedh Rasal, 2024)

{{<citation>}}

Sumedh Rasal. (2024)  
**LLM Harmony: Multi-Agent Communication for Problem Solving**  

---
Primary Category: cs.MA  
Categories: cs-MA, cs.MA  
Keywords: Language Model, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2401.01312v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have revolutionized Natural Language Processing but exhibit limitations, particularly in autonomously addressing novel challenges such as reasoning and problem-solving. Traditional techniques like chain-of-thought prompting necessitate explicit human guidance. This paper introduces a novel multi-agent communication framework, inspired by the CAMEL model, to enhance LLMs' autonomous problem-solving capabilities. The framework employs multiple LLM agents, each with a distinct persona, engaged in role-playing communication, offering a nuanced and adaptable approach to diverse problem scenarios. Extensive experimentation demonstrates the framework's superior performance and adaptability, providing valuable insights into the collaborative potential of multiple agents in overcoming the limitations of individual models.

{{</citation>}}


## cs.CR (7)



### (59/74) Experimental Validation of Sensor Fusion-based GNSS Spoofing Attack Detection Framework for Autonomous Vehicles (Sagar Dasgupta et al., 2024)

{{<citation>}}

Sagar Dasgupta, Kazi Hassan Shakib, Mizanur Rahman. (2024)  
**Experimental Validation of Sensor Fusion-based GNSS Spoofing Attack Detection Framework for Autonomous Vehicles**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2401.01304v1)  

---


**ABSTRACT**  
In this paper, we validate the performance of the a sensor fusion-based Global Navigation Satellite System (GNSS) spoofing attack detection framework for Autonomous Vehicles (AVs). To collect data, a vehicle equipped with a GNSS receiver, along with Inertial Measurement Unit (IMU) is used. The detection framework incorporates two strategies: The first strategy involves comparing the predicted location shift, which is the distance traveled between two consecutive timestamps, with the inertial sensor-based location shift. For this purpose, data from low-cost in-vehicle inertial sensors such as the accelerometer and gyroscope sensor are fused and fed into a long short-term memory (LSTM) neural network. The second strategy employs a Random-Forest supervised machine learning model to detect and classify turns, distinguishing between left and right turns using the output from the steering angle sensor. In experiments, two types of spoofing attack models: turn-by-turn and wrong turn are simulated. These spoofing attacks are modeled as SQL injection attacks, where, upon successful implementation, the navigation system perceives injected spoofed location information as legitimate while being unable to detect legitimate GNSS signals. Importantly, the IMU data remains uncompromised throughout the spoofing attack. To test the effectiveness of the detection framework, experiments are conducted in Tuscaloosa, AL, mimicking urban road structures. The results demonstrate the framework's ability to detect various sophisticated GNSS spoofing attacks, even including slow position drifting attacks. Overall, the experimental results showcase the robustness and efficacy of the sensor fusion-based spoofing attack detection approach in safeguarding AVs against GNSS spoofing threats.

{{</citation>}}


### (60/74) LLbezpeky: Leveraging Large Language Models for Vulnerability Detection (Noble Saji Mathews et al., 2024)

{{<citation>}}

Noble Saji Mathews, Yelizaveta Brus, Yousra Aafer, Mei Nagappan, Shane McIntosh. (2024)  
**LLbezpeky: Leveraging Large Language Models for Vulnerability Detection**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs-SE, cs.CR  
Keywords: AI, Language Model, Vulnerability Detection  
[Paper Link](http://arxiv.org/abs/2401.01269v1)  

---


**ABSTRACT**  
Despite the continued research and progress in building secure systems, Android applications continue to be ridden with vulnerabilities, necessitating effective detection methods. Current strategies involving static and dynamic analysis tools come with limitations like overwhelming number of false positives and limited scope of analysis which make either difficult to adopt. Over the past years, machine learning based approaches have been extensively explored for vulnerability detection, but its real-world applicability is constrained by data requirements and feature engineering challenges. Large Language Models (LLMs), with their vast parameters, have shown tremendous potential in understanding semnatics in human as well as programming languages. We dive into the efficacy of LLMs for detecting vulnerabilities in the context of Android security. We focus on building an AI-driven workflow to assist developers in identifying and rectifying vulnerabilities. Our experiments show that LLMs outperform our expectations in finding issues within applications correctly flagging insecure apps in 91.67% of cases in the Ghera benchmark. We use inferences from our experiments towards building a robust and actionable vulnerability detection system and demonstrate its effectiveness. Our experiments also shed light on how different various simple configurations can affect the True Positive (TP) and False Positive (FP) rates.

{{</citation>}}


### (61/74) PPBFL: A Privacy Protected Blockchain-based Federated Learning Model (Yang Li et al., 2024)

{{<citation>}}

Yang Li, Chunhe Xia, Wanshuang Lin, Tianbo Wang. (2024)  
**PPBFL: A Privacy Protected Blockchain-based Federated Learning Model**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2401.01204v1)  

---


**ABSTRACT**  
With the rapid development of machine learning and growing concerns about data privacy, federated learning has become an increasingly prominent focus. However, challenges such as attacks on model parameters and the lack of incentive mechanisms hinder the effectiveness of federated learning. Therefore, we propose a Privacy Protected Blockchain-based Federated Learning Model (PPBFL) to enhance the security of federated learning and promote the active participation of nodes in model training. Blockchain ensures that model parameters stored in the InterPlanetary File System (IPFS) remain unaltered. A novel adaptive differential privacy addition algorithm is simultaneously applied to local and global models, preserving the privacy of local models and preventing a decrease in the security of the global model due to the presence of numerous local models in federated learning. Additionally, we introduce a new mix transactions mechanism to better protect the identity privacy of local training clients. Security analysis and experimental results demonstrate that PPBFL outperforms baseline methods in both model performance and security.

{{</citation>}}


### (62/74) Imperio: Language-Guided Backdoor Attacks for Arbitrary Model Control (Ka-Ho Chow et al., 2024)

{{<citation>}}

Ka-Ho Chow, Wenqi Wei, Lei Yu. (2024)  
**Imperio: Language-Guided Backdoor Attacks for Arbitrary Model Control**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs.CR  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2401.01085v1)  

---


**ABSTRACT**  
Revolutionized by the transformer architecture, natural language processing (NLP) has received unprecedented attention. While advancements in NLP models have led to extensive research into their backdoor vulnerabilities, the potential for these advancements to introduce new backdoor threats remains unexplored. This paper proposes Imperio, which harnesses the language understanding capabilities of NLP models to enrich backdoor attacks. Imperio provides a new model control experience. It empowers the adversary to control the victim model with arbitrary output through language-guided instructions. This is achieved using a language model to fuel a conditional trigger generator, with optimizations designed to extend its language understanding capabilities to backdoor instruction interpretation and execution. Our experiments across three datasets, five attacks, and nine defenses confirm Imperio's effectiveness. It can produce contextually adaptive triggers from text descriptions and control the victim model with desired outputs, even in scenarios not encountered during training. The attack maintains a high success rate across complex datasets without compromising the accuracy of clean inputs and also exhibits resilience against representative defenses. The source code is available at \url{https://khchow.com/Imperio}.

{{</citation>}}


### (63/74) Detection and Defense Against Prominent Attacks on Preconditioned LLM-Integrated Virtual Assistants (Chun Fai Chan et al., 2024)

{{<citation>}}

Chun Fai Chan, Daniel Wankit Yip, Aysan Esmradi. (2024)  
**Detection and Defense Against Prominent Attacks on Preconditioned LLM-Integrated Virtual Assistants**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.00994v1)  

---


**ABSTRACT**  
The emergence of LLM (Large Language Model) integrated virtual assistants has brought about a rapid transformation in communication dynamics. During virtual assistant development, some developers prefer to leverage the system message, also known as an initial prompt or custom prompt, for preconditioning purposes. However, it is important to recognize that an excessive reliance on this functionality raises the risk of manipulation by malicious actors who can exploit it with carefully crafted prompts. Such malicious manipulation poses a significant threat, potentially compromising the accuracy and reliability of the virtual assistant's responses. Consequently, safeguarding the virtual assistants with detection and defense mechanisms becomes of paramount importance to ensure their safety and integrity. In this study, we explored three detection and defense mechanisms aimed at countering attacks that target the system message. These mechanisms include inserting a reference key, utilizing an LLM evaluator, and implementing a Self-Reminder. To showcase the efficacy of these mechanisms, they were tested against prominent attack techniques. Our findings demonstrate that the investigated mechanisms are capable of accurately identifying and counteracting the attacks. The effectiveness of these mechanisms underscores their potential in safeguarding the integrity and reliability of virtual assistants, reinforcing the importance of their implementation in real-world scenarios. By prioritizing the security of virtual assistants, organizations can maintain user trust, preserve the integrity of the application, and uphold the high standards expected in this era of transformative technologies.

{{</citation>}}


### (64/74) A Novel Evaluation Framework for Assessing Resilience Against Prompt Injection Attacks in Large Language Models (Daniel Wankit Yip et al., 2024)

{{<citation>}}

Daniel Wankit Yip, Aysan Esmradi, Chun Fai Chan. (2024)  
**A Novel Evaluation Framework for Assessing Resilience Against Prompt Injection Attacks in Large Language Models**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: GLM, Language Model  
[Paper Link](http://arxiv.org/abs/2401.00991v1)  

---


**ABSTRACT**  
Prompt injection attacks exploit vulnerabilities in large language models (LLMs) to manipulate the model into unintended actions or generate malicious content. As LLM integrated applications gain wider adoption, they face growing susceptibility to such attacks. This study introduces a novel evaluation framework for quantifying the resilience of applications. The framework incorporates innovative techniques designed to ensure representativeness, interpretability, and robustness. To ensure the representativeness of simulated attacks on the application, a meticulous selection process was employed, resulting in 115 carefully chosen attacks based on coverage and relevance. For enhanced interpretability, a second LLM was utilized to evaluate the responses generated from these simulated attacks. Unlike conventional malicious content classifiers that provide only a confidence score, the LLM-based evaluation produces a score accompanied by an explanation, thereby enhancing interpretability. Subsequently, a resilience score is computed by assigning higher weights to attacks with greater impact, thus providing a robust measurement of the application resilience. To assess the framework's efficacy, it was applied on two LLMs, namely Llama2 and ChatGLM. Results revealed that Llama2, the newer model exhibited higher resilience compared to ChatGLM. This finding substantiates the effectiveness of the framework, aligning with the prevailing notion that newer models tend to possess greater resilience. Moreover, the framework exhibited exceptional versatility, requiring only minimal adjustments to accommodate emerging attack techniques and classifications, thereby establishing itself as an effective and practical solution. Overall, the framework offers valuable insights that empower organizations to make well-informed decisions to fortify their applications against potential threats from prompt injection.

{{</citation>}}


### (65/74) CCA-Secure Hybrid Encryption in Correlated Randomness Model and KEM Combiners (Somnath Panja et al., 2024)

{{<citation>}}

Somnath Panja, Setareh Sharifian, Shaoquan Jiang, Reihaneh Safavi-Naini. (2024)  
**CCA-Secure Hybrid Encryption in Correlated Randomness Model and KEM Combiners**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2401.00983v1)  

---


**ABSTRACT**  
A hybrid encryption (HE) system is an efficient public key encryption system for arbitrarily long messages. An HE system consists of a public key component called key encapsulation mechanism (KEM), and a symmetric key component called data encapsulation mechanism (DEM). The HE encryption algorithm uses a KEM generated key k to encapsulate the message using DEM, and send the ciphertext together with the encapsulaton of k, to the decryptor who decapsulates k and uses it to decapsulate the message using the corresponding KEM and DEM components. The KEM/DEM composition theorem proves that if KEM and DEM satisfy well-defined security notions, then HE will be secure with well defined security. We introduce HE in correlated randomness model where the encryption and decryption algorithms have samples of correlated random variables that are partially leaked to the adversary. Security of the new KEM/DEM paradigm is defined against computationally unbounded or polynomially bounded adversaries. We define iKEM and cKEM with respective information theoretic computational security, and prove a composition theorem for them and a computationally secure DEM, resulting in secure HEs with proved computational security (CPA and CCA) and without any computational assumption. We construct two iKEMs that provably satisfy the required security notions of the composition theorem. The iKEMs are used to construct two efficient quantum-resistant HEs when used with an AES based DEM. We also define and construct combiners with proved security that combine the new KEM/DEM paradigm of HE with the traditional public key based paradigm of HE.

{{</citation>}}


## cs.CY (1)



### (66/74) Generative AI is already widespread in the public sector (Jonathan Bright et al., 2024)

{{<citation>}}

Jonathan Bright, Florence E. Enock, Saba Esnaashari, John Francis, Youmna Hashem, Deborah Morgan. (2024)  
**Generative AI is already widespread in the public sector**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2401.01291v1)  

---


**ABSTRACT**  
Generative AI has the potential to transform how public services are delivered by enhancing productivity and reducing time spent on bureaucracy. Furthermore, unlike other types of artificial intelligence, it is a technology that has quickly become widely available for bottom-up adoption: essentially anyone can decide to make use of it in their day to day work. But to what extent is generative AI already in use in the public sector? Our survey of 938 public service professionals within the UK (covering education, health, social work and emergency services) seeks to answer this question. We find that use of generative AI systems is already widespread: 45% of respondents were aware of generative AI usage within their area of work, while 22% actively use a generative AI system. Public sector professionals were positive about both current use of the technology and its potential to enhance their efficiency and reduce bureaucratic workload in the future. For example, those working in the NHS thought that time spent on bureaucracy could drop from 50% to 30% if generative AI was properly exploited, an equivalent of one day per week (an enormous potential impact). Our survey also found a high amount of trust (61%) around generative AI outputs, and a low fear of replacement (16%). While respondents were optimistic overall, areas of concern included feeling like the UK is missing out on opportunities to use AI to improve public services (76%), and only a minority of respondents (32%) felt like there was clear guidance on generative AI usage in their workplaces. In other words, it is clear that generative AI is already transforming the public sector, but uptake is happening in a disorganised fashion without clear guidelines. The UK's public sector urgently needs to develop more systematic methods for taking advantage of the technology.

{{</citation>}}


## cs.SI (1)



### (67/74) Deplatforming Norm-Violating Influencers on Social Media Reduces Overall Online Attention Toward Them (Manoel Horta Ribeiro et al., 2024)

{{<citation>}}

Manoel Horta Ribeiro, Shagun Jhaver, Jordi Cluet i Martinell, Marie Reignier-Tayar, Robert West. (2024)  
**Deplatforming Norm-Violating Influencers on Social Media Reduces Overall Online Attention Toward Them**  

---
Primary Category: cs.SI  
Categories: cs-CY, cs-SI, cs.SI  
Keywords: Attention, Google, Social Media  
[Paper Link](http://arxiv.org/abs/2401.01253v1)  

---


**ABSTRACT**  
From politicians to podcast hosts, online platforms have systematically banned (``deplatformed'') influential users for breaking platform guidelines. Previous inquiries on the effectiveness of this intervention are inconclusive because 1) they consider only few deplatforming events; 2) they consider only overt engagement traces (e.g., likes and posts) but not passive engagement (e.g., views); 3) they do not consider all the potential places users impacted by the deplatforming event might migrate to. We address these limitations in a longitudinal, quasi-experimental study of 165 deplatforming events targeted at 101 influencers. We collect deplatforming events from Reddit posts and then manually curate the data, ensuring the correctness of a large dataset of deplatforming events. Then, we link these events to Google Trends and Wikipedia page views, platform-agnostic measures of online attention that capture the general public's interest in specific influencers. Through a difference-in-differences approach, we find that deplatforming reduces online attention toward influencers. After 12 months, we estimate that online attention toward deplatformed influencers is reduced by -63% (95% CI [-75%,-46%]) on Google and by -43% (95% CI [-57%,-24%]) on Wikipedia. Further, as we study over a hundred deplatforming events, we can analyze in which cases deplatforming is more or less impactful, revealing nuances about the intervention. Notably, we find that both permanent and temporary deplatforming reduce online attention toward influencers; Overall, this work contributes to the ongoing effort to map the effectiveness of content moderation interventions, driving platform governance away from speculation.

{{</citation>}}


## cs.HC (1)



### (68/74) Privacy Preserving Personal Assistant with On-Device Diarization and Spoken Dialogue System for Home and Beyond (Gérard Chollet et al., 2024)

{{<citation>}}

Gérard Chollet, Hugues Sansen, Yannis Tevissen, Jérôme Boudy, Mossaab Hariz, Christophe Lohr, Fathy Yassa. (2024)  
**Privacy Preserving Personal Assistant with On-Device Diarization and Spoken Dialogue System for Home and Beyond**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2401.01146v1)  

---


**ABSTRACT**  
In the age of personal voice assistants, the question of privacy arises. These digital companions often lack memory of past interactions, while relying heavily on the internet for speech processing, raising privacy concerns. Modern smartphones now enable on-device speech processing, making cloud-based solutions unnecessary. Personal assistants for the elderly should excel at memory recall, especially in medical examinations. The e-ViTA project developed a versatile conversational application with local processing and speaker recognition. This paper highlights the importance of speaker diarization enriched with sensor data fusion for contextualized conversation preservation. The use cases applied to the e-VITA project have shown that truly personalized dialogue is pivotal for individual voice assistants. Secure local processing and sensor data fusion ensure virtual companions meet individual user needs without compromising privacy or data security.

{{</citation>}}


## eess.AS (2)



### (69/74) HAAQI-Net: A non-intrusive neural music quality assessment model for hearing aids (Dyah A. M. G. Wisnu et al., 2024)

{{<citation>}}

Dyah A. M. G. Wisnu, Epri Pratiwi, Stefano Rini, Ryandhimas E. Zezario, Hsin-Min Wang, Yu Tsao. (2024)  
**HAAQI-Net: A non-intrusive neural music quality assessment model for hearing aids**  

---
Primary Category: eess.AS  
Categories: cs-LG, cs-SD, eess-AS, eess.AS  
Keywords: LSTM, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.01145v1)  

---


**ABSTRACT**  
This paper introduces HAAQI-Net, a non-intrusive deep learning model for music quality assessment tailored to hearing aid users. In contrast to traditional methods like the Hearing Aid Audio Quality Index (HAAQI), HAAQI-Net utilizes a Bidirectional Long Short-Term Memory (BLSTM) with attention. It takes an assessed music sample and a hearing loss pattern as input, generating a predicted HAAQI score. The model employs the pre-trained Bidirectional Encoder representation from Audio Transformers (BEATs) for acoustic feature extraction. Comparing predicted scores with ground truth, HAAQI-Net achieves a Longitudinal Concordance Correlation (LCC) of 0.9257, Spearman's Rank Correlation Coefficient (SRCC) of 0.9394, and Mean Squared Error (MSE) of 0.0080. Notably, this high performance comes with a substantial reduction in inference time: from 62.52 seconds (by HAAQI) to 2.71 seconds (by HAAQI-Net), serving as an efficient music quality assessment model for hearing aid users.

{{</citation>}}


### (70/74) Efficient Parallel Audio Generation using Group Masked Language Modeling (Myeonghun Jeong et al., 2024)

{{<citation>}}

Myeonghun Jeong, Minchan Kim, Joun Yeop Lee, Nam Soo Kim. (2024)  
**Efficient Parallel Audio Generation using Group Masked Language Modeling**  

---
Primary Category: eess.AS  
Categories: cs-AI, cs-LG, eess-AS, eess.AS  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.01099v1)  

---


**ABSTRACT**  
We present a fast and high-quality codec language model for parallel audio generation. While SoundStorm, a state-of-the-art parallel audio generation model, accelerates inference speed compared to autoregressive models, it still suffers from slow inference due to iterative sampling. To resolve this problem, we propose Group-Masked Language Modeling~(G-MLM) and Group Iterative Parallel Decoding~(G-IPD) for efficient parallel audio generation. Both the training and sampling schemes enable the model to synthesize high-quality audio with a small number of iterations by effectively modeling the group-wise conditional dependencies. In addition, our model employs a cross-attention-based architecture to capture the speaker style of the prompt voice and improves computational efficiency. Experimental results demonstrate that our proposed model outperforms the baselines in prompt-based audio generation.

{{</citation>}}


## astro-ph.SR (1)



### (71/74) AI-FLARES: Artificial Intelligence for the Analysis of Solar Flares Data (Michele Piana et al., 2024)

{{<citation>}}

Michele Piana, Federico Benvenuto, Anna Maria Massone, Cristina Campi, Sabrina Guastavino, Francesco Marchetti, Paolo Massa, Emma Perracchione, Anna Volpara. (2024)  
**AI-FLARES: Artificial Intelligence for the Analysis of Solar Flares Data**  

---
Primary Category: astro-ph.SR  
Categories: 85-04, 68T01, astro-ph-SR, astro-ph.SR, cs-AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.01104v1)  

---


**ABSTRACT**  
AI-FLARES (Artificial Intelligence for the Analysis of Solar Flares Data) is a research project funded by the Agenzia Spaziale Italiana and by the Istituto Nazionale di Astrofisica within the framework of the ``Attivit\`a di Studio per la Comunit\`a Scientifica Nazionale Sole, Sistema Solare ed Esopianeti'' program. The topic addressed by this project was the development and use of computational methods for the analysis of remote sensing space data associated to solar flare emission. This paper overviews the main results obtained by the project, with specific focus on solar flare forecasting, reconstruction of morphologies of the flaring sources, and interpretation of acceleration mechanisms triggered by solar flares.

{{</citation>}}


## cs.RO (1)



### (72/74) PLE-SLAM: A Visual-Inertial SLAM Based on Point-Line Features and Efficient IMU Initialization (Jiaming He et al., 2024)

{{<citation>}}

Jiaming He, Mingrui Li, Yangyang Wang, Hongyu Wang. (2024)  
**PLE-SLAM: A Visual-Inertial SLAM Based on Point-Line Features and Efficient IMU Initialization**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2401.01081v2)  

---


**ABSTRACT**  
Visual-inertial SLAM is crucial in various fields, such as aerial vehicles, industrial robots, and autonomous driving. The fusion of camera and inertial measurement unit (IMU) makes up for the shortcomings of a signal sensor, which significantly improves the accuracy and robustness of localization in challenging environments. This article presents PLE-SLAM, an accurate and real-time visual-inertial SLAM algorithm based on point-line features and efficient IMU initialization. First, we use parallel computing methods to extract features and compute descriptors to ensure real-time performance. Adjacent short line segments are merged into long line segments, and isolated short line segments are directly deleted. Second, a rotation-translation-decoupled initialization method is extended to use both points and lines. Gyroscope bias is optimized by tightly coupling IMU measurements and image observations. Accelerometer bias and gravity direction are solved by an analytical method for efficiency. To improve the system's intelligence in handling complex environments, a scheme of leveraging semantic information and geometric constraints to eliminate dynamic features and A solution for loop detection and closed-loop frame pose estimation using CNN and GNN are integrated into the system. All networks are accelerated to ensure real-time performance. The experiment results on public datasets illustrate that PLE-SLAM is one of the state-of-the-art visual-inertial SLAM systems.

{{</citation>}}


## cs.SE (1)



### (73/74) Experimenting a New Programming Practice with LLMs (Simiao Zhang et al., 2024)

{{<citation>}}

Simiao Zhang, Jiaping Wang, Guoliang Dong, Jun Sun, Yueling Zhang, Geguang Pu. (2024)  
**Experimenting a New Programming Practice with LLMs**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.01062v1)  

---


**ABSTRACT**  
The recent development on large language models makes automatically constructing small programs possible. It thus has the potential to free software engineers from low-level coding and allow us to focus on the perhaps more interesting parts of software development, such as requirement engineering and system testing. In this project, we develop a prototype named AISD (AI-aided Software Development), which is capable of taking high-level (potentially vague) user requirements as inputs, generates detailed use cases, prototype system designs, and subsequently system implementation. Different from existing attempts, AISD is designed to keep the user in the loop, i.e., by repeatedly taking user feedback on use cases, high-level system designs, and prototype implementations through system testing. AISD has been evaluated with a novel benchmark of non-trivial software projects. The experimental results suggest that it might be possible to imagine a future where software engineering is reduced to requirement engineering and system testing only.

{{</citation>}}


## cs.SD (1)



### (74/74) Auffusion: Leveraging the Power of Diffusion and Large Language Models for Text-to-Audio Generation (Jinlong Xue et al., 2024)

{{<citation>}}

Jinlong Xue, Yayue Deng, Yingming Gao, Ya Li. (2024)  
**Auffusion: Leveraging the Power of Diffusion and Large Language Models for Text-to-Audio Generation**  

---
Primary Category: cs.SD  
Categories: cs-AI, cs-CL, cs-SD, cs.SD, eess-AS  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2401.01044v1)  

---


**ABSTRACT**  
Recent advancements in diffusion models and large language models (LLMs) have significantly propelled the field of AIGC. Text-to-Audio (TTA), a burgeoning AIGC application designed to generate audio from natural language prompts, is attracting increasing attention. However, existing TTA studies often struggle with generation quality and text-audio alignment, especially for complex textual inputs. Drawing inspiration from state-of-the-art Text-to-Image (T2I) diffusion models, we introduce Auffusion, a TTA system adapting T2I model frameworks to TTA task, by effectively leveraging their inherent generative strengths and precise cross-modal alignment. Our objective and subjective evaluations demonstrate that Auffusion surpasses previous TTA approaches using limited data and computational resource. Furthermore, previous studies in T2I recognizes the significant impact of encoder choice on cross-modal alignment, like fine-grained details and object bindings, while similar evaluation is lacking in prior TTA works. Through comprehensive ablation studies and innovative cross-attention map visualizations, we provide insightful assessments of text-audio alignment in TTA. Our findings reveal Auffusion's superior capability in generating audios that accurately match textual descriptions, which further demonstrated in several related tasks, such as audio style transfer, inpainting and other manipulations. Our implementation and demos are available at https://auffusion.github.io.

{{</citation>}}
