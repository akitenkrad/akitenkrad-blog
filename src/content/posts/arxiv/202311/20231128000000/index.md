---
draft: false
title: "arXiv @ 2023.11.28"
date: 2023-11-28
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.11.28"
    identifier: arxiv_20231128
    parent: 202311_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CV (19)](#cscv-19)
- [cs.CL (7)](#cscl-7)
- [cs.LG (7)](#cslg-7)
- [eess.IV (3)](#eessiv-3)
- [cs.IR (2)](#csir-2)
- [cs.CY (2)](#cscy-2)
- [cs.CR (3)](#cscr-3)
- [cs.RO (2)](#csro-2)
- [cs.NE (1)](#csne-1)
- [eess.SY (1)](#eesssy-1)
- [cs.SE (1)](#csse-1)
- [cs.LO (1)](#cslo-1)
- [q-fin.TR (1)](#q-fintr-1)
- [math.OC (1)](#mathoc-1)
- [math.NA (1)](#mathna-1)

## cs.CV (19)



### (1/52) DISYRE: Diffusion-Inspired SYnthetic REstoration for Unsupervised Anomaly Detection (Sergio Naval Marimont et al., 2023)

{{<citation>}}

Sergio Naval Marimont, Matthew Baugh, Vasilis Siomos, Christos Tzelepis, Bernhard Kainz, Giacomo Tarroni. (2023)  
**DISYRE: Diffusion-Inspired SYnthetic REstoration for Unsupervised Anomaly Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2311.15453v1)  

---


**ABSTRACT**  
Unsupervised Anomaly Detection (UAD) techniques aim to identify and localize anomalies without relying on annotations, only leveraging a model trained on a dataset known to be free of anomalies. Diffusion models learn to modify inputs $x$ to increase the probability of it belonging to a desired distribution, i.e., they model the score function $\nabla_x \log p(x)$. Such a score function is potentially relevant for UAD, since $\nabla_x \log p(x)$ is itself a pixel-wise anomaly score. However, diffusion models are trained to invert a corruption process based on Gaussian noise and the learned score function is unlikely to generalize to medical anomalies. This work addresses the problem of how to learn a score function relevant for UAD and proposes DISYRE: Diffusion-Inspired SYnthetic REstoration. We retain the diffusion-like pipeline but replace the Gaussian noise corruption with a gradual, synthetic anomaly corruption so the learned score function generalizes to medical, naturally occurring anomalies. We evaluate DISYRE on three common Brain MRI UAD benchmarks and substantially outperform other methods in two out of the three tasks.

{{</citation>}}


### (2/52) FLAIR: A Conditional Diffusion Framework with Applications to Face Video Restoration (Zihao Zou et al., 2023)

{{<citation>}}

Zihao Zou, Jiaming Liu, Shirin Shoushtari, Yubo Wang, Weijie Gan, Ulugbek S. Kamilov. (2023)  
**FLAIR: A Conditional Diffusion Framework with Applications to Face Video Restoration**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.15445v1)  

---


**ABSTRACT**  
Face video restoration (FVR) is a challenging but important problem where one seeks to recover a perceptually realistic face videos from a low-quality input. While diffusion probabilistic models (DPMs) have been shown to achieve remarkable performance for face image restoration, they often fail to preserve temporally coherent, high-quality videos, compromising the fidelity of reconstructed faces. We present a new conditional diffusion framework called FLAIR for FVR. FLAIR ensures temporal consistency across frames in a computationally efficient fashion by converting a traditional image DPM into a video DPM. The proposed conversion uses a recurrent video refinement layer and a temporal self-attention at different scales. FLAIR also uses a conditional iterative refinement process to balance the perceptual and distortion quality during inference. This process consists of two key components: a data-consistency module that analytically ensures that the generated video precisely matches its degraded observation and a coarse-to-fine image enhancement module specifically for facial regions. Our extensive experiments show superiority of FLAIR over the current state-of-the-art (SOTA) for video super-resolution, deblurring, JPEG restoration, and space-time frame interpolation on two high-quality face video datasets.

{{</citation>}}


### (3/52) ProtoArgNet: Interpretable Image Classification with Super-Prototypes and Argumentation [Technical Report] (Hamed Ayoobi et al., 2023)

{{<citation>}}

Hamed Ayoobi, Nico Potyka, Francesca Toni. (2023)  
**ProtoArgNet: Interpretable Image Classification with Super-Prototypes and Argumentation [Technical Report]**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2311.15438v1)  

---


**ABSTRACT**  
We propose ProtoArgNet, a novel interpretable deep neural architecture for image classification in the spirit of prototypical-part-learning as found, e.g. in ProtoPNet. While earlier approaches associate every class with multiple prototypical-parts, ProtoArgNet uses super-prototypes that combine prototypical-parts into single prototypical class representations. Furthermore, while earlier approaches use interpretable classification layers, e.g. logistic regression in ProtoPNet, ProtoArgNet improves accuracy with multi-layer perceptrons while relying upon an interpretable reading thereof based on a form of argumentation. ProtoArgNet is customisable to user cognitive requirements by a process of sparsification of the multi-layer perceptron/argumentation component. Also, as opposed to other prototypical-part-learning approaches, ProtoArgNet can recognise spatial relations between different prototypical-parts that are from different regions in images, similar to how CNNs capture relations between patterns recognized in earlier layers.

{{</citation>}}


### (4/52) Wired Perspectives: Multi-View Wire Art Embraces Generative AI (Zhiyu Qu et al., 2023)

{{<citation>}}

Zhiyu Qu, Lan Yang, Honggang Zhang, Tao Xiang, Kaiyue Pang, Yi-Zhe Song. (2023)  
**Wired Perspectives: Multi-View Wire Art Embraces Generative AI**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2311.15421v1)  

---


**ABSTRACT**  
Creating multi-view wire art (MVWA), a static 3D sculpture with diverse interpretations from different viewpoints, is a complex task even for skilled artists. In response, we present DreamWire, an AI system enabling everyone to craft MVWA easily. Users express their vision through text prompts or scribbles, freeing them from intricate 3D wire organisation. Our approach synergises 3D B\'ezier curves, Prim's algorithm, and knowledge distillation from diffusion models or their variants (e.g., ControlNet). This blend enables the system to represent 3D wire art, ensuring spatial continuity and overcoming data scarcity. Extensive evaluation and analysis are conducted to shed insight on the inner workings of the proposed system, including the trade-off between connectivity and visual aesthetics.

{{</citation>}}


### (5/52) BatchNorm-based Weakly Supervised Video Anomaly Detection (Yixuan Zhou et al., 2023)

{{<citation>}}

Yixuan Zhou, Yi Qu, Xing Xu, Fumin Shen, Jingkuan Song, Hengtao Shen. (2023)  
**BatchNorm-based Weakly Supervised Video Anomaly Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2311.15367v1)  

---


**ABSTRACT**  
In weakly supervised video anomaly detection (WVAD), where only video-level labels indicating the presence or absence of abnormal events are available, the primary challenge arises from the inherent ambiguity in temporal annotations of abnormal occurrences. Inspired by the statistical insight that temporal features of abnormal events often exhibit outlier characteristics, we propose a novel method, BN-WVAD, which incorporates BatchNorm into WVAD. In the proposed BN-WVAD, we leverage the Divergence of Feature from Mean vector (DFM) of BatchNorm as a reliable abnormality criterion to discern potential abnormal snippets in abnormal videos. The proposed DFM criterion is also discriminative for anomaly recognition and more resilient to label noise, serving as the additional anomaly score to amend the prediction of the anomaly classifier that is susceptible to noisy labels. Moreover, a batch-level selection strategy is devised to filter more abnormal snippets in videos where more abnormal events occur. The proposed BN-WVAD model demonstrates state-of-the-art performance on UCF-Crime with an AUC of 87.24%, and XD-Violence, where AP reaches up to 84.93%. Our code implementation is accessible at https://github.com/cool-xuan/BN-WVAD.

{{</citation>}}


### (6/52) Adversarial Purification of Information Masking (Sitong Liu et al., 2023)

{{<citation>}}

Sitong Liu, Zhichao Lian, Shuangquan Zhang, Liang Xiao. (2023)  
**Adversarial Purification of Information Masking**  

---
Primary Category: cs.CV  
Categories: cs-CR, cs-CV, cs-LG, cs.CV, eess-IV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2311.15339v1)  

---


**ABSTRACT**  
Adversarial attacks meticulously generate minuscule, imperceptible perturbations to images to deceive neural networks. Counteracting these, adversarial purification methods seek to transform adversarial input samples into clean output images to defend against adversarial attacks. Nonetheless, extent generative models fail to effectively eliminate adversarial perturbations, yielding less-than-ideal purification results. We emphasize the potential threat of residual adversarial perturbations to target models, quantitatively establishing a relationship between perturbation scale and attack capability. Notably, the residual perturbations on the purified image primarily stem from the same-position patch and similar patches of the adversarial sample. We propose a novel adversarial purification approach named Information Mask Purification (IMPure), aims to extensively eliminate adversarial perturbations. To obtain an adversarial sample, we first mask part of the patches information, then reconstruct the patches to resist adversarial perturbations from the patches. We reconstruct all patches in parallel to obtain a cohesive image. Then, in order to protect the purified samples against potential similar regional perturbations, we simulate this risk by randomly mixing the purified samples with the input samples before inputting them into the feature extraction network. Finally, we establish a combined constraint of pixel loss and perceptual loss to augment the model's reconstruction adaptability. Extensive experiments on the ImageNet dataset with three classifier models demonstrate that our approach achieves state-of-the-art results against nine adversarial attack methods. Implementation code and pre-trained weights can be accessed at \textcolor{blue}{https://github.com/NoWindButRain/IMPure}.

{{</citation>}}


### (7/52) Sketch Video Synthesis (Yudian Zheng et al., 2023)

{{<citation>}}

Yudian Zheng, Xiaodong Cun, Menghan Xia, Chi-Man Pun. (2023)  
**Sketch Video Synthesis**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-GR, cs.CV  
Keywords: Sketch  
[Paper Link](http://arxiv.org/abs/2311.15306v1)  

---


**ABSTRACT**  
Understanding semantic intricacies and high-level concepts is essential in image sketch generation, and this challenge becomes even more formidable when applied to the domain of videos. To address this, we propose a novel optimization-based framework for sketching videos represented by the frame-wise B\'ezier curve. In detail, we first propose a cross-frame stroke initialization approach to warm up the location and the width of each curve. Then, we optimize the locations of these curves by utilizing a semantic loss based on CLIP features and a newly designed consistency loss using the self-decomposed 2D atlas network. Built upon these design elements, the resulting sketch video showcases impressive visual abstraction and temporal coherence. Furthermore, by transforming a video into SVG lines through the sketching process, our method unlocks applications in sketch-based video editing and video doodling, enabled through video composition, as exemplified in the teaser.

{{</citation>}}


### (8/52) ChAda-ViT : Channel Adaptive Attention for Joint Representation Learning of Heterogeneous Microscopy Images (Nicolas Bourriez et al., 2023)

{{<citation>}}

Nicolas Bourriez, Ihab Bendidi, Ethan Cohen, Gabriel Watkinson, Maxime Sanchez, Guillaume Bollot, Auguste Genovesio. (2023)  
**ChAda-ViT : Channel Adaptive Attention for Joint Representation Learning of Heterogeneous Microscopy Images**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Attention, Representation Learning, Transformer  
[Paper Link](http://arxiv.org/abs/2311.15264v1)  

---


**ABSTRACT**  
Unlike color photography images, which are consistently encoded into RGB channels, biological images encompass various modalities, where the type of microscopy and the meaning of each channel varies with each experiment. Importantly, the number of channels can range from one to a dozen and their correlation is often comparatively much lower than RGB, as each of them brings specific information content. This aspect is largely overlooked by methods designed out of the bioimage field, and current solutions mostly focus on intra-channel spatial attention, often ignoring the relationship between channels, yet crucial in most biological applications. Importantly, the variable channel type and count prevent the projection of several experiments to a unified representation for large scale pre-training. In this study, we propose ChAda-ViT, a novel Channel Adaptive Vision Transformer architecture employing an Inter-Channel Attention mechanism on images with an arbitrary number, order and type of channels. We also introduce IDRCell100k, a bioimage dataset with a rich set of 79 experiments covering 7 microscope modalities, with a multitude of channel types, and channel counts varying from 1 to 10 per experiment. Our proposed architecture, trained in a self-supervised manner, outperforms existing approaches in several biologically relevant downstream tasks. Additionally, it can be used to bridge the gap for the first time between assays with different microscopes, channel numbers or types by embedding various image and experimental modalities into a unified biological image representation. The latter should facilitate interdisciplinary studies and pave the way for better adoption of deep learning in biological image-based analyses. Code and Data to be released soon.

{{</citation>}}


### (9/52) Revealing Cortical Layers In Histological Brain Images With Self-Supervised Graph Convolutional Networks Applied To Cell-Graphs (Valentina Vadori et al., 2023)

{{<citation>}}

Valentina Vadori, Antonella Peruffo, Jean-Marie Graïc, Giulia Vadori, Livio Finos, Enrico Grisan. (2023)  
**Revealing Cortical Layers In Histological Brain Images With Self-Supervised Graph Convolutional Networks Applied To Cell-Graphs**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV, q-bio-QM  
Keywords: Graph Convolutional Network, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2311.15262v1)  

---


**ABSTRACT**  
Identifying cerebral cortex layers is crucial for comparative studies of the cytoarchitecture aiming at providing insights into the relations between brain structure and function across species. The absence of extensive annotated datasets typically limits the adoption of machine learning approaches, leading to the manual delineation of cortical layers by neuroanatomists. We introduce a self-supervised approach to detect layers in 2D Nissl-stained histological slices of the cerebral cortex. It starts with the segmentation of individual cells and the creation of an attributed cell-graph. A self-supervised graph convolutional network generates cell embeddings that encode morphological and structural traits of the cellular environment and are exploited by a community detection algorithm for the final layering. Our method, the first self-supervised of its kind with no spatial transcriptomics data involved, holds the potential to accelerate cytoarchitecture analyses, sidestepping annotation needs and advancing cross-species investigation.

{{</citation>}}


### (10/52) ID-like Prompt Learning for Few-Shot Out-of-Distribution Detection (Yichen Bai et al., 2023)

{{<citation>}}

Yichen Bai, Zongbo Han, Changqing Zhang, Bing Cao, Xiaoheng Jiang, Qinghua Hu. (2023)  
**ID-like Prompt Learning for Few-Shot Out-of-Distribution Detection**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Few-Shot, ImageNet  
[Paper Link](http://arxiv.org/abs/2311.15243v1)  

---


**ABSTRACT**  
Out-of-distribution (OOD) detection methods often exploit auxiliary outliers to train model identifying OOD samples, especially discovering challenging outliers from auxiliary outliers dataset to improve OOD detection. However, they may still face limitations in effectively distinguishing between the most challenging OOD samples that are much like in-distribution (ID) data, i.e., ID-like samples. To this end, we propose a novel OOD detection framework that discovers ID-like outliers using CLIP from the vicinity space of the ID samples, thus helping to identify these most challenging OOD samples. Then a prompt learning framework is proposed that utilizes the identified ID-like outliers to further leverage the capabilities of CLIP for OOD detection. Benefiting from the powerful CLIP, we only need a small number of ID samples to learn the prompts of the model without exposing other auxiliary outlier datasets. By focusing on the most challenging ID-like OOD samples and elegantly exploiting the capabilities of CLIP, our method achieves superior few-shot learning performance on various real-world image datasets (e.g., in 4-shot OOD detection on the ImageNet-1k dataset, our method reduces the average FPR95 by 12.16% and improves the average AUROC by 2.76%, compared to state-of-the-art methods).

{{</citation>}}


### (11/52) CalibFormer: A Transformer-based Automatic LiDAR-Camera Calibration Network (Yuxuan Xiao et al., 2023)

{{<citation>}}

Yuxuan Xiao, Yao Li, Chengzhen Meng, Xingchen Li, Yanyong Zhang. (2023)  
**CalibFormer: A Transformer-based Automatic LiDAR-Camera Calibration Network**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.15241v1)  

---


**ABSTRACT**  
The fusion of LiDARs and cameras has been increasingly adopted in autonomous driving for perception tasks. The performance of such fusion-based algorithms largely depends on the accuracy of sensor calibration, which is challenging due to the difficulty of identifying common features across different data modalities. Previously, many calibration methods involved specific targets and/or manual intervention, which has proven to be cumbersome and costly. Learning-based online calibration methods have been proposed, but their performance is barely satisfactory in most cases. These methods usually suffer from issues such as sparse feature maps, unreliable cross-modality association, inaccurate calibration parameter regression, etc. In this paper, to address these issues, we propose CalibFormer, an end-to-end network for automatic LiDAR-camera calibration. We aggregate multiple layers of camera and LiDAR image features to achieve high-resolution representations. A multi-head correlation module is utilized to identify correlations between features more accurately. Lastly, we employ transformer architectures to estimate accurate calibration parameters from the correlation information. Our method achieved a mean translation error of $0.8751 \mathrm{cm}$ and a mean rotation error of $0.0562 ^{\circ}$ on the KITTI dataset, surpassing existing state-of-the-art methods and demonstrating strong robustness, accuracy, and generalization capabilities.

{{</citation>}}


### (12/52) Double Reverse Regularization Network Based on Self-Knowledge Distillation for SAR Object Classification (Bo Xu et al., 2023)

{{<citation>}}

Bo Xu, Hao Zheng, Zhigang Hu, Liu Yang, Meiguang Zheng. (2023)  
**Double Reverse Regularization Network Based on Self-Knowledge Distillation for SAR Object Classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV, eess-IV  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2311.15231v1)  

---


**ABSTRACT**  
In current synthetic aperture radar (SAR) object classification, one of the major challenges is the severe overfitting issue due to the limited dataset (few-shot) and noisy data. Considering the advantages of knowledge distillation as a learned label smoothing regularization, this paper proposes a novel Double Reverse Regularization Network based on Self-Knowledge Distillation (DRRNet-SKD). Specifically, through exploring the effect of distillation weight on the process of distillation, we are inspired to adopt the double reverse thought to implement an effective regularization network by combining offline and online distillation in a complementary way. Then, the Adaptive Weight Assignment (AWA) module is designed to adaptively assign two reverse-changing weights based on the network performance, allowing the student network to better benefit from both teachers. The experimental results on OpenSARShip and FUSAR-Ship demonstrate that DRRNet-SKD exhibits remarkable performance improvement on classical CNNs, outperforming state-of-the-art self-knowledge distillation methods.

{{</citation>}}


### (13/52) GAIA: Zero-shot Talking Avatar Generation (Tianyu He et al., 2023)

{{<citation>}}

Tianyu He, Junliang Guo, Runyi Yu, Yuchi Wang, Jialiang Zhu, Kaikai An, Leyi Li, Xu Tan, Chunyu Wang, Han Hu, HsiangTao Wu, Sheng Zhao, Jiang Bian. (2023)  
**GAIA: Zero-shot Talking Avatar Generation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2311.15230v1)  

---


**ABSTRACT**  
Zero-shot talking avatar generation aims at synthesizing natural talking videos from speech and a single portrait image. Previous methods have relied on domain-specific heuristics such as warping-based motion representation and 3D Morphable Models, which limit the naturalness and diversity of the generated avatars. In this work, we introduce GAIA (Generative AI for Avatar), which eliminates the domain priors in talking avatar generation. In light of the observation that the speech only drives the motion of the avatar while the appearance of the avatar and the background typically remain the same throughout the entire video, we divide our approach into two stages: 1) disentangling each frame into motion and appearance representations; 2) generating motion sequences conditioned on the speech and reference portrait image. We collect a large-scale high-quality talking avatar dataset and train the model on it with different scales (up to 2B parameters). Experimental results verify the superiority, scalability, and flexibility of GAIA as 1) the resulting model beats previous baseline models in terms of naturalness, diversity, lip-sync quality, and visual quality; 2) the framework is scalable since larger models yield better results; 3) it is general and enables different applications like controllable talking avatar generation and text-instructed avatar generation.

{{</citation>}}


### (14/52) One-bit Supervision for Image Classification: Problem, Solution, and Beyond (Hengtong Hu et al., 2023)

{{<citation>}}

Hengtong Hu, Lingxi Xie, Xinyue Hue, Richang Hong, Qi Tian. (2023)  
**One-bit Supervision for Image Classification: Problem, Solution, and Beyond**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2311.15225v1)  

---


**ABSTRACT**  
This paper presents one-bit supervision, a novel setting of learning with fewer labels, for image classification. Instead of training model using the accurate label of each sample, our setting requires the model to interact with the system by predicting the class label of each sample and learn from the answer whether the guess is correct, which provides one bit (yes or no) of information. An intriguing property of the setting is that the burden of annotation largely alleviates in comparison to offering the accurate label. There are two keys to one-bit supervision, which are (i) improving the guess accuracy and (ii) making good use of the incorrect guesses. To achieve these goals, we propose a multi-stage training paradigm and incorporate negative label suppression into an off-the-shelf semi-supervised learning algorithm. Theoretical analysis shows that one-bit annotation is more efficient than full-bit annotation in most cases and gives the conditions of combining our approach with active learning. Inspired by this, we further integrate the one-bit supervision framework into the self-supervised learning algorithm which yields an even more efficient training schedule. Different from training from scratch, when self-supervised learning is used for initialization, both hard example mining and class balance are verified effective in boosting the learning performance. However, these two frameworks still need full-bit labels in the initial stage. To cast off this burden, we utilize unsupervised domain adaptation to train the initial model and conduct pure one-bit annotations on the target dataset. In multiple benchmarks, the learning efficiency of the proposed approach surpasses that using full-bit, semi-supervised supervision.

{{</citation>}}


### (15/52) Insect-Foundation: A Foundation Model and Large-scale 1M Dataset for Visual Insect Understanding (Hoang-Quan Nguyen et al., 2023)

{{<citation>}}

Hoang-Quan Nguyen, Thanh-Dat Truong, Xuan Bac Nguyen, Ashley Dowling, Xin Li, Khoa Luu. (2023)  
**Insect-Foundation: A Foundation Model and Large-scale 1M Dataset for Visual Insect Understanding**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2311.15206v1)  

---


**ABSTRACT**  
In precision agriculture, the detection and recognition of insects play an essential role in the ability of crops to grow healthy and produce a high-quality yield. The current machine vision model requires a large volume of data to achieve high performance. However, there are approximately 5.5 million different insect species in the world. None of the existing insect datasets can cover even a fraction of them due to varying geographic locations and acquisition costs. In this paper, we introduce a novel ``Insect-1M'' dataset, a game-changing resource poised to revolutionize insect-related foundation model training. Covering a vast spectrum of insect species, our dataset, including 1 million images with dense identification labels of taxonomy hierarchy and insect descriptions, offers a panoramic view of entomology, enabling foundation models to comprehend visual and semantic information about insects like never before. Then, to efficiently establish an Insect Foundation Model, we develop a micro-feature self-supervised learning method with a Patch-wise Relevant Attention mechanism capable of discerning the subtle differences among insect images. In addition, we introduce Description Consistency loss to improve micro-feature modeling via insect descriptions. Through our experiments, we illustrate the effectiveness of our proposed approach in insect modeling and achieve State-of-the-Art performance on standard benchmarks of insect-related tasks. Our Insect Foundation Model and Dataset promise to empower the next generation of insect-related vision models, bringing them closer to the ultimate goal of precision agriculture.

{{</citation>}}


### (16/52) SpliceMix: A Cross-scale and Semantic Blending Augmentation Strategy for Multi-label Image Classification (Lei Wang et al., 2023)

{{<citation>}}

Lei Wang, Yibing Zhan, Leilei Ma, Dapeng Tao, Liang Ding, Chen Gong. (2023)  
**SpliceMix: A Cross-scale and Semantic Blending Augmentation Strategy for Multi-label Image Classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Augmentation, Image Classification  
[Paper Link](http://arxiv.org/abs/2311.15200v1)  

---


**ABSTRACT**  
Recently, Mix-style data augmentation methods (e.g., Mixup and CutMix) have shown promising performance in various visual tasks. However, these methods are primarily designed for single-label images, ignoring the considerable discrepancies between single- and multi-label images, i.e., a multi-label image involves multiple co-occurred categories and fickle object scales. On the other hand, previous multi-label image classification (MLIC) methods tend to design elaborate models, bringing expensive computation. In this paper, we introduce a simple but effective augmentation strategy for multi-label image classification, namely SpliceMix. The "splice" in our method is two-fold: 1) Each mixed image is a splice of several downsampled images in the form of a grid, where the semantics of images attending to mixing are blended without object deficiencies for alleviating co-occurred bias; 2) We splice mixed images and the original mini-batch to form a new SpliceMixed mini-batch, which allows an image with different scales to contribute to training together. Furthermore, such splice in our SpliceMixed mini-batch enables interactions between mixed images and original regular images. We also offer a simple and non-parametric extension based on consistency learning (SpliceMix-CL) to show the flexible extensibility of our SpliceMix. Extensive experiments on various tasks demonstrate that only using SpliceMix with a baseline model (e.g., ResNet) achieves better performance than state-of-the-art methods. Moreover, the generalizability of our SpliceMix is further validated by the improvements in current MLIC methods when married with our SpliceMix. The code is available at https://github.com/zuiran/SpliceMix.

{{</citation>}}


### (17/52) IA-LSTM: Interaction-Aware LSTM for Pedestrian Trajectory Prediction (Yuehai Chen, 2023)

{{<citation>}}

Yuehai Chen. (2023)  
**IA-LSTM: Interaction-Aware LSTM for Pedestrian Trajectory Prediction**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2311.15193v1)  

---


**ABSTRACT**  
Predicting the trajectory of pedestrians in crowd scenarios is indispensable in self-driving or autonomous mobile robot field because estimating the future locations of pedestrians around is beneficial for policy decision to avoid collision. It is a challenging issue because humans have different walking motions and the interactions between humans and objects in the current environment, especially between human themselves, are complex. Previous researches have focused on how to model the human-human interactions, however, neglecting the relative importance of interactions. In order to address this issue, we introduce a novel mechanism based on the correntropy, which not only can measure the relative importance of human-human interactions, but also can build personal space for each pedestrian. We further propose an Interaction Module including this data-driven mechanism that can effectively extract feature representations of dynamic human-human interactions in the scene and calculate corresponding weights to represent the importance of different interactions. To share such social messages among pedestrians, we design an interaction-aware architecture based on the Long Short-Term Memory (LSTM) network for trajectory prediction. We demonstrate the performance of our model on two public datasets and the experimental results demonstrate that our model can achieve better performance than several latest methods with good performance.

{{</citation>}}


### (18/52) Advancing Vision Transformers with Group-Mix Attention (Chongjian Ge et al., 2023)

{{<citation>}}

Chongjian Ge, Xiaohan Ding, Zhan Tong, Li Yuan, Jiangliu Wang, Yibing Song, Ping Luo. (2023)  
**Advancing Vision Transformers with Group-Mix Attention**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, ImageNet, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.15157v1)  

---


**ABSTRACT**  
Vision Transformers (ViTs) have been shown to enhance visual recognition through modeling long-range dependencies with multi-head self-attention (MHSA), which is typically formulated as Query-Key-Value computation. However, the attention map generated from the Query and Key captures only token-to-token correlations at one single granularity. In this paper, we argue that self-attention should have a more comprehensive mechanism to capture correlations among tokens and groups (i.e., multiple adjacent tokens) for higher representational capacity. Thereby, we propose Group-Mix Attention (GMA) as an advanced replacement for traditional self-attention, which can simultaneously capture token-to-token, token-to-group, and group-to-group correlations with various group sizes. To this end, GMA splits the Query, Key, and Value into segments uniformly and performs different group aggregations to generate group proxies. The attention map is computed based on the mixtures of tokens and group proxies and used to re-combine the tokens and groups in Value. Based on GMA, we introduce a powerful backbone, namely GroupMixFormer, which achieves state-of-the-art performance in image classification, object detection, and semantic segmentation with fewer parameters than existing models. For instance, GroupMixFormer-L (with 70.3M parameters and 384^2 input) attains 86.2% Top-1 accuracy on ImageNet-1K without external data, while GroupMixFormer-B (with 45.8M parameters) attains 51.2% mIoU on ADE20K.

{{</citation>}}


### (19/52) Self-Supervised Learning for SAR ATR with a Knowledge-Guided Predictive Architecture (Weijie Li et al., 2023)

{{<citation>}}

Weijie Li, Yang Wei, Tianpeng Liu, Yuenan Hou, Yongxiang Liu, Li Liu. (2023)  
**Self-Supervised Learning for SAR ATR with a Knowledge-Guided Predictive Architecture**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2311.15153v1)  

---


**ABSTRACT**  
Recently, the emergence of a large number of Synthetic Aperture Radar (SAR) sensors and target datasets has made it possible to unify downstream tasks with self-supervised learning techniques, which can pave the way for building the foundation model in the SAR target recognition field. The major challenge of self-supervised learning for SAR target recognition lies in the generalizable representation learning in low data quality and noise.To address the aforementioned problem, we propose a knowledge-guided predictive architecture that uses local masked patches to predict the multiscale SAR feature representations of unseen context. The core of the proposed architecture lies in combining traditional SAR domain feature extraction with state-of-the-art scalable self-supervised learning for accurate generalized feature representations. The proposed framework is validated on various downstream datasets (MSTAR, FUSAR-Ship, SAR-ACD and SSDD), and can bring consistent performance improvement for SAR target recognition. The experimental results strongly demonstrate the unified performance improvement of the self-supervised learning technique for SAR target recognition across diverse targets, scenes and sensors.

{{</citation>}}


## cs.CL (7)



### (20/52) Uncertainty-aware Language Modeling for Selective Question Answering (Qi Yang et al., 2023)

{{<citation>}}

Qi Yang, Shreya Ravikumar, Fynn Schmitt-Ulms, Satvik Lolla, Ege Demir, Iaroslav Elistratov, Alex Lavaee, Sadhana Lolla, Elaheh Ahmadi, Daniela Rus, Alexander Amini, Alejandro Perez. (2023)  
**Uncertainty-aware Language Modeling for Selective Question Answering**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: BERT, Language Model, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2311.15451v1)  

---


**ABSTRACT**  
We present an automatic large language model (LLM) conversion approach that produces uncertainty-aware LLMs capable of estimating uncertainty with every prediction. Our approach is model- and data-agnostic, is computationally-efficient, and does not rely on external models or systems. We evaluate converted models on the selective question answering setting -- to answer as many questions as possible while maintaining a given accuracy, forgoing providing predictions when necessary. As part of our results, we test BERT and Llama 2 model variants on the SQuAD extractive QA task and the TruthfulQA generative QA task. We show that using the uncertainty estimates provided by our approach to selectively answer questions leads to significantly higher accuracy over directly using model probabilities.

{{</citation>}}


### (21/52) Learning to Skip for Language Modeling (Dewen Zeng et al., 2023)

{{<citation>}}

Dewen Zeng, Nan Du, Tao Wang, Yuanzhong Xu, Tao Lei, Zhifeng Chen, Claire Cui. (2023)  
**Learning to Skip for Language Modeling**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2311.15436v1)  

---


**ABSTRACT**  
Overparameterized large-scale language models have impressive generalization performance of in-context few-shot learning. However, most language models allocate the same amount of parameters or computation to each token, disregarding the complexity or importance of the input data. We argue that in language model pretraining, a variable amount of computation should be assigned to different tokens, and this can be efficiently achieved via a simple routing mechanism. Different from conventional early stopping techniques where tokens can early exit at only early layers, we propose a more general method that dynamically skips the execution of a layer (or module) for any input token with a binary router. In our extensive evaluation across 24 NLP tasks, we demonstrate that the proposed method can significantly improve the 1-shot performance compared to other competitive baselines only at mild extra cost for inference.

{{</citation>}}


### (22/52) Machine-Generated Text Detection using Deep Learning (Raghav Gaggar et al., 2023)

{{<citation>}}

Raghav Gaggar, Ashish Bhagchandani, Harsh Oza. (2023)  
**Machine-Generated Text Detection using Deep Learning**  

---
Primary Category: cs.CL  
Categories: I-2-7; I-5-4; I-2-6, cs-CL, cs.CL  
Keywords: BERT, GPT, GPT-3.5, Language Model, QA, Twitter  
[Paper Link](http://arxiv.org/abs/2311.15425v1)  

---


**ABSTRACT**  
Our research focuses on the crucial challenge of discerning text produced by Large Language Models (LLMs) from human-generated text, which holds significance for various applications. With ongoing discussions about attaining a model with such functionality, we present supporting evidence regarding the feasibility of such models. We evaluated our models on multiple datasets, including Twitter Sentiment, Football Commentary, Project Gutenberg, PubMedQA, and SQuAD, confirming the efficacy of the enhanced detection approaches. These datasets were sampled with intricate constraints encompassing every possibility, laying the foundation for future research. We evaluate GPT-3.5-Turbo against various detectors such as SVM, RoBERTa-base, and RoBERTa-large. Based on the research findings, the results predominantly relied on the sequence length of the sentence.

{{</citation>}}


### (23/52) Learning Section Weights for Multi-Label Document Classification (Maziar Moradi Fard et al., 2023)

{{<citation>}}

Maziar Moradi Fard, Paula Sorrolla Bayod, Kiomars Motarjem, Mohammad Alian Nejadi, Saber Akhondi, Camilo Thorne. (2023)  
**Learning Section Weights for Multi-Label Document Classification**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2311.15402v1)  

---


**ABSTRACT**  
Multi-label document classification is a traditional task in NLP. Compared to single-label classification, each document can be assigned multiple classes. This problem is crucially important in various domains, such as tagging scientific articles. Documents are often structured into several sections such as abstract and title. Current approaches treat different sections equally for multi-label classification. We argue that this is not a realistic assumption, leading to sub-optimal results. Instead, we propose a new method called Learning Section Weights (LSW), leveraging the contribution of each distinct section for multi-label classification. Via multiple feed-forward layers, LSW learns to assign weights to each section of, and incorporate the weights in the prediction. We demonstrate our approach on scientific articles. Experimental results on public (arXiv) and private (Elsevier) datasets confirm the superiority of LSW, compared to state-of-the-art multi-label document classification methods. In particular, LSW achieves a 1.3% improvement in terms of macro averaged F1-score while it achieves 1.3% in terms of macro averaged recall on the publicly available arXiv dataset.

{{</citation>}}


### (24/52) Enhancing Empathetic and Emotion Support Dialogue Generation with Prophetic Commonsense Inference (Lanrui Wang et al., 2023)

{{<citation>}}

Lanrui Wang, Jiangnan Li, Chenxu Yang, Zheng Lin, Weiping Wang. (2023)  
**Enhancing Empathetic and Emotion Support Dialogue Generation with Prophetic Commonsense Inference**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue, Language Model  
[Paper Link](http://arxiv.org/abs/2311.15316v1)  

---


**ABSTRACT**  
The interest in Empathetic and Emotional Support conversations among the public has significantly increased. To offer more sensitive and understanding responses, leveraging commonsense knowledge has become a common strategy to better understand psychological aspects and causality. However, such commonsense inferences can be out of context and unable to predict upcoming dialogue themes, resulting in responses that lack coherence and empathy. To remedy this issue, we present Prophetic Commonsense Inference, an innovative paradigm for inferring commonsense knowledge. By harnessing the capabilities of Large Language Models in understanding dialogue and making commonsense deductions, we train tunable models to bridge the gap between past and potential future dialogues. Extensive experiments conducted on EmpatheticDialogues and Emotion Support Conversation show that equipping dialogue agents with our proposed prophetic commonsense inference significantly enhances the quality of their responses.

{{</citation>}}


### (25/52) UHGEval: Benchmarking the Hallucination of Chinese Large Language Models via Unconstrained Generation (Xun Liang et al., 2023)

{{<citation>}}

Xun Liang, Shichao Song, Simin Niu, Zhiyu Li, Feiyu Xiong, Bo Tang, Zhaohui Wy, Dawei He, Peng Cheng, Zhonghao Wang, Haiying Deng. (2023)  
**UHGEval: Benchmarking the Hallucination of Chinese Large Language Models via Unconstrained Generation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2311.15296v1)  

---


**ABSTRACT**  
Large language models (LLMs) have emerged as pivotal contributors in contemporary natural language processing and are increasingly being applied across a diverse range of industries. However, these large-scale probabilistic statistical models cannot currently ensure the requisite quality in professional content generation. These models often produce hallucinated text, compromising their practical utility in professional contexts. To assess the authentic reliability of LLMs in text generation, numerous initiatives have developed benchmark evaluations for hallucination phenomena. Nevertheless, these benchmarks frequently utilize constrained generation techniques due to cost and temporal constraints. These techniques encompass the use of directed hallucination induction and strategies that deliberately alter authentic text to produce hallucinations. These approaches are not congruent with the unrestricted text generation demanded by real-world applications. Furthermore, a well-established Chinese-language dataset dedicated to the evaluation of hallucinations in text generation is presently lacking. Consequently, we have developed an Unconstrained Hallucination Generation Evaluation (UHGEval) benchmark, designed to compile outputs produced with minimal restrictions by LLMs. Concurrently, we have established a comprehensive benchmark evaluation framework to aid subsequent researchers in undertaking scalable and reproducible experiments. We have also executed extensive experiments, evaluating prominent Chinese language models and the GPT series models to derive professional performance insights regarding hallucination challenges.

{{</citation>}}


### (26/52) Probabilistic Transformer: A Probabilistic Dependency Model for Contextual Word Representation (Haoyi Wu et al., 2023)

{{<citation>}}

Haoyi Wu, Kewei Tu. (2023)  
**Probabilistic Transformer: A Probabilistic Dependency Model for Contextual Word Representation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP, Transformer, Word Representation  
[Paper Link](http://arxiv.org/abs/2311.15211v1)  

---


**ABSTRACT**  
Syntactic structures used to play a vital role in natural language processing (NLP), but since the deep learning revolution, NLP has been gradually dominated by neural models that do not consider syntactic structures in their design. One vastly successful class of neural models is transformers. When used as an encoder, a transformer produces contextual representation of words in the input sentence. In this work, we propose a new model of contextual word representation, not from a neural perspective, but from a purely syntactic and probabilistic perspective. Specifically, we design a conditional random field that models discrete latent representations of all words in a sentence as well as dependency arcs between them; and we use mean field variational inference for approximate inference. Strikingly, we find that the computation graph of our model resembles transformers, with correspondences between dependencies and self-attention and between distributions over latent representations and contextual embeddings of words. Experiments show that our model performs competitively to transformers on small to medium sized datasets. We hope that our work could help bridge the gap between traditional syntactic and probabilistic approaches and cutting-edge neural approaches to NLP, and inspire more linguistically-principled neural approaches in the future.

{{</citation>}}


## cs.LG (7)



### (27/52) GGNNs : Generalizing GNNs using Residual Connections and Weighted Message Passing (Abhinav Raghuvanshi et al., 2023)

{{<citation>}}

Abhinav Raghuvanshi, Kushal Sokke Malleshappa. (2023)  
**GGNNs : Generalizing GNNs using Residual Connections and Weighted Message Passing**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2311.15448v1)  

---


**ABSTRACT**  
Many real-world phenomena can be modeled as a graph, making them extremely valuable due to their ubiquitous presence. GNNs excel at capturing those relationships and patterns within these graphs, enabling effective learning and prediction tasks. GNNs are constructed using Multi-Layer Perceptrons (MLPs) and incorporate additional layers for message passing to facilitate the flow of features among nodes. It is commonly believed that the generalizing power of GNNs is attributed to the message-passing mechanism between layers, where nodes exchange information with their neighbors, enabling them to effectively capture and propagate information across the nodes of a graph. Our technique builds on these results, modifying the message-passing mechanism further: one by weighing the messages before accumulating at each node and another by adding Residual connections. These two mechanisms show significant improvements in learning and faster convergence

{{</citation>}}


### (28/52) KOPPA: Improving Prompt-based Continual Learning with Key-Query Orthogonal Projection and Prototype-based One-Versus-All (Quyen Tran et al., 2023)

{{<citation>}}

Quyen Tran, Lam Tran, Khoat Than, Toan Tran, Dinh Phung, Trung Le. (2023)  
**KOPPA: Improving Prompt-based Continual Learning with Key-Query Orthogonal Projection and Prototype-based One-Versus-All**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.15414v1)  

---


**ABSTRACT**  
Drawing inspiration from prompt tuning techniques applied to Large Language Models, recent methods based on pre-trained ViT networks have achieved remarkable results in the field of Continual Learning. Specifically, these approaches propose to maintain a set of prompts and allocate a subset of them to learn each task using a key-query matching strategy. However, they may encounter limitations when lacking control over the correlations between old task queries and keys of future tasks, the shift of features in the latent space, and the relative separation of latent vectors learned in independent tasks. In this work, we introduce a novel key-query learning strategy based on orthogonal projection, inspired by model-agnostic meta-learning, to enhance prompt matching efficiency and address the challenge of shifting features. Furthermore, we introduce a One-Versus-All (OVA) prototype-based component that enhances the classification head distinction. Experimental results on benchmark datasets demonstrate that our method empowers the model to achieve results surpassing those of current state-of-the-art approaches by a large margin of up to 20%.

{{</citation>}}


### (29/52) Generative Modelling of Stochastic Actions with Arbitrary Constraints in Reinforcement Learning (Changyu Chen et al., 2023)

{{<citation>}}

Changyu Chen, Ramesha Karunasena, Thanh Hong Nguyen, Arunesh Sinha, Pradeep Varakantham. (2023)  
**Generative Modelling of Stochastic Actions with Arbitrary Constraints in Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.15341v1)  

---


**ABSTRACT**  
Many problems in Reinforcement Learning (RL) seek an optimal policy with large discrete multidimensional yet unordered action spaces; these include problems in randomized allocation of resources such as placements of multiple security resources and emergency response units, etc. A challenge in this setting is that the underlying action space is categorical (discrete and unordered) and large, for which existing RL methods do not perform well. Moreover, these problems require validity of the realized action (allocation); this validity constraint is often difficult to express compactly in a closed mathematical form. The allocation nature of the problem also prefers stochastic optimal policies, if one exists. In this work, we address these challenges by (1) applying a (state) conditional normalizing flow to compactly represent the stochastic policy -- the compactness arises due to the network only producing one sampled action and the corresponding log probability of the action, which is then used by an actor-critic method; and (2) employing an invalid action rejection method (via a valid action oracle) to update the base policy. The action rejection is enabled by a modified policy gradient that we derive. Finally, we conduct extensive experiments to show the scalability of our approach compared to prior methods and the ability to enforce arbitrary state-conditional constraints on the support of the distribution of actions in any state.

{{</citation>}}


### (30/52) Token Recycling for Efficient Sequential Inference with Vision Transformers (Jan Olszewski et al., 2023)

{{<citation>}}

Jan Olszewski, Dawid Rymarczyk, Piotr Wójcik, Mateusz Pach, Bartosz Zieliński. (2023)  
**Token Recycling for Efficient Sequential Inference with Vision Transformers**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.15335v1)  

---


**ABSTRACT**  
Vision Transformers (ViTs) overpass Convolutional Neural Networks in processing incomplete inputs because they do not require the imputation of missing values. Therefore, ViTs are well suited for sequential decision-making, e.g. in the Active Visual Exploration problem. However, they are computationally inefficient because they perform a full forward pass each time a piece of new sequential information arrives.   To reduce this computational inefficiency, we introduce the TOken REcycling (TORE) modification for the ViT inference, which can be used with any architecture. TORE divides ViT into two parts, iterator and aggregator. An iterator processes sequential information separately into midway tokens, which are cached. The aggregator processes midway tokens jointly to obtain the prediction. This way, we can reuse the results of computations made by iterator.   Except for efficient sequential inference, we propose a complementary training policy, which significantly reduces the computational burden associated with sequential decision-making while achieving state-of-the-art accuracy.

{{</citation>}}


### (31/52) Bias-Variance Trade-off in Physics-Informed Neural Networks with Randomized Smoothing for High-Dimensional PDEs (Zheyuan Hu et al., 2023)

{{<citation>}}

Zheyuan Hu, Zhouhao Yang, Yezhen Wang, George Em Karniadakis, Kenji Kawaguchi. (2023)  
**Bias-Variance Trade-off in Physics-Informed Neural Networks with Randomized Smoothing for High-Dimensional PDEs**  

---
Primary Category: cs.LG  
Categories: 14J60, cs-AI, cs-LG, cs-NA, cs.LG, math-DS, math-NA, stat-ML  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2311.15283v1)  

---


**ABSTRACT**  
While physics-informed neural networks (PINNs) have been proven effective for low-dimensional partial differential equations (PDEs), the computational cost remains a hurdle in high-dimensional scenarios. This is particularly pronounced when computing high-order and high-dimensional derivatives in the physics-informed loss. Randomized Smoothing PINN (RS-PINN) introduces Gaussian noise for stochastic smoothing of the original neural net model, enabling Monte Carlo methods for derivative approximation, eliminating the need for costly auto-differentiation. Despite its computational efficiency in high dimensions, RS-PINN introduces biases in both loss and gradients, negatively impacting convergence, especially when coupled with stochastic gradient descent (SGD). We present a comprehensive analysis of biases in RS-PINN, attributing them to the nonlinearity of the Mean Squared Error (MSE) loss and the PDE nonlinearity. We propose tailored bias correction techniques based on the order of PDE nonlinearity. The unbiased RS-PINN allows for a detailed examination of its pros and cons compared to the biased version. Specifically, the biased version has a lower variance and runs faster than the unbiased version, but it is less accurate due to the bias. To optimize the bias-variance trade-off, we combine the two approaches in a hybrid method that balances the rapid convergence of the biased version with the high accuracy of the unbiased version. In addition, we present an enhanced implementation of RS-PINN. Extensive experiments on diverse high-dimensional PDEs, including Fokker-Planck, HJB, viscous Burgers', Allen-Cahn, and Sine-Gordon equations, illustrate the bias-variance trade-off and highlight the effectiveness of the hybrid RS-PINN. Empirical guidelines are provided for selecting biased, unbiased, or hybrid versions, depending on the dimensionality and nonlinearity of the specific PDE problem.

{{</citation>}}


### (32/52) A Nearly Optimal and Low-Switching Algorithm for Reinforcement Learning with General Function Approximation (Heyang Zhao et al., 2023)

{{<citation>}}

Heyang Zhao, Jiafan He, Quanquan Gu. (2023)  
**A Nearly Optimal and Low-Switching Algorithm for Reinforcement Learning with General Function Approximation**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, math-OC, stat-ML  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.15238v1)  

---


**ABSTRACT**  
The exploration-exploitation dilemma has been a central challenge in reinforcement learning (RL) with complex model classes. In this paper, we propose a new algorithm, Monotonic Q-Learning with Upper Confidence Bound (MQL-UCB) for RL with general function approximation. Our key algorithmic design includes (1) a general deterministic policy-switching strategy that achieves low switching cost, (2) a monotonic value function structure with carefully controlled function class complexity, and (3) a variance-weighted regression scheme that exploits historical trajectories with high data efficiency. MQL-UCB achieves minimax optimal regret of $\tilde{O}(d\sqrt{HK})$ when $K$ is sufficiently large and near-optimal policy switching cost of $\tilde{O}(dH)$, with $d$ being the eluder dimension of the function class, $H$ being the planning horizon, and $K$ being the number of episodes.   Our work sheds light on designing provably sample-efficient and deployment-efficient Q-learning with nonlinear function approximation.

{{</citation>}}


### (33/52) Decision Tree Psychological Risk Assessment in Currency Trading (Jai Pal, 2023)

{{<citation>}}

Jai Pal. (2023)  
**Decision Tree Psychological Risk Assessment in Currency Trading**  

---
Primary Category: cs.LG  
Categories: cs-CE, cs-LG, cs.LG, q-fin-GN  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.15222v1)  

---


**ABSTRACT**  
This research paper focuses on the integration of Artificial Intelligence (AI) into the currency trading landscape, positing the development of personalized AI models, essentially functioning as intelligent personal assistants tailored to the idiosyncrasies of individual traders. The paper posits that AI models are capable of identifying nuanced patterns within the trader's historical data, facilitating a more accurate and insightful assessment of psychological risk dynamics in currency trading. The PRI is a dynamic metric that experiences fluctuations in response to market conditions that foster psychological fragility among traders. By employing sophisticated techniques, a classifying decision tree is crafted, enabling clearer decision-making boundaries within the tree structure. By incorporating the user's chronological trade entries, the model becomes adept at identifying critical junctures when psychological risks are heightened. The real-time nature of the calculations enhances the model's utility as a proactive tool, offering timely alerts to traders about impending moments of psychological risks. The implications of this research extend beyond the confines of currency trading, reaching into the realms of other industries where the judicious application of personalized modeling emerges as an efficient and strategic approach. This paper positions itself at the intersection of cutting-edge technology and the intricate nuances of human psychology, offering a transformative paradigm for decision making support in dynamic and high-pressure environments.

{{</citation>}}


## eess.IV (3)



### (34/52) Quality Modeling Under A Relaxed Natural Scene Statistics Model (Abhinau K. Venkataramanan et al., 2023)

{{<citation>}}

Abhinau K. Venkataramanan, Alan C. Bovik. (2023)  
**Quality Modeling Under A Relaxed Natural Scene Statistics Model**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV, math-ST, stat-TH  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2311.15437v1)  

---


**ABSTRACT**  
Information-theoretic image quality assessment (IQA) models such as Visual Information Fidelity (VIF) and Spatio-temporal Reduced Reference Entropic Differences (ST-RRED) have enjoyed great success by seamlessly integrating natural scene statistics (NSS) with information theory. The Gaussian Scale Mixture (GSM) model that governs the wavelet subband coefficients of natural images forms the foundation for these algorithms. However, the explosion of user-generated content on social media, which is typically distorted by one or more of many possible unknown impairments, has revealed the limitations of NSS-based IQA models that rely on the simple GSM model. Here, we seek to elaborate the VIF index by deriving useful properties of the Multivariate Generalized Gaussian Distribution (MGGD), and using them to study the behavior of VIF under a Generalized GSM (GGSM) model.

{{</citation>}}


### (35/52) Spectro-ViT: A Vision Transformer Model for GABA-edited MRS Reconstruction Using Spectrograms (Gabriel Dias et al., 2023)

{{<citation>}}

Gabriel Dias, Rodrigo Pommot Berto, Mateus Oliveira, Lucas Ueda, Sergio Dertkigil, Paula D. P. Costa, Amirmohammad Shamaei, Roberto Souza, Ashley Harris, Leticia Rittner. (2023)  
**Spectro-ViT: A Vision Transformer Model for GABA-edited MRS Reconstruction Using Spectrograms**  

---
Primary Category: eess.IV  
Categories: cs-LG, eess-IV, eess.IV, physics-med-ph  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.15386v1)  

---


**ABSTRACT**  
Purpose: To investigate the use of a Vision Transformer (ViT) to reconstruct/denoise GABA-edited magnetic resonance spectroscopy (MRS) from a quarter of the typically acquired number of transients using spectrograms.   Theory and Methods: A quarter of the typically acquired number of transients collected in GABA-edited MRS scans are pre-processed and converted to a spectrogram image representation using the Short-Time Fourier Transform (STFT). The image representation of the data allows the adaptation of a pre-trained ViT for reconstructing GABA-edited MRS spectra (Spectro-ViT). The Spectro-ViT is fine-tuned and then tested using \textit{in vivo} GABA-edited MRS data. The Spectro-ViT performance is compared against other models in the literature using spectral quality metrics and estimated metabolite concentration values.   Results: The Spectro-ViT model significantly outperformed all other models in four out of five quantitative metrics (mean squared error, shape score, GABA+/water fit error, and full width at half maximum). The metabolite concentrations estimated (GABA+/water, GABA+/Cr, and Glx/water) were consistent with the metabolite concentrations estimated using typical GABA-edited MRS scans reconstructed with the full amount of typically collected transients.   Conclusion: The proposed Spectro-ViT model achieved state-of-the-art results in reconstructing GABA-edited MRS, and the results indicate these scans could be up to four times faster.

{{</citation>}}


### (36/52) Eye Disease Prediction using Ensemble Learning and Attention on OCT Scans (Gauri Naik et al., 2023)

{{<citation>}}

Gauri Naik, Nandini Narvekar, Dimple Agarwal, Nishita Nandanwar, Himangi Pande. (2023)  
**Eye Disease Prediction using Ensemble Learning and Attention on OCT Scans**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2311.15301v1)  

---


**ABSTRACT**  
Eye diseases have posed significant challenges for decades, but advancements in technology have opened new avenues for their detection and treatment. Machine learning and deep learning algorithms have become instrumental in this domain, particularly when combined with Optical Coherent Technology (OCT) imaging. We propose a novel method for efficient detection of eye diseases from OCT images. Our technique enables the classification of patients into disease free (normal eyes) or affected by specific conditions such as Choroidal Neovascularization (CNV), Diabetic Macular Edema (DME), or Drusen. In this work, we introduce an end to end web application that utilizes machine learning and deep learning techniques for efficient eye disease prediction. The application allows patients to submit their raw OCT scanned images, which undergo segmentation using a trained custom UNet model. The segmented images are then fed into an ensemble model, comprising InceptionV3 and Xception networks, enhanced with a self attention layer. This self attention approach leverages the feature maps of individual models to achieve improved classification accuracy. The ensemble model's output is aggregated to predict and classify various eye diseases. Extensive experimentation and optimization have been conducted to ensure the application's efficiency and optimal performance. Our results demonstrate the effectiveness of the proposed approach in accurate eye disease prediction. The developed web application holds significant potential for early detection and timely intervention, thereby contributing to improved eye healthcare outcomes.

{{</citation>}}


## cs.IR (2)



### (37/52) Data Augmentation for Sample Efficient and Robust Document Ranking (Abhijit Anand et al., 2023)

{{<citation>}}

Abhijit Anand, Jurek Leonhardt, Jaspreet Singh, Koustav Rudra, Avishek Anand. (2023)  
**Data Augmentation for Sample Efficient and Robust Document Ranking**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2311.15426v1)  

---


**ABSTRACT**  
Contextual ranking models have delivered impressive performance improvements over classical models in the document ranking task. However, these highly over-parameterized models tend to be data-hungry and require large amounts of data even for fine-tuning. In this paper, we propose data-augmentation methods for effective and robust ranking performance. One of the key benefits of using data augmentation is in achieving sample efficiency or learning effectively when we have only a small amount of training data. We propose supervised and unsupervised data augmentation schemes by creating training data using parts of the relevant documents in the query-document pairs. We then adapt a family of contrastive losses for the document ranking task that can exploit the augmented data to learn an effective ranking model. Our extensive experiments on subsets of the MS MARCO and TREC-DL test sets show that data augmentation, along with the ranking-adapted contrastive losses, results in performance improvements under most dataset sizes. Apart from sample efficiency, we conclusively show that data augmentation results in robust models when transferred to out-of-domain benchmarks. Our performance improvements in in-domain and more prominently in out-of-domain benchmarks show that augmentation regularizes the ranking model and improves its robustness and generalization capability.

{{</citation>}}


### (38/52) Query-LIFE: Query-aware Language Image Fusion Embedding for E-Commerce Relevance (Hai Zhu et al., 2023)

{{<citation>}}

Hai Zhu, Yuankai Guo, Ronggang Dou, Kai Liu. (2023)  
**Query-LIFE: Query-aware Language Image Fusion Embedding for E-Commerce Relevance**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2311.14742v1)  

---


**ABSTRACT**  
Relevance module plays a fundamental role in e-commerce search as they are responsible for selecting relevant products from thousands of items based on user queries, thereby enhancing users experience and efficiency. The traditional approach models the relevance based product titles and queries, but the information in titles alone maybe insufficient to describe the products completely. A more general optimization approach is to further leverage product image information. In recent years, vision-language pre-training models have achieved impressive results in many scenarios, which leverage contrastive learning to map both textual and visual features into a joint embedding space. In e-commerce, a common practice is to fine-tune on the pre-trained model based on e-commerce data. However, the performance is sub-optimal because the vision-language pre-training models lack of alignment specifically designed for queries. In this paper, we propose a method called Query-LIFE (Query-aware Language Image Fusion Embedding) to address these challenges. Query-LIFE utilizes a query-based multimodal fusion to effectively incorporate the image and title based on the product types. Additionally, it employs query-aware modal alignment to enhance the accuracy of the comprehensive representation of products. Furthermore, we design GenFilt, which utilizes the generation capability of large models to filter out false negative samples and further improve the overall performance of the contrastive learning task in the model. Experiments have demonstrated that Query-LIFE outperforms existing baselines. We have conducted ablation studies and human evaluations to validate the effectiveness of each module within Query-LIFE. Moreover, Query-LIFE has been deployed on Miravia Search, resulting in improved both relevance and conversion efficiency.

{{</citation>}}


## cs.CY (2)



### (39/52) Increased Compute Efficiency and the Diffusion of AI Capabilities (Konstantin Pilz et al., 2023)

{{<citation>}}

Konstantin Pilz, Lennart Heim, Nicholas Brown. (2023)  
**Increased Compute Efficiency and the Diffusion of AI Capabilities**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.15377v1)  

---


**ABSTRACT**  
Training advanced AI models requires large investments in computational resources, or compute. Yet, as hardware innovation reduces the price of compute and algorithmic advances make its use more efficient, the cost of training an AI model to a given performance falls over time. To analyze this phenomenon, we introduce compute (investment) efficiency, which relates training compute investment to the resulting AI model performance. We then present a conceptual model of increases in compute efficiency and assess the social and governance implications. We find that while an access effect increases the number of actors who can train models to a given performance over time, a performance effect simultaneously increases the performance available to every actor - potentially enabling large compute investors to pioneer new capabilities and maintain a performance advantage even as capabilities diffuse. The market effects are multifaceted: while a relative performance advantage might grant outsized benefits in zero-sum competition, performance ceilings might reduce leaders' advantage. Nonetheless, we find that if the most severe risks arise from the most advanced models, large compute investors warrant particular scrutiny since they discover potentially dangerous capabilities first. Consequently, governments should require large compute investors to warn them about dangerous capabilities, thereby enabling timely preparation and potentially using their superior model performance and compute access for defensive measures. In cases of extreme risks, especially offense-dominant capabilities, the government might need to actively restrict the proliferation entirely.

{{</citation>}}


### (40/52) ChatGPT and Beyond: The Generative AI Revolution in Education (Mohammad AL-Smadi, 2023)

{{<citation>}}

Mohammad AL-Smadi. (2023)  
**ChatGPT and Beyond: The Generative AI Revolution in Education**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CL, cs-CY, cs.CY  
Keywords: AI, ChatGPT, GPT, Generative AI  
[Paper Link](http://arxiv.org/abs/2311.15198v1)  

---


**ABSTRACT**  
The wide adoption and usage of generative artificial intelligence (AI) models, particularly ChatGPT, has sparked a surge in research exploring their potential applications in the educational landscape. This survey examines academic literature published between November, 2022, and July, 2023, specifically targeting high-impact research from Scopus-indexed Q1 and Q2 journals. This survey delves into the practical applications and implications of generative AI models across a diverse range of educational contexts. Through a comprehensive and rigorous evaluation of recent academic literature, this survey seeks to illuminate the evolving role of generative AI models, particularly ChatGPT, in education. By shedding light on the potential benefits, challenges, and emerging trends in this dynamic field, the survey endeavors to contribute to the understanding of the nexus between artificial intelligence and education. The findings of this review will empower educators, researchers, and policymakers to make informed decisions about the integration of AI technologies into learning environments.

{{</citation>}}


## cs.CR (3)



### (41/52) Untargeted Code Authorship Evasion with Seq2Seq Transformation (Soohyeon Choi et al., 2023)

{{<citation>}}

Soohyeon Choi, Rhongho Jang, DaeHun Nyang, David Mohaisen. (2023)  
**Untargeted Code Authorship Evasion with Seq2Seq Transformation**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs.CR  
Keywords: Seq2Seq  
[Paper Link](http://arxiv.org/abs/2311.15366v1)  

---


**ABSTRACT**  
Code authorship attribution is the problem of identifying authors of programming language codes through the stylistic features in their codes, a topic that recently witnessed significant interest with outstanding performance. In this work, we present SCAE, a code authorship obfuscation technique that leverages a Seq2Seq code transformer called StructCoder. SCAE customizes StructCoder, a system designed initially for function-level code translation from one language to another (e.g., Java to C#), using transfer learning. SCAE improved the efficiency at a slight accuracy degradation compared to existing work. We also reduced the processing time by about 68% while maintaining an 85% transformation success rate and up to 95.77% evasion success rate in the untargeted setting.

{{</citation>}}


### (42/52) The Infrastructure Utilization of Free Contents Websites Reveal their Security Characteristics (Mohamed Alqadhi et al., 2023)

{{<citation>}}

Mohamed Alqadhi, David Mohaisen. (2023)  
**The Infrastructure Utilization of Free Contents Websites Reveal their Security Characteristics**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2311.15363v1)  

---


**ABSTRACT**  
Free Content Websites (FCWs) are a significant element of the Web, and realizing their use is essential. This study analyzes FCWs worldwide by studying how they correlate with different network sizes, cloud service providers, and countries, depending on the type of content they offer. Additionally, we compare these findings with those of premium content websites (PCWs). Our analysis concluded that FCWs correlate mainly with networks of medium size, which are associated with a higher concentration of malicious websites. Moreover, we found a strong correlation between PCWs, cloud, and country hosting patterns. At the same time, some correlations were also observed concerning FCWs but with distinct patterns contrasting each other for both types. Our investigation contributes to comprehending the FCW ecosystem through correlation analysis, and the indicative results point toward controlling the potential risks caused by these sites through adequate segregation and filtering due to their concentration.

{{</citation>}}


### (43/52) Understanding the Utilization of Cryptocurrency in the Metaverse and Security Implications (Ayodeji Adeniran et al., 2023)

{{<citation>}}

Ayodeji Adeniran, Mohammed Alkinoon, David Mohaisen. (2023)  
**Understanding the Utilization of Cryptocurrency in the Metaverse and Security Implications**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2311.15360v1)  

---


**ABSTRACT**  
We present our results on analyzing and understanding the behavior and security of various metaverse platforms incorporating cryptocurrencies. We obtained the top metaverse coins with a capitalization of at least 25 million US dollars and the top metaverse domains for the coins, and augmented our data with name registration information (via whois), including the hosting DNS IP addresses, registrant location, registrar URL, DNS service provider, expiry date and check each metaverse website for information on fiat currency for cryptocurrency. The result from virustotal.com includes the communication files, passive DNS, referrer files, and malicious detections for each metaverse domain. Among other insights, we discovered various incidents of malicious detection associated with metaverse websites. Our analysis highlights indicators of (in)security, in the correlation sense, with the files and other attributes that are potentially responsible for the malicious activities.

{{</citation>}}


## cs.RO (2)



### (44/52) Ultra-Range Gesture Recognition using an RGB Camera in Human-Robot Interaction (Eran Bamani et al., 2023)

{{<citation>}}

Eran Bamani, Eden Nissinman, Inbar Meir, Lisa Koenigsberg, Avishai Sintov. (2023)  
**Ultra-Range Gesture Recognition using an RGB Camera in Human-Robot Interaction**  

---
Primary Category: cs.RO  
Categories: cs-CV, cs-RO, cs.RO  
Keywords: Graph Convolutional Network, Transformer  
[Paper Link](http://arxiv.org/abs/2311.15361v1)  

---


**ABSTRACT**  
Hand gestures play a significant role in human interactions where non-verbal intentions, thoughts and commands are conveyed. In Human-Robot Interaction (HRI), hand gestures offer a similar and efficient medium for conveying clear and rapid directives to a robotic agent. However, state-of-the-art vision-based methods for gesture recognition have been shown to be effective only up to a user-camera distance of seven meters. Such a short distance range limits practical HRI with, for example, service robots, search and rescue robots and drones. In this work, we address the Ultra-Range Gesture Recognition (URGR) problem by aiming for a recognition distance of up to 25 meters and in the context of HRI. We propose a novel deep-learning framework for URGR using solely a simple RGB camera. First, a novel super-resolution model termed HQ-Net is used to enhance the low-resolution image of the user. Then, we propose a novel URGR classifier termed Graph Vision Transformer (GViT) which takes the enhanced image as input. GViT combines the benefits of a Graph Convolutional Network (GCN) and a modified Vision Transformer (ViT). Evaluation of the proposed framework over diverse test data yields a high recognition rate of 98.1%. The framework has also exhibited superior performance compared to human recognition in ultra-range distances. With the framework, we analyze and demonstrate the performance of an autonomous quadruped robot directed by human gestures in complex ultra-range indoor and outdoor environments.

{{</citation>}}


### (45/52) FRAC-Q-Learning: A Reinforcement Learning with Boredom Avoidance Processes for Social Robots (Akinari Onishi, 2023)

{{<citation>}}

Akinari Onishi. (2023)  
**FRAC-Q-Learning: A Reinforcement Learning with Boredom Avoidance Processes for Social Robots**  

---
Primary Category: cs.RO  
Categories: cs-HC, cs-LG, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.15327v1)  

---


**ABSTRACT**  
The reinforcement learning algorithms have often been applied to social robots. However, most reinforcement learning algorithms were not optimized for the use of social robots, and consequently they may bore users. We proposed a new reinforcement learning method specialized for the social robot, the FRAC-Q-learning, that can avoid user boredom. The proposed algorithm consists of a forgetting process in addition to randomizing and categorizing processes. This study evaluated interest and boredom hardness scores of the FRAC-Q-learning by a comparison with the traditional Q-learning. The FRAC-Q-learning showed significantly higher trend of interest score, and indicated significantly harder to bore users compared to the traditional Q-learning. Therefore, the FRAC-Q-learning can contribute to develop a social robot that will not bore users. The proposed algorithm can also find applications in Web-based communication and educational systems. This paper presents the entire process, detailed implementation and a detailed evaluation method of the of the FRAC-Q-learning for the first time.

{{</citation>}}


## cs.NE (1)



### (46/52) Algorithm Evolution Using Large Language Model (Fei Liu et al., 2023)

{{<citation>}}

Fei Liu, Xialiang Tong, Mingxuan Yuan, Qingfu Zhang. (2023)  
**Algorithm Evolution Using Large Language Model**  

---
Primary Category: cs.NE  
Categories: cs-AI, cs-LG, cs-NE, cs.NE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.15249v1)  

---


**ABSTRACT**  
Optimization can be found in many real-life applications. Designing an effective algorithm for a specific optimization problem typically requires a tedious amount of effort from human experts with domain knowledge and algorithm design skills. In this paper, we propose a novel approach called Algorithm Evolution using Large Language Model (AEL). It utilizes a large language model (LLM) to automatically generate optimization algorithms via an evolutionary framework. AEL does algorithm-level evolution without model training. Human effort and requirements for domain knowledge can be significantly reduced. We take constructive methods for the salesman traveling problem as a test example, we show that the constructive algorithm obtained by AEL outperforms simple hand-crafted and LLM-generated heuristics. Compared with other domain deep learning model-based algorithms, these methods exhibit excellent scalability across different problem sizes. AEL is also very different from previous attempts that utilize LLMs as search operators in algorithms.

{{</citation>}}


## eess.SY (1)



### (47/52) Solve Large-scale Unit Commitment Problems by Physics-informed Graph Learning (Jingtao Qin et al., 2023)

{{<citation>}}

Jingtao Qin, Nanpeng Yu. (2023)  
**Solve Large-scale Unit Commitment Problems by Physics-informed Graph Learning**  

---
Primary Category: eess.SY  
Categories: cs-LG, cs-SY, eess-SY, eess.SY  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2311.15216v1)  

---


**ABSTRACT**  
Unit commitment (UC) problems are typically formulated as mixed-integer programs (MIP) and solved by the branch-and-bound (B&B) scheme. The recent advances in graph neural networks (GNN) enable it to enhance the B&B algorithm in modern MIP solvers by learning to dive and branch. Existing GNN models that tackle MIP problems are mostly constructed from mathematical formulation, which is computationally expensive when dealing with large-scale UC problems. In this paper, we propose a physics-informed hierarchical graph convolutional network (PI-GCN) for neural diving that leverages the underlying features of various components of power systems to find high-quality variable assignments. Furthermore, we adopt the MIP model-based graph convolutional network (MB-GCN) for neural branching to select the optimal variables for branching at each node of the B&B tree. Finally, we integrate neural diving and neural branching into a modern MIP solver to establish a novel neural MIP solver designed for large-scale UC problems. Numeral studies show that PI-GCN has better performance and scalability than the baseline MB-GCN on neural diving. Moreover, the neural MIP solver yields the lowest operational cost and outperforms a modern MIP solver for all testing days after combining it with our proposed neural diving model and the baseline neural branching model.

{{</citation>}}


## cs.SE (1)



### (48/52) OpenPerf: A Benchmarking Framework for the Sustainable Development of the Open-Source Ecosystem (Fenglin Bi et al., 2023)

{{<citation>}}

Fenglin Bi, Fanyu Han, Shengyu Zhao, Jinlu Li, Yanbin Zhang, Wei Wang. (2023)  
**OpenPerf: A Benchmarking Framework for the Sustainable Development of the Open-Source Ecosystem**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: AI, ImageNet  
[Paper Link](http://arxiv.org/abs/2311.15212v1)  

---


**ABSTRACT**  
Benchmarking involves designing scientific test methods, tools, and frameworks to quantitatively and comparably assess specific performance indicators of certain test subjects. With the development of artificial intelligence, AI benchmarking datasets such as ImageNet and DataPerf have gradually become consensus standards in both academic and industrial fields. However, constructing a benchmarking framework remains a significant challenge in the open-source domain due to the diverse range of data types, the wide array of research issues, and the intricate nature of collaboration networks. This paper introduces OpenPerf, a benchmarking framework designed for the sustainable development of the open-source ecosystem. This framework defines 9 task benchmarking tasks in the open-source research, encompassing 3 data types: time series, text, and graphics, and addresses 6 research problems including regression, classification, recommendation, ranking, network building, and anomaly detection. Based on the above tasks, we implemented 3 data science task benchmarks, 2 index-based benchmarks, and 1 standard benchmark. Notably, the index-based benchmarks have been adopted by the China Electronics Standardization Institute as evaluation criteria for open-source community governance. Additionally, we have developed a comprehensive toolkit for OpenPerf, which not only offers robust data management, tool integration, and user interface capabilities but also adopts a Benchmarking-as-a-Service (BaaS) model to serve academic institutions, industries, and foundations. Through its application in renowned companies and institutions such as Alibaba, Ant Group, and East China Normal University, we have validated OpenPerf's pivotal role in the healthy evolution of the open-source ecosystem.

{{</citation>}}


## cs.LO (1)



### (49/52) Using Rely/Guarantee to Pinpoint Assumptions underlying Security Protocols (Nisansala P. Yatapanage et al., 2023)

{{<citation>}}

Nisansala P. Yatapanage, Cliff B. Jones. (2023)  
**Using Rely/Guarantee to Pinpoint Assumptions underlying Security Protocols**  

---
Primary Category: cs.LO  
Categories: cs-LO, cs.LO  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2311.15189v1)  

---


**ABSTRACT**  
The verification of security protocols is essential, in order to ensure the absence of potential attacks. However, verification results are only valid with respect to the assumptions under which the verification was performed. These assumptions are often hidden and are difficult to identify, making it unclear whether a given protocol is safe to deploy into a particular environment. Rely/guarantee provides a mechanism for abstractly reasoning about the interference from the environment. Using this approach, the assumptions are made clear and precise. This paper investigates this approach on the Needham-Schroeder Public Key protocol, showing that the technique can effectively uncover the assumptions under which the protocol can withstand attacks from intruders.

{{</citation>}}


## q-fin.TR (1)



### (50/52) Benchmarking Large Language Model Volatility (Boyang Yu, 2023)

{{<citation>}}

Boyang Yu. (2023)  
**Benchmarking Large Language Model Volatility**  

---
Primary Category: q-fin.TR  
Categories: cs-CL, q-fin-TR, q-fin.TR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.15180v1)  

---


**ABSTRACT**  
The impact of non-deterministic outputs from Large Language Models (LLMs) is not well examined for financial text understanding tasks. Through a compelling case study on investing in the US equity market via news sentiment analysis, we uncover substantial variability in sentence-level sentiment classification results, underscoring the innate volatility of LLM outputs. These uncertainties cascade downstream, leading to more significant variations in portfolio construction and return. While tweaking the temperature parameter in the language model decoder presents a potential remedy, it comes at the expense of stifled creativity. Similarly, while ensembling multiple outputs mitigates the effect of volatile outputs, it demands a notable computational investment. This work furnishes practitioners with invaluable insights for adeptly navigating uncertainty in the integration of LLMs into financial decision-making, particularly in scenarios dictated by non-deterministic information.

{{</citation>}}


## math.OC (1)



### (51/52) Optimizing Multi-Timestep Security-Constrained Optimal Power Flow for Large Power Grids (Hussein Sharadga et al., 2023)

{{<citation>}}

Hussein Sharadga, Javad Mohammadi, Constance Crozier, Kyri Baker. (2023)  
**Optimizing Multi-Timestep Security-Constrained Optimal Power Flow for Large Power Grids**  

---
Primary Category: math.OC  
Categories: cs-SY, eess-SY, math-OC, math.OC  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2311.15175v1)  

---


**ABSTRACT**  
This work proposes a novel method for scaling multi-timestep security-constrained optimal power flow in large power grids. The challenge arises from dealing with millions of variables and constraints, including binary variables and nonconvex, nonlinear characteristics. To navigate these complexities, techniques such as constraint relaxation, linearization, sequential optimization, and problem reformulation are employed. By leveraging these methods, complex power grid problems are solved while achieving high-quality solutions and meeting time constraints. The innovative solution approach showcases great robustness and consistently outperforms benchmark standards.

{{</citation>}}


## math.NA (1)



### (52/52) Reduced Augmentation Implicit Low-rank (RAIL) integrators for advection-diffusion and Fokker-Planck models (Joseph Nakao et al., 2023)

{{<citation>}}

Joseph Nakao, Jing-Mei Qiu, Lukas Einkemmer. (2023)  
**Reduced Augmentation Implicit Low-rank (RAIL) integrators for advection-diffusion and Fokker-Planck models**  

---
Primary Category: math.NA  
Categories: 65, cs-NA, math-NA, math.NA  
Keywords: AI, Augmentation  
[Paper Link](http://arxiv.org/abs/2311.15143v1)  

---


**ABSTRACT**  
This paper introduces a novel computational approach termed the Reduced Augmentation Implicit Low-rank (RAIL) method by investigating two predominant research directions in low-rank solutions to time-dependent partial differential equations (PDEs): dynamical low-rank (DLR), and step and truncation (SAT) tensor methods. The RAIL method, along with the development of the SAT approach, is designed to enhance the efficiency of traditional full-rank implicit solvers from method-of-lines discretizations of time-dependent PDEs, while maintaining accuracy and stability. We consider spectral methods for spatial discretization, and diagonally implicit Runge-Kutta (DIRK) and implicit-explicit (IMEX) RK methods for time discretization. The efficiency gain is achieved by investigating low-rank structures within solutions at each RK stage using a singular value decomposition (SVD). In particular, we develop a reduced augmentation procedure to predict the basis functions to construct projection subspaces. This procedure balances algorithm accuracy and efficiency by incorporating as many bases as possible from previous RK stages and predictions, and by optimizing the basis representation through SVD truncation. As such, one can form implicit schemes for updating basis functions in a dimension-by-dimension manner, similar in spirit to the K-L step in the DLR framework. We also apply a globally mass conservative post-processing step at the end of each RK stage. We validate the RAIL method through numerical simulations of advection-diffusion problems and a Fokker-Planck model, showcasing its ability to efficiently handle time-dependent PDEs while maintaining global mass conservation. Our approach generalizes and bridges the DLR and SAT approaches, offering a comprehensive framework for efficiently and accurately solving time-dependent PDEs with implicit treatment.

{{</citation>}}
