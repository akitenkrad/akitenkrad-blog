---
draft: false
title: "arXiv @ 2024.01.13"
date: 2024-01-13
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2024"]
menu:
  sidebar:
    name: "arXiv @ 2024.01.13"
    identifier: arxiv_20240113
    parent: 202401_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CV (17)](#cscv-17)
- [cs.LG (11)](#cslg-11)
- [cs.CL (39)](#cscl-39)
- [cs.AI (4)](#csai-4)
- [cs.SI (1)](#cssi-1)
- [cs.HC (6)](#cshc-6)
- [cs.MM (1)](#csmm-1)
- [stat.ML (1)](#statml-1)
- [cs.SE (5)](#csse-5)
- [cs.NE (1)](#csne-1)
- [math.DS (1)](#mathds-1)
- [cs.SD (2)](#cssd-2)
- [eess.SP (1)](#eesssp-1)
- [cs.IR (3)](#csir-3)
- [cs.CR (1)](#cscr-1)

## cs.CV (17)



### (1/94) Distilling Vision-Language Models on Millions of Videos (Yue Zhao et al., 2024)

{{<citation>}}

Yue Zhao, Long Zhao, Xingyi Zhou, Jialin Wu, Chun-Te Chu, Hui Miao, Florian Schroff, Hartwig Adam, Ting Liu, Boqing Gong, Philipp Krähenbühl, Liangzhe Yuan. (2024)  
**Distilling Vision-Language Models on Millions of Videos**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model, QA  
[Paper Link](http://arxiv.org/abs/2401.06129v1)  

---


**ABSTRACT**  
The recent advance in vision-language models is largely attributed to the abundance of image-text data. We aim to replicate this success for video-language models, but there simply is not enough human-curated video-text data available. We thus resort to fine-tuning a video-language model from a strong image-language baseline with synthesized instructional data. The resulting video-language model is then used to auto-label millions of videos to generate high-quality captions. We show the adapted video-language model performs well on a wide range of video-language benchmarks. For instance, it surpasses the best prior result on open-ended NExT-QA by 2.8%. Besides, our model generates detailed descriptions for previously unseen videos, which provide better textual supervision than existing methods. Experiments show that a video-language dual-encoder model contrastively trained on these auto-generated captions is 3.8% better than the strongest baseline that also leverages vision-language models. Our best model outperforms state-of-the-art methods on MSR-VTT zero-shot text-to-video retrieval by 6%.

{{</citation>}}


### (2/94) Surgical-DINO: Adapter Learning of Foundation Model for Depth Estimation in Endoscopic Surgery (Cui Beilei et al., 2024)

{{<citation>}}

Cui Beilei, Islam Mobarakol, Bai Long, Ren Hongliang. (2024)  
**Surgical-DINO: Adapter Learning of Foundation Model for Depth Estimation in Endoscopic Surgery**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.06013v1)  

---


**ABSTRACT**  
Purpose: Depth estimation in robotic surgery is vital in 3D reconstruction, surgical navigation and augmented reality visualization. Although the foundation model exhibits outstanding performance in many vision tasks, including depth estimation (e.g., DINOv2), recent works observed its limitations in medical and surgical domain-specific applications. This work presents a low-ranked adaptation (LoRA) of the foundation model for surgical depth estimation. Methods: We design a foundation model-based depth estimation method, referred to as Surgical-DINO, a low-rank adaptation of the DINOv2 for depth estimation in endoscopic surgery. We build LoRA layers and integrate them into DINO to adapt with surgery-specific domain knowledge instead of conventional fine-tuning. During training, we freeze the DINO image encoder, which shows excellent visual representation capacity, and only optimize the LoRA layers and depth decoder to integrate features from the surgical scene. Results: Our model is extensively validated on a MICCAI challenge dataset of SCARED, which is collected from da Vinci Xi endoscope surgery. We empirically show that Surgical-DINO significantly outperforms all the state-of-the-art models in endoscopic depth estimation tasks. The analysis with ablation studies has shown evidence of the remarkable effect of our LoRA layers and adaptation. Conclusion: Surgical-DINO shed some light on the successful adaptation of the foundation models into the surgical domain for depth estimation. There is clear evidence in the results that zero-shot prediction on pre-trained weights in computer vision datasets or naive fine-tuning is not sufficient to use the foundation model in the surgical domain directly. Code is available at https://github.com/BeileiCui/SurgicalDINO.

{{</citation>}}


### (3/94) Attention to detail: inter-resolution knowledge distillation (Rocío del Amor et al., 2024)

{{<citation>}}

Rocío del Amor, Julio Silva-Rodríguez, Adrián Colomer, Valery Naranjo. (2024)  
**Attention to detail: inter-resolution knowledge distillation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2401.06010v1)  

---


**ABSTRACT**  
The development of computer vision solutions for gigapixel images in digital pathology is hampered by significant computational limitations due to the large size of whole slide images. In particular, digitizing biopsies at high resolutions is a time-consuming process, which is necessary due to the worsening results from the decrease in image detail. To alleviate this issue, recent literature has proposed using knowledge distillation to enhance the model performance at reduced image resolutions. In particular, soft labels and features extracted at the highest magnification level are distilled into a model that takes lower-magnification images as input. However, this approach fails to transfer knowledge about the most discriminative image regions in the classification process, which may be lost when the resolution is decreased. In this work, we propose to distill this information by incorporating attention maps during training. In particular, our formulation leverages saliency maps of the target class via grad-CAMs, which guides the lower-resolution Student model to match the Teacher distribution by minimizing the l2 distance between them. Comprehensive experiments on prostate histology image grading demonstrate that the proposed approach substantially improves the model performance across different image resolutions compared to previous literature.

{{</citation>}}


### (4/94) A Lightweight Feature Fusion Architecture For Resource-Constrained Crowd Counting (Yashwardhan Chaudhuri et al., 2024)

{{<citation>}}

Yashwardhan Chaudhuri, Ankit Kumar, Orchid Chetia Phukan, Arun Balaji Buduru. (2024)  
**A Lightweight Feature Fusion Architecture For Resource-Constrained Crowd Counting**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Pre-Trained Model  
[Paper Link](http://arxiv.org/abs/2401.05968v1)  

---


**ABSTRACT**  
Crowd counting finds direct applications in real-world situations, making computational efficiency and performance crucial. However, most of the previous methods rely on a heavy backbone and a complex downstream architecture that restricts the deployment. To address this challenge and enhance the versatility of crowd-counting models, we introduce two lightweight models. These models maintain the same downstream architecture while incorporating two distinct backbones: MobileNet and MobileViT. We leverage Adjacent Feature Fusion to extract diverse scale features from a Pre-Trained Model (PTM) and subsequently combine these features seamlessly. This approach empowers our models to achieve improved performance while maintaining a compact and efficient design. With the comparison of our proposed models with previously available state-of-the-art (SOTA) methods on ShanghaiTech-A ShanghaiTech-B and UCF-CC-50 dataset, it achieves comparable results while being the most computationally efficient model. Finally, we present a comparative study, an extensive ablation study, along with pruning to show the effectiveness of our models.

{{</citation>}}


### (5/94) Efficient Image Deblurring Networks based on Diffusion Models (Kang Chen et al., 2024)

{{<citation>}}

Kang Chen, Yuanjie Liu. (2024)  
**Efficient Image Deblurring Networks based on Diffusion Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.05907v1)  

---


**ABSTRACT**  
This article introduces a sliding window model for defocus deblurring that achieves the best performance to date with extremely low memory usage. Named Swintormer, the method utilizes a diffusion model to generate latent prior features that assist in restoring more detailed images. It also extends the sliding window strategy to specialized Transformer blocks for efficient inference. Additionally, we have further optimized Multiply-Accumulate operations (Macs). Compared to the currently top-performing GRL method, our Swintormer model drastically reduces computational complexity from 140.35 GMACs to 8.02 GMacs, while also improving the Signal-to-Noise Ratio (SNR) for defocus deblurring from 27.04 dB to 27.07 dB. This new method allows for the processing of higher resolution images on devices with limited memory, significantly expanding potential application scenarios. The article concludes with an ablation study that provides an in-depth analysis of the impact of each network module on final performance. The source code and model will be available at the following website: https://github.com/bnm6900030/swintormer.

{{</citation>}}


### (6/94) HiCAST: Highly Customized Arbitrary Style Transfer with Adapter Enhanced Diffusion Models (Hanzhang Wang et al., 2024)

{{<citation>}}

Hanzhang Wang, Haoran Wang, Jinze Yang, Zhongrui Yu, Zeke Xie, Lei Tian, Xinyan Xiao, Junjun Jiang, Xianming Liu, Mingming Sun. (2024)  
**HiCAST: Highly Customized Arbitrary Style Transfer with Adapter Enhanced Diffusion Models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Style Transfer  
[Paper Link](http://arxiv.org/abs/2401.05870v1)  

---


**ABSTRACT**  
The goal of Arbitrary Style Transfer (AST) is injecting the artistic features of a style reference into a given image/video. Existing methods usually focus on pursuing the balance between style and content, whereas ignoring the significant demand for flexible and customized stylization results and thereby limiting their practical application. To address this critical issue, a novel AST approach namely HiCAST is proposed, which is capable of explicitly customizing the stylization results according to various source of semantic clues. In the specific, our model is constructed based on Latent Diffusion Model (LDM) and elaborately designed to absorb content and style instance as conditions of LDM. It is characterized by introducing of \textit{Style Adapter}, which allows user to flexibly manipulate the output results by aligning multi-level style information and intrinsic knowledge in LDM. Lastly, we further extend our model to perform video AST. A novel learning objective is leveraged for video diffusion model training, which significantly improve cross-frame temporal consistency in the premise of maintaining stylization strength. Qualitative and quantitative comparisons as well as comprehensive user studies demonstrate that our HiCAST outperforms the existing SoTA methods in generating visually plausible stylization results.

{{</citation>}}


### (7/94) CLIP-Driven Semantic Discovery Network for Visible-Infrared Person Re-Identification (Xiaoyan Yu et al., 2024)

{{<citation>}}

Xiaoyan Yu, Neng Dong, Liehuang Zhu, Hao Peng, Dapeng Tao. (2024)  
**CLIP-Driven Semantic Discovery Network for Visible-Infrared Person Re-Identification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2401.05806v1)  

---


**ABSTRACT**  
Visible-infrared person re-identification (VIReID) primarily deals with matching identities across person images from different modalities. Due to the modality gap between visible and infrared images, cross-modality identity matching poses significant challenges. Recognizing that high-level semantics of pedestrian appearance, such as gender, shape, and clothing style, remain consistent across modalities, this paper intends to bridge the modality gap by infusing visual features with high-level semantics. Given the capability of CLIP to sense high-level semantic information corresponding to visual representations, we explore the application of CLIP within the domain of VIReID. Consequently, we propose a CLIP-Driven Semantic Discovery Network (CSDN) that consists of Modality-specific Prompt Learner, Semantic Information Integration (SII), and High-level Semantic Embedding (HSE). Specifically, considering the diversity stemming from modality discrepancies in language descriptions, we devise bimodal learnable text tokens to capture modality-private semantic information for visible and infrared images, respectively. Additionally, acknowledging the complementary nature of semantic details across different modalities, we integrate text features from the bimodal language descriptions to achieve comprehensive semantics. Finally, we establish a connection between the integrated text features and the visual features across modalities. This process embed rich high-level semantic information into visual representations, thereby promoting the modality invariance of visual representations. The effectiveness and superiority of our proposed CSDN over existing methods have been substantiated through experimental evaluations on multiple widely used benchmarks. The code will be released at \url{https://github.com/nengdong96/CSDN}.

{{</citation>}}


### (8/94) Learn From Zoom: Decoupled Supervised Contrastive Learning For WCE Image Classification (Kunpeng Qiu et al., 2024)

{{<citation>}}

Kunpeng Qiu, Zhiying Zhou, Yongxin Guo. (2024)  
**Learn From Zoom: Decoupled Supervised Contrastive Learning For WCE Image Classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning, Image Classification  
[Paper Link](http://arxiv.org/abs/2401.05771v1)  

---


**ABSTRACT**  
Accurate lesion classification in Wireless Capsule Endoscopy (WCE) images is vital for early diagnosis and treatment of gastrointestinal (GI) cancers. However, this task is confronted with challenges like tiny lesions and background interference. Additionally, WCE images exhibit higher intra-class variance and inter-class similarities, adding complexity. To tackle these challenges, we propose Decoupled Supervised Contrastive Learning for WCE image classification, learning robust representations from zoomed-in WCE images generated by Saliency Augmentor. Specifically, We use uniformly down-sampled WCE images as anchors and WCE images from the same class, especially their zoomed-in images, as positives. This approach empowers the Feature Extractor to capture rich representations from various views of the same image, facilitated by Decoupled Supervised Contrastive Learning. Training a linear Classifier on these representations within 10 epochs yields an impressive 92.01% overall accuracy, surpassing the prior state-of-the-art (SOTA) by 0.72% on a blend of two publicly accessible WCE datasets. Code is available at: https://github.com/Qiukunpeng/DSCL.

{{</citation>}}


### (9/94) Evaluating Data Augmentation Techniques for Coffee Leaf Disease Classification (Adrian Gheorghiu et al., 2024)

{{<citation>}}

Adrian Gheorghiu, Iulian-Marius Tăiatu, Dumitru-Clementin Cercel, Iuliana Marin, Florin Pop. (2024)  
**Evaluating Data Augmentation Techniques for Coffee Leaf Disease Classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation, Transformer  
[Paper Link](http://arxiv.org/abs/2401.05768v1)  

---


**ABSTRACT**  
The detection and classification of diseases in Robusta coffee leaves are essential to ensure that plants are healthy and the crop yield is kept high. However, this job requires extensive botanical knowledge and much wasted time. Therefore, this task and others similar to it have been extensively researched subjects in image classification. Regarding leaf disease classification, most approaches have used the more popular PlantVillage dataset while completely disregarding other datasets, like the Robusta Coffee Leaf (RoCoLe) dataset. As the RoCoLe dataset is imbalanced and does not have many samples, fine-tuning of pre-trained models and multiple augmentation techniques need to be used. The current paper uses the RoCoLe dataset and approaches based on deep learning for classifying coffee leaf diseases from images, incorporating the pix2pix model for segmentation and cycle-generative adversarial network (CycleGAN) for augmentation. Our study demonstrates the effectiveness of Transformer-based models, online augmentations, and CycleGAN augmentation in improving leaf disease classification. While synthetic data has limitations, it complements real data, enhancing model performance. These findings contribute to developing robust techniques for plant disease detection and classification.

{{</citation>}}


### (10/94) Surface Normal Estimation with Transformers (Barry Shichen Hu et al., 2024)

{{<citation>}}

Barry Shichen Hu, Siyun Liang, Johannes Paetzold, Huy H. Nguyen, Isao Echizen, Jiapeng Tang. (2024)  
**Surface Normal Estimation with Transformers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.05745v1)  

---


**ABSTRACT**  
We propose the use of a Transformer to accurately predict normals from point clouds with noise and density variations. Previous learning-based methods utilize PointNet variants to explicitly extract multi-scale features at different input scales, then focus on a surface fitting method by which local point cloud neighborhoods are fitted to a geometric surface approximated by either a polynomial function or a multi-layer perceptron (MLP). However, fitting surfaces to fixed-order polynomial functions can suffer from overfitting or underfitting, and learning MLP-represented hyper-surfaces requires pre-generated per-point weights. To avoid these limitations, we first unify the design choices in previous works and then propose a simplified Transformer-based model to extract richer and more robust geometric features for the surface normal estimation task. Through extensive experiments, we demonstrate that our Transformer-based method achieves state-of-the-art performance on both the synthetic shape dataset PCPNet, and the real-world indoor scene dataset SceneNN, exhibiting more noise-resilient behavior and significantly faster inference. Most importantly, we demonstrate that the sophisticated hand-designed modules in existing works are not necessary to excel at the task of surface normal estimation.

{{</citation>}}


### (11/94) LKCA: Large Kernel Convolutional Attention (Chenghao Li et al., 2024)

{{<citation>}}

Chenghao Li, Boheng Zeng, Yi Lu, Pengbo Shi, Qingzi Chen, Jirui Liu, Lingyun Zhu. (2024)  
**LKCA: Large Kernel Convolutional Attention**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2401.05738v1)  

---


**ABSTRACT**  
We revisit the relationship between attention mechanisms and large kernel ConvNets in visual transformers and propose a new spatial attention named Large Kernel Convolutional Attention (LKCA). It simplifies the attention operation by replacing it with a single large kernel convolution. LKCA combines the advantages of convolutional neural networks and visual transformers, possessing a large receptive field, locality, and parameter sharing. We explained the superiority of LKCA from both convolution and attention perspectives, providing equivalent code implementations for each view. Experiments confirm that LKCA implemented from both the convolutional and attention perspectives exhibit equivalent performance. We extensively experimented with the LKCA variant of ViT in both classification and segmentation tasks. The experiments demonstrated that LKCA exhibits competitive performance in visual tasks. Our code will be made publicly available at https://github.com/CatworldLee/LKCA.

{{</citation>}}


### (12/94) Enhancing Contrastive Learning with Efficient Combinatorial Positive Pairing (Jaeill Kim et al., 2024)

{{<citation>}}

Jaeill Kim, Duhun Hwang, Eunjung Lee, Jangwon Suh, Jimyeong Kim, Wonjong Rhee. (2024)  
**Enhancing Contrastive Learning with Efficient Combinatorial Positive Pairing**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Contrastive Learning, ImageNet  
[Paper Link](http://arxiv.org/abs/2401.05730v1)  

---


**ABSTRACT**  
In the past few years, contrastive learning has played a central role for the success of visual unsupervised representation learning. Around the same time, high-performance non-contrastive learning methods have been developed as well. While most of the works utilize only two views, we carefully review the existing multi-view methods and propose a general multi-view strategy that can improve learning speed and performance of any contrastive or non-contrastive method. We first analyze CMC's full-graph paradigm and empirically show that the learning speed of $K$-views can be increased by $_{K}\mathrm{C}_{2}$ times for small learning rate and early training. Then, we upgrade CMC's full-graph by mixing views created by a crop-only augmentation, adopting small-size views as in SwAV multi-crop, and modifying the negative sampling. The resulting multi-view strategy is called ECPP (Efficient Combinatorial Positive Pairing). We investigate the effectiveness of ECPP by applying it to SimCLR and assessing the linear evaluation performance for CIFAR-10 and ImageNet-100. For each benchmark, we achieve a state-of-the-art performance. In case of ImageNet-100, ECPP boosted SimCLR outperforms supervised learning.

{{</citation>}}


### (13/94) Video Anomaly Detection and Explanation via Large Language Models (Hui Lv et al., 2024)

{{<citation>}}

Hui Lv, Qianru Sun. (2024)  
**Video Anomaly Detection and Explanation via Large Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Anomaly Detection, Language Model  
[Paper Link](http://arxiv.org/abs/2401.05702v1)  

---


**ABSTRACT**  
Video Anomaly Detection (VAD) aims to localize abnormal events on the timeline of long-range surveillance videos. Anomaly-scoring-based methods have been prevailing for years but suffer from the high complexity of thresholding and low explanability of detection results. In this paper, we conduct pioneer research on equipping video-based large language models (VLLMs) in the framework of VAD, making the VAD model free from thresholds and able to explain the reasons for the detected anomalies. We introduce a novel network module Long-Term Context (LTC) to mitigate the incapability of VLLMs in long-range context modeling. We design a three-phase training method to improve the efficiency of fine-tuning VLLMs by substantially minimizing the requirements for VAD data and lowering the costs of annotating instruction-tuning data. Our trained model achieves the top performance on the anomaly videos of the UCF-Crime and TAD benchmarks, with the AUC improvements of +3.86\% and +4.96\%, respectively. More impressively, our approach can provide textual explanations for detected anomalies.

{{</citation>}}


### (14/94) HiCMAE: Hierarchical Contrastive Masked Autoencoder for Self-Supervised Audio-Visual Emotion Recognition (Licai Sun et al., 2024)

{{<citation>}}

Licai Sun, Zheng Lian, Bin Liu, Jianhua Tao. (2024)  
**HiCMAE: Hierarchical Contrastive Masked Autoencoder for Self-Supervised Audio-Visual Emotion Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-HC, cs-MM, cs-SD, cs.CV, eess-AS  
Keywords: Emotion Recognition, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2401.05698v1)  

---


**ABSTRACT**  
Audio-Visual Emotion Recognition (AVER) has garnered increasing attention in recent years for its critical role in creating emotion-ware intelligent machines. Previous efforts in this area are dominated by the supervised learning paradigm. Despite significant progress, supervised learning is meeting its bottleneck due to the longstanding data scarcity issue in AVER. Motivated by recent advances in self-supervised learning, we propose Hierarchical Contrastive Masked Autoencoder (HiCMAE), a novel self-supervised framework that leverages large-scale self-supervised pre-training on vast unlabeled audio-visual data to promote the advancement of AVER. Following prior arts in self-supervised audio-visual representation learning, HiCMAE adopts two primary forms of self-supervision for pre-training, namely masked data modeling and contrastive learning. Unlike them which focus exclusively on top-layer representations while neglecting explicit guidance of intermediate layers, HiCMAE develops a three-pronged strategy to foster hierarchical audio-visual feature learning and improve the overall quality of learned representations. To verify the effectiveness of HiCMAE, we conduct extensive experiments on 9 datasets covering both categorical and dimensional AVER tasks. Experimental results show that our method significantly outperforms state-of-the-art supervised and self-supervised audio-visual methods, which indicates that HiCMAE is a powerful audio-visual emotion representation learner. Codes and models will be publicly available at https://github.com/sunlicai/HiCMAE.

{{</citation>}}


### (15/94) Parrot: Pareto-optimal Multi-Reward Reinforcement Learning Framework for Text-to-Image Generation (Seung Hyun Lee et al., 2024)

{{<citation>}}

Seung Hyun Lee, Yinxiao Li, Junjie Ke, Innfarn Yoo, Han Zhang, Jiahui Yu, Qifei Wang, Fei Deng, Glenn Entis, Junfeng He, Gang Li, Sangpil Kim, Irfan Essa, Feng Yang. (2024)  
**Parrot: Pareto-optimal Multi-Reward Reinforcement Learning Framework for Text-to-Image Generation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.05675v1)  

---


**ABSTRACT**  
Recent works demonstrate that using reinforcement learning (RL) with quality rewards can enhance the quality of generated images in text-to-image (T2I) generation. However, a simple aggregation of multiple rewards may cause over-optimization in certain metrics and degradation in others, and it is challenging to manually find the optimal weights. An effective strategy to jointly optimize multiple rewards in RL for T2I generation is highly desirable. This paper introduces Parrot, a novel multi-reward RL framework for T2I generation. Through the use of the batch-wise Pareto optimal selection, Parrot automatically identifies the optimal trade-off among different rewards during the RL optimization of the T2I generation. Additionally, Parrot employs a joint optimization approach for the T2I model and the prompt expansion network, facilitating the generation of quality-aware text prompts, thus further enhancing the final image quality. To counteract the potential catastrophic forgetting of the original user prompt due to prompt expansion, we introduce original prompt centered guidance at inference time, ensuring that the generated image remains faithful to the user input. Extensive experiments and a user study demonstrate that Parrot outperforms several baseline methods across various quality criteria, including aesthetics, human preference, image sentiment, and text-image alignment.

{{</citation>}}


### (16/94) Masked Attribute Description Embedding for Cloth-Changing Person Re-identification (Chunlei Peng et al., 2024)

{{<citation>}}

Chunlei Peng, Boyu Wang, Decheng Liu, Nannan Wang, Ruimin Hu, Xinbo Gao. (2024)  
**Masked Attribute Description Embedding for Cloth-Changing Person Re-identification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Embedding, Transformer  
[Paper Link](http://arxiv.org/abs/2401.05646v1)  

---


**ABSTRACT**  
Cloth-changing person re-identification (CC-ReID) aims to match persons who change clothes over long periods. The key challenge in CC-ReID is to extract clothing-independent features, such as face, hairstyle, body shape, and gait. Current research mainly focuses on modeling body shape using multi-modal biological features (such as silhouettes and sketches). However, it does not fully leverage the personal description information hidden in the original RGB image. Considering that there are certain attribute descriptions which remain unchanged after the changing of cloth, we propose a Masked Attribute Description Embedding (MADE) method that unifies personal visual appearance and attribute description for CC-ReID. Specifically, handling variable clothing-sensitive information, such as color and type, is challenging for effective modeling. To address this, we mask the clothing and color information in the personal attribute description extracted through an attribute detection model. The masked attribute description is then connected and embedded into Transformer blocks at various levels, fusing it with the low-level to high-level features of the image. This approach compels the model to discard clothing information. Experiments are conducted on several CC-ReID benchmarks, including PRCC, LTCC, Celeb-reID-light, and LaST. Results demonstrate that MADE effectively utilizes attribute description, enhancing cloth-changing person re-identification performance, and compares favorably with state-of-the-art methods. The code is available at https://github.com/moon-wh/MADE.

{{</citation>}}


### (17/94) Transforming Image Super-Resolution: A ConvFormer-based Efficient Approach (Gang Wu et al., 2024)

{{<citation>}}

Gang Wu, Junjun Jiang, Junpeng Jiang, Xianming Liu. (2024)  
**Transforming Image Super-Resolution: A ConvFormer-based Efficient Approach**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.05633v1)  

---


**ABSTRACT**  
Recent progress in single-image super-resolution (SISR) has achieved remarkable performance, yet the computational costs of these methods remain a challenge for deployment on resource-constrained devices. Especially for transformer-based methods, the self-attention mechanism in such models brings great breakthroughs while incurring substantial computational costs. To tackle this issue, we introduce the Convolutional Transformer layer (ConvFormer) and the ConvFormer-based Super-Resolution network (CFSR), which offer an effective and efficient solution for lightweight image super-resolution tasks. In detail, CFSR leverages the large kernel convolution as the feature mixer to replace the self-attention module, efficiently modeling long-range dependencies and extensive receptive fields with a slight computational cost. Furthermore, we propose an edge-preserving feed-forward network, simplified as EFN, to obtain local feature aggregation and simultaneously preserve more high-frequency information. Extensive experiments demonstrate that CFSR can achieve an advanced trade-off between computational cost and performance when compared to existing lightweight SR methods. Compared to state-of-the-art methods, e.g. ShuffleMixer, the proposed CFSR achieves 0.39 dB gains on Urban100 dataset for x2 SR task while containing 26% and 31% fewer parameters and FLOPs, respectively. Code and pre-trained models are available at https://github.com/Aitical/CFSR.

{{</citation>}}


## cs.LG (11)



### (18/94) Extreme Compression of Large Language Models via Additive Quantization (Vage Egiazarian et al., 2024)

{{<citation>}}

Vage Egiazarian, Andrei Panferov, Denis Kuznedelev, Elias Frantar, Artem Babenko, Dan Alistarh. (2024)  
**Extreme Compression of Large Language Models via Additive Quantization**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Language Model, Quantization  
[Paper Link](http://arxiv.org/abs/2401.06118v1)  

---


**ABSTRACT**  
The emergence of accurate open large language models (LLMs) has led to a race towards quantization techniques for such models enabling execution on end-user devices. In this paper, we revisit the problem of "extreme" LLM compression--defined as targeting extremely low bit counts, such as 2 to 3 bits per parameter, from the point of view of classic methods in Multi-Codebook Quantization (MCQ). Our work builds on top of Additive Quantization, a classic algorithm from the MCQ family, and adapts it to the quantization of language models. The resulting algorithm advances the state-of-the-art in LLM compression, outperforming all recently-proposed techniques in terms of accuracy at a given compression budget. For instance, when compressing Llama 2 models to 2 bits per parameter, our algorithm quantizes the 7B model to 6.93 perplexity (a 1.29 improvement relative to the best prior work, and 1.81 points from FP16), the 13B model to 5.70 perplexity (a .36 improvement) and the 70B model to 3.94 perplexity (a .22 improvement) on WikiText2. We release our implementation of Additive Quantization for Language Models AQLM as a baseline to facilitate future research in LLM quantization.

{{</citation>}}


### (19/94) Spatial-Aware Deep Reinforcement Learning for the Traveling Officer Problem (Niklas Strauß et al., 2024)

{{<citation>}}

Niklas Strauß, Matthias Schubert. (2024)  
**Spatial-Aware Deep Reinforcement Learning for the Traveling Officer Problem**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.05969v1)  

---


**ABSTRACT**  
The traveling officer problem (TOP) is a challenging stochastic optimization task. In this problem, a parking officer is guided through a city equipped with parking sensors to fine as many parking offenders as possible. A major challenge in TOP is the dynamic nature of parking offenses, which randomly appear and disappear after some time, regardless of whether they have been fined. Thus, solutions need to dynamically adjust to currently fineable parking offenses while also planning ahead to increase the likelihood that the officer arrives during the offense taking place. Though various solutions exist, these methods often struggle to take the implications of actions on the ability to fine future parking violations into account. This paper proposes SATOP, a novel spatial-aware deep reinforcement learning approach for TOP. Our novel state encoder creates a representation of each action, leveraging the spatial relationships between parking spots, the agent, and the action. Furthermore, we propose a novel message-passing module for learning future inter-action correlations in the given environment. Thus, the agent can estimate the potential to fine further parking violations after executing an action. We evaluate our method using an environment based on real-world data from Melbourne. Our results show that SATOP consistently outperforms state-of-the-art TOP agents and is able to fine up to 22% more parking offenses.

{{</citation>}}


### (20/94) Learning Cognitive Maps from Transformer Representations for Efficient Planning in Partially Observed Environments (Antoine Dedieu et al., 2024)

{{<citation>}}

Antoine Dedieu, Wolfgang Lehrach, Guangyao Zhou, Dileep George, Miguel Lázaro-Gredilla. (2024)  
**Learning Cognitive Maps from Transformer Representations for Efficient Planning in Partially Observed Environments**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: LSTM, Transformer  
[Paper Link](http://arxiv.org/abs/2401.05946v1)  

---


**ABSTRACT**  
Despite their stellar performance on a wide range of tasks, including in-context tasks only revealed during inference, vanilla transformers and variants trained for next-token predictions (a) do not learn an explicit world model of their environment which can be flexibly queried and (b) cannot be used for planning or navigation. In this paper, we consider partially observed environments (POEs), where an agent receives perceptually aliased observations as it navigates, which makes path planning hard. We introduce a transformer with (multiple) discrete bottleneck(s), TDB, whose latent codes learn a compressed representation of the history of observations and actions. After training a TDB to predict the future observation(s) given the history, we extract interpretable cognitive maps of the environment from its active bottleneck(s) indices. These maps are then paired with an external solver to solve (constrained) path planning problems. First, we show that a TDB trained on POEs (a) retains the near perfect predictive performance of a vanilla transformer or an LSTM while (b) solving shortest path problems exponentially faster. Second, a TDB extracts interpretable representations from text datasets, while reaching higher in-context accuracy than vanilla sequence models. Finally, in new POEs, a TDB (a) reaches near-perfect in-context accuracy, (b) learns accurate in-context cognitive maps (c) solves in-context path planning problems.

{{</citation>}}


### (21/94) Inferring Intentions to Speak Using Accelerometer Data In-the-Wild (Litian Li et al., 2024)

{{<citation>}}

Litian Li, Jord Molhoek, Jing Zhou. (2024)  
**Inferring Intentions to Speak Using Accelerometer Data In-the-Wild**  

---
Primary Category: cs.LG  
Categories: I-5-5; I-2-6, cs-AI, cs-CL, cs-HC, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.05849v1)  

---


**ABSTRACT**  
Humans have good natural intuition to recognize when another person has something to say. It would be interesting if an AI can also recognize intentions to speak. Especially in scenarios when an AI is guiding a group discussion, this can be a useful skill. This work studies the inference of successful and unsuccessful intentions to speak from accelerometer data. This is chosen because it is privacy-preserving and feasible for in-the-wild settings since it can be placed in a smart badge. Data from a real-life social networking event is used to train a machine-learning model that aims to infer intentions to speak. A subset of unsuccessful intention-to-speak cases in the data is annotated. The model is trained on the successful intentions to speak and evaluated on both the successful and unsuccessful cases. In conclusion, there is useful information in accelerometer data, but not enough to reliably capture intentions to speak. For example, posture shifts are correlated with intentions to speak, but people also often shift posture without having an intention to speak, or have an intention to speak without shifting their posture. More modalities are likely needed to reliably infer intentions to speak.

{{</citation>}}


### (22/94) Interpretable Concept Bottlenecks to Align Reinforcement Learning Agents (Quentin Delfosse et al., 2024)

{{<citation>}}

Quentin Delfosse, Sebastian Sztwiertnia, Wolfgang Stammer, Mark Rothermel, Kristian Kersting. (2024)  
**Interpretable Concept Bottlenecks to Align Reinforcement Learning Agents**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SC, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.05821v1)  

---


**ABSTRACT**  
Reward sparsity, difficult credit assignment, and misalignment are only a few of the many issues that make it difficult, if not impossible, for deep reinforcement learning (RL) agents to learn optimal policies. Unfortunately, the black-box nature of deep networks impedes the inclusion of domain experts who could interpret the model and correct wrong behavior. To this end, we introduce Successive Concept Bottlenecks Agents (SCoBots), which make the whole decision pipeline transparent via the integration of consecutive concept bottleneck layers. SCoBots make use of not only relevant object properties but also of relational concepts. Our experimental results provide strong evidence that SCoBots allow domain experts to efficiently understand and regularize their behavior, resulting in potentially better human-aligned RL. In this way, SCoBots enabled us to identify a misalignment problem in the most simple and iconic video game, Pong, and resolve it.

{{</citation>}}


### (23/94) Implications of Noise in Resistive Memory on Deep Neural Networks for Image Classification (Yannick Emonds et al., 2024)

{{<citation>}}

Yannick Emonds, Kai Xi, Holger Fröning. (2024)  
**Implications of Noise in Resistive Memory on Deep Neural Networks for Image Classification**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-ET, cs-LG, cs-PF, cs.LG  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2401.05820v1)  

---


**ABSTRACT**  
Resistive memory is a promising alternative to SRAM, but is also an inherently unstable device that requires substantial effort to ensure correct read and write operations. To avoid the associated costs in terms of area, time and energy, the present work is concerned with exploring how much noise in memory operations can be tolerated by image classification tasks based on neural networks. We introduce a special noisy operator that mimics the noise in an exemplary resistive memory unit, explore the resilience of convolutional neural networks on the CIFAR-10 classification task, and discuss a couple of countermeasures to improve this resilience.

{{</citation>}}


### (24/94) Graph Spatiotemporal Process for Multivariate Time Series Anomaly Detection with Missing Values (Yu Zheng et al., 2024)

{{<citation>}}

Yu Zheng, Huan Yee Koh, Ming Jin, Lianhua Chi, Haishuai Wang, Khoa T. Phan, Yi-Ping Phoebe Chen, Shirui Pan, Wei Xiang. (2024)  
**Graph Spatiotemporal Process for Multivariate Time Series Anomaly Detection with Missing Values**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Anomaly Detection, Time Series  
[Paper Link](http://arxiv.org/abs/2401.05800v1)  

---


**ABSTRACT**  
The detection of anomalies in multivariate time series data is crucial for various practical applications, including smart power grids, traffic flow forecasting, and industrial process control. However, real-world time series data is usually not well-structured, posting significant challenges to existing approaches: (1) The existence of missing values in multivariate time series data along variable and time dimensions hinders the effective modeling of interwoven spatial and temporal dependencies, resulting in important patterns being overlooked during model training; (2) Anomaly scoring with irregularly-sampled observations is less explored, making it difficult to use existing detectors for multivariate series without fully-observed values. In this work, we introduce a novel framework called GST-Pro, which utilizes a graph spatiotemporal process and anomaly scorer to tackle the aforementioned challenges in detecting anomalies on irregularly-sampled multivariate time series. Our approach comprises two main components. First, we propose a graph spatiotemporal process based on neural controlled differential equations. This process enables effective modeling of multivariate time series from both spatial and temporal perspectives, even when the data contains missing values. Second, we present a novel distribution-based anomaly scoring mechanism that alleviates the reliance on complete uniform observations. By analyzing the predictions of the graph spatiotemporal process, our approach allows anomalies to be easily detected. Our experimental results show that the GST-Pro method can effectively detect anomalies in time series data and outperforms state-of-the-art methods, regardless of whether there are missing values present in the data. Our code is available: https://github.com/huankoh/GST-Pro.

{{</citation>}}


### (25/94) An experimental evaluation of Deep Reinforcement Learning algorithms for HVAC control (Antonio Manjavacas et al., 2024)

{{<citation>}}

Antonio Manjavacas, Alejandro Campoy-Nieves, Javier Jiménez-Raboso, Miguel Molina-Solana, Juan Gómez-Romero. (2024)  
**An experimental evaluation of Deep Reinforcement Learning algorithms for HVAC control**  

---
Primary Category: cs.LG  
Categories: I-2-8; J-2, cs-LG, cs-SY, cs.LG, eess-SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.05737v1)  

---


**ABSTRACT**  
Heating, Ventilation, and Air Conditioning (HVAC) systems are a major driver of energy consumption in commercial and residential buildings. Recent studies have shown that Deep Reinforcement Learning (DRL) algorithms can outperform traditional reactive controllers. However, DRL-based solutions are generally designed for ad hoc setups and lack standardization for comparison. To fill this gap, this paper provides a critical and reproducible evaluation, in terms of comfort and energy consumption, of several state-of-the-art DRL algorithms for HVAC control. The study examines the controllers' robustness, adaptability, and trade-off between optimization goals by using the Sinergym framework. The results obtained confirm the potential of DRL algorithms, such as SAC and TD3, in complex scenarios and reveal several challenges related to generalization and incremental learning.

{{</citation>}}


### (26/94) Dynamic Indoor Fingerprinting Localization based on Few-Shot Meta-Learning with CSI Images (Jiyu Jiao et al., 2024)

{{<citation>}}

Jiyu Jiao, Xiaojun Wang, Chenpei Han, Yuhua Huang, Yizhuo Zhang. (2024)  
**Dynamic Indoor Fingerprinting Localization based on Few-Shot Meta-Learning with CSI Images**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, eess-SP  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2401.05711v1)  

---


**ABSTRACT**  
While fingerprinting localization is favored for its effectiveness, it is hindered by high data acquisition costs and the inaccuracy of static database-based estimates. Addressing these issues, this letter presents an innovative indoor localization method using a data-efficient meta-learning algorithm. This approach, grounded in the ``Learning to Learn'' paradigm of meta-learning, utilizes historical localization tasks to improve adaptability and learning efficiency in dynamic indoor environments. We introduce a task-weighted loss to enhance knowledge transfer within this framework. Our comprehensive experiments confirm the method's robustness and superiority over current benchmarks, achieving a notable 23.13\% average gain in Mean Euclidean Distance, particularly effective in scenarios with limited CSI data.

{{</citation>}}


### (27/94) The Distributional Reward Critic Architecture for Perturbed-Reward Reinforcement Learning (Xi Chen et al., 2024)

{{<citation>}}

Xi Chen, Zhihui Zhu, Andrew Perrault. (2024)  
**The Distributional Reward Critic Architecture for Perturbed-Reward Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.05710v1)  

---


**ABSTRACT**  
We study reinforcement learning in the presence of an unknown reward perturbation. Existing methodologies for this problem make strong assumptions including reward smoothness, known perturbations, and/or perturbations that do not modify the optimal policy. We study the case of unknown arbitrary perturbations that discretize and shuffle reward space, but have the property that the true reward belongs to the most frequently observed class after perturbation. This class of perturbations generalizes existing classes (and, in the limit, all continuous bounded perturbations) and defeats existing methods. We introduce an adaptive distributional reward critic and show theoretically that it can recover the true rewards under technical conditions. Under the targeted perturbation in discrete and continuous control tasks, we win/tie the highest return in 40/57 settings (compared to 16/57 for the best baseline). Even under the untargeted perturbation, we still win an edge over the baseline designed especially for that setting.

{{</citation>}}


### (28/94) Graph Q-Learning for Combinatorial Optimization (Victoria M. Dax et al., 2024)

{{<citation>}}

Victoria M. Dax, Jiachen Li, Kevin Leahy, Mykel J. Kochenderfer. (2024)  
**Graph Q-Learning for Combinatorial Optimization**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2401.05610v1)  

---


**ABSTRACT**  
Graph-structured data is ubiquitous throughout natural and social sciences, and Graph Neural Networks (GNNs) have recently been shown to be effective at solving prediction and inference problems on graph data. In this paper, we propose and demonstrate that GNNs can be applied to solve Combinatorial Optimization (CO) problems. CO concerns optimizing a function over a discrete solution space that is often intractably large. To learn to solve CO problems, we formulate the optimization process as a sequential decision making problem, where the return is related to how close the candidate solution is to optimality. We use a GNN to learn a policy to iteratively build increasingly promising candidate solutions. We present preliminary evidence that GNNs trained through Q-Learning can solve CO problems with performance approaching state-of-the-art heuristic-based solvers, using only a fraction of the parameters and training time.

{{</citation>}}


## cs.CL (39)



### (29/94) Axis Tour: Word Tour Determines the Order of Axes in ICA-transformed Embeddings (Hiroaki Yamagiwa et al., 2024)

{{<citation>}}

Hiroaki Yamagiwa, Yusuke Takase, Hidetoshi Shimodaira. (2024)  
**Axis Tour: Word Tour Determines the Order of Axes in ICA-transformed Embeddings**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2401.06112v1)  

---


**ABSTRACT**  
Word embedding is one of the most important components in natural language processing, but interpreting high-dimensional embeddings remains a challenging problem. To address this problem, Independent Component Analysis (ICA) is identified as an effective solution. ICA-transformed word embeddings reveal interpretable semantic axes; however, the order of these axes are arbitrary. In this study, we focus on this property and propose a novel method, Axis Tour, which optimizes the order of the axes. Inspired by Word Tour, a one-dimensional word embedding method, we aim to improve the clarity of the word embedding space by maximizing the semantic continuity of the axes. Furthermore, we show through experiments on downstream tasks that Axis Tour constructs better low-dimensional embeddings compared to both PCA and ICA.

{{</citation>}}


### (30/94) Transformers are Multi-State RNNs (Matanel Oren et al., 2024)

{{<citation>}}

Matanel Oren, Michael Hassid, Yossi Adi, Roy Schwartz. (2024)  
**Transformers are Multi-State RNNs**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.06104v1)  

---


**ABSTRACT**  
Transformers are considered conceptually different compared to the previous generation of state-of-the-art NLP models - recurrent neural networks (RNNs). In this work, we demonstrate that decoder-only transformers can in fact be conceptualized as infinite multi-state RNNs - an RNN variant with unlimited hidden state size. We further show that pretrained transformers can be converted into $\textit{finite}$ multi-state RNNs by fixing the size of their hidden state. We observe that several existing transformers cache compression techniques can be framed as such conversion policies, and introduce a novel policy, TOVA, which is simpler compared to these policies. Our experiments with several long range tasks indicate that TOVA outperforms all other baseline policies, while being nearly on par with the full (infinite) model, and using in some cases only $\frac{1}{8}$ of the original cache size. Our results indicate that transformer decoder LLMs often behave in practice as RNNs. They also lay out the option of mitigating one of their most painful computational bottlenecks - the size of their cache memory. We publicly release our code at https://github.com/schwartz-lab-NLP/TOVA.

{{</citation>}}


### (31/94) Patchscope: A Unifying Framework for Inspecting Hidden Representations of Language Models (Asma Ghandeharioun et al., 2024)

{{<citation>}}

Asma Ghandeharioun, Avi Caciularu, Adam Pearce, Lucas Dixon, Mor Geva. (2024)  
**Patchscope: A Unifying Framework for Inspecting Hidden Representations of Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.06102v1)  

---


**ABSTRACT**  
Inspecting the information encoded in hidden representations of large language models (LLMs) can explain models' behavior and verify their alignment with human values. Given the capabilities of LLMs in generating human-understandable text, we propose leveraging the model itself to explain its internal representations in natural language. We introduce a framework called Patchscopes and show how it can be used to answer a wide range of research questions about an LLM's computation. We show that prior interpretability methods based on projecting representations into the vocabulary space and intervening on the LLM computation, can be viewed as special instances of this framework. Moreover, several of their shortcomings such as failure in inspecting early layers or lack of expressivity can be mitigated by a Patchscope. Beyond unifying prior inspection techniques, Patchscopes also opens up new possibilities such as using a more capable model to explain the representations of a smaller model, and unlocks new applications such as self-correction in multi-hop reasoning.

{{</citation>}}


### (32/94) Autocompletion of Chief Complaints in the Electronic Health Records using Large Language Models (K M Sajjadul Islam et al., 2024)

{{<citation>}}

K M Sajjadul Islam, Ayesha Siddika Nipu, Praveen Madiraju, Priya Deshpande. (2024)  
**Autocompletion of Chief Complaints in the Electronic Health Records using Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: AI, BERT, GPT, GPT-4, LSTM, Language Model, QA, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.06088v1)  

---


**ABSTRACT**  
The Chief Complaint (CC) is a crucial component of a patient's medical record as it describes the main reason or concern for seeking medical care. It provides critical information for healthcare providers to make informed decisions about patient care. However, documenting CCs can be time-consuming for healthcare providers, especially in busy emergency departments. To address this issue, an autocompletion tool that suggests accurate and well-formatted phrases or sentences for clinical notes can be a valuable resource for triage nurses. In this study, we utilized text generation techniques to develop machine learning models using CC data. In our proposed work, we train a Long Short-Term Memory (LSTM) model and fine-tune three different variants of Biomedical Generative Pretrained Transformers (BioGPT), namely microsoft/biogpt, microsoft/BioGPT-Large, and microsoft/BioGPT-Large-PubMedQA. Additionally, we tune a prompt by incorporating exemplar CC sentences, utilizing the OpenAI API of GPT-4. We evaluate the models' performance based on the perplexity score, modified BERTScore, and cosine similarity score. The results show that BioGPT-Large exhibits superior performance compared to the other models. It consistently achieves a remarkably low perplexity score of 1.65 when generating CC, whereas the baseline LSTM model achieves the best perplexity score of 170. Further, we evaluate and assess the proposed models' performance and the outcome of GPT-4.0. Our study demonstrates that utilizing LLMs such as BioGPT, leads to the development of an effective autocompletion tool for generating CC documentation in healthcare settings.

{{</citation>}}


### (33/94) Improving Large Language Models via Fine-grained Reinforcement Learning with Minimum Editing Constraint (Zhipeng Chen et al., 2024)

{{<citation>}}

Zhipeng Chen, Kun Zhou, Wayne Xin Zhao, Junchen Wan, Fuzheng Zhang, Di Zhang, Ji-Rong Wen. (2024)  
**Improving Large Language Models via Fine-grained Reinforcement Learning with Minimum Editing Constraint**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Language Model, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.06081v1)  

---


**ABSTRACT**  
Reinforcement learning (RL) has been widely used in training large language models~(LLMs) for preventing unexpected outputs, \eg reducing harmfulness and errors. However, existing RL methods mostly adopt the instance-level reward, which is unable to provide fine-grained supervision for complex reasoning tasks, and can not focus on the few key tokens that lead to the incorrectness. To address it, we propose a new RL method named \textbf{RLMEC} that incorporates a generative model as the reward model, which is trained by the erroneous solution rewriting task under the minimum editing constraint, and can produce token-level rewards for RL training. Based on the generative reward model, we design the token-level RL objective for training and an imitation-based regularization for stabilizing RL process. And the both objectives focus on the learning of the key tokens for the erroneous solution, reducing the effect of other unimportant tokens. The experiment results on mathematical tasks and question-answering tasks have demonstrated the effectiveness of our approach. Our code and data are available at \url{https://github.com/RUCAIBox/RLMEC}.

{{</citation>}}


### (34/94) DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models (Damai Dai et al., 2024)

{{<citation>}}

Damai Dai, Chengqi Deng, Chenggang Zhao, R. X. Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Y. Wu, Zhenda Xie, Y. K. Li, Panpan Huang, Fuli Luo, Chong Ruan, Zhifang Sui, Wenfeng Liang. (2024)  
**DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2401.06066v1)  

---


**ABSTRACT**  
In the era of large language models, Mixture-of-Experts (MoE) is a promising architecture for managing computational costs when scaling up model parameters. However, conventional MoE architectures like GShard, which activate the top-$K$ out of $N$ experts, face challenges in ensuring expert specialization, i.e. each expert acquires non-overlapping and focused knowledge. In response, we propose the DeepSeekMoE architecture towards ultimate expert specialization. It involves two principal strategies: (1) finely segmenting the experts into $mN$ ones and activating $mK$ from them, allowing for a more flexible combination of activated experts; (2) isolating $K_s$ experts as shared ones, aiming at capturing common knowledge and mitigating redundancy in routed experts. Starting from a modest scale with 2B parameters, we demonstrate that DeepSeekMoE 2B achieves comparable performance with GShard 2.9B, which has 1.5 times the expert parameters and computation. In addition, DeepSeekMoE 2B nearly approaches the performance of its dense counterpart with the same number of total parameters, which set the upper bound of MoE models. Subsequently, we scale up DeepSeekMoE to 16B parameters and show that it achieves comparable performance with LLaMA2 7B, with only about 40% of computations. Further, our preliminary efforts to scale up DeepSeekMoE to 145B parameters consistently validate its substantial advantages over the GShard architecture, and show its performance comparable with DeepSeek 67B, using only 28.5% (maybe even 18.2%) of computations.

{{</citation>}}


### (35/94) Investigating Data Contamination for Pre-training Language Models (Minhao Jiang et al., 2024)

{{<citation>}}

Minhao Jiang, Ken Ziyu Liu, Ming Zhong, Rylan Schaeffer, Siru Ouyang, Jiawei Han, Sanmi Koyejo. (2024)  
**Investigating Data Contamination for Pre-training Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2401.06059v1)  

---


**ABSTRACT**  
Language models pre-trained on web-scale corpora demonstrate impressive capabilities on diverse downstream tasks. However, there is increasing concern whether such capabilities might arise from evaluation datasets being included in the pre-training corpus -- a phenomenon known as \textit{data contamination} -- in a manner that artificially increases performance. There has been little understanding of how this potential contamination might influence LMs' performance on downstream tasks. In this paper, we explore the impact of data contamination at the pre-training stage by pre-training a series of GPT-2 models \textit{from scratch}. We highlight the effect of both text contamination (\textit{i.e.}\ input text of the evaluation samples) and ground-truth contamination (\textit{i.e.}\ the prompts asked on the input and the desired outputs) from evaluation data. We also investigate the effects of repeating contamination for various downstream tasks. Additionally, we examine the prevailing n-gram-based definitions of contamination within current LLM reports, pinpointing their limitations and inadequacy. Our findings offer new insights into data contamination's effects on language model capabilities and underscore the need for independent, comprehensive contamination assessments in LLM studies.

{{</citation>}}


### (36/94) LinguAlchemy: Fusing Typological and Geographical Elements for Unseen Language Generalization (Muhammad Farid Adilazuarda et al., 2024)

{{<citation>}}

Muhammad Farid Adilazuarda, Samuel Cahyawijaya, Alham Fikri Aji, Genta Indra Winata, Ayu Purwarianti. (2024)  
**LinguAlchemy: Fusing Typological and Geographical Elements for Unseen Language Generalization**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2401.06034v1)  

---


**ABSTRACT**  
Pretrained language models (PLMs) have shown remarkable generalization toward multiple tasks and languages. Nonetheless, the generalization of PLMs towards unseen languages is poor, resulting in significantly worse language performance, or even generating nonsensical responses that are comparable to a random baseline. This limitation has been a longstanding problem of PLMs raising the problem of diversity and equal access to language modeling technology. In this work, we solve this limitation by introducing LinguAlchemy, a regularization technique that incorporates various aspects of languages covering typological, geographical, and phylogenetic constraining the resulting representation of PLMs to better characterize the corresponding linguistics constraints. LinguAlchemy significantly improves the accuracy performance of mBERT and XLM-R on unseen languages by ~18% and ~2%, respectively compared to fully finetuned models and displaying a high degree of unseen language generalization. We further introduce AlchemyScale and AlchemyTune, extension of LinguAlchemy which adjusts the linguistic regularization weights automatically, alleviating the need for hyperparameter search. LinguAlchemy enables better cross-lingual generalization to unseen languages which is vital for better inclusivity and accessibility of PLMs.

{{</citation>}}


### (37/94) Combating Adversarial Attacks with Multi-Agent Debate (Steffi Chern et al., 2024)

{{<citation>}}

Steffi Chern, Zhen Fan, Andy Liu. (2024)  
**Combating Adversarial Attacks with Multi-Agent Debate**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2401.05998v1)  

---


**ABSTRACT**  
While state-of-the-art language models have achieved impressive results, they remain susceptible to inference-time adversarial attacks, such as adversarial prompts generated by red teams arXiv:2209.07858. One approach proposed to improve the general quality of language model generations is multi-agent debate, where language models self-evaluate through discussion and feedback arXiv:2305.14325. We implement multi-agent debate between current state-of-the-art language models and evaluate models' susceptibility to red team attacks in both single- and multi-agent settings. We find that multi-agent debate can reduce model toxicity when jailbroken or less capable models are forced to debate with non-jailbroken or more capable models. We also find marginal improvements through the general usage of multi-agent interactions. We further perform adversarial prompt content classification via embedding clustering, and analyze the susceptibility of different models to different types of attack topics.

{{</citation>}}


### (38/94) Block-Diagonal Orthogonal Relation and Matrix Entity for Knowledge Graph Embedding (Yihua Zhu et al., 2024)

{{<citation>}}

Yihua Zhu, Hidetoshi Shimodaira. (2024)  
**Block-Diagonal Orthogonal Relation and Matrix Entity for Knowledge Graph Embedding**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-IR, cs.CL  
Keywords: Embedding, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2401.05967v1)  

---


**ABSTRACT**  
The primary aim of Knowledge Graph embeddings (KGE) is to learn low-dimensional representations of entities and relations for predicting missing facts. While rotation-based methods like RotatE and QuatE perform well in KGE, they face two challenges: limited model flexibility requiring proportional increases in relation size with entity dimension, and difficulties in generalizing the model for higher-dimensional rotations. To address these issues, we introduce OrthogonalE, a novel KGE model employing matrices for entities and block-diagonal orthogonal matrices with Riemannian optimization for relations. This approach enhances the generality and flexibility of KGE models. The experimental results indicate that our new KGE model, OrthogonalE, is both general and flexible, significantly outperforming state-of-the-art KGE models while substantially reducing the number of relation parameters.

{{</citation>}}


### (39/94) LLM-as-a-Coauthor: The Challenges of Detecting LLM-Human Mixcase (Chujie Gao et al., 2024)

{{<citation>}}

Chujie Gao, Dongping Chen, Qihui Zhang, Yue Huang, Yao Wan, Lichao Sun. (2024)  
**LLM-as-a-Coauthor: The Challenges of Detecting LLM-Human Mixcase**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.05952v1)  

---


**ABSTRACT**  
With the remarkable development and widespread applications of large language models (LLMs), the use of machine-generated text (MGT) is becoming increasingly common. This trend brings potential risks, particularly to the quality and completeness of information in fields such as news and education. Current research predominantly addresses the detection of pure MGT without adequately addressing mixed scenarios including AI-revised Human-Written Text (HWT) or human-revised MGT. To confront this challenge, we introduce mixcase, a novel concept representing a hybrid text form involving both machine-generated and human-generated content. We collected mixcase instances generated from multiple daily text-editing scenarios and composed MixSet, the first dataset dedicated to studying these mixed modification scenarios. We conduct experiments to evaluate the efficacy of popular MGT detectors, assessing their effectiveness, robustness, and generalization performance. Our findings reveal that existing detectors struggle to identify mixcase as a separate class or MGT, particularly in dealing with subtle modifications and style adaptability. This research underscores the urgent need for more fine-grain detectors tailored for mixcase, offering valuable insights for future research. Code and Models are available at https://github.com/Dongping-Chen/MixSet.

{{</citation>}}


### (40/94) Universal Vulnerabilities in Large Language Models: In-context Learning Backdoor Attacks (Shuai Zhao et al., 2024)

{{<citation>}}

Shuai Zhao, Meihuizi Jia, Luu Anh Tuan, Jinming Wen. (2024)  
**Universal Vulnerabilities in Large Language Models: In-context Learning Backdoor Attacks**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2401.05949v1)  

---


**ABSTRACT**  
In-context learning, a paradigm bridging the gap between pre-training and fine-tuning, has demonstrated high efficacy in several NLP tasks, especially in few-shot settings. Unlike traditional fine-tuning methods, in-context learning adapts pre-trained models to unseen tasks without updating any parameters. Despite being widely applied, in-context learning is vulnerable to malicious attacks. In this work, we raise security concerns regarding this paradigm. Our studies demonstrate that an attacker can manipulate the behavior of large language models by poisoning the demonstration context, without the need for fine-tuning the model. Specifically, we have designed a new backdoor attack method, named ICLAttack, to target large language models based on in-context learning. Our method encompasses two types of attacks: poisoning demonstration examples and poisoning prompts, which can make models behave in accordance with predefined intentions. ICLAttack does not require additional fine-tuning to implant a backdoor, thus preserving the model's generality. Furthermore, the poisoned examples are correctly labeled, enhancing the natural stealth of our attack method. Extensive experimental results across several language models, ranging in size from 1.3B to 40B parameters, demonstrate the effectiveness of our attack method, exemplified by a high average attack success rate of 95.0% across the three datasets on OPT models. Our findings highlight the vulnerabilities of language models, and we hope this work will raise awareness of the possible security threats associated with in-context learning.

{{</citation>}}


### (41/94) SH2: Self-Highlighted Hesitation Helps You Decode More Truthfully (Jushi Kai et al., 2024)

{{<citation>}}

Jushi Kai, Tianhang Zhang, Hai Hu, Zhouhan Lin. (2024)  
**SH2: Self-Highlighted Hesitation Helps You Decode More Truthfully**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: LLaMA  
[Paper Link](http://arxiv.org/abs/2401.05930v1)  

---


**ABSTRACT**  
Large language models (LLMs) demonstrate great performance in text generation. However, LLMs are still suffering from hallucinations. In this work, we propose an inference-time method, Self-Highlighted Hesitation (SH2), to help LLMs decode more truthfully. SH2 is based on a simple fact rooted in information theory that for an LLM, the tokens predicted with lower probabilities are prone to be more informative than others. Our analysis shows that the tokens assigned with lower probabilities by an LLM are more likely to be closely related to factual information, such as nouns, proper nouns, and adjectives. Therefore, we propose to ''highlight'' the factual information by selecting the tokens with the lowest probabilities and concatenating them to the original context, thus forcing the model to repeatedly read and hesitate on these tokens before generation. During decoding, we also adopt contrastive decoding to emphasize the difference in the output probabilities brought by the hesitation. Experimental results demonstrate that our SH2, requiring no additional data or models, can effectively help LLMs elicit factual knowledge and distinguish hallucinated contexts. Significant and consistent improvements are achieved by SH2 for LLaMA-7b and LLaMA2-7b on multiple hallucination tasks.

{{</citation>}}


### (42/94) Mitigating Unhelpfulness in Emotional Support Conversations with Multifaceted AI Feedback (Jiashuo Wang et al., 2024)

{{<citation>}}

Jiashuo Wang, Chunpu Xu, Chak Tou Leong, Wenjie Li, Jing Li. (2024)  
**Mitigating Unhelpfulness in Emotional Support Conversations with Multifaceted AI Feedback**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.05928v1)  

---


**ABSTRACT**  
An emotional support conversation system aims to alleviate users' emotional distress and assist them in addressing their challenges. To generate supportive responses, it is critical to consider multiple factors such as empathy, support strategies, and response coherence, as established in prior methods. Nonetheless, previous models occasionally generate unhelpful responses, which intend to provide support but display counterproductive effects. According to psychology and communication theories, poor performance in just one contributing factor might cause a response to be unhelpful. From the model training perspective, since these models have not been exposed to unhelpful responses during their training phase, they are unable to distinguish if the tokens they generate might result in unhelpful responses during inference. To address this issue, we introduce a novel model-agnostic framework named mitigating unhelpfulness with multifaceted AI feedback for emotional support (Muffin). Specifically, Muffin employs a multifaceted AI feedback module to assess the helpfulness of responses generated by a specific model with consideration of multiple factors. Using contrastive learning, it then reduces the likelihood of the model generating unhelpful responses compared to the helpful ones. Experimental results demonstrate that Muffin effectively mitigates the generation of unhelpful responses while slightly increasing response fluency and relevance.

{{</citation>}}


### (43/94) How Teachers Can Use Large Language Models and Bloom's Taxonomy to Create Educational Quizzes (Sabina Elkins et al., 2024)

{{<citation>}}

Sabina Elkins, Ekaterina Kochmar, Jackie C. K. Cheung, Iulian Serban. (2024)  
**How Teachers Can Use Large Language Models and Bloom's Taxonomy to Create Educational Quizzes**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.05914v1)  

---


**ABSTRACT**  
Question generation (QG) is a natural language processing task with an abundance of potential benefits and use cases in the educational domain. In order for this potential to be realized, QG systems must be designed and validated with pedagogical needs in mind. However, little research has assessed or designed QG approaches with the input from real teachers or students. This paper applies a large language model-based QG approach where questions are generated with learning goals derived from Bloom's taxonomy. The automatically generated questions are used in multiple experiments designed to assess how teachers use them in practice. The results demonstrate that teachers prefer to write quizzes with automatically generated questions, and that such quizzes have no loss in quality compared to handwritten versions. Further, several metrics indicate that automatically generated questions can even improve the quality of the quizzes created, showing the promise for large scale use of QG in the classroom setting.

{{</citation>}}


### (44/94) Prompt-based mental health screening from social media text (Wesley Ramos dos Santos et al., 2024)

{{<citation>}}

Wesley Ramos dos Santos, Ivandre Paraboni. (2024)  
**Prompt-based mental health screening from social media text**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, GPT  
[Paper Link](http://arxiv.org/abs/2401.05912v1)  

---


**ABSTRACT**  
This article presents a method for prompt-based mental health screening from a large and noisy dataset of social media text. Our method uses GPT 3.5. prompting to distinguish publications that may be more relevant to the task, and then uses a straightforward bag-of-words text classifier to predict actual user labels. Results are found to be on pair with a BERT mixture of experts classifier, and incurring only a fraction of its computational costs.

{{</citation>}}


### (45/94) EpilepsyLLM: Domain-Specific Large Language Model Fine-tuned with Epilepsy Medical Knowledge (Xuyang Zhao et al., 2024)

{{<citation>}}

Xuyang Zhao, Qibin Zhao, Toshihisa Tanaka. (2024)  
**EpilepsyLLM: Domain-Specific Large Language Model Fine-tuned with Epilepsy Medical Knowledge**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.05908v1)  

---


**ABSTRACT**  
With large training datasets and massive amounts of computing sources, large language models (LLMs) achieve remarkable performance in comprehensive and generative ability. Based on those powerful LLMs, the model fine-tuned with domain-specific datasets posseses more specialized knowledge and thus is more practical like medical LLMs. However, the existing fine-tuned medical LLMs are limited to general medical knowledge with English language. For disease-specific problems, the model's response is inaccurate and sometimes even completely irrelevant, especially when using a language other than English. In this work, we focus on the particular disease of Epilepsy with Japanese language and introduce a customized LLM termed as EpilepsyLLM. Our model is trained from the pre-trained LLM by fine-tuning technique using datasets from the epilepsy domain. The datasets contain knowledge of basic information about disease, common treatment methods and drugs, and important notes in life and work. The experimental results demonstrate that EpilepsyLLM can provide more reliable and specialized medical knowledge responses.

{{</citation>}}


### (46/94) Enhancing Personality Recognition in Dialogue by Data Augmentation and Heterogeneous Conversational Graph Networks (Yahui Fu et al., 2024)

{{<citation>}}

Yahui Fu, Haiyue Song, Tianyu Zhao, Tatsuya Kawahara. (2024)  
**Enhancing Personality Recognition in Dialogue by Data Augmentation and Heterogeneous Conversational Graph Networks**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Augmentation, Dialog, Dialogue, Personality Recognition  
[Paper Link](http://arxiv.org/abs/2401.05871v1)  

---


**ABSTRACT**  
Personality recognition is useful for enhancing robots' ability to tailor user-adaptive responses, thus fostering rich human-robot interactions. One of the challenges in this task is a limited number of speakers in existing dialogue corpora, which hampers the development of robust, speaker-independent personality recognition models. Additionally, accurately modeling both the interdependencies among interlocutors and the intra-dependencies within the speaker in dialogues remains a significant issue. To address the first challenge, we introduce personality trait interpolation for speaker data augmentation. For the second, we propose heterogeneous conversational graph networks to independently capture both contextual influences and inherent personality traits. Evaluations on the RealPersonaChat corpus demonstrate our method's significant improvements over existing baselines.

{{</citation>}}


### (47/94) Towards Boosting Many-to-Many Multilingual Machine Translation with Large Language Models (Pengzhi Gao et al., 2024)

{{<citation>}}

Pengzhi Gao, Zhongjun He, Hua Wu, Haifeng Wang. (2024)  
**Towards Boosting Many-to-Many Multilingual Machine Translation with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: LLaMA, Language Model, Machine Translation, Multilingual  
[Paper Link](http://arxiv.org/abs/2401.05861v1)  

---


**ABSTRACT**  
The training paradigm for machine translation has gradually shifted, from learning neural machine translation (NMT) models with extensive parallel corpora to instruction finetuning on pretrained multilingual large language models (LLMs) with high-quality translation pairs. In this paper, we focus on boosting the many-to-many multilingual translation performance of LLMs with an emphasis on zero-shot translation directions. We demonstrate that prompt strategies adopted during instruction finetuning are crucial to zero-shot translation performance and introduce a cross-lingual consistency regularization, XConST, to bridge the representation gap among different languages and improve zero-shot translation performance. XConST is not a new method, but a version of CrossConST (Gao et al., 2023a) adapted for multilingual finetuning on LLMs with translation instructions. Experimental results on ALMA (Xu et al., 2023) and LLaMA-2 (Touvron et al., 2023) show that our approach consistently improves translation performance. Our implementations are available at https://github.com/gpengzhi/CrossConST-LLM.

{{</citation>}}


### (48/94) Hallucination Benchmark in Medical Visual Question Answering (Jinge Wu et al., 2024)

{{<citation>}}

Jinge Wu, Yunsoo Kim, Honghan Wu. (2024)  
**Hallucination Benchmark in Medical Visual Question Answering**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CV, cs.CL  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2401.05827v1)  

---


**ABSTRACT**  
The recent success of large language and vision models on vision question answering (VQA), particularly their applications in medicine (Med-VQA), has shown a great potential of realizing effective visual assistants for healthcare. However, these models are not extensively tested on the hallucination phenomenon in clinical settings. Here, we created a hallucination benchmark of medical images paired with question-answer sets and conducted a comprehensive evaluation of the state-of-the-art models. The study provides an in-depth analysis of current models limitations and reveals the effectiveness of various prompting strategies.

{{</citation>}}


### (49/94) Tuning LLMs with Contrastive Alignment Instructions for Machine Translation in Unseen, Low-resource Languages (Zhuoyuan Mao et al., 2024)

{{<citation>}}

Zhuoyuan Mao, Yen Yu. (2024)  
**Tuning LLMs with Contrastive Alignment Instructions for Machine Translation in Unseen, Low-resource Languages**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BLOOM, Machine Translation  
[Paper Link](http://arxiv.org/abs/2401.05811v1)  

---


**ABSTRACT**  
This article introduces contrastive alignment instructions (AlignInstruct) to address two challenges in machine translation (MT) on large language models (LLMs). One is the expansion of supported languages to previously unseen ones. The second relates to the lack of data in low-resource languages. Model fine-tuning through MT instructions (MTInstruct) is a straightforward approach to the first challenge. However, MTInstruct is limited by weak cross-lingual signals inherent in the second challenge. AlignInstruct emphasizes cross-lingual supervision via a cross-lingual discriminator built using statistical word alignments. Our results based on fine-tuning the BLOOMZ models (1b1, 3b, and 7b1) in up to 24 unseen languages showed that: (1) LLMs can effectively translate unseen languages using MTInstruct; (2) AlignInstruct led to consistent improvements in translation quality across 48 translation directions involving English; (3) Discriminator-based instructions outperformed their generative counterparts as cross-lingual instructions; (4) AlignInstruct improved performance in 30 zero-shot directions.

{{</citation>}}


### (50/94) Designing Heterogeneous LLM Agents for Financial Sentiment Analysis (Frank Xing, 2024)

{{<citation>}}

Frank Xing. (2024)  
**Designing Heterogeneous LLM Agents for Financial Sentiment Analysis**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-MA, cs.CL, q-fin-GN  
Keywords: Financial, Sentiment Analysis  
[Paper Link](http://arxiv.org/abs/2401.05799v1)  

---


**ABSTRACT**  
Large language models (LLMs) have drastically changed the possible ways to design intelligent systems, shifting the focuses from massive data acquisition and new modeling training to human alignment and strategical elicitation of the full potential of existing pre-trained models. This paradigm shift, however, is not fully realized in financial sentiment analysis (FSA), due to the discriminative nature of this task and a lack of prescriptive knowledge of how to leverage generative models in such a context. This study investigates the effectiveness of the new paradigm, i.e., using LLMs without fine-tuning for FSA. Rooted in Minsky's theory of mind and emotions, a design framework with heterogeneous LLM agents is proposed. The framework instantiates specialized agents using prior domain knowledge of the types of FSA errors and reasons on the aggregated agent discussions. Comprehensive evaluation on FSA datasets show that the framework yields better accuracies, especially when the discussions are substantial. This study contributes to the design foundations and paves new avenues for LLMs-based FSA. Implications on business and management are also discussed.

{{</citation>}}


### (51/94) Discovering Low-rank Subspaces for Language-agnostic Multilingual Representations (Zhihui Xie et al., 2024)

{{<citation>}}

Zhihui Xie, Handong Zhao, Tong Yu, Shuai Li. (2024)  
**Discovering Low-rank Subspaces for Language-agnostic Multilingual Representations**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Multilingual, QA  
[Paper Link](http://arxiv.org/abs/2401.05792v1)  

---


**ABSTRACT**  
Large pretrained multilingual language models (ML-LMs) have shown remarkable capabilities of zero-shot cross-lingual transfer, without direct cross-lingual supervision. While these results are promising, follow-up works found that, within the multilingual embedding spaces, there exists strong language identity information which hinders the expression of linguistic factors shared across languages. For semantic tasks like cross-lingual sentence retrieval, it is desired to remove such language identity signals to fully leverage semantic information. In this work, we provide a novel view of projecting away language-specific factors from a multilingual embedding space. Specifically, we discover that there exists a low-rank subspace that primarily encodes information irrelevant to semantics (e.g., syntactic information). To identify this subspace, we present a simple but effective unsupervised method based on singular value decomposition with multiple monolingual corpora as input. Once the subspace is found, we can directly project the original embeddings into the null space to boost language agnosticism without finetuning. We systematically evaluate our method on various tasks including the challenging language-agnostic QA retrieval task. Empirical results show that applying our method consistently leads to improvements over commonly used ML-LMs.

{{</citation>}}


### (52/94) Evidence to Generate (E2G): A Single-agent Two-step Prompting for Context Grounded and Retrieval Augmented Reasoning (Md Rizwan Parvez, 2024)

{{<citation>}}

Md Rizwan Parvez. (2024)  
**Evidence to Generate (E2G): A Single-agent Two-step Prompting for Context Grounded and Retrieval Augmented Reasoning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4, PaLM, QA, Reasoning  
[Paper Link](http://arxiv.org/abs/2401.05787v1)  

---


**ABSTRACT**  
While chain-of-thought (CoT) prompting has revolutionized how LLMs perform reasoning tasks, its current methods and variations (e.g, Self-consistency, ReACT, Reflexion, Tree-of-Thoughts (ToT), Cumulative Reasoning (CR)) suffer from limitations like slowness, limited context grounding, hallucination and inconsistent outputs. To overcome these challenges, we introduce Evidence to Generate (E2G), a novel single-agent, two-step prompting framework. Instead of unverified reasoning claims, this innovative approach leverages the power of "evidence for decision making" by first focusing exclusively on the thought sequences (the series of intermediate steps) explicitly mentioned in the context which then serve as extracted evidence, guiding the LLM's output generation process with greater precision and efficiency. This simple yet powerful approach unlocks the true potential of chain-of-thought like prompting, paving the way for faster, more reliable, and more contextually aware reasoning in LLMs. \tool achieves remarkable results robustly across a wide range of knowledge-intensive reasoning and generation tasks, surpassing baseline approaches with state-of-the-art LLMs. For example, (i) on LogiQA benchmark using GPT-4 as backbone model, \tool achieves a new state-of-the Accuracy of 53.8% exceeding CoT by 18%, ToT by 11%, CR by 9% (ii) a variant of E2G with PaLM2 outperforms the variable-shot performance of Gemini Ultra by 0.9 F1 points, reaching an F1 score of 83.3 on a subset of DROP.

{{</citation>}}


### (53/94) Risk Taxonomy, Mitigation, and Assessment Benchmarks of Large Language Model Systems (Tianyu Cui et al., 2024)

{{<citation>}}

Tianyu Cui, Yanling Wang, Chuanpu Fu, Yong Xiao, Sijia Li, Xinhao Deng, Yunpeng Liu, Qinglin Zhang, Ziyi Qiu, Peiyang Li, Zhixing Tan, Junwu Xiong, Xinyu Kong, Zujie Wen, Ke Xu, Qi Li. (2024)  
**Risk Taxonomy, Mitigation, and Assessment Benchmarks of Large Language Model Systems**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, Google, Language Model  
[Paper Link](http://arxiv.org/abs/2401.05778v1)  

---


**ABSTRACT**  
Large language models (LLMs) have strong capabilities in solving diverse natural language processing tasks. However, the safety and security issues of LLM systems have become the major obstacle to their widespread application. Many studies have extensively investigated risks in LLM systems and developed the corresponding mitigation strategies. Leading-edge enterprises such as OpenAI, Google, Meta, and Anthropic have also made lots of efforts on responsible LLMs. Therefore, there is a growing need to organize the existing studies and establish comprehensive taxonomies for the community. In this paper, we delve into four essential modules of an LLM system, including an input module for receiving prompts, a language model trained on extensive corpora, a toolchain module for development and deployment, and an output module for exporting LLM-generated content. Based on this, we propose a comprehensive taxonomy, which systematically analyzes potential risks associated with each module of an LLM system and discusses the corresponding mitigation strategies. Furthermore, we review prevalent benchmarks, aiming to facilitate the risk assessment of LLM systems. We hope that this paper can help LLM participants embrace a systematic perspective to build their responsible LLM systems.

{{</citation>}}


### (54/94) Probing Structured Semantics Understanding and Generation of Language Models via Question Answering (Jinxin Liu et al., 2024)

{{<citation>}}

Jinxin Liu, Shulin Cao, Jiaxin Shi, Tingjian Zhang, Lei Hou, Juanzi Li. (2024)  
**Probing Structured Semantics Understanding and Generation of Language Models via Question Answering**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Question Answering  
[Paper Link](http://arxiv.org/abs/2401.05777v1)  

---


**ABSTRACT**  
Recent advancement in the capabilities of large language models (LLMs) has triggered a new surge in LLMs' evaluation. Most recent evaluation works tends to evaluate the comprehensive ability of LLMs over series of tasks. However, the deep structure understanding of natural language is rarely explored. In this work, we examine the ability of LLMs to deal with structured semantics on the tasks of question answering with the help of the human-constructed formal language. Specifically, we implement the inter-conversion of natural and formal language through in-context learning of LLMs to verify their ability to understand and generate the structured logical forms. Extensive experiments with models of different sizes and in different formal languages show that today's state-of-the-art LLMs' understanding of the logical forms can approach human level overall, but there still are plenty of room in generating correct logical forms, which suggest that it is more effective to use LLMs to generate more natural language training data to reinforce a small model than directly answering questions with LLMs. Moreover, our results also indicate that models exhibit considerable sensitivity to different formal languages. In general, the formal language with the lower the formalization level, i.e. the more similar it is to natural language, is more LLMs-friendly.

{{</citation>}}


### (55/94) A Shocking Amount of the Web is Machine Translated: Insights from Multi-Way Parallelism (Brian Thompson et al., 2024)

{{<citation>}}

Brian Thompson, Mehak Preet Dhaliwal, Peter Frisch, Tobias Domhan, Marcello Federico. (2024)  
**A Shocking Amount of the Web is Machine Translated: Insights from Multi-Way Parallelism**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Machine Translation  
[Paper Link](http://arxiv.org/abs/2401.05749v1)  

---


**ABSTRACT**  
We show that content on the web is often translated into many languages, and the low quality of these multi-way translations indicates they were likely created using Machine Translation (MT). Multi-way parallel, machine generated content not only dominates the translations in lower resource languages; it also constitutes a large fraction of the total web content in those languages. We also find evidence of a selection bias in the type of content which is translated into many languages, consistent with low quality English content being translated en masse into many lower resource languages, via MT. Our work raises serious concerns about training models such as multilingual large language models on both monolingual and bilingual data scraped from the web.

{{</citation>}}


### (56/94) Cross-modal Retrieval for Knowledge-based Visual Question Answering (Paul Lerner et al., 2024)

{{<citation>}}

Paul Lerner, Olivier Ferret, Camille Guinaudeau. (2024)  
**Cross-modal Retrieval for Knowledge-based Visual Question Answering**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-IR, cs.CL  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2401.05736v1)  

---


**ABSTRACT**  
Knowledge-based Visual Question Answering about Named Entities is a challenging task that requires retrieving information from a multimodal Knowledge Base. Named entities have diverse visual representations and are therefore difficult to recognize. We argue that cross-modal retrieval may help bridge the semantic gap between an entity and its depictions, and is foremost complementary with mono-modal retrieval. We provide empirical evidence through experiments with a multimodal dual encoder, namely CLIP, on the recent ViQuAE, InfoSeek, and Encyclopedic-VQA datasets. Additionally, we study three different strategies to fine-tune such a model: mono-modal, cross-modal, or joint training. Our method, which combines mono-and cross-modal retrieval, is competitive with billion-parameter models on the three datasets, while being conceptually simpler and computationally cheaper.

{{</citation>}}


### (57/94) CAT-LLM: Prompting Large Language Models with Text Style Definition for Chinese Article-style Transfer (Zhen Tao et al., 2024)

{{<citation>}}

Zhen Tao, Dinghao Xi, Zhiyu Li, Liumin Tang, Wei Xu. (2024)  
**CAT-LLM: Prompting Large Language Models with Text Style Definition for Chinese Article-style Transfer**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2401.05707v1)  

---


**ABSTRACT**  
Text style transfer is increasingly prominent in online entertainment and social media. However, existing research mainly concentrates on style transfer within individual English sentences, while ignoring the complexity of long Chinese texts, which limits the wider applicability of style transfer in digital media realm. To bridge this gap, we propose a Chinese Article-style Transfer framework (CAT-LLM), leveraging the capabilities of Large Language Models (LLMs). CAT-LLM incorporates a bespoke, pluggable Text Style Definition (TSD) module aimed at comprehensively analyzing text features in articles, prompting LLMs to efficiently transfer Chinese article-style. The TSD module integrates a series of machine learning algorithms to analyze article-style from both words and sentences levels, thereby aiding LLMs thoroughly grasp the target style without compromising the integrity of the original text. In addition, this module supports dynamic expansion of internal style trees, showcasing robust compatibility and allowing flexible optimization in subsequent research. Moreover, we select five Chinese articles with distinct styles and create five parallel datasets using ChatGPT, enhancing the models' performance evaluation accuracy and establishing a novel paradigm for evaluating subsequent research on article-style transfer. Extensive experimental results affirm that CAT-LLM outperforms current research in terms of transfer accuracy and content preservation, and has remarkable applicability to various types of LLMs.

{{</citation>}}


### (58/94) R-BI: Regularized Batched Inputs enhance Incremental Decoding Framework for Low-Latency Simultaneous Speech Translation (Jiaxin Guo et al., 2024)

{{<citation>}}

Jiaxin Guo, Zhanglin Wu, Zongyao Li, Hengchao Shang, Daimeng Wei, Xiaoyu Chen, Zhiqiang Rao, Shaojun Li, Hao Yang. (2024)  
**R-BI: Regularized Batched Inputs enhance Incremental Decoding Framework for Low-Latency Simultaneous Speech Translation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BLEU  
[Paper Link](http://arxiv.org/abs/2401.05700v1)  

---


**ABSTRACT**  
Incremental Decoding is an effective framework that enables the use of an offline model in a simultaneous setting without modifying the original model, making it suitable for Low-Latency Simultaneous Speech Translation. However, this framework may introduce errors when the system outputs from incomplete input. To reduce these output errors, several strategies such as Hold-$n$, LA-$n$, and SP-$n$ can be employed, but the hyper-parameter $n$ needs to be carefully selected for optimal performance. Moreover, these strategies are more suitable for end-to-end systems than cascade systems. In our paper, we propose a new adaptable and efficient policy named "Regularized Batched Inputs". Our method stands out by enhancing input diversity to mitigate output errors. We suggest particular regularization techniques for both end-to-end and cascade systems. We conducted experiments on IWSLT Simultaneous Speech Translation (SimulST) tasks, which demonstrate that our approach achieves low latency while maintaining no more than 2 BLEU points loss compared to offline systems. Furthermore, our SimulST systems attained several new state-of-the-art results in various language directions.

{{</citation>}}


### (59/94) Integrating Physician Diagnostic Logic into Large Language Models: Preference Learning from Process Feedback (Chengfeng Dou et al., 2024)

{{<citation>}}

Chengfeng Dou, Zhi Jin, Wenpin Jiao, Haiyan Zhao, Yongqiang Zhao, Zhenwei Tao. (2024)  
**Integrating Physician Diagnostic Logic into Large Language Models: Preference Learning from Process Feedback**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.05695v1)  

---


**ABSTRACT**  
The use of large language models in medical dialogue generation has garnered significant attention, with a focus on improving response quality and fluency. While previous studies have made progress in optimizing model performance for single-round medical Q&A tasks, there is a need to enhance the model's capability for multi-round conversations to avoid logical inconsistencies. To address this, we propose an approach called preference learning from process feedback~(PLPF), which integrates the doctor's diagnostic logic into LLMs. PLPF involves rule modeling, preference data generation, and preference alignment to train the model to adhere to the diagnostic process. Experimental results using Standardized Patient Testing show that PLPF enhances the diagnostic accuracy of the baseline model in medical conversations by 17.6%, outperforming traditional reinforcement learning from human feedback. Additionally, PLPF demonstrates effectiveness in both multi-round and single-round dialogue tasks, showcasing its potential for improving medical dialogue generation.

{{</citation>}}


### (60/94) UCorrect: An Unsupervised Framework for Automatic Speech Recognition Error Correction (Jiaxin Guo et al., 2024)

{{<citation>}}

Jiaxin Guo, Minghan Wang, Xiaosong Qiao, Daimeng Wei, Hengchao Shang, Zongyao Li, Zhengzhe Yu, Yinglu Li, Chang Su, Min Zhang, Shimin Tao, Hao Yang. (2024)  
**UCorrect: An Unsupervised Framework for Automatic Speech Recognition Error Correction**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: AI, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2401.05689v1)  

---


**ABSTRACT**  
Error correction techniques have been used to refine the output sentences from automatic speech recognition (ASR) models and achieve a lower word error rate (WER). Previous works usually adopt end-to-end models and has strong dependency on Pseudo Paired Data and Original Paired Data. But when only pre-training on Pseudo Paired Data, previous models have negative effect on correction. While fine-tuning on Original Paired Data, the source side data must be transcribed by a well-trained ASR model, which takes a lot of time and not universal. In this paper, we propose UCorrect, an unsupervised Detector-Generator-Selector framework for ASR Error Correction. UCorrect has no dependency on the training data mentioned before. The whole procedure is first to detect whether the character is erroneous, then to generate some candidate characters and finally to select the most confident one to replace the error character. Experiments on the public AISHELL-1 dataset and WenetSpeech dataset show the effectiveness of UCorrect for ASR error correction: 1) it achieves significant WER reduction, achieves 6.83\% even without fine-tuning and 14.29\% after fine-tuning; 2) it outperforms the popular NAR correction models by a large margin with a competitive low latency; and 3) it is an universal method, as it reduces all WERs of the ASR model with different decoding strategies and reduces all WERs of ASR models trained on different scale datasets.

{{</citation>}}


### (61/94) ConcEPT: Concept-Enhanced Pre-Training for Language Models (Xintao Wang et al., 2024)

{{<citation>}}

Xintao Wang, Zhouhong Gu, Jiaqing Liang, Dakuan Lu, Yanghua Xiao, Wei Wang. (2024)  
**ConcEPT: Concept-Enhanced Pre-Training for Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.05669v1)  

---


**ABSTRACT**  
Pre-trained language models (PLMs) have been prevailing in state-of-the-art methods for natural language processing, and knowledge-enhanced PLMs are further proposed to promote model performance in knowledge-intensive tasks. However, conceptual knowledge, one essential kind of knowledge for human cognition, still remains understudied in this line of research. This limits PLMs' performance in scenarios requiring human-like cognition, such as understanding long-tail entities with concepts. In this paper, we propose ConcEPT, which stands for Concept-Enhanced Pre-Training for language models, to infuse conceptual knowledge into PLMs. ConcEPT exploits external taxonomies with entity concept prediction, a novel pre-training objective to predict the concepts of entities mentioned in the pre-training contexts. Unlike previous concept-enhanced methods, ConcEPT can be readily adapted to various downstream applications without entity linking or concept mapping. Results of extensive experiments show the effectiveness of ConcEPT in four tasks such as entity typing, which validates that our model gains improved conceptual knowledge with concept-enhanced pre-training.

{{</citation>}}


### (62/94) On Detecting Cherry-picking in News Coverage Using Large Language Models (Israa Jaradat et al., 2024)

{{<citation>}}

Israa Jaradat, Haiqi Zhang, Chengkai Li. (2024)  
**On Detecting Cherry-picking in News Coverage Using Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.05650v1)  

---


**ABSTRACT**  
Cherry-picking refers to the deliberate selection of evidence or facts that favor a particular viewpoint while ignoring or distorting evidence that supports an opposing perspective. Manually identifying instances of cherry-picked statements in news stories can be challenging, particularly when the opposing viewpoint's story is absent. This study introduces Cherry, an innovative approach for automatically detecting cherry-picked statements in news articles by finding missing important statements in the target news story. Cherry utilizes the analysis of news coverage from multiple sources to identify instances of cherry-picking. Our approach relies on language models that consider contextual information from other news sources to classify statements based on their importance to the event covered in the target news story. Furthermore, this research introduces a novel dataset specifically designed for cherry-picking detection, which was used to train and evaluate the performance of the models. Our best performing model achieves an F-1 score of about %89 in detecting important statements when tested on unseen set of news stories. Moreover, results show the importance incorporating external knowledge from alternative unbiased narratives when assessing a statement's importance.

{{</citation>}}


### (63/94) Natural Language Processing for Dialects of a Language: A Survey (Aditya Joshi et al., 2024)

{{<citation>}}

Aditya Joshi, Raj Dabre, Diptesh Kanojia, Zhuang Li, Haolan Zhan, Gholamreza Haffari, Doris Dippold. (2024)  
**Natural Language Processing for Dialects of a Language: A Survey**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP, NLU, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2401.05632v1)  

---


**ABSTRACT**  
State-of-the-art natural language processing (NLP) models are trained on massive training corpora, and report a superlative performance on evaluation datasets. This survey delves into an important attribute of these datasets: the dialect of a language. Motivated by the performance degradation of NLP models for dialectic datasets and its implications for the equity of language technologies, we survey past research in NLP for dialects in terms of datasets, and approaches. We describe a wide range of NLP tasks in terms of two categories: natural language understanding (NLU) (for tasks such as dialect classification, sentiment analysis, parsing, and NLU benchmarks) and natural language generation (NLG) (for summarisation, machine translation, and dialogue systems). The survey is also broad in its coverage of languages which include English, Arabic, German among others. We observe that past work in NLP concerning dialects goes deeper than mere dialect classification, and . This includes early approaches that used sentence transduction that lead to the recent approaches that integrate hypernetworks into LoRA. We expect that this survey will be useful to NLP researchers interested in building equitable language technologies by rethinking LLM benchmarks and model architectures.

{{</citation>}}


### (64/94) The Benefits of a Concise Chain of Thought on Problem-Solving in Large Language Models (Matthew Renze et al., 2024)

{{<citation>}}

Matthew Renze, Erhan Guven. (2024)  
**The Benefits of a Concise Chain of Thought on Problem-Solving in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, GPT, GPT-3.5, GPT-4, Language Model, QA  
[Paper Link](http://arxiv.org/abs/2401.05618v1)  

---


**ABSTRACT**  
In this paper, we introduce Concise Chain-of-Thought (CCoT) prompting. We compared standard CoT and CCoT prompts to see how conciseness impacts response length and correct-answer accuracy. We evaluated this using GPT-3.5 and GPT-4 with a multiple-choice question-and-answer (MCQA) benchmark. CCoT reduced average response length by 48.70% for both GPT-3.5 and GPT-4 while having a negligible impact on problem-solving performance. However, on math problems, GPT-3.5 with CCoT incurs a performance penalty of 27.69%. Overall, CCoT leads to an average per-token cost reduction of 22.67%. These results have practical implications for AI systems engineers using LLMs to solve real-world problems with CoT prompt-engineering techniques. In addition, these results provide more general insight for AI researchers studying the emergent behavior of step-by-step reasoning in LLMs.

{{</citation>}}


### (65/94) Scaling Laws for Forgetting When Fine-Tuning Large Language Models (Damjan Kalajdzievski, 2024)

{{<citation>}}

Damjan Kalajdzievski. (2024)  
**Scaling Laws for Forgetting When Fine-Tuning Large Language Models**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.05605v1)  

---


**ABSTRACT**  
We study and quantify the problem of forgetting when fine-tuning pre-trained large language models (LLMs) on a downstream task. We find that parameter-efficient fine-tuning (PEFT) strategies, such as Low-Rank Adapters (LoRA), still suffer from catastrophic forgetting. In particular, we identify a strong inverse linear relationship between the fine-tuning performance and the amount of forgetting when fine-tuning LLMs with LoRA. We further obtain precise scaling laws that show forgetting increases as a shifted power law in the number of parameters fine-tuned and the number of update steps. We also examine the impact of forgetting on knowledge, reasoning, and the safety guardrails trained into Llama 2 7B chat. Our study suggests that forgetting cannot be avoided through early stopping or by varying the number of parameters fine-tuned. We believe this opens up an important safety-critical direction for future research to evaluate and develop fine-tuning schemes which mitigate forgetting

{{</citation>}}


### (66/94) REBUS: A Robust Evaluation Benchmark of Understanding Symbols (Andrew Gritsevskiy et al., 2024)

{{<citation>}}

Andrew Gritsevskiy, Arjun Panickssery, Aaron Kirtland, Derik Kauffman, Hans Gundlach, Irina Gritsevskaya, Joe Cavanagh, Jonathan Chiang, Lydia La Roux, Michelle Hung. (2024)  
**REBUS: A Robust Evaluation Benchmark of Understanding Symbols**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CV, cs-CY, cs.CL  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2401.05604v1)  

---


**ABSTRACT**  
We propose a new benchmark evaluating the performance of multimodal large language models on rebus puzzles. The dataset covers 333 original examples of image-based wordplay, cluing 13 categories such as movies, composers, major cities, and food. To achieve good performance on the benchmark of identifying the clued word or phrase, models must combine image recognition and string manipulation with hypothesis testing, multi-step reasoning, and an understanding of human cognition, making for a complex, multimodal evaluation of capabilities. We find that proprietary models such as GPT-4V and Gemini Pro significantly outperform all other tested models. However, even the best model has a final accuracy of just 24%, highlighting the need for substantial improvements in reasoning. Further, models rarely understand all parts of a puzzle, and are almost always incapable of retroactively explaining the correct answer. Our benchmark can therefore be used to identify major shortcomings in the knowledge and reasoning of multimodal large language models.

{{</citation>}}


### (67/94) POMP: Probability-driven Meta-graph Prompter for LLMs in Low-resource Unsupervised Neural Machine Translation (Shilong Pan et al., 2024)

{{<citation>}}

Shilong Pan, Zhiliang Tian, Liang Ding, Zhen Huang, Zhihua Wen, Dongsheng Li. (2024)  
**POMP: Probability-driven Meta-graph Prompter for LLMs in Low-resource Unsupervised Neural Machine Translation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BLEU, Language Model, Machine Translation  
[Paper Link](http://arxiv.org/abs/2401.05596v1)  

---


**ABSTRACT**  
Low-resource languages (LRLs) face challenges in supervised neural machine translation due to limited parallel data, prompting research into unsupervised methods. Unsupervised neural machine translation (UNMT) methods, including back-translation, transfer learning, and pivot-based translation, offer practical solutions for LRL translation, but they are hindered by issues like synthetic data noise, language bias, and error propagation, which can potentially be mitigated by Large Language Models (LLMs). LLMs have advanced NMT with in-context learning (ICL) and supervised fine-tuning methods, but insufficient training data results in poor performance in LRLs. We argue that LLMs can mitigate the linguistic noise with auxiliary languages to improve translations in LRLs. In this paper, we propose Probability-driven Meta-graph Prompter (POMP), a novel approach employing a dynamic, sampling-based graph of multiple auxiliary languages to enhance LLMs' translation capabilities for LRLs. POMP involves constructing a directed acyclic meta-graph for each source language, from which we dynamically sample multiple paths to prompt LLMs to mitigate the linguistic noise and improve translations during training. We use the BLEURT metric to evaluate the translations and back-propagate rewards, estimated by scores, to update the probabilities of auxiliary languages in the paths. Our experiments show significant improvements in the translation quality of three LRLs, demonstrating the effectiveness of our approach.

{{</citation>}}


## cs.AI (4)



### (68/94) Secrets of RLHF in Large Language Models Part II: Reward Modeling (Binghai Wang et al., 2024)

{{<citation>}}

Binghai Wang, Rui Zheng, Lu Chen, Yan Liu, Shihan Dou, Caishuang Huang, Wei Shen, Senjie Jin, Enyu Zhou, Chenyu Shi, Songyang Gao, Nuo Xu, Yuhao Zhou, Xiaoran Fan, Zhiheng Xi, Jun Zhao, Xiao Wang, Tao Ji, Hang Yan, Lixing Shen, Zhan Chen, Tao Gui, Qi Zhang, Xipeng Qiu, Xuanjing Huang, Zuxuan Wu, Yu-Gang Jiang. (2024)  
**Secrets of RLHF in Large Language Models Part II: Reward Modeling**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Language Model, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.06080v1)  

---


**ABSTRACT**  
Reinforcement Learning from Human Feedback (RLHF) has become a crucial technology for aligning language models with human values and intentions, enabling models to produce more helpful and harmless responses. Reward models are trained as proxies for human preferences to drive reinforcement learning optimization. While reward models are often considered central to achieving high performance, they face the following challenges in practical applications: (1) Incorrect and ambiguous preference pairs in the dataset may hinder the reward model from accurately capturing human intent. (2) Reward models trained on data from a specific distribution often struggle to generalize to examples outside that distribution and are not suitable for iterative RLHF training.   In this report, we attempt to address these two issues. (1) From a data perspective, we propose a method to measure the strength of preferences within the data, based on a voting mechanism of multiple reward models. Experimental results confirm that data with varying preference strengths have different impacts on reward model performance. We introduce a series of novel methods to mitigate the influence of incorrect and ambiguous preferences in the dataset and fully leverage high-quality preference data. (2) From an algorithmic standpoint, we introduce contrastive learning to enhance the ability of reward models to distinguish between chosen and rejected responses, thereby improving model generalization. Furthermore, we employ meta-learning to enable the reward model to maintain the ability to differentiate subtle differences in out-of-distribution samples, and this approach can be utilized for iterative RLHF optimization.

{{</citation>}}


### (69/94) Chain of History: Learning and Forecasting with LLMs for Temporal Knowledge Graph Completion (Ruilin Luo et al., 2024)

{{<citation>}}

Ruilin Luo, Tianle Gu, Haoling Li, Junzhe Li, Zicheng Lin, Jiayi Li, Yujiu Yang. (2024)  
**Chain of History: Learning and Forecasting with LLMs for Temporal Knowledge Graph Completion**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2401.06072v1)  

---


**ABSTRACT**  
Temporal Knowledge Graph Completion (TKGC) is a challenging task of predicting missing event links at future timestamps by leveraging established temporal structural knowledge. Given the formidable generative capabilities inherent in LLMs (LLMs), this paper proposes a novel approach to conceptualize temporal link prediction as an event generation task within the context of a historical event chain. We employ efficient fine-tuning methods to make LLMs adapt to specific graph textual information and patterns discovered in temporal timelines. Furthermore, we introduce structure-based historical data augmentation and the integration of reverse knowledge to emphasize LLMs' awareness of structural information, thereby enhancing their reasoning capabilities. We conduct thorough experiments on multiple widely used datasets and find that our fine-tuned model outperforms existing embedding-based models on multiple metrics, achieving SOTA results. We also carry out sufficient ablation experiments to explore the key influencing factors when LLMs perform structured temporal knowledge inference tasks.

{{</citation>}}


### (70/94) Machine Learning Insides OptVerse AI Solver: Design Principles and Applications (Xijun Li et al., 2024)

{{<citation>}}

Xijun Li, Fangzhou Zhu, Hui-Ling Zhen, Weilin Luo, Meng Lu, Yimin Huang, Zhenan Fan, Zirui Zhou, Yufei Kuang, Zhihai Wang, Zijie Geng, Yang Li, Haoyang Liu, Zhiwu An, Muming Yang, Jianshu Li, Jie Wang, Junchi Yan, Defeng Sun, Tao Zhong, Yong Zhang, Jia Zeng, Mingxuan Yuan, Jianye Hao, Jun Yao, Kun Mao. (2024)  
**Machine Learning Insides OptVerse AI Solver: Design Principles and Applications**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.05960v1)  

---


**ABSTRACT**  
In an era of digital ubiquity, efficient resource management and decision-making are paramount across numerous industries. To this end, we present a comprehensive study on the integration of machine learning (ML) techniques into Huawei Cloud's OptVerse AI Solver, which aims to mitigate the scarcity of real-world mathematical programming instances, and to surpass the capabilities of traditional optimization techniques. We showcase our methods for generating complex SAT and MILP instances utilizing generative models that mirror multifaceted structures of real-world problem. Furthermore, we introduce a training framework leveraging augmentation policies to maintain solvers' utility in dynamic environments. Besides the data generation and augmentation, our proposed approaches also include novel ML-driven policies for personalized solver strategies, with an emphasis on applications like graph convolutional networks for initial basis selection and reinforcement learning for advanced presolving and cut selection. Additionally, we detail the incorporation of state-of-the-art parameter tuning algorithms which markedly elevate solver performance. Compared with traditional solvers such as Gurobi and SCIP, our ML-augmented OptVerse AI Solver demonstrates superior speed and precision across both established benchmarks and real-world scenarios, reinforcing the practical imperative and effectiveness of machine learning techniques in mathematical programming solvers.

{{</citation>}}


### (71/94) Towards Conversational Diagnostic AI (Tao Tu et al., 2024)

{{<citation>}}

Tao Tu, Anil Palepu, Mike Schaekermann, Khaled Saab, Jan Freyberg, Ryutaro Tanno, Amy Wang, Brenna Li, Mohamed Amin, Nenad Tomasev, Shekoofeh Azizi, Karan Singhal, Yong Cheng, Le Hou, Albert Webson, Kavita Kulkarni, S Sara Mahdavi, Christopher Semturs, Juraj Gottweis, Joelle Barral, Katherine Chou, Greg S Corrado, Yossi Matias, Alan Karthikesalingam, Vivek Natarajan. (2024)  
**Towards Conversational Diagnostic AI**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-LG, cs.AI  
Keywords: AI, Clinical, Language Model  
[Paper Link](http://arxiv.org/abs/2401.05654v1)  

---


**ABSTRACT**  
At the heart of medicine lies the physician-patient dialogue, where skillful history-taking paves the way for accurate diagnosis, effective management, and enduring trust. Artificial Intelligence (AI) systems capable of diagnostic dialogue could increase accessibility, consistency, and quality of care. However, approximating clinicians' expertise is an outstanding grand challenge. Here, we introduce AMIE (Articulate Medical Intelligence Explorer), a Large Language Model (LLM) based AI system optimized for diagnostic dialogue.   AMIE uses a novel self-play based simulated environment with automated feedback mechanisms for scaling learning across diverse disease conditions, specialties, and contexts. We designed a framework for evaluating clinically-meaningful axes of performance including history-taking, diagnostic accuracy, management reasoning, communication skills, and empathy. We compared AMIE's performance to that of primary care physicians (PCPs) in a randomized, double-blind crossover study of text-based consultations with validated patient actors in the style of an Objective Structured Clinical Examination (OSCE). The study included 149 case scenarios from clinical providers in Canada, the UK, and India, 20 PCPs for comparison with AMIE, and evaluations by specialist physicians and patient actors. AMIE demonstrated greater diagnostic accuracy and superior performance on 28 of 32 axes according to specialist physicians and 24 of 26 axes according to patient actors. Our research has several limitations and should be interpreted with appropriate caution. Clinicians were limited to unfamiliar synchronous text-chat which permits large-scale LLM-patient interactions but is not representative of usual clinical practice. While further research is required before AMIE could be translated to real-world settings, the results represent a milestone towards conversational diagnostic AI.

{{</citation>}}


## cs.SI (1)



### (72/94) On the Power of Graph Neural Networks and Feature Augmentation Strategies to Classify Social Networks (Walid Guettala et al., 2024)

{{<citation>}}

Walid Guettala, László Gulyás. (2024)  
**On the Power of Graph Neural Networks and Feature Augmentation Strategies to Classify Social Networks**  

---
Primary Category: cs.SI  
Categories: cs-AI, cs-LG, cs-SI, cs.SI  
Keywords: Augmentation, GNN, Graph Neural Network, Graph Neural Networks, Social Network  
[Paper Link](http://arxiv.org/abs/2401.06048v1)  

---


**ABSTRACT**  
This paper studies four Graph Neural Network architectures (GNNs) for a graph classification task on a synthetic dataset created using classic generative models of Network Science. Since the synthetic networks do not contain (node or edge) features, five different augmentation strategies (artificial feature types) are applied to nodes. All combinations of the 4 GNNs (GCN with Hierarchical and Global aggregation, GIN and GATv2) and the 5 feature types (constant 1, noise, degree, normalized degree and ID -- a vector of the number of cycles of various lengths) are studied and their performances compared as a function of the hidden dimension of artificial neural networks used in the GNNs. The generalisation ability of these models is also analysed using a second synthetic network dataset (containing networks of different sizes).Our results point towards the balanced importance of the computational power of the GNN architecture and the the information level provided by the artificial features. GNN architectures with higher computational power, like GIN and GATv2, perform well for most augmentation strategies. On the other hand, artificial features with higher information content, like ID or degree, not only consistently outperform other augmentation strategies, but can also help GNN architectures with lower computational power to achieve good performance.

{{</citation>}}


## cs.HC (6)



### (73/94) Boosting Mixed-Initiative Co-Creativity in Game Design: A Tutorial (Solange Margarido et al., 2024)

{{<citation>}}

Solange Margarido, Licínio Roque, Penousal Machado, Pedro Martins. (2024)  
**Boosting Mixed-Initiative Co-Creativity in Game Design: A Tutorial**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.05999v1)  

---


**ABSTRACT**  
In recent years, there has been a growing application of mixed-initiative co-creative approaches in the creation of video games. The rapid advances in the capabilities of artificial intelligence (AI) systems further propel creative collaboration between humans and computational agents. In this tutorial, we present guidelines for researchers and practitioners to develop game design tools with a high degree of mixed-initiative co-creativity (MI-CCy). We begin by reviewing a selection of current works that will serve as case studies and categorize them by the type of game content they address. We introduce the MI-CCy Quantifier, a framework that can be used by researchers and developers to assess co-creative tools on their level of MI-CCy through a visual scheme of quantifiable criteria scales. We demonstrate the usage of the MI-CCy Quantifier by applying it to the selected works. This analysis enabled us to discern prevalent patterns within these tools, as well as features that contribute to a higher level of MI-CCy. We highlight current gaps in MI-CCy approaches within game design, which we propose as pivotal aspects to tackle in the development of forthcoming approaches.

{{</citation>}}


### (74/94) Decoding AI's Nudge: A Unified Framework to Predict Human Behavior in AI-assisted Decision Making (Zhuoyan Li et al., 2024)

{{<citation>}}

Zhuoyan Li, Zhuoran Lu, Ming Yin. (2024)  
**Decoding AI's Nudge: A Unified Framework to Predict Human Behavior in AI-assisted Decision Making**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.05840v1)  

---


**ABSTRACT**  
With the rapid development of AI-based decision aids, different forms of AI assistance have been increasingly integrated into the human decision making processes. To best support humans in decision making, it is essential to quantitatively understand how diverse forms of AI assistance influence humans' decision making behavior. To this end, much of the current research focuses on the end-to-end prediction of human behavior using ``black-box'' models, often lacking interpretations of the nuanced ways in which AI assistance impacts the human decision making process. Meanwhile, methods that prioritize the interpretability of human behavior predictions are often tailored for one specific form of AI assistance, making adaptations to other forms of assistance difficult. In this paper, we propose a computational framework that can provide an interpretable characterization of the influence of different forms of AI assistance on decision makers in AI-assisted decision making. By conceptualizing AI assistance as the ``{\em nudge}'' in human decision making processes, our approach centers around modelling how different forms of AI assistance modify humans' strategy in weighing different information in making their decisions. Evaluations on behavior data collected from real human decision makers show that the proposed framework outperforms various baselines in accurately predicting human behavior in AI-assisted decision making. Based on the proposed framework, we further provide insights into how individuals with different cognitive styles are nudged by AI assistance differently.

{{</citation>}}


### (75/94) How to write a CHI paper (asking for a friend) (Raquel Robinson et al., 2024)

{{<citation>}}

Raquel Robinson, Alberto Alvarez, Elisa Mekler. (2024)  
**How to write a CHI paper (asking for a friend)**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.05818v1)  

---


**ABSTRACT**  
Writing and genre conventions are extant to any scientific community, and CHI is no different. In this paper, we present the early phases of an AI tool we created called KITSUNE, which supports authors in placing their work into the format of a CHI paper, taking into account many conventions that are ever-present in CHI papers. We describe the development of the tool with the intent to promote discussion around how writing conventions are upheld and unquestioned by the CHI community, and how this translates to the work produced. In addition, we bring up questions surrounding how the introduction of LLMs into academic writing fundamentally change how conventions will be upheld now and in the future

{{</citation>}}


### (76/94) DrawTalking: Building Interactive Worlds by Sketching and Speaking (Karl Toby Rosenberg et al., 2024)

{{<citation>}}

Karl Toby Rosenberg, Rubaiat Habib Kazi, Li-Yi Wei, Haijun Xia, Ken Perlin. (2024)  
**DrawTalking: Building Interactive Worlds by Sketching and Speaking**  

---
Primary Category: cs.HC  
Categories: H-5-2; D-2-2; I-2-7; D-1-7; H-5-1, cs-AI, cs-CL, cs-GR, cs-HC, cs.HC  
Keywords: Sketch  
[Paper Link](http://arxiv.org/abs/2401.05631v1)  

---


**ABSTRACT**  
We introduce an interactive approach, DrawTalking, in which the user builds interactive worlds by sketching and speaking. It emphasizes user control and flexibility, and gives programming-like capability without code. We implemented it on the iPad. An open-ended study shows the mechanics resonate and are applicable to many creative-exploratory use cases. We hope to inspire and inform research in future natural user-centered interfaces.

{{</citation>}}


### (77/94) Designing for Appropriate Reliance:Designing for Appropriate Reliance: The Roles of AI Uncertainty Presentation, Initial User Decision, and User Demographics in AI-Assisted Decision-Making (Shiye Cao et al., 2024)

{{<citation>}}

Shiye Cao, Anqi Liu, Chien-Ming Huang. (2024)  
**Designing for Appropriate Reliance:Designing for Appropriate Reliance: The Roles of AI Uncertainty Presentation, Initial User Decision, and User Demographics in AI-Assisted Decision-Making**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.05612v1)  

---


**ABSTRACT**  
Appropriate reliance is critical to achieving synergistic human-AI collaboration. For instance, when users over-rely on AI assistance, their human-AI team performance is bounded by the model's capability. This work studies how the presentation of model uncertainty may steer users' decision-making toward fostering appropriate reliance. Our results demonstrate that showing the calibrated model uncertainty alone is inadequate. Rather, calibrating model uncertainty and presenting it in a frequency format allow users to adjust their reliance accordingly and help reduce the effect of confirmation bias on their decisions. Furthermore, the critical nature of our skin cancer screening task skews participants' judgment, causing their reliance to vary depending on their initial decision. Additionally, step-wise multiple regression analyses revealed how user demographics such as age and familiarity with probability and statistics influence human-AI collaborative decision-making. We discuss the potential for model uncertainty presentation, initial user decision, and user demographics to be incorporated in designing personalized AI aids for appropriate reliance.

{{</citation>}}


### (78/94) Exploring How FoMO, Social Media Addiction, and Subjective Norms Influence Personal Moderation Configurations (Shagun Jhaver, 2024)

{{<citation>}}

Shagun Jhaver. (2024)  
**Exploring How FoMO, Social Media Addiction, and Subjective Norms Influence Personal Moderation Configurations**  

---
Primary Category: cs.HC  
Categories: cs-CY, cs-HC, cs.HC  
Keywords: Social Media  
[Paper Link](http://arxiv.org/abs/2401.05603v1)  

---


**ABSTRACT**  
Personal moderation tools on social media platforms allow users to control their feeds by configuring the acceptable toxicity thresholds for their feed content or muting inappropriate accounts. This research examines how the end-user configuration of these tools is shaped by four critical psychosocial factors - fear of missing out (FoMO), social media addiction, subjective norms, and trust in moderation systems. Findings from a nationally representative sample of 1,061 participants show that FoMO and social media addiction make Facebook users more vulnerable to content-based harms by reducing their likelihood of adopting personal moderation tools to hide inappropriate posts. In contrast, descriptive and injunctive norms positively influence the use of these tools. Further, trust in Facebook's moderation systems also significantly affects users' engagement with personal moderation. This analysis highlights qualitatively different pathways through which FoMO and social media addiction make affected users disproportionately unsafe and offers design and policy solutions to address this challenge.

{{</citation>}}


## cs.MM (1)



### (79/94) A Multi-Embedding Convergence Network on Siamese Architecture for Fake Reviews (Sankarshan Dasgupta et al., 2024)

{{<citation>}}

Sankarshan Dasgupta, James Buckley. (2024)  
**A Multi-Embedding Convergence Network on Siamese Architecture for Fake Reviews**  

---
Primary Category: cs.MM  
Categories: cs-MM, cs.MM  
Keywords: BERT, Embedding, LSTM  
[Paper Link](http://arxiv.org/abs/2401.05995v1)  

---


**ABSTRACT**  
In this new digital era, accessibility to real-world events is moving towards web-based modules. This is mostly visible on e-commerce websites where there is limited availability of physical verification. With this unforeseen development, we depend on the verification in the virtual world to influence our decisions. One of the decision making process is deeply based on review reading. Reviews play an important part in this transactional process. And seeking a real review can be very tenuous work for the user. On the other hand, fake review heavily impacts these transaction records of a product. The article presents an implementation of a Siamese network for detecting fake reviews. The fake reviews dataset, consisting of 40K reviews, preprocessed with different techniques. The cleaned data is passed through embeddings generated by MiniLM BERT for contextual relationship and Word2Vec for semantic relationship to form vectors. Further, the embeddings are trained in a Siamese network with LSTM layers connected to fuzzy logic for decision-making. The results show that fake reviews can be detected with high accuracy on a siamese network for prediction and verification.

{{</citation>}}


## stat.ML (1)



### (80/94) A tree-based varying coefficient model (Henning Zakrisson et al., 2024)

{{<citation>}}

Henning Zakrisson, Mathias Lindholm. (2024)  
**A tree-based varying coefficient model**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-ML, stat.ML  
Keywords: GLM  
[Paper Link](http://arxiv.org/abs/2401.05982v1)  

---


**ABSTRACT**  
The paper introduces a tree-based varying coefficient model (VCM) where the varying coefficients are modelled using the cyclic gradient boosting machine (CGBM) from Delong et al. (2023). Modelling the coefficient functions using a CGBM allows for dimension-wise early stopping and feature importance scores. The dimension-wise early stopping not only reduces the risk of dimension-specific overfitting, but also reveals differences in model complexity across dimensions. The use of feature importance scores allows for simple feature selection and easy model interpretation. The model is evaluated on the same simulated and real data examples as those used in Richman and W\"uthrich (2023), and the results show that it produces results in terms of out of sample loss that are comparable to those of their neural network-based VCM called LocalGLMnet.

{{</citation>}}


## cs.SE (5)



### (81/94) Mutation-based Consistency Testing for Evaluating the Code Understanding Capability of LLMs (Ziyu Li et al., 2024)

{{<citation>}}

Ziyu Li, Donghwan Shin. (2024)  
**Mutation-based Consistency Testing for Evaluating the Code Understanding Capability of LLMs**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: GPT, GPT-3.5, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2401.05940v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have shown remarkable capabilities in processing both natural and programming languages, which have enabled various applications in software engineering, such as requirement engineering, code generation, and software testing. However, existing code generation benchmarks do not necessarily assess the code understanding performance of LLMs, especially for the subtle inconsistencies that may arise between code and its semantics described in natural language.   In this paper, we propose a novel method to systematically assess the code understanding performance of LLMs, particularly focusing on subtle differences between code and its descriptions, by introducing code mutations to existing code generation datasets. Code mutations are small changes that alter the semantics of the original code, creating a mismatch with the natural language description. We apply different types of code mutations, such as operator replacement and statement deletion, to generate inconsistent code-description pairs. We then use these pairs to test the ability of LLMs to correctly detect the inconsistencies.   We propose a new LLM testing method, called Mutation-based Consistency Testing (MCT), and conduct a case study on the two popular LLMs, GPT-3.5 and GPT-4, using the state-of-the-art code generation benchmark, HumanEval-X, which consists of six programming languages (Python, C++, Java, Go, JavaScript, and Rust). We compare the performance of the LLMs across different types of code mutations and programming languages and analyze the results. We find that the LLMs show significant variation in their code understanding performance and that they have different strengths and weaknesses depending on the mutation type and language.

{{</citation>}}


### (82/94) Using Large Language Models for Commit Message Generation: A Preliminary Study (Linghao Zhang et al., 2024)

{{<citation>}}

Linghao Zhang, Jingshu Zhao, Chong Wang, Peng Liang. (2024)  
**Using Large Language Models for Commit Message Generation: A Preliminary Study**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: BLEU, ChatGPT, GPT, Language Model, Rouge, Rouge-L  
[Paper Link](http://arxiv.org/abs/2401.05926v1)  

---


**ABSTRACT**  
A commit message is a textual description of the code changes in a commit, which is a key part of the Git version control system (VCS). It captures the essence of software updating. Therefore, it can help developers understand code evolution and facilitate efficient collaboration between developers. However, it is time-consuming and labor-intensive to write good and valuable commit messages. Some researchers have conducted extensive studies on the automatic generation of commit messages and proposed several methods for this purpose, such as generation-based and retrieval-based models. However, seldom studies explored whether large language models (LLMs) can be effectively used for the automatic generation of commit messages. To this end, this paper designed and conducted a series of experiments to comprehensively evaluate the performance of popular open-source and closed-source LLMs, i.e., Llama 2 and ChatGPT, in commit message generation. The results indicate that considering the BLEU and Rouge-L metrics, LLMs surpass existing methods in certain indicators but lag behind in others. After human evaluations, however, LLMs show a distinct advantage over all these existing methods. Especially, in 78% of the 366 samples, the commit messages generated by LLMs were evaluated by humans as the best. This work not only reveals the promising potential of using LLMs to generate commit messages, but also explores the limitations of commonly used metrics in evaluating the quality of automatically generated commit messages.

{{</citation>}}


### (83/94) Seven Failure Points When Engineering a Retrieval Augmented Generation System (Scott Barnett et al., 2024)

{{<citation>}}

Scott Barnett, Stefanus Kurniawan, Srikanth Thudumu, Zach Brannelly, Mohamed Abdelrazek. (2024)  
**Seven Failure Points When Engineering a Retrieval Augmented Generation System**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2401.05856v1)  

---


**ABSTRACT**  
Software engineers are increasingly adding semantic search capabilities to applications using a strategy known as Retrieval Augmented Generation (RAG). A RAG system involves finding documents that semantically match a query and then passing the documents to a large language model (LLM) such as ChatGPT to extract the right answer using an LLM. RAG systems aim to: a) reduce the problem of hallucinated responses from LLMs, b) link sources/references to generated responses, and c) remove the need for annotating documents with meta-data. However, RAG systems suffer from limitations inherent to information retrieval systems and from reliance on LLMs. In this paper, we present an experience report on the failure points of RAG systems from three case studies from separate domains: research, education, and biomedical. We share the lessons learned and present 7 failure points to consider when designing a RAG system. The two key takeaways arising from our work are: 1) validation of a RAG system is only feasible during operation, and 2) the robustness of a RAG system evolves rather than designed in at the start. We conclude with a list of potential research directions on RAG systems for the software engineering community.

{{</citation>}}


### (84/94) Development in times of hype: How freelancers explore Generative AI? (Mateusz Dolata et al., 2024)

{{<citation>}}

Mateusz Dolata, Norbert Lange, Gerhard Schwabe. (2024)  
**Development in times of hype: How freelancers explore Generative AI?**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2401.05790v1)  

---


**ABSTRACT**  
The rise of generative AI has led many companies to hire freelancers to harness its potential. However, this technology presents unique challenges to developers who have not previously engaged with it. Freelancers may find these challenges daunting due to the absence of organizational support and their reliance on positive client feedback. In a study involving 52 freelance developers, we identified multiple challenges associated with developing solutions based on generative AI. Freelancers often struggle with aspects they perceive as unique to generative AI such as unpredictability of its output, the occurrence of hallucinations, and the inconsistent effort required due to trial-and-error prompting cycles. Further, the limitations of specific frameworks, such as token limits and long response times, add to the complexity. Hype-related issues, such as inflated client expectations and a rapidly evolving technological ecosystem, further exacerbate the difficulties. To address these issues, we propose Software Engineering for Generative AI (SE4GenAI) and Hype-Induced Software Engineering (HypeSE) as areas where the software engineering community can provide effective guidance. This support is essential for freelancers working with generative AI and other emerging technologies.

{{</citation>}}


### (85/94) Cross-Inlining Binary Function Similarity Detection (Ang Jia et al., 2024)

{{<citation>}}

Ang Jia, Ming Fan, Xi Xu, Wuxia Jin, Haijun Wang, Ting Liu. (2024)  
**Cross-Inlining Binary Function Similarity Detection**  

---
Primary Category: cs.SE  
Categories: cs-CR, cs-SE, cs.SE  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2401.05739v1)  

---


**ABSTRACT**  
Binary function similarity detection plays an important role in a wide range of security applications. Existing works usually assume that the query function and target function share equal semantics and compare their full semantics to obtain the similarity. However, we find that the function mapping is more complex, especially when function inlining happens.   In this paper, we will systematically investigate cross-inlining binary function similarity detection. We first construct a cross-inlining dataset by compiling 51 projects using 9 compilers, with 4 optimizations, to 6 architectures, with 2 inlining flags, which results in two datasets both with 216 combinations. Then we construct the cross-inlining function mappings by linking the common source functions in these two datasets. Through analysis of this dataset, we find that three cross-inlining patterns widely exist while existing work suffers when detecting cross-inlining binary function similarity. Next, we propose a pattern-based model named CI-Detector for cross-inlining matching. CI-Detector uses the attributed CFG to represent the semantics of binary functions and GNN to embed binary functions into vectors. CI-Detector respectively trains a model for these three cross-inlining patterns. Finally, the testing pairs are input to these three models and all the produced similarities are aggregated to produce the final similarity. We conduct several experiments to evaluate CI-Detector. Results show that CI-Detector can detect cross-inlining pairs with a precision of 81% and a recall of 97%, which exceeds all state-of-the-art works.

{{</citation>}}


## cs.NE (1)



### (86/94) Time Series Forecasting of HIV/AIDS in the Philippines Using Deep Learning: Does COVID-19 Epidemic Matter? (Sales G. Aribe Jr. et al., 2024)

{{<citation>}}

Sales G. Aribe Jr., Bobby D. Gerardo, Ruji P. Medina. (2024)  
**Time Series Forecasting of HIV/AIDS in the Philippines Using Deep Learning: Does COVID-19 Epidemic Matter?**  

---
Primary Category: cs.NE  
Categories: cs-LG, cs-NE, cs.NE  
Keywords: AI, Time Series  
[Paper Link](http://arxiv.org/abs/2401.05933v1)  

---


**ABSTRACT**  
With a 676% growth rate in HIV incidence between 2010 and 2021, the HIV/AIDS epidemic in the Philippines is the one that is spreading the quickest in the western Pacific. Although the full effects of COVID-19 on HIV services and development are still unknown, it is predicted that such disruptions could lead to a significant increase in HIV casualties. Therefore, the nation needs some modeling and forecasting techniques to foresee the spread pattern and enhance the governments prevention, treatment, testing, and care program. In this study, the researcher uses Multilayer Perceptron Neural Network to forecast time series during the period when the COVID-19 pandemic strikes the nation, using statistics taken from the HIV/AIDS and ART Registry of the Philippines. After training, validation, and testing of data, the study finds that the predicted cumulative cases in the nation by 2030 will reach 145,273. Additionally, there is very little difference between observed and anticipated HIV epidemic levels, as evidenced by reduced RMSE, MAE, and MAPE values as well as a greater coefficient of determination. Further research revealed that the Philippines seems far from achieving Sustainable Development Goal 3 of Project 2030 due to an increase in the nations rate of new HIV infections. Despite the detrimental effects of COVID-19 spread on HIV/AIDS efforts nationwide, the Philippine government, under the Marcos administration, must continue to adhere to the United Nations 90-90-90 targets by enhancing its ART program and ensuring that all vital health services are readily accessible and available.

{{</citation>}}


## math.DS (1)



### (87/94) A Geometric Embedding Approach to Multiple Games and Multiple Populations (Bastian Boll et al., 2024)

{{<citation>}}

Bastian Boll, Jonas Cassel, Peter Albers, Stefania Petra, Christoph Schnörr. (2024)  
**A Geometric Embedding Approach to Multiple Games and Multiple Populations**  

---
Primary Category: math.DS  
Categories: cs-GT, math-DS, math.DS, nlin-AO  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2401.05918v1)  

---


**ABSTRACT**  
This paper studies a meta-simplex concept and geometric embedding framework for multi-population replicator dynamics. Central results are two embedding theorems which constitute a formal reduction of multi-population replicator dynamics to single-population ones. In conjunction with a robust mathematical formalism, this provides a toolset for analyzing complex multi-population models. Our framework provides a unifying perspective on different population dynamics in the literature which in particular enables to establish a formal link between multi-population and multi-game dynamics.

{{</citation>}}


## cs.SD (2)



### (88/94) Contrastive Loss Based Frame-wise Feature disentanglement for Polyphonic Sound Event Detection (Yadong Guan et al., 2024)

{{<citation>}}

Yadong Guan, Jiqing Han, Hongwei Song, Wenjie Song, Guibin Zheng, Tieran Zheng, Yongjun He. (2024)  
**Contrastive Loss Based Frame-wise Feature disentanglement for Polyphonic Sound Event Detection**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Event Detection  
[Paper Link](http://arxiv.org/abs/2401.05850v1)  

---


**ABSTRACT**  
Overlapping sound events are ubiquitous in real-world environments, but existing end-to-end sound event detection (SED) methods still struggle to detect them effectively. A critical reason is that these methods represent overlapping events using shared and entangled frame-wise features, which degrades the feature discrimination. To solve the problem, we propose a disentangled feature learning framework to learn a category-specific representation. Specifically, we employ different projectors to learn the frame-wise features for each category. To ensure that these feature does not contain information of other categories, we maximize the common information between frame-wise features within the same category and propose a frame-wise contrastive loss. In addition, considering that the labeled data used by the proposed method is limited, we propose a semi-supervised frame-wise contrastive loss that can leverage large amounts of unlabeled data to achieve feature disentanglement. The experimental results demonstrate the effectiveness of our method.

{{</citation>}}


### (89/94) Self-Attention and Hybrid Features for Replay and Deep-Fake Audio Detection (Lian Huang et al., 2024)

{{<citation>}}

Lian Huang, Chi-Man Pun. (2024)  
**Self-Attention and Hybrid Features for Replay and Deep-Fake Audio Detection**  

---
Primary Category: cs.SD  
Categories: cs-MM, cs-SD, cs.SD, eess-AS  
Keywords: Attention, Self-Attention  
[Paper Link](http://arxiv.org/abs/2401.05614v1)  

---


**ABSTRACT**  
Due to the successful application of deep learning, audio spoofing detection has made significant progress. Spoofed audio with speech synthesis or voice conversion can be well detected by many countermeasures. However, an automatic speaker verification system is still vulnerable to spoofing attacks such as replay or Deep-Fake audio. Deep-Fake audio means that the spoofed utterances are generated using text-to-speech (TTS) and voice conversion (VC) algorithms. Here, we propose a novel framework based on hybrid features with the self-attention mechanism. It is expected that hybrid features can be used to get more discrimination capacity. Firstly, instead of only one type of conventional feature, deep learning features and Mel-spectrogram features will be extracted by two parallel paths: convolution neural networks and a short-time Fourier transform (STFT) followed by Mel-frequency. Secondly, features will be concatenated by a max-pooling layer. Thirdly, there is a Self-attention mechanism for focusing on essential elements. Finally, ResNet and a linear layer are built to get the results. Experimental results reveal that the hybrid features, compared with conventional features, can cover more details of an utterance. We achieve the best Equal Error Rate (EER) of 9.67\% in the physical access (PA) scenario and 8.94\% in the Deep fake task on the ASVspoof 2021 dataset. Compared with the best baseline system, the proposed approach improves by 74.60\% and 60.05\%, respectively.

{{</citation>}}


## eess.SP (1)



### (90/94) TAnet: A New Temporal Attention Network for EEG-based Auditory Spatial Attention Decoding with a Short Decision Window (Yuting Ding et al., 2024)

{{<citation>}}

Yuting Ding, Fei Chen. (2024)  
**TAnet: A New Temporal Attention Network for EEG-based Auditory Spatial Attention Decoding with a Short Decision Window**  

---
Primary Category: eess.SP  
Categories: cs-LG, eess-SP, eess.SP  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2401.05819v1)  

---


**ABSTRACT**  
Auditory spatial attention detection (ASAD) is used to determine the direction of a listener's attention to a speaker by analyzing her/his electroencephalographic (EEG) signals. This study aimed to further improve the performance of ASAD with a short decision window (i.e., <1 s) rather than with long decision windows in previous studies. An end-to-end temporal attention network (i.e., TAnet) was introduced in this work. TAnet employs a multi-head attention (MHA) mechanism, which can more effectively capture the interactions among time steps in collected EEG signals and efficiently assign corresponding weights to those EEG time steps. Experiments demonstrated that, compared with the CNN-based method and recent ASAD methods, TAnet provided improved decoding performance in the KUL dataset, with decoding accuracies of 92.4% (decision window 0.1 s), 94.9% (0.25 s), 95.1% (0.3 s), 95.4% (0.4 s), and 95.5% (0.5 s) with short decision windows (i.e., <1 s). As a new ASAD model with a short decision window, TAnet can potentially facilitate the design of EEG-controlled intelligent hearing aids and sound recognition systems.

{{</citation>}}


## cs.IR (3)



### (91/94) What Else Would I Like? A User Simulator using Alternatives for Improved Evaluation of Fashion Conversational Recommendation Systems (Maria Vlachou et al., 2024)

{{<citation>}}

Maria Vlachou, Craig Macdonald. (2024)  
**What Else Would I Like? A User Simulator using Alternatives for Improved Evaluation of Fashion Conversational Recommendation Systems**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Conversational Recommendation  
[Paper Link](http://arxiv.org/abs/2401.05783v1)  

---


**ABSTRACT**  
In Conversational Recommendation Systems (CRS), a user can provide feedback on recommended items at each interaction turn, leading the CRS towards more desirable recommendations. Currently, different types of CRS offer various possibilities for feedback, i.e., natural language feedback, or answering clarifying questions. In most cases, a user simulator is employed for training as well as evaluating the CRS. Such user simulators typically critique the current retrieved items based on knowledge of a single target item. Still, evaluating systems in offline settings with simulators suffers from problems, such as focusing entirely on a single target item (not addressing the exploratory nature of a recommender system), and exhibiting extreme patience (consistent feedback over a large number of turns). To overcome these limitations, we obtain extra judgements for a selection of alternative items in common CRS datasets, namely Shoes and Fashion IQ Dresses. Going further, we propose improved user simulators that allow simulated users not only to express their preferences about alternative items to their original target, but also to change their mind and level of patience. In our experiments using the relative image captioning CRS setting and different CRS models, we find that using the knowledge of alternatives by the simulator can have a considerable impact on the evaluation of existing CRS models, specifically that the existing single-target evaluation underestimates their effectiveness, and when simulated users are allowed to instead consider alternatives, the system can rapidly respond to more quickly satisfy the user.

{{</citation>}}


### (92/94) Large Language Models vs. Search Engines: Evaluating User Preferences Across Varied Information Retrieval Scenarios (Kevin Matthe Caramancion, 2024)

{{<citation>}}

Kevin Matthe Caramancion. (2024)  
**Large Language Models vs. Search Engines: Evaluating User Preferences Across Varied Information Retrieval Scenarios**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Information Retrieval, Language Model  
[Paper Link](http://arxiv.org/abs/2401.05761v1)  

---


**ABSTRACT**  
This study embarked on a comprehensive exploration of user preferences between Search Engines and Large Language Models (LLMs) in the context of various information retrieval scenarios. Conducted with a sample size of 100 internet users (N=100) from across the United States, the research delved into 20 distinct use cases ranging from factual searches, such as looking up COVID-19 guidelines, to more subjective tasks, like seeking interpretations of complex concepts in layman's terms. Participants were asked to state their preference between using a traditional search engine or an LLM for each scenario. This approach allowed for a nuanced understanding of how users perceive and utilize these two predominant digital tools in differing contexts. The use cases were carefully selected to cover a broad spectrum of typical online queries, thus ensuring a comprehensive analysis of user preferences. The findings reveal intriguing patterns in user choices, highlighting a clear tendency for participants to favor search engines for direct, fact-based queries, while LLMs were more often preferred for tasks requiring nuanced understanding and language processing. These results offer valuable insights into the current state of digital information retrieval and pave the way for future innovations in this field. This study not only sheds light on the specific contexts in which each tool is favored but also hints at the potential for developing hybrid models that leverage the strengths of both search engines and LLMs. The insights gained from this research are pivotal for developers, researchers, and policymakers in understanding the evolving landscape of digital information retrieval and user interaction with these technologies.

{{</citation>}}


### (93/94) Attention Is Not the Only Choice: Counterfactual Reasoning for Path-Based Explainable Recommendation (Yicong Li et al., 2024)

{{<citation>}}

Yicong Li, Xiangguo Sun, Hongxu Chen, Sixiao Zhang, Yu Yang, Guandong Xu. (2024)  
**Attention Is Not the Only Choice: Counterfactual Reasoning for Path-Based Explainable Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Attention, Reasoning  
[Paper Link](http://arxiv.org/abs/2401.05744v1)  

---


**ABSTRACT**  
Compared with only pursuing recommendation accuracy, the explainability of a recommendation model has drawn more attention in recent years. Many graph-based recommendations resort to informative paths with the attention mechanism for the explanation. Unfortunately, these attention weights are intentionally designed for model accuracy but not explainability. Recently, some researchers have started to question attention-based explainability because the attention weights are unstable for different reproductions, and they may not always align with human intuition. Inspired by the counterfactual reasoning from causality learning theory, we propose a novel explainable framework targeting path-based recommendations, wherein the explainable weights of paths are learned to replace attention weights. Specifically, we design two counterfactual reasoning algorithms from both path representation and path topological structure perspectives. Moreover, unlike traditional case studies, we also propose a package of explainability evaluation solutions with both qualitative and quantitative methods. We conduct extensive experiments on three real-world datasets, the results of which further demonstrate the effectiveness and reliability of our method.

{{</citation>}}


## cs.CR (1)



### (94/94) Use of Graph Neural Networks in Aiding Defensive Cyber Operations (Shaswata Mitra et al., 2024)

{{<citation>}}

Shaswata Mitra, Trisha Chakraborty, Subash Neupane, Aritran Piplai, Sudip Mittal. (2024)  
**Use of Graph Neural Networks in Aiding Defensive Cyber Operations**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs-LG, cs-NE, cs.CR  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2401.05680v1)  

---


**ABSTRACT**  
In an increasingly interconnected world, where information is the lifeblood of modern society, regular cyber-attacks sabotage the confidentiality, integrity, and availability of digital systems and information. Additionally, cyber-attacks differ depending on the objective and evolve rapidly to disguise defensive systems. However, a typical cyber-attack demonstrates a series of stages from attack initiation to final resolution, called an attack life cycle. These diverse characteristics and the relentless evolution of cyber attacks have led cyber defense to adopt modern approaches like Machine Learning to bolster defensive measures and break the attack life cycle. Among the adopted ML approaches, Graph Neural Networks have emerged as a promising approach for enhancing the effectiveness of defensive measures due to their ability to process and learn from heterogeneous cyber threat data. In this paper, we look into the application of GNNs in aiding to break each stage of one of the most renowned attack life cycles, the Lockheed Martin Cyber Kill Chain. We address each phase of CKC and discuss how GNNs contribute to preparing and preventing an attack from a defensive standpoint. Furthermore, We also discuss open research areas and further improvement scopes.

{{</citation>}}
