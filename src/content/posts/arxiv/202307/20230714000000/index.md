---
draft: false
title: "arXiv @ 2023.07.14"
date: 2023-07-14
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.07.14"
    identifier: arxiv_20230714
    parent: 202307_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CV (26)](#cscv-26)
- [cs.DS (1)](#csds-1)
- [cs.LG (17)](#cslg-17)
- [cs.AI (3)](#csai-3)
- [cs.CL (19)](#cscl-19)
- [physics.med-ph (1)](#physicsmed-ph-1)
- [physics.optics (1)](#physicsoptics-1)
- [cs.RO (5)](#csro-5)
- [cs.SE (3)](#csse-3)
- [cs.SI (1)](#cssi-1)
- [eess.SP (2)](#eesssp-2)
- [cs.IT (1)](#csit-1)
- [math.OC (1)](#mathoc-1)
- [cs.CR (3)](#cscr-3)
- [cs.IR (1)](#csir-1)
- [eess.AS (2)](#eessas-2)
- [eess.IV (2)](#eessiv-2)
- [cs.SD (1)](#cssd-1)
- [cs.HC (1)](#cshc-1)
- [cs.NE (2)](#csne-2)
- [cs.DL (1)](#csdl-1)
- [physics.soc-ph (1)](#physicssoc-ph-1)
- [math.NA (1)](#mathna-1)
- [cs.DC (1)](#csdc-1)
- [stat.AP (1)](#statap-1)
- [cs.CY (1)](#cscy-1)

## cs.CV (26)



### (1/99) MPDIoU: A Loss for Efficient and Accurate Bounding Box Regression (Ma Siliang et al., 2023)

{{<citation>}}

Ma Siliang, Xu Yong. (2023)  
**MPDIoU: A Loss for Efficient and Accurate Bounding Box Regression**  

---
Primary Category: cs.CV
Categories: cs-AI, cs-CV, cs.CV  
Keywords: T5  
[Paper Link](http://arxiv.org/abs/2307.07662v1)  

---


**ABSTRACT**  
Bounding box regression (BBR) has been widely used in object detection and instance segmentation, which is an important step in object localization. However, most of the existing loss functions for bounding box regression cannot be optimized when the predicted box has the same aspect ratio as the groundtruth box, but the width and height values are exactly different. In order to tackle the issues mentioned above, we fully explore the geometric features of horizontal rectangle and propose a novel bounding box similarity comparison metric MPDIoU based on minimum point distance, which contains all of the relevant factors considered in the existing loss functions, namely overlapping or non-overlapping area, central points distance, and deviation of width and height, while simplifying the calculation process. On this basis, we propose a bounding box regression loss function based on MPDIoU, called LMPDIoU . Experimental results show that the MPDIoU loss function is applied to state-of-the-art instance segmentation (e.g., YOLACT) and object detection (e.g., YOLOv7) model trained on PASCAL VOC, MS COCO, and IIIT5k outperforms existing loss functions.

{{</citation>}}


### (2/99) RFLA: A Stealthy Reflected Light Adversarial Attack in the Physical World (Donghua Wang et al., 2023)

{{<citation>}}

Donghua Wang, Wen Yao, Tingsong Jiang, Chao Li, Xiaoqian Chen. (2023)  
**RFLA: A Stealthy Reflected Light Adversarial Attack in the Physical World**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2307.07653v1)  

---


**ABSTRACT**  
Physical adversarial attacks against deep neural networks (DNNs) have recently gained increasing attention. The current mainstream physical attacks use printed adversarial patches or camouflage to alter the appearance of the target object. However, these approaches generate conspicuous adversarial patterns that show poor stealthiness. Another physical deployable attack is the optical attack, featuring stealthiness while exhibiting weakly in the daytime with sunlight. In this paper, we propose a novel Reflected Light Attack (RFLA), featuring effective and stealthy in both the digital and physical world, which is implemented by placing the color transparent plastic sheet and a paper cut of a specific shape in front of the mirror to create different colored geometries on the target object. To achieve these goals, we devise a general framework based on the circle to model the reflected light on the target object. Specifically, we optimize a circle (composed of a coordinate and radius) to carry various geometrical shapes determined by the optimized angle. The fill color of the geometry shape and its corresponding transparency are also optimized. We extensively evaluate the effectiveness of RFLA on different datasets and models. Experiment results suggest that the proposed method achieves over 99% success rate on different datasets and models in the digital world. Additionally, we verify the effectiveness of the proposed method in different physical environments by using sunlight or a flashlight.

{{</citation>}}


### (3/99) ACF-Net: An Attention-enhanced Co-interactive Fusion Network for Automated Structural Condition Assessment in Visual Inspection (Chenyu Zhang et al., 2023)

{{<citation>}}

Chenyu Zhang, Zhaozheng Yin, Ruwen Qin. (2023)  
**ACF-Net: An Attention-enhanced Co-interactive Fusion Network for Automated Structural Condition Assessment in Visual Inspection**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2307.07643v1)  

---


**ABSTRACT**  
Efficiently monitoring the condition of civil infrastructures necessitates automating the structural condition assessment in visual inspection. This paper proposes an Attention-enhanced Co-interactive Fusion Network (ACF-Net) for automatic structural condition assessment in visual bridge inspection. The ACF-Net can simultaneously parse structural elements and segment surface defects on the elements in inspection images. It integrates two task-specific relearning subnets to extract task-specific features from an overall feature embedding and a co-interactive feature fusion module to capture the spatial correlation and facilitate information sharing between tasks. Experimental results demonstrate that the proposed ACF-Net outperforms the current state-of-the-art approaches, achieving promising performance with 92.11% mIoU for element parsing and 87.16% mIoU for corrosion segmentation on the new benchmark dataset Steel Bridge Condition Inspection Visual (SBCIV) testing set. An ablation study reveals the strengths of ACF-Net, and a case study showcases its capability to automate structural condition assessment. The code will be open-source after acceptance.

{{</citation>}}


### (4/99) Gastrointestinal Disease Classification through Explainable and Cost-Sensitive Deep Neural Networks with Supervised Contrastive Learning (Dibya Nath et al., 2023)

{{<citation>}}

Dibya Nath, G. M. Shahariar. (2023)  
**Gastrointestinal Disease Classification through Explainable and Cost-Sensitive Deep Neural Networks with Supervised Contrastive Learning**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: AI, Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2307.07603v1)  

---


**ABSTRACT**  
Gastrointestinal diseases pose significant healthcare chall-enges as they manifest in diverse ways and can lead to potential complications. Ensuring precise and timely classification of these diseases is pivotal in guiding treatment choices and enhancing patient outcomes. This paper introduces a novel approach on classifying gastrointestinal diseases by leveraging cost-sensitive pre-trained deep convolutional neural network (CNN) architectures with supervised contrastive learning. Our approach enables the network to learn representations that capture vital disease-related features, while also considering the relationships of similarity between samples. To tackle the challenges posed by imbalanced datasets and the cost-sensitive nature of misclassification errors in healthcare, we incorporate cost-sensitive learning. By assigning distinct costs to misclassifications based on the disease class, we prioritize accurate classification of critical conditions. Furthermore, we enhance the interpretability of our model by integrating gradient-based techniques from explainable artificial intelligence (AI). This inclusion provides valuable insights into the decision-making process of the network, aiding in understanding the features that contribute to disease classification. To assess the effectiveness of our proposed approach, we perform extensive experiments on a comprehensive gastrointestinal disease dataset, such as the Hyper-Kvasir dataset. Through thorough comparisons with existing works, we demonstrate the strong classification accuracy, robustness and interpretability of our model. We have made the implementation of our proposed approach publicly available at https://github.com/dibya404/Gastrointestinal-Disease-Classification-through-Explainable-and-Cost-Sensitive-DNN-with-SCL

{{</citation>}}


### (5/99) TALL: Thumbnail Layout for Deepfake Video Detection (Yuting Xu et al., 2023)

{{<citation>}}

Yuting Xu, Jian Liang, Gengyun Jia, Ziming Yang, Yanhao Zhang, Ran He. (2023)  
**TALL: Thumbnail Layout for Deepfake Video Detection**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.07494v1)  

---


**ABSTRACT**  
The growing threats of deepfakes to society and cybersecurity have raised enormous public concerns, and increasing efforts have been devoted to this critical topic of deepfake video detection. Existing video methods achieve good performance but are computationally intensive. This paper introduces a simple yet effective strategy named Thumbnail Layout (TALL), which transforms a video clip into a pre-defined layout to realize the preservation of spatial and temporal dependencies. Specifically, consecutive frames are masked in a fixed position in each frame to improve generalization, then resized to sub-images and rearranged into a pre-defined layout as the thumbnail. TALL is model-agnostic and extremely simple by only modifying a few lines of code. Inspired by the success of vision transformers, we incorporate TALL into Swin Transformer, forming an efficient and effective method TALL-Swin. Extensive experiments on intra-dataset and cross-dataset validate the validity and superiority of TALL and SOTA TALL-Swin. TALL-Swin achieves 90.79$\%$ AUC on the challenging cross-dataset task, FaceForensics++ $\to$ Celeb-DF. The code is available at https://github.com/rainy-xu/TALL4Deepfake.

{{</citation>}}


### (6/99) DreamTeacher: Pretraining Image Backbones with Deep Generative Models (Daiqing Li et al., 2023)

{{<citation>}}

Daiqing Li, Huan Ling, Amlan Kar, David Acuna, Seung Wook Kim, Karsten Kreis, Antonio Torralba, Sanja Fidler. (2023)  
**DreamTeacher: Pretraining Image Backbones with Deep Generative Models**  

---
Primary Category: cs.CV
Categories: cs-CV, cs-LG, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2307.07487v1)  

---


**ABSTRACT**  
In this work, we introduce a self-supervised feature representation learning framework DreamTeacher that utilizes generative networks for pre-training downstream image backbones. We propose to distill knowledge from a trained generative model into standard image backbones that have been well engineered for specific perception tasks. We investigate two types of knowledge distillation: 1) distilling learned generative features onto target image backbones as an alternative to pretraining these backbones on large labeled datasets such as ImageNet, and 2) distilling labels obtained from generative networks with task heads onto logits of target backbones. We perform extensive analyses on multiple generative models, dense prediction benchmarks, and several pre-training regimes. We empirically find that our DreamTeacher significantly outperforms existing self-supervised representation learning approaches across the board. Unsupervised ImageNet pre-training with DreamTeacher leads to significant improvements over ImageNet classification pre-training on downstream datasets, showcasing generative models, and diffusion generative models specifically, as a promising approach to representation learning on large, diverse datasets without requiring manual annotation.

{{</citation>}}


### (7/99) Multimodal Distillation for Egocentric Action Recognition (Gorjan Radevski et al., 2023)

{{<citation>}}

Gorjan Radevski, Dusan Grujicic, Marie-Francine Moens, Matthew Blaschko, Tinne Tuytelaars. (2023)  
**Multimodal Distillation for Egocentric Action Recognition**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.07483v2)  

---


**ABSTRACT**  
The focal point of egocentric video understanding is modelling hand-object interactions. Standard models, e.g. CNNs or Vision Transformers, which receive RGB frames as input perform well. However, their performance improves further by employing additional input modalities that provide complementary cues, such as object detections, optical flow, audio, etc. The added complexity of the modality-specific modules, on the other hand, makes these models impractical for deployment. The goal of this work is to retain the performance of such a multimodal approach, while using only the RGB frames as input at inference time. We demonstrate that for egocentric action recognition on the Epic-Kitchens and the Something-Something datasets, students which are taught by multimodal teachers tend to be more accurate and better calibrated than architecturally equivalent models trained on ground truth labels in a unimodal or multimodal fashion. We further adopt a principled multimodal knowledge distillation framework, allowing us to deal with issues which occur when applying multimodal knowledge distillation in a naive manner. Lastly, we demonstrate the achieved reduction in computational complexity, and show that our approach maintains higher performance with the reduction of the number of input views. We release our code at https://github.com/gorjanradevski/multimodal-distillation.

{{</citation>}}


### (8/99) Dual-Query Multiple Instance Learning for Dynamic Meta-Embedding based Tumor Classification (Simon Holdenried-Krafft et al., 2023)

{{<citation>}}

Simon Holdenried-Krafft, Peter Somers, Ivonne A. Montes-Majarro, Diana Silimon, Cristina Tarín, Falko Fend, Hendrik P. A. Lensch. (2023)  
**Dual-Query Multiple Instance Learning for Dynamic Meta-Embedding based Tumor Classification**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2307.07482v1)  

---


**ABSTRACT**  
Whole slide image (WSI) assessment is a challenging and crucial step in cancer diagnosis and treatment planning. WSIs require high magnifications to facilitate sub-cellular analysis. Precise annotations for patch- or even pixel-level classifications in the context of gigapixel WSIs are tedious to acquire and require domain experts. Coarse-grained labels, on the other hand, are easily accessible, which makes WSI classification an ideal use case for multiple instance learning (MIL). In our work, we propose a novel embedding-based Dual-Query MIL pipeline (DQ-MIL). We contribute to both the embedding and aggregation steps. Since all-purpose visual feature representations are not yet available, embedding models are currently limited in terms of generalizability. With our work, we explore the potential of dynamic meta-embedding based on cutting-edge self-supervised pre-trained models in the context of MIL. Moreover, we propose a new MIL architecture capable of combining MIL-attention with correlated self-attention. The Dual-Query Perceiver design of our approach allows us to leverage the concept of self-distillation and to combine the advantages of a small model in the context of a low data regime with the rich feature representation of a larger model. We demonstrate the superior performance of our approach on three histopathological datasets, where we show improvement of up to 10% over state-of-the-art approaches.

{{</citation>}}


### (9/99) Interactive Spatiotemporal Token Attention Network for Skeleton-based General Interactive Action Recognition (Yuhang Wen et al., 2023)

{{<citation>}}

Yuhang Wen, Zixuan Tang, Yunsheng Pang, Beichen Ding, Mengyuan Liu. (2023)  
**Interactive Spatiotemporal Token Attention Network for Skeleton-based General Interactive Action Recognition**  

---
Primary Category: cs.CV
Categories: cs-AI, cs-CV, cs-RO, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2307.07469v1)  

---


**ABSTRACT**  
Recognizing interactive action plays an important role in human-robot interaction and collaboration. Previous methods use late fusion and co-attention mechanism to capture interactive relations, which have limited learning capability or inefficiency to adapt to more interacting entities. With assumption that priors of each entity are already known, they also lack evaluations on a more general setting addressing the diversity of subjects. To address these problems, we propose an Interactive Spatiotemporal Token Attention Network (ISTA-Net), which simultaneously model spatial, temporal, and interactive relations. Specifically, our network contains a tokenizer to partition Interactive Spatiotemporal Tokens (ISTs), which is a unified way to represent motions of multiple diverse entities. By extending the entity dimension, ISTs provide better interactive representations. To jointly learn along three dimensions in ISTs, multi-head self-attention blocks integrated with 3D convolutions are designed to capture inter-token correlations. When modeling correlations, a strict entity ordering is usually irrelevant for recognizing interactive actions. To this end, Entity Rearrangement is proposed to eliminate the orderliness in ISTs for interchangeable entities. Extensive experiments on four datasets verify the effectiveness of ISTA-Net by outperforming state-of-the-art methods. Our code is publicly available at https://github.com/Necolizer/ISTA-Net

{{</citation>}}


### (10/99) Combining multitemporal optical and SAR data for LAI imputation with BiLSTM network (W. Zhao et al., 2023)

{{<citation>}}

W. Zhao, F. Yin, H. Ma, Q. Wu, J. Gomez-Dans, P. Lewis. (2023)  
**Combining multitemporal optical and SAR data for LAI imputation with BiLSTM network**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV, eess-IV  
Keywords: AI, LSTM  
[Paper Link](http://arxiv.org/abs/2307.07434v1)  

---


**ABSTRACT**  
The Leaf Area Index (LAI) is vital for predicting winter wheat yield. Acquisition of crop conditions via Sentinel-2 remote sensing images can be hindered by persistent clouds, affecting yield predictions. Synthetic Aperture Radar (SAR) provides all-weather imagery, and the ratio between its cross- and co-polarized channels (C-band) shows a high correlation with time series LAI over winter wheat regions. This study evaluates the use of time series Sentinel-1 VH/VV for LAI imputation, aiming to increase spatial-temporal density. We utilize a bidirectional LSTM (BiLSTM) network to impute time series LAI and use half mean squared error for each time step as the loss function. We trained models on data from southern Germany and the North China Plain using only LAI data generated by Sentinel-1 VH/VV and Sentinel-2. Experimental results show BiLSTM outperforms traditional regression methods, capturing nonlinear dynamics between multiple time series. It proves robust in various growing conditions and is effective even with limited Sentinel-2 images. BiLSTM's performance surpasses that of LSTM, particularly over the senescence period. Therefore, BiLSTM can be used to impute LAI with time-series Sentinel-1 VH/VV and Sentinel-2 data, and this method could be applied to other time-series imputation issues.

{{</citation>}}


### (11/99) Improving Zero-Shot Generalization for CLIP with Synthesized Prompts (Zhengbo Wang et al., 2023)

{{<citation>}}

Zhengbo Wang, Jian Liang, Ran He, Nan Xu, Zilei Wang, Tieniu Tan. (2023)  
**Improving Zero-Shot Generalization for CLIP with Synthesized Prompts**  

---
Primary Category: cs.CV
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2307.07397v1)  

---


**ABSTRACT**  
With the growing interest in pretrained vision-language models like CLIP, recent research has focused on adapting these models to downstream tasks. Despite achieving promising results, most existing methods require labeled data for all classes, which may not hold in real-world applications due to the long tail and Zipf's law. For example, some classes may lack labeled data entirely, such as emerging concepts. To address this problem, we propose a plug-and-play generative approach called \textbf{S}ynt\textbf{H}es\textbf{I}zed \textbf{P}rompts~(\textbf{SHIP}) to improve existing fine-tuning methods. Specifically, we follow variational autoencoders to introduce a generator that reconstructs the visual features by inputting the synthesized prompts and the corresponding class names to the textual encoder of CLIP. In this manner, we easily obtain the synthesized features for the remaining label-only classes. Thereafter, we fine-tune CLIP with off-the-shelf methods by combining labeled and synthesized features. Extensive experiments on base-to-new generalization, cross-dataset transfer learning, and generalized zero-shot learning demonstrate the superiority of our approach. The code is available at \url{https://github.com/mrflogs/SHIP}.

{{</citation>}}


### (12/99) L-DAWA: Layer-wise Divergence Aware Weight Aggregation in Federated Self-Supervised Visual Representation Learning (Yasar Abbas Ur Rehman et al., 2023)

{{<citation>}}

Yasar Abbas Ur Rehman, Yan Gao, Pedro Porto Buarque de Gusmão, Mina Alibeigi, Jiajun Shen, Nicholas D. Lane. (2023)  
**L-DAWA: Layer-wise Divergence Aware Weight Aggregation in Federated Self-Supervised Visual Representation Learning**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Representation Learning, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2307.07393v1)  

---


**ABSTRACT**  
The ubiquity of camera-enabled devices has led to large amounts of unlabeled image data being produced at the edge. The integration of self-supervised learning (SSL) and federated learning (FL) into one coherent system can potentially offer data privacy guarantees while also advancing the quality and robustness of the learned visual representations without needing to move data around. However, client bias and divergence during FL aggregation caused by data heterogeneity limits the performance of learned visual representations on downstream tasks. In this paper, we propose a new aggregation strategy termed Layer-wise Divergence Aware Weight Aggregation (L-DAWA) to mitigate the influence of client bias and divergence during FL aggregation. The proposed method aggregates weights at the layer-level according to the measure of angular divergence between the clients' model and the global model. Extensive experiments with cross-silo and cross-device settings on CIFAR-10/100 and Tiny ImageNet datasets demonstrate that our methods are effective and obtain new SOTA performance on both contrastive and non-contrastive SSL approaches.

{{</citation>}}


### (13/99) AIC-AB NET: A Neural Network for Image Captioning with Spatial Attention and Text Attributes (Guoyun Tu et al., 2023)

{{<citation>}}

Guoyun Tu, Ying Liu, Vladimir Vlassov. (2023)  
**AIC-AB NET: A Neural Network for Image Captioning with Spatial Attention and Text Attributes**  

---
Primary Category: cs.CV
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: AI, Attention, Image Captioning  
[Paper Link](http://arxiv.org/abs/2307.07370v1)  

---


**ABSTRACT**  
Image captioning is a significant field across computer vision and natural language processing. We propose and present AIC-AB NET, a novel Attribute-Information-Combined Attention-Based Network that combines spatial attention architecture and text attributes in an encoder-decoder. For caption generation, adaptive spatial attention determines which image region best represents the image and whether to attend to the visual features or the visual sentinel. Text attribute information is synchronously fed into the decoder to help image recognition and reduce uncertainty. We have tested and evaluated our AICAB NET on the MS COCO dataset and a new proposed Fashion dataset. The Fashion dataset is employed as a benchmark of single-object images. The results show the superior performance of the proposed model compared to the state-of-the-art baseline and ablated models on both the images from MSCOCO and our single-object images. Our AIC-AB NET outperforms the baseline adaptive attention network by 0.017 (CIDEr score) on the MS COCO dataset and 0.095 (CIDEr score) on the Fashion dataset.

{{</citation>}}


### (14/99) ConTrack: Contextual Transformer for Device Tracking in X-ray (Marc Demoustier et al., 2023)

{{<citation>}}

Marc Demoustier, Yue Zhang, Venkatesh Narasimha Murthy, Florin C. Ghesu, Dorin Comaniciu. (2023)  
**ConTrack: Contextual Transformer for Device Tracking in X-ray**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.07541v1)  

---


**ABSTRACT**  
Device tracking is an important prerequisite for guidance during endovascular procedures. Especially during cardiac interventions, detection and tracking of guiding the catheter tip in 2D fluoroscopic images is important for applications such as mapping vessels from angiography (high dose with contrast) to fluoroscopy (low dose without contrast). Tracking the catheter tip poses different challenges: the tip can be occluded by contrast during angiography or interventional devices; and it is always in continuous movement due to the cardiac and respiratory motions. To overcome these challenges, we propose ConTrack, a transformer-based network that uses both spatial and temporal contextual information for accurate device detection and tracking in both X-ray fluoroscopy and angiography. The spatial information comes from the template frames and the segmentation module: the template frames define the surroundings of the device, whereas the segmentation module detects the entire device to bring more context for the tip prediction. Using multiple templates makes the model more robust to the change in appearance of the device when it is occluded by the contrast agent. The flow information computed on the segmented catheter mask between the current and the previous frame helps in further refining the prediction by compensating for the respiratory and cardiac motions. The experiments show that our method achieves 45% or higher accuracy in detection and tracking when compared to state-of-the-art tracking models.

{{</citation>}}


### (15/99) A scoping review on multimodal deep learning in biomedical images and texts (Zhaoyi Sun et al., 2023)

{{<citation>}}

Zhaoyi Sun, Mingquan Lin, Qingqing Zhu, Qianqian Xie, Fei Wang, Zhiyong Lu, Yifan Peng. (2023)  
**A scoping review on multimodal deep learning in biomedical images and texts**  

---
Primary Category: cs.CV
Categories: cs-CL, cs-CV, cs.CV  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2307.07362v1)  

---


**ABSTRACT**  
Computer-assisted diagnostic and prognostic systems of the future should be capable of simultaneously processing multimodal data. Multimodal deep learning (MDL), which involves the integration of multiple sources of data, such as images and text, has the potential to revolutionize the analysis and interpretation of biomedical data. However, it only caught researchers' attention recently. To this end, there is a critical need to conduct a systematic review on this topic, identify the limitations of current work, and explore future directions. In this scoping review, we aim to provide a comprehensive overview of the current state of the field and identify key concepts, types of studies, and research gaps with a focus on biomedical images and texts joint learning, mainly because these two were the most commonly available data types in MDL research. This study reviewed the current uses of multimodal deep learning on five tasks: (1) Report generation, (2) Visual question answering, (3) Cross-modal retrieval, (4) Computer-aided diagnosis, and (5) Semantic segmentation. Our results highlight the diverse applications and potential of MDL and suggest directions for future research in the field. We hope our review will facilitate the collaboration of natural language processing (NLP) and medical imaging communities and support the next generation of decision-making and computer-assisted diagnostic system development.

{{</citation>}}


### (16/99) Gloss Attention for Gloss-free Sign Language Translation (Aoxiong Yin et al., 2023)

{{<citation>}}

Aoxiong Yin, Tianyun Zhong, Li Tang, Weike Jin, Tao Jin, Zhou Zhao. (2023)  
**Gloss Attention for Gloss-free Sign Language Translation**  

---
Primary Category: cs.CV
Categories: cs-CL, cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2307.07361v1)  

---


**ABSTRACT**  
Most sign language translation (SLT) methods to date require the use of gloss annotations to provide additional supervision information, however, the acquisition of gloss is not easy. To solve this problem, we first perform an analysis of existing models to confirm how gloss annotations make SLT easier. We find that it can provide two aspects of information for the model, 1) it can help the model implicitly learn the location of semantic boundaries in continuous sign language videos, 2) it can help the model understand the sign language video globally. We then propose \emph{gloss attention}, which enables the model to keep its attention within video segments that have the same semantics locally, just as gloss helps existing models do. Furthermore, we transfer the knowledge of sentence-to-sentence similarity from the natural language model to our gloss attention SLT network (GASLT) to help it understand sign language videos at the sentence level. Experimental results on multiple large-scale sign language datasets show that our proposed GASLT model significantly outperforms existing methods. Our code is provided in \url{https://github.com/YinAoXiong/GASLT}.

{{</citation>}}


### (17/99) LEST: Large-scale LiDAR Semantic Segmentation with Transformer (Chuanyu Luo et al., 2023)

{{<citation>}}

Chuanyu Luo, Nuo Cheng, Sikun Ma, Han Li, Xiaohan Li, Shengguang Lei, Pu Li. (2023)  
**LEST: Large-scale LiDAR Semantic Segmentation with Transformer**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation, Transformer  
[Paper Link](http://arxiv.org/abs/2307.09367v1)  

---


**ABSTRACT**  
Large-scale LiDAR-based point cloud semantic segmentation is a critical task in autonomous driving perception. Almost all of the previous state-of-the-art LiDAR semantic segmentation methods are variants of sparse 3D convolution. Although the Transformer architecture is becoming popular in the field of natural language processing and 2D computer vision, its application to large-scale point cloud semantic segmentation is still limited. In this paper, we propose a LiDAR sEmantic Segmentation architecture with pure Transformer, LEST. LEST comprises two novel components: a Space Filling Curve (SFC) Grouping strategy and a Distance-based Cosine Linear Transformer, DISCO. On the public nuScenes semantic segmentation validation set and SemanticKITTI test set, our model outperforms all the other state-of-the-art methods.

{{</citation>}}


### (18/99) SynTable: A Synthetic Data Generation Pipeline for Unseen Object Amodal Instance Segmentation of Cluttered Tabletop Scenes (Zhili Ng et al., 2023)

{{<citation>}}

Zhili Ng, Haozhe Wang, Zhengshen Zhang, Francis Tay Eng Hock, Marcelo H. Ang Jr. (2023)  
**SynTable: A Synthetic Data Generation Pipeline for Unseen Object Amodal Instance Segmentation of Cluttered Tabletop Scenes**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.07333v1)  

---


**ABSTRACT**  
In this work, we present SynTable, a unified and flexible Python-based dataset generator built using NVIDIA's Isaac Sim Replicator Composer for generating high-quality synthetic datasets for unseen object amodal instance segmentation of cluttered tabletop scenes. Our dataset generation tool can render a complex 3D scene containing object meshes, materials, textures, lighting, and backgrounds. Metadata, such as modal and amodal instance segmentation masks, occlusion masks, depth maps, bounding boxes, and material properties, can be generated to automatically annotate the scene according to the users' requirements. Our tool eliminates the need for manual labeling in the dataset generation process while ensuring the quality and accuracy of the dataset. In this work, we discuss our design goals, framework architecture, and the performance of our tool. We demonstrate the use of a sample dataset generated using SynTable by ray tracing for training a state-of-the-art model, UOAIS-Net. The results show significantly improved performance in Sim-to-Real transfer when evaluated on the OSD-Amodal dataset. We offer this tool as an open-source, easy-to-use, photorealistic dataset generator for advancing research in deep learning and synthetic data generation.

{{</citation>}}


### (19/99) HEAL-SWIN: A Vision Transformer On The Sphere (Oscar Carlsson et al., 2023)

{{<citation>}}

Oscar Carlsson, Jan E. Gerken, Hampus Linander, Heiner Spieß, Fredrik Ohlsson, Christoffer Petersson, Daniel Persson. (2023)  
**HEAL-SWIN: A Vision Transformer On The Sphere**  

---
Primary Category: cs.CV
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.07313v1)  

---


**ABSTRACT**  
High-resolution wide-angle fisheye images are becoming more and more important for robotics applications such as autonomous driving. However, using ordinary convolutional neural networks or vision transformers on this data is problematic due to projection and distortion losses introduced when projecting to a rectangular grid on the plane. We introduce the HEAL-SWIN transformer, which combines the highly uniform Hierarchical Equal Area iso-Latitude Pixelation (HEALPix) grid used in astrophysics and cosmology with the Hierarchical Shifted-Window (SWIN) transformer to yield an efficient and flexible model capable of training on high-resolution, distortion-free spherical data. In HEAL-SWIN, the nested structure of the HEALPix grid is used to perform the patching and windowing operations of the SWIN transformer, resulting in a one-dimensional representation of the spherical data with minimal computational overhead. We demonstrate the superior performance of our model for semantic segmentation and depth regression tasks on both synthetic and real automotive datasets. Our code is available at https://github.com/JanEGerken/HEAL-SWIN.

{{</citation>}}


### (20/99) FreeCOS: Self-Supervised Learning from Fractals and Unlabeled Images for Curvilinear Object Segmentation (Tianyi Shi et al., 2023)

{{<citation>}}

Tianyi Shi, Xiaohuan Ding, Liang Zhang, Xin Yang. (2023)  
**FreeCOS: Self-Supervised Learning from Fractals and Unlabeled Images for Curvilinear Object Segmentation**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2307.07245v1)  

---


**ABSTRACT**  
Curvilinear object segmentation is critical for many applications. However, manually annotating curvilinear objects is very time-consuming and error-prone, yielding insufficiently available annotated datasets for existing supervised methods and domain adaptation methods. This paper proposes a self-supervised curvilinear object segmentation method that learns robust and distinctive features from fractals and unlabeled images (FreeCOS). The key contributions include a novel Fractal-FDA synthesis (FFS) module and a geometric information alignment (GIA) approach. FFS generates curvilinear structures based on the parametric Fractal L-system and integrates the generated structures into unlabeled images to obtain synthetic training images via Fourier Domain Adaptation. GIA reduces the intensity differences between the synthetic and unlabeled images by comparing the intensity order of a given pixel to the values of its nearby neighbors. Such image alignment can explicitly remove the dependency on absolute intensity values and enhance the inherent geometric characteristics which are common in both synthetic and real images. In addition, GIA aligns features of synthetic and real images via the prediction space adaptation loss (PSAL) and the curvilinear mask contrastive loss (CMCL). Extensive experimental results on four public datasets, i.e., XCAD, DRIVE, STARE and CrackTree demonstrate that our method outperforms the state-of-the-art unsupervised methods, self-supervised methods and traditional methods by a large margin. The source code of this work is available at https://github.com/TY-Shi/FreeCOS.

{{</citation>}}


### (21/99) Multimodal Motion Conditioned Diffusion Model for Skeleton-based Video Anomaly Detection (Alessandro Flaborea et al., 2023)

{{<citation>}}

Alessandro Flaborea, Luca Collorone, Guido D'Amely, Stefano D'Arrigo, Bardh Prenkaj, Fabio Galasso. (2023)  
**Multimodal Motion Conditioned Diffusion Model for Skeleton-based Video Anomaly Detection**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2307.07205v1)  

---


**ABSTRACT**  
Anomalies are rare and anomaly detection is often therefore framed as One-Class Classification (OCC), i.e. trained solely on normalcy. Leading OCC techniques constrain the latent representations of normal motions to limited volumes and detect as abnormal anything outside, which accounts satisfactorily for the openset'ness of anomalies. But normalcy shares the same openset'ness property, since humans can perform the same action in several ways, which the leading techniques neglect. We propose a novel generative model for video anomaly detection (VAD), which assumes that both normality and abnormality are multimodal. We consider skeletal representations and leverage state-of-the-art diffusion probabilistic models to generate multimodal future human poses. We contribute a novel conditioning on the past motion of people, and exploit the improved mode coverage capabilities of diffusion processes to generate different-but-plausible future motions. Upon the statistical aggregation of future modes, anomaly is detected when the generated set of motions is not pertinent to the actual future. We validate our model on 4 established benchmarks: UBnormal, HR-UBnormal, HR-STC, and HR-Avenue, with extensive experiments surpassing state-of-the-art results.

{{</citation>}}


### (22/99) LightFormer: An End-to-End Model for Intersection Right-of-Way Recognition Using Traffic Light Signals and an Attention Mechanism (Zhenxing Ming et al., 2023)

{{<citation>}}

Zhenxing Ming, Julie Stephany Berrio, Mao Shan, Eduardo Nebot, Stewart Worrall. (2023)  
**LightFormer: An End-to-End Model for Intersection Right-of-Way Recognition Using Traffic Light Signals and an Attention Mechanism**  

---
Primary Category: cs.CV
Categories: cs-CV, cs-RO, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2307.07196v1)  

---


**ABSTRACT**  
For smart vehicles driving through signalised intersections, it is crucial to determine whether the vehicle has right of way given the state of the traffic lights. To address this issue, camera based sensors can be used to determine whether the vehicle has permission to proceed straight, turn left or turn right. This paper proposes a novel end to end intersection right of way recognition model called LightFormer to generate right of way status for available driving directions in complex urban intersections. The model includes a spatial temporal inner structure with an attention mechanism, which incorporates features from past image to contribute to the classification of the current frame right of way status. In addition, a modified, multi weight arcface loss is introduced to enhance the model classification performance. Finally, the proposed LightFormer is trained and tested on two public traffic light datasets with manually augmented labels to demonstrate its effectiveness.

{{</citation>}}


### (23/99) TVPR: Text-to-Video Person Retrieval and a New Benchmark (Fan Ni et al., 2023)

{{<citation>}}

Fan Ni, Xu Zhang, Jianhui Wu, Guan-Nan Dong, Aichun Zhu, Hui Liu, Yue Zhang. (2023)  
**TVPR: Text-to-Video Person Retrieval and a New Benchmark**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2307.07184v1)  

---


**ABSTRACT**  
Most existing methods for text-based person retrieval focus on text-to-image person retrieval. Nevertheless, due to the lack of dynamic information provided by isolated frames, the performance is hampered when the person is obscured in isolated frames or variable motion details are given in the textual description. In this paper, we propose a new task called Text-to-Video Person Retrieval(TVPR) which aims to effectively overcome the limitations of isolated frames. Since there is no dataset or benchmark that describes person videos with natural language, we construct a large-scale cross-modal person video dataset containing detailed natural language annotations, such as person's appearance, actions and interactions with environment, etc., termed as Text-to-Video Person Re-identification (TVPReid) dataset, which will be publicly available. To this end, a Text-to-Video Person Retrieval Network (TVPRN) is proposed. Specifically, TVPRN acquires video representations by fusing visual and motion representations of person videos, which can deal with temporal occlusion and the absence of variable motion details in isolated frames. Meanwhile, we employ the pre-trained BERT to obtain caption representations and the relationship between caption and video representations to reveal the most relevant person videos. To evaluate the effectiveness of the proposed TVPRN, extensive experiments have been conducted on TVPReid dataset. To the best of our knowledge, TVPRN is the first successful attempt to use video for text-based person retrieval task and has achieved state-of-the-art performance on TVPReid dataset. The TVPReid dataset will be publicly available to benefit future research.

{{</citation>}}


### (24/99) TriFormer: A Multi-modal Transformer Framework For Mild Cognitive Impairment Conversion Prediction (Linfeng Liu et al., 2023)

{{<citation>}}

Linfeng Liu, Junyan Lyu, Siyu Liu, Xiaoying Tang, Shekhar S. Chandra, Fatima A. Nasrallah. (2023)  
**TriFormer: A Multi-modal Transformer Framework For Mild Cognitive Impairment Conversion Prediction**  

---
Primary Category: cs.CV
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.07177v1)  

---


**ABSTRACT**  
The prediction of mild cognitive impairment (MCI) conversion to Alzheimer's disease (AD) is important for early treatment to prevent or slow the progression of AD. To accurately predict the MCI conversion to stable MCI or progressive MCI, we propose Triformer, a novel transformer-based framework with three specialized transformers to incorporate multi-model data. Triformer uses I) an image transformer to extract multi-view image features from medical scans, II) a clinical transformer to embed and correlate multi-modal clinical data, and III) a modality fusion transformer that produces an accurate prediction based on fusing the outputs from the image and clinical transformers. Triformer is evaluated on the Alzheimer's Disease Neuroimaging Initiative (ANDI)1 and ADNI2 datasets and outperforms previous state-of-the-art single and multi-modal methods.

{{</citation>}}


### (25/99) Adaptive Region Selection for Active Learning in Whole Slide Image Semantic Segmentation (Jingna Qiu et al., 2023)

{{<citation>}}

Jingna Qiu, Frauke Wilm, Mathias Öttl, Maja Schlereth, Chang Liu, Tobias Heimann, Marc Aubreville, Katharina Breininger. (2023)  
**Adaptive Region Selection for Active Learning in Whole Slide Image Semantic Segmentation**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: Active Learning, Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2307.07168v1)  

---


**ABSTRACT**  
The process of annotating histological gigapixel-sized whole slide images (WSIs) at the pixel level for the purpose of training a supervised segmentation model is time-consuming. Region-based active learning (AL) involves training the model on a limited number of annotated image regions instead of requesting annotations of the entire images. These annotation regions are iteratively selected, with the goal of optimizing model performance while minimizing the annotated area. The standard method for region selection evaluates the informativeness of all square regions of a specified size and then selects a specific quantity of the most informative regions. We find that the efficiency of this method highly depends on the choice of AL step size (i.e., the combination of region size and the number of selected regions per WSI), and a suboptimal AL step size can result in redundant annotation requests or inflated computation costs. This paper introduces a novel technique for selecting annotation regions adaptively, mitigating the reliance on this AL hyperparameter. Specifically, we dynamically determine each region by first identifying an informative area and then detecting its optimal bounding box, as opposed to selecting regions of a uniform predefined shape and size as in the standard method. We evaluate our method using the task of breast cancer metastases segmentation on the public CAMELYON16 dataset and show that it consistently achieves higher sampling efficiency than the standard method across various AL step sizes. With only 2.6\% of tissue area annotated, we achieve full annotation performance and thereby substantially reduce the costs of annotating a WSI dataset. The source code is available at https://github.com/DeepMicroscopy/AdaptiveRegionSelection.

{{</citation>}}


### (26/99) CFI2P: Coarse-to-Fine Cross-Modal Correspondence Learning for Image-to-Point Cloud Registration (Gongxin Yao et al., 2023)

{{<citation>}}

Gongxin Yao, Yixin Xuan, Yiwei Chen, Yu Pan. (2023)  
**CFI2P: Coarse-to-Fine Cross-Modal Correspondence Learning for Image-to-Point Cloud Registration**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.07142v1)  

---


**ABSTRACT**  
In the context of image-to-point cloud registration, acquiring point-to-pixel correspondences presents a challenging task since the similarity between individual points and pixels is ambiguous due to the visual differences in data modalities. Nevertheless, the same object present in the two data formats can be readily identified from the local perspective of point sets and pixel patches. Motivated by this intuition, we propose a coarse-to-fine framework that emphasizes the establishment of correspondences between local point sets and pixel patches, followed by the refinement of results at both the point and pixel levels. On a coarse scale, we mimic the classic Visual Transformer to translate both image and point cloud into two sequences of local representations, namely point and pixel proxies, and employ attention to capture global and cross-modal contexts. To supervise the coarse matching, we propose a novel projected point proportion loss, which guides to match point sets with pixel patches where more points can be projected into. On a finer scale, point-to-pixel correspondences are then refined from a smaller search space (i.e., the coarsely matched sets and patches) via well-designed sampling, attentional learning and fine matching, where sampling masks are embedded in the last two steps to mitigate the negative effect of sampling. With the high-quality correspondences, the registration problem is then resolved by EPnP algorithm within RANSAC. Experimental results on large-scale outdoor benchmarks demonstrate our superiority over existing methods.

{{</citation>}}


## cs.DS (1)



### (27/99) Zip-zip Trees: Making Zip Trees More Balanced, Biased, Compact, or Persistent (Ofek Gila et al., 2023)

{{<citation>}}

Ofek Gila, Michael T. Goodrich, Robert E. Tarjan. (2023)  
**Zip-zip Trees: Making Zip Trees More Balanced, Biased, Compact, or Persistent**  

---
Primary Category: cs.DS
Categories: E-1, cs-DS, cs.DS  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2307.07660v1)  

---


**ABSTRACT**  
We define simple variants of zip trees, called zip-zip trees, which provide several advantages over zip trees, including overcoming a bias that favors smaller keys over larger ones. We analyze zip-zip trees theoretically and empirically, showing, e.g., that the expected depth of a node in an $n$-node zip-zip tree is at most $1.3863\log n-1+o(1)$, which matches the expected depth of treaps and binary search trees built by uniformly random insertions. Unlike these other data structures, however, zip-zip trees achieve their bounds using only $O(\log\log n)$ bits of metadata per node, w.h.p., as compared to the $\Theta(\log n)$ bits per node required by treaps. In fact, we even describe a ``just-in-time'' zip-zip tree variant, which needs just an expected $O(1)$ number of bits of metadata per node. Moreover, we can define zip-zip trees to be strongly history independent, whereas treaps are generally only weakly history independent. We also introduce \emph{biased zip-zip trees}, which have an explicit bias based on key weights, so the expected depth of a key, $k$, with weight, $w_k$, is $O(\log (W/w_k))$, where $W$ is the weight of all keys in the weighted zip-zip tree. Finally, we show that one can easily make zip-zip trees partially persistent with only $O(n)$ space overhead w.h.p.

{{</citation>}}


## cs.LG (17)



### (28/99) DistTGL: Distributed Memory-Based Temporal Graph Neural Network Training (Hongkuan Zhou et al., 2023)

{{<citation>}}

Hongkuan Zhou, Da Zheng, Xiang Song, George Karypis, Viktor Prasanna. (2023)  
**DistTGL: Distributed Memory-Based Temporal Graph Neural Network Training**  

---
Primary Category: cs.LG
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2307.07649v1)  

---


**ABSTRACT**  
Memory-based Temporal Graph Neural Networks are powerful tools in dynamic graph representation learning and have demonstrated superior performance in many real-world applications. However, their node memory favors smaller batch sizes to capture more dependencies in graph events and needs to be maintained synchronously across all trainers. As a result, existing frameworks suffer from accuracy loss when scaling to multiple GPUs. Evenworse, the tremendous overhead to synchronize the node memory make it impractical to be deployed to distributed GPU clusters. In this work, we propose DistTGL -- an efficient and scalable solution to train memory-based TGNNs on distributed GPU clusters. DistTGL has three improvements over existing solutions: an enhanced TGNN model, a novel training algorithm, and an optimized system. In experiments, DistTGL achieves near-linear convergence speedup, outperforming state-of-the-art single-machine method by 14.5% in accuracy and 10.17x in training throughput.

{{</citation>}}


### (29/99) Generalizable Embeddings with Cross-batch Metric Learning (Yeti Z. Gurbuz et al., 2023)

{{<citation>}}

Yeti Z. Gurbuz, A. Aydin Alatan. (2023)  
**Generalizable Embeddings with Cross-batch Metric Learning**  

---
Primary Category: cs.LG
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2307.07620v1)  

---


**ABSTRACT**  
Global average pooling (GAP) is a popular component in deep metric learning (DML) for aggregating features. Its effectiveness is often attributed to treating each feature vector as a distinct semantic entity and GAP as a combination of them. Albeit substantiated, such an explanation's algorithmic implications to learn generalizable entities to represent unseen classes, a crucial DML goal, remain unclear. To address this, we formulate GAP as a convex combination of learnable prototypes. We then show that the prototype learning can be expressed as a recursive process fitting a linear predictor to a batch of samples. Building on that perspective, we consider two batches of disjoint classes at each iteration and regularize the learning by expressing the samples of a batch with the prototypes that are fitted to the other batch. We validate our approach on 4 popular DML benchmarks.

{{</citation>}}


### (30/99) Population Expansion for Training Language Models with Private Federated Learning (Tatsuki Koga et al., 2023)

{{<citation>}}

Tatsuki Koga, Congzheng Song, Martin Pelikan, Mona Chitnis. (2023)  
**Population Expansion for Training Language Models with Private Federated Learning**  

---
Primary Category: cs.LG
Categories: cs-CL, cs-CR, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.07477v1)  

---


**ABSTRACT**  
Federated learning (FL) combined with differential privacy (DP) offers machine learning (ML) training with distributed devices and with a formal privacy guarantee. With a large population of devices, FL with DP produces a performant model in a timely manner. However, for applications with a smaller population, not only does the model utility degrade as the DP noise is inversely proportional to population, but also the training latency increases since waiting for enough clients to become available from a smaller pool is slower. In this work, we thus propose expanding the population based on domain adaptation techniques to speed up the training and improves the final model quality when training with small populations. We empirically demonstrate that our techniques can improve the utility by 13% to 30% on real-world language modeling datasets.

{{</citation>}}


### (31/99) Structured Pruning of Neural Networks for Constraints Learning (Matteo Cacciola et al., 2023)

{{<citation>}}

Matteo Cacciola, Antonio Frangioni, Andrea Lodi. (2023)  
**Structured Pruning of Neural Networks for Constraints Learning**  

---
Primary Category: cs.LG
Categories: cs-AI, cs-LG, cs.LG, math-OC  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2307.07457v1)  

---


**ABSTRACT**  
In recent years, the integration of Machine Learning (ML) models with Operation Research (OR) tools has gained popularity across diverse applications, including cancer treatment, algorithmic configuration, and chemical process optimization. In this domain, the combination of ML and OR often relies on representing the ML model output using Mixed Integer Programming (MIP) formulations. Numerous studies in the literature have developed such formulations for many ML predictors, with a particular emphasis on Artificial Neural Networks (ANNs) due to their significant interest in many applications. However, ANNs frequently contain a large number of parameters, resulting in MIP formulations that are impractical to solve, thereby impeding scalability. In fact, the ML community has already introduced several techniques to reduce the parameter count of ANNs without compromising their performance, since the substantial size of modern ANNs presents challenges for ML applications as it significantly impacts computational efforts during training and necessitates significant memory resources for storage. In this paper, we showcase the effectiveness of pruning, one of these techniques, when applied to ANNs prior to their integration into MIPs. By pruning the ANN, we achieve significant improvements in the speed of the solution process. We discuss why pruning is more suitable in this context compared to other ML compression techniques, and we identify the most appropriate pruning strategies. To highlight the potential of this approach, we conduct experiments using feed-forward neural networks with multiple layers to construct adversarial examples. Our results demonstrate that pruning offers remarkable reductions in solution times without hindering the quality of the final decision, enabling the resolution of previously unsolvable instances.

{{</citation>}}


### (32/99) Can Large Language Models Empower Molecular Property Prediction? (Chen Qian et al., 2023)

{{<citation>}}

Chen Qian, Huayi Tang, Zhirui Yang, Hong Liang, Yong Liu. (2023)  
**Can Large Language Models Empower Molecular Property Prediction?**  

---
Primary Category: cs.LG
Categories: cs-AI, cs-LG, cs.LG, q-bio-QM  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2307.07443v1)  

---


**ABSTRACT**  
Molecular property prediction has gained significant attention due to its transformative potential in multiple scientific disciplines. Conventionally, a molecule graph can be represented either as a graph-structured data or a SMILES text. Recently, the rapid development of Large Language Models (LLMs) has revolutionized the field of NLP. Although it is natural to utilize LLMs to assist in understanding molecules represented by SMILES, the exploration of how LLMs will impact molecular property prediction is still in its early stage. In this work, we advance towards this objective through two perspectives: zero/few-shot molecular classification, and using the new explanations generated by LLMs as representations of molecules. To be specific, we first prompt LLMs to do in-context molecular classification and evaluate their performance. After that, we employ LLMs to generate semantically enriched explanations for the original SMILES and then leverage that to fine-tune a small-scale LM model for multiple downstream tasks. The experimental results highlight the superiority of text explanations as molecular representations across multiple benchmark datasets, and confirm the immense potential of LLMs in molecular property prediction tasks. Codes are available at \url{https://github.com/ChnQ/LLM4Mol}.

{{</citation>}}


### (33/99) Exploiting Counter-Examples for Active Learning with Partial labels (Fei Zhang et al., 2023)

{{<citation>}}

Fei Zhang, Yunjie Ye, Lei Feng, Zhongwen Rao, Jieming Zhu, Marcus Kalander, Chen Gong, Jianye Hao, Bo Han. (2023)  
**Exploiting Counter-Examples for Active Learning with Partial labels**  

---
Primary Category: cs.LG
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2307.07413v1)  

---


**ABSTRACT**  
This paper studies a new problem, \emph{active learning with partial labels} (ALPL). In this setting, an oracle annotates the query samples with partial labels, relaxing the oracle from the demanding accurate labeling process. To address ALPL, we first build an intuitive baseline that can be seamlessly incorporated into existing AL frameworks. Though effective, this baseline is still susceptible to the \emph{overfitting}, and falls short of the representative partial-label-based samples during the query process. Drawing inspiration from human inference in cognitive science, where accurate inferences can be explicitly derived from \emph{counter-examples} (CEs), our objective is to leverage this human-like learning pattern to tackle the \emph{overfitting} while enhancing the process of selecting representative samples in ALPL. Specifically, we construct CEs by reversing the partial labels for each instance, and then we propose a simple but effective WorseNet to directly learn from this complementary pattern. By leveraging the distribution gap between WorseNet and the predictor, this adversarial evaluation manner could enhance both the performance of the predictor itself and the sample selection process, allowing the predictor to capture more accurate patterns in the data. Experimental results on five real-world datasets and four benchmark datasets show that our proposed method achieves comprehensive improvements over ten representative AL frameworks, highlighting the superiority of WorseNet. The source code will be available at \url{https://github.com/Ferenas/APLL}.

{{</citation>}}


### (34/99) HuCurl: Human-induced Curriculum Discovery (Mohamed Elgaar et al., 2023)

{{<citation>}}

Mohamed Elgaar, Hadi Amiri. (2023)  
**HuCurl: Human-induced Curriculum Discovery**  

---
Primary Category: cs.LG
Categories: cs-CL, cs-LG, cs.LG  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2307.07412v1)  

---


**ABSTRACT**  
We introduce the problem of curriculum discovery and describe a curriculum learning framework capable of discovering effective curricula in a curriculum space based on prior knowledge about sample difficulty. Using annotation entropy and loss as measures of difficulty, we show that (i): the top-performing discovered curricula for a given model and dataset are often non-monotonic as opposed to monotonic curricula in existing literature, (ii): the prevailing easy-to-hard or hard-to-easy transition curricula are often at the risk of underperforming, and (iii): the curricula discovered for smaller datasets and models perform well on larger datasets and models respectively. The proposed framework encompasses some of the existing curriculum learning approaches and can discover curricula that outperform them across several NLP tasks.

{{</citation>}}


### (35/99) Performance of $\ell_1$ Regularization for Sparse Convex Optimization (Kyriakos Axiotis et al., 2023)

{{<citation>}}

Kyriakos Axiotis, Taisuke Yasuda. (2023)  
**Performance of $\ell_1$ Regularization for Sparse Convex Optimization**  

---
Primary Category: cs.LG
Categories: cs-DS, cs-LG, cs.LG, math-ST, stat-ML, stat-TH  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2307.07405v1)  

---


**ABSTRACT**  
Despite widespread adoption in practice, guarantees for the LASSO and Group LASSO are strikingly lacking in settings beyond statistical problems, and these algorithms are usually considered to be a heuristic in the context of sparse convex optimization on deterministic inputs. We give the first recovery guarantees for the Group LASSO for sparse convex optimization with vector-valued features. We show that if a sufficiently large Group LASSO regularization is applied when minimizing a strictly convex function $l$, then the minimizer is a sparse vector supported on vector-valued features with the largest $\ell_2$ norm of the gradient. Thus, repeating this procedure selects the same set of features as the Orthogonal Matching Pursuit algorithm, which admits recovery guarantees for any function $l$ with restricted strong convexity and smoothness via weak submodularity arguments. This answers open questions of Tibshirani et al. and Yasuda et al. Our result is the first to theoretically explain the empirical success of the Group LASSO for convex functions under general input instances assuming only restricted strong convexity and smoothness. Our result also generalizes provable guarantees for the Sequential Attention algorithm, which is a feature selection algorithm inspired by the attention mechanism proposed by Yasuda et al.   As an application of our result, we give new results for the column subset selection problem, which is well-studied when the loss is the Frobenius norm or other entrywise matrix losses. We give the first result for general loss functions for this problem that requires only restricted strong convexity and smoothness.

{{</citation>}}


### (36/99) On Interpolating Experts and Multi-Armed Bandits (Houshuang Chen et al., 2023)

{{<citation>}}

Houshuang Chen, Yuchen He, Chihao Zhang. (2023)  
**On Interpolating Experts and Multi-Armed Bandits**  

---
Primary Category: cs.LG
Categories: cs-DS, cs-LG, cs.LG, stat-ML  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.07264v1)  

---


**ABSTRACT**  
Learning with expert advice and multi-armed bandit are two classic online decision problems which differ on how the information is observed in each round of the game. We study a family of problems interpolating the two. For a vector $\mathbf{m}=(m_1,\dots,m_K)\in \mathbb{N}^K$, an instance of $\mathbf{m}$-MAB indicates that the arms are partitioned into $K$ groups and the $i$-th group contains $m_i$ arms. Once an arm is pulled, the losses of all arms in the same group are observed. We prove tight minimax regret bounds for $\mathbf{m}$-MAB and design an optimal PAC algorithm for its pure exploration version, $\mathbf{m}$-BAI, where the goal is to identify the arm with minimum loss with as few rounds as possible. We show that the minimax regret of $\mathbf{m}$-MAB is $\Theta\left(\sqrt{T\sum_{k=1}^K\log (m_k+1)}\right)$ and the minimum number of pulls for an $(\epsilon,0.05)$-PAC algorithm of $\mathbf{m}$-BAI is $\Theta\left(\frac{1}{\epsilon^2}\cdot \sum_{k=1}^K\log (m_k+1)\right)$. Both our upper bounds and lower bounds for $\mathbf{m}$-MAB can be extended to a more general setting, namely the bandit with graph feedback, in terms of the clique cover and related graph parameters. As consequences, we obtained tight minimax regret bounds for several families of feedback graphs.

{{</citation>}}


### (37/99) Mitigating Adversarial Vulnerability through Causal Parameter Estimation by Adversarial Double Machine Learning (Byung-Kwan Lee et al., 2023)

{{<citation>}}

Byung-Kwan Lee, Junho Kim, Yong Man Ro. (2023)  
**Mitigating Adversarial Vulnerability through Causal Parameter Estimation by Adversarial Double Machine Learning**  

---
Primary Category: cs.LG
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.07250v2)  

---


**ABSTRACT**  
Adversarial examples derived from deliberately crafted perturbations on visual inputs can easily harm decision process of deep neural networks. To prevent potential threats, various adversarial training-based defense methods have grown rapidly and become a de facto standard approach for robustness. Despite recent competitive achievements, we observe that adversarial vulnerability varies across targets and certain vulnerabilities remain prevalent. Intriguingly, such peculiar phenomenon cannot be relieved even with deeper architectures and advanced defense methods. To address this issue, in this paper, we introduce a causal approach called Adversarial Double Machine Learning (ADML), which allows us to quantify the degree of adversarial vulnerability for network predictions and capture the effect of treatments on outcome of interests. ADML can directly estimate causal parameter of adversarial perturbations per se and mitigate negative effects that can potentially damage robustness, bridging a causal perspective into the adversarial vulnerability. Through extensive experiments on various CNN and Transformer architectures, we corroborate that ADML improves adversarial robustness with large margins and relieve the empirical observation.

{{</citation>}}


### (38/99) Omnipotent Adversarial Training for Unknown Label-noisy and Imbalanced Datasets (Guanlin Li et al., 2023)

{{<citation>}}

Guanlin Li, Kangjie Chen, Yuan Xu, Han Qiu, Tianwei Zhang. (2023)  
**Omnipotent Adversarial Training for Unknown Label-noisy and Imbalanced Datasets**  

---
Primary Category: cs.LG
Categories: cs-CR, cs-CV, cs-LG, cs.LG  
Keywords: Adversarial Training  
[Paper Link](http://arxiv.org/abs/2307.08596v1)  

---


**ABSTRACT**  
Adversarial training is an important topic in robust deep learning, but the community lacks attention to its practical usage. In this paper, we aim to resolve a real-world application challenge, i.e., training a model on an imbalanced and noisy dataset to achieve high clean accuracy and robustness, with our proposed Omnipotent Adversarial Training (OAT). Our strategy consists of two innovative methodologies to address the label noise and data imbalance in the training set. We first introduce an oracle into the adversarial training process to help the model learn a correct data-label conditional distribution. This carefully-designed oracle can provide correct label annotations for adversarial training. We further propose logits adjustment adversarial training to overcome the data imbalance challenge, which can help the model learn a Bayes-optimal distribution. Our comprehensive evaluation results show that OAT outperforms other baselines by more than 20% clean accuracy improvement and 10% robust accuracy improvement under the complex combinations of data imbalance and label noise scenarios. The code can be found in https://github.com/GuanlinLee/OAT.

{{</citation>}}


### (39/99) Adversarial Training Over Long-Tailed Distribution (Guanlin Li et al., 2023)

{{<citation>}}

Guanlin Li, Guowen Xu, Tianwei Zhang. (2023)  
**Adversarial Training Over Long-Tailed Distribution**  

---
Primary Category: cs.LG
Categories: cs-CR, cs-CV, cs-LG, cs.LG  
Keywords: Adversarial Training  
[Paper Link](http://arxiv.org/abs/2307.10205v1)  

---


**ABSTRACT**  
In this paper, we study adversarial training on datasets that obey the long-tailed distribution, which is practical but rarely explored in previous works. Compared with conventional adversarial training on balanced datasets, this process falls into the dilemma of generating uneven adversarial examples (AEs) and an unbalanced feature embedding space, causing the resulting model to exhibit low robustness and accuracy on tail data. To combat that, we propose a new adversarial training framework -- Re-balancing Adversarial Training (REAT). This framework consists of two components: (1) a new training strategy inspired by the term effective number to guide the model to generate more balanced and informative AEs; (2) a carefully constructed penalty function to force a satisfactory feature space. Evaluation results on different datasets and model structures prove that REAT can effectively enhance the model's robustness and preserve the model's clean accuracy. The code can be found in https://github.com/GuanlinLee/REAT.

{{</citation>}}


### (40/99) Safe DreamerV3: Safe Reinforcement Learning with World Models (Weidong Huang et al., 2023)

{{<citation>}}

Weidong Huang, Jiaming Ji, Borong Zhang, Chunhe Xia, Yaodong Yang. (2023)  
**Safe DreamerV3: Safe Reinforcement Learning with World Models**  

---
Primary Category: cs.LG
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.07176v1)  

---


**ABSTRACT**  
The widespread application of Reinforcement Learning (RL) in real-world situations is yet to come to fruition, largely as a result of its failure to satisfy the essential safety demands of such systems. Existing safe reinforcement learning (SafeRL) methods, employing cost functions to enhance safety, fail to achieve zero-cost in complex scenarios, including vision-only tasks, even with comprehensive data sampling and training. To address this, we introduce Safe DreamerV3, a novel algorithm that integrates both Lagrangian-based and planning-based methods within a world model. Our methodology represents a significant advancement in SafeRL as the first algorithm to achieve nearly zero-cost in both low-dimensional and vision-only tasks within the Safety-Gymnasium benchmark. Our project website can be found in: https://sites.google.com/view/safedreamerv3.

{{</citation>}}


### (41/99) HYTREL: Hypergraph-enhanced Tabular Data Representation Learning (Pei Chen et al., 2023)

{{<citation>}}

Pei Chen, Soumajyoti Sarkar, Leonard Lausen, Balasubramaniam Srinivasan, Sheng Zha, Ruihong Huang, George Karypis. (2023)  
**HYTREL: Hypergraph-enhanced Tabular Data Representation Learning**  

---
Primary Category: cs.LG
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2307.08623v1)  

---


**ABSTRACT**  
Language models pretrained on large collections of tabular data have demonstrated their effectiveness in several downstream tasks. However, many of these models do not take into account the row/column permutation invariances, hierarchical structure, etc. that exist in tabular data. To alleviate these limitations, we propose HYTREL, a tabular language model, that captures the permutation invariances and three more structural properties of tabular data by using hypergraphs - where the table cells make up the nodes and the cells occurring jointly together in each row, column, and the entire table are used to form three different types of hyperedges. We show that HYTREL is maximally invariant under certain conditions for tabular data, i.e., two tables obtain the same representations via HYTREL iff the two tables are identical up to permutations. Our empirical results demonstrate that HYTREL consistently outperforms other competitive baselines on four downstream tasks with minimal pretraining, illustrating the advantages of incorporating the inductive biases associated with tabular data into the representations. Finally, our qualitative analyses showcase that HYTREL can assimilate the table structures to generate robust representations for the cells, rows, columns, and the entire table.

{{</citation>}}


### (42/99) Vulnerability-Aware Instance Reweighting For Adversarial Training (Olukorede Fakorede et al., 2023)

{{<citation>}}

Olukorede Fakorede, Ashutosh Kumar Nirala, Modeste Atsague, Jin Tian. (2023)  
**Vulnerability-Aware Instance Reweighting For Adversarial Training**  

---
Primary Category: cs.LG
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: Adversarial Training  
[Paper Link](http://arxiv.org/abs/2307.07167v1)  

---


**ABSTRACT**  
Adversarial Training (AT) has been found to substantially improve the robustness of deep learning classifiers against adversarial attacks. AT involves obtaining robustness by including adversarial examples in training a classifier. Most variants of AT algorithms treat every training example equally. However, recent works have shown that better performance is achievable by treating them unequally. In addition, it has been observed that AT exerts an uneven influence on different classes in a training set and unfairly hurts examples corresponding to classes that are inherently harder to classify. Consequently, various reweighting schemes have been proposed that assign unequal weights to robust losses of individual examples in a training set. In this work, we propose a novel instance-wise reweighting scheme. It considers the vulnerability of each natural example and the resulting information loss on its adversarial counterpart occasioned by adversarial attacks. Through extensive experiments, we show that our proposed method significantly improves over existing reweighting schemes, especially against strong white and black-box attacks.

{{</citation>}}


### (43/99) Looking deeper into interpretable deep learning in neuroimaging: a comprehensive survey (Md. Mahfuzur Rahman et al., 2023)

{{<citation>}}

Md. Mahfuzur Rahman, Vince D. Calhoun, Sergey M. Plis. (2023)  
**Looking deeper into interpretable deep learning in neuroimaging: a comprehensive survey**  

---
Primary Category: cs.LG
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.09615v1)  

---


**ABSTRACT**  
Deep learning (DL) models have been popular due to their ability to learn directly from the raw data in an end-to-end paradigm, alleviating the concern of a separate error-prone feature extraction phase. Recent DL-based neuroimaging studies have also witnessed a noticeable performance advancement over traditional machine learning algorithms. But the challenges of deep learning models still exist because of the lack of transparency in these models for their successful deployment in real-world applications. In recent years, Explainable AI (XAI) has undergone a surge of developments mainly to get intuitions of how the models reached the decisions, which is essential for safety-critical domains such as healthcare, finance, and law enforcement agencies. While the interpretability domain is advancing noticeably, researchers are still unclear about what aspect of model learning a post hoc method reveals and how to validate its reliability. This paper comprehensively reviews interpretable deep learning models in the neuroimaging domain. Firstly, we summarize the current status of interpretability resources in general, focusing on the progression of methods, associated challenges, and opinions. Secondly, we discuss how multiple recent neuroimaging studies leveraged model interpretability to capture anatomical and functional brain alterations most relevant to model predictions. Finally, we discuss the limitations of the current practices and offer some valuable insights and guidance on how we can steer our future research directions to make deep learning models substantially interpretable and thus advance scientific understanding of brain disorders.

{{</citation>}}


### (44/99) Graph Positional and Structural Encoder (Renming Liu et al., 2023)

{{<citation>}}

Renming Liu, Semih Cantürk, Olivier Lapointe-Gagné, Vincent Létourneau, Guy Wolf, Dominique Beaini, Ladislav Rampášek. (2023)  
**Graph Positional and Structural Encoder**  

---
Primary Category: cs.LG
Categories: cs-LG, cs.LG  
Keywords: GNN, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.07107v1)  

---


**ABSTRACT**  
Positional and structural encodings (PSE) enable better identifiability of nodes within a graph, as in general graphs lack a canonical node ordering. This renders PSEs essential tools for empowering modern GNNs, and in particular graph Transformers. However, designing PSEs that work optimally for a variety of graph prediction tasks is a challenging and unsolved problem. Here, we present the graph positional and structural encoder (GPSE), a first-ever attempt to train a graph encoder that captures rich PSE representations for augmenting any GNN. GPSE can effectively learn a common latent representation for multiple PSEs, and is highly transferable. The encoder trained on a particular graph dataset can be used effectively on datasets drawn from significantly different distributions and even modalities. We show that across a wide range of benchmarks, GPSE-enhanced models can significantly improve the performance in certain tasks, while performing on par with those that employ explicitly computed PSEs in other cases. Our results pave the way for the development of large pre-trained models for extracting graph positional and structural information and highlight their potential as a viable alternative to explicitly computed PSEs as well as to existing self-supervised pre-training approaches.

{{</citation>}}


## cs.AI (3)



### (45/99) `It is currently hodgepodge'': Examining AI/ML Practitioners' Challenges during Co-production of Responsible AI Values (Rama Adithya Varanasi et al., 2023)

{{<citation>}}

Rama Adithya Varanasi, Nitesh Goyal. (2023)  
**`It is currently hodgepodge'': Examining AI/ML Practitioners' Challenges during Co-production of Responsible AI Values**  

---
Primary Category: cs.AI
Categories: I-2; K-4, cs-AI, cs-HC, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.10221v1)  

---


**ABSTRACT**  
Recently, the AI/ML research community has indicated an urgent need to establish Responsible AI (RAI) values and practices as part of the AI/ML lifecycle. Several organizations and communities are responding to this call by sharing RAI guidelines. However, there are gaps in awareness, deliberation, and execution of such practices for multi-disciplinary ML practitioners. This work contributes to the discussion by unpacking co-production challenges faced by practitioners as they align their RAI values. We interviewed 23 individuals, across 10 organizations, tasked to ship AI/ML based products while upholding RAI norms and found that both top-down and bottom-up institutional structures create burden for different roles preventing them from upholding RAI values, a challenge that is further exacerbated when executing conflicted values. We share multiple value levers used as strategies by the practitioners to resolve their challenges. We end our paper with recommendations for inclusive and equitable RAI value-practices, creating supportive organizational structures and opportunities to further aid practitioners.

{{</citation>}}


### (46/99) Exploring Link Prediction over Hyper-Relational Temporal Knowledge Graphs Enhanced with Time-Invariant Relational Knowledge (Zifeng Ding et al., 2023)

{{<citation>}}

Zifeng Ding, Jingcheng Wu, Jingpei Wu, Yan Xia, Volker Tresp. (2023)  
**Exploring Link Prediction over Hyper-Relational Temporal Knowledge Graphs Enhanced with Time-Invariant Relational Knowledge**  

---
Primary Category: cs.AI
Categories: cs-AI, cs-LG, cs.AI  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2307.10219v1)  

---


**ABSTRACT**  
Stemming from traditional knowledge graphs (KGs), hyper-relational KGs (HKGs) provide additional key-value pairs (i.e., qualifiers) for each KG fact that help to better restrict the fact validity. In recent years, there has been an increasing interest in studying graph reasoning over HKGs. In the meantime, due to the ever-evolving nature of world knowledge, extensive parallel works have been focusing on reasoning over temporal KGs (TKGs), where each TKG fact can be viewed as a KG fact coupled with a timestamp (or time period) specifying its time validity. The existing HKG reasoning approaches do not consider temporal information because it is not explicitly specified in previous benchmark datasets. Besides, all the previous TKG reasoning methods only lay emphasis on temporal reasoning and have no way to learn from qualifiers. To this end, we aim to fill the gap between TKG reasoning and HKG reasoning. We develop two new benchmark hyper-relational TKG (HTKG) datasets, i.e., Wiki-hy and YAGO-hy, and propose a HTKG reasoning model that efficiently models both temporal facts and qualifiers. We further exploit additional time-invariant relational knowledge from the Wikidata knowledge base and study its effectiveness in HTKG reasoning. Time-invariant relational knowledge serves as the knowledge that remains unchanged in time (e.g., Sasha Obama is the child of Barack Obama), and it has never been fully explored in previous TKG reasoning benchmarks and approaches. Experimental results show that our model substantially outperforms previous related methods on HTKG link prediction and can be enhanced by jointly leveraging both temporal and time-invariant relational knowledge.

{{</citation>}}


### (47/99) Value-based Fast and Slow AI Nudging (Marianna B. Ganapini et al., 2023)

{{<citation>}}

Marianna B. Ganapini, Francesco Fabiano, Lior Horesh, Andrea Loreggia, Nicholas Mattei, Keerthiram Murugesan, Vishal Pallagani, Francesca Rossi, Biplav Srivastava, Brent Venable. (2023)  
**Value-based Fast and Slow AI Nudging**  

---
Primary Category: cs.AI
Categories: cs-AI, cs-CY, cs-HC, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.07628v1)  

---


**ABSTRACT**  
Nudging is a behavioral strategy aimed at influencing people's thoughts and actions. Nudging techniques can be found in many situations in our daily lives, and these nudging techniques can targeted at human fast and unconscious thinking, e.g., by using images to generate fear or the more careful and effortful slow thinking, e.g., by releasing information that makes us reflect on our choices. In this paper, we propose and discuss a value-based AI-human collaborative framework where AI systems nudge humans by proposing decision recommendations. Three different nudging modalities, based on when recommendations are presented to the human, are intended to stimulate human fast thinking, slow thinking, or meta-cognition. Values that are relevant to a specific decision scenario are used to decide when and how to use each of these nudging modalities. Examples of values are decision quality, speed, human upskilling and learning, human agency, and privacy. Several values can be present at the same time, and their priorities can vary over time. The framework treats values as parameters to be instantiated in a specific decision environment.

{{</citation>}}


## cs.CL (19)



### (48/99) QontSum: On Contrasting Salient Content for Query-focused Summarization (Sajad Sotudeh et al., 2023)

{{<citation>}}

Sajad Sotudeh, Nazli Goharian. (2023)  
**QontSum: On Contrasting Salient Content for Query-focused Summarization**  

---
Primary Category: cs.CL
Categories: cs-CL, cs.CL  
Keywords: Information Retrieval, Summarization  
[Paper Link](http://arxiv.org/abs/2307.07586v1)  

---


**ABSTRACT**  
Query-focused summarization (QFS) is a challenging task in natural language processing that generates summaries to address specific queries. The broader field of Generative Information Retrieval (Gen-IR) aims to revolutionize information extraction from vast document corpora through generative approaches, encompassing Generative Document Retrieval (GDR) and Grounded Answer Retrieval (GAR). This paper highlights the role of QFS in Grounded Answer Generation (GAR), a key subdomain of Gen-IR that produces human-readable answers in direct correspondence with queries, grounded in relevant documents. In this study, we propose QontSum, a novel approach for QFS that leverages contrastive learning to help the model attend to the most relevant regions of the input document. We evaluate our approach on a couple of benchmark datasets for QFS and demonstrate that it either outperforms existing state-of-the-art or exhibits a comparable performance with considerably reduced computational cost through enhancements in the fine-tuning stage, rather than relying on large-scale pre-training experiments, which is the focus of current SOTA. Moreover, we conducted a human study and identified improvements in the relevance of generated summaries to the posed queries without compromising fluency. We further conduct an error analysis study to understand our model's limitations and propose avenues for future research.

{{</citation>}}


### (49/99) Towards spoken dialect identification of Irish (Liam Lonergan et al., 2023)

{{<citation>}}

Liam Lonergan, Mengjie Qian, Neasa Ní Chiaráin, Christer Gobl, Ailbhe Ní Chasaide. (2023)  
**Towards spoken dialect identification of Irish**  

---
Primary Category: cs.CL
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2307.07436v1)  

---


**ABSTRACT**  
The Irish language is rich in its diversity of dialects and accents. This compounds the difficulty of creating a speech recognition system for the low-resource language, as such a system must contend with a high degree of variability with limited corpora. A recent study investigating dialect bias in Irish ASR found that balanced training corpora gave rise to unequal dialect performance, with performance for the Ulster dialect being consistently worse than for the Connacht or Munster dialects. Motivated by this, the present experiments investigate spoken dialect identification of Irish, with a view to incorporating such a system into the speech recognition pipeline. Two acoustic classification models are tested, XLS-R and ECAPA-TDNN, in conjunction with a text-based classifier using a pretrained Irish-language BERT model. The ECAPA-TDNN, particularly a model pretrained for language identification on the VoxLingua107 dataset, performed best overall, with an accuracy of 73%. This was further improved to 76% by fusing the model's outputs with the text-based model. The Ulster dialect was most accurately identified, with an accuracy of 94%, however the model struggled to disambiguate between the Connacht and Munster dialects, suggesting a more nuanced approach may be necessary to robustly distinguish between the dialects of Irish.

{{</citation>}}


### (50/99) Rank Your Summaries: Enhancing Bengali Text Summarization via Ranking-based Approach (G. M. Shahariar et al., 2023)

{{<citation>}}

G. M. Shahariar, Tonmoy Talukder, Rafin Alam Khan Sotez, Md. Tanvir Rouf Shawon. (2023)  
**Rank Your Summaries: Enhancing Bengali Text Summarization via Ranking-based Approach**  

---
Primary Category: cs.CL
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, BLEU, Summarization, Text Summarization  
[Paper Link](http://arxiv.org/abs/2307.07392v1)  

---


**ABSTRACT**  
With the increasing need for text summarization techniques that are both efficient and accurate, it becomes crucial to explore avenues that enhance the quality and precision of pre-trained models specifically tailored for summarizing Bengali texts. When it comes to text summarization tasks, there are numerous pre-trained transformer models at one's disposal. Consequently, it becomes quite a challenge to discern the most informative and relevant summary for a given text among the various options generated by these pre-trained summarization models. This paper aims to identify the most accurate and informative summary for a given text by utilizing a simple but effective ranking-based approach that compares the output of four different pre-trained Bengali text summarization models. The process begins by carrying out preprocessing of the input text that involves eliminating unnecessary elements such as special characters and punctuation marks. Next, we utilize four pre-trained summarization models to generate summaries, followed by applying a text ranking algorithm to identify the most suitable summary. Ultimately, the summary with the highest ranking score is chosen as the final one. To evaluate the effectiveness of this approach, the generated summaries are compared against human-annotated summaries using standard NLG metrics such as BLEU, ROUGE, BERTScore, WIL, WER, and METEOR. Experimental results suggest that by leveraging the strengths of each pre-trained transformer model and combining them using a ranking-based approach, our methodology significantly improves the accuracy and effectiveness of the Bengali text summarization.

{{</citation>}}


### (51/99) Composition-contrastive Learning for Sentence Embeddings (Sachin J. Chanchani et al., 2023)

{{<citation>}}

Sachin J. Chanchani, Ruihong Huang. (2023)  
**Composition-contrastive Learning for Sentence Embeddings**  

---
Primary Category: cs.CL
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Embedding, Sentence Embedding  
[Paper Link](http://arxiv.org/abs/2307.07380v1)  

---


**ABSTRACT**  
Vector representations of natural language are ubiquitous in search applications. Recently, various methods based on contrastive learning have been proposed to learn textual representations from unlabelled data; by maximizing alignment between minimally-perturbed embeddings of the same text, and encouraging a uniform distribution of embeddings across a broader corpus. Differently, we propose maximizing alignment between texts and a composition of their phrasal constituents. We consider several realizations of this objective and elaborate the impact on representations in each case. Experimental results on semantic textual similarity tasks show improvements over baselines that are comparable with state-of-the-art approaches. Moreover, this work is the first to do so without incurring costs in auxiliary training objectives or additional network parameters.

{{</citation>}}


### (52/99) Mitigating Bias in Conversations: A Hate Speech Classifier and Debiaser with Prompts (Shaina Raza et al., 2023)

{{<citation>}}

Shaina Raza, Chen Ding, Deval Pandya. (2023)  
**Mitigating Bias in Conversations: A Hate Speech Classifier and Debiaser with Prompts**  

---
Primary Category: cs.CL
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2307.10213v1)  

---


**ABSTRACT**  
Discriminatory language and biases are often present in hate speech during conversations, which usually lead to negative impacts on targeted groups such as those based on race, gender, and religion. To tackle this issue, we propose an approach that involves a two-step process: first, detecting hate speech using a classifier, and then utilizing a debiasing component that generates less biased or unbiased alternatives through prompts. We evaluated our approach on a benchmark dataset and observed reduction in negativity due to hate speech comments. The proposed method contributes to the ongoing efforts to reduce biases in online discourse and promote a more inclusive and fair environment for communication.

{{</citation>}}


### (53/99) Unsupervised Domain Adaptation using Lexical Transformations and Label Injection for Twitter Data (Akshat Gupta et al., 2023)

{{<citation>}}

Akshat Gupta, Xiaomo Liu, Sameena Shah. (2023)  
**Unsupervised Domain Adaptation using Lexical Transformations and Label Injection for Twitter Data**  

---
Primary Category: cs.CL
Categories: cs-CL, cs.CL  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2307.10210v1)  

---


**ABSTRACT**  
Domain adaptation is an important and widely studied problem in natural language processing. A large body of literature tries to solve this problem by adapting models trained on the source domain to the target domain. In this paper, we instead solve this problem from a dataset perspective. We modify the source domain dataset with simple lexical transformations to reduce the domain shift between the source dataset distribution and the target dataset distribution. We find that models trained on the transformed source domain dataset performs significantly better than zero-shot models. Using our proposed transformations to convert standard English to tweets, we reach an unsupervised part-of-speech (POS) tagging accuracy of 92.14% (from 81.54% zero shot accuracy), which is only slightly below the supervised performance of 94.45%. We also use our proposed transformations to synthetically generate tweets and augment the Twitter dataset to achieve state-of-the-art performance for POS tagging.

{{</citation>}}


### (54/99) How Different Is Stereotypical Bias Across Languages? (Ibrahim Tolga Öztürk et al., 2023)

{{<citation>}}

Ibrahim Tolga Öztürk, Rostislav Nedelchev, Christian Heumann, Esteban Garces Arias, Marius Roger, Bernd Bischl, Matthias Aßenmacher. (2023)  
**How Different Is Stereotypical Bias Across Languages?**  

---
Primary Category: cs.CL
Categories: cs-CL, cs-CY, cs-LG, cs.CL, stat-ML  
Keywords: Bias, GPT  
[Paper Link](http://arxiv.org/abs/2307.07331v1)  

---


**ABSTRACT**  
Recent studies have demonstrated how to assess the stereotypical bias in pre-trained English language models. In this work, we extend this branch of research in multiple different dimensions by systematically investigating (a) mono- and multilingual models of (b) different underlying architectures with respect to their bias in (c) multiple different languages. To that end, we make use of the English StereoSet data set (Nadeem et al., 2021), which we semi-automatically translate into German, French, Spanish, and Turkish. We find that it is of major importance to conduct this type of analysis in a multilingual setting, as our experiments show a much more nuanced picture as well as notable differences from the English-only analysis. The main takeaways from our analysis are that mGPT-2 (partly) shows surprising anti-stereotypical behavior across languages, English (monolingual) models exhibit the strongest bias, and the stereotypes reflected in the data set are least present in Turkish models. Finally, we release our codebase alongside the translated data sets and practical guidelines for the semi-automatic translation to encourage a further extension of our work to other languages.

{{</citation>}}


### (55/99) Using Large Language Models for Zero-Shot Natural Language Generation from Knowledge Graphs (Agnes Axelsson et al., 2023)

{{<citation>}}

Agnes Axelsson, Gabriel Skantze. (2023)  
**Using Large Language Models for Zero-Shot Natural Language Generation from Knowledge Graphs**  

---
Primary Category: cs.CL
Categories: 68T50, I-2-7; I-2-4, cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Knowledge Graph, Language Model, Natural Language Generation, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2307.07312v1)  

---


**ABSTRACT**  
In any system that uses structured knowledge graph (KG) data as its underlying knowledge representation, KG-to-text generation is a useful tool for turning parts of the graph data into text that can be understood by humans. Recent work has shown that models that make use of pretraining on large amounts of text data can perform well on the KG-to-text task even with relatively small sets of training data on the specific graph-to-text task. In this paper, we build on this concept by using large language models to perform zero-shot generation based on nothing but the model's understanding of the triple structure from what it can read. We show that ChatGPT achieves near state-of-the-art performance on some measures of the WebNLG 2020 challenge, but falls behind on others. Additionally, we compare factual, counter-factual and fictional statements, and show that there is a significant connection between what the LLM already knows about the data it is parsing and the quality of the output text.

{{</citation>}}


### (56/99) C3: Zero-shot Text-to-SQL with ChatGPT (Xuemei Dong et al., 2023)

{{<citation>}}

Xuemei Dong, Chao Zhang, Yuhang Ge, Yuren Mao, Yunjun Gao, lu Chen, Jinshu Lin, Dongfang Lou. (2023)  
**C3: Zero-shot Text-to-SQL with ChatGPT**  

---
Primary Category: cs.CL
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2307.07306v1)  

---


**ABSTRACT**  
This paper proposes a ChatGPT-based zero-shot Text-to-SQL method, dubbed C3, which achieves 82.3\% in terms of execution accuracy on the holdout test set of Spider and becomes the state-of-the-art zero-shot Text-to-SQL method on the Spider Challenge. C3 consists of three key components: Clear Prompting (CP), Calibration with Hints (CH), and Consistent Output (CO), which are corresponding to the model input, model bias and model output respectively. It provides a systematic treatment for zero-shot Text-to-SQL. Extensive experiments have been conducted to verify the effectiveness and efficiency of our proposed method.

{{</citation>}}


### (57/99) Replay to Remember: Continual Layer-Specific Fine-tuning for German Speech Recognition (Theresa Pekarek Rosin et al., 2023)

{{<citation>}}

Theresa Pekarek Rosin, Stefan Wermter. (2023)  
**Replay to Remember: Continual Layer-Specific Fine-tuning for German Speech Recognition**  

---
Primary Category: cs.CL
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2307.07280v1)  

---


**ABSTRACT**  
While Automatic Speech Recognition (ASR) models have shown significant advances with the introduction of unsupervised or self-supervised training techniques, these improvements are still only limited to a subsection of languages and speakers. Transfer learning enables the adaptation of large-scale multilingual models to not only low-resource languages but also to more specific speaker groups. However, fine-tuning on data from new domains is usually accompanied by a decrease in performance on the original domain. Therefore, in our experiments, we examine how well the performance of large-scale ASR models can be approximated for smaller domains, with our own dataset of German Senior Voice Commands (SVC-de), and how much of the general speech recognition performance can be preserved by selectively freezing parts of the model during training. To further increase the robustness of the ASR model to vocabulary and speakers outside of the fine-tuned domain, we apply Experience Replay for continual learning. By adding only a fraction of data from the original domain, we are able to reach Word-Error-Rates (WERs) below 5\% on the new domain, while stabilizing performance for general speech recognition at acceptable WERs.

{{</citation>}}


### (58/99) Are words equally surprising in audio and audio-visual comprehension? (Pranava Madhyastha et al., 2023)

{{<citation>}}

Pranava Madhyastha, Ye Zhang, Gabriella Vigliocco. (2023)  
**Are words equally surprising in audio and audio-visual comprehension?**  

---
Primary Category: cs.CL
Categories: cs-CL, cs.CL  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.07277v1)  

---


**ABSTRACT**  
We report a controlled study investigating the effect of visual information (i.e., seeing the speaker) on spoken language comprehension. We compare the ERP signature (N400) associated with each word in audio-only and audio-visual presentations of the same verbal stimuli. We assess the extent to which surprisal measures (which quantify the predictability of words in their lexical context) are generated on the basis of different types of language models (specifically n-gram and Transformer models) that predict N400 responses for each word. Our results indicate that cognitive effort differs significantly between multimodal and unimodal settings. In addition, our findings suggest that while Transformer-based models, which have access to a larger lexical context, provide a better fit in the audio-only setting, 2-gram language models are more effective in the multimodal setting. This highlights the significant impact of local lexical context on cognitive processing in a multimodal environment.

{{</citation>}}


### (59/99) MorphPiece : Moving away from Statistical Language Representation (Haris Jabbar, 2023)

{{<citation>}}

Haris Jabbar. (2023)  
**MorphPiece : Moving away from Statistical Language Representation**  

---
Primary Category: cs.CL
Categories: cs-CL, cs.CL  
Keywords: GPT, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2307.07262v1)  

---


**ABSTRACT**  
Tokenization is a critical part of modern NLP pipelines. However, contemporary tokenizers for Large Language Models are based on statistical analysis of text corpora, without much consideration to the linguistic features. We propose a linguistically motivated tokenization scheme, MorphPiece, which is based partly on morphological segmentation of the underlying text. A GPT-style causal language model trained on this tokenizer (called MorphGPT) shows superior convergence compared to the same architecture trained on a standard BPE tokenizer. Specifically we get Language Modeling performance comparable to a 6 times larger model. Additionally, we evaluate MorphGPT on a variety of NLP tasks in supervised and unsupervised settings and find superior performance across the board, compared to GPT-2 model.

{{</citation>}}


### (60/99) Improving BERT with Hybrid Pooling Network and Drop Mask (Qian Chen et al., 2023)

{{<citation>}}

Qian Chen, Wen Wang, Qinglin Zhang, Chong Deng, Ma Yukun, Siqi Zheng. (2023)  
**Improving BERT with Hybrid Pooling Network and Drop Mask**  

---
Primary Category: cs.CL
Categories: cs-CL, cs.CL  
Keywords: BERT, Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2307.07258v1)  

---


**ABSTRACT**  
Transformer-based pre-trained language models, such as BERT, achieve great success in various natural language understanding tasks. Prior research found that BERT captures a rich hierarchy of linguistic information at different layers. However, the vanilla BERT uses the same self-attention mechanism for each layer to model the different contextual features. In this paper, we propose a HybridBERT model which combines self-attention and pooling networks to encode different contextual features in each layer. Additionally, we propose a simple DropMask method to address the mismatch between pre-training and fine-tuning caused by excessive use of special mask tokens during Masked Language Modeling pre-training. Experiments show that HybridBERT outperforms BERT in pre-training with lower loss, faster training speed (8% relative), lower memory cost (13% relative), and also in transfer learning with 1.5% relative higher accuracies on downstream tasks. Additionally, DropMask improves accuracies of BERT on downstream tasks across various masking rates.

{{</citation>}}


### (61/99) Dialogue Agents 101: A Beginner's Guide to Critical Ingredients for Designing Effective Conversational Systems (Shivani Kumar et al., 2023)

{{<citation>}}

Shivani Kumar, Sumit Bhatia, Milan Aggarwal, Tanmoy Chakraborty. (2023)  
**Dialogue Agents 101: A Beginner's Guide to Critical Ingredients for Designing Effective Conversational Systems**  

---
Primary Category: cs.CL
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2307.07255v1)  

---


**ABSTRACT**  
Sharing ideas through communication with peers is the primary mode of human interaction. Consequently, extensive research has been conducted in the area of conversational AI, leading to an increase in the availability and diversity of conversational tasks, datasets, and methods. However, with numerous tasks being explored simultaneously, the current landscape of conversational AI becomes fragmented. Therefore, initiating a well-thought-out model for a dialogue agent can pose significant challenges for a practitioner. Towards highlighting the critical ingredients needed for a practitioner to design a dialogue agent from scratch, the current study provides a comprehensive overview of the primary characteristics of a dialogue agent, the supporting tasks, their corresponding open-domain datasets, and the methods used to benchmark these datasets. We observe that different methods have been used to tackle distinct dialogue tasks. However, building separate models for each task is costly and does not leverage the correlation among the several tasks of a dialogue agent. As a result, recent trends suggest a shift towards building unified foundation models. To this end, we propose UNIT, a UNified dIalogue dataseT constructed from conversations of existing datasets for different dialogue tasks capturing the nuances for each of them. We also examine the evaluation strategies used to measure the performance of dialogue agents and highlight the scope for future research in the area of conversational AI.

{{</citation>}}


### (62/99) Certified Robustness for Large Language Models with Self-Denoising (Zhen Zhang et al., 2023)

{{<citation>}}

Zhen Zhang, Guanhua Zhang, Bairu Hou, Wenqi Fan, Qing Li, Sijia Liu, Yang Zhang, Shiyu Chang. (2023)  
**Certified Robustness for Large Language Models with Self-Denoising**  

---
Primary Category: cs.CL
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2307.07171v1)  

---


**ABSTRACT**  
Although large language models (LLMs) have achieved great success in vast real-world applications, their vulnerabilities towards noisy inputs have significantly limited their uses, especially in high-stake environments. In these contexts, it is crucial to ensure that every prediction made by large language models is stable, i.e., LLM predictions should be consistent given minor differences in the input. This largely falls into the study of certified robust LLMs, i.e., all predictions of LLM are certified to be correct in a local region around the input. Randomized smoothing has demonstrated great potential in certifying the robustness and prediction stability of LLMs. However, randomized smoothing requires adding noise to the input before model prediction, and its certification performance depends largely on the model's performance on corrupted data. As a result, its direct application to LLMs remains challenging and often results in a small certification radius. To address this issue, we take advantage of the multitasking nature of LLMs and propose to denoise the corrupted inputs with LLMs in a self-denoising manner. Different from previous works like denoised smoothing, which requires training a separate model to robustify LLM, our method enjoys far better efficiency and flexibility. Our experiment results show that our method outperforms the existing certification methods under both certified robustness and empirical robustness. The codes are available at https://github.com/UCSB-NLP-Chang/SelfDenoise.

{{</citation>}}


### (63/99) Learning to Retrieve In-Context Examples for Large Language Models (Liang Wang et al., 2023)

{{<citation>}}

Liang Wang, Nan Yang, Furu Wei. (2023)  
**Learning to Retrieve In-Context Examples for Large Language Models**  

---
Primary Category: cs.CL
Categories: cs-CL, cs-IR, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.07164v1)  

---


**ABSTRACT**  
Large language models (LLMs) have demonstrated their ability to learn in-context, allowing them to perform various tasks based on a few input-output examples. However, the effectiveness of in-context learning is heavily reliant on the quality of the selected examples. In this paper, we propose a novel framework to iteratively train dense retrievers that can identify high-quality in-context examples for LLMs. Our framework initially trains a reward model based on LLM feedback to evaluate the quality of candidate examples, followed by knowledge distillation to train a bi-encoder based dense retriever. Our experiments on a suite of 30 tasks demonstrate that our framework significantly enhances in-context learning performance. Furthermore, we show the generalization ability of our framework to unseen tasks during training. An in-depth analysis reveals that our model improves performance by retrieving examples with similar patterns, and the gains are consistent across LLMs of varying sizes.

{{</citation>}}


### (64/99) Do not Mask Randomly: Effective Domain-adaptive Pre-training by Masking In-domain Keywords (Shahriar Golchin et al., 2023)

{{<citation>}}

Shahriar Golchin, Mihai Surdeanu, Nazgol Tavabi, Ata Kiapour. (2023)  
**Do not Mask Randomly: Effective Domain-adaptive Pre-training by Masking In-domain Keywords**  

---
Primary Category: cs.CL
Categories: cs-CL, cs-LG, cs.CL  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2307.07160v1)  

---


**ABSTRACT**  
We propose a novel task-agnostic in-domain pre-training method that sits between generic pre-training and fine-tuning. Our approach selectively masks in-domain keywords, i.e., words that provide a compact representation of the target domain. We identify such keywords using KeyBERT (Grootendorst, 2020). We evaluate our approach using six different settings: three datasets combined with two distinct pre-trained language models (PLMs). Our results reveal that the fine-tuned PLMs adapted using our in-domain pre-training strategy outperform PLMs that used in-domain pre-training with random masking as well as those that followed the common pre-train-then-fine-tune paradigm. Further, the overhead of identifying in-domain keywords is reasonable, e.g., 7-15% of the pre-training time (for two epochs) for BERT Large (Devlin et al., 2019).

{{</citation>}}


### (65/99) MMSD2.0: Towards a Reliable Multi-modal Sarcasm Detection System (Libo Qin et al., 2023)

{{<citation>}}

Libo Qin, Shijue Huang, Qiguang Chen, Chenran Cai, Yudi Zhang, Bin Liang, Wanxiang Che, Ruifeng Xu. (2023)  
**MMSD2.0: Towards a Reliable Multi-modal Sarcasm Detection System**  

---
Primary Category: cs.CL
Categories: cs-CL, cs.CL  
Keywords: Sarcasm Detection  
[Paper Link](http://arxiv.org/abs/2307.07135v1)  

---


**ABSTRACT**  
Multi-modal sarcasm detection has attracted much recent attention. Nevertheless, the existing benchmark (MMSD) has some shortcomings that hinder the development of reliable multi-modal sarcasm detection system: (1) There are some spurious cues in MMSD, leading to the model bias learning; (2) The negative samples in MMSD are not always reasonable. To solve the aforementioned issues, we introduce MMSD2.0, a correction dataset that fixes the shortcomings of MMSD, by removing the spurious cues and re-annotating the unreasonable samples. Meanwhile, we present a novel framework called multi-view CLIP that is capable of leveraging multi-grained cues from multiple perspectives (i.e., text, image, and text-image interaction view) for multi-modal sarcasm detection. Extensive experiments show that MMSD2.0 is a valuable benchmark for building reliable multi-modal sarcasm detection systems and multi-view CLIP can significantly outperform the previous best baselines.

{{</citation>}}


### (66/99) Generating Efficient Training Data via LLM-based Attribute Manipulation (Letian Peng et al., 2023)

{{<citation>}}

Letian Peng, Yuwei Zhang, Jingbo Shang. (2023)  
**Generating Efficient Training Data via LLM-based Attribute Manipulation**  

---
Primary Category: cs.CL
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.07099v1)  

---


**ABSTRACT**  
In this paper, we propose a novel method, Chain-of-Thoughts Attribute Manipulation (CoTAM), to guide few-shot learning by carefully crafted data from Large Language Models (LLMs). The main idea is to create data with changes only in the attribute targeted by the task. Inspired by facial attribute manipulation, our approach generates label-switched data by leveraging LLMs to manipulate task-specific attributes and reconstruct new sentences in a controlled manner. Instead of conventional latent representation controlling, we implement chain-of-thoughts decomposition and reconstruction to adapt the procedure to LLMs. Extensive results on text classification and other tasks verify the advantage of CoTAM over other LLM-based text generation methods with the same number of training examples. Analysis visualizes the attribute manipulation effectiveness of CoTAM and presents the potential of LLM-guided learning with even less supervision.

{{</citation>}}


## physics.med-ph (1)



### (67/99) Reconstruction of 3-Axis Seismocardiogram from Right-to-left and Head-to-foot Components Using A Long Short-Term Memory Network (Mohammad Muntasir Rahman et al., 2023)

{{<citation>}}

Mohammad Muntasir Rahman, Amirtahà Taebi. (2023)  
**Reconstruction of 3-Axis Seismocardiogram from Right-to-left and Head-to-foot Components Using A Long Short-Term Memory Network**  

---
Primary Category: physics.med-ph
Categories: cs-LG, eess-SP, physics-med-ph, physics.med-ph  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2307.07566v1)  

---


**ABSTRACT**  
This pilot study aims to develop a deep learning model for predicting seismocardiogram (SCG) signals in the dorsoventral direction from the SCG signals in the right-to-left and head-to-foot directions ($\textrm{SCG}_x$ and $\textrm{SCG}_y$). The dataset used for the training and validation of the model was obtained from 15 healthy adult subjects. The SCG signals were recorded using tri-axial accelerometers placed on the chest of each subject. The signals were then segmented using electrocardiogram R waves, and the segments were downsampled, normalized, and centered around zero. The resulting dataset was used to train and validate a long short-term memory (LSTM) network with two layers and a dropout layer to prevent overfitting. The network took as input 100-time steps of $\textrm{SCG}_x$ and $\textrm{SCG}_y$, representing one cardiac cycle, and outputted a vector that mapped to the target variable being predicted. The results showed that the LSTM model had a mean square error of 0.09 between the predicted and actual SCG segments in the dorsoventral direction. The study demonstrates the potential of deep learning models for reconstructing 3-axis SCG signals using the data obtained from dual-axis accelerometers.

{{</citation>}}


## physics.optics (1)



### (68/99) Reinforcement Learning for Photonic Component Design (Donald Witt et al., 2023)

{{<citation>}}

Donald Witt, Jeff Young, Lukas Chrostowski. (2023)  
**Reinforcement Learning for Photonic Component Design**  

---
Primary Category: physics.optics
Categories: cs-LG, physics-app-ph, physics-optics, physics.optics  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.11075v1)  

---


**ABSTRACT**  
We present a new fab-in-the-loop reinforcement learning algorithm for the design of nano-photonic components that accounts for the imperfections present in nanofabrication processes. As a demonstration of the potential of this technique, we apply it to the design of photonic crystal grating couplers (PhCGC) fabricated on a 220nm silicon on insulator (SOI) single etch platform. This fab-in-the-loop algorithm improves the insertion loss from 8.8 dB to 3.24 dB. The widest bandwidth designs produced using our fab-in-the-loop algorithm are able to cover a 150nm bandwidth with less than 10.2 dB of loss at their lowest point.

{{</citation>}}


## cs.RO (5)



### (69/99) SGGNet$^2$: Speech-Scene Graph Grounding Network for Speech-guided Navigation (Dohyun Kim et al., 2023)

{{<citation>}}

Dohyun Kim, Yeseung Kim, Jaehwi Jang, Minjae Song, Woojin Choi, Daehyung Park. (2023)  
**SGGNet$^2$: Speech-Scene Graph Grounding Network for Speech-guided Navigation**  

---
Primary Category: cs.RO
Categories: cs-RO, cs.RO  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2307.07468v1)  

---


**ABSTRACT**  
The spoken language serves as an accessible and efficient interface, enabling non-experts and disabled users to interact with complex assistant robots. However, accurately grounding language utterances gives a significant challenge due to the acoustic variability in speakers' voices and environmental noise. In this work, we propose a novel speech-scene graph grounding network (SGGNet$^2$) that robustly grounds spoken utterances by leveraging the acoustic similarity between correctly recognized and misrecognized words obtained from automatic speech recognition (ASR) systems. To incorporate the acoustic similarity, we extend our previous grounding model, the scene-graph-based grounding network (SGGNet), with the ASR model from NVIDIA NeMo. We accomplish this by feeding the latent vector of speech pronunciations into the BERT-based grounding network within SGGNet. We evaluate the effectiveness of using latent vectors of speech commands in grounding through qualitative and quantitative studies. We also demonstrate the capability of SGGNet$^2$ in a speech-based navigation task using a real quadruped robot, RBQ-3, from Rainbow Robotics.

{{</citation>}}


### (70/99) Learn from Incomplete Tactile Data: Tactile Representation Learning with Masked Autoencoders (Guanqun Cao et al., 2023)

{{<citation>}}

Guanqun Cao, Jiaqi Jiang, Danushka Bollegala, Shan Luo. (2023)  
**Learn from Incomplete Tactile Data: Tactile Representation Learning with Masked Autoencoders**  

---
Primary Category: cs.RO
Categories: cs-RO, cs.RO  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2307.07358v1)  

---


**ABSTRACT**  
The missing signal caused by the objects being occluded or an unstable sensor is a common challenge during data collection. Such missing signals will adversely affect the results obtained from the data, and this issue is observed more frequently in robotic tactile perception. In tactile perception, due to the limited working space and the dynamic environment, the contact between the tactile sensor and the object is frequently insufficient and unstable, which causes the partial loss of signals, thus leading to incomplete tactile data. The tactile data will therefore contain fewer tactile cues with low information density. In this paper, we propose a tactile representation learning method, named TacMAE, based on Masked Autoencoder to address the problem of incomplete tactile data in tactile perception. In our framework, a portion of the tactile image is masked out to simulate the missing contact region. By reconstructing the missing signals in the tactile image, the trained model can achieve a high-level understanding of surface geometry and tactile properties from limited tactile cues. The experimental results of tactile texture recognition show that our proposed TacMAE can achieve a high recognition accuracy of 71.4% in the zero-shot transfer and 85.8% after fine-tuning, which are 15.2% and 8.2% higher than the results without using masked modeling. The extensive experiments on YCB objects demonstrate the knowledge transferability of our proposed method and the potential to improve efficiency in tactile exploration.

{{</citation>}}


### (71/99) Reinforcement Learning with Frontier-Based Exploration via Autonomous Environment (Kenji Leong, 2023)

{{<citation>}}

Kenji Leong. (2023)  
**Reinforcement Learning with Frontier-Based Exploration via Autonomous Environment**  

---
Primary Category: cs.RO
Categories: cs-LG, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.07296v1)  

---


**ABSTRACT**  
Active Simultaneous Localisation and Mapping (SLAM) is a critical problem in autonomous robotics, enabling robots to navigate to new regions while building an accurate model of their surroundings. Visual SLAM is a popular technique that uses virtual elements to enhance the experience. However, existing frontier-based exploration strategies can lead to a non-optimal path in scenarios where there are multiple frontiers with similar distance. This issue can impact the efficiency and accuracy of Visual SLAM, which is crucial for a wide range of robotic applications, such as search and rescue, exploration, and mapping. To address this issue, this research combines both an existing Visual-Graph SLAM known as ExploreORB with reinforcement learning. The proposed algorithm allows the robot to learn and optimize exploration routes through a reward-based system to create an accurate map of the environment with proper frontier selection. Frontier-based exploration is used to detect unexplored areas, while reinforcement learning optimizes the robot's movement by assigning rewards for optimal frontier points. Graph SLAM is then used to integrate the robot's sensory data and build an accurate map of the environment. The proposed algorithm aims to improve the efficiency and accuracy of ExploreORB by optimizing the exploration process of frontiers to build a more accurate map. To evaluate the effectiveness of the proposed approach, experiments will be conducted in various virtual environments using Gazebo, a robot simulation software. Results of these experiments will be compared with existing methods to demonstrate the potential of the proposed approach as an optimal solution for SLAM in autonomous robotics.

{{</citation>}}


### (72/99) Switching Head-Tail Funnel UNITER for Dual Referring Expression Comprehension with Fetch-and-Carry Tasks (Ryosuke Korekata et al., 2023)

{{<citation>}}

Ryosuke Korekata, Motonari Kambara, Yu Yoshida, Shintaro Ishikawa, Yosuke Kawasaki, Masaki Takahashi, Komei Sugiura. (2023)  
**Switching Head-Tail Funnel UNITER for Dual Referring Expression Comprehension with Fetch-and-Carry Tasks**  

---
Primary Category: cs.RO
Categories: cs-CL, cs-CV, cs-RO, cs.RO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.07166v1)  

---


**ABSTRACT**  
This paper describes a domestic service robot (DSR) that fetches everyday objects and carries them to specified destinations according to free-form natural language instructions. Given an instruction such as "Move the bottle on the left side of the plate to the empty chair," the DSR is expected to identify the bottle and the chair from multiple candidates in the environment and carry the target object to the destination. Most of the existing multimodal language understanding methods are impractical in terms of computational complexity because they require inferences for all combinations of target object candidates and destination candidates. We propose Switching Head-Tail Funnel UNITER, which solves the task by predicting the target object and the destination individually using a single model. Our method is validated on a newly-built dataset consisting of object manipulation instructions and semi photo-realistic images captured in a standard Embodied AI simulator. The results show that our method outperforms the baseline method in terms of language comprehension accuracy. Furthermore, we conduct physical experiments in which a DSR delivers standardized everyday objects in a standardized domestic environment as requested by instructions with referring expressions. The experimental results show that the object grasping and placing actions are achieved with success rates of more than 90%.

{{</citation>}}


### (73/99) Drive Like a Human: Rethinking Autonomous Driving with Large Language Models (Daocheng Fu et al., 2023)

{{<citation>}}

Daocheng Fu, Xin Li, Licheng Wen, Min Dou, Pinlong Cai, Botian Shi, Yu Qiao. (2023)  
**Drive Like a Human: Rethinking Autonomous Driving with Large Language Models**  

---
Primary Category: cs.RO
Categories: cs-CL, cs-RO, cs.RO  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.07162v1)  

---


**ABSTRACT**  
In this paper, we explore the potential of using a large language model (LLM) to understand the driving environment in a human-like manner and analyze its ability to reason, interpret, and memorize when facing complex scenarios. We argue that traditional optimization-based and modular autonomous driving (AD) systems face inherent performance limitations when dealing with long-tail corner cases. To address this problem, we propose that an ideal AD system should drive like a human, accumulating experience through continuous driving and using common sense to solve problems. To achieve this goal, we identify three key abilities necessary for an AD system: reasoning, interpretation, and memorization. We demonstrate the feasibility of employing an LLM in driving scenarios by building a closed-loop system to showcase its comprehension and environment-interaction abilities. Our extensive experiments show that the LLM exhibits the impressive ability to reason and solve long-tailed cases, providing valuable insights for the development of human-like autonomous driving. The related code are available at https://github.com/PJLab-ADG/DriveLikeAHuman .

{{</citation>}}


## cs.SE (3)



### (74/99) Investigating ChatGPT's Potential to Assist in Requirements Elicitation Processes (Krishna Ronanki et al., 2023)

{{<citation>}}

Krishna Ronanki, Christian Berger, Jennifer Horkoff. (2023)  
**Investigating ChatGPT's Potential to Assist in Requirements Elicitation Processes**  

---
Primary Category: cs.SE
Categories: cs-SE, cs.SE  
Keywords: AI, ChatGPT, GPT, Generative AI, Language Model, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2307.07381v1)  

---


**ABSTRACT**  
Natural Language Processing (NLP) for Requirements Engineering (RE) (NLP4RE) seeks to apply NLP tools, techniques, and resources to the RE process to increase the quality of the requirements. There is little research involving the utilization of Generative AI-based NLP tools and techniques for requirements elicitation. In recent times, Large Language Models (LLM) like ChatGPT have gained significant recognition due to their notably improved performance in NLP tasks. To explore the potential of ChatGPT to assist in requirements elicitation processes, we formulated six questions to elicit requirements using ChatGPT. Using the same six questions, we conducted interview-based surveys with five RE experts from academia and industry and collected 30 responses containing requirements. The quality of these 36 responses (human-formulated + ChatGPT-generated) was evaluated over seven different requirements quality attributes by another five RE experts through a second round of interview-based surveys. In comparing the quality of requirements generated by ChatGPT with those formulated by human experts, we found that ChatGPT-generated requirements are highly Abstract, Atomic, Consistent, Correct, and Understandable. Based on these results, we present the most pressing issues related to LLMs and what future research should focus on to leverage the emergent behaviour of LLMs more effectively in natural language-based RE activities.

{{</citation>}}


### (75/99) Software Testing with Large Language Model: Survey, Landscape, and Vision (Junjie Wang et al., 2023)

{{<citation>}}

Junjie Wang, Yuchao Huang, Chunyang Chen, Zhe Liu, Song Wang, Qing Wang. (2023)  
**Software Testing with Large Language Model: Survey, Landscape, and Vision**  

---
Primary Category: cs.SE
Categories: cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.07221v1)  

---


**ABSTRACT**  
Pre-trained large language models (LLMs) have recently emerged as a breakthrough technology in natural language processing and artificial intelligence, with the ability to handle large-scale datasets and exhibit remarkable performance across a wide range of tasks. Meanwhile, software testing is a crucial undertaking that serves as a cornerstone for ensuring the quality and reliability of software products. As the scope and complexity of software systems continue to grow, the need for more effective software testing techniques becomes increasingly urgent, and making it an area ripe for innovative approaches such as the use of LLMs. This paper provides a comprehensive review of the utilization of LLMs in software testing. It analyzes 52 relevant studies that have used LLMs for software testing, from both the software testing and LLMs perspectives. The paper presents a detailed discussion of the software testing tasks for which LLMs are commonly used, among which test case preparation and program repair are the most representative ones. It also analyzes the commonly used LLMs, the types of prompt engineering that are employed, as well as the accompanied techniques with these LLMs. It also summarizes the key challenges and potential opportunities in this direction. This work can serve as a roadmap for future research in this area, highlighting potential avenues for exploration, and identifying gaps in our current understanding of the use of LLMs in software testing.

{{</citation>}}


### (76/99) When Conversations Turn Into Work: A Taxonomy of Converted Discussions and Issues in GitHub (Dong Wang et al., 2023)

{{<citation>}}

Dong Wang, Masanari Kondo, Yasutaka Kamei, Raula Gaikovina Kula, Naoyasu Ubayashi. (2023)  
**When Conversations Turn Into Work: A Taxonomy of Converted Discussions and Issues in GitHub**  

---
Primary Category: cs.SE
Categories: cs-SE, cs.SE  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2307.07117v1)  

---


**ABSTRACT**  
Popular and large contemporary open-source projects now embrace a diverse set of documentation for communication channels. Examples include contribution guidelines (i.e., commit message guidelines, coding rules, submission guidelines), code of conduct (i.e., rules and behavior expectations), governance policies, and Q&A forum. In 2020, GitHub released Discussion to distinguish between communication and collaboration. However, it remains unclear how developers maintain these channels, how trivial it is, and whether deciding on conversion takes time. We conducted an empirical study on 259 NPM and 148 PyPI repositories, devising two taxonomies of reasons for converting discussions into issues and vice-versa. The most frequent conversion from a discussion to an issue is when developers request a contributor to clarify their idea into an issue (Reporting a Clarification Request -35.1% and 34.7%, respectively), while agreeing that having non actionable topic (QA, ideas, feature requests -55.0% and 42.0%, respectively}) is the most frequent reason of converting an issue into a discussion. Furthermore, we show that not all reasons for conversion are trivial (e.g., not a bug), and raising a conversion intent potentially takes time (i.e., a median of 15.2 and 35.1 hours, respectively, taken from issues to discussions). Our work contributes to complementing the GitHub guidelines and helping developers effectively utilize the Issue and Discussion communication channels to maintain their collaboration.

{{</citation>}}


## cs.SI (1)



### (77/99) Are Large Language Models a Threat to Digital Public Goods? Evidence from Activity on Stack Overflow (Maria del Rio-Chanona et al., 2023)

{{<citation>}}

Maria del Rio-Chanona, Nadzeya Laurentsyeva, Johannes Wachs. (2023)  
**Are Large Language Models a Threat to Digital Public Goods? Evidence from Activity on Stack Overflow**  

---
Primary Category: cs.SI
Categories: cs-AI, cs-CY, cs-SI, cs.SI  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2307.07367v1)  

---


**ABSTRACT**  
Large language models like ChatGPT efficiently provide users with information about various topics, presenting a potential substitute for searching the web and asking people for help online. But since users interact privately with the model, these models may drastically reduce the amount of publicly available human-generated data and knowledge resources. This substitution can present a significant problem in securing training data for future models. In this work, we investigate how the release of ChatGPT changed human-generated open data on the web by analyzing the activity on Stack Overflow, the leading online Q\&A platform for computer programming. We find that relative to its Russian and Chinese counterparts, where access to ChatGPT is limited, and to similar forums for mathematics, where ChatGPT is less capable, activity on Stack Overflow significantly decreased. A difference-in-differences model estimates a 16\% decrease in weekly posts on Stack Overflow. This effect increases in magnitude over time, and is larger for posts related to the most widely used programming languages. Posts made after ChatGPT get similar voting scores than before, suggesting that ChatGPT is not merely displacing duplicate or low-quality content. These results suggest that more users are adopting large language models to answer questions and they are better substitutes for Stack Overflow for languages for which they have more training data. Using models like ChatGPT may be more efficient for solving certain programming problems, but its widespread adoption and the resulting shift away from public exchange on the web will limit the open data people and models can learn from in the future.

{{</citation>}}


## eess.SP (2)



### (78/99) Source-Free Domain Adaptation with Temporal Imputation for Time Series Data (Mohamed Ragab et al., 2023)

{{<citation>}}

Mohamed Ragab, Emadeldeen Eldele, Min Wu, Chuan-Sheng Foo, Xiaoli Li, Zhenghua Chen. (2023)  
**Source-Free Domain Adaptation with Temporal Imputation for Time Series Data**  

---
Primary Category: eess.SP
Categories: cs-AI, cs-LG, eess-SP, eess.SP  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2307.07542v1)  

---


**ABSTRACT**  
Source-free domain adaptation (SFDA) aims to adapt a pretrained model from a labeled source domain to an unlabeled target domain without access to the source domain data, preserving source domain privacy. Despite its prevalence in visual applications, SFDA is largely unexplored in time series applications. The existing SFDA methods that are mainly designed for visual applications may fail to handle the temporal dynamics in time series, leading to impaired adaptation performance. To address this challenge, this paper presents a simple yet effective approach for source-free domain adaptation on time series data, namely MAsk and imPUte (MAPU). First, to capture temporal information of the source domain, our method performs random masking on the time series signals while leveraging a novel temporal imputer to recover the original signal from a masked version in the embedding space. Second, in the adaptation step, the imputer network is leveraged to guide the target model to produce target features that are temporally consistent with the source features. To this end, our MAPU can explicitly account for temporal dependency during the adaptation while avoiding the imputation in the noisy input space. Our method is the first to handle temporal consistency in SFDA for time series data and can be seamlessly equipped with other existing SFDA methods. Extensive experiments conducted on three real-world time series datasets demonstrate that our MAPU achieves significant performance gain over existing methods. Our code is available at \url{https://github.com/mohamedr002/MAPU_SFDA_TS}.

{{</citation>}}


### (79/99) Polarization-Based Security: Safeguarding Wireless Communications at the Physical Layer (Pol Henarejos et al., 2023)

{{<citation>}}

Pol Henarejos, Ana I. Pérez-Neira. (2023)  
**Polarization-Based Security: Safeguarding Wireless Communications at the Physical Layer**  

---
Primary Category: eess.SP
Categories: cs-CR, eess-SP, eess.SP  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2307.07244v1)  

---


**ABSTRACT**  
Physical layer security is a field of study that continues to gain importance over time. It encompasses a range of algorithms applicable to various aspects of communication systems. While research in the physical layer has predominantly focused on secrecy capacity, which involves logical and digital manipulations to achieve secure communication, there is limited exploration of directly manipulating electromagnetic fields to enhance security against eavesdroppers. In this paper, we propose a novel system that utilizes the Mueller calculation to establish a theoretical framework for manipulating electromagnetic fields in the context of physical layer security. We develop fundamental expressions and introduce new metrics to analyze the system's performance analytically. Additionally, we present three techniques that leverage polarization to enhance physical layer security.

{{</citation>}}


## cs.IT (1)



### (80/99) From Multilayer Perceptron to GPT: A Reflection on Deep Learning Research for Wireless Physical Layer (Mohamed Akrout et al., 2023)

{{<citation>}}

Mohamed Akrout, Amine Mezghani, Ekram Hossain, Faouzi Bellili, Robert W. Heath. (2023)  
**From Multilayer Perceptron to GPT: A Reflection on Deep Learning Research for Wireless Physical Layer**  

---
Primary Category: cs.IT
Categories: cs-IT, cs.IT, math-IT  
Keywords: AI, GPT  
[Paper Link](http://arxiv.org/abs/2307.07359v1)  

---


**ABSTRACT**  
Most research studies on deep learning (DL) applied to the physical layer of wireless communication do not put forward the critical role of the accuracy-generalization trade-off in developing and evaluating practical algorithms. To highlight the disadvantage of this common practice, we revisit a data decoding example from one of the first papers introducing DL-based end-to-end wireless communication systems to the research community and promoting the use of artificial intelligence (AI)/DL for the wireless physical layer. We then put forward two key trade-offs in designing DL models for communication, namely, accuracy versus generalization and compression versus latency. We discuss their relevance in the context of wireless communications use cases using emerging DL models including large language models (LLMs). Finally, we summarize our proposed evaluation guidelines to enhance the research impact of DL on wireless communications. These guidelines are an attempt to reconcile the empirical nature of DL research with the rigorous requirement metrics of wireless communications systems.

{{</citation>}}


## math.OC (1)



### (81/99) Inverse Optimization for Routing Problems (Pedro Zattoni Scroccaro et al., 2023)

{{<citation>}}

Pedro Zattoni Scroccaro, Piet van Beek, Peyman Mohajerin Esfahani, Bilge Atasoy. (2023)  
**Inverse Optimization for Routing Problems**  

---
Primary Category: math.OC
Categories: cs-LG, math-OC, math.OC  
Keywords: Amazon  
[Paper Link](http://arxiv.org/abs/2307.07357v1)  

---


**ABSTRACT**  
We propose a method for learning decision-makers' behavior in routing problems using Inverse Optimization (IO). The IO framework falls into the supervised learning category and builds on the premise that the target behavior is an optimizer of an unknown cost function. This cost function is to be learned through historical data, and in the context of routing problems, can be interpreted as the routing preferences of the decision-makers. In this view, the main contributions of this study are to propose an IO methodology with a hypothesis function, loss function, and stochastic first-order algorithm tailored to routing problems. We further test our IO approach in the Amazon Last Mile Routing Research Challenge, where the goal is to learn models that replicate the routing preferences of human drivers, using thousands of real-world routing examples. Our final IO-learned routing model achieves a score that ranks 2nd compared with the 48 models that qualified for the final round of the challenge. Our results showcase the flexibility and real-world potential of the proposed IO methodology to learn from decision-makers' decisions in routing problems.

{{</citation>}}


## cs.CR (3)



### (82/99) Time for aCTIon: Automated Analysis of Cyber Threat Intelligence in the Wild (Giuseppe Siracusano et al., 2023)

{{<citation>}}

Giuseppe Siracusano, Davide Sanvito, Roberto Gonzalez, Manikantan Srinivasan, Sivakaman Kamatchi, Wataru Takahashi, Masaru Kawakita, Takahiro Kakumaru, Roberto Bifulco. (2023)  
**Time for aCTIon: Automated Analysis of Cyber Threat Intelligence in the Wild**  

---
Primary Category: cs.CR
Categories: cs-CR, cs-LG, cs.CR  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2307.10214v1)  

---


**ABSTRACT**  
Cyber Threat Intelligence (CTI) plays a crucial role in assessing risks and enhancing security for organizations. However, the process of extracting relevant information from unstructured text sources can be expensive and time-consuming. Our empirical experience shows that existing tools for automated structured CTI extraction have performance limitations. Furthermore, the community lacks a common benchmark to quantitatively assess their performance. We fill these gaps providing a new large open benchmark dataset and aCTIon, a structured CTI information extraction tool. The dataset includes 204 real-world publicly available reports and their corresponding structured CTI information in STIX format. Our team curated the dataset involving three independent groups of CTI analysts working over the course of several months. To the best of our knowledge, this dataset is two orders of magnitude larger than previously released open source datasets. We then design aCTIon, leveraging recently introduced large language models (GPT3.5) in the context of two custom information extraction pipelines. We compare our method with 10 solutions presented in previous work, for which we develop our own implementations when open-source implementations were lacking. Our results show that aCTIon outperforms previous work for structured CTI extraction with an improvement of the F1-score from 10%points to 50%points across all tasks.

{{</citation>}}


### (83/99) On the Sensitivity of Deep Load Disaggregation to Adversarial Attacks (Hafsa Bousbiat et al., 2023)

{{<citation>}}

Hafsa Bousbiat, Yassine Himeur, Abbes Amira, Wathiq Mansoor. (2023)  
**On the Sensitivity of Deep Load Disaggregation to Adversarial Attacks**  

---
Primary Category: cs.CR
Categories: cs-CR, cs-LG, cs.CR  
Keywords: Adversarial Attack, Sequence-to-Sequence  
[Paper Link](http://arxiv.org/abs/2307.10209v1)  

---


**ABSTRACT**  
Non-intrusive Load Monitoring (NILM) algorithms, commonly referred to as load disaggregation algorithms, are fundamental tools for effective energy management. Despite the success of deep models in load disaggregation, they face various challenges, particularly those pertaining to privacy and security. This paper investigates the sensitivity of prominent deep NILM baselines to adversarial attacks, which have proven to be a significant threat in domains such as computer vision and speech recognition. Adversarial attacks entail the introduction of imperceptible noise into the input data with the aim of misleading the neural network into generating erroneous outputs. We investigate the Fast Gradient Sign Method (FGSM), a well-known adversarial attack, to perturb the input sequences fed into two commonly employed CNN-based NILM baselines: the Sequence-to-Sequence (S2S) and Sequence-to-Point (S2P) models. Our findings provide compelling evidence for the vulnerability of these models, particularly the S2P model which exhibits an average decline of 20\% in the F1-score even with small amounts of noise. Such weakness has the potential to generate profound implications for energy management systems in residential and industrial sectors reliant on NILM models.

{{</citation>}}


### (84/99) Understanding Multi-Turn Toxic Behaviors in Open-Domain Chatbots (Bocheng Chen et al., 2023)

{{<citation>}}

Bocheng Chen, Guangjing Wang, Hanqing Guo, Yuanda Wang, Qiben Yan. (2023)  
**Understanding Multi-Turn Toxic Behaviors in Open-Domain Chatbots**  

---
Primary Category: cs.CR
Categories: cs-AI, cs-CL, cs-CR, cs.CR  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2307.09579v1)  

---


**ABSTRACT**  
Recent advances in natural language processing and machine learning have led to the development of chatbot models, such as ChatGPT, that can engage in conversational dialogue with human users. However, the ability of these models to generate toxic or harmful responses during a non-toxic multi-turn conversation remains an open research question. Existing research focuses on single-turn sentence testing, while we find that 82\% of the individual non-toxic sentences that elicit toxic behaviors in a conversation are considered safe by existing tools. In this paper, we design a new attack, \toxicbot, by fine-tuning a chatbot to engage in conversation with a target open-domain chatbot. The chatbot is fine-tuned with a collection of crafted conversation sequences. Particularly, each conversation begins with a sentence from a crafted prompt sentences dataset. Our extensive evaluation shows that open-domain chatbot models can be triggered to generate toxic responses in a multi-turn conversation. In the best scenario, \toxicbot achieves a 67\% activation rate. The conversation sequences in the fine-tuning stage help trigger the toxicity in a conversation, which allows the attack to bypass two defense methods. Our findings suggest that further research is needed to address chatbot toxicity in a dynamic interactive environment. The proposed \toxicbot can be used by both industry and researchers to develop methods for detecting and mitigating toxic responses in conversational dialogue and improve the robustness of chatbots for end users.

{{</citation>}}


## cs.IR (1)



### (85/99) PiTL: Cross-modal Retrieval with Weakly-supervised Vision-language Pre-training via Prompting (Zixin Guo et al., 2023)

{{<citation>}}

Zixin Guo, Tzu-Jui Julius Wang, Selen Pehlivan, Abduljalil Radman, Jorma Laaksonen. (2023)  
**PiTL: Cross-modal Retrieval with Weakly-supervised Vision-language Pre-training via Prompting**  

---
Primary Category: cs.IR
Categories: cs-CV, cs-IR, cs.IR  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2307.07341v1)  

---


**ABSTRACT**  
Vision-language (VL) Pre-training (VLP) has shown to well generalize VL models over a wide range of VL downstream tasks, especially for cross-modal retrieval. However, it hinges on a huge amount of image-text pairs, which requires tedious and costly curation. On the contrary, weakly-supervised VLP (W-VLP) explores means with object tags generated by a pre-trained object detector (OD) from images. Yet, they still require paired information, i.e. images and object-level annotations, as supervision to train an OD.   To further reduce the amount of supervision, we propose Prompts-in-The-Loop (PiTL) that prompts knowledge from large language models (LLMs) to describe images. Concretely, given a category label of an image, e.g. refinery, the knowledge, e.g. a refinery could be seen with large storage tanks, pipework, and ..., extracted by LLMs is used as the language counterpart. The knowledge supplements, e.g. the common relations among entities most likely appearing in a scene. We create IN14K, a new VL dataset of 9M images and 1M descriptions of 14K categories from ImageNet21K with PiTL. Empirically, the VL models pre-trained with PiTL-generated pairs are strongly favored over other W-VLP works on image-to-text (I2T) and text-to-image (T2I) retrieval tasks, with less supervision. The results reveal the effectiveness of PiTL-generated pairs for VLP.

{{</citation>}}


## eess.AS (2)



### (86/99) Representation Learning With Hidden Unit Clustering For Low Resource Speech Applications (Varun Krishna et al., 2023)

{{<citation>}}

Varun Krishna, Tarun Sai, Sriram Ganapathy. (2023)  
**Representation Learning With Hidden Unit Clustering For Low Resource Speech Applications**  

---
Primary Category: eess.AS
Categories: cs-AI, cs-LG, eess-AS, eess.AS  
Keywords: BERT, LSTM, Representation Learning  
[Paper Link](http://arxiv.org/abs/2307.07325v1)  

---


**ABSTRACT**  
The representation learning of speech, without textual resources, is an area of significant interest for many low resource speech applications. In this paper, we describe an approach to self-supervised representation learning from raw audio using a hidden unit clustering (HUC) framework. The input to the model consists of audio samples that are windowed and processed with 1-D convolutional layers. The learned "time-frequency" representations from the convolutional neural network (CNN) module are further processed with long short term memory (LSTM) layers which generate a contextual vector representation for every windowed segment. The HUC framework, allowing the categorization of the representations into a small number of phoneme-like units, is used to train the model for learning semantically rich speech representations. The targets consist of phoneme-like pseudo labels for each audio segment and these are generated with an iterative k-means algorithm. We explore techniques that improve the speaker invariance of the learned representations and illustrate the effectiveness of the proposed approach on two settings, i) completely unsupervised speech applications on the sub-tasks described as part of the ZeroSpeech 2021 challenge and ii) semi-supervised automatic speech recognition (ASR) applications on the TIMIT dataset and on the GramVaani challenge Hindi dataset. In these experiments, we achieve state-of-art results for various ZeroSpeech tasks. Further, on the ASR experiments, the HUC representations are shown to improve significantly over other established benchmarks based on Wav2vec, HuBERT and Best-RQ.

{{</citation>}}


### (87/99) Mega-TTS 2: Zero-Shot Text-to-Speech with Arbitrary Length Speech Prompts (Ziyue Jiang et al., 2023)

{{<citation>}}

Ziyue Jiang, Jinglin Liu, Yi Ren, Jinzheng He, Chen Zhang, Zhenhui Ye, Pengfei Wei, Chunfeng Wang, Xiang Yin, Zejun Ma, Zhou Zhao. (2023)  
**Mega-TTS 2: Zero-Shot Text-to-Speech with Arbitrary Length Speech Prompts**  

---
Primary Category: eess.AS
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2307.07218v1)  

---


**ABSTRACT**  
Zero-shot text-to-speech aims at synthesizing voices with unseen speech prompts. Previous large-scale multispeaker TTS models have successfully achieved this goal with an enrolled recording within 10 seconds. However, most of them are designed to utilize only short speech prompts. The limited information in short speech prompts significantly hinders the performance of fine-grained identity imitation. In this paper, we introduce Mega-TTS 2, a generic zero-shot multispeaker TTS model that is capable of synthesizing speech for unseen speakers with arbitrary-length prompts. Specifically, we 1) design a multi-reference timbre encoder to extract timbre information from multiple reference speeches; 2) and train a prosody language model with arbitrary-length speech prompts; With these designs, our model is suitable for prompts of different lengths, which extends the upper bound of speech quality for zero-shot text-to-speech. Besides arbitrary-length prompts, we introduce arbitrary-source prompts, which leverages the probabilities derived from multiple P-LLM outputs to produce expressive and controlled prosody. Furthermore, we propose a phoneme-level auto-regressive duration model to introduce in-context learning capabilities to duration modeling. Experiments demonstrate that our method could not only synthesize identity-preserving speech with a short prompt of an unseen speaker but also achieve improved performance with longer speech prompts. Audio samples can be found in https://mega-tts.github.io/mega2_demo/.

{{</citation>}}


## eess.IV (2)



### (88/99) Frequency Domain Adversarial Training for Robust Volumetric Medical Segmentation (Asif Hanif et al., 2023)

{{<citation>}}

Asif Hanif, Muzammal Naseer, Salman Khan, Mubarak Shah, Fahad Shahbaz Khan. (2023)  
**Frequency Domain Adversarial Training for Robust Volumetric Medical Segmentation**  

---
Primary Category: eess.IV
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Adversarial Training  
[Paper Link](http://arxiv.org/abs/2307.07269v2)  

---


**ABSTRACT**  
It is imperative to ensure the robustness of deep learning models in critical applications such as, healthcare. While recent advances in deep learning have improved the performance of volumetric medical image segmentation models, these models cannot be deployed for real-world applications immediately due to their vulnerability to adversarial attacks. We present a 3D frequency domain adversarial attack for volumetric medical image segmentation models and demonstrate its advantages over conventional input or voxel domain attacks. Using our proposed attack, we introduce a novel frequency domain adversarial training approach for optimizing a robust model against voxel and frequency domain attacks. Moreover, we propose frequency consistency loss to regulate our frequency domain adversarial training that achieves a better tradeoff between model's performance on clean and adversarial samples. Code is publicly available at https://github.com/asif-hanif/vafa.

{{</citation>}}


### (89/99) Masked Autoencoders for Unsupervised Anomaly Detection in Medical Images (Mariana-Iuliana Georgescu, 2023)

{{<citation>}}

Mariana-Iuliana Georgescu. (2023)  
**Masked Autoencoders for Unsupervised Anomaly Detection in Medical Images**  

---
Primary Category: eess.IV
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2307.07534v1)  

---


**ABSTRACT**  
Pathological anomalies exhibit diverse appearances in medical imaging, making it difficult to collect and annotate a representative amount of data required to train deep learning models in a supervised setting. Therefore, in this work, we tackle anomaly detection in medical images training our framework using only healthy samples. We propose to use the Masked Autoencoder model to learn the structure of the normal samples, then train an anomaly classifier on top of the difference between the original image and the reconstruction provided by the masked autoencoder. We train the anomaly classifier in a supervised manner using as negative samples the reconstruction of the healthy scans, while as positive samples, we use pseudo-abnormal scans obtained via our novel pseudo-abnormal module. The pseudo-abnormal module alters the reconstruction of the normal samples by changing the intensity of several regions. We conduct experiments on two medical image data sets, namely BRATS2020 and LUNA16 and compare our method with four state-of-the-art anomaly detection frameworks, namely AST, RD4AD, AnoVAEGAN and f-AnoGAN.

{{</citation>}}


## cs.SD (1)



### (90/99) AudioInceptionNeXt: TCL AI LAB Submission to EPIC-SOUND Audio-Based-Interaction-Recognition Challenge 2023 (Kin Wai Lau et al., 2023)

{{<citation>}}

Kin Wai Lau, Yasar Abbas Ur Rehman, Yuyang Xie, Lan Ma. (2023)  
**AudioInceptionNeXt: TCL AI LAB Submission to EPIC-SOUND Audio-Based-Interaction-Recognition Challenge 2023**  

---
Primary Category: cs.SD
Categories: cs-AI, cs-SD, cs.SD, eess-AS  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.07265v1)  

---


**ABSTRACT**  
This report presents the technical details of our submission to the 2023 Epic-Kitchen EPIC-SOUNDS Audio-Based Interaction Recognition Challenge. The task is to learn the mapping from audio samples to their corresponding action labels. To achieve this goal, we propose a simple yet effective single-stream CNN-based architecture called AudioInceptionNeXt that operates on the time-frequency log-mel-spectrogram of the audio samples. Motivated by the design of the InceptionNeXt, we propose parallel multi-scale depthwise separable convolutional kernels in the AudioInceptionNeXt block, which enable the model to learn the time and frequency information more effectively. The large-scale separable kernels capture the long duration of activities and the global frequency semantic information, while the small-scale separable kernels capture the short duration of activities and local details of frequency information. Our approach achieved 55.43% of top-1 accuracy on the challenge test set, ranked as 1st on the public leaderboard. Codes are available anonymously at https://github.com/StevenLauHKHK/AudioInceptionNeXt.git.

{{</citation>}}


## cs.HC (1)



### (91/99) Visual Explanations with Attributions and Counterfactuals on Time Series Classification (Udo Schlegel et al., 2023)

{{<citation>}}

Udo Schlegel, Daniela Oelke, Daniel A. Keim, Mennatallah El-Assady. (2023)  
**Visual Explanations with Attributions and Counterfactuals on Time Series Classification**  

---
Primary Category: cs.HC
Categories: cs-HC, cs-LG, cs.HC  
Keywords: AI, Time Series  
[Paper Link](http://arxiv.org/abs/2307.08494v1)  

---


**ABSTRACT**  
With the rising necessity of explainable artificial intelligence (XAI), we see an increase in task-dependent XAI methods on varying abstraction levels. XAI techniques on a global level explain model behavior and on a local level explain sample predictions. We propose a visual analytics workflow to support seamless transitions between global and local explanations, focusing on attributions and counterfactuals on time series classification. In particular, we adapt local XAI techniques (attributions) that are developed for traditional datasets (images, text) to analyze time series classification, a data type that is typically less intelligible to humans. To generate a global overview, we apply local attribution methods to the data, creating explanations for the whole dataset. These explanations are projected onto two dimensions, depicting model behavior trends, strategies, and decision boundaries. To further inspect the model decision-making as well as potential data errors, a what-if analysis facilitates hypothesis generation and verification on both the global and local levels. We constantly collected and incorporated expert user feedback, as well as insights based on their domain knowledge, resulting in a tailored analysis workflow and system that tightly integrates time series transformations into explanations. Lastly, we present three use cases, verifying that our technique enables users to (1)~explore data transformations and feature relevance, (2)~identify model behavior and decision boundaries, as well as, (3)~the reason for misclassifications.

{{</citation>}}


## cs.NE (2)



### (92/99) Long Short-term Memory with Two-Compartment Spiking Neuron (Shimin Zhang et al., 2023)

{{<citation>}}

Shimin Zhang, Qu Yang, Chenxiang Ma, Jibin Wu, Haizhou Li, Kay Chen Tan. (2023)  
**Long Short-term Memory with Two-Compartment Spiking Neuron**  

---
Primary Category: cs.NE
Categories: cs-NE, cs.NE  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2307.07231v1)  

---


**ABSTRACT**  
The identification of sensory cues associated with potential opportunities and dangers is frequently complicated by unrelated events that separate useful cues by long delays. As a result, it remains a challenging task for state-of-the-art spiking neural networks (SNNs) to identify long-term temporal dependencies since bridging the temporal gap necessitates an extended memory capacity. To address this challenge, we propose a novel biologically inspired Long Short-Term Memory Leaky Integrate-and-Fire spiking neuron model, dubbed LSTM-LIF. Our model incorporates carefully designed somatic and dendritic compartments that are tailored to retain short- and long-term memories. The theoretical analysis further confirms its effectiveness in addressing the notorious vanishing gradient problem. Our experimental results, on a diverse range of temporal classification tasks, demonstrate superior temporal classification capability, rapid training convergence, strong network generalizability, and high energy efficiency of the proposed LSTM-LIF model. This work, therefore, opens up a myriad of opportunities for resolving challenging temporal processing tasks on emerging neuromorphic computing machines.

{{</citation>}}


### (93/99) SLSSNN: High energy efficiency spike-train level spiking neural networks with spatio-temporal conversion (Changqing Xu et al., 2023)

{{<citation>}}

Changqing Xu, Yi Liu, Yintang Yang. (2023)  
**SLSSNN: High energy efficiency spike-train level spiking neural networks with spatio-temporal conversion**  

---
Primary Category: cs.NE
Categories: cs-NE, cs.NE  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2307.07136v1)  

---


**ABSTRACT**  
Brain-inspired spiking neuron networks (SNNs) have attracted widespread research interest due to their low power features, high biological plausibility, and strong spatiotemporal information processing capability. Although adopting a surrogate gradient (SG) makes the non-differentiability SNN trainable, achieving comparable accuracy for ANNs and keeping low-power features simultaneously is still tricky. In this paper, we proposed an energy-efficient spike-train level spiking neural network (SLSSNN) with low computational cost and high accuracy. In the SLSSNN, spatio-temporal conversion blocks (STCBs) are applied to replace the convolutional and ReLU layers to keep the low power features of SNNs and improve accuracy. However, SLSSNN cannot adopt backpropagation algorithms directly due to the non-differentiability nature of spike trains. We proposed a suitable learning rule for SLSSNNs by deducing the equivalent gradient of STCB. We evaluate the proposed SLSSNN on static and neuromorphic datasets, including Fashion-Mnist, Cifar10, Cifar100, TinyImageNet, and DVS-Cifar10. The experiment results show that our proposed SLSSNN outperforms the state-of-the-art accuracy on nearly all datasets, using fewer time steps and being highly energy-efficient.

{{</citation>}}


## cs.DL (1)



### (94/99) Aspect-Driven Structuring of Historical Dutch Newspaper Archives (Hermann Kroll et al., 2023)

{{<citation>}}

Hermann Kroll, Christin Katharina Kreutz, Mirjam Cuper, Bill Matthias Thang, Wolf-Tilo Balke. (2023)  
**Aspect-Driven Structuring of Historical Dutch Newspaper Archives**  

---
Primary Category: cs.DL
Categories: cs-DL, cs.DL  
Keywords: OCR  
[Paper Link](http://arxiv.org/abs/2307.09203v1)  

---


**ABSTRACT**  
Digital libraries oftentimes provide access to historical newspaper archives via keyword-based search. Historical figures and their roles are particularly interesting cognitive access points in historical research. Structuring and clustering news articles would allow more sophisticated access for users to explore such information. However, real-world limitations such as the lack of training data, licensing restrictions and non-English text with OCR errors make the composition of such a system difficult and cost-intensive in practice. In this work we tackle these issues with the showcase of the National Library of the Netherlands by introducing a role-based interface that structures news articles on historical persons. In-depth, component-wise evaluations and interviews with domain experts highlighted our prototype's effectiveness and appropriateness for a real-world digital library collection.

{{</citation>}}


## physics.soc-ph (1)



### (95/99) Global path preference and local response: A reward decomposition approach for network path choice analysis in the presence of locally perceived attributes (Yuki Oyama, 2023)

{{<citation>}}

Yuki Oyama. (2023)  
**Global path preference and local response: A reward decomposition approach for network path choice analysis in the presence of locally perceived attributes**  

---
Primary Category: physics.soc-ph
Categories: cs-LG, econ-EM, physics-soc-ph, physics.soc-ph  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2307.08646v1)  

---


**ABSTRACT**  
This study performs an attribute-level analysis of the global and local path preferences of network travelers. To this end, a reward decomposition approach is proposed and integrated into a link-based recursive (Markovian) path choice model. The approach decomposes the instantaneous reward function associated with each state-action pair into the global utility, a function of attributes globally perceived from anywhere in the network, and the local utility, a function of attributes that are only locally perceived from the current state. Only the global utility then enters the value function of each state, representing the future expected utility toward the destination. This global-local path choice model with decomposed reward functions allows us to analyze to what extent and which attributes affect the global and local path choices of agents. Moreover, unlike most adaptive path choice models, the proposed model can be estimated based on revealed path observations (without the information of plans) and as efficiently as deterministic recursive path choice models. The model was applied to the real pedestrian path choice observations in an urban street network where the green view index was extracted as a visual street quality from Google Street View images. The result revealed that pedestrians locally perceive and react to the visual street quality, rather than they have the pre-trip global perception on it. Furthermore, the simulation results using the estimated models suggested the importance of location selection of interventions when policy-related attributes are only locally perceived by travelers.

{{</citation>}}


## math.NA (1)



### (96/99) A Simple Embedding Method for Scalar Hyperbolic Conservation Laws on Implicit Surfaces (Chun Kit Hung et al., 2023)

{{<citation>}}

Chun Kit Hung, Shingyu Leung. (2023)  
**A Simple Embedding Method for Scalar Hyperbolic Conservation Laws on Implicit Surfaces**  

---
Primary Category: math.NA
Categories: cs-NA, math-NA, math.NA  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2307.07151v1)  

---


**ABSTRACT**  
We have developed a new embedding method for solving scalar hyperbolic conservation laws on surfaces. The approach represents the interface implicitly by a signed distance function following the typical level set method and some embedding methods. Instead of solving the equation explicitly on the surface, we introduce a modified partial differential equation in a small neighborhood of the interface. This embedding equation is developed based on a push-forward operator that can extend any tangential flux vectors from the surface to a neighboring level surface. This operator is easy to compute and involves only the level set function and the corresponding Hessian. The resulting solution is constant in the normal direction of the interface. To demonstrate the accuracy and effectiveness of our method, we provide some two- and three-dimensional examples.

{{</citation>}}


## cs.DC (1)



### (97/99) Federated Learning-Empowered AI-Generated Content in Wireless Networks (Xumin Huang et al., 2023)

{{<citation>}}

Xumin Huang, Peichun Li, Hongyang Du, Jiawen Kang, Dusit Niyato, Dong In Kim, Yuan Wu. (2023)  
**Federated Learning-Empowered AI-Generated Content in Wireless Networks**  

---
Primary Category: cs.DC
Categories: cs-AI, cs-DC, cs.DC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.07146v1)  

---


**ABSTRACT**  
Artificial intelligence generated content (AIGC) has emerged as a promising technology to improve the efficiency, quality, diversity and flexibility of the content creation process by adopting a variety of generative AI models. Deploying AIGC services in wireless networks has been expected to enhance the user experience. However, the existing AIGC service provision suffers from several limitations, e.g., the centralized training in the pre-training, fine-tuning and inference processes, especially their implementations in wireless networks with privacy preservation. Federated learning (FL), as a collaborative learning framework where the model training is distributed to cooperative data owners without the need for data sharing, can be leveraged to simultaneously improve learning efficiency and achieve privacy protection for AIGC. To this end, we present FL-based techniques for empowering AIGC, and aim to enable users to generate diverse, personalized, and high-quality content. Furthermore, we conduct a case study of FL-aided AIGC fine-tuning by using the state-of-the-art AIGC model, i.e., stable diffusion model. Numerical results show that our scheme achieves advantages in effectively reducing the communication cost and training latency and privacy protection. Finally, we highlight several major research directions and open issues for the convergence of FL and AIGC.

{{</citation>}}


## stat.AP (1)



### (98/99) Digital Health Discussion Through Articles Published Until the Year 2021: A Digital Topic Modeling Approach (Junhyoun Sung et al., 2023)

{{<citation>}}

Junhyoun Sung, Hyoungsook Kim. (2023)  
**Digital Health Discussion Through Articles Published Until the Year 2021: A Digital Topic Modeling Approach**  

---
Primary Category: stat.AP
Categories: cs-IR, stat-AP, stat.AP  
Keywords: Topic Model, Topic Modeling  
[Paper Link](http://arxiv.org/abs/2307.07130v1)  

---


**ABSTRACT**  
The digital health industry has grown in popularity since the 2010s, but there has been limited analysis of the topics discussed in the field across academic disciplines. This study aims to analyze the research trends of digital health-related articles published on the Web of Science until 2021, in order to understand the concentration, scope, and characteristics of the research. 15,950 digital health-related papers from the top 10 academic fields were analyzed using the Web of Science. The papers were grouped into three domains: public health, medicine, and electrical engineering and computer science (EECS). Two time periods (2012-2016 and 2017-2021) were compared using Latent Dirichlet Allocation (LDA) for topic modeling. The number of topics was determined based on coherence score, and topic compositions were compared using a homogeneity test. The number of optimal topics varied across domains and time periods. For public health, the first and second halves had 13 and 19 topics, respectively. Medicine had 14 and 25 topics, and EECS had 7 and 21 topics. Text analysis revealed shared topics among the domains, but with variations in composition. The homogeneity test confirmed significant differences between the groups (p<2.2e-16). Six dominant themes emerged, including journal article methodology, information technology, medical issues, population demographics, social phenomena, and healthcare. Digital health research is expanding and evolving, particularly in relation to Covid-19, where topics such as depression and mental disorders, education, and physical activity have gained prominence. There was no bias in topic composition among the three domains, but other fields like kinesiology or psychology could contribute to future digital health research. Exploring expanded topics that reflect people's needs for digital health over time will be crucial.

{{</citation>}}


## cs.CY (1)



### (99/99) Ethics in the Age of AI: An Analysis of AI Practitioners' Awareness and Challenges (Aastha Pant et al., 2023)

{{<citation>}}

Aastha Pant, Rashina Hoda, Simone V. Spiegler, Chakkrit Tantithamthavorn, Burak Turhan. (2023)  
**Ethics in the Age of AI: An Analysis of AI Practitioners' Awareness and Challenges**  

---
Primary Category: cs.CY
Categories: cs-AI, cs-CY, cs-SE, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.10057v1)  

---


**ABSTRACT**  
Ethics in AI has become a debated topic of public and expert discourse in recent years. But what do people who build AI - AI practitioners - have to say about their understanding of AI ethics and the challenges associated with incorporating it in the AI-based systems they develop? Understanding AI practitioners' views on AI ethics is important as they are the ones closest to the AI systems and can bring about changes and improvements. We conducted a survey aimed at understanding AI practitioners' awareness of AI ethics and their challenges in incorporating ethics. Based on 100 AI practitioners' responses, our findings indicate that majority of AI practitioners had a reasonable familiarity with the concept of AI ethics, primarily due to workplace rules and policies. Privacy protection and security was the ethical principle that majority of them were aware of. Formal education/training was considered somewhat helpful in preparing practitioners to incorporate AI ethics. The challenges that AI practitioners faced in the development of ethical AI-based systems included (i) general challenges, (ii) technology-related challenges and (iii) human-related challenges. We also identified areas needing further investigation and provided recommendations to assist AI practitioners and companies in incorporating ethics into AI development.

{{</citation>}}
