---
draft: false
title: "arXiv @ 2023.08.28"
date: 2023-08-28
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.08.28"
    identifier: arxiv_20230828
    parent: 202308_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CV (14)](#cscv-14)
- [cs.LG (8)](#cslg-8)
- [cs.SE (6)](#csse-6)
- [cs.CL (7)](#cscl-7)
- [cs.SI (1)](#cssi-1)
- [cs.AR (1)](#csar-1)
- [cs.AI (6)](#csai-6)
- [eess.SP (2)](#eesssp-2)
- [eess.IV (2)](#eessiv-2)
- [cs.HC (2)](#cshc-2)
- [cs.DC (1)](#csdc-1)
- [cs.SD (1)](#cssd-1)
- [cs.RO (1)](#csro-1)

## cs.CV (14)



### (1/52) Fixating on Attention: Integrating Human Eye Tracking into Vision Transformers (Sharath Koorathota et al., 2023)

{{<citation>}}

Sharath Koorathota, Nikolas Papadopoulos, Jia Li Ma, Shruti Kumar, Xiaoxiao Sun, Arunesh Mittal, Patrick Adelman, Paul Sajda. (2023)  
**Fixating on Attention: Integrating Human Eye Tracking into Vision Transformers**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Attention, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.13969v1)  

---


**ABSTRACT**  
Modern transformer-based models designed for computer vision have outperformed humans across a spectrum of visual tasks. However, critical tasks, such as medical image interpretation or autonomous driving, still require reliance on human judgments. This work demonstrates how human visual input, specifically fixations collected from an eye-tracking device, can be integrated into transformer models to improve accuracy across multiple driving situations and datasets. First, we establish the significance of fixation regions in left-right driving decisions, as observed in both human subjects and a Vision Transformer (ViT). By comparing the similarity between human fixation maps and ViT attention weights, we reveal the dynamics of overlap across individual heads and layers. This overlap is exploited for model pruning without compromising accuracy. Thereafter, we incorporate information from the driving scene with fixation data, employing a "joint space-fixation" (JSF) attention setup. Lastly, we propose a "fixation-attention intersection" (FAX) loss to train the ViT model to attend to the same regions that humans fixated on. We find that the ViT performance is improved in accuracy and number of training epochs when using JSF and FAX. These results hold significant implications for human-guided artificial intelligence.

{{</citation>}}


### (2/52) Transfer Learning for Microstructure Segmentation with CS-UNet: A Hybrid Algorithm with Transformer and CNN Encoders (Khaled Alrfou et al., 2023)

{{<citation>}}

Khaled Alrfou, Tian Zhao, Amir Kordijazi. (2023)  
**Transfer Learning for Microstructure Segmentation with CS-UNet: A Hybrid Algorithm with Transformer and CNN Encoders**  

---
Primary Category: cs.CV  
Categories: cond-mat-mtrl-sci, cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.13917v1)  

---


**ABSTRACT**  
Transfer learning improves the performance of deep learning models by initializing them with parameters pre-trained on larger datasets. Intuitively, transfer learning is more effective when pre-training is on the in-domain datasets. A recent study by NASA has demonstrated that the microstructure segmentation with encoder-decoder algorithms benefits more from CNN encoders pre-trained on microscopy images than from those pre-trained on natural images. However, CNN models only capture the local spatial relations in images. In recent years, attention networks such as Transformers are increasingly used in image analysis to capture the long-range relations between pixels. In this study, we compare the segmentation performance of Transformer and CNN models pre-trained on microscopy images with those pre-trained on natural images. Our result partially confirms the NASA study that the segmentation performance of out-of-distribution images (taken under different imaging and sample conditions) is significantly improved when pre-training on microscopy images. However, the performance gain for one-shot and few-shot learning is more modest with Transformers. We also find that for image segmentation, the combination of pre-trained Transformers and CNN encoders are consistently better than pre-trained CNN encoders alone. Our dataset (of about 50,000 images) combines the public portion of the NASA dataset with additional images we collected. Even with much less training data, our pre-trained models have significantly better performance for image segmentation. This result suggests that Transformers and CNN complement each other and when pre-trained on microscopy images, they are more beneficial to the downstream tasks.

{{</citation>}}


### (3/52) Exploring Human Crowd Patterns and Categorization in Video Footage for Enhanced Security and Surveillance using Computer Vision and Machine Learning (Afnan Alazbah et al., 2023)

{{<citation>}}

Afnan Alazbah, Khalid Fakeeh, Osama Rabie. (2023)  
**Exploring Human Crowd Patterns and Categorization in Video Footage for Enhanced Security and Surveillance using Computer Vision and Machine Learning**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Computer Vision, Security  
[Paper Link](http://arxiv.org/abs/2308.13910v1)  

---


**ABSTRACT**  
Computer vision and machine learning have brought revolutionary shifts in perception for researchers, scientists, and the general populace. Once thought to be unattainable, these technologies have achieved the seemingly impossible. Their exceptional applications in diverse fields like security, agriculture, and education are a testament to their impact. However, the full potential of computer vision remains untapped. This paper explores computer vision's potential in security and surveillance, presenting a novel approach to track motion in videos. By categorizing motion into Arcs, Lanes, Converging/Diverging, and Random/Block motions using Motion Information Images and Blockwise dominant motion data, the paper examines different optical flow techniques, CNN models, and machine learning models. Successfully achieving its objectives with promising accuracy, the results can train anomaly-detection models, provide behavioral insights based on motion, and enhance scene comprehension.

{{</citation>}}


### (4/52) Semi-Supervised Semantic Segmentation via Marginal Contextual Information (Moshe Kimhi et al., 2023)

{{<citation>}}

Moshe Kimhi, Shai Kimhi, Evgenii Zheltonozhskii, Or Litany, Chaim Baskin. (2023)  
**Semi-Supervised Semantic Segmentation via Marginal Contextual Information**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Semantic Segmentation, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2308.13900v1)  

---


**ABSTRACT**  
We present a novel confidence refinement scheme that enhances pseudo-labels in semi-supervised semantic segmentation. Unlike current leading methods, which filter pixels with low-confidence predictions in isolation, our approach leverages the spatial correlation of labels in segmentation maps by grouping neighboring pixels and considering their pseudo-labels collectively. With this contextual information, our method, named S4MC, increases the amount of unlabeled data used during training while maintaining the quality of the pseudo-labels, all with negligible computational overhead. Through extensive experiments on standard benchmarks, we demonstrate that S4MC outperforms existing state-of-the-art semi-supervised learning approaches, offering a promising solution for reducing the cost of acquiring dense annotations. For example, S4MC achieves a 1.29 mIoU improvement over the prior state-of-the-art method on PASCAL VOC 12 with 366 annotated images. The code to reproduce our experiments is available at https://s4mcontext.github.io/

{{</citation>}}


### (5/52) Joint Gaze-Location and Gaze-Object Detection (Danyang Tu et al., 2023)

{{<citation>}}

Danyang Tu, Wei Shen, Wei Sun, Xiongkuo Min, Guangtao Zhai. (2023)  
**Joint Gaze-Location and Gaze-Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Object Detection  
[Paper Link](http://arxiv.org/abs/2308.13857v1)  

---


**ABSTRACT**  
This paper proposes an efficient and effective method for joint gaze location detection (GL-D) and gaze object detection (GO-D), \emph{i.e.}, gaze following detection. Current approaches frame GL-D and GO-D as two separate tasks, employing a multi-stage framework where human head crops must first be detected and then be fed into a subsequent GL-D sub-network, which is further followed by an additional object detector for GO-D. In contrast, we reframe the gaze following detection task as detecting human head locations and their gaze followings simultaneously, aiming at jointly detect human gaze location and gaze object in a unified and single-stage pipeline. To this end, we propose GTR, short for \underline{G}aze following detection \underline{TR}ansformer, streamlining the gaze following detection pipeline by eliminating all additional components, leading to the first unified paradigm that unites GL-D and GO-D in a fully end-to-end manner. GTR enables an iterative interaction between holistic semantics and human head features through a hierarchical structure, inferring the relations of salient objects and human gaze from the global image context and resulting in an impressive accuracy. Concretely, GTR achieves a 12.1 mAP gain ($\mathbf{25.1}\%$) on GazeFollowing and a 18.2 mAP gain ($\mathbf{43.3\%}$) on VideoAttentionTarget for GL-D, as well as a 19 mAP improvement ($\mathbf{45.2\%}$) on GOO-Real for GO-D. Meanwhile, unlike existing systems detecting gaze following sequentially due to the need for a human head as input, GTR has the flexibility to comprehend any number of people's gaze followings simultaneously, resulting in high efficiency. Specifically, GTR introduces over a $\times 9$ improvement in FPS and the relative gap becomes more pronounced as the human number grows.

{{</citation>}}


### (6/52) Point-Query Quadtree for Crowd Counting, Localization, and More (Chengxin Liu et al., 2023)

{{<citation>}}

Chengxin Liu, Hao Lu, Zhiguo Cao, Tongliang Liu. (2023)  
**Point-Query Quadtree for Crowd Counting, Localization, and More**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.13814v1)  

---


**ABSTRACT**  
We show that crowd counting can be viewed as a decomposable point querying process. This formulation enables arbitrary points as input and jointly reasons whether the points are crowd and where they locate. The querying processing, however, raises an underlying problem on the number of necessary querying points. Too few imply underestimation; too many increase computational overhead. To address this dilemma, we introduce a decomposable structure, i.e., the point-query quadtree, and propose a new counting model, termed Point quEry Transformer (PET). PET implements decomposable point querying via data-dependent quadtree splitting, where each querying point could split into four new points when necessary, thus enabling dynamic processing of sparse and dense regions. Such a querying process yields an intuitive, universal modeling of crowd as both the input and output are interpretable and steerable. We demonstrate the applications of PET on a number of crowd-related tasks, including fully-supervised crowd counting and localization, partial annotation learning, and point annotation refinement, and also report state-of-the-art performance. For the first time, we show that a single counting model can address multiple crowd-related tasks across different learning paradigms. Code is available at https://github.com/cxliu0/PET.

{{</citation>}}


### (7/52) VIDES: Virtual Interior Design via Natural Language and Visual Guidance (Minh-Hien Le et al., 2023)

{{<citation>}}

Minh-Hien Le, Chi-Bien Chu, Khanh-Duy Le, Tam V. Nguyen, Minh-Triet Tran, Trung-Nghia Le. (2023)  
**VIDES: Virtual Interior Design via Natural Language and Visual Guidance**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.13795v1)  

---


**ABSTRACT**  
Interior design is crucial in creating aesthetically pleasing and functional indoor spaces. However, developing and editing interior design concepts requires significant time and expertise. We propose Virtual Interior DESign (VIDES) system in response to this challenge. Leveraging cutting-edge technology in generative AI, our system can assist users in generating and editing indoor scene concepts quickly, given user text description and visual guidance. Using both visual guidance and language as the conditional inputs significantly enhances the accuracy and coherence of the generated scenes, resulting in visually appealing designs. Through extensive experimentation, we demonstrate the effectiveness of VIDES in developing new indoor concepts, changing indoor styles, and replacing and removing interior objects. The system successfully captures the essence of users' descriptions while providing flexibility for customization. Consequently, this system can potentially reduce the entry barrier for indoor design, making it more accessible to users with limited technical skills and reducing the time required to create high-quality images. Individuals who have a background in design can now easily communicate their ideas visually and effectively present their design concepts. https://sites.google.com/view/ltnghia/research/VIDES

{{</citation>}}


### (8/52) SOGDet: Semantic-Occupancy Guided Multi-view 3D Object Detection (Qiu Zhou et al., 2023)

{{<citation>}}

Qiu Zhou, Jinming Cao, Hanchao Leng, Yifang Yin, Yu Kun, Roger Zimmermann. (2023)  
**SOGDet: Semantic-Occupancy Guided Multi-view 3D Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2308.13794v1)  

---


**ABSTRACT**  
In the field of autonomous driving, accurate and comprehensive perception of the 3D environment is crucial. Bird's Eye View (BEV) based methods have emerged as a promising solution for 3D object detection using multi-view images as input. However, existing 3D object detection methods often ignore the physical context in the environment, such as sidewalk and vegetation, resulting in sub-optimal performance. In this paper, we propose a novel approach called SOGDet (Semantic-Occupancy Guided Multi-view 3D Object Detection), that leverages a 3D semantic-occupancy branch to improve the accuracy of 3D object detection. In particular, the physical context modeled by semantic occupancy helps the detector to perceive the scenes in a more holistic view. Our SOGDet is flexible to use and can be seamlessly integrated with most existing BEV-based methods. To evaluate its effectiveness, we apply this approach to several state-of-the-art baselines and conduct extensive experiments on the exclusive nuScenes dataset. Our results show that SOGDet consistently enhance the performance of three baseline methods in terms of nuScenes Detection Score (NDS) and mean Average Precision (mAP). This indicates that the combination of 3D object detection and 3D semantic occupancy leads to a more comprehensive perception of the 3D environment, thereby aiding build more robust autonomous driving systems. The codes are available at: https://github.com/zhouqiu/SOGDet.

{{</citation>}}


### (9/52) Zero-Shot Edge Detection with SCESAME: Spectral Clustering-based Ensemble for Segment Anything Model Estimation (Hiroaki Yamagiwa et al., 2023)

{{<citation>}}

Hiroaki Yamagiwa, Yusuke Takase, Hiroyuki Kambe, Ryosuke Nakamoto. (2023)  
**Zero-Shot Edge Detection with SCESAME: Spectral Clustering-based Ensemble for Segment Anything Model Estimation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2308.13779v1)  

---


**ABSTRACT**  
This paper proposes a novel zero-shot edge detection with SCESAME, which stands for Spectral Clustering-based Ensemble for Segment Anything Model Estimation, based on the recently proposed Segment Anything Model (SAM). SAM is a foundation model for segmentation tasks, and one of the interesting applications of SAM is Automatic Mask Generation (AMG), which generates zero-shot segmentation masks of an entire image. AMG can be applied to edge detection, but suffers from the problem of overdetecting edges. Edge detection with SCESAME overcomes this problem by three steps: (1) eliminating small generated masks, (2) combining masks by spectral clustering, taking into account mask positions and overlaps, and (3) removing artifacts after edge detection. We performed edge detection experiments on two datasets, BSDS500 and NYUDv2. Although our zero-shot approach is simple, the experimental results on BSDS500 showed almost identical performance to human performance and CNN-based methods from seven years ago. In the NYUDv2 experiments, it performed almost as well as recent CNN-based methods. These results indicate that our method has the potential to be a strong baseline for future zero-shot edge detection methods. Furthermore, SCESAME is not only applicable to edge detection, but also to other downstream zero-shot tasks.

{{</citation>}}


### (10/52) Bengali Document Layout Analysis with Detectron2 (Md Ataullha et al., 2023)

{{<citation>}}

Md Ataullha, Mahedi Hassan Rabby, Mushfiqur Rahman, Tahsina Bintay Azam. (2023)  
**Bengali Document Layout Analysis with Detectron2**  

---
Primary Category: cs.CV  
Categories: I-4; I-4-6; I-7, cs-CV, cs-LG, cs.CV  
Keywords: OCR  
[Paper Link](http://arxiv.org/abs/2308.13769v1)  

---


**ABSTRACT**  
Document digitization is vital for preserving historical records, efficient document management, and advancing OCR (Optical Character Recognition) research. Document Layout Analysis (DLA) involves segmenting documents into meaningful units like text boxes, paragraphs, images, and tables. Challenges arise when dealing with diverse layouts, historical documents, and unique scripts like Bengali, hindered by the lack of comprehensive Bengali DLA datasets. We improved the accuracy of the DLA model for Bengali documents by utilizing advanced Mask R-CNN models available in the Detectron2 library. Our evaluation involved three variants: Mask R-CNN R-50, R-101, and X-101, both with and without pretrained weights from PubLayNet, on the BaDLAD dataset, which contains human-annotated Bengali documents in four categories: text boxes, paragraphs, images, and tables. Results show the effectiveness of these models in accurately segmenting Bengali documents. We discuss speed-accuracy tradeoffs and underscore the significance of pretrained weights. Our findings expand the applicability of Mask R-CNN in document layout analysis, efficient document management, and OCR research while suggesting future avenues for fine-tuning and data augmentation.

{{</citation>}}


### (11/52) Unified Single-Stage Transformer Network for Efficient RGB-T Tracking (Jianqiang Xia et al., 2023)

{{<citation>}}

Jianqiang Xia, DianXi Shi, Ke Song, Linna Song, XiaoLei Wang, Songchang Jin, Li Zhou, Yu Cheng, Lei Jin, Zheng Zhu, Jianan Li, Gang Wang, Junliang Xing, Jian Zhao. (2023)  
**Unified Single-Stage Transformer Network for Efficient RGB-T Tracking**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.13764v1)  

---


**ABSTRACT**  
Most existing RGB-T tracking networks extract modality features in a separate manner, which lacks interaction and mutual guidance between modalities. This limits the network's ability to adapt to the diverse dual-modality appearances of targets and the dynamic relationships between the modalities. Additionally, the three-stage fusion tracking paradigm followed by these networks significantly restricts the tracking speed. To overcome these problems, we propose a unified single-stage Transformer RGB-T tracking network, namely USTrack, which unifies the above three stages into a single ViT (Vision Transformer) backbone with a dual embedding layer through self-attention mechanism. With this structure, the network can extract fusion features of the template and search region under the mutual interaction of modalities. Simultaneously, relation modeling is performed between these features, efficiently obtaining the search region fusion features with better target-background discriminability for prediction. Furthermore, we introduce a novel feature selection mechanism based on modality reliability to mitigate the influence of invalid modalities for prediction, further improving the tracking performance. Extensive experiments on three popular RGB-T tracking benchmarks demonstrate that our method achieves new state-of-the-art performance while maintaining the fastest inference speed 84.2FPS. In particular, MPR/MSR on the short-term and long-term subsets of VTUAV dataset increased by 11.1$\%$/11.7$\%$ and 11.3$\%$/9.7$\%$.

{{</citation>}}


### (12/52) SamDSK: Combining Segment Anything Model with Domain-Specific Knowledge for Semi-Supervised Learning in Medical Image Segmentation (Yizhe Zhang et al., 2023)

{{<citation>}}

Yizhe Zhang, Tao Zhou, Shuo Wang, Ye Wu, Pengfei Gu, Danny Z. Chen. (2023)  
**SamDSK: Combining Segment Anything Model with Domain-Specific Knowledge for Semi-Supervised Learning in Medical Image Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2308.13759v1)  

---


**ABSTRACT**  
The Segment Anything Model (SAM) exhibits a capability to segment a wide array of objects in natural images, serving as a versatile perceptual tool for various downstream image segmentation tasks. In contrast, medical image segmentation tasks often rely on domain-specific knowledge (DSK). In this paper, we propose a novel method that combines the segmentation foundation model (i.e., SAM) with domain-specific knowledge for reliable utilization of unlabeled images in building a medical image segmentation model. Our new method is iterative and consists of two main stages: (1) segmentation model training; (2) expanding the labeled set by using the trained segmentation model, an unlabeled set, SAM, and domain-specific knowledge. These two stages are repeated until no more samples are added to the labeled set. A novel optimal-matching-based method is developed for combining the SAM-generated segmentation proposals and pixel-level and image-level DSK for constructing annotations of unlabeled images in the iterative stage (2). In experiments, we demonstrate the effectiveness of our proposed method for breast cancer segmentation in ultrasound images, polyp segmentation in endoscopic images, and skin lesion segmentation in dermoscopic images. Our work initiates a new direction of semi-supervised learning for medical image segmentation: the segmentation foundation model can be harnessed as a valuable tool for label-efficient segmentation learning in medical image segmentation.

{{</citation>}}


### (13/52) PE-MED: Prompt Enhancement for Interactive Medical Image Segmentation (Ao Chang et al., 2023)

{{<citation>}}

Ao Chang, Xing Tao, Xin Yang, Yuhao Huang, Xinrui Zhou, Jiajun Zeng, Ruobing Huang, Dong Ni. (2023)  
**PE-MED: Prompt Enhancement for Interactive Medical Image Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Attention, Time Series  
[Paper Link](http://arxiv.org/abs/2308.13746v1)  

---


**ABSTRACT**  
Interactive medical image segmentation refers to the accurate segmentation of the target of interest through interaction (e.g., click) between the user and the image. It has been widely studied in recent years as it is less dependent on abundant annotated data and more flexible than fully automated segmentation. However, current studies have not fully explored user-provided prompt information (e.g., points), including the knowledge mined in one interaction, and the relationship between multiple interactions. Thus, in this paper, we introduce a novel framework equipped with prompt enhancement, called PE-MED, for interactive medical image segmentation. First, we introduce a Self-Loop strategy to generate warm initial segmentation results based on the first prompt. It can prevent the highly unfavorable scenarios, such as encountering a blank mask as the initial input after the first interaction. Second, we propose a novel Prompt Attention Learning Module (PALM) to mine useful prompt information in one interaction, enhancing the responsiveness of the network to user clicks. Last, we build a Time Series Information Propagation (TSIP) mechanism to extract the temporal relationships between multiple interactions and increase the model stability. Comparative experiments with other state-of-the-art (SOTA) medical image segmentation algorithms show that our method exhibits better segmentation accuracy and stability.

{{</citation>}}


### (14/52) Devignet: High-Resolution Vignetting Removal via a Dual Aggregated Fusion Transformer With Adaptive Channel Expansion (Shenghong Luo et al., 2023)

{{<citation>}}

Shenghong Luo, Xuhang Chen, Weiwen Chen, Zinuo Li, Shuqiang Wang, Chi-Man Pun. (2023)  
**Devignet: High-Resolution Vignetting Removal via a Dual Aggregated Fusion Transformer With Adaptive Channel Expansion**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.13739v1)  

---


**ABSTRACT**  
Vignetting commonly occurs as a degradation in images resulting from factors such as lens design, improper lens hood usage, and limitations in camera sensors. This degradation affects image details, color accuracy, and presents challenges in computational photography. Existing vignetting removal algorithms predominantly rely on ideal physics assumptions and hand-crafted parameters, resulting in ineffective removal of irregular vignetting and suboptimal results. Moreover, the substantial lack of real-world vignetting datasets hinders the objective and comprehensive evaluation of vignetting removal. To address these challenges, we present Vigset, a pioneering dataset for vignette removal. Vigset includes 983 pairs of both vignetting and vignetting-free high-resolution ($5340\times3697$) real-world images under various conditions. In addition, We introduce DeVigNet, a novel frequency-aware Transformer architecture designed for vignetting removal. Through the Laplacian Pyramid decomposition, we propose the Dual Aggregated Fusion Transformer to handle global features and remove vignetting in the low-frequency domain. Additionally, we introduce the Adaptive Channel Expansion Module to enhance details in the high-frequency domain. The experiments demonstrate that the proposed model outperforms existing state-of-the-art methods.

{{</citation>}}


## cs.LG (8)



### (15/52) Multivariate time series classification with dual attention network (Mojtaba A. Farahani et al., 2023)

{{<citation>}}

Mojtaba A. Farahani, Tara Eslaminokandeh. (2023)  
**Multivariate time series classification with dual attention network**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Attention, Self-Attention  
[Paper Link](http://arxiv.org/abs/2308.13968v1)  

---


**ABSTRACT**  
One of the topics in machine learning that is becoming more and more relevant is multivariate time series classification. Current techniques concentrate on identifying the local important sequence segments or establishing the global long-range dependencies. They frequently disregard the merged data from both global and local features, though. Using dual attention, we explore a novel network (DA-Net) in this research to extract local and global features for multivariate time series classification. The two distinct layers that make up DA-Net are the Squeeze-Excitation Window Attention (SEWA) layer and the Sparse Self-Attention within Windows (SSAW) layer. DA- Net can mine essential local sequence fragments that are necessary for establishing global long-range dependencies based on the two expanded layers.

{{</citation>}}


### (16/52) Drug Interaction Vectors Neural Network: DrIVeNN (Natalie Wang et al., 2023)

{{<citation>}}

Natalie Wang, Casey Overby Taylor. (2023)  
**Drug Interaction Vectors Neural Network: DrIVeNN**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, q-bio-QM  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2308.13891v1)  

---


**ABSTRACT**  
Polypharmacy, the concurrent use of multiple drugs to treat a single condition, is common in patients managing multiple or complex conditions. However, as more drugs are added to the treatment plan, the risk of adverse drug events (ADEs) rises rapidly. Many serious ADEs associated with polypharmacy only become known after the drugs are in use. It is impractical to test every possible drug combination during clinical trials. This issue is particularly prevalent among older adults with cardiovascular disease (CVD) where polypharmacy and ADEs are commonly observed. In this research, our primary objective was to identify key drug features to build and evaluate a model for modeling polypharmacy ADEs. Our secondary objective was to assess our model on a domain-specific case study. We developed a two-layer neural network that incorporated drug features such as molecular structure, drug-protein interactions, and mono drug side effects (DrIVeNN). We assessed DrIVeNN using publicly available side effect databases and determined Principal Component Analysis (PCA) with a variance threshold of 0.95 as the most effective feature selection method. DrIVeNN performed moderately better than state-of-the-art models like RESCAL, DEDICOM, DeepWalk, Decagon, DeepDDI, KGDDI, and KGNN in terms of AUROC for the drug-drug interaction prediction task. We also conducted a domain-specific case study centered on the treatment of cardiovascular disease (CVD). When the best performing model architecture was applied to the CVD treatment cohort, there was a significant increase in performance from the general model. We observed an average AUROC for CVD drug pair prediction increasing from 0.826 (general model) to 0.975 (CVD specific model). Our findings indicate the strong potential of domain-specific models for improving the accuracy of drug-drug interaction predictions.

{{</citation>}}


### (17/52) Price-Discrimination Game for Distributed Resource Management in Federated Learning (Han Zhang et al., 2023)

{{<citation>}}

Han Zhang, Halvin Yang, Guopeng Zhang. (2023)  
**Price-Discrimination Game for Distributed Resource Management in Federated Learning**  

---
Primary Category: cs.LG  
Categories: cs-GT, cs-LG, cs.LG  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2308.13838v1)  

---


**ABSTRACT**  
In vanilla federated learning (FL) such as FedAvg, the parameter server (PS) and multiple distributed clients can form a typical buyer's market, where the number of PS/buyers of FL services is far less than the number of clients/sellers. In order to improve the performance of FL and reduce the cost of motivating clients to participate in FL, this paper proposes to differentiate the pricing for services provided by different clients rather than simply providing the same service pricing for different clients. The price is differentiated based on the performance improvements brought to FL and their heterogeneity in computing and communication capabilities. To this end, a price-discrimination game (PDG) is formulated to comprehensively address the distributed resource management problems in FL, including multi-objective trade-off, client selection, and incentive mechanism. As the PDG is a mixed-integer nonlinear programming (MINLP) problem, a distributed semi-heuristic algorithm with low computational complexity and low communication overhead is designed to solve it. The simulation result verifies the effectiveness of the proposed approach.

{{</citation>}}


### (18/52) Deep Learning for Structure-Preserving Universal Stable Koopman-Inspired Embeddings for Nonlinear Canonical Hamiltonian Dynamics (Pawan Goyal et al., 2023)

{{<citation>}}

Pawan Goyal, Süleyman Yıldız, Peter Benner. (2023)  
**Deep Learning for Structure-Preserving Universal Stable Koopman-Inspired Embeddings for Nonlinear Canonical Hamiltonian Dynamics**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, math-DS  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2308.13835v1)  

---


**ABSTRACT**  
Discovering a suitable coordinate transformation for nonlinear systems enables the construction of simpler models, facilitating prediction, control, and optimization for complex nonlinear systems. To that end, Koopman operator theory offers a framework for global linearization for nonlinear systems, thereby allowing the usage of linear tools for design studies. In this work, we focus on the identification of global linearized embeddings for canonical nonlinear Hamiltonian systems through a symplectic transformation. While this task is often challenging, we leverage the power of deep learning to discover the desired embeddings. Furthermore, to overcome the shortcomings of Koopman operators for systems with continuous spectra, we apply the lifting principle and learn global cubicized embeddings. Additionally, a key emphasis is paid to enforce the bounded stability for the dynamics of the discovered embeddings. We demonstrate the capabilities of deep learning in acquiring compact symplectic coordinate transformation and the corresponding simple dynamical models, fostering data-driven learning of nonlinear canonical Hamiltonian systems, even those with continuous spectra.

{{</citation>}}


### (19/52) Homological Convolutional Neural Networks (Antonio Briola et al., 2023)

{{<citation>}}

Antonio Briola, Yuanrong Wang, Silvia Bartolucci, Tomaso Aste. (2023)  
**Homological Convolutional Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CC, cs-LG, cs.LG  
Keywords: Financial  
[Paper Link](http://arxiv.org/abs/2308.13816v1)  

---


**ABSTRACT**  
Deep learning methods have demonstrated outstanding performances on classification and regression tasks on homogeneous data types (e.g., image, audio, and text data). However, tabular data still poses a challenge with classic machine learning approaches being often computationally cheaper and equally effective than increasingly complex deep learning architectures. The challenge arises from the fact that, in tabular data, the correlation among features is weaker than the one from spatial or semantic relationships in images or natural languages, and the dependency structures need to be modeled without any prior information. In this work, we propose a novel deep learning architecture that exploits the data structural organization through topologically constrained network representations to gain spatial information from sparse tabular data. The resulting model leverages the power of convolutions and is centered on a limited number of concepts from network topology to guarantee (i) a data-centric, deterministic building pipeline; (ii) a high level of interpretability over the inference process; and (iii) an adequate room for scalability. We test our model on 18 benchmark datasets against 5 classic machine learning and 3 deep learning models demonstrating that our approach reaches state-of-the-art performances on these challenging datasets. The code to reproduce all our experiments is provided at https://github.com/FinancialComputingUCL/HomologicalCNN.

{{</citation>}}


### (20/52) DeLELSTM: Decomposition-based Linear Explainable LSTM to Capture Instantaneous and Long-term Effects in Time Series (Chaoqun Wang et al., 2023)

{{<citation>}}

Chaoqun Wang, Yijun Li, Xiangqian Sun, Qi Wu, Dongdong Wang, Zhixiang Huang. (2023)  
**DeLELSTM: Decomposition-based Linear Explainable LSTM to Capture Instantaneous and Long-term Effects in Time Series**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: LSTM, Time Series  
[Paper Link](http://arxiv.org/abs/2308.13797v1)  

---


**ABSTRACT**  
Time series forecasting is prevalent in various real-world applications. Despite the promising results of deep learning models in time series forecasting, especially the Recurrent Neural Networks (RNNs), the explanations of time series models, which are critical in high-stakes applications, have received little attention. In this paper, we propose a Decomposition-based Linear Explainable LSTM (DeLELSTM) to improve the interpretability of LSTM. Conventionally, the interpretability of RNNs only concentrates on the variable importance and time importance. We additionally distinguish between the instantaneous influence of new coming data and the long-term effects of historical data. Specifically, DeLELSTM consists of two components, i.e., standard LSTM and tensorized LSTM. The tensorized LSTM assigns each variable with a unique hidden state making up a matrix $\mathbf{h}_t$, and the standard LSTM models all the variables with a shared hidden state $\mathbf{H}_t$. By decomposing the $\mathbf{H}_t$ into the linear combination of past information $\mathbf{h}_{t-1}$ and the fresh information $\mathbf{h}_{t}-\mathbf{h}_{t-1}$, we can get the instantaneous influence and the long-term effect of each variable. In addition, the advantage of linear regression also makes the explanation transparent and clear. We demonstrate the effectiveness and interpretability of DeLELSTM on three empirical datasets. Extensive experiments show that the proposed method achieves competitive performance against the baseline methods and provides a reliable explanation relative to domain knowledge.

{{</citation>}}


### (21/52) Muffin: A Framework Toward Multi-Dimension AI Fairness by Uniting Off-the-Shelf Models (Yi Sheng et al., 2023)

{{<citation>}}

Yi Sheng, Junhuan Yang, Lei Yang, Yiyu Shi, Jingtongf Hu, Weiwen Jiang. (2023)  
**Muffin: A Framework Toward Multi-Dimension AI Fairness by Uniting Off-the-Shelf Models**  

---
Primary Category: cs.LG  
Categories: cs-CY, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.13730v1)  

---


**ABSTRACT**  
Model fairness (a.k.a., bias) has become one of the most critical problems in a wide range of AI applications. An unfair model in autonomous driving may cause a traffic accident if corner cases (e.g., extreme weather) cannot be fairly regarded; or it will incur healthcare disparities if the AI model misdiagnoses a certain group of people (e.g., brown and black skin). In recent years, there have been emerging research works on addressing unfairness, and they mainly focus on a single unfair attribute, like skin tone; however, real-world data commonly have multiple attributes, among which unfairness can exist in more than one attribute, called 'multi-dimensional fairness'. In this paper, we first reveal a strong correlation between the different unfair attributes, i.e., optimizing fairness on one attribute will lead to the collapse of others. Then, we propose a novel Multi-Dimension Fairness framework, namely Muffin, which includes an automatic tool to unite off-the-shelf models to improve the fairness on multiple attributes simultaneously. Case studies on dermatology datasets with two unfair attributes show that the existing approach can achieve 21.05% fairness improvement on the first attribute while it makes the second attribute unfair by 1.85%. On the other hand, the proposed Muffin can unite multiple models to achieve simultaneously 26.32% and 20.37% fairness improvement on both attributes; meanwhile, it obtains 5.58% accuracy gain.

{{</citation>}}


### (22/52) Time-to-Pattern: Information-Theoretic Unsupervised Learning for Scalable Time Series Summarization (Alireza Ghods et al., 2023)

{{<citation>}}

Alireza Ghods, Trong Nghia Hoang, Diane Cook. (2023)  
**Time-to-Pattern: Information-Theoretic Unsupervised Learning for Scalable Time Series Summarization**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Summarization, Time Series  
[Paper Link](http://arxiv.org/abs/2308.13722v1)  

---


**ABSTRACT**  
Data summarization is the process of generating interpretable and representative subsets from a dataset. Existing time series summarization approaches often search for recurring subsequences using a set of manually devised similarity functions to summarize the data. However, such approaches are fraught with limitations stemming from an exhaustive search coupled with a heuristic definition of series similarity. Such approaches affect the diversity and comprehensiveness of the generated data summaries. To mitigate these limitations, we introduce an approach to time series summarization, called Time-to-Pattern (T2P), which aims to find a set of diverse patterns that together encode the most salient information, following the notion of minimum description length. T2P is implemented as a deep generative model that learns informative embeddings of the discrete time series on a latent space specifically designed to be interpretable. Our synthetic and real-world experiments reveal that T2P discovers informative patterns, even in noisy and complex settings. Furthermore, our results also showcase the improved performance of T2P over previous work in pattern diversity and processing scalability, which conclusively demonstrate the algorithm's effectiveness for time series summarization.

{{</citation>}}


## cs.SE (6)



### (23/52) GPTCloneBench: A comprehensive benchmark of semantic clones and cross-language clones using GPT-3 model and SemanticCloneBench (Ajmain Inqiad Alam et al., 2023)

{{<citation>}}

Ajmain Inqiad Alam, Palash Ranjan Roy, Farouq Al-omari, Chanchal Kumar Roy, Banani Roy, Kevin Schneider. (2023)  
**GPTCloneBench: A comprehensive benchmark of semantic clones and cross-language clones using GPT-3 model and SemanticCloneBench**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: AI, GPT  
[Paper Link](http://arxiv.org/abs/2308.13963v1)  

---


**ABSTRACT**  
Machine learning (ML) has made BigCloneBench popular for semantic clone detection tools. However, BigCloneBench only has a few Java semantic clones. In addition, due to the design principles of how the benchmark was created, imbalance issues have been identified, including the ambiguity in the definition of semantic clones. Thus, ML-based clone detection algorithms trained on BigCloneBench may overlook semantic clones or report incorrect results. The SemanticCloneBench features Stack Overflow clones of several languages. However, it lacks samples for ML-based clone detection. There is also a marked lack of cross-language clone benchmarks. The widely used CLCDSA dataset lacks reusable examples that can't be used in real-world software systems, making it inadequate for ML-based clone detection. The OpenAI GPT-3 model has shown outstanding text production, including code generation and summarization. In this paper, we used the GPT-3 model to generate a complete benchmark for both semantic and cross-language clones. Using SemanticCloneBench's genuine language clones, we tested several prompts to see which yielded better results using GPT-3 question formulation. Then, we used NiCad to filter Type-1 and Type-2 clones from GPT-3 output. We used a GUI-assisted Clone Validator tool to manually validate all clone pairings with nine judges. Functionality testing and CloneCognition verified our benchmark has no syntactic clones. Later, we validated SourcererCC, Oreo and CLCDSA tools on our benchmark. The poor performance of these tools suggests GPTCloneBench has no syntactic clone. From 77,207 Clone pairs of SemanticCloneBench/GPT-3 output, we created a benchmark with 37,149 genuine semantic clone pairs, 19,288 false semantic pairs, and 20,770 cross-language clones across four languages (Java, C, C#, and Python).

{{</citation>}}


### (24/52) A Comparative Study on Reward Models for UI Adaptation with Reinforcement Learning (Daniel Gaspar Figueiredo et al., 2023)

{{<citation>}}

Daniel Gaspar Figueiredo, Silvia Abrahão, Marta Fernández-Diego, Emilio Insfran. (2023)  
**A Comparative Study on Reward Models for UI Adaptation with Reinforcement Learning**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.13937v1)  

---


**ABSTRACT**  
Adapting the User Interface (UI) of software systems to user requirements and the context of use is challenging. The main difficulty consists of suggesting the right adaptation at the right time in the right place in order to make it valuable for end-users. We believe that recent progress in Machine Learning techniques provides useful ways in which to support adaptation more effectively. In particular, Reinforcement learning (RL) can be used to personalise interfaces for each context of use in order to improve the user experience (UX). However, determining the reward of each adaptation alternative is a challenge in RL for UI adaptation. Recent research has explored the use of reward models to address this challenge, but there is currently no empirical evidence on this type of model. In this paper, we propose a confirmatory study design that aims to investigate the effectiveness of two different approaches for the generation of reward models in the context of UI adaptation using RL: (1) by employing a reward model derived exclusively from predictive Human-Computer Interaction (HCI) models (HCI), and (2) by employing predictive HCI models augmented by Human Feedback (HCI&HF). The controlled experiment will use an AB/BA crossover design with two treatments: HCI and HCI&HF. We shall determine how the manipulation of these two treatments will affect the UX when interacting with adaptive user interfaces (AUI). The UX will be measured in terms of user engagement and user satisfaction, which will be operationalized by means of predictive HCI models and the Questionnaire for User Interaction Satisfaction (QUIS), respectively. By comparing the performance of two reward models in terms of their ability to adapt to user preferences with the purpose of improving the UX, our study contributes to the understanding of how reward modelling can facilitate UI adaptation using RL.

{{</citation>}}


### (25/52) Modeling Programmer Attention as Scanpath Prediction (Aakash Bansal et al., 2023)

{{<citation>}}

Aakash Bansal, Chia-Yi Su, Zachary Karas, Yifan Zhang, Yu Huang, Toby Jia-Jun Li, Collin McMillan. (2023)  
**Modeling Programmer Attention as Scanpath Prediction**  

---
Primary Category: cs.SE  
Categories: cs-HC, cs-SE, cs.SE  
Keywords: AI, Attention  
[Paper Link](http://arxiv.org/abs/2308.13920v1)  

---


**ABSTRACT**  
This paper launches a new effort at modeling programmer attention by predicting eye movement scanpaths. Programmer attention refers to what information people intake when performing programming tasks. Models of programmer attention refer to machine prediction of what information is important to people. Models of programmer attention are important because they help researchers build better interfaces, assistive technologies, and more human-like AI. For many years, researchers in SE have built these models based on features such as mouse clicks, key logging, and IDE interactions. Yet the holy grail in this area is scanpath prediction -- the prediction of the sequence of eye fixations a person would take over a visual stimulus. A person's eye movements are considered the most concrete evidence that a person is taking in a piece of information. Scanpath prediction is a notoriously difficult problem, but we believe that the emergence of lower-cost, higher-accuracy eye tracking equipment and better large language models of source code brings a solution within grasp. We present an eye tracking experiment with 27 programmers and a prototype scanpath predictor to present preliminary results and obtain early community feedback.

{{</citation>}}


### (26/52) Which is a better programming assistant? A comparative study between chatgpt and stack overflow (Jinrun Liu et al., 2023)

{{<citation>}}

Jinrun Liu, Xinyu Tang, Linlin Li, Panpan Chen, Yepang Liu. (2023)  
**Which is a better programming assistant? A comparative study between chatgpt and stack overflow**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2308.13851v1)  

---


**ABSTRACT**  
Programmers often seek help from Q\&A websites to resolve issues they encounter during programming. Stack Overflow has been a widely used platform for this purpose for over a decade. Recently, revolutionary AI-powered platforms like ChatGPT have quickly gained popularity among programmers for their efficient and personalized programming assistance via natural language interactions. Both platforms can offer valuable assistance to programmers, but it's unclear which is more effective at enhancing programmer productivity. In our paper, we conducted an exploratory user study to compare the performance of Stack Overflow and ChatGPT in enhancing programmer productivity. Two groups of students with similar programming abilities were instructed to use the two platforms to solve three different types of programming tasks: algorithmic challenges, library usage, and debugging. During the experiments, we measured and compared the quality of code produced and the time taken to complete tasks for the two groups. The results show that, concerning code quality, ChatGPT outperforms Stack Overflow significantly in helping complete algorithmic and library-related tasks, while Stack Overflow is better for debugging tasks. Regarding task completion speed, the ChatGPT group is obviously faster than the Stack Overflow group in the algorithmic challenge, but the two groups have a similar performance in the other two tasks. Additionally, we conducted a post-experiment survey with the participants to understand how the platforms have helped them complete the programming tasks. We analyzed the questionnaires to summarize ChatGPT and Stack Overflow's strengths and weaknesses pointed out by the participants. By comparing these, we identified the reasons behind the two platforms' divergent performances in programming assistance.

{{</citation>}}


### (27/52) EditSum: A Retrieve-and-Edit Framework for Source Code Summarization (Jia Allen Li et al., 2023)

{{<citation>}}

Jia Allen Li, Yongmin Li, Ge Li, Xing Hu, Xin Xia, Zhi Jin. (2023)  
**EditSum: A Retrieve-and-Edit Framework for Source Code Summarization**  

---
Primary Category: cs.SE  
Categories: cs-CL, cs-SE, cs.SE  
Keywords: Summarization  
[Paper Link](http://arxiv.org/abs/2308.13775v1)  

---


**ABSTRACT**  
Existing studies show that code summaries help developers understand and maintain source code. Unfortunately, these summaries are often missing or outdated in software projects. Code summarization aims to generate natural language descriptions automatically for source code. Code summaries are highly structured and have repetitive patterns. Besides the patternized words, a code summary also contains important keywords, which are the key to reflecting the functionality of the code. However, the state-of-the-art approaches perform poorly on predicting the keywords, which leads to the generated summaries suffering a loss in informativeness. To alleviate this problem, this paper proposes a novel retrieve-and-edit approach named EditSum for code summarization. Specifically, EditSum first retrieves a similar code snippet from a pre-defined corpus and treats its summary as a prototype summary to learn the pattern. Then, EditSum edits the prototype automatically to combine the pattern in the prototype with the semantic information of input code. Our motivation is that the retrieved prototype provides a good start-point for post-generation because the summaries of similar code snippets often have the same pattern. The post-editing process further reuses the patternized words in the prototype and generates keywords based on the semantic information of input code. We conduct experiments on a large-scale Java corpus and experimental results demonstrate that EditSum outperforms the state-of-the-art approaches by a substantial margin. The human evaluation also proves the summaries generated by EditSum are more informative and useful. We also verify that EditSum performs well on predicting the patternized words and keywords.

{{</citation>}}


### (28/52) ZC3: Zero-Shot Cross-Language Code Clone Detection (Jia Li et al., 2023)

{{<citation>}}

Jia Li, Chongyang Tao, Zhi Jin, Fang Liu, Jia Allen Li, Ge Li. (2023)  
**ZC3: Zero-Shot Cross-Language Code Clone Detection**  

---
Primary Category: cs.SE  
Categories: cs-CL, cs-IR, cs-SE, cs.SE  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2308.13754v1)  

---


**ABSTRACT**  
Developers introduce code clones to improve programming productivity. Many existing studies have achieved impressive performance in monolingual code clone detection. However, during software development, more and more developers write semantically equivalent programs with different languages to support different platforms and help developers translate projects from one language to another. Considering that collecting cross-language parallel data, especially for low-resource languages, is expensive and time-consuming, how designing an effective cross-language model that does not rely on any parallel data is a significant problem. In this paper, we propose a novel method named ZC3 for Zero-shot Cross-language Code Clone detection. ZC3 designs the contrastive snippet prediction to form an isomorphic representation space among different programming languages. Based on this, ZC3 exploits domain-aware learning and cycle consistency learning to further constrain the model to generate representations that are aligned among different languages meanwhile are diacritical for different types of clones. To evaluate our approach, we conduct extensive experiments on four representative cross-language clone detection datasets. Experimental results show that ZC3 outperforms the state-of-the-art baselines by 67.12%, 51.39%, 14.85%, and 53.01% on the MAP score, respectively. We further investigate the representational distribution of different languages and discuss the effectiveness of our method.

{{</citation>}}


## cs.CL (7)



### (29/52) Translate Meanings, Not Just Words: IdiomKB's Role in Optimizing Idiomatic Translation with Language Models (Shuang Li et al., 2023)

{{<citation>}}

Shuang Li, Jiangjie Chen, Siyu Yuan, Xinyi Wu, Hao Yang, Shimin Tao, Yanghua Xiao. (2023)  
**Translate Meanings, Not Just Words: IdiomKB's Role in Optimizing Idiomatic Translation with Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BLOOM, GPT, GPT-4, Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2308.13961v1)  

---


**ABSTRACT**  
To translate well, machine translation (MT) systems and general-purposed language models (LMs) need a deep understanding of both source and target languages and cultures. Therefore, idioms, with their non-compositional nature, pose particular challenges for Transformer-based systems, as literal translations often miss the intended meaning. Traditional methods, which replace idioms using existing knowledge bases (KBs), often lack scale and context awareness. Addressing these challenges, our approach prioritizes context awareness and scalability, allowing for offline storage of idioms in a manageable KB size. This ensures efficient serving with smaller models and provides a more comprehensive understanding of idiomatic expressions. We introduce a multilingual idiom KB (IdiomKB) developed using large LMs to address this. This KB facilitates better translation by smaller models, such as BLOOMZ (7.1B), Alpaca (7B), and InstructGPT (6.7B), by retrieving idioms' figurative meanings. We present a novel, GPT-4-powered metric for human-aligned evaluation, demonstrating that IdiomKB considerably boosts model performance. Human evaluations further validate our KB's quality.

{{</citation>}}


### (30/52) Improving Knowledge Distillation for BERT Models: Loss Functions, Mapping Methods, and Weight Tuning (Apoorv Dankar et al., 2023)

{{<citation>}}

Apoorv Dankar, Adeem Jassani, Kartikaeya Kumar. (2023)  
**Improving Knowledge Distillation for BERT Models: Loss Functions, Mapping Methods, and Weight Tuning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, GLUE, GPT, Knowledge Distillation, T5  
[Paper Link](http://arxiv.org/abs/2308.13958v1)  

---


**ABSTRACT**  
The use of large transformer-based models such as BERT, GPT, and T5 has led to significant advancements in natural language processing. However, these models are computationally expensive, necessitating model compression techniques that reduce their size and complexity while maintaining accuracy. This project investigates and applies knowledge distillation for BERT model compression, specifically focusing on the TinyBERT student model. We explore various techniques to improve knowledge distillation, including experimentation with loss functions, transformer layer mapping methods, and tuning the weights of attention and representation loss and evaluate our proposed techniques on a selection of downstream tasks from the GLUE benchmark. The goal of this work is to improve the efficiency and effectiveness of knowledge distillation, enabling the development of more efficient and accurate models for a range of natural language processing tasks.

{{</citation>}}


### (31/52) Exploring Large Language Models for Knowledge Graph Completion (Liang Yao et al., 2023)

{{<citation>}}

Liang Yao, Jiazhen Peng, Chengsheng Mao, Yuan Luo. (2023)  
**Exploring Large Language Models for Knowledge Graph Completion**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GLM, GPT, GPT-4, Knowledge Graph, LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2308.13916v2)  

---


**ABSTRACT**  
Knowledge graphs play a vital role in numerous artificial intelligence tasks, yet they frequently face the issue of incompleteness. In this study, we explore utilizing Large Language Models (LLM) for knowledge graph completion. We consider triples in knowledge graphs as text sequences and introduce an innovative framework called Knowledge Graph LLM (KG-LLM) to model these triples. Our technique employs entity and relation descriptions of a triple as prompts and utilizes the response for predictions. Experiments on various benchmark knowledge graphs demonstrate that our method attains state-of-the-art performance in tasks such as triple classification and relation prediction. We also find that fine-tuning relatively smaller models (e.g., LLaMA-7B, ChatGLM-6B) outperforms recent ChatGPT and GPT-4.

{{</citation>}}


### (32/52) LMSanitator: Defending Prompt-Tuning Against Task-Agnostic Backdoors (Chengkun Wei et al., 2023)

{{<citation>}}

Chengkun Wei, Wenlong Meng, Zhikun Zhang, Min Chen, Minghu Zhao, Wenjing Fang, Lei Wang, Zihui Zhang, Wenzhi Chen. (2023)  
**LMSanitator: Defending Prompt-Tuning Against Task-Agnostic Backdoors**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CR, cs-LG, cs.CL  
Keywords: NLP, Transformer  
[Paper Link](http://arxiv.org/abs/2308.13904v1)  

---


**ABSTRACT**  
Prompt-tuning has emerged as an attractive paradigm for deploying large-scale language models due to its strong downstream task performance and efficient multitask serving ability. Despite its wide adoption, we empirically show that prompt-tuning is vulnerable to downstream task-agnostic backdoors, which reside in the pretrained models and can affect arbitrary downstream tasks. The state-of-the-art backdoor detection approaches cannot defend against task-agnostic backdoors since they hardly converge in reversing the backdoor triggers. To address this issue, we propose LMSanitator, a novel approach for detecting and removing task-agnostic backdoors on Transformer models. Instead of directly inversing the triggers, LMSanitator aims to inverse the predefined attack vectors (pretrained models' output when the input is embedded with triggers) of the task-agnostic backdoors, which achieves much better convergence performance and backdoor detection accuracy. LMSanitator further leverages prompt-tuning's property of freezing the pretrained model to perform accurate and fast output monitoring and input purging during the inference phase. Extensive experiments on multiple language models and NLP tasks illustrate the effectiveness of LMSanitator. For instance, LMSanitator achieves 92.8% backdoor detection accuracy on 960 models and decreases the attack success rate to less than 1% in most scenarios.

{{</citation>}}


### (33/52) Solving Math Word Problem with Problem Type Classification (Jie Yao et al., 2023)

{{<citation>}}

Jie Yao, Zihao Zhou, Qiufeng Wang. (2023)  
**Solving Math Word Problem with Problem Type Classification**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2308.13844v1)  

---


**ABSTRACT**  
Math word problems (MWPs) require analyzing text descriptions and generating mathematical equations to derive solutions. Existing works focus on solving MWPs with two types of solvers: tree-based solver and large language model (LLM) solver. However, these approaches always solve MWPs by a single solver, which will bring the following problems: (1) Single type of solver is hard to solve all types of MWPs well. (2) A single solver will result in poor performance due to over-fitting. To address these challenges, this paper utilizes multiple ensemble approaches to improve MWP-solving ability. Firstly, We propose a problem type classifier that combines the strengths of the tree-based solver and the LLM solver. This ensemble approach leverages their respective advantages and broadens the range of MWPs that can be solved. Furthermore, we also apply ensemble techniques to both tree-based solver and LLM solver to improve their performance. For the tree-based solver, we propose an ensemble learning framework based on ten-fold cross-validation and voting mechanism. In the LLM solver, we adopt self-consistency (SC) method to improve answer selection. Experimental results demonstrate the effectiveness of these ensemble approaches in enhancing MWP-solving ability. The comprehensive evaluation showcases improved performance, validating the advantages of our proposed approach. Our code is available at this url: https://github.com/zhouzihao501/NLPCC2023-Shared-Task3-ChineseMWP.

{{</citation>}}


### (34/52) Planning with Logical Graph-based Language Model for Instruction Generation (Fan Zhang et al., 2023)

{{<citation>}}

Fan Zhang, Kebing Jin, Hankz Hankui Zhuo. (2023)  
**Planning with Logical Graph-based Language Model for Instruction Generation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GLM, Language Model  
[Paper Link](http://arxiv.org/abs/2308.13782v1)  

---


**ABSTRACT**  
Despite the superior performance of large language models to generate natural language texts, it is hard to generate texts with correct logic according to a given task, due to the difficulties for neural models to capture implied rules from free-form texts. In this paper, we propose a novel graph-based language model, Logical-GLM, to infuse logic into language models for more valid text generation and interpretability. Specifically, we first capture information from natural language instructions and construct logical bayes graphs that generally describe domains. Next, we generate logical skeletons to guide language model training, infusing domain knowledge into language models. Finally, we alternately optimize the searching policy of graphs and language models until convergence. The experimental results show that Logical-GLM is both effective and efficient compared with traditional language models, despite using smaller-scale training data and fewer parameters. Our approach can generate instructional texts with more correct logic owing to the internalized domain knowledge. Moreover, the usage of logical graphs reflects the inner mechanism of the language models, which improves the interpretability of black-box models.

{{</citation>}}


### (35/52) Adversarial Fine-Tuning of Language Models: An Iterative Optimisation Approach for the Generation and Detection of Problematic Content (Charles O'Neill et al., 2023)

{{<citation>}}

Charles O'Neill, Jack Miller, Ioana Ciuca, Yuan-Sen Ting, Thang Bui. (2023)  
**Adversarial Fine-Tuning of Language Models: An Iterative Optimisation Approach for the Generation and Detection of Problematic Content**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2308.13768v1)  

---


**ABSTRACT**  
In this paper, we tackle the emerging challenge of unintended harmful content generation in Large Language Models (LLMs) with a novel dual-stage optimisation technique using adversarial fine-tuning. Our two-pronged approach employs an adversarial model, fine-tuned to generate potentially harmful prompts, and a judge model, iteratively optimised to discern these prompts. In this adversarial cycle, the two models seek to outperform each other in the prompting phase, generating a dataset of rich examples which are then used for fine-tuning. This iterative application of prompting and fine-tuning allows continuous refinement and improved performance. The performance of our approach is evaluated through classification accuracy on a dataset consisting of problematic prompts not detected by GPT-4, as well as a selection of contentious but unproblematic prompts. We show considerable increase in classification accuracy of the judge model on this challenging dataset as it undergoes the optimisation process. Furthermore, we show that a rudimentary model \texttt{ada} can achieve 13\% higher accuracy on the hold-out test set than GPT-4 after only a few rounds of this process, and that this fine-tuning improves performance in parallel tasks such as toxic comment identification.

{{</citation>}}


## cs.SI (1)



### (36/52) Quantifying the Influence of User Behaviors on the Dissemination of Fake News on Twitter with Multivariate Hawkes Processes (Yichen Jiang et al., 2023)

{{<citation>}}

Yichen Jiang, Michael D. Porter. (2023)  
**Quantifying the Influence of User Behaviors on the Dissemination of Fake News on Twitter with Multivariate Hawkes Processes**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI, stat-AP  
Keywords: Fake News, Social Network, Twitter  
[Paper Link](http://arxiv.org/abs/2308.13927v1)  

---


**ABSTRACT**  
Fake news has emerged as a pervasive problem within Online Social Networks, leading to a surge of research interest in this area. Understanding the dissemination mechanisms of fake news is crucial in comprehending the propagation of disinformation/misinformation and its impact on users in Online Social Networks. This knowledge can facilitate the development of interventions to curtail the spread of false information and inform affected users to remain vigilant against fraudulent/malicious content. In this paper, we specifically target the Twitter platform and propose a Multivariate Hawkes Point Processes model that incorporates essential factors such as user networks, response tweet types, and user stances as model parameters. Our objective is to investigate and quantify their influence on the dissemination process of fake news. We derive parameter estimation expressions using an Expectation Maximization algorithm and validate them on a simulated dataset. Furthermore, we conduct a case study using a real dataset of fake news collected from Twitter to explore the impact of user stances and tweet types on dissemination patterns. This analysis provides valuable insights into how users are influenced by or influence the dissemination process of disinformation/misinformation, and demonstrates how our model can aid in intervening in this process.

{{</citation>}}


## cs.AR (1)



### (37/52) An Efficient FPGA-Based Accelerator for Swin Transformer (Zhiyang Liu et al., 2023)

{{<citation>}}

Zhiyang Liu, Pengyu Yin, Zhenhua Ren. (2023)  
**An Efficient FPGA-Based Accelerator for Swin Transformer**  

---
Primary Category: cs.AR  
Categories: cs-AR, cs.AR  
Keywords: ImageNet, Transformer  
[Paper Link](http://arxiv.org/abs/2308.13922v1)  

---


**ABSTRACT**  
Since introduced, Swin Transformer has achieved remarkable results in the field of computer vision, it has sparked the need for dedicated hardware accelerators, specifically catering to edge computing demands. For the advantages of flexibility, low power consumption, FPGAs have been widely employed to accelerate the inference of convolutional neural networks (CNNs) and show potential in Transformer-based models. Unlike CNNs, which mainly involve multiply and accumulate (MAC) operations, Transformer involve non-linear computations such as Layer Normalization (LN), Softmax, and GELU. These nonlinear computations do pose challenges for accelerator design. In this paper, to propose an efficient FPGA-based hardware accelerator for Swin Transformer, we focused on using different strategies to deal with these nonlinear calculations and efficiently handling MAC computations to achieve the best acceleration results. We replaced LN with BN, Given that Batch Normalization (BN) can be fused with linear layers during inference to optimize inference efficiency. The modified Swin-T, Swin-S, and Swin-B respectively achieved Top-1 accuracy rates of 80.7%, 82.7%, and 82.8% in ImageNet. Furthermore, We employed strategies for approximate computation to design hardware-friendly architectures for Softmax and GELU computations. We also designed an efficient Matrix Multiplication Unit to handle all linear computations in Swin Transformer. As a conclude, compared with CPU (AMD Ryzen 5700X), our accelerator achieved 1.76x, 1.66x, and 1.25x speedup and achieved 20.45x, 18.60x, and 14.63x energy efficiency (FPS/power consumption) improvement on Swin-T, Swin-S, and Swin-B models, respectively. Compared to GPU (Nvidia RTX 2080 Ti), we achieved 5.05x, 4.42x, and 3.00x energy efficiency improvement respectively. As far as we know, the accelerator we proposed is the fastest FPGA-based accelerator for Swin Transformer.

{{</citation>}}


## cs.AI (6)



### (38/52) A Wide Evaluation of ChatGPT on Affective Computing Tasks (Mostafa M. Amin et al., 2023)

{{<citation>}}

Mostafa M. Amin, Rui Mao, Erik Cambria, Björn W. Schuller. (2023)  
**A Wide Evaluation of ChatGPT on Affective Computing Tasks**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: ChatGPT, GPT, GPT-3.5, GPT-4, NLP  
[Paper Link](http://arxiv.org/abs/2308.13911v1)  

---


**ABSTRACT**  
With the rise of foundation models, a new artificial intelligence paradigm has emerged, by simply using general purpose foundation models with prompting to solve problems instead of training a separate machine learning model for each problem. Such models have been shown to have emergent properties of solving problems that they were not initially trained on. The studies for the effectiveness of such models are still quite limited. In this work, we widely study the capabilities of the ChatGPT models, namely GPT-4 and GPT-3.5, on 13 affective computing problems, namely aspect extraction, aspect polarity classification, opinion extraction, sentiment analysis, sentiment intensity ranking, emotions intensity ranking, suicide tendency detection, toxicity detection, well-being assessment, engagement measurement, personality assessment, sarcasm detection, and subjectivity detection. We introduce a framework to evaluate the ChatGPT models on regression-based problems, such as intensity ranking problems, by modelling them as pairwise ranking classification. We compare ChatGPT against more traditional NLP methods, such as end-to-end recurrent neural networks and transformers. The results demonstrate the emergent abilities of the ChatGPT models on a wide range of affective computing problems, where GPT-3.5 and especially GPT-4 have shown strong performance on many problems, particularly the ones related to sentiment, emotions, or toxicity. The ChatGPT models fell short for problems with implicit signals, such as engagement measurement and subjectivity detection.

{{</citation>}}


### (39/52) Federated Fine-tuning of Billion-Sized Language Models across Mobile Devices (Mengwei Xu et al., 2023)

{{<citation>}}

Mengwei Xu, Yaozong Wu, Dongqi Cai, Xiang Li, Shangguang Wang. (2023)  
**Federated Fine-tuning of Billion-Sized Language Models across Mobile Devices**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: LLaMA, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2308.13894v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) are transforming the landscape of mobile intelligence. Federated Learning (FL), a method to preserve user data privacy, is often employed in fine-tuning LLMs to downstream mobile tasks, an approach known as FedLLM. Though recent efforts have addressed the network issue induced by the vast model size, they have not practically mitigated vital challenges concerning integration with mobile devices, such as significant memory consumption and sluggish model convergence.   In response to these challenges, this work introduces FwdLLM, an innovative FL protocol designed to enhance the FedLLM efficiency. The key idea of FwdLLM to employ backpropagation (BP)-free training methods, requiring devices only to execute ``perturbed inferences''. Consequently, FwdLLM delivers way better memory efficiency and time efficiency (expedited by mobile NPUs and an expanded array of participant devices). FwdLLM centers around three key designs: (1) it combines BP-free training with parameter-efficient training methods, an essential way to scale the approach to the LLM era; (2) it systematically and adaptively allocates computational loads across devices, striking a careful balance between convergence speed and accuracy; (3) it discriminatively samples perturbed predictions that are more valuable to model convergence. Comprehensive experiments with five LLMs and three NLP tasks illustrate FwdLLM's significant advantages over conventional methods, including up to three orders of magnitude faster convergence and a 14.6x reduction in memory footprint. Uniquely, FwdLLM paves the way for federated learning of billion-parameter LLMs such as LLaMA on COTS mobile devices -- a feat previously unattained.

{{</citation>}}


### (40/52) Graph Edit Distance Learning via Different Attention (Jiaxi Lv et al., 2023)

{{<citation>}}

Jiaxi Lv, Liang Zhang, Yi Huang, Jiancheng Huang, Shifeng Chen. (2023)  
**Graph Edit Distance Learning via Different Attention**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Attention, GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2308.13871v1)  

---


**ABSTRACT**  
Recently, more and more research has focused on using Graph Neural Networks (GNN) to solve the Graph Similarity Computation problem (GSC), i.e., computing the Graph Edit Distance (GED) between two graphs. These methods treat GSC as an end-to-end learnable task, and the core of their architecture is the feature fusion modules to interact with the features of two graphs. Existing methods consider that graph-level embedding is difficult to capture the differences in local small structures between two graphs, and thus perform fine-grained feature fusion on node-level embedding can improve the accuracy, but leads to greater time and memory consumption in the training and inference phases. However, this paper proposes a novel graph-level fusion module Different Attention (DiffAtt), and demonstrates that graph-level fusion embeddings can substantially outperform these complex node-level fusion embeddings. We posit that the relative difference structure of the two graphs plays an important role in calculating their GED values. To this end, DiffAtt uses the difference between two graph-level embeddings as an attentional mechanism to capture the graph structural difference of the two graphs. Based on DiffAtt, a new GSC method, named Graph Edit Distance Learning via Different Attention (REDRAFT), is proposed, and experimental results demonstrate that REDRAFT achieves state-of-the-art performance in 23 out of 25 metrics in five benchmark datasets. Especially on MSE, it respectively outperforms the second best by 19.9%, 48.8%, 29.1%, 31.6%, and 2.2%. Moreover, we propose a quantitative test Remaining Subgraph Alignment Test (RESAT) to verify that among all graph-level fusion modules, the fusion embedding generated by DiffAtt can best capture the structural differences between two graphs.

{{</citation>}}


### (41/52) Empowering Dynamics-aware Text-to-Video Diffusion with Large Language Models (Hao Fei et al., 2023)

{{<citation>}}

Hao Fei, Shengqiong Wu, Wei Ji, Hanwang Zhang, Tat-Seng Chua. (2023)  
**Empowering Dynamics-aware Text-to-Video Diffusion with Large Language Models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CV, cs.AI  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2308.13812v1)  

---


**ABSTRACT**  
Text-to-video (T2V) synthesis has gained increasing attention in the community, in which the recently emerged diffusion models (DMs) have promisingly shown stronger performance than the past approaches. While existing state-of-the-art DMs are competent to achieve high-resolution video generation, they may largely suffer from key limitations (e.g., action occurrence disorders, crude video motions) with respect to the intricate temporal dynamics modeling, one of the crux of video synthesis. In this work, we investigate strengthening the awareness of video dynamics for DMs, for high-quality T2V generation. Inspired by human intuition, we design an innovative dynamic scene manager (dubbed as Dysen) module, which includes (step-1) extracting from input text the key actions with proper time-order arrangement, (step-2) transforming the action schedules into the dynamic scene graph (DSG) representations, and (step-3) enriching the scenes in the DSG with sufficient and reasonable details. Taking advantage of the existing powerful LLMs (e.g., ChatGPT) via in-context learning, Dysen realizes (nearly) human-level temporal dynamics understanding. Finally, the resulting video DSG with rich action scene details is encoded as fine-grained spatio-temporal features, integrated into the backbone T2V DM for video generating. Experiments on popular T2V datasets suggest that our framework consistently outperforms prior arts with significant margins, especially in the scenario with complex actions. Project page at https://haofei.vip/Dysen-VDM

{{</citation>}}


### (42/52) Reinforcement Learning Based Multi-modal Feature Fusion Network for Novel Class Discovery (Qiang Li et al., 2023)

{{<citation>}}

Qiang Li, Qiuyang Ma, Weizhi Nie, Anan Liu. (2023)  
**Reinforcement Learning Based Multi-modal Feature Fusion Network for Novel Class Discovery**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-MM, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.13801v1)  

---


**ABSTRACT**  
With the development of deep learning techniques, supervised learning has achieved performances surpassing those of humans. Researchers have designed numerous corresponding models for different data modalities, achieving excellent results in supervised tasks. However, with the exponential increase of data in multiple fields, the recognition and classification of unlabeled data have gradually become a hot topic. In this paper, we employed a Reinforcement Learning framework to simulate the cognitive processes of humans for effectively addressing novel class discovery in the Open-set domain. We deployed a Member-to-Leader Multi-Agent framework to extract and fuse features from multi-modal information, aiming to acquire a more comprehensive understanding of the feature space. Furthermore, this approach facilitated the incorporation of self-supervised learning to enhance model training. We employed a clustering method with varying constraint conditions, ranging from strict to loose, allowing for the generation of dependable labels for a subset of unlabeled data during the training phase. This iterative process is similar to human exploratory learning of unknown data. These mechanisms collectively update the network parameters based on rewards received from environmental feedback. This process enables effective control over the extent of exploration learning, ensuring the accuracy of learning in unknown data categories. We demonstrate the performance of our approach in both the 3D and 2D domains by employing the OS-MN40, OS-MN40-Miss, and Cifar10 datasets. Our approach achieves competitive competitive results.

{{</citation>}}


### (43/52) i-Align: an interpretable knowledge graph alignment model (Bayu Distiawan Trisedya et al., 2023)

{{<citation>}}

Bayu Distiawan Trisedya, Flora D Salim, Jeffrey Chan, Damiano Spina, Falk Scholer, Mark Sanderson. (2023)  
**i-Align: an interpretable knowledge graph alignment model**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2308.13755v1)  

---


**ABSTRACT**  
Knowledge graphs (KGs) are becoming essential resources for many downstream applications. However, their incompleteness may limit their potential. Thus, continuous curation is needed to mitigate this problem. One of the strategies to address this problem is KG alignment, i.e., forming a more complete KG by merging two or more KGs. This paper proposes i-Align, an interpretable KG alignment model. Unlike the existing KG alignment models, i-Align provides an explanation for each alignment prediction while maintaining high alignment performance. Experts can use the explanation to check the correctness of the alignment prediction. Thus, the high quality of a KG can be maintained during the curation process (e.g., the merging process of two KGs). To this end, a novel Transformer-based Graph Encoder (Trans-GE) is proposed as a key component of i-Align for aggregating information from entities' neighbors (structures). Trans-GE uses Edge-gated Attention that combines the adjacency matrix and the self-attention matrix to learn a gating mechanism to control the information aggregation from the neighboring entities. It also uses historical embeddings, allowing Trans-GE to be trained over mini-batches, or smaller sub-graphs, to address the scalability issue when encoding a large KG. Another component of i-Align is a Transformer encoder for aggregating entities' attributes. This way, i-Align can generate explanations in the form of a set of the most influential attributes/neighbors based on attention weights. Extensive experiments are conducted to show the power of i-Align. The experiments include several aspects, such as the model's effectiveness for aligning KGs, the quality of the generated explanations, and its practicality for aligning large KGs. The results show the effectiveness of i-Align in these aspects.

{{</citation>}}


## eess.SP (2)



### (44/52) A Two-Dimensional Deep Network for RF-based Drone Detection and Identification Towards Secure Coverage Extension (Zixiao Zhao et al., 2023)

{{<citation>}}

Zixiao Zhao, Qinghe Du, Xiang Yao, Lei Lu, Shijiao Zhang. (2023)  
**A Two-Dimensional Deep Network for RF-based Drone Detection and Identification Towards Secure Coverage Extension**  

---
Primary Category: eess.SP  
Categories: cs-LG, eess-SP, eess.SP  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2308.13906v1)  

---


**ABSTRACT**  
As drones become increasingly prevalent in human life, they also raises security concerns such as unauthorized access and control, as well as collisions and interference with manned aircraft. Therefore, ensuring the ability to accurately detect and identify between different drones holds significant implications for coverage extension. Assisted by machine learning, radio frequency (RF) detection can recognize the type and flight mode of drones based on the sampled drone signals. In this paper, we first utilize Short-Time Fourier. Transform (STFT) to extract two-dimensional features from the raw signals, which contain both time-domain and frequency-domain information. Then, we employ a Convolutional Neural Network (CNN) built with ResNet structure to achieve multi-class classifications. Our experimental results show that the proposed ResNet-STFT can achieve higher accuracy and faster convergence on the extended dataset. Additionally, it exhibits balanced performance compared to other baselines on the raw dataset.

{{</citation>}}


### (45/52) Self-Supervised Scalable Deep Compressed Sensing (Bin Chen et al., 2023)

{{<citation>}}

Bin Chen, Xuanyu Zhang, Shuai Liu, Yongbing Zhang, Jian Zhang. (2023)  
**Self-Supervised Scalable Deep Compressed Sensing**  

---
Primary Category: eess.SP  
Categories: cs-CV, cs-LG, eess-SP, eess.SP  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2308.13777v1)  

---


**ABSTRACT**  
Compressed sensing (CS) is a promising tool for reducing sampling costs. Current deep neural network (NN)-based CS methods face challenges in collecting labeled measurement-ground truth (GT) data and generalizing to real applications. This paper proposes a novel $\mathbf{S}$elf-supervised s$\mathbf{C}$alable deep CS method, comprising a $\mathbf{L}$earning scheme called $\mathbf{SCL}$ and a family of $\mathbf{Net}$works named $\mathbf{SCNet}$, which does not require GT and can handle arbitrary sampling ratios and matrices once trained on a partial measurement set. Our SCL contains a dual-domain loss and a four-stage recovery strategy. The former encourages a cross-consistency on two measurement parts and a sampling-reconstruction cycle-consistency regarding arbitrary ratios and matrices to maximize data/information utilization. The latter can progressively leverage common signal prior in external measurements and internal characteristics of test samples and learned NNs to improve accuracy. SCNet combines the explicit guidance from optimization algorithms with implicit regularization from advanced NN blocks to learn a collaborative signal representation. Our theoretical analyses and experiments on simulated and real captured data, covering 1-/2-/3-D natural and scientific signals, demonstrate the effectiveness, superior performance, flexibility, and generalization ability of our method over existing self-supervised methods and its significant potential in competing against state-of-the-art supervised methods.

{{</citation>}}


## eess.IV (2)



### (46/52) ReFuSeg: Regularized Multi-Modal Fusion for Precise Brain Tumour Segmentation (Aditya Kasliwal et al., 2023)

{{<citation>}}

Aditya Kasliwal, Sankarshanaa Sagaram, Laven Srivastava, Pratinav Seth, Adil Khan. (2023)  
**ReFuSeg: Regularized Multi-Modal Fusion for Precise Brain Tumour Segmentation**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.13883v1)  

---


**ABSTRACT**  
Semantic segmentation of brain tumours is a fundamental task in medical image analysis that can help clinicians in diagnosing the patient and tracking the progression of any malignant entities. Accurate segmentation of brain lesions is essential for medical diagnosis and treatment planning. However, failure to acquire specific MRI imaging modalities can prevent applications from operating in critical situations, raising concerns about their reliability and overall trustworthiness. This paper presents a novel multi-modal approach for brain lesion segmentation that leverages information from four distinct imaging modalities while being robust to real-world scenarios of missing modalities, such as T1, T1c, T2, and FLAIR MRI of brains. Our proposed method can help address the challenges posed by artifacts in medical imagery due to data acquisition errors (such as patient motion) or a reconstruction algorithm's inability to represent the anatomy while ensuring a trade-off in accuracy. Our proposed regularization module makes it robust to these scenarios and ensures the reliability of lesion segmentation.

{{</citation>}}


### (47/52) Bias in Unsupervised Anomaly Detection in Brain MRI (Cosmin I. Bercea et al., 2023)

{{<citation>}}

Cosmin I. Bercea, Esther Puyol-Antón, Benedikt Wiestler, Daniel Rueckert, Julia A. Schnabel, Andrew P. King. (2023)  
**Bias in Unsupervised Anomaly Detection in Brain MRI**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Anomaly Detection, Bias  
[Paper Link](http://arxiv.org/abs/2308.13861v1)  

---


**ABSTRACT**  
Unsupervised anomaly detection methods offer a promising and flexible alternative to supervised approaches, holding the potential to revolutionize medical scan analysis and enhance diagnostic performance.   In the current landscape, it is commonly assumed that differences between a test case and the training distribution are attributed solely to pathological conditions, implying that any disparity indicates an anomaly. However, the presence of other potential sources of distributional shift, including scanner, age, sex, or race, is frequently overlooked. These shifts can significantly impact the accuracy of the anomaly detection task. Prominent instances of such failures have sparked concerns regarding the bias, credibility, and fairness of anomaly detection.   This work presents a novel analysis of biases in unsupervised anomaly detection. By examining potential non-pathological distributional shifts between the training and testing distributions, we shed light on the extent of these biases and their influence on anomaly detection results. Moreover, this study examines the algorithmic limitations that arise due to biases, providing valuable insights into the challenges encountered by anomaly detection algorithms in accurately learning and capturing the entire range of variability present in the normative distribution. Through this analysis, we aim to enhance the understanding of these biases and pave the way for future improvements in the field. Here, we specifically investigate Alzheimer's disease detection from brain MR imaging as a case study, revealing significant biases related to sex, race, and scanner variations that substantially impact the results. These findings align with the broader goal of improving the reliability, fairness, and effectiveness of anomaly detection in medical imaging.

{{</citation>}}


## cs.HC (2)



### (48/52) Digital Twin conceptual framework for the O&M process of cubature building objects (Andrzej Szymon Borkowski, 2023)

{{<citation>}}

Andrzej Szymon Borkowski. (2023)  
**Digital Twin conceptual framework for the O&M process of cubature building objects**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.13873v1)  

---


**ABSTRACT**  
The broader construction industry is struggling with data loss, inefficient processes and low productivity in asset management. The remedy to these problems seems to be the idea of Digital Twin (DT). However, the frameworks proposed so far do not always support a solution to these problems. This paper conducts an extensive literature review to develop a conceptual framework for the Operation and Maintenance (O&M) phase for cubic facilities. The conceptual framework takes into account the increasingly popular Internet of Things (IoT) and Artificial Intelligence (AI) technologies. The presented framework, after appropriate modifications, can also be applied to infrastructure facilities or city fragments. The paper presents limitations and directions for further research. The DT paradigm has been adopted and its adoption is ongoing. Its implementation will progress in the coming years.

{{</citation>}}


### (49/52) Cura: Curation at Social Media Scale (Wanrong He et al., 2023)

{{<citation>}}

Wanrong He, Mitchell L. Gordon, Lindsay Popowski, Michael S. Bernstein. (2023)  
**Cura: Curation at Social Media Scale**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: Social Media  
[Paper Link](http://arxiv.org/abs/2308.13841v1)  

---


**ABSTRACT**  
How can online communities execute a focused vision for their space? Curation offers one approach, where community leaders manually select content to share with the community. Curation enables leaders to shape a space that matches their taste, norms, and values, but the practice is often intractable at social media scale: curators cannot realistically sift through hundreds or thousands of submissions daily. In this paper, we contribute algorithmic and interface foundations enabling curation at scale, and manifest these foundations in a system called Cura. Our approach draws on the observation that, while curators' attention is limited, other community members' upvotes are plentiful and informative of curators' likely opinions. We thus contribute a transformer-based curation model that predicts whether each curator will upvote a post based on previous community upvotes. Cura applies this curation model to create a feed of content that it predicts the curator would want in the community. Evaluations demonstrate that the curation model accurately estimates opinions of diverse curators, that changing curators for a community results in clearly recognizable shifts in the community's content, and that, consequently, curation can reduce anti-social behavior by half without extra moderation effort. By sampling different types of curators, Cura lowers the threshold to genres of curated social media ranging from editorial groups to stakeholder roundtables to democracies.

{{</citation>}}


## cs.DC (1)



### (50/52) Throughput Maximization of DNN Inference: Batching or Multi-Tenancy? (Seyed Morteza Nabavinejad et al., 2023)

{{<citation>}}

Seyed Morteza Nabavinejad, Masoumeh Ebrahimi, Sherief Reda. (2023)  
**Throughput Maximization of DNN Inference: Batching or Multi-Tenancy?**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.13803v1)  

---


**ABSTRACT**  
Deployment of real-time ML services on warehouse-scale infrastructures is on the increase. Therefore, decreasing latency and increasing throughput of deep neural network (DNN) inference applications that empower those services have attracted attention from both academia and industry. A common solution to address this challenge is leveraging hardware accelerators such as GPUs. To improve the inference throughput of DNNs deployed on GPU accelerators, two common approaches are employed: Batching and Multi-Tenancy. Our preliminary experiments show that the effect of these approaches on the throughput depends on the DNN architecture. Taking this observation into account, we design and implement DNNScaler which aims to maximize the throughput of interactive AI-powered services while meeting their latency requirements. DNNScaler first detects the suitable approach (Batching or Multi-Tenancy) that would be most beneficial for a DNN regarding throughput improvement. Then, it adjusts the control knob of the detected approach (batch size for Batching and number of co-located instances for Multi-Tenancy) to maintain the latency while increasing the throughput. Conducting an extensive set of experiments using well-known DNNs from a variety of domains, several popular datasets, and a cutting-edge GPU, the results indicate that DNNScaler can improve the throughput by up to 14x (218% on average) compared with the previously proposed approach, while meeting the latency requirements of the services.

{{</citation>}}


## cs.SD (1)



### (51/52) A Comprehensive Survey for Evaluation Methodologies of AI-Generated Music (Zeyu Xiong et al., 2023)

{{<citation>}}

Zeyu Xiong, Weitao Wang, Jing Yu, Yue Lin, Ziyan Wang. (2023)  
**A Comprehensive Survey for Evaluation Methodologies of AI-Generated Music**  

---
Primary Category: cs.SD  
Categories: cs-AI, cs-HC, cs-SD, cs.SD, eess-AS  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.13736v1)  

---


**ABSTRACT**  
In recent years, AI-generated music has made significant progress, with several models performing well in multimodal and complex musical genres and scenes. While objective metrics can be used to evaluate generative music, they often lack interpretability for musical evaluation. Therefore, researchers often resort to subjective user studies to assess the quality of the generated works, which can be resource-intensive and less reproducible than objective metrics. This study aims to comprehensively evaluate the subjective, objective, and combined methodologies for assessing AI-generated music, highlighting the advantages and disadvantages of each approach. Ultimately, this study provides a valuable reference for unifying generative AI in the field of music evaluation.

{{</citation>}}


## cs.RO (1)



### (52/52) ISR-LLM: Iterative Self-Refined Large Language Model for Long-Horizon Sequential Task Planning (Zhehua Zhou et al., 2023)

{{<citation>}}

Zhehua Zhou, Jiayang Song, Kunpeng Yao, Zhan Shu, Lei Ma. (2023)  
**ISR-LLM: Iterative Self-Refined Large Language Model for Long-Horizon Sequential Task Planning**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.13724v1)  

---


**ABSTRACT**  
Motivated by the substantial achievements observed in Large Language Models (LLMs) in the field of natural language processing, recent research has commenced investigations into the application of LLMs for complex, long-horizon sequential task planning challenges in robotics. LLMs are advantageous in offering the potential to enhance the generalizability as task-agnostic planners and facilitate flexible interaction between human instructors and planning systems. However, task plans generated by LLMs often lack feasibility and correctness. To address this challenge, we introduce ISR-LLM, a novel framework that improves LLM-based planning through an iterative self-refinement process. The framework operates through three sequential steps: preprocessing, planning, and iterative self-refinement. During preprocessing, an LLM translator is employed to convert natural language input into a Planning Domain Definition Language (PDDL) formulation. In the planning phase, an LLM planner formulates an initial plan, which is then assessed and refined in the iterative self-refinement step by using a validator. We examine the performance of ISR-LLM across three distinct planning domains. The results show that ISR-LLM is able to achieve markedly higher success rates in task accomplishments compared to state-of-the-art LLM-based planners. Moreover, it also preserves the broad applicability and generalizability of working with natural language instructions.

{{</citation>}}
