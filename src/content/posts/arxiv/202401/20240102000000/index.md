---
draft: false
title: "arXiv @ 2024.01.02"
date: 2024-01-02
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2024"]
menu:
  sidebar:
    name: "arXiv @ 2024.01.02"
    identifier: arxiv_20240102
    parent: 202401_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CV (12)](#cscv-12)
- [cs.LG (9)](#cslg-9)
- [cs.AI (2)](#csai-2)
- [eess.IV (1)](#eessiv-1)
- [cs.CL (9)](#cscl-9)
- [cs.CR (3)](#cscr-3)
- [cs.SI (2)](#cssi-2)
- [physics.chem-ph (1)](#physicschem-ph-1)
- [cs.SD (2)](#cssd-2)
- [cs.SE (1)](#csse-1)
- [cs.NI (1)](#csni-1)

## cs.CV (12)



### (1/43) WoodScape Motion Segmentation for Autonomous Driving -- CVPR 2023 OmniCV Workshop Challenge (Saravanabalagi Ramachandran et al., 2023)

{{<citation>}}

Saravanabalagi Ramachandran, Nathaniel Cibik, Ganesh Sistu, John McDonald. (2023)  
**WoodScape Motion Segmentation for Autonomous Driving -- CVPR 2023 OmniCV Workshop Challenge**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Computer Vision  
[Paper Link](http://arxiv.org/abs/2401.00910v1)  

---


**ABSTRACT**  
Motion segmentation is a complex yet indispensable task in autonomous driving. The challenges introduced by the ego-motion of the cameras, radial distortion in fisheye lenses, and the need for temporal consistency make the task more complicated, rendering traditional and standard Convolutional Neural Network (CNN) approaches less effective. The consequent laborious data labeling, representation of diverse and uncommon scenarios, and extensive data capture requirements underscore the imperative of synthetic data for improving machine learning model performance. To this end, we employ the PD-WoodScape synthetic dataset developed by Parallel Domain, alongside the WoodScape fisheye dataset. Thus, we present the WoodScape fisheye motion segmentation challenge for autonomous driving, held as part of the CVPR 2023 Workshop on Omnidirectional Computer Vision (OmniCV). As one of the first competitions focused on fisheye motion segmentation, we aim to explore and evaluate the potential and impact of utilizing synthetic data in this domain. In this paper, we provide a detailed analysis on the competition which attracted the participation of 112 global teams and a total of 234 submissions. This study delineates the complexities inherent in the task of motion segmentation, emphasizes the significance of fisheye datasets, articulate the necessity for synthetic datasets and the resultant domain gap they engender, outlining the foundational blueprint for devising successful solutions. Subsequently, we delve into the details of the baseline experiments and winning methods evaluating their qualitative and quantitative results, providing with useful insights.

{{</citation>}}


### (2/43) Bringing Back the Context: Camera Trap Species Identification as Link Prediction on Multimodal Knowledge Graphs (Vardaan Pahuja et al., 2023)

{{<citation>}}

Vardaan Pahuja, Weidi Luo, Yu Gu, Cheng-Hao Tu, Hong-You Chen, Tanya Berger-Wolf, Charles Stewart, Song Gao, Wei-Lun Chao, Yu Su. (2023)  
**Bringing Back the Context: Camera Trap Species Identification as Link Prediction on Multimodal Knowledge Graphs**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2401.00608v1)  

---


**ABSTRACT**  
Camera traps are valuable tools in animal ecology for biodiversity monitoring and conservation. However, challenges like poor generalization to deployment at new unseen locations limit their practical application. Images are naturally associated with heterogeneous forms of context possibly in different modalities. In this work, we leverage the structured context associated with the camera trap images to improve out-of-distribution generalization for the task of species identification in camera traps. For example, a photo of a wild animal may be associated with information about where and when it was taken, as well as structured biology knowledge about the animal species. While typically overlooked by existing work, bringing back such context offers several potential benefits for better image understanding, such as addressing data scarcity and enhancing generalization. However, effectively integrating such heterogeneous context into the visual domain is a challenging problem. To address this, we propose a novel framework that reformulates species classification as link prediction in a multimodal knowledge graph (KG). This framework seamlessly integrates various forms of multimodal context for visual recognition. We apply this framework for out-of-distribution species classification on the iWildCam2020-WILDS and Snapshot Mountain Zebra datasets and achieve competitive performance with state-of-the-art approaches. Furthermore, our framework successfully incorporates biological taxonomy for improved generalization and enhances sample efficiency for recognizing under-represented species.

{{</citation>}}


### (3/43) Masked Modeling for Self-supervised Representation Learning on Vision and Beyond (Siyuan Li et al., 2023)

{{<citation>}}

Siyuan Li, Luyuan Zhang, Zedong Wang, Di Wu, Lirong Wu, Zicheng Liu, Jun Xia, Cheng Tan, Yang Liu, Baigui Sun, Stan Z. Li. (2023)  
**Masked Modeling for Self-supervised Representation Learning on Vision and Beyond**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2401.00897v1)  

---


**ABSTRACT**  
As the deep learning revolution marches on, self-supervised learning has garnered increasing attention in recent years thanks to its remarkable representation learning ability and the low dependence on labeled data. Among these varied self-supervised techniques, masked modeling has emerged as a distinctive approach that involves predicting parts of the original data that are proportionally masked during training. This paradigm enables deep models to learn robust representations and has demonstrated exceptional performance in the context of computer vision, natural language processing, and other modalities. In this survey, we present a comprehensive review of the masked modeling framework and its methodology. We elaborate on the details of techniques within masked modeling, including diverse masking strategies, recovering targets, network architectures, and more. Then, we systematically investigate its wide-ranging applications across domains. Furthermore, we also explore the commonalities and differences between masked modeling methods in different fields. Toward the end of this paper, we conclude by discussing the limitations of current techniques and point out several potential avenues for advancing masked modeling research. A paper list project with this survey is available at \url{https://github.com/Lupin1998/Awesome-MIM}.

{{</citation>}}


### (4/43) Analyzing Local Representations of Self-supervised Vision Transformers (Ani Vanyan et al., 2023)

{{<citation>}}

Ani Vanyan, Alvard Barseghyan, Hakob Tamazyan, Vahan Huroyan, Hrant Khachatrian, Martin Danelljan. (2023)  
**Analyzing Local Representations of Self-supervised Vision Transformers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.00463v1)  

---


**ABSTRACT**  
In this paper, we present a comparative analysis of various self-supervised Vision Transformers (ViTs), focusing on their local representative power. Inspired by large language models, we examine the abilities of ViTs to perform various computer vision tasks with little to no fine-tuning. We design an evaluation framework to analyze the quality of local, i.e. patch-level, representations in the context of few-shot semantic segmentation, instance identification, object retrieval, and tracking. We discover that contrastive learning based methods like DINO produce more universal patch representations that can be immediately applied for downstream tasks with no parameter tuning, compared to masked image modeling. The embeddings learned using the latter approach, e.g. in masked autoencoders, have high variance features that harm distance-based algorithms, such as k-NN, and do not contain useful information for most downstream tasks. Furthermore, we demonstrate that removing these high-variance features enhances k-NN by providing an analysis of the benchmarks for this work and for Scale-MAE, a recent extension of masked autoencoders. Finally, we find an object instance retrieval setting where DINOv2, a model pretrained on two orders of magnitude more data, performs worse than its less compute-intensive counterpart DINO.

{{</citation>}}


### (5/43) Bidirectional Trained Tree-Structured Decoder for Handwritten Mathematical Expression Recognition (Hanbo Cheng et al., 2023)

{{<citation>}}

Hanbo Cheng, Chenyu Liu, Pengfei Hu, Zhenrong Zhang, Jiefeng Ma, Jun Du. (2023)  
**Bidirectional Trained Tree-Structured Decoder for Handwritten Mathematical Expression Recognition**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Language Model, OCR  
[Paper Link](http://arxiv.org/abs/2401.00435v1)  

---


**ABSTRACT**  
The Handwritten Mathematical Expression Recognition (HMER) task is a critical branch in the field of OCR. Recent studies have demonstrated that incorporating bidirectional context information significantly improves the performance of HMER models. However, existing methods fail to effectively utilize bidirectional context information during the inference stage. Furthermore, current bidirectional training methods are primarily designed for string decoders and cannot adequately generalize to tree decoders, which offer superior generalization capabilities and structural analysis capacity. In order to overcome these limitations, we propose the Mirror-Flipped Symbol Layout Tree (MF-SLT) and Bidirectional Asynchronous Training (BAT) structure. Our method extends the bidirectional training strategy to the tree decoder, allowing for more effective training by leveraging bidirectional information. Additionally, we analyze the impact of the visual and linguistic perception of the HMER model separately and introduce the Shared Language Modeling (SLM) mechanism. Through the SLM, we enhance the model's robustness and generalization when dealing with visual ambiguity, particularly in scenarios with abundant training data. Our approach has been validated through extensive experiments, demonstrating its ability to achieve new state-of-the-art results on the CROHME 2014, 2016, and 2019 datasets, as well as the HME100K dataset. The code used in our experiments will be publicly available.

{{</citation>}}


### (6/43) SVFAP: Self-supervised Video Facial Affect Perceiver (Licai Sun et al., 2023)

{{<citation>}}

Licai Sun, Zheng Lian, Kexin Wang, Yu He, Mingyu Xu, Haiyang Sun, Bin Liu, Jianhua Tao. (2023)  
**SVFAP: Self-supervised Video Facial Affect Perceiver**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-HC, cs-MM, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.00416v1)  

---


**ABSTRACT**  
Video-based facial affect analysis has recently attracted increasing attention owing to its critical role in human-computer interaction. Previous studies mainly focus on developing various deep learning architectures and training them in a fully supervised manner. Although significant progress has been achieved by these supervised methods, the longstanding lack of large-scale high-quality labeled data severely hinders their further improvements. Motivated by the recent success of self-supervised learning in computer vision, this paper introduces a self-supervised approach, termed Self-supervised Video Facial Affect Perceiver (SVFAP), to address the dilemma faced by supervised methods. Specifically, SVFAP leverages masked facial video autoencoding to perform self-supervised pre-training on massive unlabeled facial videos. Considering that large spatiotemporal redundancy exists in facial videos, we propose a novel temporal pyramid and spatial bottleneck Transformer as the encoder of SVFAP, which not only enjoys low computational cost but also achieves excellent performance. To verify the effectiveness of our method, we conduct experiments on nine datasets spanning three downstream tasks, including dynamic facial expression recognition, dimensional emotion recognition, and personality recognition. Comprehensive results demonstrate that SVFAP can learn powerful affect-related representations via large-scale self-supervised pre-training and it significantly outperforms previous state-of-the-art methods on all datasets. Codes will be available at https://github.com/sunlicai/SVFAP.

{{</citation>}}


### (7/43) Is It Possible to Backdoor Face Forgery Detection with Natural Triggers? (Xiaoxuan Han et al., 2023)

{{<citation>}}

Xiaoxuan Han, Songlin Yang, Wei Wang, Ziwen He, Jing Dong. (2023)  
**Is It Possible to Backdoor Face Forgery Detection with Natural Triggers?**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.00414v1)  

---


**ABSTRACT**  
Deep neural networks have significantly improved the performance of face forgery detection models in discriminating Artificial Intelligent Generated Content (AIGC). However, their security is significantly threatened by the injection of triggers during model training (i.e., backdoor attacks). Although existing backdoor defenses and manual data selection can mitigate those using human-eye-sensitive triggers, such as patches or adversarial noises, the more challenging natural backdoor triggers remain insufficiently researched. To further investigate natural triggers, we propose a novel analysis-by-synthesis backdoor attack against face forgery detection models, which embeds natural triggers in the latent space. We thoroughly study such backdoor vulnerability from two perspectives: (1) Model Discrimination (Optimization-Based Trigger): we adopt a substitute detection model and find the trigger by minimizing the cross-entropy loss; (2) Data Distribution (Custom Trigger): we manipulate the uncommon facial attributes in the long-tailed distribution to generate poisoned samples without the supervision from detection models. Furthermore, to completely evaluate the detection models towards the latest AIGC, we utilize both state-of-the-art StyleGAN and Stable Diffusion for trigger generation. Finally, these backdoor triggers introduce specific semantic features to the generated poisoned samples (e.g., skin textures and smile), which are more natural and robust. Extensive experiments show that our method is superior from three levels: (1) Attack Success Rate: ours achieves a high attack success rate (over 99%) and incurs a small model accuracy drop (below 0.2%) with a low poisoning rate (less than 3%); (2) Backdoor Defense: ours shows better robust performance when faced with existing backdoor defense methods; (3) Human Inspection: ours is less human-eye-sensitive from a comprehensive user study.

{{</citation>}}


### (8/43) A Two-stream Hybrid CNN-Transformer Network for Skeleton-based Human Interaction Recognition (Ruoqi Yin et al., 2023)

{{<citation>}}

Ruoqi Yin, Jianqin Yin. (2023)  
**A Two-stream Hybrid CNN-Transformer Network for Skeleton-based Human Interaction Recognition**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.00409v1)  

---


**ABSTRACT**  
Human Interaction Recognition is the process of identifying interactive actions between multiple participants in a specific situation. The aim is to recognise the action interactions between multiple entities and their meaning. Many single Convolutional Neural Network has issues, such as the inability to capture global instance interaction features or difficulty in training, leading to ambiguity in action semantics. In addition, the computational complexity of the Transformer cannot be ignored, and its ability to capture local information and motion features in the image is poor. In this work, we propose a Two-stream Hybrid CNN-Transformer Network (THCT-Net), which exploits the local specificity of CNN and models global dependencies through the Transformer. CNN and Transformer simultaneously model the entity, time and space relationships between interactive entities respectively. Specifically, Transformer-based stream integrates 3D convolutions with multi-head self-attention to learn inter-token correlations; We propose a new multi-branch CNN framework for CNN-based streams that automatically learns joint spatio-temporal features from skeleton sequences. The convolutional layer independently learns the local features of each joint neighborhood and aggregates the features of all joints. And the raw skeleton coordinates as well as their temporal difference are integrated with a dual-branch paradigm to fuse the motion features of the skeleton. Besides, a residual structure is added to speed up training convergence. Finally, the recognition results of the two branches are fused using parallel splicing. Experimental results on diverse and challenging datasets, demonstrate that the proposed method can better comprehend and infer the meaning and context of various actions, outperforming state-of-the-art methods.

{{</citation>}}


### (9/43) Generative Model-Driven Synthetic Training Image Generation: An Approach to Cognition in Rail Defect Detection (Rahatara Ferdousi et al., 2023)

{{<citation>}}

Rahatara Ferdousi, Chunsheng Yang, M. Anwar Hossain, Fedwa Laamarti, M. Shamim Hossain, Abdulmotaleb El Saddik. (2023)  
**Generative Model-Driven Synthetic Training Image Generation: An Approach to Cognition in Rail Defect Detection**  

---
Primary Category: cs.CV  
Categories: 68T05, 94A08, 90B25, I-2-6; I-2-10; I-5-4; I-4-10, cs-AI, cs-CV, cs-LG, cs-MM, cs.CV, eess-IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.00393v1)  

---


**ABSTRACT**  
Recent advancements in cognitive computing, with the integration of deep learning techniques, have facilitated the development of intelligent cognitive systems (ICS). This is particularly beneficial in the context of rail defect detection, where the ICS would emulate human-like analysis of image data for defect patterns. Despite the success of Convolutional Neural Networks (CNN) in visual defect classification, the scarcity of large datasets for rail defect detection remains a challenge due to infrequent accident events that would result in defective parts and images. Contemporary researchers have addressed this data scarcity challenge by exploring rule-based and generative data augmentation models. Among these, Variational Autoencoder (VAE) models can generate realistic data without extensive baseline datasets for noise modeling. This study proposes a VAE-based synthetic image generation technique for rail defects, incorporating weight decay regularization and image reconstruction loss to prevent overfitting. The proposed method is applied to create a synthetic dataset for the Canadian Pacific Railway (CPR) with just 50 real samples across five classes. Remarkably, 500 synthetic samples are generated with a minimal reconstruction loss of 0.021. A Visual Transformer (ViT) model underwent fine-tuning using this synthetic CPR dataset, achieving high accuracy rates (98%-99%) in classifying the five defect classes. This research offers a promising solution to the data scarcity challenge in rail defect detection, showcasing the potential for robust ICS development in this domain.

{{</citation>}}


### (10/43) Horizontal Federated Computer Vision (Paul K. Mandal et al., 2023)

{{<citation>}}

Paul K. Mandal, Cole Leo, Connor Hurley. (2023)  
**Horizontal Federated Computer Vision**  

---
Primary Category: cs.CV  
Categories: C-2-4; I-2-8; I-4; I-4-8, cs-AI, cs-CV, cs-DC, cs-LG, cs.CV  
Keywords: Computer Vision  
[Paper Link](http://arxiv.org/abs/2401.00390v1)  

---


**ABSTRACT**  
In the modern world, the amount of visual data recorded has been rapidly increasing. In many cases, data is stored in geographically distinct locations and thus requires a large amount of time and space to consolidate. Sometimes, there are also regulations for privacy protection which prevent data consolidation. In this work, we present federated implementations for object detection and recognition using a federated Faster R-CNN (FRCNN) and image segmentation using a federated Fully Convolutional Network (FCN). Our FRCNN was trained on 5000 examples of the COCO2017 dataset while our FCN was trained on the entire train set of the CamVid dataset. The proposed federated models address the challenges posed by the increasing volume and decentralized nature of visual data, offering efficient solutions in compliance with privacy regulations.

{{</citation>}}


### (11/43) EMAGE: Towards Unified Holistic Co-Speech Gesture Generation via Masked Audio Gesture Modeling (Haiyang Liu et al., 2023)

{{<citation>}}

Haiyang Liu, Zihao Zhu, Giorgio Becherini, Yichen Peng, Mingyang Su, You Zhou, Naoya Iwamoto, Bo Zheng, Michael J. Black. (2023)  
**EMAGE: Towards Unified Holistic Co-Speech Gesture Generation via Masked Audio Gesture Modeling**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.00374v2)  

---


**ABSTRACT**  
We propose EMAGE, a framework to generate full-body human gestures from audio and masked gestures, encompassing facial, local body, hands, and global movements. To achieve this, we first introduce BEATX (BEAT-SMPLX-FLAME), a new mesh-level holistic co-speech dataset. BEATX combines MoShed SMPLX body with FLAME head parameters and further refines the modeling of head, neck, and finger movements, offering a community-standardized, high-quality 3D motion captured dataset. EMAGE leverages masked body gesture priors during training to boost inference performance. It involves a Masked Audio Gesture Transformer, facilitating joint training on audio-to-gesture generation and masked gesture reconstruction to effectively encode audio and body gesture hints. Encoded body hints from masked gestures are then separately employed to generate facial and body movements. Moreover, EMAGE adaptively merges speech features from the audio's rhythm and content and utilizes four compositional VQ-VAEs to enhance the results' fidelity and diversity. Experiments demonstrate that EMAGE generates holistic gestures with state-of-the-art performance and is flexible in accepting predefined spatial-temporal gesture inputs, generating complete, audio-synchronized results. Our code and dataset are available at https://pantomatrix.github.io/EMAGE/

{{</citation>}}


### (12/43) Multi-Granularity Representation Learning for Sketch-based Dynamic Face Image Retrieval (Liang Wang et al., 2023)

{{<citation>}}

Liang Wang, Dawei Dai, Shiyu Fu, Guoyin Wang. (2023)  
**Multi-Granularity Representation Learning for Sketch-based Dynamic Face Image Retrieval**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Representation Learning, Sketch  
[Paper Link](http://arxiv.org/abs/2401.00371v1)  

---


**ABSTRACT**  
In specific scenarios, face sketch can be used to identify a person. However, drawing a face sketch often requires exceptional skill and is time-consuming, limiting its widespread applications in actual scenarios. The new framework of sketch less face image retrieval (SLFIR)[1] attempts to overcome the barriers by providing a means for humans and machines to interact during the drawing process. Considering SLFIR problem, there is a large gap between a partial sketch with few strokes and any whole face photo, resulting in poor performance at the early stages. In this study, we propose a multigranularity (MG) representation learning (MGRL) method to address the SLFIR problem, in which we learn the representation of different granularity regions for a partial sketch, and then, by combining all MG regions of the sketches and images, the final distance was determined. In the experiments, our method outperformed state-of-the-art baselines in terms of early retrieval on two accessible datasets. Codes are available at https://github.com/ddw2AIGROUP2CQUPT/MGRL.

{{</citation>}}


## cs.LG (9)



### (13/43) LaFFi: Leveraging Hybrid Natural Language Feedback for Fine-tuning Language Models (Qianxi Li et al., 2023)

{{<citation>}}

Qianxi Li, Yingyue Cao, Jikun Kang, Tianpei Yang, Xi Chen, Jun Jin, Matthew E. Taylor. (2023)  
**LaFFi: Leveraging Hybrid Natural Language Feedback for Fine-tuning Language Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.00907v1)  

---


**ABSTRACT**  
Fine-tuning Large Language Models (LLMs) adapts a trained model to specific downstream tasks, significantly improving task-specific performance. Supervised Fine-Tuning (SFT) is a common approach, where an LLM is trained to produce desired answers. However, LLMs trained with SFT sometimes make simple mistakes and result in hallucinations on reasoning tasks such as question-answering. Without external feedback, it is difficult for SFT to learn a good mapping between the question and the desired answer, especially with a small dataset. This paper introduces an alternative to SFT called Natural Language Feedback for Finetuning LLMs (LaFFi). LaFFi has LLMs directly predict the feedback they will receive from an annotator. We find that requiring such reflection can significantly improve the accuracy in in-domain question-answering tasks, providing a promising direction for the application of natural language feedback in the realm of SFT LLMs. Additional ablation studies show that the portion of human-annotated data in the annotated datasets affects the fine-tuning performance.

{{</citation>}}


### (14/43) Financial Time-Series Forecasting: Towards Synergizing Performance And Interpretability Within a Hybrid Machine Learning Approach (Shun Liu et al., 2023)

{{<citation>}}

Shun Liu, Kexin Wu, Chufeng Jiang, Bin Huang, Danqing Ma. (2023)  
**Financial Time-Series Forecasting: Towards Synergizing Performance And Interpretability Within a Hybrid Machine Learning Approach**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, q-fin-ST  
Keywords: Financial, LSTM  
[Paper Link](http://arxiv.org/abs/2401.00534v1)  

---


**ABSTRACT**  
In the realm of cryptocurrency, the prediction of Bitcoin prices has garnered substantial attention due to its potential impact on financial markets and investment strategies. This paper propose a comparative study on hybrid machine learning algorithms and leverage on enhancing model interpretability. Specifically, linear regression(OLS, LASSO), long-short term memory(LSTM), decision tree regressors are introduced. Through the grounded experiments, we observe linear regressor achieves the best performance among candidate models. For the interpretability, we carry out a systematic overview on the preprocessing techniques of time-series statistics, including decomposition, auto-correlational function, exponential triple forecasting, which aim to excavate latent relations and complex patterns appeared in the financial time-series forecasting. We believe this work may derive more attention and inspire more researches in the realm of time-series analysis and its realistic applications.

{{</citation>}}


### (15/43) GraphGPT: Graph Learning with Generative Pre-trained Transformers (Qifang Zhao et al., 2023)

{{<citation>}}

Qifang Zhao, Weidong Ren, Tianyu Li, Xiaoxiao Xu, Hong Liu. (2023)  
**GraphGPT: Graph Learning with Generative Pre-trained Transformers**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, GPT, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.00529v1)  

---


**ABSTRACT**  
We introduce \textit{GraphGPT}, a novel model for Graph learning by self-supervised Generative Pre-training Transformers. Our model transforms each graph or sampled subgraph into a sequence of tokens representing the node, edge and attributes reversibly using the Eulerian path first. Then we feed the tokens into a standard transformer decoder and pre-train it with the next-token-prediction (NTP) task. Lastly, we fine-tune the GraphGPT model with the supervised tasks. This intuitive, yet effective model achieves superior or close results to the state-of-the-art methods for the graph-, edge- and node-level tasks on the large scale molecular dataset PCQM4Mv2, the protein-protein association dataset ogbl-ppa and the ogbn-proteins dataset from the Open Graph Benchmark (OGB). Furthermore, the generative pre-training enables us to train GraphGPT up to 400M+ parameters with consistently increasing performance, which is beyond the capability of GNNs and previous graph transformers. The source code and pre-trained checkpoints will be released soon\footnote{\url{https://github.com/alibaba/graph-gpt}} to pave the way for the graph foundation model research, and also to assist the scientific discovery in pharmaceutical, chemistry, material and bio-informatics domains, etc.

{{</citation>}}


### (16/43) Multi-spatial Multi-temporal Air Quality Forecasting with Integrated Monitoring and Reanalysis Data (Yuxiao Hu et al., 2023)

{{<citation>}}

Yuxiao Hu, Qian Li, Xiaodan Shi, Jinyue Yan, Yuntian Chen. (2023)  
**Multi-spatial Multi-temporal Air Quality Forecasting with Integrated Monitoring and Reanalysis Data**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, stat-AP  
Keywords: Graph Convolutional Network  
[Paper Link](http://arxiv.org/abs/2401.00521v1)  

---


**ABSTRACT**  
Accurate air quality forecasting is crucial for public health, environmental monitoring and protection, and urban planning. However, existing methods fail to effectively utilize multi-scale information, both spatially and temporally. Spatially, there is a lack of integration between individual monitoring stations and city-wide scales. Temporally, the periodic nature of air quality variations is often overlooked or inadequately considered. To address these limitations, we present a novel Multi-spatial Multi-temporal air quality forecasting method based on Graph Convolutional Networks and Gated Recurrent Units (M2G2), bridging the gap in air quality forecasting across spatial and temporal scales. The proposed framework consists of two modules: Multi-scale Spatial GCN (MS-GCN) for spatial information fusion and Multi-scale Temporal GRU(MT-GRU) for temporal information integration. In the spatial dimension, the MS-GCN module employs a bidirectional learnable structure and a residual structure, enabling comprehensive information exchange between individual monitoring stations and the city-scale graph. Regarding the temporal dimension, the MT-GRU module adaptively combines information from different temporal scales through parallel hidden states. Leveraging meteorological indicators and four air quality indicators, we present comprehensive comparative analyses and ablation experiments, showcasing the higher accuracy of M2G2 in comparison to nine currently available advanced approaches across all aspects. The improvements of M2G2 over the second-best method on RMSE of the 24h/48h/72h are as follows: PM2.5: (7.72%, 6.67%, 10.45%); PM10: (6.43%, 5.68%, 7.73%); NO2: (5.07%, 7.76%, 16.60%); O3: (6.46%, 6.86%, 9.79%). Furthermore, we demonstrate the effectiveness of each module of M2G2 by ablation study.

{{</citation>}}


### (17/43) Viz: A QLoRA-based Copyright Marketplace for Legally Compliant Generative AI (Dipankar Sarkar, 2023)

{{<citation>}}

Dipankar Sarkar. (2023)  
**Viz: A QLoRA-based Copyright Marketplace for Legally Compliant Generative AI**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI, Generative AI, Legal  
[Paper Link](http://arxiv.org/abs/2401.00503v1)  

---


**ABSTRACT**  
This paper aims to introduce and analyze the Viz system in a comprehensive way, a novel system architecture that integrates Quantized Low-Rank Adapters (QLoRA) to fine-tune large language models (LLM) within a legally compliant and resource efficient marketplace. Viz represents a significant contribution to the field of artificial intelligence, particularly in addressing the challenges of computational efficiency, legal compliance, and economic sustainability in the utilization and monetization of LLMs. The paper delineates the scholarly discourse and developments that have informed the creation of Viz, focusing primarily on the advancements in LLM models, copyright issues in AI training (NYT case, 2023), and the evolution of model fine-tuning techniques, particularly low-rank adapters and quantized low-rank adapters, to create a sustainable and economically compliant framework for LLM utilization. The economic model it proposes benefits content creators, AI developers, and end-users, delineating a harmonious integration of technology, economy, and law, offering a comprehensive solution to the complex challenges of today's AI landscape.

{{</citation>}}


### (18/43) Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws (Nikhil Sardana et al., 2023)

{{<citation>}}

Nikhil Sardana, Jonathan Frankle. (2023)  
**Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.00448v1)  

---


**ABSTRACT**  
Large language model (LLM) scaling laws are empirical formulas that estimate changes in model quality as a result of increasing parameter count and training data. However, these formulas, including the popular DeepMind Chinchilla scaling laws, neglect to include the cost of inference. We modify the Chinchilla scaling laws to calculate the optimal LLM parameter count and pre-training data size to train and deploy a model of a given quality and inference demand. We conduct our analysis both in terms of a compute budget and real-world costs and find that LLM researchers expecting reasonably large inference demand (~1B requests) should train models smaller and longer than Chinchilla-optimal.

{{</citation>}}


### (19/43) MSGNet: Learning Multi-Scale Inter-Series Correlations for Multivariate Time Series Forecasting (Wanlin Cai et al., 2023)

{{<citation>}}

Wanlin Cai, Yuxuan Liang, Xianggen Liu, Jianshuai Feng, Yuankai Wu. (2023)  
**MSGNet: Learning Multi-Scale Inter-Series Correlations for Multivariate Time Series Forecasting**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2401.00423v1)  

---


**ABSTRACT**  
Multivariate time series forecasting poses an ongoing challenge across various disciplines. Time series data often exhibit diverse intra-series and inter-series correlations, contributing to intricate and interwoven dependencies that have been the focus of numerous studies. Nevertheless, a significant research gap remains in comprehending the varying inter-series correlations across different time scales among multiple time series, an area that has received limited attention in the literature. To bridge this gap, this paper introduces MSGNet, an advanced deep learning model designed to capture the varying inter-series correlations across multiple time scales using frequency domain analysis and adaptive graph convolution. By leveraging frequency domain analysis, MSGNet effectively extracts salient periodic patterns and decomposes the time series into distinct time scales. The model incorporates a self-attention mechanism to capture intra-series dependencies, while introducing an adaptive mixhop graph convolution layer to autonomously learn diverse inter-series correlations within each time scale. Extensive experiments are conducted on several real-world datasets to showcase the effectiveness of MSGNet. Furthermore, MSGNet possesses the ability to automatically learn explainable multi-scale inter-series correlations, exhibiting strong generalization capabilities even when applied to out-of-distribution samples.

{{</citation>}}


### (20/43) HQ-VAE: Hierarchical Discrete Representation Learning with Variational Bayes (Yuhta Takida et al., 2023)

{{<citation>}}

Yuhta Takida, Yukara Ikemiya, Takashi Shibuya, Kazuki Shimada, Woosung Choi, Chieh-Hsin Lai, Naoki Murata, Toshimitsu Uesaka, Kengo Uchida, Wei-Hsiang Liao, Yuki Mitsufuji. (2023)  
**HQ-VAE: Hierarchical Discrete Representation Learning with Variational Bayes**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2401.00365v1)  

---


**ABSTRACT**  
Vector quantization (VQ) is a technique to deterministically learn features with discrete codebook representations. It is commonly performed with a variational autoencoding model, VQ-VAE, which can be further extended to hierarchical structures for making high-fidelity reconstructions. However, such hierarchical extensions of VQ-VAE often suffer from the codebook/layer collapse issue, where the codebook is not efficiently used to express the data, and hence degrades reconstruction accuracy. To mitigate this problem, we propose a novel unified framework to stochastically learn hierarchical discrete representation on the basis of the variational Bayes framework, called hierarchically quantized variational autoencoder (HQ-VAE). HQ-VAE naturally generalizes the hierarchical variants of VQ-VAE, such as VQ-VAE-2 and residual-quantized VAE (RQ-VAE), and provides them with a Bayesian training scheme. Our comprehensive experiments on image datasets show that HQ-VAE enhances codebook usage and improves reconstruction performance. We also validated HQ-VAE in terms of its applicability to a different modality with an audio dataset.

{{</citation>}}


### (21/43) Tight Finite Time Bounds of Two-Time-Scale Linear Stochastic Approximation with Markovian Noise (Shaan Ul Haque et al., 2023)

{{<citation>}}

Shaan Ul Haque, Sajad Khodadadian, Siva Theja Maguluri. (2023)  
**Tight Finite Time Bounds of Two-Time-Scale Linear Stochastic Approximation with Markovian Noise**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SY, cs.LG, eess-SY, math-OC  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.00364v1)  

---


**ABSTRACT**  
Stochastic approximation (SA) is an iterative algorithm to find the fixed point of an operator given noisy samples of this operator. SA appears in many areas such as optimization and Reinforcement Learning (RL). When implemented in practice, the noise that appears in the update of RL algorithms is naturally Markovian. Furthermore, in some settings, such as gradient TD, SA is employed in a two-time-scale manner. The mix of Markovian noise along with the two-time-scale structure results in an algorithm which is complex to analyze theoretically. In this paper, we characterize a tight convergence bound for the iterations of linear two-time-scale SA with Markovian noise. Our results show the convergence behavior of this algorithm given various choices of step sizes. Applying our result to the well-known TDC algorithm, we show the first $O(1/\epsilon)$ sample complexity for the convergence of this algorithm, outperforming all the previous work. Similarly, our results can be applied to establish the convergence behavior of a variety of RL algorithms, such as TD-learning with Polyak averaging, GTD, and GTD2.

{{</citation>}}


## cs.AI (2)



### (22/43) Fairness in Serving Large Language Models (Ying Sheng et al., 2023)

{{<citation>}}

Ying Sheng, Shiyi Cao, Dacheng Li, Banghua Zhu, Zhuohan Li, Danyang Zhuo, Joseph E. Gonzalez, Ion Stoica. (2023)  
**Fairness in Serving Large Language Models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs-PF, cs.AI  
Keywords: BARD, ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2401.00588v1)  

---


**ABSTRACT**  
High-demand LLM inference services (e.g., ChatGPT and BARD) support a wide range of requests from short chat conversations to long document reading. To ensure that all client requests are processed fairly, most major LLM inference services have request rate limits, to ensure that no client can dominate the request queue. However, this rudimentary notion of fairness also results in under-utilization of the resources and poor client experience when there is spare capacity. While there is a rich literature on fair scheduling, serving LLMs presents new challenges due to their unpredictable request lengths and their unique batching characteristics on parallel accelerators. This paper introduces the definition of LLM serving fairness based on a cost function that accounts for the number of input and output tokens processed. To achieve fairness in serving, we propose a novel scheduling algorithm, the Virtual Token Counter (VTC), a fair scheduler based on the continuous batching mechanism. We prove a 2x tight upper bound on the service difference between two backlogged clients, adhering to the requirement of work-conserving. Through extensive experiments, we demonstrate the superior performance of VTC in ensuring fairness, especially in contrast to other baseline methods, which exhibit shortcomings under various conditions.

{{</citation>}}


### (23/43) Brain-Conditional Multimodal Synthesis: A Survey and Taxonomy (Weijian Mai et al., 2023)

{{<citation>}}

Weijian Mai, Jian Zhang, Pengfei Fang, Zhijun Zhang. (2023)  
**Brain-Conditional Multimodal Synthesis: A Survey and Taxonomy**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.00430v1)  

---


**ABSTRACT**  
In the era of Artificial Intelligence Generated Content (AIGC), conditional multimodal synthesis technologies (e.g., text-to-image, text-to-video, text-to-audio, etc) are gradually reshaping the natural content in the real world. The key to multimodal synthesis technology is to establish the mapping relationship between different modalities. Brain signals, serving as potential reflections of how the brain interprets external information, exhibit a distinctive One-to-Many correspondence with various external modalities. This correspondence makes brain signals emerge as a promising guiding condition for multimodal content synthesis. Brian-conditional multimodal synthesis refers to decoding brain signals back to perceptual experience, which is crucial for developing practical brain-computer interface systems and unraveling complex mechanisms underlying how the brain perceives and comprehends external stimuli. This survey comprehensively examines the emerging field of AIGC-based Brain-conditional Multimodal Synthesis, termed AIGC-Brain, to delineate the current landscape and future directions. To begin, related brain neuroimaging datasets, functional brain regions, and mainstream generative models are introduced as the foundation of AIGC-Brain decoding and analysis. Next, we provide a comprehensive taxonomy for AIGC-Brain decoding models and present task-specific representative work and detailed implementation strategies to facilitate comparison and in-depth analysis. Quality assessments are then introduced for both qualitative and quantitative evaluation. Finally, this survey explores insights gained, providing current challenges and outlining prospects of AIGC-Brain. Being the inaugural survey in this domain, this paper paves the way for the progress of AIGC-Brain research, offering a foundational overview to guide future work.

{{</citation>}}


## eess.IV (1)



### (24/43) Brain Tumor Segmentation Based on Deep Learning, Attention Mechanisms, and Energy-Based Uncertainty Prediction (Zachary Schwehr et al., 2023)

{{<citation>}}

Zachary Schwehr, Sriman Achanta. (2023)  
**Brain Tumor Segmentation Based on Deep Learning, Attention Mechanisms, and Energy-Based Uncertainty Prediction**  

---
Primary Category: eess.IV  
Categories: cs-AI, eess-IV, eess.IV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2401.00587v1)  

---


**ABSTRACT**  
Brain tumors are one of the deadliest forms of cancer with a mortality rate of over 80%. A quick and accurate diagnosis is crucial to increase the chance of survival. However, in medical analysis, the manual annotation and segmentation of a brain tumor can be a complicated task. Multiple MRI modalities are typically analyzed as they provide unique information regarding the tumor regions. Although these MRI modalities are helpful for segmenting gliomas, they tend to increase overfitting and computation. This paper proposes a region of interest detection algorithm that is implemented during data preprocessing to locate salient features and remove extraneous MRI data. This decreases the input size, allowing for more aggressive data augmentations and deeper neural networks. Following the preprocessing of the MRI modalities, a fully convolutional autoencoder with soft attention segments the different brain MRIs. When these deep learning algorithms are implemented in practice, analysts and physicians cannot differentiate between accurate and inaccurate predictions. Subsequently, test time augmentations and an energy-based model were used for voxel-based uncertainty predictions. Experimentation was conducted on the BraTS benchmarks and achieved state-of-the-art segmentation performance. Additionally, qualitative results were used to assess the segmentation models and uncertainty predictions.

{{</citation>}}


## cs.CL (9)



### (25/43) An Analysis of Embedding Layers and Similarity Scores using Siamese Neural Networks (Yash Bingi et al., 2023)

{{<citation>}}

Yash Bingi, Yiqiao Yin. (2023)  
**An Analysis of Embedding Layers and Similarity Scores using Siamese Neural Networks**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: AI, BERT, Embedding, Google, PaLM  
[Paper Link](http://arxiv.org/abs/2401.00582v1)  

---


**ABSTRACT**  
Large Lanugage Models (LLMs) are gaining increasing popularity in a variety of use cases, from language understanding and writing to assistance in application development. One of the most important aspects for optimal funcionality of LLMs is embedding layers. Word embeddings are distributed representations of words in a continuous vector space. In the context of LLMs, words or tokens from the input text are transformed into high-dimensional vectors using unique algorithms specific to the model. Our research examines the embedding algorithms from leading companies in the industry, such as OpenAI, Google's PaLM, and BERT. Using medical data, we have analyzed similarity scores of each embedding layer, observing differences in performance among each algorithm. To enhance each model and provide an additional encoding layer, we also implemented Siamese Neural Networks. After observing changes in performance with the addition of the model, we measured the carbon footage per epoch of training. The carbon footprint associated with large language models (LLMs) is a significant concern, and should be taken into consideration when selecting algorithms for a variety of use cases. Overall, our research compared the accuracy different, leading embedding algorithms and their carbon footage, allowing for a holistic review of each embedding algorithm.

{{</citation>}}


### (26/43) Exploring the Effectiveness of Instruction Tuning in Biomedical Language Processing (Omid Rohanian et al., 2023)

{{<citation>}}

Omid Rohanian, Mohammadmahdi Nouriborji, David A. Clifton. (2023)  
**Exploring the Effectiveness of Instruction Tuning in Biomedical Language Processing**  

---
Primary Category: cs.CL  
Categories: 68T50, I-2-7, cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: BERT, ChatGPT, Clinical, GPT, Language Model, NER, NLI, NLP, Named Entity Recognition, Natural Language Inference, Natural Language Processing, Relation Extraction  
[Paper Link](http://arxiv.org/abs/2401.00579v1)  

---


**ABSTRACT**  
Large Language Models (LLMs), particularly those similar to ChatGPT, have significantly influenced the field of Natural Language Processing (NLP). While these models excel in general language tasks, their performance in domain-specific downstream tasks such as biomedical and clinical Named Entity Recognition (NER), Relation Extraction (RE), and Medical Natural Language Inference (NLI) is still evolving. In this context, our study investigates the potential of instruction tuning for biomedical language processing, applying this technique to two general LLMs of substantial scale. We present a comprehensive, instruction-based model trained on a dataset that consists of approximately $200,000$ instruction-focused samples. This dataset represents a carefully curated compilation of existing data, meticulously adapted and reformatted to align with the specific requirements of our instruction-based tasks. This initiative represents an important step in utilising such models to achieve results on par with specialised encoder-only models like BioBERT and BioClinicalBERT for various classical biomedical NLP tasks. Our work includes an analysis of the dataset's composition and its impact on model performance, providing insights into the intricacies of instruction tuning. By sharing our codes, models, and the distinctively assembled instruction-based dataset, we seek to encourage ongoing research and development in this area.

{{</citation>}}


### (27/43) HSC-GPT: A Large Language Model for Human Settlements Construction (Chen Ran et al., 2023)

{{<citation>}}

Chen Ran, Yao Xueqi, Jiang Xuhui, Han Zhengqi, Guo Jingze, Zhang Xianyue, Lin Chunyu, Liu Chumin, Zhao Jing, Lian Zeke, Zhang Jingjing, Li Keke. (2023)  
**HSC-GPT: A Large Language Model for Human Settlements Construction**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, GPT, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2401.00504v1)  

---


**ABSTRACT**  
The field of human settlement construction encompasses a range of spatial designs and management tasks, including urban planning and landscape architecture design. These tasks involve a plethora of instructions and descriptions presented in natural language, which are essential for understanding design requirements and producing effective design solutions. Recent research has sought to integrate natural language processing (NLP) and generative artificial intelligence (AI) into human settlement construction tasks. Due to the efficient processing and analysis capabilities of AI with data, significant successes have been achieved in design within this domain. However, this task still faces several fundamental challenges. The semantic information involved includes complex spatial details, diverse data source formats, high sensitivity to regional culture, and demanding requirements for innovation and rigor in work scenarios. These factors lead to limitations when applying general generative AI in this field, further exacerbated by a lack of high-quality data for model training. To address these challenges, this paper first proposes HSC-GPT, a large-scale language model framework specifically designed for tasks in human settlement construction, considering the unique characteristics of this domain.

{{</citation>}}


### (28/43) GeoGalactica: A Scientific Large Language Model in Geoscience (Zhouhan Lin et al., 2023)

{{<citation>}}

Zhouhan Lin, Cheng Deng, Le Zhou, Tianhang Zhang, Yi Xu, Yutong Xu, Zhongmou He, Yuanyuan Shi, Beiya Dai, Yunchong Song, Boyi Zeng, Qiyuan Chen, Tao Shi, Tianyu Huang, Yiwei Xu, Shu Wang, Luoyi Fu, Weinan Zhang, Junxian He, Chao Ma, Yunqiang Zhu, Xinbing Wang, Chenghu Zhou. (2023)  
**GeoGalactica: A Scientific Large Language Model in Geoscience**  

---
Primary Category: cs.CL  
Categories: I-2-7; F-4-1, cs-CL, cs.CL  
Keywords: AI, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2401.00434v1)  

---


**ABSTRACT**  
Large language models (LLMs) have achieved huge success for their general knowledge and ability to solve a wide spectrum of tasks in natural language processing (NLP). Due to their impressive abilities, LLMs have shed light on potential inter-discipline applications to foster scientific discoveries of a specific domain by using artificial intelligence (AI for science, AI4S). In the meantime, utilizing NLP techniques in geoscience research and practice is wide and convoluted, contributing from knowledge extraction and document classification to question answering and knowledge discovery. In this work, we take the initial step to leverage LLM for science, through a rather straightforward approach. We try to specialize an LLM into geoscience, by further pre-training the model with a vast amount of texts in geoscience, as well as supervised fine-tuning (SFT) the resulting model with our custom collected instruction tuning dataset. These efforts result in a model GeoGalactica consisting of 30 billion parameters. To our best knowledge, it is the largest language model for the geoscience domain. More specifically, GeoGalactica is from further pre-training of Galactica. We train GeoGalactica over a geoscience-related text corpus containing 65 billion tokens curated from extensive data sources in the big science project Deep-time Digital Earth (DDE), preserving as the largest geoscience-specific text corpus. Then we fine-tune the model with 1 million pairs of instruction-tuning data consisting of questions that demand professional geoscience knowledge to answer. In this technical report, we will illustrate in detail all aspects of GeoGalactica, including data collection, data cleaning, base model selection, pre-training, SFT, and evaluation. We open-source our data curation tools and the checkpoints of GeoGalactica during the first 3/4 of pre-training.

{{</citation>}}


### (29/43) keqing: knowledge-based question answering is a nature chain-of-thought mentor of LLM (Chaojie Wang et al., 2023)

{{<citation>}}

Chaojie Wang, Yishi Xu, Zhong Peng, Chenxi Zhang, Bo Chen, Xinrun Wang, Lei Feng, Bo An. (2023)  
**keqing: knowledge-based question answering is a nature chain-of-thought mentor of LLM**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Information Retrieval, NLP, QA  
[Paper Link](http://arxiv.org/abs/2401.00426v1)  

---


**ABSTRACT**  
Large language models (LLMs) have exhibited remarkable performance on various natural language processing (NLP) tasks, especially for question answering. However, in the face of problems beyond the scope of knowledge, these LLMs tend to talk nonsense with a straight face, where the potential solution could be incorporating an Information Retrieval (IR) module and generating response based on these retrieved knowledge. In this paper, we present a novel framework to assist LLMs, such as ChatGPT, to retrieve question-related structured information on the knowledge graph, and demonstrate that Knowledge-based question answering (Keqing) could be a nature Chain-of-Thought (CoT) mentor to guide the LLM to sequentially find the answer entities of a complex question through interpretable logical chains. Specifically, the workflow of Keqing will execute decomposing a complex question according to predefined templates, retrieving candidate entities on knowledge graph, reasoning answers of sub-questions, and finally generating response with reasoning paths, which greatly improves the reliability of LLM's response. The experimental results on KBQA datasets show that Keqing can achieve competitive performance and illustrate the logic of answering each question.

{{</citation>}}


### (30/43) SDIF-DA: A Shallow-to-Deep Interaction Framework with Data Augmentation for Multi-modal Intent Detection (Shijue Huang et al., 2023)

{{<citation>}}

Shijue Huang, Libo Qin, Bingbing Wang, Geng Tu, Ruifeng Xu. (2023)  
**SDIF-DA: A Shallow-to-Deep Interaction Framework with Data Augmentation for Multi-modal Intent Detection**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Augmentation, ChatGPT, GPT, Intent Detection  
[Paper Link](http://arxiv.org/abs/2401.00424v1)  

---


**ABSTRACT**  
Multi-modal intent detection aims to utilize various modalities to understand the user's intentions, which is essential for the deployment of dialogue systems in real-world scenarios. The two core challenges for multi-modal intent detection are (1) how to effectively align and fuse different features of modalities and (2) the limited labeled multi-modal intent training data. In this work, we introduce a shallow-to-deep interaction framework with data augmentation (SDIF-DA) to address the above challenges. Firstly, SDIF-DA leverages a shallow-to-deep interaction module to progressively and effectively align and fuse features across text, video, and audio modalities. Secondly, we propose a ChatGPT-based data augmentation approach to automatically augment sufficient training data. Experimental results demonstrate that SDIF-DA can effectively align and fuse multi-modal features by achieving state-of-the-art performance. In addition, extensive analyses show that the introduced data augmentation approach can successfully distill knowledge from the large language model.

{{</citation>}}


### (31/43) RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models (Yuanhao Wu et al., 2023)

{{<citation>}}

Yuanhao Wu, Juno Zhu, Siliang Xu, Kashun Shum, Cheng Niu, Randy Zhong, Juntong Song, Tong Zhang. (2023)  
**RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2401.00396v1)  

---


**ABSTRACT**  
Retrieval-augmented generation (RAG) has become a main technique for alleviating hallucinations in large language models (LLMs). Despite the integration of RAG, LLMs may still present unsupported or contradictory claims to the retrieved contents. In order to develop effective hallucination prevention strategies under RAG, it is important to create benchmark datasets that can measure the extent of hallucination. This paper presents RAGTruth, a corpus tailored for analyzing word-level hallucinations in various domains and tasks within the standard RAG frameworks for LLM applications. RAGTruth comprises nearly 18,000 naturally generated responses from diverse LLMs using RAG. These responses have undergone meticulous manual annotations at both the individual cases and word levels, incorporating evaluations of hallucination intensity. We not only benchmark hallucination frequencies across different LLMs, but also critically assess the effectiveness of several existing hallucination detection methodologies. Furthermore, we show that using a high-quality dataset such as RAGTruth, it is possible to finetune a relatively small LLM and achieve a competitive level of performance in hallucination detection when compared to the existing prompt-based approaches using state-of-the-art large language models such as GPT-4.

{{</citation>}}


### (32/43) FusionMind -- Improving question and answering with external context fusion (Shreyas Verma et al., 2023)

{{<citation>}}

Shreyas Verma, Manoj Parmar, Palash Choudhary, Sanchita Porwal. (2023)  
**FusionMind -- Improving question and answering with external context fusion**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GNN, Graph Neural Network, Graph Neural Networks, QA  
[Paper Link](http://arxiv.org/abs/2401.00388v1)  

---


**ABSTRACT**  
Answering questions using pre-trained language models (LMs) and knowledge graphs (KGs) presents challenges in identifying relevant knowledge and performing joint reasoning.We compared LMs (fine-tuned for the task) with the previously published QAGNN method for the Question-answering (QA) objective and further measured the impact of additional factual context on the QAGNN performance. The QAGNN method employs LMs to encode QA context and estimate KG node importance, and effectively update the question choice entity representations using Graph Neural Networks (GNNs). We further experimented with enhancing the QA context encoding by incorporating relevant knowledge facts for the question stem. The models are trained on the OpenbookQA dataset, which contains ~6000 4-way multiple choice questions and is widely used as a benchmark for QA tasks. Through our experimentation, we found that incorporating knowledge facts context led to a significant improvement in performance. In contrast, the addition of knowledge graphs to language models resulted in only a modest increase. This suggests that the integration of contextual knowledge facts may be more impactful for enhancing question answering performance compared to solely adding knowledge graphs.

{{</citation>}}


### (33/43) Improving Text Embeddings with Large Language Models (Liang Wang et al., 2023)

{{<citation>}}

Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, Furu Wei. (2023)  
**Improving Text Embeddings with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-IR, cs.CL  
Keywords: Embedding, Language Model  
[Paper Link](http://arxiv.org/abs/2401.00368v1)  

---


**ABSTRACT**  
In this paper, we introduce a novel and simple method for obtaining high-quality text embeddings using only synthetic data and less than 1k training steps. Unlike existing methods that often depend on multi-stage intermediate pre-training with billions of weakly-supervised text pairs, followed by fine-tuning with a few labeled datasets, our method does not require building complex training pipelines or relying on manually collected datasets that are often constrained by task diversity and language coverage. We leverage proprietary LLMs to generate diverse synthetic data for hundreds of thousands of text embedding tasks across nearly 100 languages. We then fine-tune open-source decoder-only LLMs on the synthetic data using standard contrastive loss. Experiments demonstrate that our method achieves strong performance on highly competitive text embedding benchmarks without using any labeled data. Furthermore, when fine-tuned with a mixture of synthetic and labeled data, our model sets new state-of-the-art results on the BEIR and MTEB benchmarks.

{{</citation>}}


## cs.CR (3)



### (34/43) KernelGPT: Enhanced Kernel Fuzzing via Large Language Models (Chenyuan Yang et al., 2023)

{{<citation>}}

Chenyuan Yang, Zijie Zhao, Lingming Zhang. (2023)  
**KernelGPT: Enhanced Kernel Fuzzing via Large Language Models**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs-SE, cs.CR  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2401.00563v1)  

---


**ABSTRACT**  
Bugs in operating system kernels can affect billions of devices and users all over the world. As a result, a large body of research has been focused on kernel fuzzing, i.e., automatically generating syscall (system call) sequences to detect potential kernel bugs or vulnerabilities. Syzkaller, one of the most widely studied kernel fuzzers, aims to generate valid syscall sequences based on predefined specifications written in syzlang, a domain-specific language for defining syscalls, their arguments, and the relationships between them. While there has been existing work trying to automate Syzkaller specification generation, this still remains largely manual work and a large number of important syscalls are still uncovered. In this paper, we propose KernelGPT, the first approach to automatically inferring Syzkaller specifications via Large Language Models (LLMs) for enhanced kernel fuzzing. Our basic insight is that LLMs have seen massive kernel code, documentation, and use cases during pre-training, and thus can automatically distill the necessary information for making valid syscalls. More specifically, KernelGPT leverages an iterative approach to automatically infer all the necessary specification components, and further leverages the validation feedback to repair/refine the initial specifications. Our preliminary results demonstrate that KernelGPT can help Syzkaller achieve higher coverage and find multiple previously unknown bugs. Moreover, we also received a request from the Syzkaller team to upstream specifications inferred by KernelGPT.

{{</citation>}}


### (35/43) Opening A Pandora's Box: Things You Should Know in the Era of Custom GPTs (Guanhong Tao et al., 2023)

{{<citation>}}

Guanhong Tao, Siyuan Cheng, Zhuo Zhang, Junmin Zhu, Guangyu Shen, Xiangyu Zhang. (2023)  
**Opening A Pandora's Box: Things You Should Know in the Era of Custom GPTs**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: AI, GPT  
[Paper Link](http://arxiv.org/abs/2401.00905v1)  

---


**ABSTRACT**  
The emergence of large language models (LLMs) has significantly accelerated the development of a wide range of applications across various fields. There is a growing trend in the construction of specialized platforms based on LLMs, such as the newly introduced custom GPTs by OpenAI. While custom GPTs provide various functionalities like web browsing and code execution, they also introduce significant security threats. In this paper, we conduct a comprehensive analysis of the security and privacy issues arising from the custom GPT platform. Our systematic examination categorizes potential attack scenarios into three threat models based on the role of the malicious actor, and identifies critical data exchange channels in custom GPTs. Utilizing the STRIDE threat modeling framework, we identify 26 potential attack vectors, with 19 being partially or fully validated in real-world settings. Our findings emphasize the urgent need for robust security and privacy measures in the custom GPT ecosystem, especially in light of the forthcoming launch of the official GPT store by OpenAI.

{{</citation>}}


### (36/43) Blockchain and Deep Learning-Based IDS for Securing SDN-Enabled Industrial IoT Environments (Samira Kamali Poorazad et al., 2023)

{{<citation>}}

Samira Kamali Poorazad, Chafika Benzaıd, Tarik Taleb. (2023)  
**Blockchain and Deep Learning-Based IDS for Securing SDN-Enabled Industrial IoT Environments**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-NI, cs.CR  
Keywords: Intrusion Detection  
[Paper Link](http://arxiv.org/abs/2401.00468v1)  

---


**ABSTRACT**  
The industrial Internet of Things (IIoT) involves the integration of Internet of Things (IoT) technologies into industrial settings. However, given the high sensitivity of the industry to the security of industrial control system networks and IIoT, the use of software-defined networking (SDN) technology can provide improved security and automation of communication processes. Despite this, the architecture of SDN can give rise to various security threats. Therefore, it is of paramount importance to consider the impact of these threats on SDN-based IIoT environments. Unlike previous research, which focused on security in IIoT and SDN architectures separately, we propose an integrated method including two components that work together seamlessly for better detecting and preventing security threats associated with SDN-based IIoT architectures. The two components consist in a convolutional neural network-based Intrusion Detection System (IDS) implemented as an SDN application and a Blockchain-based system (BS) to empower application layer and network layer security, respectively. A significant advantage of the proposed method lies in jointly minimizing the impact of attacks such as command injection and rule injection on SDN-based IIoT architecture layers. The proposed IDS exhibits superior classification accuracy in both binary and multiclass categories.

{{</citation>}}


## cs.SI (2)



### (37/43) Pack and Measure: An Effective Approach for Influence Propagation in Social Networks (Faisal N. Abu-Khzam et al., 2023)

{{<citation>}}

Faisal N. Abu-Khzam, Ghinwa Bou Matar, Sergio Thoumi. (2023)  
**Pack and Measure: An Effective Approach for Influence Propagation in Social Networks**  

---
Primary Category: cs.SI  
Categories: cs-AI, cs-DS, cs-SI, cs.SI  
Keywords: Social Network  
[Paper Link](http://arxiv.org/abs/2401.00525v1)  

---


**ABSTRACT**  
The Influence Maximization problem under the Independent Cascade model (IC) is considered. The problem asks for a minimal set of vertices to serve as "seed set" from which a maximum influence propagation is expected. New seed-set selection methods are introduced based on the notions of a $d$-packing and vertex centrality. In particular, we focus on selecting seed-vertices that are far apart and whose influence-values are the highest in their local communities. Our best results are achieved via an initial computation of a $d$-Packing followed by selecting either vertices of high degree or high centrality in their respective closed neighborhoods. This overall "Pack and Measure" approach proves highly effective as a seed selection method.

{{</citation>}}


### (38/43) Social-LLM: Modeling User Behavior at Scale using Language Models and Social Network Data (Julie Jiang et al., 2023)

{{<citation>}}

Julie Jiang, Emilio Ferrara. (2023)  
**Social-LLM: Modeling User Behavior at Scale using Language Models and Social Network Data**  

---
Primary Category: cs.SI  
Categories: cs-AI, cs-SI, cs.SI  
Keywords: Language Model, Social Network  
[Paper Link](http://arxiv.org/abs/2401.00893v1)  

---


**ABSTRACT**  
The proliferation of social network data has unlocked unprecedented opportunities for extensive, data-driven exploration of human behavior. The structural intricacies of social networks offer insights into various computational social science issues, particularly concerning social influence and information diffusion. However, modeling large-scale social network data comes with computational challenges. Though large language models make it easier than ever to model textual content, any advanced network representation methods struggle with scalability and efficient deployment to out-of-sample users. In response, we introduce a novel approach tailored for modeling social network data in user detection tasks. This innovative method integrates localized social network interactions with the capabilities of large language models. Operating under the premise of social network homophily, which posits that socially connected users share similarities, our approach is designed to address these challenges. We conduct a thorough evaluation of our method across seven real-world social network datasets, spanning a diverse range of topics and detection tasks, showcasing its applicability to advance research in computational social science.

{{</citation>}}


## physics.chem-ph (1)



### (39/43) Generating High-Precision Force Fields for Molecular Dynamics Simulations to Study Chemical Reaction Mechanisms using Molecular Configuration Transformer (Sihao Yuan et al., 2023)

{{<citation>}}

Sihao Yuan, Xu Han, Zhaoxin Xie, Cheng Fan, Yi Issac Yang, Yi Qin Gao. (2023)  
**Generating High-Precision Force Fields for Molecular Dynamics Simulations to Study Chemical Reaction Mechanisms using Molecular Configuration Transformer**  

---
Primary Category: physics.chem-ph  
Categories: cond-mat-soft, cs-AI, physics-chem-ph, physics.chem-ph  
Keywords: AI, Transformer  
[Paper Link](http://arxiv.org/abs/2401.00499v1)  

---


**ABSTRACT**  
Theoretical studies on chemical reaction mechanisms have been crucial in organic chemistry. Traditionally, calculating the manually constructed molecular conformations of transition states for chemical reactions using quantum chemical calculations is the most commonly used method. However, this way is heavily dependent on individual experience and chemical intuition. In our previous study, we proposed a research paradigm that uses enhanced sampling in QM/MM molecular dynamics simulations to study chemical reactions. This approach can directly simulate the entire process of a chemical reaction. However, the computational speed limits the use of high-precision potential energy functions for simulations. To address this issue, we present a scheme for training high-precision force fields for molecular modeling using our developed graph-neural-network-based molecular model, molecular configuration transformer. This potential energy function allows for highly accurate simulations at a low computational cost, leading to more precise calculations of the mechanism of chemical reactions. We have used this approach to study a Cope rearrangement reaction and a Carbonyl insertion reaction catalyzed by Manganese. This "AI+Physics" based simulation approach is expected to become a new trend in the theoretical study of organic chemical reaction mechanisms.

{{</citation>}}


## cs.SD (2)



### (40/43) E-chat: Emotion-sensitive Spoken Dialogue System with Large Language Models (Hongfei Xue et al., 2023)

{{<citation>}}

Hongfei Xue, Yuhao Liang, Bingshen Mu, Shiliang Zhang, Qian Chen, Lei Xie. (2023)  
**E-chat: Emotion-sensitive Spoken Dialogue System with Large Language Models**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Dialog, Dialogue, Language Model  
[Paper Link](http://arxiv.org/abs/2401.00475v1)  

---


**ABSTRACT**  
This study focuses on emotion-sensitive spoken dialogue in human-machine speech interaction. With the advancement of Large Language Models (LLMs), dialogue systems can handle multimodal data, including audio. Recent models have enhanced the understanding of complex audio signals through the integration of various audio events. However, they are unable to generate appropriate responses based on emotional speech. To address this, we introduce the Emotional chat Model (E-chat), a novel spoken dialogue system capable of comprehending and responding to emotions conveyed from speech. This model leverages an emotion embedding extracted by a speech encoder, combined with LLMs, enabling it to respond according to different emotional contexts. Additionally, we introduce the E-chat200 dataset, designed explicitly for emotion-sensitive spoken dialogue. In various evaluation metrics, E-chat consistently outperforms baseline LLMs, demonstrating its potential in emotional comprehension and human-machine interaction.

{{</citation>}}


### (41/43) Online Symbolic Music Alignment with Offline Reinforcement Learning (Silvan David Peter, 2023)

{{<citation>}}

Silvan David Peter. (2023)  
**Online Symbolic Music Alignment with Offline Reinforcement Learning**  

---
Primary Category: cs.SD  
Categories: cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.00466v1)  

---


**ABSTRACT**  
Symbolic Music Alignment is the process of matching performed MIDI notes to corresponding score notes. In this paper, we introduce a reinforcement learning (RL)-based online symbolic music alignment technique. The RL agent - an attention-based neural network - iteratively estimates the current score position from local score and performance contexts. For this symbolic alignment task, environment states can be sampled exhaustively and the reward is dense, rendering a formulation as a simplified offline RL problem straightforward. We evaluate the trained agent in three ways. First, in its capacity to identify correct score positions for sampled test contexts; second, as the core technique of a complete algorithm for symbolic online note-wise alignment; and finally, as a real-time symbolic score follower. We further investigate the pitch-based score and performance representations used as the agent's inputs. To this end, we develop a second model, a two-step Dynamic Time Warping (DTW)-based offline alignment algorithm leveraging the same input representation. The proposed model outperforms a state-of-the-art reference model of offline symbolic music alignment.

{{</citation>}}


## cs.SE (1)



### (42/43) Exploring the Need of Accessibility Education in the Software Industry: Insights from a Survey of Software Professionals in India (Parthasarathy P D et al., 2023)

{{<citation>}}

Parthasarathy P D, Swaroop Joshi. (2023)  
**Exploring the Need of Accessibility Education in the Software Industry: Insights from a Survey of Software Professionals in India**  

---
Primary Category: cs.SE  
Categories: K-3-2; K-4-2, cs-CY, cs-SE, cs.SE  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.00451v1)  

---


**ABSTRACT**  
A UserWay study in 2021 indicates that an annual global e-commerce revenue loss of approximately $16 billion can be attributed to inaccessible websites and applications. According to the 2023 WebAIM study, only 3.7% of the world's top one million website homepages are fully accessible. This shows that many software developers use poor coding practices that don't adhere to the Web Content Accessibility Guidelines (WCAG). This research centers on software professionals and their role in addressing accessibility. This work seeks to understand (a) who within the software development community actively practices accessibility, (b) when and how accessibility is considered in the software development lifecycle, (c) the various challenges encountered in building accessible software, and (d) the resources required by software professionals to enhance product accessibility. Our survey of 269 software professionals from India sheds light on the pressing need for accessibility education within the software industry. A substantial majority (69.9%, N=269) of respondents express the need for training materials, workshops, and bootcamps to enhance their accessibility skills. We present a list of actionable recommendations that can be implemented within the industry to promote accessibility awareness and skills. We also open source our raw data for further research, encouraging continued exploration in this domain.

{{</citation>}}


## cs.NI (1)



### (43/43) Deeper and Wider Networks for Performance Metrics Prediction in Communication Networks (Aijia Liu et al., 2023)

{{<citation>}}

Aijia Liu, Shiqing Liu, Xiaobing Pei. (2023)  
**Deeper and Wider Networks for Performance Metrics Prediction in Communication Networks**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI, eess-SP  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2401.00429v1)  

---


**ABSTRACT**  
In today's era, users have increasingly high expectations regarding the performance and efficiency of communication networks. Network operators aspire to achieve efficient network planning, operation, and optimization through Digital Twin Networks (DTN). The effectiveness of DTN heavily relies on the network model, with graph neural networks (GNN) playing a crucial role in network modeling. However, existing network modeling methods still lack a comprehensive understanding of communication networks. In this paper, we propose DWNet (Deeper and Wider Networks), a heterogeneous graph neural network modeling method based on data-driven approaches that aims to address end-to-end latency and jitter prediction in network models. This method stands out due to two distinctive features: firstly, it introduces deeper levels of state participation in the message passing process; secondly, it extensively integrates relevant features during the feature fusion process. Through experimental validation and evaluation, our model achieves higher prediction accuracy compared to previous research achievements, particularly when dealing with unseen network topologies during model training. Our model not only provides more accurate predictions but also demonstrates stronger generalization capabilities across diverse topological structures.

{{</citation>}}
