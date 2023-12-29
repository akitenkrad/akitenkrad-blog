---
draft: false
title: "arXiv @ 2023.12.27"
date: 2023-12-27
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.12.27"
    identifier: arxiv_20231227
    parent: 202312_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.IR (4)](#csir-4)
- [cs.CV (20)](#cscv-20)
- [cs.SD (3)](#cssd-3)
- [cs.SI (1)](#cssi-1)
- [cs.CL (13)](#cscl-13)
- [cs.RO (1)](#csro-1)
- [cs.NI (1)](#csni-1)
- [cs.DM (1)](#csdm-1)
- [cs.LG (5)](#cslg-5)
- [cs.AI (4)](#csai-4)
- [cs.SE (1)](#csse-1)
- [cs.MA (1)](#csma-1)
- [eess.SY (1)](#eesssy-1)
- [cs.MM (1)](#csmm-1)
- [cs.HC (1)](#cshc-1)

## cs.IR (4)



### (1/58) Adversarial Item Promotion on Visually-Aware Recommender Systems by Guided Diffusion (Lijian Chen et al., 2023)

{{<citation>}}

Lijian Chen, Wei Yuan, Tong Chen, Quoc Viet Hung Nguyen, Lizhen Cui, Hongzhi Yin. (2023)  
**Adversarial Item Promotion on Visually-Aware Recommender Systems by Guided Diffusion**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Amazon  
[Paper Link](http://arxiv.org/abs/2312.15826v1)  

---


**ABSTRACT**  
Visually-aware recommender systems have found widespread application in domains where visual elements significantly contribute to the inference of users' potential preferences. While the incorporation of visual information holds the promise of enhancing recommendation accuracy and alleviating the cold-start problem, it is essential to point out that the inclusion of item images may introduce substantial security challenges. Some existing works have shown that the item provider can manipulate item exposure rates to its advantage by constructing adversarial images. However, these works cannot reveal the real vulnerability of visually-aware recommender systems because (1) The generated adversarial images are markedly distorted, rendering them easily detectable by human observers; (2) The effectiveness of the attacks is inconsistent and even ineffective in some scenarios. To shed light on the real vulnerabilities of visually-aware recommender systems when confronted with adversarial images, this paper introduces a novel attack method, IPDGI (Item Promotion by Diffusion Generated Image). Specifically, IPDGI employs a guided diffusion model to generate adversarial samples designed to deceive visually-aware recommender systems. Taking advantage of accurately modeling benign images' distribution by diffusion models, the generated adversarial images have high fidelity with original images, ensuring the stealth of our IPDGI. To demonstrate the effectiveness of our proposed methods, we conduct extensive experiments on two commonly used e-commerce recommendation datasets (Amazon Beauty and Amazon Baby) with several typical visually-aware recommender systems. The experimental results show that our attack method has a significant improvement in both the performance of promoting the long-tailed (i.e., unpopular) items and the quality of generated adversarial images.

{{</citation>}}


### (2/58) Large Language Models are Not Stable Recommender Systems (Tianhui Ma et al., 2023)

{{<citation>}}

Tianhui Ma, Yuan Cheng, Hengshu Zhu, Hui Xiong. (2023)  
**Large Language Models are Not Stable Recommender Systems**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.15746v1)  

---


**ABSTRACT**  
With the significant successes of large language models (LLMs) in many natural language processing tasks, there is growing interest among researchers in exploring LLMs for novel recommender systems. However, we have observed that directly using LLMs as a recommender system is usually unstable due to its inherent position bias. To this end, we introduce exploratory research and find consistent patterns of positional bias in LLMs that influence the performance of recommendation across a range of scenarios. Then, we propose a Bayesian probabilistic framework, STELLA (Stable LLM for Recommendation), which involves a two-stage pipeline. During the first probing stage, we identify patterns in a transition matrix using a probing detection dataset. And in the second recommendation stage, a Bayesian strategy is employed to adjust the biased output of LLMs with an entropy indicator. Therefore, our framework can capitalize on existing pattern information to calibrate instability of LLMs, and enhance recommendation performance. Finally, extensive experiments clearly validate the effectiveness of our framework.

{{</citation>}}


### (3/58) Unlocking the Potential of Large Language Models for Explainable Recommendations (Yucong Luo et al., 2023)

{{<citation>}}

Yucong Luo, Mingyue Cheng, Hao Zhang, Junyu Lu, Qi Liu, Enhong Chen. (2023)  
**Unlocking the Potential of Large Language Models for Explainable Recommendations**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.15661v2)  

---


**ABSTRACT**  
Generating user-friendly explanations regarding why an item is recommended has become increasingly common, largely due to advances in language generation technology, which can enhance user trust and facilitate more informed decision-making when using online services. However, existing explainable recommendation systems focus on using small-size language models. It remains uncertain what impact replacing the explanation generator with the recently emerging large language models (LLMs) would have. Can we expect unprecedented results?   In this study, we propose LLMXRec, a simple yet effective two-stage explainable recommendation framework aimed at further boosting the explanation quality by employing LLMs. Unlike most existing LLM-based recommendation works, a key characteristic of LLMXRec is its emphasis on the close collaboration between previous recommender models and LLM-based explanation generators. Specifically, by adopting several key fine-tuning techniques, including parameter-efficient instructing tuning and personalized prompt techniques, controllable and fluent explanations can be well generated to achieve the goal of explanation recommendation. Most notably, we provide three different perspectives to evaluate the effectiveness of the explanations. Finally, we conduct extensive experiments over several benchmark recommender models and publicly available datasets. The experimental results not only yield positive results in terms of effectiveness and efficiency but also uncover some previously unknown outcomes. To facilitate further explorations in this area, the full code and detailed original results are open-sourced at https://anonymous.4open.science/r/LLM_rec_explanation-7028/

{{</citation>}}


### (4/58) Preliminary Study on Incremental Learning for Large Language Model-based Recommender Systems (Tianhao Shi et al., 2023)

{{<citation>}}

Tianhao Shi, Yang Zhang, Zhijian Xu, Chong Chen, Fuli Feng, Xiangnan He, Qi Tian. (2023)  
**Preliminary Study on Incremental Learning for Large Language Model-based Recommender Systems**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.15599v1)  

---


**ABSTRACT**  
Adapting Large Language Models for recommendation (LLM4Rec)has garnered substantial attention and demonstrated promising results. However, the challenges of practically deploying LLM4Rec are largely unexplored, with the need for incremental adaptation to evolving user preferences being a critical concern. Nevertheless, the suitability of traditional incremental learning within LLM4Rec remains ambiguous, given the unique characteristics of LLMs. In this study, we empirically evaluate the commonly used incremental learning strategies (full retraining and fine-tuning) for LLM4Rec. Surprisingly, neither approach leads to evident improvements in LLM4Rec's performance. Rather than directly dismissing the role of incremental learning, we ascribe this lack of anticipated performance improvement to the mismatch between the LLM4Recarchitecture and incremental learning: LLM4Rec employs a single adaptation module for learning recommendation, hampering its ability to simultaneously capture long-term and short-term user preferences in the incremental learning context. To validate this speculation, we develop a Long- and Short-term Adaptation-aware Tuning (LSAT) framework for LLM4Rec incremental learning. Instead of relying on a single adaptation module, LSAT utilizes two adaptation modules to separately learn long-term and short-term user preferences. Empirical results demonstrate that LSAT could enhance performance, validating our speculation.

{{</citation>}}


## cs.CV (20)



### (5/58) Comparative Analysis of Radiomic Features and Gene Expression Profiles in Histopathology Data Using Graph Neural Networks (Luis Carlos Rivera Monroy et al., 2023)

{{<citation>}}

Luis Carlos Rivera Monroy, Leonhard Rist, Martin Eberhardt, Christian Ostalecki, Andreas Bauer, Julio Vera, Katharina Breininger, Andreas Maier. (2023)  
**Comparative Analysis of Radiomic Features and Gene Expression Profiles in Histopathology Data Using Graph Neural Networks**  

---
Primary Category: cs.CV  
Categories: cs-CE, cs-CV, cs-LG, cs.CV  
Keywords: Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2312.15825v1)  

---


**ABSTRACT**  
This study leverages graph neural networks to integrate MELC data with Radiomic-extracted features for melanoma classification, focusing on cell-wise analysis. It assesses the effectiveness of gene expression profiles and Radiomic features, revealing that Radiomic features, particularly when combined with UMAP for dimensionality reduction, significantly enhance classification performance. Notably, using Radiomics contributes to increased diagnostic accuracy and computational efficiency, as it allows for the extraction of critical data from fewer stains, thereby reducing operational costs. This methodology marks an advancement in computational dermatology for melanoma cell classification, setting the stage for future research and potential developments.

{{</citation>}}


### (6/58) WebVLN: Vision-and-Language Navigation on Websites (Qi Chen et al., 2023)

{{<citation>}}

Qi Chen, Dileepa Pitawela, Chongyang Zhao, Gengze Zhou, Hsiang-Ting Chen, Qi Wu. (2023)  
**WebVLN: Vision-and-Language Navigation on Websites**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.15820v1)  

---


**ABSTRACT**  
Vision-and-Language Navigation (VLN) task aims to enable AI agents to accurately understand and follow natural language instructions to navigate through real-world environments, ultimately reaching specific target locations. We recognise a promising opportunity to extend VLN to a comparable navigation task that holds substantial significance in our daily lives, albeit within the virtual realm: navigating websites on the Internet. This paper proposes a new task named Vision-and-Language Navigation on Websites (WebVLN), where we use question-based instructions to train an agent, emulating how users naturally browse websites. Unlike the existing VLN task that only pays attention to vision and instruction (language), the WebVLN agent further considers underlying web-specific content like HTML, which could not be seen on the rendered web pages yet contains rich visual and textual information. Toward this goal, we contribute a dataset, WebVLN-v1, and introduce a novel approach called Website-aware VLN Network (WebVLN-Net), which is built upon the foundation of state-of-the-art VLN techniques. Experimental results show that WebVLN-Net outperforms current VLN and web-related navigation methods. We believe that the introduction of the new WebVLN task and its dataset will establish a new dimension within the VLN domain and contribute to the broader vision-and-language research community. The code is available at: https://github.com/WebVLN/WebVLN.

{{</citation>}}


### (7/58) Contrastive Learning-Based Framework for Sim-to-Real Mapping of Lidar Point Clouds in Autonomous Driving Systems (Hamed Haghighi et al., 2023)

{{<citation>}}

Hamed Haghighi, Mehrdad Dianati, Kurt Debattista, Valentina Donzella. (2023)  
**Contrastive Learning-Based Framework for Sim-to-Real Mapping of Lidar Point Clouds in Autonomous Driving Systems**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs-RO, cs.CV, eess-IV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2312.15817v1)  

---


**ABSTRACT**  
Perception sensor models are essential elements of automotive simulation environments; they also serve as powerful tools for creating synthetic datasets to train deep learning-based perception models. Developing realistic perception sensor models poses a significant challenge due to the large gap between simulated sensor data and real-world sensor outputs, known as the sim-to-real gap. To address this problem, learning-based models have emerged as promising solutions in recent years, with unparalleled potential to map low-fidelity simulated sensor data into highly realistic outputs. Motivated by this potential, this paper focuses on sim-to-real mapping of Lidar point clouds, a widely used perception sensor in automated driving systems. We introduce a novel Contrastive-Learning-based Sim-to-Real mapping framework, namely CLS2R, inspired by the recent advancements in image-to-image translation techniques. The proposed CLS2R framework employs a lossless representation of Lidar point clouds, considering all essential Lidar attributes such as depth, reflectance, and raydrop. We extensively evaluate the proposed framework, comparing it with state-of-the-art image-to-image translation methods using a diverse range of metrics to assess realness, faithfulness, and the impact on the performance of a downstream task. Our results show that CLS2R demonstrates superior performance across nearly all metrics. Source code is available at https://github.com/hamedhaghighi/CLS2R.git.

{{</citation>}}


### (8/58) MetaScript: Few-Shot Handwritten Chinese Content Generation via Generative Adversarial Networks (Xiangyuan Xue et al., 2023)

{{<citation>}}

Xiangyuan Xue, Kailing Wang, Jiazi Bu, Qirui Li, Zhiyuan Zhang. (2023)  
**MetaScript: Few-Shot Handwritten Chinese Content Generation via Generative Adversarial Networks**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2312.16251v1)  

---


**ABSTRACT**  
In this work, we propose MetaScript, a novel Chinese content generation system designed to address the diminishing presence of personal handwriting styles in the digital representation of Chinese characters. Our approach harnesses the power of few-shot learning to generate Chinese characters that not only retain the individual's unique handwriting style but also maintain the efficiency of digital typing. Trained on a diverse dataset of handwritten styles, MetaScript is adept at producing high-quality stylistic imitations from minimal style references and standard fonts. Our work demonstrates a practical solution to the challenges of digital typography in preserving the personal touch in written communication, particularly in the context of Chinese script. Notably, our system has demonstrated superior performance in various evaluations, including recognition accuracy, inception score, and Frechet inception distance. At the same time, the training conditions of our model are easy to meet and facilitate generalization to real applications.

{{</citation>}}


### (9/58) A Recipe for Scaling up Text-to-Video Generation with Text-free Videos (Xiang Wang et al., 2023)

{{<citation>}}

Xiang Wang, Shiwei Zhang, Hangjie Yuan, Zhiwu Qing, Biao Gong, Yingya Zhang, Yujun Shen, Changxin Gao, Nong Sang. (2023)  
**A Recipe for Scaling up Text-to-Video Generation with Text-free Videos**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.15770v1)  

---


**ABSTRACT**  
Diffusion-based text-to-video generation has witnessed impressive progress in the past year yet still falls behind text-to-image generation. One of the key reasons is the limited scale of publicly available data (e.g., 10M video-text pairs in WebVid10M vs. 5B image-text pairs in LAION), considering the high cost of video captioning. Instead, it could be far easier to collect unlabeled clips from video platforms like YouTube. Motivated by this, we come up with a novel text-to-video generation framework, termed TF-T2V, which can directly learn with text-free videos. The rationale behind is to separate the process of text decoding from that of temporal modeling. To this end, we employ a content branch and a motion branch, which are jointly optimized with weights shared. Following such a pipeline, we study the effect of doubling the scale of training set (i.e., video-only WebVid10M) with some randomly collected text-free videos and are encouraged to observe the performance improvement (FID from 9.67 to 8.19 and FVD from 484 to 441), demonstrating the scalability of our approach. We also find that our model could enjoy sustainable performance gain (FID from 8.19 to 7.64 and FVD from 441 to 366) after reintroducing some text labels for training. Finally, we validate the effectiveness and generalizability of our ideology on both native text-to-video generation and compositional video synthesis paradigms. Code and models will be publicly available at https://tf-t2v.github.io/.

{{</citation>}}


### (10/58) DI-V2X: Learning Domain-Invariant Representation for Vehicle-Infrastructure Collaborative 3D Object Detection (Li Xiang et al., 2023)

{{<citation>}}

Li Xiang, Junbo Yin, Wei Li, Cheng-Zhong Xu, Ruigang Yang, Jianbing Shen. (2023)  
**DI-V2X: Learning Domain-Invariant Representation for Vehicle-Infrastructure Collaborative 3D Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI, Object Detection  
[Paper Link](http://arxiv.org/abs/2312.15742v1)  

---


**ABSTRACT**  
Vehicle-to-Everything (V2X) collaborative perception has recently gained significant attention due to its capability to enhance scene understanding by integrating information from various agents, e.g., vehicles, and infrastructure. However, current works often treat the information from each agent equally, ignoring the inherent domain gap caused by the utilization of different LiDAR sensors of each agent, thus leading to suboptimal performance. In this paper, we propose DI-V2X, that aims to learn Domain-Invariant representations through a new distillation framework to mitigate the domain discrepancy in the context of V2X 3D object detection. DI-V2X comprises three essential components: a domain-mixing instance augmentation (DMA) module, a progressive domain-invariant distillation (PDD) module, and a domain-adaptive fusion (DAF) module. Specifically, DMA builds a domain-mixing 3D instance bank for the teacher and student models during training, resulting in aligned data representation. Next, PDD encourages the student models from different domains to gradually learn a domain-invariant feature representation towards the teacher, where the overlapping regions between agents are employed as guidance to facilitate the distillation process. Furthermore, DAF closes the domain gap between the students by incorporating calibration-aware domain-adaptive attention. Extensive experiments on the challenging DAIR-V2X and V2XSet benchmark datasets demonstrate DI-V2X achieves remarkable performance, outperforming all the previous V2X models. Code is available at https://github.com/Serenos/DI-V2X

{{</citation>}}


### (11/58) Adaptive FSS: A Novel Few-Shot Segmentation Framework via Prototype Enhancement (Jing Wang et al., 2023)

{{<citation>}}

Jing Wang, Jinagyun Li, Chen Chen, Yisi Zhang, Haoran Shen, Tianxiang Zhang. (2023)  
**Adaptive FSS: A Novel Few-Shot Segmentation Framework via Prototype Enhancement**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2312.15731v2)  

---


**ABSTRACT**  
The Few-Shot Segmentation (FSS) aims to accomplish the novel class segmentation task with a few annotated images. Current FSS research based on meta-learning focus on designing a complex interaction mechanism between the query and support feature. However, unlike humans who can rapidly learn new things from limited samples, the existing approach relies solely on fixed feature matching to tackle new tasks, lacking adaptability. In this paper, we propose a novel framework based on the adapter mechanism, namely Adaptive FSS, which can efficiently adapt the existing FSS model to the novel classes. In detail, we design the Prototype Adaptive Module (PAM), which utilizes accurate category information provided by the support set to derive class prototypes, enhancing class-specific information in the multi-stage representation. In addition, our approach is compatible with in diverse FSS methods with different backbones by simply inserting PAM between the layers of the encoder. Experiments demonstrate that our method effectively improves the performance of the FSS models (e.g., MSANet, HDMNet, FPTrans, and DCAMA) and achieve new state-of-the-art (SOTA) results (i.e., 72.4\% and 79.1\% mIoU on PASCAL-5$^i$ 1-shot and 5-shot settings, 52.7\% and 60.0\% mIoU on COCO-20$^i$ 1-shot and 5-shot settings). Our code can be available at https://github.com/jingw193/AdaptiveFSS.

{{</citation>}}


### (12/58) UniRef++: Segment Every Reference Object in Spatial and Temporal Spaces (Jiannan Wu et al., 2023)

{{<citation>}}

Jiannan Wu, Yi Jiang, Bin Yan, Huchuan Lu, Zehuan Yuan, Ping Luo. (2023)  
**UniRef++: Segment Every Reference Object in Spatial and Temporal Spaces**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.15715v1)  

---


**ABSTRACT**  
The reference-based object segmentation tasks, namely referring image segmentation (RIS), few-shot image segmentation (FSS), referring video object segmentation (RVOS), and video object segmentation (VOS), aim to segment a specific object by utilizing either language or annotated masks as references. Despite significant progress in each respective field, current methods are task-specifically designed and developed in different directions, which hinders the activation of multi-task capabilities for these tasks. In this work, we end the current fragmented situation and propose UniRef++ to unify the four reference-based object segmentation tasks with a single architecture. At the heart of our approach is the proposed UniFusion module which performs multiway-fusion for handling different tasks with respect to their specified references. And a unified Transformer architecture is then adopted for achieving instance-level segmentation. With the unified designs, UniRef++ can be jointly trained on a broad range of benchmarks and can flexibly complete multiple tasks at run-time by specifying the corresponding references. We evaluate our unified models on various benchmarks. Extensive experimental results indicate that our proposed UniRef++ achieves state-of-the-art performance on RIS and RVOS, and performs competitively on FSS and VOS with a parameter-shared network. Moreover, we showcase that the proposed UniFusion module could be easily incorporated into the current advanced foundation model SAM and obtain satisfactory results with parameter-efficient finetuning. Codes and models are available at \url{https://github.com/FoundationVision/UniRef}.

{{</citation>}}


### (13/58) Nighttime Person Re-Identification via Collaborative Enhancement Network with Multi-domain Learning (Andong Lu et al., 2023)

{{<citation>}}

Andong Lu, Tianrui Zha, Chenglong Li, Jin Tang, Xiaofeng Wang, Bin Luo. (2023)  
**Nighttime Person Re-Identification via Collaborative Enhancement Network with Multi-domain Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.16246v1)  

---


**ABSTRACT**  
Prevalent nighttime ReID methods typically combine relighting networks and ReID networks in a sequential manner, which not only restricts the ReID performance by the quality of relighting images, but also neglects the effective collaborative modeling between image relighting and person ReID tasks. To handle these problems, we propose a novel Collaborative Enhancement Network called CENet, which performs the multilevel feature interactions in a parallel framework, for nighttime person ReID. In particular, CENet is a parallel Transformer network, in which the designed parallel structure can avoid the impact of the quality of relighting images on ReID performance. To perform effective collaborative modeling between image relighting and person ReID tasks, we integrate the multilevel feature interactions in CENet. Specifically, we share the Transformer encoder to build the low-level feature interaction, and then perform the feature distillation to transfer the high-level features from image relighting to ReID. In addition, the sizes of existing real-world nighttime person ReID datasets are small, and large-scale synthetic ones exhibit substantial domain gaps with real-world data. To leverage both small-scale real-world and large-scale synthetic training data, we develop a multi-domain learning algorithm, which alternately utilizes both kinds of data to reduce the inter-domain difference in the training of CENet. Extensive experiments on two real nighttime datasets, \textit{Night600} and \textit{RGBNT201$_{rgb}$}, and a synthetic nighttime ReID dataset are conducted to validate the effectiveness of CENet. We will release the code and synthetic dataset.

{{</citation>}}


### (14/58) Three Heads Are Better Than One: Complementary Experts for Long-Tailed Semi-supervised Learning (Chengcheng Ma et al., 2023)

{{<citation>}}

Chengcheng Ma, Ismail Elezi, Jiankang Deng, Weiming Dong, Changsheng Xu. (2023)  
**Three Heads Are Better Than One: Complementary Experts for Long-Tailed Semi-supervised Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2312.15702v1)  

---


**ABSTRACT**  
We address the challenging problem of Long-Tailed Semi-Supervised Learning (LTSSL) where labeled data exhibit imbalanced class distribution and unlabeled data follow an unknown distribution. Unlike in balanced SSL, the generated pseudo-labels are skewed towards head classes, intensifying the training bias. Such a phenomenon is even amplified as more unlabeled data will be mislabeled as head classes when the class distribution of labeled and unlabeled datasets are mismatched. To solve this problem, we propose a novel method named ComPlementary Experts (CPE). Specifically, we train multiple experts to model various class distributions, each of them yielding high-quality pseudo-labels within one form of class distribution. Besides, we introduce Classwise Batch Normalization for CPE to avoid performance degradation caused by feature distribution mismatch between head and non-head classes. CPE achieves state-of-the-art performances on CIFAR-10-LT, CIFAR-100-LT, and STL-10-LT dataset benchmarks. For instance, on CIFAR-10-LT, CPE improves test accuracy by over >2.22% compared to baselines. Code is available at https://github.com/machengcheng2016/CPE-LTSSL.

{{</citation>}}


### (15/58) Word length-aware text spotting: Enhancing detection and recognition in dense text image (Hao Wang et al., 2023)

{{<citation>}}

Hao Wang, Huabing Zhou, Yanduo Zhang, Tao Lu, Jiayi Ma. (2023)  
**Word length-aware text spotting: Enhancing detection and recognition in dense text image**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.15690v1)  

---


**ABSTRACT**  
Scene text spotting is essential in various computer vision applications, enabling extracting and interpreting textual information from images. However, existing methods often neglect the spatial semantics of word images, leading to suboptimal detection recall rates for long and short words within long-tailed word length distributions that exist prominently in dense scenes. In this paper, we present WordLenSpotter, a novel word length-aware spotter for scene text image detection and recognition, improving the spotting capabilities for long and short words, particularly in the tail data of dense text images. We first design an image encoder equipped with a dilated convolutional fusion module to integrate multiscale text image features effectively. Then, leveraging the Transformer framework, we synergistically optimize text detection and recognition accuracy after iteratively refining text region image features using the word length prior. Specially, we design a Spatial Length Predictor module (SLP) using character count prior tailored to different word lengths to constrain the regions of interest effectively. Furthermore, we introduce a specialized word Length-aware Segmentation (LenSeg) proposal head, enhancing the network's capacity to capture the distinctive features of long and short terms within categories characterized by long-tailed distributions. Comprehensive experiments on public datasets and our dense text spotting dataset DSTD1500 demonstrate the superiority of our proposed methods, particularly in dense text image detection and recognition tasks involving long-tailed word length distributions encompassing a range of long and short words.

{{</citation>}}


### (16/58) Partial Fine-Tuning: A Successor to Full Fine-Tuning for Vision Transformers (Peng Ye et al., 2023)

{{<citation>}}

Peng Ye, Yongqi Huang, Chongjun Tu, Minglei Li, Tao Chen, Tong He, Wanli Ouyang. (2023)  
**Partial Fine-Tuning: A Successor to Full Fine-Tuning for Vision Transformers**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.15681v1)  

---


**ABSTRACT**  
Fine-tuning pre-trained foundation models has gained significant popularity in various research fields. Existing methods for fine-tuning can be roughly divided into two categories, namely Parameter-Efficient Fine-Tuning and High-Performance Fine-Tuning. The former aims at improving efficiency, while the latter focuses on enhancing performance. Beyond these methods, we demonstrate that Partial Fine-Tuning can be an innovative and promising direction capable of concurrently enhancing both efficiency and accuracy. We first validate eight manually-defined partial fine-tuning strategies across kinds of datasets and vision transformer architectures, and find that some partial fine-tuning strategies (e.g., ffn only or attention only) can achieve better performance with fewer tuned parameters than full fine-tuning, and selecting appropriate layers is critical to partial fine-tuning. Thus, we propose a novel fine-tuned angle metric to guide the selection of appropriate layers for partial fine-tuning, making it flexible to be adapted to various scenarios for more practicable partial fine-tuning. Additionally, we show that partial fine-tuning can serve as a new dimension for Model Soups, improving both the model performance and generalization with fewer tuned parameters. Comprehensive experiments on a wide range of datasets and models validate the great potential of partial fine-tuning.

{{</citation>}}


### (17/58) Merging Vision Transformers from Different Tasks and Domains (Peng Ye et al., 2023)

{{<citation>}}

Peng Ye, Chenyu Huang, Mingzhu Shen, Tao Chen, Yongqi Huang, Yuning Zhang, Wanli Ouyang. (2023)  
**Merging Vision Transformers from Different Tasks and Domains**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Attention, Embedding, NLP, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.16240v1)  

---


**ABSTRACT**  
This work targets to merge various Vision Transformers (ViTs) trained on different tasks (i.e., datasets with different object categories) or domains (i.e., datasets with the same categories but different environments) into one unified model, yielding still good performance on each task or domain. Previous model merging works focus on either CNNs or NLP models, leaving the ViTs merging research untouched. To fill this gap, we first explore and find that existing model merging methods cannot well handle the merging of the whole ViT models and still have improvement space. To enable the merging of the whole ViT, we propose a simple-but-effective gating network that can both merge all kinds of layers (e.g., Embedding, Norm, Attention, and MLP) and select the suitable classifier. Specifically, the gating network is trained by unlabeled datasets from all the tasks (domains), and predicts the probability of which task (domain) the input belongs to for merging the models during inference. To further boost the performance of the merged model, especially when the difficulty of merging tasks increases, we design a novel metric of model weight similarity, and utilize it to realize controllable and combined weight merging. Comprehensive experiments on kinds of newly established benchmarks, validate the superiority of the proposed ViT merging framework for different tasks and domains. Our method can even merge beyond 10 ViT models from different vision tasks with a negligible effect on the performance of each task.

{{</citation>}}


### (18/58) Open-Vocabulary Video Relation Extraction (Wentao Tian et al., 2023)

{{<citation>}}

Wentao Tian, Zheng Wang, Yuqian Fu, Jingjing Chen, Lechao Cheng. (2023)  
**Open-Vocabulary Video Relation Extraction**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Relation Extraction  
[Paper Link](http://arxiv.org/abs/2312.15670v1)  

---


**ABSTRACT**  
A comprehensive understanding of videos is inseparable from describing the action with its contextual action-object interactions. However, many current video understanding tasks prioritize general action classification and overlook the actors and relationships that shape the nature of the action, resulting in a superficial understanding of the action. Motivated by this, we introduce Open-vocabulary Video Relation Extraction (OVRE), a novel task that views action understanding through the lens of action-centric relation triplets. OVRE focuses on pairwise relations that take part in the action and describes these relation triplets with natural languages. Moreover, we curate the Moments-OVRE dataset, which comprises 180K videos with action-centric relation triplets, sourced from a multi-label action classification dataset. With Moments-OVRE, we further propose a crossmodal mapping model to generate relation triplets as a sequence. Finally, we benchmark existing cross-modal generation models on the new task of OVRE.

{{</citation>}}


### (19/58) IQAGPT: Image Quality Assessment with Vision-language and ChatGPT Models (Zhihao Chen et al., 2023)

{{<citation>}}

Zhihao Chen, Bin Hu, Chuang Niu, Tao Chen, Yuxin Li, Hongming Shan, Ge Wang. (2023)  
**IQAGPT: Image Quality Assessment with Vision-language and ChatGPT Models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: ChatGPT, GPT, GPT-4, QA  
[Paper Link](http://arxiv.org/abs/2312.15663v1)  

---


**ABSTRACT**  
Large language models (LLMs), such as ChatGPT, have demonstrated impressive capabilities in various tasks and attracted an increasing interest as a natural language interface across many domains. Recently, large vision-language models (VLMs) like BLIP-2 and GPT-4 have been intensively investigated, which learn rich vision-language correlation from image-text pairs. However, despite these developments, the application of LLMs and VLMs in image quality assessment (IQA), particularly in medical imaging, remains to be explored, which is valuable for objective performance evaluation and potential supplement or even replacement of radiologists' opinions. To this end, this paper introduces IQAGPT, an innovative image quality assessment system integrating an image quality captioning VLM with ChatGPT for generating quality scores and textual reports. First, we build a CT-IQA dataset for training and evaluation, comprising 1,000 CT slices with diverse quality levels professionally annotated. To better leverage the capabilities of LLMs, we convert annotated quality scores into semantically rich text descriptions using a prompt template. Second, we fine-tune the image quality captioning VLM on the CT-IQA dataset to generate quality descriptions. The captioning model fuses the image and text features through cross-modal attention. Third, based on the quality descriptions, users can talk with ChatGPT to rate image quality scores or produce a radiological quality report. Our preliminary results demonstrate the feasibility of assessing image quality with large models. Remarkably, our IQAGPT outperforms GPT-4 and CLIP-IQA, as well as the multi-task classification and regression models that solely rely on images.

{{</citation>}}


### (20/58) Lifting by Image -- Leveraging Image Cues for Accurate 3D Human Pose Estimation (Feng Zhou et al., 2023)

{{<citation>}}

Feng Zhou, Jianqin Yin, Peiyang Li. (2023)  
**Lifting by Image -- Leveraging Image Cues for Accurate 3D Human Pose Estimation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.15636v1)  

---


**ABSTRACT**  
The "lifting from 2D pose" method has been the dominant approach to 3D Human Pose Estimation (3DHPE) due to the powerful visual analysis ability of 2D pose estimators. Widely known, there exists a depth ambiguity problem when estimating solely from 2D pose, where one 2D pose can be mapped to multiple 3D poses. Intuitively, the rich semantic and texture information in images can contribute to a more accurate "lifting" procedure. Yet, existing research encounters two primary challenges. Firstly, the distribution of image data in 3D motion capture datasets is too narrow because of the laboratorial environment, which leads to poor generalization ability of methods trained with image information. Secondly, effective strategies for leveraging image information are lacking. In this paper, we give new insight into the cause of poor generalization problems and the effectiveness of image features. Based on that, we propose an advanced framework. Specifically, the framework consists of two stages. First, we enable the keypoints to query and select the beneficial features from all image patches. To reduce the keypoints attention to inconsequential background features, we design a novel Pose-guided Transformer Layer, which adaptively limits the updates to unimportant image patches. Then, through a designed Adaptive Feature Selection Module, we prune less significant image patches from the feature map. In the second stage, we allow the keypoints to further emphasize the retained critical image features. This progressive learning approach prevents further training on insignificant image features. Experimental results show that our model achieves state-of-the-art performance on both the Human3.6M dataset and the MPI-INF-3DHP dataset.

{{</citation>}}


### (21/58) MuLA-GAN: Multi-Level Attention GAN for Enhanced Underwater Visibility (Ahsan Baidar Bakht et al., 2023)

{{<citation>}}

Ahsan Baidar Bakht, Zikai Jia, Muhayy ud Din, Waseem Akram, Lyes Saad Soud, Lakmal Seneviratne, Defu Lin, Shaoming He, Irfan Hussain. (2023)  
**MuLA-GAN: Multi-Level Attention GAN for Enhanced Underwater Visibility**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.15633v1)  

---


**ABSTRACT**  
The underwater environment presents unique challenges, including color distortions, reduced contrast, and blurriness, hindering accurate analysis. In this work, we introduce MuLA-GAN, a novel approach that leverages the synergistic power of Generative Adversarial Networks (GANs) and Multi-Level Attention mechanisms for comprehensive underwater image enhancement. The integration of Multi-Level Attention within the GAN architecture significantly enhances the model's capacity to learn discriminative features crucial for precise image restoration. By selectively focusing on relevant spatial and multi-level features, our model excels in capturing and preserving intricate details in underwater imagery, essential for various applications. Extensive qualitative and quantitative analyses on diverse datasets, including UIEB test dataset, UIEB challenge dataset, U45, and UCCS dataset, highlight the superior performance of MuLA-GAN compared to existing state-of-the-art methods. Experimental evaluations on a specialized dataset tailored for bio-fouling and aquaculture applications demonstrate the model's robustness in challenging environmental conditions. On the UIEB test dataset, MuLA-GAN achieves exceptional PSNR (25.59) and SSIM (0.893) scores, surpassing Water-Net, the second-best model, with scores of 24.36 and 0.885, respectively. This work not only addresses a significant research gap in underwater image enhancement but also underscores the pivotal role of Multi-Level Attention in enhancing GANs, providing a novel and comprehensive framework for restoring underwater image quality.

{{</citation>}}


### (22/58) APTv2: Benchmarking Animal Pose Estimation and Tracking with a Large-scale Dataset and Beyond (Yuxiang Yang et al., 2023)

{{<citation>}}

Yuxiang Yang, Yingqi Deng, Yufei Xu, Jing Zhang. (2023)  
**APTv2: Benchmarking Animal Pose Estimation and Tracking with a Large-scale Dataset and Beyond**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.15612v1)  

---


**ABSTRACT**  
Animal Pose Estimation and Tracking (APT) is a critical task in detecting and monitoring the keypoints of animals across a series of video frames, which is essential for understanding animal behavior. Past works relating to animals have primarily focused on either animal tracking or single-frame animal pose estimation only, neglecting the integration of both aspects. The absence of comprehensive APT datasets inhibits the progression and evaluation of animal pose estimation and tracking methods based on videos, thereby constraining their real-world applications. To fill this gap, we introduce APTv2, the pioneering large-scale benchmark for animal pose estimation and tracking. APTv2 comprises 2,749 video clips filtered and collected from 30 distinct animal species. Each video clip includes 15 frames, culminating in a total of 41,235 frames. Following meticulous manual annotation and stringent verification, we provide high-quality keypoint and tracking annotations for a total of 84,611 animal instances, split into easy and hard subsets based on the number of instances that exists in the frame. With APTv2 as the foundation, we establish a simple baseline method named \posetrackmethodname and provide benchmarks for representative models across three tracks: (1) single-frame animal pose estimation track to evaluate both intra- and inter-domain transfer learning performance, (2) low-data transfer and generalization track to evaluate the inter-species domain generalization performance, and (3) animal pose tracking track. Our experimental results deliver key empirical insights, demonstrating that APTv2 serves as a valuable benchmark for animal pose estimation and tracking. It also presents new challenges and opportunities for future research. The code and dataset are released at \href{https://github.com/ViTAE-Transformer/APTv2}{https://github.com/ViTAE-Transformer/APTv2}.

{{</citation>}}


### (23/58) A Target Detection Algorithm in Traffic Scenes Based on Deep Reinforcement Learning (Xinyu Ren et al., 2023)

{{<citation>}}

Xinyu Ren, Ruixuan Wang. (2023)  
**A Target Detection Algorithm in Traffic Scenes Based on Deep Reinforcement Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: LSTM, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.15606v1)  

---


**ABSTRACT**  
This research presents a novel active detection model utilizing deep reinforcement learning to accurately detect traffic objects in real-world scenarios. The model employs a deep Q-network based on LSTM-CNN that identifies and aligns target zones with specific categories of traffic objects through implementing a top-down approach with efficient feature extraction of the environment. The model integrates historical and current actions and observations to make a comprehensive analysis. The design of the state space and reward function takes into account the impact of time steps to enable the model to complete the task in fewer steps. Tests conducted demonstrate the model's proficiency, exhibiting exceptional precision and performance in locating traffic signal lights and speed limit signs. The findings of this study highlight the efficacy and potential of the deep reinforcement learning-based active detection model in traffic-related applications, underscoring its robust detection abilities and promising performance.

{{</citation>}}


### (24/58) Deep Structure and Attention Aware Subspace Clustering (Wenhao Wu et al., 2023)

{{<citation>}}

Wenhao Wu, Weiwei Wang, Shengjiang Kong. (2023)  
**Deep Structure and Attention Aware Subspace Clustering**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.15577v1)  

---


**ABSTRACT**  
Clustering is a fundamental unsupervised representation learning task with wide application in computer vision and pattern recognition. Deep clustering utilizes deep neural networks to learn latent representation, which is suitable for clustering. However, previous deep clustering methods, especially image clustering, focus on the features of the data itself and ignore the relationship between the data, which is crucial for clustering. In this paper, we propose a novel Deep Structure and Attention aware Subspace Clustering (DSASC), which simultaneously considers data content and structure information. We use a vision transformer to extract features, and the extracted features are divided into two parts, structure features, and content features. The two features are used to learn a more efficient subspace structure for spectral clustering. Extensive experimental results demonstrate that our method significantly outperforms state-of-the-art methods. Our code will be available at https://github.com/cs-whh/DSASC

{{</citation>}}


## cs.SD (3)



### (25/58) Self-Supervised Learning for Few-Shot Bird Sound Classification (Ilyass Moummad et al., 2023)

{{<citation>}}

Ilyass Moummad, Romain Serizel, Nicolas Farrugia. (2023)  
**Self-Supervised Learning for Few-Shot Bird Sound Classification**  

---
Primary Category: cs.SD  
Categories: cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Few-Shot, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2312.15824v2)  

---


**ABSTRACT**  
Self-supervised learning (SSL) in audio holds significant potential across various domains, particularly in situations where abundant, unlabeled data is readily available at no cost. This is particularly pertinent in bioacoustics, where biologists routinely collect extensive sound datasets from the natural environment. In this study, we demonstrate that SSL is capable of acquiring meaningful representations of bird sounds from audio recordings without the need for annotations. Our experiments showcase that these learned representations exhibit the capacity to generalize to new bird species in few-shot learning (FSL) scenarios. Additionally, we show that selecting windows with high bird activation for self-supervised learning, using a pretrained audio neural network, significantly enhances the quality of the learned representations.

{{</citation>}}


### (26/58) Uncertainty as a Predictor: Leveraging Self-Supervised Learning for Zero-Shot MOS Prediction (Aditya Ravuri et al., 2023)

{{<citation>}}

Aditya Ravuri, Erica Cooper, Junichi Yamagishi. (2023)  
**Uncertainty as a Predictor: Leveraging Self-Supervised Learning for Zero-Shot MOS Prediction**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS, stat-ML  
Keywords: Self-Supervised, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2312.15616v1)  

---


**ABSTRACT**  
Predicting audio quality in voice synthesis and conversion systems is a critical yet challenging task, especially when traditional methods like Mean Opinion Scores (MOS) are cumbersome to collect at scale. This paper addresses the gap in efficient audio quality prediction, especially in low-resource settings where extensive MOS data from large-scale listening tests may be unavailable. We demonstrate that uncertainty measures derived from out-of-the-box pretrained self-supervised learning (SSL) models, such as wav2vec, correlate with MOS scores. These findings are based on data from the 2022 and 2023 VoiceMOS challenges. We explore the extent of this correlation across different models and language contexts, revealing insights into how inherent uncertainties in SSL models can serve as effective proxies for audio quality assessment. In particular, we show that the contrastive wav2vec models are the most performant in all settings.

{{</citation>}}


### (27/58) DSNet: Disentangled Siamese Network with Neutral Calibration for Speech Emotion Recognition (Chengxin Chen et al., 2023)

{{<citation>}}

Chengxin Chen, Pengyuan Zhang. (2023)  
**DSNet: Disentangled Siamese Network with Neutral Calibration for Speech Emotion Recognition**  

---
Primary Category: cs.SD  
Categories: cs-AI, cs-SD, cs.SD, eess-AS  
Keywords: Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2312.15593v1)  

---


**ABSTRACT**  
One persistent challenge in deep learning based speech emotion recognition (SER) is the unconscious encoding of emotion-irrelevant factors (e.g., speaker or phonetic variability), which limits the generalization of SER in practical use. In this paper, we propose DSNet, a Disentangled Siamese Network with neutral calibration, to meet the demand for a more robust and explainable SER model. Specifically, we introduce an orthogonal feature disentanglement module to explicitly project the high-level representation into two distinct subspaces. Later, we propose a novel neutral calibration mechanism to encourage one subspace to capture sufficient emotion-irrelevant information. In this way, the other one can better isolate and emphasize the emotion-relevant information within speech signals. Experimental results on two popular benchmark datasets demonstrate the superiority of DSNet over various state-of-the-art methods for speaker-independent SER.

{{</citation>}}


## cs.SI (1)



### (28/58) Viral Marketing in Social Networks with Competing Products (Ahad N. Zehmakan et al., 2023)

{{<citation>}}

Ahad N. Zehmakan, Xiaotian Zhou, Zhongzhi Zhang. (2023)  
**Viral Marketing in Social Networks with Competing Products**  

---
Primary Category: cs.SI  
Categories: cs-AI, cs-DS, cs-SI, cs.SI  
Keywords: Social Network  
[Paper Link](http://arxiv.org/abs/2312.15819v1)  

---


**ABSTRACT**  
Consider a directed network where each node is either red (using the red product), blue (using the blue product), or uncolored (undecided). Then in each round, an uncolored node chooses red (resp. blue) with some probability proportional to the number of its red (resp. blue) out-neighbors. What is the best strategy to maximize the expected final number of red nodes given the budget to select $k$ red seed nodes? After proving that this problem is computationally hard, we provide a polynomial time approximation algorithm with the best possible approximation guarantee, building on the monotonicity and submodularity of the objective function and exploiting the Monte Carlo method. Furthermore, our experiments on various real-world and synthetic networks demonstrate that our proposed algorithm outperforms other algorithms. Additionally, we investigate the convergence time of the aforementioned process both theoretically and experimentally. In particular, we prove several tight bounds on the convergence time in terms of different graph parameters, such as the number of nodes/edges, maximum out-degree and diameter, by developing novel proof techniques.

{{</citation>}}


## cs.CL (13)



### (29/58) TEILP: Time Prediction over Knowledge Graphs via Logical Reasoning (Siheng Xiong et al., 2023)

{{<citation>}}

Siheng Xiong, Yuan Yang, Ali Payani, James C Kerce, Faramarz Fekri. (2023)  
**TEILP: Time Prediction over Knowledge Graphs via Logical Reasoning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Knowledge Graph, Reasoning  
[Paper Link](http://arxiv.org/abs/2312.15816v1)  

---


**ABSTRACT**  
Conventional embedding-based models approach event time prediction in temporal knowledge graphs (TKGs) as a ranking problem. However, they often fall short in capturing essential temporal relationships such as order and distance. In this paper, we propose TEILP, a logical reasoning framework that naturaly integrates such temporal elements into knowledge graph predictions. We first convert TKGs into a temporal event knowledge graph (TEKG) which has a more explicit representation of time in term of nodes of the graph. The TEKG equips us to develop a differentiable random walk approach to time prediction. Finally, we introduce conditional probability density functions, associated with the logical rules involving the query interval, using which we arrive at the time prediction. We compare TEILP with state-of-the-art methods on five benchmark datasets. We show that our model achieves a significant improvement over baselines while providing interpretable explanations. In particular, we consider several scenarios where training samples are limited, event types are imbalanced, and forecasting the time of future events based on only past events is desired. In all these cases, TEILP outperforms state-of-the-art methods in terms of robustness.

{{</citation>}}


### (30/58) Compositional Generalization in Spoken Language Understanding (Avik Ray et al., 2023)

{{<citation>}}

Avik Ray, Yilin Shen, Hongxia Jin. (2023)  
**Compositional Generalization in Spoken Language Understanding**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, Spoken Language Understanding  
[Paper Link](http://arxiv.org/abs/2312.15815v1)  

---


**ABSTRACT**  
State-of-the-art spoken language understanding (SLU) models have shown tremendous success in benchmark SLU datasets, yet they still fail in many practical scenario due to the lack of model compositionality when trained on limited training data. In this paper, we study two types of compositionality: (a) novel slot combination, and (b) length generalization. We first conduct in-depth analysis, and find that state-of-the-art SLU models often learn spurious slot correlations during training, which leads to poor performance in both compositional cases. To mitigate these limitations, we create the first compositional splits of benchmark SLU datasets and we propose the first compositional SLU model, including compositional loss and paired training that tackle each compositional case respectively. On both benchmark and compositional splits in ATIS and SNIPS, we show that our compositional SLU model significantly outperforms (up to $5\%$ F1 score) state-of-the-art BERT SLU model.

{{</citation>}}


### (31/58) AHAM: Adapt, Help, Ask, Model -- Harvesting LLMs for literature mining (Boshko Koloski et al., 2023)

{{<citation>}}

Boshko Koloski, Nada Lavrač, Bojan Cestnik, Senja Pollak, Blaž Škrlj, Andrej Kastrin. (2023)  
**AHAM: Adapt, Help, Ask, Model -- Harvesting LLMs for literature mining**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2312.15784v1)  

---


**ABSTRACT**  
In an era marked by a rapid increase in scientific publications, researchers grapple with the challenge of keeping pace with field-specific advances. We present the `AHAM' methodology and a metric that guides the domain-specific \textbf{adapt}ation of the BERTopic topic modeling framework to improve scientific text analysis. By utilizing the LLaMa2 generative language model, we generate topic definitions via one-shot learning by crafting prompts with the \textbf{help} of domain experts to guide the LLM for literature mining by \textbf{asking} it to model the topic names. For inter-topic similarity evaluation, we leverage metrics from language generation and translation processes to assess lexical and semantic similarity of the generated topics. Our system aims to reduce both the ratio of outlier topics to the total number of topics and the similarity between topic definitions. The methodology has been assessed on a newly gathered corpus of scientific papers on literature-based discovery. Through rigorous evaluation by domain experts, AHAM has been validated as effective in uncovering intriguing and novel insights within broad research areas. We explore the impact of domain adaptation of sentence-transformers for the task of topic \textbf{model}ing using two datasets, each specialized to specific scientific domains within arXiv and medarxiv. We evaluate the impact of data size, the niche of adaptation, and the importance of domain adaptation. Our results suggest a strong interaction between domain adaptation and topic modeling precision in terms of outliers and topic definitions.

{{</citation>}}


### (32/58) Design and Implementation of a Tool for Extracting Uzbek Syllables (Ulugbek Salaev et al., 2023)

{{<citation>}}

Ulugbek Salaev, Elmurod Kuriyozov, Gayrat Matlatipov. (2023)  
**Design and Implementation of a Tool for Extracting Uzbek Syllables**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2312.15779v1)  

---


**ABSTRACT**  
The accurate syllabification of words plays a vital role in various Natural Language Processing applications. Syllabification is a versatile linguistic tool with applications in linguistic research, language technology, education, and various fields where understanding and processing language is essential. In this paper, we present a comprehensive approach to syllabification for the Uzbek language, including rule-based techniques and machine learning algorithms. Our rule-based approach utilizes advanced methods for dividing words into syllables, generating hyphenations for line breaks and count of syllables. Additionally, we collected a dataset for evaluating and training using machine learning algorithms comprising word-syllable mappings, hyphenations, and syllable counts to predict syllable counts as well as for the evaluation of the proposed model. Our results demonstrate the effectiveness and efficiency of both approaches in achieving accurate syllabification. The results of our experiments show that both approaches achieved a high level of accuracy, exceeding 99%. This study provides valuable insights and recommendations for future research on syllabification and related areas in not only the Uzbek language itself, but also in other closely-related Turkic languages with low-resource factor.

{{</citation>}}


### (33/58) Solving Label Variation in Scientific Information Extraction via Multi-Task Learning (Dong Pham et al., 2023)

{{<citation>}}

Dong Pham, Xanh Ho, Quang-Thuy Ha, Akiko Aizawa. (2023)  
**Solving Label Variation in Scientific Information Extraction via Multi-Task Learning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Information Extraction  
[Paper Link](http://arxiv.org/abs/2312.15751v1)  

---


**ABSTRACT**  
Scientific Information Extraction (ScientificIE) is a critical task that involves the identification of scientific entities and their relationships. The complexity of this task is compounded by the necessity for domain-specific knowledge and the limited availability of annotated data. Two of the most popular datasets for ScientificIE are SemEval-2018 Task-7 and SciERC. They have overlapping samples and differ in their annotation schemes, which leads to conflicts. In this study, we first introduced a novel approach based on multi-task learning to address label variations. We then proposed a soft labeling technique that converts inconsistent labels into probabilistic distributions. The experimental results demonstrated that the proposed method can enhance the model robustness to label noise and improve the end-to-end performance in both ScientificIE tasks. The analysis revealed that label variations can be particularly effective in handling ambiguous instances. Furthermore, the richness of the information captured by label variations can potentially reduce data size requirements. The findings highlight the importance of releasing variation labels and promote future research on other tasks in other domains. Overall, this study demonstrates the effectiveness of multi-task learning and the potential of label variations to enhance the performance of ScientificIE.

{{</citation>}}


### (34/58) PersianLLaMA: Towards Building First Persian Large Language Model (Mohammad Amin Abbasi et al., 2023)

{{<citation>}}

Mohammad Amin Abbasi, Arash Ghafouri, Mahdi Firouzmandi, Hassan Naderi, Behrouz Minaei Bidgoli. (2023)  
**PersianLLaMA: Towards Building First Persian Large Language Model**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2312.15713v1)  

---


**ABSTRACT**  
Despite the widespread use of the Persian language by millions globally, limited efforts have been made in natural language processing for this language. The use of large language models as effective tools in various natural language processing tasks typically requires extensive textual data and robust hardware resources. Consequently, the scarcity of Persian textual data and the unavailability of powerful hardware resources have hindered the development of large language models for Persian. This paper introduces the first large Persian language model, named PersianLLaMA, trained on a collection of Persian texts and datasets. This foundational model comes in two versions, with 7 and 13 billion parameters, trained on formal and colloquial Persian texts using two different approaches. PersianLLaMA has been evaluated for natural language generation tasks based on the latest evaluation methods, namely using larger language models, and for natural language understanding tasks based on automated machine metrics. The results indicate that PersianLLaMA significantly outperforms its competitors in both understanding and generating Persian text. PersianLLaMA marks an important step in the development of Persian natural language processing and can be a valuable resource for the Persian-speaking community. This large language model can be used for various natural language processing tasks, especially text generation like chatbots, question-answering, machine translation, and text summarization

{{</citation>}}


### (35/58) Alleviating Hallucinations of Large Language Models through Induced Hallucinations (Yue Zhang et al., 2023)

{{<citation>}}

Yue Zhang, Leyang Cui, Wei Bi, Shuming Shi. (2023)  
**Alleviating Hallucinations of Large Language Models through Induced Hallucinations**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model, QA  
[Paper Link](http://arxiv.org/abs/2312.15710v1)  

---


**ABSTRACT**  
Despite their impressive capabilities, large language models (LLMs) have been observed to generate responses that include inaccurate or fabricated information, a phenomenon commonly known as ``hallucination''. In this work, we propose a simple \textit{Induce-then-Contrast} Decoding (ICD) strategy to alleviate hallucinations. We first construct a factually weak LLM by inducing hallucinations from the original LLMs. Then, we penalize these induced hallucinations during decoding to enhance the factuality of the generated content. Concretely, we determine the final next-token predictions by amplifying the predictions from the original model and downplaying the induced untruthful predictions via contrastive decoding. Experimental results on both discrimination-based and generation-based hallucination evaluation benchmarks, such as TruthfulQA and \textsc{FActScore}, demonstrate that our proposed ICD methods can effectively enhance the factuality of LLMs across various model sizes and families. For example, when equipped with ICD, Llama2-7B-Chat and Mistral-7B-Instruct achieve performance comparable to ChatGPT and GPT4 on TruthfulQA, respectively.

{{</citation>}}


### (36/58) EcomGPT-CT: Continual Pre-training of E-commerce Large Language Models with Semi-structured Data (Shirong Ma et al., 2023)

{{<citation>}}

Shirong Ma, Shen Huang, Shulin Huang, Xiaobin Wang, Yangning Li, Hai-Tao Zheng, Pengjun Xie, Fei Huang, Yong Jiang. (2023)  
**EcomGPT-CT: Continual Pre-training of E-commerce Large Language Models with Semi-structured Data**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2312.15696v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) pre-trained on massive corpora have exhibited remarkable performance on various NLP tasks. However, applying these models to specific domains still poses significant challenges, such as lack of domain knowledge, limited capacity to leverage domain knowledge and inadequate adaptation to domain-specific data formats. Considering the exorbitant cost of training LLMs from scratch and the scarcity of annotated data within particular domains, in this work, we focus on domain-specific continual pre-training of LLMs using E-commerce domain as an exemplar. Specifically, we explore the impact of continual pre-training on LLMs employing unlabeled general and E-commercial corpora. Furthermore, we design a mixing strategy among different data sources to better leverage E-commercial semi-structured data. We construct multiple tasks to assess LLMs' few-shot In-context Learning ability and their zero-shot performance after instruction tuning in E-commerce domain. Experimental results demonstrate the effectiveness of continual pre-training of E-commerce LLMs and the efficacy of our devised data mixing strategy.

{{</citation>}}


### (37/58) What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning (Wei Liu et al., 2023)

{{<citation>}}

Wei Liu, Weihao Zeng, Keqing He, Yong Jiang, Junxian He. (2023)  
**What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: LLaMA  
[Paper Link](http://arxiv.org/abs/2312.15685v1)  

---


**ABSTRACT**  
Instruction tuning is a standard technique employed to align large language models to end tasks and user preferences after the initial pretraining phase. Recent research indicates the critical role of data engineering in instruction tuning -- when appropriately selected, only limited data is necessary to achieve superior performance. However, we still lack a principled understanding of what makes good instruction tuning data for alignment, and how we should select data automatically and effectively. In this work, we delve deeply into automatic data selection strategies for alignment. We start with controlled studies to measure data across three dimensions: complexity, quality, and diversity, along which we examine existing methods and introduce novel techniques for enhanced data measurement. Subsequently, we propose a simple strategy to select data samples based on the measurement. We present deita (short for Data-Efficient Instruction Tuning for Alignment), a series of models fine-tuned from LLaMA and Mistral models using data samples automatically selected with our proposed approach. Empirically, deita performs better or on par with the state-of-the-art open-source alignment models with only 6K SFT training data samples -- over 10x less than the data used in the baselines. When further trained with direct preference optimization (DPO), deita-Mistral-7B + DPO trained with 6K SFT and 10K DPO samples achieve 7.55 MT-Bench and 90.06% AlpacaEval scores. We anticipate this work to provide tools on automatic data selection, facilitating data-efficient alignment. We release our models as well as the selected datasets for future researches to effectively align models more efficiently.

{{</citation>}}


### (38/58) Conditional Variational Autoencoder for Sign Language Translation with Cross-Modal Alignment (Rui Zhao et al., 2023)

{{<citation>}}

Rui Zhao, Liang Zhang, Biao Fu, Cong Hu, Jinsong Su, Yidong Chen. (2023)  
**Conditional Variational Autoencoder for Sign Language Translation with Cross-Modal Alignment**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.15645v1)  

---


**ABSTRACT**  
Sign language translation (SLT) aims to convert continuous sign language videos into textual sentences. As a typical multi-modal task, there exists an inherent modality gap between sign language videos and spoken language text, which makes the cross-modal alignment between visual and textual modalities crucial. However, previous studies tend to rely on an intermediate sign gloss representation to help alleviate the cross-modal problem thereby neglecting the alignment across modalities that may lead to compromised results. To address this issue, we propose a novel framework based on Conditional Variational autoencoder for SLT (CV-SLT) that facilitates direct and sufficient cross-modal alignment between sign language videos and spoken language text. Specifically, our CV-SLT consists of two paths with two Kullback-Leibler (KL) divergences to regularize the outputs of the encoder and decoder, respectively. In the prior path, the model solely relies on visual information to predict the target text; whereas in the posterior path, it simultaneously encodes visual information and textual knowledge to reconstruct the target text. The first KL divergence optimizes the conditional variational autoencoder and regularizes the encoder outputs, while the second KL divergence performs a self-distillation from the posterior path to the prior path, ensuring the consistency of decoder outputs. We further enhance the integration of textual information to the posterior path by employing a shared Attention Residual Gaussian Distribution (ARGD), which considers the textual information in the posterior path as a residual component relative to the prior path. Extensive experiments conducted on public datasets (PHOENIX14T and CSL-daily) demonstrate the effectiveness of our framework, achieving new state-of-the-art results while significantly alleviating the cross-modal representation discrepancy.

{{</citation>}}


### (39/58) A Split-and-Privatize Framework for Large Language Model Fine-Tuning (Xicong Shen et al., 2023)

{{<citation>}}

Xicong Shen, Yang Liu, Huiqi Liu, Jue Hong, Bing Duan, Zirui Huang, Yunlong Mao, Ye Wu, Di Wu. (2023)  
**A Split-and-Privatize Framework for Large Language Model Fine-Tuning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.15603v1)  

---


**ABSTRACT**  
Fine-tuning is a prominent technique to adapt a pre-trained language model to downstream scenarios. In parameter-efficient fine-tuning, only a small subset of modules are trained over the downstream datasets, while leaving the rest of the pre-trained model frozen to save computation resources. In recent years, a popular productization form arises as Model-as-a-Service (MaaS), in which vendors provide abundant pre-trained language models, server resources and core functions, and customers can fine-tune, deploy and invoke their customized model by accessing the one-stop MaaS with their own private dataset. In this paper, we identify the model and data privacy leakage risks in MaaS fine-tuning, and propose a Split-and-Privatize (SAP) framework, which manage to mitigate the privacy issues by adapting the existing split learning architecture. The proposed SAP framework is sufficiently investigated by experiments, and the results indicate that it can enhance the empirical privacy by 62% at the cost of 1% model performance degradation on the Stanford Sentiment Treebank dataset.

{{</citation>}}


### (40/58) Chatbot is Not All You Need: Information-rich Prompting for More Realistic Responses (Seokhoon Jeong et al., 2023)

{{<citation>}}

Seokhoon Jeong, Assentay Makhmud. (2023)  
**Chatbot is Not All You Need: Information-rich Prompting for More Realistic Responses**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.16233v1)  

---


**ABSTRACT**  
Recent Large Language Models (LLMs) have shown remarkable capabilities in mimicking fictional characters or real humans in conversational settings. However, the realism and consistency of these responses can be further enhanced by providing richer information of the agent being mimicked. In this paper, we propose a novel approach to generate more realistic and consistent responses from LLMs, leveraging five senses, attributes, emotional states, relationship with the interlocutor, and memories. By incorporating these factors, we aim to increase the LLM's capacity for generating natural and realistic reactions in conversational exchanges. Through our research, we expect to contribute to the development of LLMs that demonstrate improved capabilities in mimicking fictional characters. We release a new benchmark dataset and all our codes, prompts, and sample results on our Github: https://github.com/srafsasm/InfoRichBot

{{</citation>}}


### (41/58) Reducing LLM Hallucinations using Epistemic Neural Networks (Shreyas Verma et al., 2023)

{{<citation>}}

Shreyas Verma, Kien Tran, Yusuf Ali, Guangyu Min. (2023)  
**Reducing LLM Hallucinations using Epistemic Neural Networks**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2312.15576v1)  

---


**ABSTRACT**  
Reducing and detecting hallucinations in large language models is an open research problem. In this project, we attempt to leverage recent advances in the field of uncertainty estimation to reduce hallucinations in frozen large language models. Epistemic neural networks have recently been proposed to improve output joint distributions for large pre-trained models. ENNs are small networks attached to large, frozen models to improve the model's joint distributions and uncertainty estimates. In this work, we train an epistemic neural network on top of the Llama-2 7B model combined with a contrastive decoding feature enhancement technique. We are the first to train an ENN for the next token prediction task and explore the efficacy of this method in reducing hallucinations on the TruthfulQA dataset. In essence, we provide a method that leverages a pre-trained model's latent embeddings to reduce hallucinations.

{{</citation>}}


## cs.RO (1)



### (42/58) A Closed-Loop Multi-perspective Visual Servoing Approach with Reinforcement Learning (Lei Zhang et al., 2023)

{{<citation>}}

Lei Zhang, Jiacheng Pei, Kaixin Bai, Zhaopeng Chen, Jianwei Zhang. (2023)  
**A Closed-Loop Multi-perspective Visual Servoing Approach with Reinforcement Learning**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.15809v1)  

---


**ABSTRACT**  
Traditional visual servoing methods suffer from serving between scenes from multiple perspectives, which humans can complete with visual signals alone. In this paper, we investigated how multi-perspective visual servoing could be solved under robot-specific constraints, including self-collision, singularity problems. We presented a novel learning-based multi-perspective visual servoing framework, which iteratively estimates robot actions from latent space representations of visual states using reinforcement learning. Furthermore, our approaches were trained and validated in a Gazebo simulation environment with connection to OpenAI/Gym. Through simulation experiments, we showed that our method can successfully learn an optimal control policy given initial images from different perspectives, and it outperformed the Direct Visual Servoing algorithm with mean success rate of 97.0%.

{{</citation>}}


## cs.NI (1)



### (43/58) Quantum-Assisted Online Task Offloading and Resource Allocation in MEC-Enabled Satellite-Aerial-Terrestrial Integrated Networks (Yu Zhang et al., 2023)

{{<citation>}}

Yu Zhang, Yanmin Gong, Lei Fan, Yu Wang, Zhu Han, Yuanxiong Guo. (2023)  
**Quantum-Assisted Online Task Offloading and Resource Allocation in MEC-Enabled Satellite-Aerial-Terrestrial Integrated Networks**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2312.15808v1)  

---


**ABSTRACT**  
In the era of Internet of Things (IoT), multi-access edge computing (MEC)-enabled satellite-aerial-terrestrial integrated network (SATIN) has emerged as a promising technology to provide massive IoT devices with seamless and reliable communication and computation services. This paper investigates the cooperation of low Earth orbit (LEO) satellites, high altitude platforms (HAPs), and terrestrial base stations (BSs) to provide relaying and computation services for vastly distributed IoT devices. Considering the uncertainty in dynamic SATIN systems, we formulate a stochastic optimization problem to minimize the time-average expected service delay by jointly optimizing resource allocation and task offloading while satisfying the energy constraints. To solve the formulated problem, we first develop a Lyapunov-based online control algorithm to decompose it into multiple one-slot problems. Since each one-slot problem is a large-scale mixed-integer nonlinear program (MINLP) that is intractable for classical computers, we further propose novel hybrid quantum-classical generalized Benders' decomposition (HQCGBD) algorithms to solve the problem efficiently by leveraging quantum advantages in parallel computing. Numerical results validate the effectiveness of the proposed MEC-enabled SATIN schemes.

{{</citation>}}


## cs.DM (1)



### (44/58) Embedding 1-Planar Graphs in Ten Pages (Franz J. Brandenburg, 2023)

{{<citation>}}

Franz J. Brandenburg. (2023)  
**Embedding 1-Planar Graphs in Ten Pages**  

---
Primary Category: cs.DM  
Categories: 05C62, 68R10, F-2-2, cs-DM, cs.DM  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2312.15786v1)  

---


**ABSTRACT**  
Every planar graph has a 4-page book embedding and this bound is tight. We show that every 1-planar graph, which is a graph that admits a drawing with at most one crossing per edge, has a 10-page book embedding. In addition, four pages are sometimes necessary and always sufficient if the planar skeleton, obtained from a 1-planar drawing by removing all crossed edges, has a Hamiltonian cycle.

{{</citation>}}


## cs.LG (5)



### (45/58) XuanCe: A Comprehensive and Unified Deep Reinforcement Learning Library (Wenzhang Liu et al., 2023)

{{<citation>}}

Wenzhang Liu, Wenzhe Cai, Kun Jiang, Guangran Cheng, Yuanda Wang, Jiawei Wang, Jingyu Cao, Lele Xu, Chaoxu Mu, Changyin Sun. (2023)  
**XuanCe: A Comprehensive and Unified Deep Reinforcement Learning Library**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-DL, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.16248v1)  

---


**ABSTRACT**  
In this paper, we present XuanCe, a comprehensive and unified deep reinforcement learning (DRL) library designed to be compatible with PyTorch, TensorFlow, and MindSpore. XuanCe offers a wide range of functionalities, including over 40 classical DRL and multi-agent DRL algorithms, with the flexibility to easily incorporate new algorithms and environments. It is a versatile DRL library that supports CPU, GPU, and Ascend, and can be executed on various operating systems such as Ubuntu, Windows, MacOS, and EulerOS. Extensive benchmarks conducted on popular environments including MuJoCo, Atari, and StarCraftII multi-agent challenge demonstrate the library's impressive performance. XuanCe is open-source and can be accessed at https://github.com/agi-brain/xuance.git.

{{</citation>}}


### (46/58) TimesURL: Self-supervised Contrastive Learning for Universal Time Series Representation Learning (Jiexi Liu et al., 2023)

{{<citation>}}

Jiexi Liu, Songcan Chen. (2023)  
**TimesURL: Self-supervised Contrastive Learning for Universal Time Series Representation Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Computer Vision, Contrastive Learning, NLP, Natural Language Processing, Representation Learning, Time Series  
[Paper Link](http://arxiv.org/abs/2312.15709v1)  

---


**ABSTRACT**  
Learning universal time series representations applicable to various types of downstream tasks is challenging but valuable in real applications. Recently, researchers have attempted to leverage the success of self-supervised contrastive learning (SSCL) in Computer Vision(CV) and Natural Language Processing(NLP) to tackle time series representation. Nevertheless, due to the special temporal characteristics, relying solely on empirical guidance from other domains may be ineffective for time series and difficult to adapt to multiple downstream tasks. To this end, we review three parts involved in SSCL including 1) designing augmentation methods for positive pairs, 2) constructing (hard) negative pairs, and 3) designing SSCL loss. For 1) and 2), we find that unsuitable positive and negative pair construction may introduce inappropriate inductive biases, which neither preserve temporal properties nor provide sufficient discriminative features. For 3), just exploring segment- or instance-level semantics information is not enough for learning universal representation. To remedy the above issues, we propose a novel self-supervised framework named TimesURL. Specifically, we first introduce a frequency-temporal-based augmentation to keep the temporal property unchanged. And then, we construct double Universums as a special kind of hard negative to guide better contrastive learning. Additionally, we introduce time reconstruction as a joint optimization objective with contrastive learning to capture both segment-level and instance-level information. As a result, TimesURL can learn high-quality universal representations and achieve state-of-the-art performance in 6 different downstream tasks, including short- and long-term forecasting, imputation, classification, anomaly detection and transfer learning.

{{</citation>}}


### (47/58) Revisiting Knowledge Distillation under Distribution Shift (Songming Zhang et al., 2023)

{{<citation>}}

Songming Zhang, Ziyu Lyu, Xiaofeng Chen. (2023)  
**Revisiting Knowledge Distillation under Distribution Shift**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2312.16242v1)  

---


**ABSTRACT**  
Knowledge distillation transfers knowledge from large models into small models, and has recently made remarkable achievements. However, few studies has investigated the mechanism of knowledge distillation against distribution shift. Distribution shift refers to the data distribution drifts between training and testing phases. In this paper, we reconsider the paradigm of knowledge distillation by reformulating the objective function in shift situations. Under the real scenarios, we propose a unified and systematic framework to benchmark knowledge distillation against two general distributional shifts including diversity and correlation shift. The evaluation benchmark covers more than 30 methods from algorithmic, data-driven, and optimization perspectives for five benchmark datasets. Overall, we conduct extensive experiments on the student model. We reveal intriguing observations of poor teaching performance under distribution shifts; in particular, complex algorithms and data augmentation offer limited gains in many cases.

{{</citation>}}


### (48/58) Swap-based Deep Reinforcement Learning for Facility Location Problems in Networks (Wenxuan Guo et al., 2023)

{{<citation>}}

Wenxuan Guo, Yanyan Xu, Yaohui Jin. (2023)  
**Swap-based Deep Reinforcement Learning for Facility Location Problems in Networks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, math-CO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.15658v1)  

---


**ABSTRACT**  
Facility location problems on graphs are ubiquitous in real world and hold significant importance, yet their resolution is often impeded by NP-hardness. Recently, machine learning methods have been proposed to tackle such classical problems, but they are limited to the myopic constructive pattern and only consider the problems in Euclidean space. To overcome these limitations, we propose a general swap-based framework that addresses the p-median problem and the facility relocation problem on graphs and a novel reinforcement learning model demonstrating a keen awareness of complex graph structures. Striking a harmonious balance between solution quality and running time, our method surpasses handcrafted heuristics on intricate graph datasets. Additionally, we introduce a graph generation process to simulate real-world urban road networks with demand, facilitating the construction of large datasets for the classic problem. For the initialization of the locations of facilities, we introduce a physics-inspired strategy for the p-median problem, reaching more stable solutions than the random strategy. The proposed pipeline coupling the classic swap-based method with deep reinforcement learning marks a significant step forward in addressing the practical challenges associated with facility location on graphs.

{{</citation>}}


### (49/58) Context-aware Communication for Multi-agent Reinforcement Learning (Xinran Li et al., 2023)

{{<citation>}}

Xinran Li, Jun Zhang. (2023)  
**Context-aware Communication for Multi-agent Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-MA, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.15600v1)  

---


**ABSTRACT**  
Effective communication protocols in multi-agent reinforcement learning (MARL) are critical to fostering cooperation and enhancing team performance. To leverage communication, many previous works have proposed to compress local information into a single message and broadcast it to all reachable agents. This simplistic messaging mechanism, however, may fail to provide adequate, critical, and relevant information to individual agents, especially in severely bandwidth-limited scenarios. This motivates us to develop context-aware communication schemes for MARL, aiming to deliver personalized messages to different agents. Our communication protocol, named CACOM, consists of two stages. In the first stage, agents exchange coarse representations in a broadcast fashion, providing context for the second stage. Following this, agents utilize attention mechanisms in the second stage to selectively generate messages personalized for the receivers. Furthermore, we employ the learned step size quantization (LSQ) technique for message quantization to reduce the communication overhead. To evaluate the effectiveness of CACOM, we integrate it with both actor-critic and value-based MARL algorithms. Empirical results on cooperative benchmark tasks demonstrate that CACOM provides evident performance gains over baselines under communication-constrained scenarios.

{{</citation>}}


## cs.AI (4)



### (50/58) Spatial-Temporal Interplay in Human Mobility: A Hierarchical Reinforcement Learning Approach with Hypergraph Representation (Zhaofan Zhang et al., 2023)

{{<citation>}}

Zhaofan Zhang, Yanan Xiao, Lu Jiang, Dingqi Yang, Minghao Yin, Pengyang Wang. (2023)  
**Spatial-Temporal Interplay in Human Mobility: A Hierarchical Reinforcement Learning Approach with Hypergraph Representation**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CY, cs-LG, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.15717v1)  

---


**ABSTRACT**  
In the realm of human mobility, the decision-making process for selecting the next-visit location is intricately influenced by a trade-off between spatial and temporal constraints, which are reflective of individual needs and preferences. This trade-off, however, varies across individuals, making the modeling of these spatial-temporal dynamics a formidable challenge. To address the problem, in this work, we introduce the "Spatial-temporal Induced Hierarchical Reinforcement Learning" (STI-HRL) framework, for capturing the interplay between spatial and temporal factors in human mobility decision-making. Specifically, STI-HRL employs a two-tiered decision-making process: the low-level focuses on disentangling spatial and temporal preferences using dedicated agents, while the high-level integrates these considerations to finalize the decision. To complement the hierarchical decision setting, we construct a hypergraph to organize historical data, encapsulating the multi-aspect semantics of human mobility. We propose a cross-channel hypergraph embedding module to learn the representations as the states to facilitate the decision-making cycle. Our extensive experiments on two real-world datasets validate the superiority of STI-HRL over state-of-the-art methods in predicting users' next visits across various performance metrics.

{{</citation>}}


### (51/58) Instruction Fusion: Advancing Prompt Evolution through Hybridization (Weidong Guo et al., 2023)

{{<citation>}}

Weidong Guo, Jiuding Yang, Kaitong Yang, Xiangyang Li, Zhuwei Rao, Yu Xu, Di Niu. (2023)  
**Instruction Fusion: Advancing Prompt Evolution through Hybridization**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.15692v2)  

---


**ABSTRACT**  
The fine-tuning of Large Language Models (LLMs) specialized in code generation has seen notable advancements through the use of open-domain coding queries. Despite the successes, existing methodologies like Evol-Instruct encounter performance limitations, impeding further enhancements in code generation tasks. This paper examines the constraints of existing prompt evolution techniques and introduces a novel approach, Instruction Fusion (IF). IF innovatively combines two distinct prompts through a hybridization process, thereby enhancing the evolution of training prompts for code LLMs. Our experimental results reveal that the proposed novel method effectively addresses the shortcomings of prior methods, significantly improving the performance of Code LLMs across five code generation benchmarks, namely HumanEval, HumanEval+, MBPP, MBPP+ and MultiPL-E, which underscore the effectiveness of Instruction Fusion in advancing the capabilities of LLMs in code generation.

{{</citation>}}


### (52/58) Abductive Logical Reasoning on Knowledge Graphs (Jiaxin Bai et al., 2023)

{{<citation>}}

Jiaxin Bai, Yicheng Wang, Tianshi Zheng, Yue Guo, Xin Liu, Yangqiu Song. (2023)  
**Abductive Logical Reasoning on Knowledge Graphs**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: Knowledge Graph, Reasoning, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.15643v1)  

---


**ABSTRACT**  
Abductive reasoning is logical reasoning that makes educated guesses to infer the most likely reasons to explain the observations. However, the abductive logical reasoning over knowledge graphs (KGs) is underexplored in KG literature. In this paper, we initially and formally raise the task of abductive logical reasoning over KGs, which involves inferring the most probable logic hypothesis from the KGs to explain an observed entity set. Traditional approaches use symbolic methods, like searching, to tackle the knowledge graph problem. However, the symbolic methods are unsuitable for this task, because the KGs are naturally incomplete, and the logical hypotheses can be complex with multiple variables and relations. To address these issues, we propose a generative approach to create logical expressions based on observations. First, we sample hypothesis-observation pairs from the KG and use supervised training to train a generative model that generates hypotheses from observations. Since supervised learning only minimizes structural differences between generated and reference hypotheses, higher structural similarity does not guarantee a better explanation for observations. To tackle this issue, we introduce the Reinforcement Learning from the Knowledge Graph (RLF-KG) method, which minimizes the differences between observations and conclusions drawn from the generated hypotheses according to the KG. Experimental results demonstrate that transformer-based generative models can generate logical explanations robustly and efficiently. Moreover, with the assistance of RLF-KG, the generated hypothesis can provide better explanations for the observations, and the method of supervised learning with RLF-KG achieves state-of-the-art results on abductive knowledge graph reasoning on three widely used KGs.

{{</citation>}}


### (53/58) RDF-star2Vec: RDF-star Graph Embeddings for Data Mining (Shusaku Egami et al., 2023)

{{<citation>}}

Shusaku Egami, Takanori Ugai, Masateru Oota, Kyoumoto Matsushita, Takahiro Kawamura, Kouji Kozaki, Ken Fukuda. (2023)  
**RDF-star2Vec: RDF-star Graph Embeddings for Data Mining**  

---
Primary Category: cs.AI  
Categories: I-2-7; I-2-4; I-2-6, cs-AI, cs-CL, cs-IR, cs-LG, cs.AI  
Keywords: Embedding, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2312.15626v1)  

---


**ABSTRACT**  
Knowledge Graphs (KGs) such as Resource Description Framework (RDF) data represent relationships between various entities through the structure of triples (<subject, predicate, object>). Knowledge graph embedding (KGE) is crucial in machine learning applications, specifically in node classification and link prediction tasks. KGE remains a vital research topic within the semantic web community. RDF-star introduces the concept of a quoted triple (QT), a specific form of triple employed either as the subject or object within another triple. Moreover, RDF-star permits a QT to act as compositional entities within another QT, thereby enabling the representation of recursive, hyper-relational KGs with nested structures. However, existing KGE models fail to adequately learn the semantics of QTs and entities, primarily because they do not account for RDF-star graphs containing multi-leveled nested QTs and QT-QT relationships. This study introduces RDF-star2Vec, a novel KGE model specifically designed for RDF-star graphs. RDF-star2Vec introduces graph walk techniques that enable probabilistic transitions between a QT and its compositional entities. Feature vectors for QTs, entities, and relations are derived from generated sequences through the structured skip-gram model. Additionally, we provide a dataset and a benchmarking framework for data mining tasks focused on complex RDF-star graphs. Evaluative experiments demonstrated that RDF-star2Vec yielded superior performance compared to recent extensions of RDF2Vec in various tasks including classification, clustering, entity relatedness, and QT similarity.

{{</citation>}}


## cs.SE (1)



### (54/58) RepairLLaMA: Efficient Representations and Fine-Tuned Adapters for Program Repair (André Silva et al., 2023)

{{<citation>}}

André Silva, Sen Fang, Martin Monperrus. (2023)  
**RepairLLaMA: Efficient Representations and Fine-Tuned Adapters for Program Repair**  

---
Primary Category: cs.SE  
Categories: cs-LG, cs-SE, cs.SE  
Keywords: LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2312.15698v1)  

---


**ABSTRACT**  
Automated Program Repair (APR) has evolved significantly with the advent of Large Language Models (LLMs). Fine-tuning LLMs for program repair is a recent avenue of research, with many dimensions which have not been explored. Existing work mostly fine-tunes LLMs with naive code representations and is fundamentally limited in its ability to fine-tune larger LLMs. To address this problem, we propose RepairLLaMA, a novel program repair approach that combines 1) code representations for APR and 2) the state-of-the-art parameter-efficient LLM fine-tuning technique called LoRA. This results in RepairLLaMA producing a highly effective `program repair adapter' for fixing bugs with language models. Our experiments demonstrate the validity of both concepts. First, fine-tuning adapters with program repair specific code representations enables the model to use meaningful repair signals. Second, parameter-efficient fine-tuning helps fine-tuning to converge and contributes to the effectiveness of the repair adapter to fix data-points outside the fine-tuning data distribution. Overall, RepairLLaMA correctly fixes 125 Defects4J v2 and 82 HumanEval-Java bugs, outperforming all baselines.

{{</citation>}}


## cs.MA (1)



### (55/58) Multi-Task Multi-Agent Shared Layers are Universal Cognition of Multi-Agent Coordination (Jiawei Wang et al., 2023)

{{<citation>}}

Jiawei Wang, Jian Zhao, Zhengtao Cao, Ruili Feng, Rongjun Qin, Yang Yu. (2023)  
**Multi-Task Multi-Agent Shared Layers are Universal Cognition of Multi-Agent Coordination**  

---
Primary Category: cs.MA  
Categories: cs-MA, cs.MA  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2312.15674v1)  

---


**ABSTRACT**  
Multi-agent reinforcement learning shines as the pinnacle of multi-agent systems, conquering intricate real-world challenges, fostering collaboration and coordination among agents, and unleashing the potential for intelligent decision-making across domains. However, training a multi-agent reinforcement learning network is a formidable endeavor, demanding substantial computational resources to interact with diverse environmental variables, extract state representations, and acquire decision-making knowledge. The recent breakthroughs in large-scale pre-trained models ignite our curiosity: Can we uncover shared knowledge in multi-agent reinforcement learning and leverage pre-trained models to expedite training for future tasks? Addressing this issue, we present an innovative multi-task learning approach that aims to extract and harness common decision-making knowledge, like cooperation and competition, across different tasks. Our approach involves concurrent training of multiple multi-agent tasks, with each task employing independent front-end perception layers while sharing back-end decision-making layers. This effective decoupling of state representation extraction from decision-making allows for more efficient training and better transferability. To evaluate the efficacy of our proposed approach, we conduct comprehensive experiments in two distinct environments: the StarCraft Multi-agent Challenge (SMAC) and the Google Research Football (GRF) environments. The experimental results unequivocally demonstrate the smooth transferability of the shared decision-making network to other tasks, thereby significantly reducing training costs and improving final performance. Furthermore, visualizations authenticate the presence of general multi-agent decision-making knowledge within the shared network layers, further validating the effectiveness of our approach.

{{</citation>}}


## eess.SY (1)



### (56/58) Coordinated Planning of Offshore Charging Stations and Electrified Ships: A Case Study on Shanghai-Busan Maritime Route (Hao Li et al., 2023)

{{<citation>}}

Hao Li, Hanqi Tao, Wentao Huang, Hongcai Zhang, Ran Li. (2023)  
**Coordinated Planning of Offshore Charging Stations and Electrified Ships: A Case Study on Shanghai-Busan Maritime Route**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.15639v1)  

---


**ABSTRACT**  
Despite the success of electric vehicles on land, electrification of maritime ships is challenged by the dilemma of range anxiety and cargo-carrying capacity. The longer range requires larger batteries, which inevitably eat up the precious cargo space and weight. This paper breaks new ground by proposing a coordinated planning model for offshore charging stations (OCSs) and electric ships (ESs), marking a first in this field. Strategically situated OCS can partition a long maritime route into several shorter segments, which in turn lead to smaller batteries and thus larger cargo capacities. The research analyzed the impact of maritime geographical conditions on the placement and sizing process and provided insights into the trade-offs between battery size, cargo-carrying capacity, and the cruising range of different types of electrified ships. Using real Automatic Identification System (AIS) data, we estimated the economic feasibility of the Shanghai-Busan high-traffic maritime route and conducted a sensitivity analysis on factors affecting its economic viability. The results show that installing OCS can significantly reduce the propulsion cost compared with ESs without OCS and traditional internal combustion engine (ICE) ships.

{{</citation>}}


## cs.MM (1)



### (57/58) RMNAS: A Multimodal Neural Architecture Search Framework For Robust Multimodal Sentiment Analysis (Haiyang Sun et al., 2023)

{{<citation>}}

Haiyang Sun, Zheng Lian, Licai Sun, Bin Liu, Jianhua Tao. (2023)  
**RMNAS: A Multimodal Neural Architecture Search Framework For Robust Multimodal Sentiment Analysis**  

---
Primary Category: cs.MM  
Categories: cs-MM, cs.MM  
Keywords: Sentiment Analysis, Transformer  
[Paper Link](http://arxiv.org/abs/2312.15583v1)  

---


**ABSTRACT**  
Multimodal sentiment analysis (MSA) finds extensive applications, but the presence of missing modalities in real-world environments requires researchers to enhance the robustness of models, often demanding significant efforts. Multimodal neural architecture search (MNAS) is a more efficient approach. However, current MNAS methods, while effective in integrating multi-level information, are incapable of simultaneously searching for optimal operations to extract modality-specific information. This weakens the robustness of the model in addressing diverse scenarios. Moreover, these methods also fall short in enhancing the capture of emotional cues. In this paper, we propose robust-sentiment multimodal neural architecture search (RMNAS) framework. Specifically, we utilize the Transformer as a unified architecture for various modalities and incorporate a search for token mixers to enhance the encoding capacity of individual modalities and improve robustness across diverse scenarios. Subsequently, we leverage BM-NAS to integrate multi-level information. Furthermore, we incorporate local sentiment variation trends to guide the token mixers computation, enhancing the model's ability to capture sentiment context. Experimental results demonstrate that our approach outperforms or competitively matches existing state-of-the-art approaches in incomplete multimodal learning, both in sentence-level and dialogue-level MSA tasks, without the need for knowledge of incomplete learning.

{{</citation>}}


## cs.HC (1)



### (58/58) Conversational Co-Speech Gesture Generation via Modeling Dialog Intention, Emotion, and Context with Diffusion Models (Haiwei Xue et al., 2023)

{{<citation>}}

Haiwei Xue, Sicheng Yang, Zhensong Zhang, Zhiyong Wu, Minglei Li, Zonghong Dai, Helen Meng. (2023)  
**Conversational Co-Speech Gesture Generation via Modeling Dialog Intention, Emotion, and Context with Diffusion Models**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: Dialog  
[Paper Link](http://arxiv.org/abs/2312.15567v1)  

---


**ABSTRACT**  
Audio-driven co-speech human gesture generation has made remarkable advancements recently. However, most previous works only focus on single person audio-driven gesture generation. We aim at solving the problem of conversational co-speech gesture generation that considers multiple participants in a conversation, which is a novel and challenging task due to the difficulty of simultaneously incorporating semantic information and other relevant features from both the primary speaker and the interlocutor. To this end, we propose CoDiffuseGesture, a diffusion model-based approach for speech-driven interaction gesture generation via modeling bilateral conversational intention, emotion, and semantic context. Our method synthesizes appropriate interactive, speech-matched, high-quality gestures for conversational motions through the intention perception module and emotion reasoning module at the sentence level by a pretrained language model. Experimental results demonstrate the promising performance of the proposed method.

{{</citation>}}
