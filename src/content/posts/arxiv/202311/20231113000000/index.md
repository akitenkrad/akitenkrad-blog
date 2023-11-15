---
draft: false
title: "arXiv @ 2023.11.13"
date: 2023-11-13
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.11.13"
    identifier: arxiv_20231113
    parent: 202311_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CV (13)](#cscv-13)
- [cs.IR (3)](#csir-3)
- [cs.RO (2)](#csro-2)
- [cs.LG (5)](#cslg-5)
- [cs.CL (13)](#cscl-13)
- [cs.AI (3)](#csai-3)
- [cs.HC (1)](#cshc-1)
- [cs.PL (1)](#cspl-1)
- [eess.IV (1)](#eessiv-1)
- [q-bio.QM (1)](#q-bioqm-1)
- [physics.flu-dyn (1)](#physicsflu-dyn-1)
- [cs.SE (2)](#csse-2)
- [cs.DB (1)](#csdb-1)
- [cs.NI (1)](#csni-1)
- [cs.SI (1)](#cssi-1)
- [cs.IT (1)](#csit-1)
- [cs.CY (2)](#cscy-2)

## cs.CV (13)



### (1/52) Automatized Self-Supervised Learning for Skin Lesion Screening (Vullnet Useini et al., 2023)

{{<citation>}}

Vullnet Useini, Stephanie Tanadini-Lang, Quentin Lohmeyer, Mirko Meboldt, Nicolaus Andratschke, Ralph P. Braun, Javier Barranco García. (2023)  
**Automatized Self-Supervised Learning for Skin Lesion Screening**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2311.06691v1)  

---


**ABSTRACT**  
The incidence rates of melanoma, the deadliest form of skin cancer, have been increasing steadily worldwide, presenting a significant challenge to dermatologists. Early detection of melanoma is crucial for improving patient survival rates, but identifying suspicious lesions through ugly duckling (UD) screening, the current method used for skin cancer screening, can be challenging and often requires expertise in pigmented lesions. To address these challenges and improve patient outcomes, an artificial intelligence (AI) decision support tool was developed to assist dermatologists in identifying UD from wide-field patient images. The tool uses a state-of-the-art object detection algorithm to identify and extract all skin lesions from patient images, which are then sorted by suspiciousness using a self-supervised AI algorithm. A clinical validation study was conducted to evaluate the tool's performance, which demonstrated an average sensitivity of 93% for the top-10 AI-identified UDs on skin lesions selected by the majority of experts in pigmented skin lesions. The study also found that dermatologists confidence increased, and the average majority agreement with the top-10 AI-identified UDs improved to 100% when assisted by AI. The development of this AI decision support tool aims to address the shortage of specialists, enable at-risk patients to receive faster consultations and understand the impact of AI-assisted screening. The tool's automation can assist dermatologists in identifying suspicious lesions and provide a more objective assessment, reducing subjectivity in the screening process. The future steps for this project include expanding the dataset to include histologically confirmed melanoma cases and increasing the number of participants for clinical validation to strengthen the tool's reliability and adapt it for real-world consultation.

{{</citation>}}


### (2/52) Unsupervised and semi-supervised co-salient object detection via segmentation frequency statistics (Souradeep Chakraborty et al., 2023)

{{<citation>}}

Souradeep Chakraborty, Shujon Naha, Muhammet Bastan, Amit Kumar K C, Dimitris Samaras. (2023)  
**Unsupervised and semi-supervised co-salient object detection via segmentation frequency statistics**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2311.06654v1)  

---


**ABSTRACT**  
In this paper, we address the detection of co-occurring salient objects (CoSOD) in an image group using frequency statistics in an unsupervised manner, which further enable us to develop a semi-supervised method. While previous works have mostly focused on fully supervised CoSOD, less attention has been allocated to detecting co-salient objects when limited segmentation annotations are available for training. Our simple yet effective unsupervised method US-CoSOD combines the object co-occurrence frequency statistics of unsupervised single-image semantic segmentations with salient foreground detections using self-supervised feature learning. For the first time, we show that a large unlabeled dataset e.g. ImageNet-1k can be effectively leveraged to significantly improve unsupervised CoSOD performance. Our unsupervised model is a great pre-training initialization for our semi-supervised model SS-CoSOD, especially when very limited labeled data is available for training. To avoid propagating erroneous signals from predictions on unlabeled data, we propose a confidence estimation module to guide our semi-supervised training. Extensive experiments on three CoSOD benchmark datasets show that both of our unsupervised and semi-supervised models outperform the corresponding state-of-the-art models by a significant margin (e.g., on the Cosal2015 dataset, our US-CoSOD model has an 8.8% F-measure gain over a SOTA unsupervised co-segmentation model and our SS-CoSOD model has an 11.81% F-measure gain over a SOTA semi-supervised CoSOD model).

{{</citation>}}


### (3/52) Traffic Sign Recognition Using Local Vision Transformer (Ali Farzipour et al., 2023)

{{<citation>}}

Ali Farzipour, Omid Nejati Manzari, Shahriar B. Shokouhi. (2023)  
**Traffic Sign Recognition Using Local Vision Transformer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.06651v1)  

---


**ABSTRACT**  
Recognition of traffic signs is a crucial aspect of self-driving cars and driver assistance systems, and machine vision tasks such as traffic sign recognition have gained significant attention. CNNs have been frequently used in machine vision, but introducing vision transformers has provided an alternative approach to global feature learning. This paper proposes a new novel model that blends the advantages of both convolutional and transformer-based networks for traffic sign recognition. The proposed model includes convolutional blocks for capturing local correlations and transformer-based blocks for learning global dependencies. Additionally, a locality module is incorporated to enhance local perception. The performance of the suggested model is evaluated on the Persian Traffic Sign Dataset and German Traffic Sign Recognition Benchmark and compared with SOTA convolutional and transformer-based models. The experimental evaluations demonstrate that the hybrid network with the locality module outperforms pure transformer-based models and some of the best convolutional networks in accuracy. Specifically, our proposed final model reached 99.66% accuracy in the German traffic sign recognition benchmark and 99.8% in the Persian traffic sign dataset, higher than the best convolutional models. Moreover, it outperforms existing CNNs and ViTs while maintaining fast inference speed. Consequently, the proposed model proves to be significantly faster and more suitable for real-world applications.

{{</citation>}}


### (4/52) VT-Former: A Transformer-based Vehicle Trajectory Prediction Approach For Intelligent Highway Transportation Systems (Armin Danesh Pazho et al., 2023)

{{<citation>}}

Armin Danesh Pazho, Vinit Katariya, Ghazal Alinezhad Noghre, Hamed Tabkhi. (2023)  
**VT-Former: A Transformer-based Vehicle Trajectory Prediction Approach For Intelligent Highway Transportation Systems**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI, Transformer  
[Paper Link](http://arxiv.org/abs/2311.06623v1)  

---


**ABSTRACT**  
Enhancing roadway safety and traffic management has become an essential focus area for a broad range of modern cyber-physical systems and intelligent transportation systems. Vehicle Trajectory Prediction is a pivotal element within numerous applications for highway and road safety. These applications encompass a wide range of use cases, spanning from traffic management and accident prevention to enhancing work-zone safety and optimizing energy conservation. The ability to implement intelligent management in this context has been greatly advanced by the developments in the field of Artificial Intelligence (AI), alongside the increasing deployment of surveillance cameras across road networks. In this paper, we introduce a novel transformer-based approach for vehicle trajectory prediction for highway safety and surveillance, denoted as VT-Former. In addition to utilizing transformers to capture long-range temporal patterns, a new Graph Attentive Tokenization (GAT) module has been proposed to capture intricate social interactions among vehicles. Combining these two core components culminates in a precise approach for vehicle trajectory prediction. Our study on three benchmark datasets with three different viewpoints demonstrates the State-of-The-Art (SoTA) performance of VT-Former in vehicle trajectory prediction and its generalizability and robustness. We also evaluate VT-Former's efficiency on embedded boards and explore its potential for vehicle anomaly detection as a sample application, showcasing its broad applicability.

{{</citation>}}


### (5/52) Computer Vision for Particle Size Analysis of Coarse-Grained Soils (Sompote Youwai et al., 2023)

{{<citation>}}

Sompote Youwai, Parchya Makam. (2023)  
**Computer Vision for Particle Size Analysis of Coarse-Grained Soils**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Computer Vision  
[Paper Link](http://arxiv.org/abs/2311.06613v1)  

---


**ABSTRACT**  
Particle size analysis (PSA) is a fundamental technique for evaluating the physical characteristics of soils. However, traditional methods like sieving can be time-consuming and labor-intensive. In this study, we present a novel approach that utilizes computer vision (CV) and the Python programming language for PSA of coarse-grained soils, employing a standard mobile phone camera. By eliminating the need for a high-performance camera, our method offers convenience and cost savings. Our methodology involves using the OPENCV library to detect and measure soil particles in digital photographs taken under ordinary lighting conditions. For accurate particle size determination, a calibration target with known dimensions is placed on a plain paper alongside 20 different sand samples. The proposed method is compared with traditional sieve analysis and exhibits satisfactory performance for soil particles larger than 2 mm, with a mean absolute percent error (MAPE) of approximately 6%. However, particles smaller than 2 mm result in higher MAPE, reaching up to 60%. To address this limitation, we recommend using a higher-resolution camera to capture images of the smaller soil particles. Furthermore, we discuss the advantages, limitations, and potential future improvements of our method. Remarkably, the program can be executed on a mobile phone, providing immediate results without the need to send soil samples to a laboratory. This field-friendly feature makes our approach highly convenient for on-site usage, outside of a traditional laboratory setting. Ultimately, this novel method represents an initial disruption to the industry, enabling efficient particle size analysis of soil without the reliance on laboratory-based sieve analysis. KEYWORDS: Computer vision, Grain size, ARUCO

{{</citation>}}


### (6/52) PerceptionGPT: Effectively Fusing Visual Perception into LLM (Renjie Pi et al., 2023)

{{<citation>}}

Renjie Pi, Lewei Yao, Jiahui Gao, Jipeng Zhang, Tong Zhang. (2023)  
**PerceptionGPT: Effectively Fusing Visual Perception into LLM**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2311.06612v1)  

---


**ABSTRACT**  
The integration of visual inputs with large language models (LLMs) has led to remarkable advancements in multi-modal capabilities, giving rise to visual large language models (VLLMs). However, effectively harnessing VLLMs for intricate visual perception tasks remains a challenge. In this paper, we present a novel end-to-end framework named PerceptionGPT, which efficiently and effectively equips the VLLMs with visual perception abilities by leveraging the representation power of LLMs' token embedding. Our proposed method treats the token embedding of the LLM as the carrier of spatial information, then leverage lightweight visual task encoders and decoders to perform visual perception tasks (e.g., detection, segmentation). Our approach significantly alleviates the training difficulty suffered by previous approaches that formulate the visual outputs as discrete tokens, and enables achieving superior performance with fewer trainable parameters, less training data and shorted training time. Moreover, as only one token embedding is required to decode the visual outputs, the resulting sequence length during inference is significantly reduced. Consequently, our approach enables accurate and flexible representations, seamless integration of visual perception tasks, and efficient handling of a multiple of visual outputs. We validate the effectiveness and efficiency of our approach through extensive experiments. The results demonstrate significant improvements over previous methods with much fewer trainable parameters and GPU hours, which facilitates future research in enabling LLMs with visual perception abilities.

{{</citation>}}


### (7/52) Monkey: Image Resolution and Text Label Are Important Things for Large Multi-modal Models (Zhang Li et al., 2023)

{{<citation>}}

Zhang Li, Biao Yang, Qiang Liu, Zhiyin Ma, Shuo Zhang, Jingxu Yang, Yabo Sun, Yuliang Liu, Xiang Bai. (2023)  
**Monkey: Image Resolution and Text Label Are Important Things for Large Multi-modal Models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs.CV  
Keywords: Image Captioning, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2311.06607v1)  

---


**ABSTRACT**  
Large Multimodal Models have demonstrated impressive capabilities in understanding general vision-language tasks. However, due to the limitation of supported input resolution (e.g., 448 x 448) as well as the inexhaustive description of the training image-text pair, these models often encounter challenges when dealing with intricate scene understandings and narratives. Here we address the problem by proposing the Monkey. Our contributions are two-fold: 1) without pretraining from the start, our method can be built upon an existing vision encoder (e.g., vit-BigHuge) to effectively improve the input resolution capacity up to 896 x 1344 pixels; 2) we propose a multi-level description generation method, which automatically provides rich information that can guide model to learn contextual association between scenes and objects. Our extensive testing across more than 16 distinct datasets reveals that Monkey achieves consistently competitive performance over the existing LMMs on fundamental tasks, such as Image Captioning, General Visual Question Answering (VQA), and Document-oriented VQA. Models, interactive demo, and the source code are provided at the following https://github.com/Yuliang-Liu/Monkey.

{{</citation>}}


### (8/52) OR Residual Connection Achieving Comparable Accuracy to ADD Residual Connection in Deep Residual Spiking Neural Networks (Yimeng Shan et al., 2023)

{{<citation>}}

Yimeng Shan, Xuerui Qiu, Rui-jie Zhu, Ruike Li, Meng Wang, Haicheng Qu. (2023)  
**OR Residual Connection Achieving Comparable Accuracy to ADD Residual Connection in Deep Residual Spiking Neural Networks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2311.06570v1)  

---


**ABSTRACT**  
Spiking Neural Networks (SNNs) have garnered substantial attention in brain-like computing for their biological fidelity and the capacity to execute energy-efficient spike-driven operations. As the demand for heightened performance in SNNs surges, the trend towards training deeper networks becomes imperative, while residual learning stands as a pivotal method for training deep neural networks. In our investigation, we identified that the SEW-ResNet, a prominent representative of deep residual spiking neural networks, incorporates non-event-driven operations. To rectify this, we introduce the OR Residual connection (ORRC) to the architecture. Additionally, we propose the Synergistic Attention (SynA) module, an amalgamation of the Inhibitory Attention (IA) module and the Multi-dimensional Attention (MA) module, to offset energy loss stemming from high quantization. When integrating SynA into the network, we observed the phenomenon of "natural pruning", where after training, some or all of the shortcuts in the network naturally drop out without affecting the model's classification accuracy. This significantly reduces computational overhead and makes it more suitable for deployment on edge devices. Experimental results on various public datasets confirmed that the SynA enhanced OR-Spiking ResNet achieved single-sample classification with as little as 0.8 spikes per neuron. Moreover, when compared to other spike residual models, it exhibited higher accuracy and lower power consumption. Codes are available at https://github.com/Ym-Shan/ORRC-SynA-natural-pruning.

{{</citation>}}


### (9/52) Visual Commonsense based Heterogeneous Graph Contrastive Learning (Zongzhao Li et al., 2023)

{{<citation>}}

Zongzhao Li, Xiangyu Zhu, Xi Zhang, Zhaoxiang Zhang, Zhen Lei. (2023)  
**Visual Commonsense based Heterogeneous Graph Contrastive Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning, QA  
[Paper Link](http://arxiv.org/abs/2311.06553v1)  

---


**ABSTRACT**  
How to select relevant key objects and reason about the complex relationships cross vision and linguistic domain are two key issues in many multi-modality applications such as visual question answering (VQA). In this work, we incorporate the visual commonsense information and propose a heterogeneous graph contrastive learning method to better finish the visual reasoning task. Our method is designed as a plug-and-play way, so that it can be quickly and easily combined with a wide range of representative methods. Specifically, our model contains two key components: the Commonsense-based Contrastive Learning and the Graph Relation Network. Using contrastive learning, we guide the model concentrate more on discriminative objects and relevant visual commonsense attributes. Besides, thanks to the introduction of the Graph Relation Network, the model reasons about the correlations between homogeneous edges and the similarities between heterogeneous edges, which makes information transmission more effective. Extensive experiments on four benchmarks show that our method greatly improves seven representative VQA models, demonstrating its effectiveness and generalizability.

{{</citation>}}


### (10/52) Generation Of Colors using Bidirectional Long Short Term Memory Networks (A. Sinha, 2023)

{{<citation>}}

A. Sinha. (2023)  
**Generation Of Colors using Bidirectional Long Short Term Memory Networks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: LSTM, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2311.06542v1)  

---


**ABSTRACT**  
Human vision can distinguish between a vast spectrum of colours, estimated to be between 2 to 7 million discernible shades. However, this impressive range does not inherently imply that all these colours have been precisely named and described within our lexicon. We often associate colours with familiar objects and concepts in our daily lives. This research endeavors to bridge the gap between our visual perception of countless shades and our ability to articulate and name them accurately. A novel model has been developed to achieve this goal, leveraging Bidirectional Long Short-Term Memory (BiLSTM) networks with Active learning. This model operates on a proprietary dataset meticulously curated for this study. The primary objective of this research is to create a versatile tool for categorizing and naming previously unnamed colours or identifying intermediate shades that elude traditional colour terminology. The findings underscore the potential of this innovative approach in revolutionizing our understanding of colour perception and language. Through rigorous experimentation and analysis, this study illuminates a promising avenue for Natural Language Processing (NLP) applications in diverse industries. By facilitating the exploration of the vast colour spectrum the potential applications of NLP are extended beyond conventional boundaries.

{{</citation>}}


### (11/52) LayoutPrompter: Awaken the Design Ability of Large Language Models (Jiawei Lin et al., 2023)

{{<citation>}}

Jiawei Lin, Jiaqi Guo, Shizhao Sun, Zijiang James Yang, Jian-Guang Lou, Dongmei Zhang. (2023)  
**LayoutPrompter: Awaken the Design Ability of Large Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.06495v1)  

---


**ABSTRACT**  
Conditional graphic layout generation, which automatically maps user constraints to high-quality layouts, has attracted widespread attention today. Although recent works have achieved promising performance, the lack of versatility and data efficiency hinders their practical applications. In this work, we propose LayoutPrompter, which leverages large language models (LLMs) to address the above problems through in-context learning. LayoutPrompter is made up of three key components, namely input-output serialization, dynamic exemplar selection and layout ranking. Specifically, the input-output serialization component meticulously designs the input and output formats for each layout generation task. Dynamic exemplar selection is responsible for selecting the most helpful prompting exemplars for a given input. And a layout ranker is used to pick the highest quality layout from multiple outputs of LLMs. We conduct experiments on all existing layout generation tasks using four public datasets. Despite the simplicity of our approach, experimental results show that LayoutPrompter can compete with or even outperform state-of-the-art approaches on these tasks without any model training or fine-tuning. This demonstrates the effectiveness of this versatile and training-free approach. In addition, the ablation studies show that LayoutPrompter is significantly superior to the training-based baseline in a low-data regime, further indicating the data efficiency of LayoutPrompter. Our project is available at https://github.com/microsoft/LayoutGeneration/tree/main/LayoutPrompter.

{{</citation>}}


### (12/52) PECoP: Parameter Efficient Continual Pretraining for Action Quality Assessment (Amirhossein Dadashzadeh et al., 2023)

{{<citation>}}

Amirhossein Dadashzadeh, Shuchao Duan, Alan Whone, Majid Mirmehdi. (2023)  
**PECoP: Parameter Efficient Continual Pretraining for Action Quality Assessment**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AWS, QA  
[Paper Link](http://arxiv.org/abs/2311.07603v1)  

---


**ABSTRACT**  
The limited availability of labelled data in Action Quality Assessment (AQA), has forced previous works to fine-tune their models pretrained on large-scale domain-general datasets. This common approach results in weak generalisation, particularly when there is a significant domain shift. We propose a novel, parameter efficient, continual pretraining framework, PECoP, to reduce such domain shift via an additional pretraining stage. In PECoP, we introduce 3D-Adapters, inserted into the pretrained model, to learn spatiotemporal, in-domain information via self-supervised learning where only the adapter modules' parameters are updated. We demonstrate PECoP's ability to enhance the performance of recent state-of-the-art methods (MUSDL, CoRe, and TSA) applied to AQA, leading to considerable improvements on benchmark datasets, JIGSAWS ($\uparrow6.0\%$), MTL-AQA ($\uparrow0.99\%$), and FineDiving ($\uparrow2.54\%$). We also present a new Parkinson's Disease dataset, PD4T, of real patients performing four various actions, where we surpass ($\uparrow3.56\%$) the state-of-the-art in comparison. Our code, pretrained models, and the PD4T dataset are available at https://github.com/Plrbear/PECoP.

{{</citation>}}


### (13/52) CVTHead: One-shot Controllable Head Avatar with Vertex-feature Transformer (Haoyu Ma et al., 2023)

{{<citation>}}

Haoyu Ma, Tong Zhang, Shanlin Sun, Xiangyi Yan, Kun Han, Xiaohui Xie. (2023)  
**CVTHead: One-shot Controllable Head Avatar with Vertex-feature Transformer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.06443v1)  

---


**ABSTRACT**  
Reconstructing personalized animatable head avatars has significant implications in the fields of AR/VR. Existing methods for achieving explicit face control of 3D Morphable Models (3DMM) typically rely on multi-view images or videos of a single subject, making the reconstruction process complex. Additionally, the traditional rendering pipeline is time-consuming, limiting real-time animation possibilities. In this paper, we introduce CVTHead, a novel approach that generates controllable neural head avatars from a single reference image using point-based neural rendering. CVTHead considers the sparse vertices of mesh as the point set and employs the proposed Vertex-feature Transformer to learn local feature descriptors for each vertex. This enables the modeling of long-range dependencies among all the vertices. Experimental results on the VoxCeleb dataset demonstrate that CVTHead achieves comparable performance to state-of-the-art graphics-based methods. Moreover, it enables efficient rendering of novel human heads with various expressions, head poses, and camera views. These attributes can be explicitly controlled using the coefficients of 3DMMs, facilitating versatile and realistic animation in real-time scenarios.

{{</citation>}}


## cs.IR (3)



### (14/52) Metric Optimization and Mainstream Bias Mitigation in Recommender Systems (Roger Zhe Li, 2023)

{{<citation>}}

Roger Zhe Li. (2023)  
**Metric Optimization and Mainstream Bias Mitigation in Recommender Systems**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2311.06689v1)  

---


**ABSTRACT**  
The first part of this thesis focuses on maximizing the overall recommendation accuracy. This accuracy is usually evaluated with some user-oriented metric tailored to the recommendation scenario, but because recommendation is usually treated as a machine learning problem, recommendation models are trained to maximize some other generic criteria that does not necessarily align with the criteria ultimately captured by the user-oriented evaluation metric. Recent research aims at bridging this gap between training and evaluation via direct ranking optimization, but still assumes that the metric used for evaluation should also be the metric used for training. We challenge this assumption, mainly because some metrics are more informative than others. Indeed, we show that models trained via the optimization of a loss inspired by Rank-Biased Precision (RBP) tend to yield higher accuracy, even when accuracy is measured with metrics other than RBP. However, the superiority of this RBP-inspired loss stems from further benefiting users who are already well-served, rather than helping those who are not.   This observation inspires the second part of this thesis, where our focus turns to helping non-mainstream users. These are users who are difficult to recommend to either because there is not enough data to model them, or because they have niche taste and thus few similar users to look at when recommending in a collaborative way. These differences in mainstreamness introduce a bias reflected in an accuracy gap between users or user groups, which we try to narrow.

{{</citation>}}


### (15/52) An Empirical Study of Using ChatGPT for Fact Verification Task (Mohna Chakraborty et al., 2023)

{{<citation>}}

Mohna Chakraborty, Adithya Kulkarni, Qi Li. (2023)  
**An Empirical Study of Using ChatGPT for Fact Verification Task**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: ChatGPT, Fact Verification, GPT, NLP  
[Paper Link](http://arxiv.org/abs/2311.06592v1)  

---


**ABSTRACT**  
ChatGPT has recently emerged as a powerful tool for performing diverse NLP tasks. However, ChatGPT has been criticized for generating nonfactual responses, raising concerns about its usability for sensitive tasks like fact verification. This study investigates three key research questions: (1) Can ChatGPT be used for fact verification tasks? (2) What are different prompts performance using ChatGPT for fact verification tasks? (3) For the best-performing prompt, what common mistakes does ChatGPT make? Specifically, this study focuses on conducting a comprehensive and systematic analysis by designing and comparing the performance of three different prompts for fact verification tasks on the benchmark FEVER dataset using ChatGPT.

{{</citation>}}


### (16/52) Mitigating Pooling Bias in E-commerce Search via False Negative Estimation (Xiaochen Wang et al., 2023)

{{<citation>}}

Xiaochen Wang, Xiao Xiao, Ruhan Zhang, Xuan Zhang, Taesik Na, Tejaswi Tenneti, Haixun Wang, Fenglong Ma. (2023)  
**Mitigating Pooling Bias in E-commerce Search via False Negative Estimation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs-LG, cs.IR  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2311.06444v1)  

---


**ABSTRACT**  
Efficient and accurate product relevance assessment is critical for user experiences and business success. Training a proficient relevance assessment model requires high-quality query-product pairs, often obtained through negative sampling strategies. Unfortunately, current methods introduce pooling bias by mistakenly sampling false negatives, diminishing performance and business impact. To address this, we present Bias-mitigating Hard Negative Sampling (BHNS), a novel negative sampling strategy tailored to identify and adjust for false negatives, building upon our original False Negative Estimation algorithm. Our experiments in the Instacart search setting confirm BHNS as effective for practical e-commerce use. Furthermore, comparative analyses on public dataset showcase its domain-agnostic potential for diverse applications.

{{</citation>}}


## cs.RO (2)



### (17/52) The Effect of Trust and its Antecedents on Robot Acceptance (Katrin Fischer et al., 2023)

{{<citation>}}

Katrin Fischer, Donggyu Kim, Joo-Wha Hong. (2023)  
**The Effect of Trust and its Antecedents on Robot Acceptance**  

---
Primary Category: cs.RO  
Categories: cs-HC, cs-RO, cs.RO  
Keywords: Amazon  
[Paper Link](http://arxiv.org/abs/2311.06688v1)  

---


**ABSTRACT**  
As social and socially assistive robots are becoming more prevalent in our society, it is beneficial to understand how people form first impressions of them and eventually come to trust and accept them. This paper describes an Amazon Mechanical Turk study (n = 239) that investigated trust and its antecedents trustworthiness and first impressions. Participants evaluated the social robot Pepper's warmth and competence as well as trustworthiness characteristics ability, benevolence and integrity followed by their trust in and intention to use the robot. Mediation analyses assessed to what degree participants' first impressions affected their willingness to trust and use it. Known constructs from user acceptance and trust research were introduced to explain the pathways in which one perception predicted the next. Results showed that trustworthiness and trust, in serial, mediated the relationship between first impressions and behavioral intention.

{{</citation>}}


### (18/52) NewsGPT: ChatGPT Integration for Robot-Reporter (Abdelhadi Hireche et al., 2023)

{{<citation>}}

Abdelhadi Hireche, Abdelkader Nasreddine Belkacem, Sadia Jamil, Chao Chen. (2023)  
**NewsGPT: ChatGPT Integration for Robot-Reporter**  

---
Primary Category: cs.RO  
Categories: cs-HC, cs-MM, cs-RO, cs.RO  
Keywords: AI, ChatGPT, GPT, Google  
[Paper Link](http://arxiv.org/abs/2311.06640v1)  

---


**ABSTRACT**  
The integration of large language models (LLMs) with social robots has emerged as a promising avenue for enhancing human-robot interactions at a time when news reports generated by artificial intelligence (AI) are gaining in credibility. This integration is expected to intensify and become a more productive resource for journalism, media, communication, and education. In this paper a novel system is proposed that integrates AI's generative pretrained transformer (GPT) model with the Pepper robot, with the aim of improving the robot's natural language understanding and response generation capabilities for enhanced social interactions. By leveraging GPT's powerful language processing capabilities, this system offers a comprehensive pipeline that incorporates voice input recording, speech-to-text transcription, context analysis, and text-to-speech synthesis action generation. The Pepper robot is enabled to comprehend user queries, generate informative responses with general knowledge, maintain contextually relevant conversations, and act as a more domain-oriented news reporter. It is also linked with a news resource and powered with a Google search capability. To evaluate the performance of the framework, experiments were conducted involving a set of diverse questions. The robot's responses were assessed on the basis of eight criteria, including relevance, context, and fluency. Despite some identified limitations, this system contributes to the field of journalism and human-robot interaction by showcasing the potential of integrating LLMs with social robots. The proposed framework opens up opportunities for improving the conversational capabilities of robots, enabling interactions that are smoother, more engaging, and more context aware.

{{</citation>}}


## cs.LG (5)



### (19/52) Dream to Adapt: Meta Reinforcement Learning by Latent Context Imagination and MDP Imagination (Lu Wen et al., 2023)

{{<citation>}}

Lu Wen, Songan Zhang, H. Eric Tseng, Huei Peng. (2023)  
**Dream to Adapt: Meta Reinforcement Learning by Latent Context Imagination and MDP Imagination**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-RO, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.06673v1)  

---


**ABSTRACT**  
Meta reinforcement learning (Meta RL) has been amply explored to quickly learn an unseen task by transferring previously learned knowledge from similar tasks. However, most state-of-the-art algorithms require the meta-training tasks to have a dense coverage on the task distribution and a great amount of data for each of them. In this paper, we propose MetaDreamer, a context-based Meta RL algorithm that requires less real training tasks and data by doing meta-imagination and MDP-imagination. We perform meta-imagination by interpolating on the learned latent context space with disentangled properties, as well as MDP-imagination through the generative world model where physical knowledge is added to plain VAE networks. Our experiments with various benchmarks show that MetaDreamer outperforms existing approaches in data efficiency and interpolated generalization.

{{</citation>}}


### (20/52) The Pros and Cons of Using Machine Learning and Interpretable Machine Learning Methods in psychiatry detection applications, specifically depression disorder: A Brief Review (Hossein Simchi et al., 2023)

{{<citation>}}

Hossein Simchi, Samira Tajik. (2023)  
**The Pros and Cons of Using Machine Learning and Interpretable Machine Learning Methods in psychiatry detection applications, specifically depression disorder: A Brief Review**  

---
Primary Category: cs.LG  
Categories: 68T01, cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.06633v1)  

---


**ABSTRACT**  
The COVID-19 pandemic has forced many people to limit their social activities, which has resulted in a rise in mental illnesses, particularly depression. To diagnose these illnesses with accuracy and speed, and prevent severe outcomes such as suicide, the use of machine learning has become increasingly important. Additionally, to provide precise and understandable diagnoses for better treatment, AI scientists and researchers must develop interpretable AI-based solutions. This article provides an overview of relevant articles in the field of machine learning and interpretable AI, which helps to understand the advantages and disadvantages of using AI in psychiatry disorder detection applications.

{{</citation>}}


### (21/52) Graph ODE with Factorized Prototypes for Modeling Complicated Interacting Dynamics (Xiao Luo et al., 2023)

{{<citation>}}

Xiao Luo, Yiyang Gu, Huiyu Jiang, Jinsheng Huang, Wei Ju, Ming Zhang, Yizhou Sun. (2023)  
**Graph ODE with Factorized Prototypes for Modeling Complicated Interacting Dynamics**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2311.06554v1)  

---


**ABSTRACT**  
This paper studies the problem of modeling interacting dynamical systems, which is critical for understanding physical dynamics and biological processes. Recent research predominantly uses geometric graphs to represent these interactions, which are then captured by powerful graph neural networks (GNNs). However, predicting interacting dynamics in challenging scenarios such as out-of-distribution shift and complicated underlying rules remains unsolved. In this paper, we propose a new approach named Graph ODE with factorized prototypes (GOAT) to address the problem. The core of GOAT is to incorporate factorized prototypes from contextual knowledge into a continuous graph ODE framework. Specifically, GOAT employs representation disentanglement and system parameters to extract both object-level and system-level contexts from historical trajectories, which allows us to explicitly model their independent influence and thus enhances the generalization capability under system changes. Then, we integrate these disentangled latent representations into a graph ODE model, which determines a combination of various interacting prototypes for enhanced model expressivity. The entire model is optimized using an end-to-end variational inference framework to maximize the likelihood. Extensive experiments in both in-distribution and out-of-distribution settings validate the superiority of GOAT.

{{</citation>}}


### (22/52) MuST: Multimodal Spatiotemporal Graph-Transformer for Hospital Readmission Prediction (Yan Miao et al., 2023)

{{<citation>}}

Yan Miao, Lequan Yu. (2023)  
**MuST: Multimodal Spatiotemporal Graph-Transformer for Hospital Readmission Prediction**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.07608v1)  

---


**ABSTRACT**  
Hospital readmission prediction is considered an essential approach to decreasing readmission rates, which is a key factor in assessing the quality and efficacy of a healthcare system. Previous studies have extensively utilized three primary modalities, namely electronic health records (EHR), medical images, and clinical notes, to predict hospital readmissions. However, the majority of these studies did not integrate information from all three modalities or utilize the spatiotemporal relationships present in the dataset. This study introduces a novel model called the Multimodal Spatiotemporal Graph-Transformer (MuST) for predicting hospital readmissions. By employing Graph Convolution Networks and temporal transformers, we can effectively capture spatial and temporal dependencies in EHR and chest radiographs. We then propose a fusion transformer to combine the spatiotemporal features from the two modalities mentioned above with the features from clinical notes extracted by a pre-trained, domain-specific transformer. We assess the effectiveness of our methods using the latest publicly available dataset, MIMIC-IV. The experimental results indicate that the inclusion of multimodal features in MuST improves its performance in comparison to unimodal methods. Furthermore, our proposed pipeline outperforms the current leading methods in the prediction of hospital readmissions.

{{</citation>}}


### (23/52) Finetuning Text-to-Image Diffusion Models for Fairness (Xudong Shen et al., 2023)

{{<citation>}}

Xudong Shen, Chao Du, Tianyu Pang, Min Lin, Yongkang Wong, Mohan Kankanhalli. (2023)  
**Finetuning Text-to-Image Diffusion Models for Fairness**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-CY, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.07604v1)  

---


**ABSTRACT**  
The rapid adoption of text-to-image diffusion models in society underscores an urgent need to address their biases. Without interventions, these biases could propagate a distorted worldview and limit opportunities for minority groups. In this work, we frame fairness as a distributional alignment problem. Our solution consists of two main technical contributions: (1) a distributional alignment loss that steers specific characteristics of the generated images towards a user-defined target distribution, and (2) biased direct finetuning of diffusion model's sampling process, which leverages a biased gradient to more effectively optimize losses defined on the generated images. Empirically, our method markedly reduces gender, racial, and their intersectional biases for occupational prompts. Gender bias is significantly reduced even when finetuning just five soft tokens. Crucially, our method supports diverse perspectives of fairness beyond absolute equality, which is demonstrated by controlling age to a $75\%$ young and $25\%$ old distribution while simultaneously debiasing gender and race. Finally, our method is scalable: it can debias multiple concepts at once by simply including these prompts in the finetuning data. We hope our work facilitates the social alignment of T2I generative AI. We will share code and various debiased diffusion model adaptors.

{{</citation>}}


## cs.CL (13)



### (24/52) Intentional Biases in LLM Responses (Nicklaus Badyal et al., 2023)

{{<citation>}}

Nicklaus Badyal, Derek Jacoby, Yvonne Coady. (2023)  
**Intentional Biases in LLM Responses**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Bias, Falcon, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2311.07611v1)  

---


**ABSTRACT**  
In this study we intentionally introduce biases into large language model responses in an attempt to create specific personas for interactive media purposes. We explore the differences between open source models such as Falcon-7b and the GPT-4 model from Open AI, and we quantify some differences in responses afforded by the two systems. We find that the guardrails in the GPT-4 mixture of experts models with a supervisor, while useful in assuring AI alignment in general, are detrimental in trying to construct personas with a variety of uncommon viewpoints. This study aims to set the groundwork for future exploration in intentional biases of large language models such that these practices can be applied in the creative field, and new forms of media.

{{</citation>}}


### (25/52) Robust Text Classification: Analyzing Prototype-Based Networks (Zhivar Sourati et al., 2023)

{{<citation>}}

Zhivar Sourati, Darshan Deshpande, Filip Ilievski, Kiril Gashteovski, Sascha Saralajew. (2023)  
**Robust Text Classification: Analyzing Prototype-Based Networks**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP, Text Classification  
[Paper Link](http://arxiv.org/abs/2311.06647v1)  

---


**ABSTRACT**  
Downstream applications often require text classification models to be accurate, robust, and interpretable. While the accuracy of the stateof-the-art language models approximates human performance, they are not designed to be interpretable and often exhibit a drop in performance on noisy data. The family of PrototypeBased Networks (PBNs) that classify examples based on their similarity to prototypical examples of a class (prototypes) is natively interpretable and shown to be robust to noise, which enabled its wide usage for computer vision tasks. In this paper, we study whether the robustness properties of PBNs transfer to text classification tasks. We design a modular and comprehensive framework for studying PBNs, which includes different backbone architectures, backbone sizes, and objective functions. Our evaluation protocol assesses the robustness of models against character-, word-, and sentence-level perturbations. Our experiments on three benchmarks show that the robustness of PBNs transfers to NLP classification tasks facing realistic perturbations. Moreover, the robustness of PBNs is supported mostly by the objective function that keeps prototypes interpretable, while the robustness superiority of PBNs over vanilla models becomes more salient as datasets get more complex.

{{</citation>}}


### (26/52) BizBench: A Quantitative Reasoning Benchmark for Business and Finance (Rik Koncel-Kedziorski et al., 2023)

{{<citation>}}

Rik Koncel-Kedziorski, Michael Krumdick, Viet Lai, Varshini Reddy, Charles Lovering, Chris Tanner. (2023)  
**BizBench: A Quantitative Reasoning Benchmark for Business and Finance**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP, QA, Reasoning  
[Paper Link](http://arxiv.org/abs/2311.06602v1)  

---


**ABSTRACT**  
As large language models (LLMs) impact a growing number of complex domains, it is becoming increasingly important to have fair, accurate, and rigorous evaluation benchmarks. Evaluating the reasoning skills required for business and financial NLP stands out as a particularly difficult challenge. We introduce BizBench, a new benchmark for evaluating models' ability to reason about realistic financial problems. BizBench comprises 8 quantitative reasoning tasks. Notably, BizBench targets the complex task of question-answering (QA) for structured and unstructured financial data via program synthesis (i.e., code generation). We introduce three diverse financially-themed code-generation tasks from newly collected and augmented QA data. Additionally, we isolate distinct financial reasoning capabilities required to solve these QA tasks: reading comprehension of financial text and tables, which is required to extract correct intermediate values; and understanding domain knowledge (e.g., financial formulas) needed to calculate complex solutions. Collectively, these tasks evaluate a model's financial background knowledge, ability to extract numeric entities from financial documents, and capacity to solve problems with code. We conduct an in-depth evaluation of open-source and commercial LLMs, illustrating that BizBench is a challenging benchmark for quantitative reasoning in the finance and business domain.

{{</citation>}}


### (27/52) From Classification to Generation: Insights into Crosslingual Retrieval Augmented ICL (Xiaoqian Li et al., 2023)

{{<citation>}}

Xiaoqian Li, Ercong Nie, Sheng Liang. (2023)  
**From Classification to Generation: Insights into Crosslingual Retrieval Augmented ICL**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.06595v2)  

---


**ABSTRACT**  
The remarkable ability of Large Language Models (LLMs) to understand and follow instructions has sometimes been limited by their in-context learning (ICL) performance in low-resource languages. To address this, we introduce a novel approach that leverages cross-lingual retrieval-augmented in-context learning (CREA-ICL). By extracting semantically similar prompts from high-resource languages, we aim to improve the zero-shot performance of multilingual pre-trained language models (MPLMs) across diverse tasks. Though our approach yields steady improvements in classification tasks, it faces challenges in generation tasks. Our evaluation offers insights into the performance dynamics of retrieval-augmented in-context learning across both classification and generation domains.

{{</citation>}}


### (28/52) Heuristics-Driven Link-of-Analogy Prompting: Enhancing Large Language Models for Document-Level Event Argument Extraction (Hanzhang Zhou et al., 2023)

{{<citation>}}

Hanzhang Zhou, Junlang Qian, Zijian Feng, Hui Lu, Zixiao Zhu, Kezhi Mao. (2023)  
**Heuristics-Driven Link-of-Analogy Prompting: Enhancing Large Language Models for Document-Level Event Argument Extraction**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.06555v1)  

---


**ABSTRACT**  
In this study, we investigate in-context learning (ICL) in document-level event argument extraction (EAE). The paper identifies key challenges in this problem, including example selection, context length limitation, abundance of event types, and the limitation of Chain-of-Thought (CoT) prompting in non-reasoning tasks. To address these challenges, we introduce the Heuristic-Driven Link-of-Analogy (HD-LoA) prompting method. Specifically, we hypothesize and validate that LLMs learn task-specific heuristics from demonstrations via ICL. Building upon this hypothesis, we introduce an explicit heuristic-driven demonstration construction approach, which transforms the haphazard example selection process into a methodical method that emphasizes task heuristics. Additionally, inspired by the analogical reasoning of human, we propose the link-of-analogy prompting, which enables LLMs to process new situations by drawing analogies to known situations, enhancing their adaptability. Extensive experiments show that our method outperforms the existing prompting methods and few-shot supervised learning methods, exhibiting F1 score improvements of 4.53% and 9.38% on the document-level EAE dataset. Furthermore, when applied to sentiment analysis and natural language inference tasks, the HD-LoA prompting achieves accuracy gains of 2.87% and 2.63%, indicating its effectiveness across different tasks.

{{</citation>}}


### (29/52) Zero-Shot Cross-Lingual Sentiment Classification under Distribution Shift: an Exploratory Study (Maarten De Raedt et al., 2023)

{{<citation>}}

Maarten De Raedt, Semere Kiros Bitew, Fréderic Godin, Thomas Demeester, Chris Develder. (2023)  
**Zero-Shot Cross-Lingual Sentiment Classification under Distribution Shift: an Exploratory Study**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Amazon, BERT, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2311.06549v1)  

---


**ABSTRACT**  
The brittleness of finetuned language model performance on out-of-distribution (OOD) test samples in unseen domains has been well-studied for English, yet is unexplored for multi-lingual models. Therefore, we study generalization to OOD test data specifically in zero-shot cross-lingual transfer settings, analyzing performance impacts of both language and domain shifts between train and test data. We further assess the effectiveness of counterfactually augmented data (CAD) in improving OOD generalization for the cross-lingual setting, since CAD has been shown to benefit in a monolingual English setting. Finally, we propose two new approaches for OOD generalization that avoid the costly annotation process associated with CAD, by exploiting the power of recent large language models (LLMs). We experiment with 3 multilingual models, LaBSE, mBERT, and XLM-R trained on English IMDb movie reviews, and evaluate on OOD test sets in 13 languages: Amazon product reviews, Tweets, and Restaurant reviews. Results echo the OOD performance decline observed in the monolingual English setting. Further, (i) counterfactuals from the original high-resource language do improve OOD generalization in the low-resource language, and (ii) our newly proposed cost-effective approaches reach similar or up to +3.1% better accuracy than CAD for Amazon and Restaurant reviews.

{{</citation>}}


### (30/52) Enhancing Public Understanding of Court Opinions with Automated Summarizers (Elliott Ash et al., 2023)

{{<citation>}}

Elliott Ash, Aniket Kesari, Suresh Naidu, Lena Song, Dominik Stammbach. (2023)  
**Enhancing Public Understanding of Court Opinions with Automated Summarizers**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.06534v1)  

---


**ABSTRACT**  
Written judicial opinions are an important tool for building public trust in court decisions, yet they can be difficult for non-experts to understand. We present a pipeline for using an AI assistant to generate simplified summaries of judicial opinions. These are more accessible to the public and more easily understood by non-experts, We show in a survey experiment that the simplified summaries help respondents understand the key features of a ruling. We discuss how to integrate legal domain knowledge into studies using large language models. Our results suggest a role both for AI assistants to inform the public, and for lawyers to guide the process of generating accessible summaries.

{{</citation>}}


### (31/52) Added Toxicity Mitigation at Inference Time for Multimodal and Massively Multilingual Translation (Marta R. Costa-jussà et al., 2023)

{{<citation>}}

Marta R. Costa-jussà, David Dale, Maha Elbayad, Bokai Yu. (2023)  
**Added Toxicity Mitigation at Inference Time for Multimodal and Massively Multilingual Translation**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-CL, cs.CL  
Keywords: Multilingual  
[Paper Link](http://arxiv.org/abs/2311.06532v1)  

---


**ABSTRACT**  
Added toxicity in the context of translation refers to the fact of producing a translation output with more toxicity than there exists in the input. In this paper, we present MinTox which is a novel pipeline to identify added toxicity and mitigate this issue which works at inference time. MinTox uses a toxicity detection classifier which is multimodal (speech and text) and works in languages at scale. The mitigation method is applied to languages at scale and directly in text outputs. MinTox is applied to SEAMLESSM4T, which is the latest multimodal and massively multilingual machine translation system. For this system, MinTox achieves significant added toxicity mitigation across domains, modalities and language directions. MinTox manages to approximately filter out from 25% to 95% of added toxicity (depending on the modality and domain) while keeping translation quality.

{{</citation>}}


### (32/52) Step by Step to Fairness: Attributing Societal Bias in Task-oriented Dialogue Systems (Hsuan Su et al., 2023)

{{<citation>}}

Hsuan Su, Rebecca Qian, Chinnadhurai Sankar, Shahin Shayandeh, Shang-Tse Chen, Hung-yi Lee, Daniel M. Bikel. (2023)  
**Step by Step to Fairness: Attributing Societal Bias in Task-oriented Dialogue Systems**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Bias, Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2311.06513v2)  

---


**ABSTRACT**  
Recent works have shown considerable improvements in task-oriented dialogue (TOD) systems by utilizing pretrained large language models (LLMs) in an end-to-end manner. However, the biased behavior of each component in a TOD system and the error propagation issue in the end-to-end framework can lead to seriously biased TOD responses. Existing works of fairness only focus on the total bias of a system. In this paper, we propose a diagnosis method to attribute bias to each component of a TOD system. With the proposed attribution method, we can gain a deeper understanding of the sources of bias. Additionally, researchers can mitigate biased model behavior at a more granular level. We conduct experiments to attribute the TOD system's bias toward three demographic axes: gender, age, and race. Experimental results show that the bias of a TOD system usually comes from the response generation model.

{{</citation>}}


### (33/52) Knowledgeable Preference Alignment for LLMs in Domain-specific Question Answering (Yichi Zhang et al., 2023)

{{<citation>}}

Yichi Zhang, Zhuo Chen, Yin Fang, Lei Cheng, Yanxi Lu, Fangming Li, Wen Zhang, Huajun Chen. (2023)  
**Knowledgeable Preference Alignment for LLMs in Domain-specific Question Answering**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2311.06503v1)  

---


**ABSTRACT**  
Recently, the development of large language models (LLMs) has attracted wide attention in academia and industry. Deploying LLMs to real scenarios is one of the key directions in the current Internet industry. In this paper, we present a novel pipeline to apply LLMs for domain-specific question answering (QA) that incorporates domain knowledge graphs (KGs), addressing an important direction of LLM application. As a real-world application, the content generated by LLMs should be user-friendly to serve the customers. Additionally, the model needs to utilize domain knowledge properly to generate reliable answers. These two issues are the two major difficulties in the LLM application as vanilla fine-tuning can not adequately address them. We think both requirements can be unified as the model preference problem that needs to align with humans to achieve practical application. Thus, we introduce Knowledgeable Preference AlignmenT (KnowPAT), which constructs two kinds of preference set called style preference set and knowledge preference set respectively to tackle the two issues. Besides, we design a new alignment objective to align the LLM preference with human preference, aiming to train a better LLM for real-scenario domain-specific QA to generate reliable and user-friendly answers. Adequate experiments and comprehensive with 15 baseline methods demonstrate that our KnowPAT is an outperforming pipeline for real-scenario domain-specific QA with LLMs. Our code is open-source at https://github.com/zjukg/KnowPAT.

{{</citation>}}


### (34/52) L3 Ensembles: Lifelong Learning Approach for Ensemble of Foundational Language Models (Aidin Shiri et al., 2023)

{{<citation>}}

Aidin Shiri, Kaushik Roy, Amit Sheth, Manas Gaur. (2023)  
**L3 Ensembles: Lifelong Learning Approach for Ensemble of Foundational Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GLUE, Language Model, NLP, Natural Language Processing, SuperGLUE, T5  
[Paper Link](http://arxiv.org/abs/2311.06493v1)  

---


**ABSTRACT**  
Fine-tuning pre-trained foundational language models (FLM) for specific tasks is often impractical, especially for resource-constrained devices. This necessitates the development of a Lifelong Learning (L3) framework that continuously adapts to a stream of Natural Language Processing (NLP) tasks efficiently. We propose an approach that focuses on extracting meaningful representations from unseen data, constructing a structured knowledge base, and improving task performance incrementally. We conducted experiments on various NLP tasks to validate its effectiveness, including benchmarks like GLUE and SuperGLUE. We measured good performance across the accuracy, training efficiency, and knowledge transfer metrics. Initial experimental results show that the proposed L3 ensemble method increases the model accuracy by 4% ~ 36% compared to the fine-tuned FLM. Furthermore, L3 model outperforms naive fine-tuning approaches while maintaining competitive or superior performance (up to 15.4% increase in accuracy) compared to the state-of-the-art language model (T5) for the given task, STS benchmark.

{{</citation>}}


### (35/52) THOS: A Benchmark Dataset for Targeted Hate and Offensive Speech (Saad Almohaimeed et al., 2023)

{{<citation>}}

Saad Almohaimeed, Saleh Almohaimeed, Ashfaq Ali Shafin, Bogdan Carbunar, Ladislau Bölöni. (2023)  
**THOS: A Benchmark Dataset for Targeted Hate and Offensive Speech**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, Twitter  
[Paper Link](http://arxiv.org/abs/2311.06446v1)  

---


**ABSTRACT**  
Detecting harmful content on social media, such as Twitter, is made difficult by the fact that the seemingly simple yes/no classification conceals a significant amount of complexity. Unfortunately, while several datasets have been collected for training classifiers in hate and offensive speech, there is a scarcity of datasets labeled with a finer granularity of target classes and specific targets. In this paper, we introduce THOS, a dataset of 8.3k tweets manually labeled with fine-grained annotations about the target of the message. We demonstrate that this dataset makes it feasible to train classifiers, based on Large Language Models, to perform classification at this level of granularity.

{{</citation>}}


### (36/52) Separating the Wheat from the Chaff with BREAD: An open-source benchmark and metrics to detect redundancy in text (Isaac Caswell et al., 2023)

{{<citation>}}

Isaac Caswell, Lisa Wang, Isabel Papadimitriou. (2023)  
**Separating the Wheat from the Chaff with BREAD: An open-source benchmark and metrics to detect redundancy in text**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2311.06440v1)  

---


**ABSTRACT**  
Data quality is a problem that perpetually resurfaces throughout the field of NLP, regardless of task, domain, or architecture, and remains especially severe for lower-resource languages. A typical and insidious issue, affecting both training data and model output, is data that is repetitive and dominated by linguistically uninteresting boilerplate, such as price catalogs or computer-generated log files. Though this problem permeates many web-scraped corpora, there has yet to be a benchmark to test against, or a systematic study to find simple metrics that generalize across languages and agree with human judgements of data quality. In the present work, we create and release BREAD, a human-labeled benchmark on repetitive boilerplate vs. plausible linguistic content, spanning 360 languages. We release several baseline CRED (Character REDundancy) scores along with it, and evaluate their effectiveness on BREAD. We hope that the community will use this resource to develop better filtering methods, and that our reference implementations of CRED scores can become standard corpus evaluation tools, driving the development of cleaner language modeling corpora, especially in low-resource languages.

{{</citation>}}


## cs.AI (3)



### (37/52) TrainerAgent: Customizable and Efficient Model Training through LLM-Powered Multi-Agent System (Haoyuan Li et al., 2023)

{{<citation>}}

Haoyuan Li, Hao Jiang, Tianke Zhang, Zhelun Yu, Aoxiong Yin, Hao Cheng, Siming Fu, Yuhao Zhang, Wanggui He. (2023)  
**TrainerAgent: Customizable and Efficient Model Training through LLM-Powered Multi-Agent System**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2311.06622v1)  

---


**ABSTRACT**  
Training AI models has always been challenging, especially when there is a need for custom models to provide personalized services. Algorithm engineers often face a lengthy process to iteratively develop models tailored to specific business requirements, making it even more difficult for non-experts. The quest for high-quality and efficient model development, along with the emergence of Large Language Model (LLM) Agents, has become a key focus in the industry. Leveraging the powerful analytical, planning, and decision-making capabilities of LLM, we propose a TrainerAgent system comprising a multi-agent framework including Task, Data, Model and Server agents. These agents analyze user-defined tasks, input data, and requirements (e.g., accuracy, speed), optimizing them comprehensively from both data and model perspectives to obtain satisfactory models, and finally deploy these models as online service. Experimental evaluations on classical discriminative and generative tasks in computer vision and natural language processing domains demonstrate that our system consistently produces models that meet the desired criteria. Furthermore, the system exhibits the ability to critically identify and reject unattainable tasks, such as fantastical scenarios or unethical requests, ensuring robustness and safety. This research presents a significant advancement in achieving desired models with increased efficiency and quality as compared to traditional model development, facilitated by the integration of LLM-powered analysis, decision-making, and execution capabilities, as well as the collaboration among four agents. We anticipate that our work will contribute to the advancement of research on TrainerAgent in both academic and industry communities, potentially establishing it as a new paradigm for model development in the field of AI.

{{</citation>}}


### (38/52) An Intelligent Social Learning-based Optimization Strategy for Black-box Robotic Control with Reinforcement Learning (Xubo Yang et al., 2023)

{{<citation>}}

Xubo Yang, Jian Gao, Ting Wang, Yaozhen He. (2023)  
**An Intelligent Social Learning-based Optimization Strategy for Black-box Robotic Control with Reinforcement Learning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-NE, cs-RO, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.06576v1)  

---


**ABSTRACT**  
Implementing intelligent control of robots is a difficult task, especially when dealing with complex black-box systems, because of the lack of visibility and understanding of how these robots work internally. This paper proposes an Intelligent Social Learning (ISL) algorithm to enable intelligent control of black-box robotic systems. Inspired by mutual learning among individuals in human social groups, ISL includes learning, imitation, and self-study styles. Individuals in the learning style use the Levy flight search strategy to learn from the best performer and form the closest relationships. In the imitation style, individuals mimic the best performer with a second-level rapport by employing a random perturbation strategy. In the self-study style, individuals learn independently using a normal distribution sampling method while maintaining a distant relationship with the best performer. Individuals in the population are regarded as autonomous intelligent agents in each style. Neural networks perform strategic actions in three styles to interact with the environment and the robot and iteratively optimize the network policy. Overall, ISL builds on the principles of intelligent optimization, incorporating ideas from reinforcement learning, and possesses strong search capabilities, fast computation speed, fewer hyperparameters, and insensitivity to sparse rewards. The proposed ISL algorithm is compared with four state-of-the-art methods on six continuous control benchmark cases in MuJoCo to verify its effectiveness and advantages. Furthermore, ISL is adopted in the simulation and experimental grasping tasks of the UR3 robot for validations, and satisfactory solutions are yielded.

{{</citation>}}


### (39/52) Modeling Choice via Self-Attention (Joohwan Ko et al., 2023)

{{<citation>}}

Joohwan Ko, Andrew A. Li. (2023)  
**Modeling Choice via Self-Attention**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: Attention, Self-Attention  
[Paper Link](http://arxiv.org/abs/2311.07607v1)  

---


**ABSTRACT**  
Models of choice are a fundamental input to many now-canonical optimization problems in the field of Operations Management, including assortment, inventory, and price optimization. Naturally, accurate estimation of these models from data is a critical step in the application of these optimization problems in practice, and so it is perhaps surprising that such choice estimation has to now been accomplished almost exclusively, both in theory and in practice, (a) without the use of deep learning in any meaningful way, and (b) via evaluation on limited data with constantly-changing metrics. This is in stark contrast to the vast majority of similar learning applications, for which the practice of machine learning suggests that (a) neural network-based models are typically state-of-the-art, and (b) strict standardization on evaluation procedures (datasets, metrics, etc.) is crucial. Thus motivated, we first propose a choice model that is the first to successfully (both theoretically and practically) leverage a modern neural network architectural concept (self-attention). Theoretically, we show that our attention-based choice model is a low-rank generalization of the Halo Multinomial Logit model, a recent model that parsimoniously captures irrational choice effects and has seen empirical success. We prove that whereas the Halo-MNL requires $\Omega(m^2)$ data samples to estimate, where $m$ is the number of products, our model supports a natural nonconvex estimator (in particular, that which a standard neural network implementation would apply) which admits a near-optimal stationary point with $O(m)$ samples. We then establish the first realistic-scale benchmark for choice estimation on real data and use this benchmark to run the largest evaluation of existing choice models to date. We find that the model we propose is dominant over both short-term and long-term data periods.

{{</citation>}}


## cs.HC (1)



### (40/52) The Power of Attention: Bridging Cognitive Load, Multimedia Learning, and AI (Herbert dos Santos Macedo et al., 2023)

{{<citation>}}

Herbert dos Santos Macedo, Italo Thiago Felix dos Santos, Edgard Luciano Oliveira da Silva. (2023)  
**The Power of Attention: Bridging Cognitive Load, Multimedia Learning, and AI**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs-MM, cs.HC  
Keywords: AI, Attention, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.06586v1)  

---


**ABSTRACT**  
This article addresses the intersection of various educational theories and their relationship with the education of computer science students, with a focus on the importance of understanding computational thinking and its application in education. The historical context and fundamental concepts of Cognitive Load Theory, Multimedia Learning, and Constructivism are explored, highlighting their underlying biological assumptions about human learning. It also examines how these theories can be integrated with the use of Artificial Intelligence (AI) in education, with a particular emphasis on the attention mechanisms and abstract learning present in AI models like Transformers. Lastly, the relevance of these theories and practices for computer education student training is discussed, emphasizing how the development of computational thinking can contribute to a more effective approach in teaching and learning.

{{</citation>}}


## cs.PL (1)



### (41/52) Sparse Attention-Based Neural Networks for Code Classification (Ziyang Xiang et al., 2023)

{{<citation>}}

Ziyang Xiang, Zaixi Zhang, Qi Liu. (2023)  
**Sparse Attention-Based Neural Networks for Code Classification**  

---
Primary Category: cs.PL  
Categories: cs-LG, cs-PL, cs.PL  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2311.06575v1)  

---


**ABSTRACT**  
Categorizing source codes accurately and efficiently is a challenging problem in real-world programming education platform management. In recent years, model-based approaches utilizing abstract syntax trees (ASTs) have been widely applied to code classification tasks. We introduce an approach named the Sparse Attention-based neural network for Code Classification (SACC) in this paper. The approach involves two main steps: In the first step, source code undergoes syntax parsing and preprocessing. The generated abstract syntax tree is split into sequences of subtrees and then encoded using a recursive neural network to obtain a high-dimensional representation. This step simultaneously considers both the logical structure and lexical level information contained within the code. In the second step, the encoded sequences of subtrees are fed into a Transformer model that incorporates sparse attention mechanisms for the purpose of classification. This method efficiently reduces the computational cost of the self-attention mechanisms, thus improving the training speed while preserving effectiveness. Our work introduces a carefully designed sparse attention pattern that is specifically designed to meet the unique needs of code classification tasks. This design helps reduce the influence of redundant information and enhances the overall performance of the model. Finally, we also deal with problems in previous related research, which include issues like incomplete classification labels and a small dataset size. We annotated the CodeNet dataset with algorithm-related labeling categories, which contains a significantly large amount of data. Extensive comparative experimental results demonstrate the effectiveness and efficiency of SACC for the code classification tasks.

{{</citation>}}


## eess.IV (1)



### (42/52) Swin UNETR++: Advancing Transformer-Based Dense Dose Prediction Towards Fully Automated Radiation Oncology Treatments (Kuancheng Wang et al., 2023)

{{<citation>}}

Kuancheng Wang, Hai Siong Tan, Rafe Mcbeth. (2023)  
**Swin UNETR++: Advancing Transformer-Based Dense Dose Prediction Towards Fully Automated Radiation Oncology Treatments**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2311.06572v1)  

---


**ABSTRACT**  
The field of Radiation Oncology is uniquely positioned to benefit from the use of artificial intelligence to fully automate the creation of radiation treatment plans for cancer therapy. This time-consuming and specialized task combines patient imaging with organ and tumor segmentation to generate a 3D radiation dose distribution to meet clinical treatment goals, similar to voxel-level dense prediction. In this work, we propose Swin UNETR++, that contains a lightweight 3D Dual Cross-Attention (DCA) module to capture the intra and inter-volume relationships of each patient's unique anatomy, which fully convolutional neural networks lack. Our model was trained, validated, and tested on the Open Knowledge-Based Planning dataset. In addition to metrics of Dose Score $\overline{S_{\text{Dose}}}$ and DVH Score $\overline{S_{\text{DVH}}}$ that quantitatively measure the difference between the predicted and ground-truth 3D radiation dose distribution, we propose the qualitative metrics of average volume-wise acceptance rate $\overline{R_{\text{VA}}}$ and average patient-wise clinical acceptance rate $\overline{R_{\text{PA}}}$ to assess the clinical reliability of the predictions. Swin UNETR++ demonstrates near-state-of-the-art performance on validation and test dataset (validation: $\overline{S_{\text{DVH}}}$=1.492 Gy, $\overline{S_{\text{Dose}}}$=2.649 Gy, $\overline{R_{\text{VA}}}$=88.58%, $\overline{R_{\text{PA}}}$=100.0%; test: $\overline{S_{\text{DVH}}}$=1.634 Gy, $\overline{S_{\text{Dose}}}$=2.757 Gy, $\overline{R_{\text{VA}}}$=90.50%, $\overline{R_{\text{PA}}}$=98.0%), establishing a basis for future studies to translate 3D dose predictions into a deliverable treatment plan, facilitating full automation.

{{</citation>}}


## q-bio.QM (1)



### (43/52) Artificial Intelligence in Assessing Cardiovascular Diseases and Risk Factors via Retinal Fundus Images: A Review of the Last Decade (Mirsaeed Abdollahi et al., 2023)

{{<citation>}}

Mirsaeed Abdollahi, Ali Jafarizadeh, Amirhosein Ghafouri Asbagh, Navid Sobhi, Keysan Pourmoghtader, Siamak Pedrammehr, Houshyar Asadi, Roohallah Alizadehsani, Ru-San Tan, U. Rajendra Acharya. (2023)  
**Artificial Intelligence in Assessing Cardiovascular Diseases and Risk Factors via Retinal Fundus Images: A Review of the Last Decade**  

---
Primary Category: q-bio.QM  
Categories: J-3-2; J-3-3, cs-CV, eess-IV, physics-med-ph, q-bio-QM, q-bio.QM  
Keywords: AI, Google  
[Paper Link](http://arxiv.org/abs/2311.07609v1)  

---


**ABSTRACT**  
Background: Cardiovascular diseases (CVDs) continue to be the leading cause of mortality on a global scale. In recent years, the application of artificial intelligence (AI) techniques, particularly deep learning (DL), has gained considerable popularity for evaluating the various aspects of CVDs. Moreover, using fundus images and optical coherence tomography angiography (OCTA) to diagnose retinal diseases has been extensively studied. To better understand heart function and anticipate changes based on microvascular characteristics and function, researchers are currently exploring the integration of AI with non-invasive retinal scanning. Leveraging AI-assisted early detection and prediction of cardiovascular diseases on a large scale holds excellent potential to mitigate cardiovascular events and alleviate the economic burden on healthcare systems. Method: A comprehensive search was conducted across various databases, including PubMed, Medline, Google Scholar, Scopus, Web of Sciences, IEEE Xplore, and ACM Digital Library, using specific keywords related to cardiovascular diseases and artificial intelligence. Results: A total of 87 English-language publications, selected for relevance were included in the study, and additional references were considered. This study presents an overview of the current advancements and challenges in employing retinal imaging and artificial intelligence to identify cardiovascular disorders and provides insights for further exploration in this field. Conclusion: Researchers aim to develop precise disease prognosis patterns as the aging population and global CVD burden increase. AI and deep learning are transforming healthcare, offering the potential for single retinal image-based diagnosis of various CVDs, albeit with the need for accelerated adoption in healthcare systems.

{{</citation>}}


## physics.flu-dyn (1)



### (44/52) Identification of vortex in unstructured mesh with graph neural networks (Lianfa Wang et al., 2023)

{{<citation>}}

Lianfa Wang, Yvan Fournier, Jean-Francois Wald, Youssef Mesri. (2023)  
**Identification of vortex in unstructured mesh with graph neural networks**  

---
Primary Category: physics.flu-dyn  
Categories: cs-CV, physics-flu-dyn, physics.flu-dyn  
Keywords: GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2311.06557v1)  

---


**ABSTRACT**  
Deep learning has been employed to identify flow characteristics from Computational Fluid Dynamics (CFD) databases to assist the researcher to better understand the flow field, to optimize the geometry design and to select the correct CFD configuration for corresponding flow characteristics. Convolutional Neural Network (CNN) is one of the most popular algorithms used to extract and identify flow features. However its use, without any additional flow field interpolation, is limited to the simple domain geometry and regular meshes which limits its application to real industrial cases where complex geometry and irregular meshes are usually used. Aiming at the aforementioned problems, we present a Graph Neural Network (GNN) based model with U-Net architecture to identify the vortex in CFD results on unstructured meshes. The graph generation and graph hierarchy construction using algebraic multigrid method from CFD meshes are introduced. A vortex auto-labeling method is proposed to label vortex regions in 2D CFD meshes. We precise our approach by firstly optimizing the input set on CNNs, then benchmarking current GNN kernels against CNN model and evaluating the performances of GNN kernels in terms of classification accuracy, training efficiency and identified vortex morphology. Finally, we demonstrate the adaptability of our approach to unstructured meshes and generality to unseen cases with different turbulence models at different Reynolds numbers.

{{</citation>}}


## cs.SE (2)



### (45/52) How ChatGPT is Solving Vulnerability Management Problem (Peiyu Liu et al., 2023)

{{<citation>}}

Peiyu Liu, Junming Liu, Lirong Fu, Kangjie Lu, Yifan Xia, Xuhong Zhang, Wenzhi Chen, Haiqin Weng, Shouling Ji, Wenhai Wang. (2023)  
**How ChatGPT is Solving Vulnerability Management Problem**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-CL, cs-CR, cs-SE, cs.SE  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2311.06530v1)  

---


**ABSTRACT**  
Recently, ChatGPT has attracted great attention from the code analysis domain. Prior works show that ChatGPT has the capabilities of processing foundational code analysis tasks, such as abstract syntax tree generation, which indicates the potential of using ChatGPT to comprehend code syntax and static behaviors. However, it is unclear whether ChatGPT can complete more complicated real-world vulnerability management tasks, such as the prediction of security relevance and patch correctness, which require an all-encompassing understanding of various aspects, including code syntax, program semantics, and related manual comments.   In this paper, we explore ChatGPT's capabilities on 6 tasks involving the complete vulnerability management process with a large-scale dataset containing 78,445 samples. For each task, we compare ChatGPT against SOTA approaches, investigate the impact of different prompts, and explore the difficulties. The results suggest promising potential in leveraging ChatGPT to assist vulnerability management. One notable example is ChatGPT's proficiency in tasks like generating titles for software bug reports. Furthermore, our findings reveal the difficulties encountered by ChatGPT and shed light on promising future directions. For instance, directly providing random demonstration examples in the prompt cannot consistently guarantee good performance in vulnerability management. By contrast, leveraging ChatGPT in a self-heuristic way -- extracting expertise from demonstration examples itself and integrating the extracted expertise in the prompt is a promising research direction. Besides, ChatGPT may misunderstand and misuse the information in the prompt. Consequently, effectively guiding ChatGPT to focus on helpful information rather than the irrelevant content is still an open problem.

{{</citation>}}


### (46/52) Conceptual Model Interpreter for Large Language Models (Felix Härer, 2023)

{{<citation>}}

Felix Härer. (2023)  
**Conceptual Model Interpreter for Large Language Models**  

---
Primary Category: cs.SE  
Categories: I-2-1, cs-AI, cs-PL, cs-SE, cs.SE  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2311.07605v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) recently demonstrated capabilities for generating source code in common programming languages. Additionally, commercial products such as ChatGPT 4 started to provide code interpreters, allowing for the automatic execution of generated code fragments, instant feedback, and the possibility to develop and refine in a conversational fashion. With an exploratory research approach, this paper applies code generation and interpretation to conceptual models. The concept and prototype of a conceptual model interpreter is explored, capable of rendering visual models generated in textual syntax by state-of-the-art LLMs such as Llama~2 and ChatGPT 4. In particular, these LLMs can generate textual syntax for the PlantUML and Graphviz modeling software that is automatically rendered within a conversational user interface. The first result is an architecture describing the components necessary to interact with interpreters and LLMs through APIs or locally, providing support for many commercial and open source LLMs and interpreters. Secondly, experimental results for models generated with ChatGPT 4 and Llama 2 are discussed in two cases covering UML and, on an instance level, graphs created from custom data. The results indicate the possibility of modeling iteratively in a conversational fashion.

{{</citation>}}


## cs.DB (1)



### (47/52) Towards a Personal Health Knowledge Graph Framework for Patient Monitoring (Daniel Bloor et al., 2023)

{{<citation>}}

Daniel Bloor, Nnamdi Ugwuoke, David Taylor, Keir Lewis, Luis Mur, Chuan Lu. (2023)  
**Towards a Personal Health Knowledge Graph Framework for Patient Monitoring**  

---
Primary Category: cs.DB  
Categories: 68-06, E-1-3, cs-DB, cs.DB  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2311.06524v1)  

---


**ABSTRACT**  
Healthcare providers face significant challenges with monitoring and managing patient data outside of clinics, particularly with insufficient resources and limited feedback on their patients' conditions. Effective management of these symptoms and exploration of larger bodies of data are vital for maintaining long-term quality of life and preventing late interventions. In this paper, we propose a framework for constructing personal health knowledge graphs from heterogeneous data sources. Our approach integrates clinical databases, relevant ontologies and standard healthcare guidelines to support alert generation, clinician interpretation and querying of patient data. Through a use case of monitoring Chronic Obstructive Pulmonary Disease (COPD) patients, we demonstrate that inference and reasoning on personal health knowledge graphs built with our framework can aid in patient monitoring and enhance the efficacy and accuracy of patient data queries.

{{</citation>}}


## cs.NI (1)



### (48/52) Generative AI for Space-Air-Ground Integrated Networks (SAGIN) (Ruichen Zhang et al., 2023)

{{<citation>}}

Ruichen Zhang, Hongyang Du, Dusit Niyato, Jiawen Kang, Zehui Xiong, Abbas Jamalipour, Ping Zhang, Dong In Kim. (2023)  
**Generative AI for Space-Air-Ground Integrated Networks (SAGIN)**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI, eess-SP  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2311.06523v1)  

---


**ABSTRACT**  
Recently, generative AI technologies have emerged as a significant advancement in artificial intelligence field, renowned for their language and image generation capabilities. Meantime, space-air-ground integrated network (SAGIN) is an integral part of future B5G/6G for achieving ubiquitous connectivity. Inspired by this, this article explores an integration of generative AI in SAGIN, focusing on potential applications and case study. We first provide a comprehensive review of SAGIN and generative AI models, highlighting their capabilities and opportunities of their integration. Benefiting from generative AI's ability to generate useful data and facilitate advanced decision-making processes, it can be applied to various scenarios of SAGIN. Accordingly, we present a concise survey on their integration, including channel modeling and channel state information (CSI) estimation, joint air-space-ground resource allocation, intelligent network deployment, semantic communications, image extraction and processing, security and privacy enhancement. Next, we propose a framework that utilizes a Generative Diffusion Model (GDM) to construct channel information map to enhance quality of service for SAGIN. Simulation results demonstrate the effectiveness of the proposed framework. Finally, we discuss potential research directions for generative AI-enabled SAGIN.

{{</citation>}}


## cs.SI (1)



### (49/52) Dynamics of toxic behavior in the Covid-19 vaccination debate (Azza Bouleimen et al., 2023)

{{<citation>}}

Azza Bouleimen, Nicolò Pagan, Stefano Cresci, Aleksandra Urman, Silvia Giordano. (2023)  
**Dynamics of toxic behavior in the Covid-19 vaccination debate**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI, physics-soc-ph  
Keywords: Social Network  
[Paper Link](http://arxiv.org/abs/2311.06520v1)  

---


**ABSTRACT**  
In this paper, we study the behavior of users on Online Social Networks in the context of Covid-19 vaccines in Italy. We identify two main polarized communities: Provax and Novax. We find that Novax users are more active, more clustered in the network, and share less reliable information compared to the Provax users. On average, Novax are more toxic than Provax. However, starting from June 2021, the Provax became more toxic than the Novax. We show that the change in trend is explained by the aggregation of some contagion effects and the change in the activity level within communities. In fact, we establish that Provax users who increase their intensity of activity after May 2021 are significantly more toxic than the other users, shifting the toxicity up within the Provax community. Our study suggests that users presenting a spiky activity pattern tend to be more toxic.

{{</citation>}}


## cs.IT (1)



### (50/52) Knowledge Distillation and Training Balance for Heterogeneous Decentralized Multi-Modal Learning over Wireless Networks (Benshun Yin et al., 2023)

{{<citation>}}

Benshun Yin, Zhiyong Chen, Meixia Tao. (2023)  
**Knowledge Distillation and Training Balance for Heterogeneous Decentralized Multi-Modal Learning over Wireless Networks**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, math-IT  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2311.06500v1)  

---


**ABSTRACT**  
Decentralized learning is widely employed for collaboratively training models using distributed data over wireless networks. Existing decentralized learning methods primarily focus on training single-modal networks. For the decentralized multi-modal learning (DMML), the modality heterogeneity and the non-independent and non-identically distributed (non-IID) data across devices make it difficult for the training model to capture the correlated features across different modalities. Moreover, modality competition can result in training imbalance among different modalities, which can significantly impact the performance of DMML. To improve the training performance in the presence of non-IID data and modality heterogeneity, we propose a novel DMML with knowledge distillation (DMML-KD) framework, which decomposes the extracted feature into the modality-common and the modality-specific components. In the proposed DMML-KD, a generator is applied to learn the global conditional distribution of the modality-common features, thereby guiding the modality-common features of different devices towards the same distribution. Meanwhile, we propose to decrease the number of local iterations for the modalities with fast training speed in DMML-KD to address the imbalanced training. We design a balance metric based on the parameter variation to evaluate the training speed of different modalities in DMML-KD. Using this metric, we optimize the number of local iterations for different modalities on each device under the constraint of remaining energy on devices. Experimental results demonstrate that the proposed DMML-KD with training balance can effectively improve the training performance of DMML.

{{</citation>}}


## cs.CY (2)



### (51/52) Report of the 1st Workshop on Generative AI and Law (A. Feder Cooper et al., 2023)

{{<citation>}}

A. Feder Cooper, Katherine Lee, James Grimmelmann, Daphne Ippolito, Christopher Callison-Burch, Christopher A. Choquette-Choo, Niloofar Mireshghallah, Miles Brundage, David Mimno, Madiha Zahrah Choksi, Jack M. Balkin, Nicholas Carlini, Christopher De Sa, Jonathan Frankle, Deep Ganguli, Bryant Gipson, Andres Guadamuz, Swee Leng Harris, Abigail Z. Jacobs, Elizabeth Joh, Gautam Kamath, Mark Lemley, Cass Matthews, Christine McLeavey, Corynne McSherry, Milad Nasr, Paul Ohm, Adam Roberts, Tom Rubin, Pamela Samuelson, Ludwig Schubert, Kristen Vaccaro, Luis Villa, Felix Wu, Elana Zeide. (2023)  
**Report of the 1st Workshop on Generative AI and Law**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2311.06477v2)  

---


**ABSTRACT**  
This report presents the takeaways of the inaugural Workshop on Generative AI and Law (GenLaw), held in July 2023. A cross-disciplinary group of practitioners and scholars from computer science and law convened to discuss the technical, doctrinal, and policy challenges presented by law for Generative AI, and by Generative AI for law, with an emphasis on U.S. law in particular. We begin the report with a high-level statement about why Generative AI is both immensely significant and immensely challenging for law. To meet these challenges, we conclude that there is an essential need for 1) a shared knowledge base that provides a common conceptual language for experts across disciplines; 2) clarification of the distinctive technical capabilities of generative-AI systems, as compared and contrasted to other computer and AI systems; 3) a logical taxonomy of the legal issues these systems raise; and, 4) a concrete research agenda to promote collaboration and knowledge-sharing on emerging issues at the intersection of Generative AI and law. In this report, we synthesize the key takeaways from the GenLaw workshop that begin to address these needs. All of the listed authors contributed to the workshop upon which this report is based, but they and their organizations do not necessarily endorse all of the specific claims in this report.

{{</citation>}}


### (52/52) Online Advertisements with LLMs: Opportunities and Challenges (Soheil Feizi et al., 2023)

{{<citation>}}

Soheil Feizi, MohammadTaghi Hajiaghayi, Keivan Rezaei, Suho Shin. (2023)  
**Online Advertisements with LLMs: Opportunities and Challenges**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CY, cs.CY  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.07601v1)  

---


**ABSTRACT**  
This paper explores the potential for leveraging Large Language Models (LLM) in the realm of online advertising systems. We delve into essential requirements including privacy, latency, reliability, users and advertisers' satisfaction, which such a system must fulfill. We further introduce a general framework for LLM advertisement, consisting of modification, bidding, prediction, and auction modules. Different design considerations for each module is presented, with an in-depth examination of their practicality and the technical challenges inherent to their implementation.

{{</citation>}}
