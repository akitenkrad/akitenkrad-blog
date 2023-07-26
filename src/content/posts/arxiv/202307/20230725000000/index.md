---
draft: false
title: "arXiv @ 2023.07.25"
date: 2023-07-25
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.07.25"
    identifier: arxiv_20230725
    parent: 202307_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CV (11)](#cscv-11)
- [quant-ph (1)](#quant-ph-1)
- [cs.CL (8)](#cscl-8)
- [cs.SE (1)](#csse-1)
- [cs.NI (2)](#csni-2)
- [cs.LG (4)](#cslg-4)
- [cs.CR (1)](#cscr-1)
- [cs.SD (2)](#cssd-2)
- [eess.IV (1)](#eessiv-1)
- [cs.CY (1)](#cscy-1)

## cs.CV (11)



### (1/32) ProtoFL: Unsupervised Federated Learning via Prototypical Distillation (Hansol Kim et al., 2023)

{{<citation>}}

Hansol Kim, Youngjun Kwak, Minyoung Jung, Jinho Shin, Youngsung Kim, Changick Kim. (2023)  
**ProtoFL: Unsupervised Federated Learning via Prototypical Distillation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2307.12450v1)  

---


**ABSTRACT**  
Federated learning (FL) is a promising approach for enhancing data privacy preservation, particularly for authentication systems. However, limited round communications, scarce representation, and scalability pose significant challenges to its deployment, hindering its full potential. In this paper, we propose 'ProtoFL', Prototypical Representation Distillation based unsupervised Federated Learning to enhance the representation power of a global model and reduce round communication costs. Additionally, we introduce a local one-class classifier based on normalizing flows to improve performance with limited data. Our study represents the first investigation of using FL to improve one-class classification performance. We conduct extensive experiments on five widely used benchmarks, namely MNIST, CIFAR-10, CIFAR-100, ImageNet-30, and Keystroke-Dynamics, to demonstrate the superior performance of our proposed framework over previous methods in the literature.

{{</citation>}}


### (2/32) SwIPE: Efficient and Robust Medical Image Segmentation with Implicit Patch Embeddings (Yejia Zhang et al., 2023)

{{<citation>}}

Yejia Zhang, Pengfei Gu, Nishchal Sapkota, Danny Z. Chen. (2023)  
**SwIPE: Efficient and Robust Medical Image Segmentation with Implicit Patch Embeddings**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2307.12429v1)  

---


**ABSTRACT**  
Modern medical image segmentation methods primarily use discrete representations in the form of rasterized masks to learn features and generate predictions. Although effective, this paradigm is spatially inflexible, scales poorly to higher-resolution images, and lacks direct understanding of object shapes. To address these limitations, some recent works utilized implicit neural representations (INRs) to learn continuous representations for segmentation. However, these methods often directly adopted components designed for 3D shape reconstruction. More importantly, these formulations were also constrained to either point-based or global contexts, lacking contextual understanding or local fine-grained details, respectively--both critical for accurate segmentation. To remedy this, we propose a novel approach, SwIPE (Segmentation with Implicit Patch Embeddings), that leverages the advantages of INRs and predicts shapes at the patch level--rather than at the point level or image level--to enable both accurate local boundary delineation and global shape coherence. Extensive evaluations on two tasks (2D polyp segmentation and 3D abdominal organ segmentation) show that SwIPE significantly improves over recent implicit approaches and outperforms state-of-the-art discrete methods with over 10x fewer parameters. Our method also demonstrates superior data efficiency and improved robustness to data shifts across image resolutions and datasets. Code is available on Github.

{{</citation>}}


### (3/32) Augmented Box Replay: Overcoming Foreground Shift for Incremental Object Detection (Liu Yuyang et al., 2023)

{{<citation>}}

Liu Yuyang, Cong Yang, Goswami Dipam, Liu Xialei, Joost van de Weijer. (2023)  
**Augmented Box Replay: Overcoming Foreground Shift for Incremental Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2307.12427v1)  

---


**ABSTRACT**  
In incremental learning, replaying stored samples from previous tasks together with current task samples is one of the most efficient approaches to address catastrophic forgetting. However, unlike incremental classification, image replay has not been successfully applied to incremental object detection (IOD). In this paper, we identify the overlooked problem of foreground shift as the main reason for this. Foreground shift only occurs when replaying images of previous tasks and refers to the fact that their background might contain foreground objects of the current task. To overcome this problem, a novel and efficient Augmented Box Replay (ABR) method is developed that only stores and replays foreground objects and thereby circumvents the foreground shift problem. In addition, we propose an innovative Attentive RoI Distillation loss that uses spatial attention from region-of-interest (RoI) features to constrain current model to focus on the most important information from old model. ABR significantly reduces forgetting of previous classes while maintaining high plasticity in current classes. Moreover, it considerably reduces the storage requirements when compared to standard image replay. Comprehensive experiments on Pascal-VOC and COCO datasets support the state-of-the-art performance of our model.

{{</citation>}}


### (4/32) ComPtr: Towards Diverse Bi-source Dense Prediction Tasks via A Simple yet General Complementary Transformer (Youwei Pang et al., 2023)

{{<citation>}}

Youwei Pang, Xiaoqi Zhao, Lihe Zhang, Huchuan Lu. (2023)  
**ComPtr: Towards Diverse Bi-source Dense Prediction Tasks via A Simple yet General Complementary Transformer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.12349v1)  

---


**ABSTRACT**  
Deep learning (DL) has advanced the field of dense prediction, while gradually dissolving the inherent barriers between different tasks. However, most existing works focus on designing architectures and constructing visual cues only for the specific task, which ignores the potential uniformity introduced by the DL paradigm. In this paper, we attempt to construct a novel \underline{ComP}lementary \underline{tr}ansformer, \textbf{ComPtr}, for diverse bi-source dense prediction tasks. Specifically, unlike existing methods that over-specialize in a single task or a subset of tasks, ComPtr starts from the more general concept of bi-source dense prediction. Based on the basic dependence on information complementarity, we propose consistency enhancement and difference awareness components with which ComPtr can evacuate and collect important visual semantic cues from different image sources for diverse tasks, respectively. ComPtr treats different inputs equally and builds an efficient dense interaction model in the form of sequence-to-sequence on top of the transformer. This task-generic design provides a smooth foundation for constructing the unified model that can simultaneously deal with various bi-source information. In extensive experiments across several representative vision tasks, i.e. remote sensing change detection, RGB-T crowd counting, RGB-D/T salient object detection, and RGB-D semantic segmentation, the proposed method consistently obtains favorable performance. The code will be available at \url{https://github.com/lartpang/ComPtr}.

{{</citation>}}


### (5/32) Towards Generic and Controllable Attacks Against Object Detection (Guopeng Li et al., 2023)

{{<citation>}}

Guopeng Li, Yue Xu, Jian Ding, Gui-Song Xia. (2023)  
**Towards Generic and Controllable Attacks Against Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2307.12342v1)  

---


**ABSTRACT**  
Existing adversarial attacks against Object Detectors (ODs) suffer from two inherent limitations. Firstly, ODs have complicated meta-structure designs, hence most advanced attacks for ODs concentrate on attacking specific detector-intrinsic structures, which makes it hard for them to work on other detectors and motivates us to design a generic attack against ODs. Secondly, most works against ODs make Adversarial Examples (AEs) by generalizing image-level attacks from classification to detection, which brings redundant computations and perturbations in semantically meaningless areas (e.g., backgrounds) and leads to an emergency for seeking controllable attacks for ODs. To this end, we propose a generic white-box attack, LGP (local perturbations with adaptively global attacks), to blind mainstream object detectors with controllable perturbations. For a detector-agnostic attack, LGP tracks high-quality proposals and optimizes three heterogeneous losses simultaneously. In this way, we can fool the crucial components of ODs with a part of their outputs without the limitations of specific structures. Regarding controllability, we establish an object-wise constraint that exploits foreground-background separation adaptively to induce the attachment of perturbations to foregrounds. Experimentally, the proposed LGP successfully attacked sixteen state-of-the-art object detectors on MS-COCO and DOTA datasets, with promising imperceptibility and transferability obtained. Codes are publicly released in https://github.com/liguopeng0923/LGP.git

{{</citation>}}


### (6/32) TransHuman: A Transformer-based Human Representation for Generalizable Neural Human Rendering (Xiao Pan et al., 2023)

{{<citation>}}

Xiao Pan, Zongxin Yang, Jianxin Ma, Chang Zhou, Yi Yang. (2023)  
**TransHuman: A Transformer-based Human Representation for Generalizable Neural Human Rendering**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.12291v1)  

---


**ABSTRACT**  
In this paper, we focus on the task of generalizable neural human rendering which trains conditional Neural Radiance Fields (NeRF) from multi-view videos of different characters. To handle the dynamic human motion, previous methods have primarily used a SparseConvNet (SPC)-based human representation to process the painted SMPL. However, such SPC-based representation i) optimizes under the volatile observation space which leads to the pose-misalignment between training and inference stages, and ii) lacks the global relationships among human parts that is critical for handling the incomplete painted SMPL. Tackling these issues, we present a brand-new framework named TransHuman, which learns the painted SMPL under the canonical space and captures the global relationships between human parts with transformers. Specifically, TransHuman is mainly composed of Transformer-based Human Encoding (TransHE), Deformable Partial Radiance Fields (DPaRF), and Fine-grained Detail Integration (FDI). TransHE first processes the painted SMPL under the canonical space via transformers for capturing the global relationships between human parts. Then, DPaRF binds each output token with a deformable radiance field for encoding the query point under the observation space. Finally, the FDI is employed to further integrate fine-grained information from reference images. Extensive experiments on ZJU-MoCap and H36M show that our TransHuman achieves a significantly new state-of-the-art performance with high efficiency. Project page: https://pansanity666.github.io/TransHuman/

{{</citation>}}


### (7/32) DQ-Det: Learning Dynamic Query Combinations for Transformer-based Object Detection and Segmentation (Yiming Cui et al., 2023)

{{<citation>}}

Yiming Cui, Linjie Yang, Haichao Yu. (2023)  
**DQ-Det: Learning Dynamic Query Combinations for Transformer-based Object Detection and Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection, Transformer  
[Paper Link](http://arxiv.org/abs/2307.12239v1)  

---


**ABSTRACT**  
Transformer-based detection and segmentation methods use a list of learned detection queries to retrieve information from the transformer network and learn to predict the location and category of one specific object from each query. We empirically find that random convex combinations of the learned queries are still good for the corresponding models. We then propose to learn a convex combination with dynamic coefficients based on the high-level semantics of the image. The generated dynamic queries, named modulated queries, better capture the prior of object locations and categories in the different images. Equipped with our modulated queries, a wide range of DETR-based models achieve consistent and superior performance across multiple tasks including object detection, instance segmentation, panoptic segmentation, and video instance segmentation.

{{</citation>}}


### (8/32) EchoGLAD: Hierarchical Graph Neural Networks for Left Ventricle Landmark Detection on Echocardiograms (Masoud Mokhtari et al., 2023)

{{<citation>}}

Masoud Mokhtari, Mobina Mahdavi, Hooman Vaseli, Christina Luong, Purang Abolmaesumi, Teresa S. M. Tsang, Renjie Liao. (2023)  
**EchoGLAD: Hierarchical Graph Neural Networks for Left Ventricle Landmark Detection on Echocardiograms**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2307.12229v1)  

---


**ABSTRACT**  
The functional assessment of the left ventricle chamber of the heart requires detecting four landmark locations and measuring the internal dimension of the left ventricle and the approximate mass of the surrounding muscle. The key challenge of automating this task with machine learning is the sparsity of clinical labels, i.e., only a few landmark pixels in a high-dimensional image are annotated, leading many prior works to heavily rely on isotropic label smoothing. However, such a label smoothing strategy ignores the anatomical information of the image and induces some bias. To address this challenge, we introduce an echocardiogram-based, hierarchical graph neural network (GNN) for left ventricle landmark detection (EchoGLAD). Our main contributions are: 1) a hierarchical graph representation learning framework for multi-resolution landmark detection via GNNs; 2) induced hierarchical supervision at different levels of granularity using a multi-level loss. We evaluate our model on a public and a private dataset under the in-distribution (ID) and out-of-distribution (OOD) settings. For the ID setting, we achieve the state-of-the-art mean absolute errors (MAEs) of 1.46 mm and 1.86 mm on the two datasets. Our model also shows better OOD generalization than prior works with a testing MAE of 4.3 mm.

{{</citation>}}


### (9/32) Expediting Building Footprint Segmentation from High-resolution Remote Sensing Images via progressive lenient supervision (Haonan Guo et al., 2023)

{{<citation>}}

Haonan Guo, Bo Du, Chen Wu, Xin Su, Liangpei Zhang. (2023)  
**Expediting Building Footprint Segmentation from High-resolution Remote Sensing Images via progressive lenient supervision**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2307.12220v1)  

---


**ABSTRACT**  
The efficacy of building footprint segmentation from remotely sensed images has been hindered by model transfer effectiveness. Many existing building segmentation methods were developed upon the encoder-decoder architecture of U-Net, in which the encoder is finetuned from the newly developed backbone networks that are pre-trained on ImageNet. However, the heavy computational burden of the existing decoder designs hampers the successful transfer of these modern encoder networks to remote sensing tasks. Even the widely-adopted deep supervision strategy fails to mitigate these challenges due to its invalid loss in hybrid regions where foreground and background pixels are intermixed. In this paper, we conduct a comprehensive evaluation of existing decoder network designs for building footprint segmentation and propose an efficient framework denoted as BFSeg to enhance learning efficiency and effectiveness. Specifically, a densely-connected coarse-to-fine feature fusion decoder network that facilitates easy and fast feature fusion across scales is proposed. Moreover, considering the invalidity of hybrid regions in the down-sampled ground truth during the deep supervision process, we present a lenient deep supervision and distillation strategy that enables the network to learn proper knowledge from deep supervision. Building upon these advancements, we have developed a new family of building segmentation networks, which consistently surpass prior works with outstanding performance and efficiency across a wide range of newly developed encoder networks. The code will be released on https://github.com/HaonanGuo/BFSeg-Efficient-Building-Footprint-Segmentation-Framework.

{{</citation>}}


### (10/32) LoLep: Single-View View Synthesis with Locally-Learned Planes and Self-Attention Occlusion Inference (Cong Wang et al., 2023)

{{<citation>}}

Cong Wang, Yu-Ping Wang, Dinesh Manocha. (2023)  
**LoLep: Single-View View Synthesis with Locally-Learned Planes and Self-Attention Occlusion Inference**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Self-Attention  
[Paper Link](http://arxiv.org/abs/2307.12217v1)  

---


**ABSTRACT**  
We propose a novel method, LoLep, which regresses Locally-Learned planes from a single RGB image to represent scenes accurately, thus generating better novel views. Without the depth information, regressing appropriate plane locations is a challenging problem. To solve this issue, we pre-partition the disparity space into bins and design a disparity sampler to regress local offsets for multiple planes in each bin. However, only using such a sampler makes the network not convergent; we further propose two optimizing strategies that combine with different disparity distributions of datasets and propose an occlusion-aware reprojection loss as a simple yet effective geometric supervision technique. We also introduce a self-attention mechanism to improve occlusion inference and present a Block-Sampling Self-Attention (BS-SA) module to address the problem of applying self-attention to large feature maps. We demonstrate the effectiveness of our approach and generate state-of-the-art results on different datasets. Compared to MINE, our approach has an LPIPS reduction of 4.8%-9.0% and an RV reduction of 83.1%-84.7%. We also evaluate the performance on real-world images and demonstrate the benefits.

{{</citation>}}


### (11/32) LIST: Learning Implicitly from Spatial Transformers for Single-View 3D Reconstruction (Mohammad Samiul Arshad et al., 2023)

{{<citation>}}

Mohammad Samiul Arshad, William J. Beksi. (2023)  
**LIST: Learning Implicitly from Spatial Transformers for Single-View 3D Reconstruction**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-GR, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.12194v1)  

---


**ABSTRACT**  
Accurate reconstruction of both the geometric and topological details of a 3D object from a single 2D image embodies a fundamental challenge in computer vision. Existing explicit/implicit solutions to this problem struggle to recover self-occluded geometry and/or faithfully reconstruct topological shape structures. To resolve this dilemma, we introduce LIST, a novel neural architecture that leverages local and global image features to accurately reconstruct the geometric and topological structure of a 3D object from a single image. We utilize global 2D features to predict a coarse shape of the target object and then use it as a base for higher-resolution reconstruction. By leveraging both local 2D features from the image and 3D features from the coarse prediction, we can predict the signed distance between an arbitrary point and the target surface via an implicit predictor with great accuracy. Furthermore, our model does not require camera estimation or pixel alignment. It provides an uninfluenced reconstruction from the input-view direction. Through qualitative and quantitative analysis, we show the superiority of our model in reconstructing 3D objects from both synthetic and real-world images against the state of the art.

{{</citation>}}


## quant-ph (1)



### (12/32) WEPRO: Weight Prediction for Efficient Optimization of Hybrid Quantum-Classical Algorithms (Satwik Kundu et al., 2023)

{{<citation>}}

Satwik Kundu, Debarshi Kundu, Swaroop Ghosh. (2023)  
**WEPRO: Weight Prediction for Efficient Optimization of Hybrid Quantum-Classical Algorithms**  

---
Primary Category: quant-ph  
Categories: cs-LG, quant-ph, quant-ph  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2307.12449v1)  

---


**ABSTRACT**  
The exponential run time of quantum simulators on classical machines and long queue depths and high costs of real quantum devices present significant challenges in the effective training of Variational Quantum Algorithms (VQAs) like Quantum Neural Networks (QNNs), Variational Quantum Eigensolver (VQE) and Quantum Approximate Optimization Algorithm (QAOA). To address these limitations, we propose a new approach, WEPRO (Weight Prediction), which accelerates the convergence of VQAs by exploiting regular trends in the parameter weights. We introduce two techniques for optimal prediction performance namely, Naive Prediction (NaP) and Adaptive Prediction (AdaP). Through extensive experimentation and training of multiple QNN models on various datasets, we demonstrate that WEPRO offers a speedup of approximately $2.25\times$ compared to standard training methods, while also providing improved accuracy (up to $2.3\%$ higher) and loss (up to $6.1\%$ lower) with low storage and computational overheads. We also evaluate WEPRO's effectiveness in VQE for molecular ground-state energy estimation and in QAOA for graph MaxCut. Our results show that WEPRO leads to speed improvements of up to $3.1\times$ for VQE and $2.91\times$ for QAOA, compared to traditional optimization techniques, while using up to $3.3\times$ less number of shots (i.e., repeated circuit executions) per training iteration.

{{</citation>}}


## cs.CL (8)



### (13/32) On the Effectiveness of Offline RL for Dialogue Response Generation (Paloma Sodhi et al., 2023)

{{<citation>}}

Paloma Sodhi, Felix Wu, Ethan R. Elenberg, Kilian Q. Weinberger, Ryan McDonald. (2023)  
**On the Effectiveness of Offline RL for Dialogue Response Generation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2307.12425v1)  

---


**ABSTRACT**  
A common training technique for language models is teacher forcing (TF). TF attempts to match human language exactly, even though identical meanings can be expressed in different ways. This motivates use of sequence-level objectives for dialogue response generation. In this paper, we study the efficacy of various offline reinforcement learning (RL) methods to maximize such objectives. We present a comprehensive evaluation across multiple datasets, models, and metrics. Offline RL shows a clear performance improvement over teacher forcing while not inducing training instability or sacrificing practical training budgets.

{{</citation>}}


### (14/32) CommonsenseVIS: Visualizing and Understanding Commonsense Reasoning Capabilities of Natural Language Models (Xingbo Wang et al., 2023)

{{<citation>}}

Xingbo Wang, Renfei Huang, Zhihua Jin, Tianqing Fang, Huamin Qu. (2023)  
**CommonsenseVIS: Visualizing and Understanding Commonsense Reasoning Capabilities of Natural Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-HC, cs.CL  
Keywords: Language Model, NLP, Reasoning  
[Paper Link](http://arxiv.org/abs/2307.12382v1)  

---


**ABSTRACT**  
Recently, large pretrained language models have achieved compelling performance on commonsense benchmarks. Nevertheless, it is unclear what commonsense knowledge the models learn and whether they solely exploit spurious patterns. Feature attributions are popular explainability techniques that identify important input concepts for model outputs. However, commonsense knowledge tends to be implicit and rarely explicitly presented in inputs. These methods cannot infer models' implicit reasoning over mentioned concepts. We present CommonsenseVIS, a visual explanatory system that utilizes external commonsense knowledge bases to contextualize model behavior for commonsense question-answering. Specifically, we extract relevant commonsense knowledge in inputs as references to align model behavior with human knowledge. Our system features multi-level visualization and interactive model probing and editing for different concepts and their underlying relations. Through a user study, we show that CommonsenseVIS helps NLP experts conduct a systematic and scalable visual analysis of models' relational reasoning over concepts in different situations.

{{</citation>}}


### (15/32) In-Context Learning in Large Language Models Learns Label Relationships but Is Not Conventional Learning (Jannik Kossen et al., 2023)

{{<citation>}}

Jannik Kossen, Tom Rainforth, Yarin Gal. (2023)  
**In-Context Learning in Large Language Models Learns Label Relationships but Is Not Conventional Learning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.12375v1)  

---


**ABSTRACT**  
The performance of Large Language Models (LLMs) on downstream tasks often improves significantly when including examples of the input-label relationship in the context. However, there is currently no consensus about how this in-context learning (ICL) ability of LLMs works: for example, while Xie et al. (2021) liken ICL to a general-purpose learning algorithm, Min et al. (2022b) argue ICL does not even learn label relationships from in-context examples. In this paper, we study (1) how labels of in-context examples affect predictions, (2) how label relationships learned during pre-training interact with input-label examples provided in-context, and (3) how ICL aggregates label information across in-context examples. Our findings suggests LLMs usually incorporate information from in-context labels, but that pre-training and in-context label relationships are treated differently, and that the model does not consider all in-context information equally. Our results give insights into understanding and aligning LLM behavior.

{{</citation>}}


### (16/32) Evaluating Emotional Nuances in Dialogue Summarization (Yongxin Zhou et al., 2023)

{{<citation>}}

Yongxin Zhou, Fabien Ringeval, François Portet. (2023)  
**Evaluating Emotional Nuances in Dialogue Summarization**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue, Summarization  
[Paper Link](http://arxiv.org/abs/2307.12371v1)  

---


**ABSTRACT**  
Automatic dialogue summarization is a well-established task that aims to identify the most important content from human conversations to create a short textual summary. Despite recent progress in the field, we show that most of the research has focused on summarizing the factual information, leaving aside the affective content, which can yet convey useful information to analyse, monitor, or support human interactions. In this paper, we propose and evaluate a set of measures $PEmo$, to quantify how much emotion is preserved in dialog summaries. Results show that, summarization models of the state-of-the-art do not preserve well the emotional content in the summaries. We also show that by reducing the training set to only emotional dialogues, the emotional content is better preserved in the generated summaries, while conserving the most salient factual information.

{{</citation>}}


### (17/32) X-CapsNet For Fake News Detection (Mohammad Hadi Goldani et al., 2023)

{{<citation>}}

Mohammad Hadi Goldani, Reza Safabakhsh, Saeedeh Momtazi. (2023)  
**X-CapsNet For Fake News Detection**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs.CL  
Keywords: Fake News  
[Paper Link](http://arxiv.org/abs/2307.12332v1)  

---


**ABSTRACT**  
News consumption has significantly increased with the growing popularity and use of web-based forums and social media. This sets the stage for misinforming and confusing people. To help reduce the impact of misinformation on users' potential health-related decisions and other intents, it is desired to have machine learning models to detect and combat fake news automatically. This paper proposes a novel transformer-based model using Capsule neural Networks(CapsNet) called X-CapsNet. This model includes a CapsNet with dynamic routing algorithm paralyzed with a size-based classifier for detecting short and long fake news statements. We use two size-based classifiers, a Deep Convolutional Neural Network (DCNN) for detecting long fake news statements and a Multi-Layer Perceptron (MLP) for detecting short news statements. To resolve the problem of representing short news statements, we use indirect features of news created by concatenating the vector of news speaker profiles and a vector of polarity, sentiment, and counting words of news statements. For evaluating the proposed architecture, we use the Covid-19 and the Liar datasets. The results in terms of the F1-score for the Covid-19 dataset and accuracy for the Liar dataset show that models perform better than the state-of-the-art baselines.

{{</citation>}}


### (18/32) Towards Automatic Boundary Detection for Human-AI Hybrid Essay in Education (Zijie Zeng et al., 2023)

{{<citation>}}

Zijie Zeng, Lele Sha, Yuheng Li, Kaixun Yang, Dragan Gašević, Guanliang Chen. (2023)  
**Towards Automatic Boundary Detection for Human-AI Hybrid Essay in Education**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2307.12267v1)  

---


**ABSTRACT**  
Human-AI collaborative writing has been greatly facilitated with the help of modern large language models (LLM), e.g., ChatGPT. While admitting the convenience brought by technology advancement, educators also have concerns that students might leverage LLM to partially complete their writing assignment and pass off the human-AI hybrid text as their original work. Driven by such concerns, in this study, we investigated the automatic detection of Human-AI hybrid text in education, where we formalized the hybrid text detection as a boundary detection problem, i.e., identifying the transition points between human-written content and AI-generated content. We constructed a hybrid essay dataset by partially removing sentences from the original student-written essays and then instructing ChatGPT to fill in for the incomplete essays. Then we proposed a two-step detection approach where we (1) Separated AI-generated content from human-written content during the embedding learning process; and (2) Calculated the distances between every two adjacent prototypes (a prototype is the mean of a set of consecutive sentences from the hybrid text in the embedding space) and assumed that the boundaries exist between the two prototypes that have the furthest distance from each other. Through extensive experiments, we summarized the following main findings: (1) The proposed approach consistently outperformed the baseline methods across different experiment settings; (2) The embedding learning process (i.e., step 1) can significantly boost the performance of the proposed approach; (3) When detecting boundaries for single-boundary hybrid essays, the performance of the proposed approach could be enhanced by adopting a relatively large prototype size, leading to a $22$\% improvement (against the second-best baseline method) in the in-domain setting and an $18$\% improvement in the out-of-domain setting.

{{</citation>}}


### (19/32) Transformer-based Joint Source Channel Coding for Textual Semantic Communication (Shicong Liu et al., 2023)

{{<citation>}}

Shicong Liu, Zhen Gao, Gaojie Chen, Yu Su, Lu Peng. (2023)  
**Transformer-based Joint Source Channel Coding for Textual Semantic Communication**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL, eess-SP  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.12266v1)  

---


**ABSTRACT**  
The Space-Air-Ground-Sea integrated network calls for more robust and secure transmission techniques against jamming. In this paper, we propose a textual semantic transmission framework for robust transmission, which utilizes the advanced natural language processing techniques to model and encode sentences. Specifically, the textual sentences are firstly split into tokens using wordpiece algorithm, and are embedded to token vectors for semantic extraction by Transformer-based encoder. The encoded data are quantized to a fixed length binary sequence for transmission, where binary erasure, symmetric, and deletion channels are considered for transmission. The received binary sequences are further decoded by the transformer decoders into tokens used for sentence reconstruction. Our proposed approach leverages the power of neural networks and attention mechanism to provide reliable and efficient communication of textual data in challenging wireless environments, and simulation results on semantic similarity and bilingual evaluation understudy prove the superiority of the proposed model in semantic transmission.

{{</citation>}}


### (20/32) FATRER: Full-Attention Topic Regularizer for Accurate and Robust Conversational Emotion Recognition (Yuzhao Mao et al., 2023)

{{<citation>}}

Yuzhao Mao, Di Lu, Xiaojie Wang, Yang Zhang. (2023)  
**FATRER: Full-Attention Topic Regularizer for Accurate and Robust Conversational Emotion Recognition**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Attention, Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2307.12221v1)  

---


**ABSTRACT**  
This paper concentrates on the understanding of interlocutors' emotions evoked in conversational utterances. Previous studies in this literature mainly focus on more accurate emotional predictions, while ignoring model robustness when the local context is corrupted by adversarial attacks. To maintain robustness while ensuring accuracy, we propose an emotion recognizer augmented by a full-attention topic regularizer, which enables an emotion-related global view when modeling the local context in a conversation. A joint topic modeling strategy is introduced to implement regularization from both representation and loss perspectives. To avoid over-regularization, we drop the constraints on prior distributions that exist in traditional topic modeling and perform probabilistic approximations based entirely on attention alignment. Experiments show that our models obtain more favorable results than state-of-the-art models, and gain convincing robustness under three types of adversarial attacks.

{{</citation>}}


## cs.SE (1)



### (21/32) Testing Hateful Speeches against Policies (Jiangrui Zheng et al., 2023)

{{<citation>}}

Jiangrui Zheng, Xueqing Liu, Girish Budhrani, Wei Yang, Ravishka Rathnasuriya. (2023)  
**Testing Hateful Speeches against Policies**  

---
Primary Category: cs.SE  
Categories: cs-CL, cs-SE, cs.SE  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.12418v1)  

---


**ABSTRACT**  
In the recent years, many software systems have adopted AI techniques, especially deep learning techniques. Due to their black-box nature, AI-based systems brought challenges to traceability, because AI system behaviors are based on models and data, whereas the requirements or policies are rules in the form of natural or programming language. To the best of our knowledge, there is a limited amount of studies on how AI and deep neural network-based systems behave against rule-based requirements/policies. This experience paper examines deep neural network behaviors against rule-based requirements described in natural language policies. In particular, we focus on a case study to check AI-based content moderation software against content moderation policies. First, using crowdsourcing, we collect natural language test cases which match each moderation policy, we name this dataset HateModerate; second, using the test cases in HateModerate, we test the failure rates of state-of-the-art hate speech detection software, and we find that these models have high failure rates for certain policies; finally, since manual labeling is costly, we further proposed an automated approach to augument HateModerate by finetuning OpenAI's large language models to automatically match new examples to policies. The dataset and code of this work can be found on our anonymous website: \url{https://sites.google.com/view/content-moderation-project}.

{{</citation>}}


## cs.NI (2)



### (22/32) Practical Commercial 5G Standalone (SA) Uplink Throughput Prediction (Kasidis Arunruangsirilert et al., 2023)

{{<citation>}}

Kasidis Arunruangsirilert, Jiro Katto. (2023)  
**Practical Commercial 5G Standalone (SA) Uplink Throughput Prediction**  

---
Primary Category: cs.NI  
Categories: cs-LG, cs-NI, cs.NI, eess-SP  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2307.12417v1)  

---


**ABSTRACT**  
While the 5G New Radio (NR) network promises a huge uplift of the uplink throughput, the improvement can only be seen when the User Equipment (UE) is connected to the high-frequency millimeter wave (mmWave) band. With the rise of uplink-intensive smartphone applications such as the real-time transmission of UHD 4K/8K videos, and Virtual Reality (VR)/Augmented Reality (AR) contents, uplink throughput prediction plays a huge role in maximizing the users' quality of experience (QoE). In this paper, we propose using a ConvLSTM-based neural network to predict the future uplink throughput based on past uplink throughput and RF parameters. The network is trained using the data from real-world drive tests on commercial 5G SA networks while riding commuter trains, which accounted for various frequency bands, handover, and blind spots. To make sure our model can be practically implemented, we then limited our model to only use the information available via Android API, then evaluate our model using the data from both commuter trains and other methods of transportation. The results show that our model reaches an average prediction accuracy of 98.9\% with an average RMSE of 1.80 Mbps across all unseen evaluation scenarios.

{{</citation>}}


### (23/32) Semantic Communication-Empowered Traffic Management using Vehicle Count Prediction (Sachin Kadam et al., 2023)

{{<citation>}}

Sachin Kadam, Dong In Kim. (2023)  
**Semantic Communication-Empowered Traffic Management using Vehicle Count Prediction**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI, eess-SP  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2307.12254v1)  

---


**ABSTRACT**  
Vehicle count prediction is an important aspect of smart city traffic management. Most major roads are monitored by cameras with computing and transmitting capabilities. These cameras provide data to the central traffic controller (CTC), which is in charge of traffic control management. In this paper, we propose a joint CNN-LSTM-based semantic communication (SemCom) model in which the semantic encoder of a camera extracts the relevant semantics from raw images. The encoded semantics are then sent to the CTC by the transmitter in the form of symbols. The semantic decoder of the CTC predicts the vehicle count on each road based on the sequence of received symbols and develops a traffic management strategy accordingly. An optimization problem to improve the quality of experience (QoE) is introduced and numerically solved, taking into account constraints such as vehicle user safety, transmit power of camera devices, vehicle count prediction accuracy, and semantic entropy. Using numerical results, we show that the proposed SemCom model reduces overhead by $54.42\%$ when compared to source encoder/decoder methods. Also, we demonstrate through simulations that the proposed model outperforms state-of-the-art models in terms of mean absolute error (MAE) and QoE.

{{</citation>}}


## cs.LG (4)



### (24/32) Uncertainty-aware Grounded Action Transformation towards Sim-to-Real Transfer for Traffic Signal Control (Longchao Da et al., 2023)

{{<citation>}}

Longchao Da, Hao Mei, Romir Sharma, Hua Wei. (2023)  
**Uncertainty-aware Grounded Action Transformation towards Sim-to-Real Transfer for Traffic Signal Control**  

---
Primary Category: cs.LG  
Categories: H-4-0, cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.12388v1)  

---


**ABSTRACT**  
Traffic signal control (TSC) is a complex and important task that affects the daily lives of millions of people. Reinforcement Learning (RL) has shown promising results in optimizing traffic signal control, but current RL-based TSC methods are mainly trained in simulation and suffer from the performance gap between simulation and the real world. In this paper, we propose a simulation-to-real-world (sim-to-real) transfer approach called UGAT, which transfers a learned policy trained from a simulated environment to a real-world environment by dynamically transforming actions in the simulation with uncertainty to mitigate the domain gap of transition dynamics. We evaluate our method on a simulated traffic environment and show that it significantly improves the performance of the transferred RL policy in the real world.

{{</citation>}}


### (25/32) TabADM: Unsupervised Tabular Anomaly Detection with Diffusion Models (Guy Zamberg et al., 2023)

{{<citation>}}

Guy Zamberg, Moshe Salhov, Ofir Lindenbaum, Amir Averbuch. (2023)  
**TabADM: Unsupervised Tabular Anomaly Detection with Diffusion Models**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2307.12336v1)  

---


**ABSTRACT**  
Tables are an abundant form of data with use cases across all scientific fields. Real-world datasets often contain anomalous samples that can negatively affect downstream analysis. In this work, we only assume access to contaminated data and present a diffusion-based probabilistic model effective for unsupervised anomaly detection. Our model is trained to learn the density of normal samples by utilizing a unique rejection scheme to attenuate the influence of anomalies on the density estimation. At inference, we identify anomalies as samples in low-density regions. We use real data to demonstrate that our method improves detection capabilities over baselines. Furthermore, our method is relatively stable to the dimension of the data and does not require extensive hyperparameter tuning.

{{</citation>}}


### (26/32) Geometry-Aware Adaptation for Pretrained Models (Nicholas Roberts et al., 2023)

{{<citation>}}

Nicholas Roberts, Xintong Li, Dyah Adila, Sonia Cromp, Tzu-Heng Huang, Jitian Zhao, Frederic Sala. (2023)  
**Geometry-Aware Adaptation for Pretrained Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, stat-ML  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2307.12226v1)  

---


**ABSTRACT**  
Machine learning models -- including prominent zero-shot models -- are often trained on datasets whose labels are only a small proportion of a larger label space. Such spaces are commonly equipped with a metric that relates the labels via distances between them. We propose a simple approach to exploit this information to adapt the trained model to reliably predict new classes -- or, in the case of zero-shot prediction, to improve its performance -- without any additional training. Our technique is a drop-in replacement of the standard prediction rule, swapping argmax with the Fr\'echet mean. We provide a comprehensive theoretical analysis for this approach, studying (i) learning-theoretic results trading off label space diameter, sample complexity, and model dimension, (ii) characterizations of the full range of scenarios in which it is possible to predict any unobserved class, and (iii) an optimal active learning-like next class selection procedure to obtain optimal training classes for when it is not possible to predict the entire range of unobserved classes. Empirically, using easily-available external metrics, our proposed approach, Loki, gains up to 29.7% relative improvement over SimCLR on ImageNet and scales to hundreds of thousands of classes. When no such metric is available, Loki can use self-derived metrics from class embeddings and obtains a 10.5% improvement on pretrained zero-shot models such as CLIP.

{{</citation>}}


### (27/32) Adversarial Agents For Attacking Inaudible Voice Activated Devices (Forrest McKee et al., 2023)

{{<citation>}}

Forrest McKee, David Noever. (2023)  
**Adversarial Agents For Attacking Inaudible Voice Activated Devices**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SD, cs.LG, eess-AS  
Keywords: Microsoft  
[Paper Link](http://arxiv.org/abs/2307.12204v2)  

---


**ABSTRACT**  
The paper applies reinforcement learning to novel Internet of Thing configurations. Our analysis of inaudible attacks on voice-activated devices confirms the alarming risk factor of 7.6 out of 10, underlining significant security vulnerabilities scored independently by NIST National Vulnerability Database (NVD). Our baseline network model showcases a scenario in which an attacker uses inaudible voice commands to gain unauthorized access to confidential information on a secured laptop. We simulated many attack scenarios on this baseline network model, revealing the potential for mass exploitation of interconnected devices to discover and own privileged information through physical access without adding new hardware or amplifying device skills. Using Microsoft's CyberBattleSim framework, we evaluated six reinforcement learning algorithms and found that Deep-Q learning with exploitation proved optimal, leading to rapid ownership of all nodes in fewer steps. Our findings underscore the critical need for understanding non-conventional networks and new cybersecurity measures in an ever-expanding digital landscape, particularly those characterized by mobile devices, voice activation, and non-linear microphones susceptible to malicious actors operating stealth attacks in the near-ultrasound or inaudible ranges. By 2024, this new attack surface might encompass more digital voice assistants than people on the planet yet offer fewer remedies than conventional patching or firmware fixes since the inaudible attacks arise inherently from the microphone design and digital signal processing.

{{</citation>}}


## cs.CR (1)



### (28/32) An Efficient Authentication Protocol for Smart Grid Communication Based on On-Chip-Error-Correcting Physical Unclonable Function (Masoud Kaveh et al., 2023)

{{<citation>}}

Masoud Kaveh, Mohammad Reza Mosavi, Diego Martin, Saeed Aghapour. (2023)  
**An Efficient Authentication Protocol for Smart Grid Communication Based on On-Chip-Error-Correcting Physical Unclonable Function**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2307.12374v1)  

---


**ABSTRACT**  
Security has become a main concern for the smart grid to move from research and development to industry. The concept of security has usually referred to resistance to threats by an active or passive attacker. However, since smart meters (SMs) are often placed in unprotected areas, physical security has become one of the important security goals in the smart grid. Physical unclonable functions (PUFs) have been largely utilized for ensuring physical security in recent years, though their reliability has remained a major problem to be practically used in cryptographic applications. Although fuzzy extractors have been considered as a solution to solve the reliability problem of PUFs, they put a considerable computational cost to the resource-constrained SMs. To that end, we first propose an on-chip-error-correcting (OCEC) PUF that efficiently generates stable digits for the authentication process. Afterward, we introduce a lightweight authentication protocol between the SMs and neighborhood gateway (NG) based on the proposed PUF. The provable security analysis shows that not only the proposed protocol can stand secure in the Canetti-Krawczyk (CK) adversary model but also provides additional security features. Also, the performance evaluation demonstrates the significant improvement of the proposed scheme in comparison with the state-of-the-art.

{{</citation>}}


## cs.SD (2)



### (29/32) Self-Supervised Learning for Audio-Based Emotion Recognition (Peranut Nimitsurachat et al., 2023)

{{<citation>}}

Peranut Nimitsurachat, Peter Washington. (2023)  
**Self-Supervised Learning for Audio-Based Emotion Recognition**  

---
Primary Category: cs.SD  
Categories: cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Emotion Recognition, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2307.12343v1)  

---


**ABSTRACT**  
Emotion recognition models using audio input data can enable the development of interactive systems with applications in mental healthcare, marketing, gaming, and social media analysis. While the field of affective computing using audio data is rich, a major barrier to achieve consistently high-performance models is the paucity of available training labels. Self-supervised learning (SSL) is a family of methods which can learn despite a scarcity of supervised labels by predicting properties of the data itself. To understand the utility of self-supervised learning for audio-based emotion recognition, we have applied self-supervised learning pre-training to the classification of emotions from the CMU- MOSEI's acoustic modality. Unlike prior papers that have experimented with raw acoustic data, our technique has been applied to encoded acoustic data. Our model is first pretrained to uncover the randomly-masked timestamps of the acoustic data. The pre-trained model is then fine-tuned using a small sample of annotated data. The performance of the final model is then evaluated via several evaluation metrics against a baseline deep learning model with an identical backbone architecture. We find that self-supervised learning consistently improves the performance of the model across all metrics. This work shows the utility of self-supervised learning for affective computing, demonstrating that self-supervised learning is most useful when the number of training examples is small, and that the effect is most pronounced for emotions which are easier to classify such as happy, sad and anger. This work further demonstrates that self-supervised learning works when applied to embedded feature representations rather than the traditional approach of pre-training on the raw input space.

{{</citation>}}


### (30/32) Exploring the Integration of Speech Separation and Recognition with Self-Supervised Learning Representation (Yoshiki Masuyama et al., 2023)

{{<citation>}}

Yoshiki Masuyama, Xuankai Chang, Wangyou Zhang, Samuele Cornell, Zhong-Qiu Wang, Nobutaka Ono, Yanmin Qian, Shinji Watanabe. (2023)  
**Exploring the Integration of Speech Separation and Recognition with Self-Supervised Learning Representation**  

---
Primary Category: cs.SD  
Categories: cs-CL, cs-SD, cs.SD, eess-AS  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2307.12231v1)  

---


**ABSTRACT**  
Neural speech separation has made remarkable progress and its integration with automatic speech recognition (ASR) is an important direction towards realizing multi-speaker ASR. This work provides an insightful investigation of speech separation in reverberant and noisy-reverberant scenarios as an ASR front-end. In detail, we explore multi-channel separation methods, mask-based beamforming and complex spectral mapping, as well as the best features to use in the ASR back-end model. We employ the recent self-supervised learning representation (SSLR) as a feature and improve the recognition performance from the case with filterbank features. To further improve multi-speaker recognition performance, we present a carefully designed training strategy for integrating speech separation and recognition with SSLR. The proposed integration using TF-GridNet-based complex spectral mapping and WavLM-based SSLR achieves a 2.5% word error rate in reverberant WHAMR! test set, significantly outperforming an existing mask-based MVDR beamforming and filterbank integration (28.9%).

{{</citation>}}


## eess.IV (1)



### (31/32) ASCON: Anatomy-aware Supervised Contrastive Learning Framework for Low-dose CT Denoising (Zhihao Chen et al., 2023)

{{<citation>}}

Zhihao Chen, Qi Gao, Yi Zhang, Hongming Shan. (2023)  
**ASCON: Anatomy-aware Supervised Contrastive Learning Framework for Low-dose CT Denoising**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2307.12225v1)  

---


**ABSTRACT**  
While various deep learning methods have been proposed for low-dose computed tomography (CT) denoising, most of them leverage the normal-dose CT images as the ground-truth to supervise the denoising process. These methods typically ignore the inherent correlation within a single CT image, especially the anatomical semantics of human tissues, and lack the interpretability on the denoising process. In this paper, we propose a novel Anatomy-aware Supervised CONtrastive learning framework, termed ASCON, which can explore the anatomical semantics for low-dose CT denoising while providing anatomical interpretability. The proposed ASCON consists of two novel designs: an efficient self-attention-based U-Net (ESAU-Net) and a multi-scale anatomical contrastive network (MAC-Net). First, to better capture global-local interactions and adapt to the high-resolution input, an efficient ESAU-Net is introduced by using a channel-wise self-attention mechanism. Second, MAC-Net incorporates a patch-wise non-contrastive module to capture inherent anatomical information and a pixel-wise contrastive module to maintain intrinsic anatomical consistency. Extensive experimental results on two public low-dose CT denoising datasets demonstrate superior performance of ASCON over state-of-the-art models. Remarkably, our ASCON provides anatomical interpretability for low-dose CT denoising for the first time. Source code is available at https://github.com/hao1635/ASCON.

{{</citation>}}


## cs.CY (1)



### (32/32) A Comprehensive Review and Systematic Analysis of Artificial Intelligence Regulation Policies (Weiyue Wu et al., 2023)

{{<citation>}}

Weiyue Wu, Shaoshan Liu. (2023)  
**A Comprehensive Review and Systematic Analysis of Artificial Intelligence Regulation Policies**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.12218v1)  

---


**ABSTRACT**  
Due to the cultural and governance differences of countries around the world, there currently exists a wide spectrum of AI regulation policy proposals that have created a chaos in the global AI regulatory space. Properly regulating AI technologies is extremely challenging, as it requires a delicate balance between legal restrictions and technological developments. In this article, we first present a comprehensive review of AI regulation proposals from different geographical locations and cultural backgrounds. Then, drawing from historical lessons, we develop a framework to facilitate a thorough analysis of AI regulation proposals. Finally, we perform a systematic analysis of these AI regulation proposals to understand how each proposal may fail. This study, containing historical lessons and analysis methods, aims to help governing bodies untangling the AI regulatory chaos through a divide-and-conquer manner.

{{</citation>}}
