---
draft: false
title: "arXiv @ 2023.09.26"
date: 2023-09-26
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.09.26"
    identifier: arxiv_20230926
    parent: 202309_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.LG (7)](#cslg-7)
- [cs.CV (22)](#cscv-22)
- [cs.CL (11)](#cscl-11)
- [cs.RO (2)](#csro-2)
- [math.OC (1)](#mathoc-1)
- [cs.HC (2)](#cshc-2)
- [eess.AS (3)](#eessas-3)
- [cs.CR (2)](#cscr-2)
- [math.NA (1)](#mathna-1)
- [eess.IV (1)](#eessiv-1)
- [cs.IT (1)](#csit-1)

## cs.LG (7)



### (1/53) GHN-QAT: Training Graph Hypernetworks to Predict Quantization-Robust Parameters of Unseen Limited Precision Neural Networks (Stone Yun et al., 2023)

{{<citation>}}

Stone Yun, Alexander Wong. (2023)  
**GHN-QAT: Training Graph Hypernetworks to Predict Quantization-Robust Parameters of Unseen Limited Precision Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: QA, Quantization  
[Paper Link](http://arxiv.org/abs/2309.13773v1)  

---


**ABSTRACT**  
Graph Hypernetworks (GHN) can predict the parameters of varying unseen CNN architectures with surprisingly good accuracy at a fraction of the cost of iterative optimization. Following these successes, preliminary research has explored the use of GHNs to predict quantization-robust parameters for 8-bit and 4-bit quantized CNNs. However, this early work leveraged full-precision float32 training and only quantized for testing. We explore the impact of quantization-aware training and/or other quantization-based training strategies on quantized robustness and performance of GHN predicted parameters for low-precision CNNs. We show that quantization-aware training can significantly improve quantized accuracy for GHN predicted parameters of 4-bit quantized CNNs and even lead to greater-than-random accuracy for 2-bit quantized CNNs. These promising results open the door for future explorations such as investigating the use of GHN predicted parameters as initialization for further quantized training of individual CNNs, further exploration of "extreme bitwidth" quantization, and mixed precision quantization schemes.

{{</citation>}}


### (2/53) Devil in the Number: Towards Robust Multi-modality Data Filter (Yichen Xu et al., 2023)

{{<citation>}}

Yichen Xu, Zihan Xu, Wenhao Chai, Zhonghan Zhao, Enxin Song, Gaoang Wang. (2023)  
**Devil in the Number: Towards Robust Multi-modality Data Filter**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: AI, ImageNet  
[Paper Link](http://arxiv.org/abs/2309.13770v1)  

---


**ABSTRACT**  
In order to appropriately filter multi-modality data sets on a web-scale, it becomes crucial to employ suitable filtering methods to boost performance and reduce training costs. For instance, LAION papers employs the CLIP score filter to select data with CLIP scores surpassing a certain threshold. On the other hand, T-MARS achieves high-quality data filtering by detecting and masking text within images and then filtering by CLIP score. Through analyzing the dataset, we observe a significant proportion of redundant information, such as numbers, present in the textual content. Our experiments on a subset of the data unveil the profound impact of these redundant elements on the CLIP scores. A logical approach would involve reevaluating the CLIP scores after eliminating these influences. Experimentally, our text-based CLIP filter outperforms the top-ranked method on the ``small scale" of DataComp (a data filtering benchmark) on ImageNet distribution shifts, achieving a 3.6% performance improvement. The results also demonstrate that our proposed text-masked filter outperforms the original CLIP score filter when selecting the top 40% of the data. The impact of numbers on CLIP and their handling provide valuable insights for improving the effectiveness of CLIP training, including language rewrite techniques.

{{</citation>}}


### (3/53) Accelerating Large Batch Training via Gradient Signal to Noise Ratio (GSNR) (Guo-qing Jiang et al., 2023)

{{<citation>}}

Guo-qing Jiang, Jinlong Liu, Zixiang Ding, Lin Guo, Wei Lin. (2023)  
**Accelerating Large Batch Training via Gradient Signal to Noise Ratio (GSNR)**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: BERT, ImageNet, NLP  
[Paper Link](http://arxiv.org/abs/2309.13681v1)  

---


**ABSTRACT**  
As models for nature language processing (NLP), computer vision (CV) and recommendation systems (RS) require surging computation, a large number of GPUs/TPUs are paralleled as a large batch (LB) to improve training throughput. However, training such LB tasks often meets large generalization gap and downgrades final precision, which limits enlarging the batch size. In this work, we develop the variance reduced gradient descent technique (VRGD) based on the gradient signal to noise ratio (GSNR) and apply it onto popular optimizers such as SGD/Adam/LARS/LAMB. We carry out a theoretical analysis of convergence rate to explain its fast training dynamics, and a generalization analysis to demonstrate its smaller generalization gap on LB training. Comprehensive experiments demonstrate that VRGD can accelerate training ($1\sim 2 \times$), narrow generalization gap and improve final accuracy. We push the batch size limit of BERT pretraining up to 128k/64k and DLRM to 512k without noticeable accuracy loss. We improve ImageNet Top-1 accuracy at 96k by $0.52pp$ than LARS. The generalization gap of BERT and ImageNet training is significantly reduce by over $65\%$.

{{</citation>}}


### (4/53) From Cluster Assumption to Graph Convolution: Graph-based Semi-Supervised Learning Revisited (Zheng Wang et al., 2023)

{{<citation>}}

Zheng Wang, Hongming Ding, Li Pan, Jianhua Li, Zhiguo Gong, Philip S. Yu. (2023)  
**From Cluster Assumption to Graph Convolution: Graph-based Semi-Supervised Learning Revisited**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2309.13599v1)  

---


**ABSTRACT**  
Graph-based semi-supervised learning (GSSL) has long been a hot research topic. Traditional methods are generally shallow learners, based on the cluster assumption. Recently, graph convolutional networks (GCNs) have become the predominant techniques for their promising performance. In this paper, we theoretically discuss the relationship between these two types of methods in a unified optimization framework. One of the most intriguing findings is that, unlike traditional ones, typical GCNs may not jointly consider the graph structure and label information at each layer. Motivated by this, we further propose three simple but powerful graph convolution methods. The first is a supervised method OGC which guides the graph convolution process with labels. The others are two unsupervised methods: GGC and its multi-scale version GGCM, both aiming to preserve the graph structure information during the convolution process. Finally, we conduct extensive experiments to show the effectiveness of our methods.

{{</citation>}}


### (5/53) Probabilistic Weight Fixing: Large-scale training of neural network weight uncertainties for quantization (Christopher Subia-Waud et al., 2023)

{{<citation>}}

Christopher Subia-Waud, Srinandan Dasmahapatra. (2023)  
**Probabilistic Weight Fixing: Large-scale training of neural network weight uncertainties for quantization**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2309.13575v1)  

---


**ABSTRACT**  
Weight-sharing quantization has emerged as a technique to reduce energy expenditure during inference in large neural networks by constraining their weights to a limited set of values. However, existing methods for weight-sharing quantization often make assumptions about the treatment of weights based on value alone that neglect the unique role weight position plays. This paper proposes a probabilistic framework based on Bayesian neural networks (BNNs) and a variational relaxation to identify which weights can be moved to which cluster centre and to what degree based on their individual position-specific learned uncertainty distributions. We introduce a new initialisation setting and a regularisation term which allow for the training of BNNs under complex dataset-model combinations. By leveraging the flexibility of weight values captured through a probability distribution, we enhance noise resilience and downstream compressibility. Our iterative clustering procedure demonstrates superior compressibility and higher accuracy compared to state-of-the-art methods on both ResNet models and the more complex transformer-based architectures. In particular, our method outperforms the state-of-the-art quantization method top-1 accuracy by 1.6% on ImageNet using DeiT-Tiny, with its 5 million+ weights now represented by only 296 unique values.

{{</citation>}}


### (6/53) Iterative Reachability Estimation for Safe Reinforcement Learning (Milan Ganai et al., 2023)

{{<citation>}}

Milan Ganai, Zheng Gong, Chenning Yu, Sylvia Herbert, Sicun Gao. (2023)  
**Iterative Reachability Estimation for Safe Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-RO, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.13528v1)  

---


**ABSTRACT**  
Ensuring safety is important for the practical deployment of reinforcement learning (RL). Various challenges must be addressed, such as handling stochasticity in the environments, providing rigorous guarantees of persistent state-wise safety satisfaction, and avoiding overly conservative behaviors that sacrifice performance. We propose a new framework, Reachability Estimation for Safe Policy Optimization (RESPO), for safety-constrained RL in general stochastic settings. In the feasible set where there exist violation-free policies, we optimize for rewards while maintaining persistent safety. Outside this feasible set, our optimization produces the safest behavior by guaranteeing entrance into the feasible set whenever possible with the least cumulative discounted violations. We introduce a class of algorithms using our novel reachability estimation function to optimize in our proposed framework and in similar frameworks such as those concurrently handling multiple hard and soft constraints. We theoretically establish that our algorithms almost surely converge to locally optimal policies of our safe optimization framework. We evaluate the proposed methods on a diverse suite of safe RL environments from Safety Gym, PyBullet, and MuJoCo, and show the benefits in improving both reward performance and safety compared with state-of-the-art baselines.

{{</citation>}}


### (7/53) Guided Cooperation in Hierarchical Reinforcement Learning via Model-based Rollout (Haoran Wang et al., 2023)

{{<citation>}}

Haoran Wang, Yaoru Sun, Fang Wang, Yeming Chen. (2023)  
**Guided Cooperation in Hierarchical Reinforcement Learning via Model-based Rollout**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.13508v1)  

---


**ABSTRACT**  
Goal-conditioned hierarchical reinforcement learning (HRL) presents a promising approach for enabling effective exploration in complex long-horizon reinforcement learning (RL) tasks via temporal abstraction. Yet, most goal-conditioned HRL algorithms focused on the subgoal discovery, regardless of inter-level coupling. In essence, for hierarchical systems, the increased inter-level communication and coordination can induce more stable and robust policy improvement. Here, we present a goal-conditioned HRL framework with Guided Cooperation via Model-based Rollout (GCMR), which estimates forward dynamics to promote inter-level cooperation. The GCMR alleviates the state-transition error within off-policy correction through a model-based rollout, further improving the sample efficiency. Meanwhile, to avoid being disrupted by these corrected but possibly unseen or faraway goals, lower-level Q-function gradients are constrained using a gradient penalty with a model-inferred upper bound, leading to a more stable behavioral policy. Besides, we propose a one-step rollout-based planning to further facilitate inter-level cooperation, where the higher-level Q-function is used to guide the lower-level policy by estimating the value of future states so that global task information is transmitted downwards to avoid local pitfalls. Experimental results demonstrate that incorporating the proposed GCMR framework with ACLG, a disentangled variant of HIGL, yields more stable and robust policy improvement than baselines and substantially outperforms previous state-of-the-art (SOTA) HRL algorithms in both hard-exploration problems and robotic control.

{{</citation>}}


## cs.CV (22)



### (8/53) Combining Two Adversarial Attacks Against Person Re-Identification Systems (Eduardo de O. Andrade et al., 2023)

{{<citation>}}

Eduardo de O. Andrade, Igor Garcia Ballhausen Sampaio, Joris Guérin, José Viterbo. (2023)  
**Combining Two Adversarial Attacks Against Person Re-Identification Systems**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2309.13763v1)  

---


**ABSTRACT**  
The field of Person Re-Identification (Re-ID) has received much attention recently, driven by the progress of deep neural networks, especially for image classification. The problem of Re-ID consists in identifying individuals through images captured by surveillance cameras in different scenarios. Governments and companies are investing a lot of time and money in Re-ID systems for use in public safety and identifying missing persons. However, several challenges remain for successfully implementing Re-ID, such as occlusions and light reflections in people's images. In this work, we focus on adversarial attacks on Re-ID systems, which can be a critical threat to the performance of these systems. In particular, we explore the combination of adversarial attacks against Re-ID models, trying to strengthen the decrease in the classification results. We conduct our experiments on three datasets: DukeMTMC-ReID, Market-1501, and CUHK03. We combine the use of two types of adversarial attacks, P-FGSM and Deep Mis-Ranking, applied to two popular Re-ID models: IDE (ResNet-50) and AlignedReID. The best result demonstrates a decrease of 3.36% in the Rank-10 metric for AlignedReID applied to CUHK03. We also try to use Dropout during the inference as a defense method.

{{</citation>}}


### (9/53) A Systematic Literature Review of Computer Vision Applications in Robotized Wire Harness Assembly (Hao Wang et al., 2023)

{{<citation>}}

Hao Wang, Omkar Salunkhe, Walter Quadrini, Björn Johansson, Dan Lämkull, Fredrik Ore, Mélanie Despeisse, Luca Fumagalli, Johan Stahre. (2023)  
**A Systematic Literature Review of Computer Vision Applications in Robotized Wire Harness Assembly**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-RO, cs.CV  
Keywords: Computer Vision  
[Paper Link](http://arxiv.org/abs/2309.13744v1)  

---


**ABSTRACT**  
This article presents a systematic literature review on computer vision applications that have been proposed for robotized wire harness assembly, derives challenges from existing studies, and identifies opportunities for future research to promote a more practical robotized assembly of wire harnesses.

{{</citation>}}


### (10/53) MOSAIC: Multi-Object Segmented Arbitrary Stylization Using CLIP (Prajwal Ganugula et al., 2023)

{{<citation>}}

Prajwal Ganugula, Y S S S Santosh Kumar, N K Sagar Reddy, Prabhath Chellingi, Avinash Thakur, Neeraj Kasera, C Shyam Anand. (2023)  
**MOSAIC: Multi-Object Segmented Arbitrary Stylization Using CLIP**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.13716v1)  

---


**ABSTRACT**  
Style transfer driven by text prompts paved a new path for creatively stylizing the images without collecting an actual style image. Despite having promising results, with text-driven stylization, the user has no control over the stylization. If a user wants to create an artistic image, the user requires fine control over the stylization of various entities individually in the content image, which is not addressed by the current state-of-the-art approaches. On the other hand, diffusion style transfer methods also suffer from the same issue because the regional stylization control over the stylized output is ineffective. To address this problem, We propose a new method Multi-Object Segmented Arbitrary Stylization Using CLIP (MOSAIC), that can apply styles to different objects in the image based on the context extracted from the input prompt. Text-based segmentation and stylization modules which are based on vision transformer architecture, were used to segment and stylize the objects. Our method can extend to any arbitrary objects, styles and produce high-quality images compared to the current state of art methods. To our knowledge, this is the first attempt to perform text-guided arbitrary object-wise stylization. We demonstrate the effectiveness of our approach through qualitative and quantitative analysis, showing that it can generate visually appealing stylized images with enhanced control over stylization and the ability to generalize to unseen object classes.

{{</citation>}}


### (11/53) Sound-Print: Generalised Face Presentation Attack Detection using Deep Representation of Sound Echoes (Raghavendra Ramachandra et al., 2023)

{{<citation>}}

Raghavendra Ramachandra, Jag Mohan Singh, Sushma Venkatesh. (2023)  
**Sound-Print: Generalised Face Presentation Attack Detection using Deep Representation of Sound Echoes**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.13704v1)  

---


**ABSTRACT**  
Facial biometrics are widely deployed in smartphone-based applications because of their usability and increased verification accuracy in unconstrained scenarios. The evolving applications of smartphone-based facial recognition have also increased Presentation Attacks (PAs), where an attacker can present a Presentation Attack Instrument (PAI) to maliciously gain access to the application. Because the materials used to generate PAI are not deterministic, the detection of unknown presentation attacks is challenging. In this paper, we present an acoustic echo-based face Presentation Attack Detection (PAD) on a smartphone in which the PAs are detected based on the reflection profiles of the transmitted signal. We propose a novel transmission signal based on the wide pulse that allows us to model the background noise before transmitting the signal and increase the Signal-to-Noise Ratio (SNR). The received signal reflections were processed to remove background noise and accurately represent reflection characteristics. The reflection profiles of the bona fide and PAs are different owing to the different reflection characteristics of the human skin and artefact materials. Extensive experiments are presented using the newly collected Acoustic Sound Echo Dataset (ASED) with 4807 samples captured from bona fide and four different types of PAIs, including print (two types), display, and silicone face-mask attacks. The obtained results indicate the robustness of the proposed method for detecting unknown face presentation attacks.

{{</citation>}}


### (12/53) Causal-DFQ: Causality Guided Data-free Network Quantization (Yuzhang Shang et al., 2023)

{{<citation>}}

Yuzhang Shang, Bingxin Xu, Gaowen Liu, Ramana Kompella, Yan Yan. (2023)  
**Causal-DFQ: Causality Guided Data-free Network Quantization**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2309.13682v1)  

---


**ABSTRACT**  
Model quantization, which aims to compress deep neural networks and accelerate inference speed, has greatly facilitated the development of cumbersome models on mobile and edge devices. There is a common assumption in quantization methods from prior works that training data is available. In practice, however, this assumption cannot always be fulfilled due to reasons of privacy and security, rendering these methods inapplicable in real-life situations. Thus, data-free network quantization has recently received significant attention in neural network compression. Causal reasoning provides an intuitive way to model causal relationships to eliminate data-driven correlations, making causality an essential component of analyzing data-free problems. However, causal formulations of data-free quantization are inadequate in the literature. To bridge this gap, we construct a causal graph to model the data generation and discrepancy reduction between the pre-trained and quantized models. Inspired by the causal understanding, we propose the Causality-guided Data-free Network Quantization method, Causal-DFQ, to eliminate the reliance on data via approaching an equilibrium of causality-driven intervened distributions. Specifically, we design a content-style-decoupled generator, synthesizing images conditioned on the relevant and irrelevant factors; then we propose a discrepancy reduction loss to align the intervened distributions of the pre-trained and quantized models. It is worth noting that our work is the first attempt towards introducing causality to data-free quantization problem. Extensive experiments demonstrate the efficacy of Causal-DFQ. The code is available at https://github.com/42Shawn/Causal-DFQ.

{{</citation>}}


### (13/53) Deep Reinforcement Learning for Image-to-Image Translation (Xin Wang et al., 2023)

{{<citation>}}

Xin Wang, Ziwei Luo, Jing Hu, Chengming Feng, Shu Hu, Bin Zhu, Xi Wu, Siwei Lyu. (2023)  
**Deep Reinforcement Learning for Image-to-Image Translation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.13672v1)  

---


**ABSTRACT**  
Most existing Image-to-Image Translation (I2IT) methods generate images in a single run of a deep learning (DL) model. However, designing such a single-step model is always challenging, requiring a huge number of parameters and easily falling into bad global minimums and overfitting. In this work, we reformulate I2IT as a step-wise decision-making problem via deep reinforcement learning (DRL) and propose a novel framework that performs RL-based I2IT (RL-I2IT). The key feature in the RL-I2IT framework is to decompose a monolithic learning process into small steps with a lightweight model to progressively transform a source image successively to a target image. Considering that it is challenging to handle high dimensional continuous state and action spaces in the conventional RL framework, we introduce meta policy with a new concept Plan to the standard Actor-Critic model, which is of a lower dimension than the original image and can facilitate the actor to generate a tractable high dimensional action. In the RL-I2IT framework, we also employ a task-specific auxiliary learning strategy to stabilize the training process and improve the performance of the corresponding task. Experiments on several I2IT tasks demonstrate the effectiveness and robustness of the proposed method when facing high-dimensional continuous action space problems.

{{</citation>}}


### (14/53) GraphAdapter: Tuning Vision-Language Models With Dual Knowledge Graph (Xin Li et al., 2023)

{{<citation>}}

Xin Li, Dongze Lian, Zhihe Lu, Jiawang Bai, Zhibo Chen, Xinchao Wang. (2023)  
**GraphAdapter: Tuning Vision-Language Models With Dual Knowledge Graph**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Knowledge Graph, Language Model  
[Paper Link](http://arxiv.org/abs/2309.13625v1)  

---


**ABSTRACT**  
Adapter-style efficient transfer learning (ETL) has shown excellent performance in the tuning of vision-language models (VLMs) under the low-data regime, where only a few additional parameters are introduced to excavate the task-specific knowledge based on the general and powerful representation of VLMs. However, most adapter-style works face two limitations: (i) modeling task-specific knowledge with a single modality only; and (ii) overlooking the exploitation of the inter-class relationships in downstream tasks, thereby leading to sub-optimal solutions. To mitigate that, we propose an effective adapter-style tuning strategy, dubbed GraphAdapter, which performs the textual adapter by explicitly modeling the dual-modality structure knowledge (i.e., the correlation of different semantics/classes in textual and visual modalities) with a dual knowledge graph. In particular, the dual knowledge graph is established with two sub-graphs, i.e., a textual knowledge sub-graph, and a visual knowledge sub-graph, where the nodes and edges represent the semantics/classes and their correlations in two modalities, respectively. This enables the textual feature of each prompt to leverage the task-specific structure knowledge from both textual and visual modalities, yielding a more effective classifier for downstream tasks. Extensive experimental results on 11 benchmark datasets reveal that our GraphAdapter significantly outperforms previous adapter-based methods. The code will be released at https://github.com/lixinustc/GraphAdapter

{{</citation>}}


### (15/53) PRIS: Practical robust invertible network for image steganography (Hang Yang et al., 2023)

{{<citation>}}

Hang Yang, Yitian Xu, Xuhua Liu, Xiaodong Ma. (2023)  
**PRIS: Practical robust invertible network for image steganography**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CR, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.13620v1)  

---


**ABSTRACT**  
Image steganography is a technique of hiding secret information inside another image, so that the secret is not visible to human eyes and can be recovered when needed. Most of the existing image steganography methods have low hiding robustness when the container images affected by distortion. Such as Gaussian noise and lossy compression. This paper proposed PRIS to improve the robustness of image steganography, it based on invertible neural networks, and put two enhance modules before and after the extraction process with a 3-step training strategy. Moreover, rounding error is considered which is always ignored by existing methods, but actually it is unavoidable in practical. A gradient approximation function (GAF) is also proposed to overcome the undifferentiable issue of rounding distortion. Experimental results show that our PRIS outperforms the state-of-the-art robust image steganography method in both robustness and practicability. Codes are available at https://github.com/yanghangAI/PRIS, demonstration of our model in practical at http://yanghang.site/hide/.

{{</citation>}}


### (16/53) Changes-Aware Transformer: Learning Generalized Changes Representation (Dan Wang et al., 2023)

{{<citation>}}

Dan Wang, Licheng Jiao, Jie Chen, Shuyuan Yang, Fang Liu. (2023)  
**Changes-Aware Transformer: Learning Generalized Changes Representation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.13619v1)  

---


**ABSTRACT**  
Difference features obtained by comparing the images of two periods play an indispensable role in the change detection (CD) task. However, a pair of bi-temporal images can exhibit diverse changes, which may cause various difference features. Identifying changed pixels with differ difference features to be the same category is thus a challenge for CD. Most nowadays' methods acquire distinctive difference features in implicit ways like enhancing image representation or supervision information. Nevertheless, informative image features only guarantee object semantics are modeled and can not guarantee that changed pixels have similar semantics in the difference feature space and are distinct from those unchanged ones. In this work, the generalized representation of various changes is learned straightforwardly in the difference feature space, and a novel Changes-Aware Transformer (CAT) for refining difference features is proposed. This generalized representation can perceive which pixels are changed and which are unchanged and further guide the update of pixels' difference features. CAT effectively accomplishes this refinement process through the stacked cosine cross-attention layer and self-attention layer. After refinement, the changed pixels in the difference feature space are closer to each other, which facilitates change detection. In addition, CAT is compatible with various backbone networks and existing CD methods. Experiments on remote sensing CD data set and street scene CD data set show that our method achieves state-of-the-art performance and has excellent generalization.

{{</citation>}}


### (17/53) VisionKG: Unleashing the Power of Visual Datasets via Knowledge Graph (Jicheng Yuan et al., 2023)

{{<citation>}}

Jicheng Yuan, Anh Le-Tuan, Manh Nguyen-Duc, Trung-Kien Tran, Manfred Hauswirth, Danh Le-Phuoc. (2023)  
**VisionKG: Unleashing the Power of Visual Datasets via Knowledge Graph**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2309.13610v1)  

---


**ABSTRACT**  
The availability of vast amounts of visual data with heterogeneous features is a key factor for developing, testing, and benchmarking of new computer vision (CV) algorithms and architectures. Most visual datasets are created and curated for specific tasks or with limited image data distribution for very specific situations, and there is no unified approach to manage and access them across diverse sources, tasks, and taxonomies. This not only creates unnecessary overheads when building robust visual recognition systems, but also introduces biases into learning systems and limits the capabilities of data-centric AI. To address these problems, we propose the Vision Knowledge Graph (VisionKG), a novel resource that interlinks, organizes and manages visual datasets via knowledge graphs and Semantic Web technologies. It can serve as a unified framework facilitating simple access and querying of state-of-the-art visual datasets, regardless of their heterogeneous formats and taxonomies. One of the key differences between our approach and existing methods is that ours is knowledge-based rather than metadatabased. It enhances the enrichment of the semantics at both image and instance levels and offers various data retrieval and exploratory services via SPARQL. VisionKG currently contains 519 million RDF triples that describe approximately 40 million entities, and are accessible at https://vision.semkg.org and through APIs. With the integration of 30 datasets and four popular CV tasks, we demonstrate its usefulness across various scenarios when working with CV pipelines.

{{</citation>}}


### (18/53) Vulnerabilities in Video Quality Assessment Models: The Challenge of Adversarial Attacks (Ao-Xiang Zhang et al., 2023)

{{<citation>}}

Ao-Xiang Zhang, Yu Ran, Weixuan Tang, Yuan-Gen Wang. (2023)  
**Vulnerabilities in Video Quality Assessment Models: The Challenge of Adversarial Attacks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: Adversarial Attack, QA, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.13609v1)  

---


**ABSTRACT**  
No-Reference Video Quality Assessment (NR-VQA) plays an essential role in improving the viewing experience of end-users. Driven by deep learning, recent NR-VQA models based on Convolutional Neural Networks (CNNs) and Transformers have achieved outstanding performance. To build a reliable and practical assessment system, it is of great necessity to evaluate their robustness. However, such issue has received little attention in the academic community. In this paper, we make the first attempt to evaluate the robustness of NR-VQA models against adversarial attacks under black-box setting, and propose a patch-based random search method for black-box attack. Specifically, considering both the attack effect on quality score and the visual quality of adversarial video, the attack problem is formulated as misleading the estimated quality score under the constraint of just-noticeable difference (JND). Built upon such formulation, a novel loss function called Score-Reversed Boundary Loss is designed to push the adversarial video's estimated quality score far away from its ground-truth score towards a specific boundary, and the JND constraint is modeled as a strict $L_2$ and $L_\infty$ norm restriction. By this means, both white-box and black-box attacks can be launched in an effective and imperceptible manner. The source code is available at https://github.com/GZHU-DVL/AttackVQA.

{{</citation>}}


### (19/53) MM-NeRF: Multimodal-Guided 3D Multi-Style Transfer of Neural Radiance Field (Zijiang Yang et al., 2023)

{{<citation>}}

Zijiang Yang, Zhongwei Qiu, Chang Xu, Dongmei Fu. (2023)  
**MM-NeRF: Multimodal-Guided 3D Multi-Style Transfer of Neural Radiance Field**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Style Transfer  
[Paper Link](http://arxiv.org/abs/2309.13607v1)  

---


**ABSTRACT**  
3D style transfer aims to render stylized novel views of 3D scenes with the specified style, which requires high-quality rendering and keeping multi-view consistency. Benefiting from the ability of 3D representation from Neural Radiance Field (NeRF), existing methods learn the stylized NeRF by giving a reference style from an image. However, they suffer the challenges of high-quality stylization with texture details for multi-style transfer and stylization with multimodal guidance. In this paper, we reveal that the same objects in 3D scenes show various states (color tone, details, etc.) from different views after stylization since previous methods optimized by single-view image-based style loss functions, leading NeRF to tend to smooth texture details, further resulting in low-quality rendering. To tackle these problems, we propose a novel Multimodal-guided 3D Multi-style transfer of NeRF, termed MM-NeRF, which achieves high-quality 3D multi-style rendering with texture details and can be driven by multimodal-style guidance. First, MM-NeRF adopts a unified framework to project multimodal guidance into CLIP space and extracts multimodal style features to guide the multi-style stylization. To relieve the problem of lacking details, we propose a novel Multi-Head Learning Scheme (MLS), in which each style head predicts the parameters of the color head of NeRF. MLS decomposes the learning difficulty caused by the inconsistency of multi-style transfer and improves the quality of stylization. In addition, the MLS can generalize pre-trained MM-NeRF to any new styles by adding heads with small training costs (a few minutes). Extensive experiments on three real-world 3D scene datasets show that MM-NeRF achieves high-quality 3D multi-style stylization with multimodal guidance, keeps multi-view consistency, and keeps semantic consistency of multimodal style guidance. Codes will be released later.

{{</citation>}}


### (20/53) Distribution-Aware Continual Test Time Adaptation for Semantic Segmentation (Jiayi Ni et al., 2023)

{{<citation>}}

Jiayi Ni, Senqiao Yang, Jiaming Liu, Xiaoqi Li, Wenyu Jiao, Ran Xu, Zehui Chen, Yi Liu, Shanghang Zhang. (2023)  
**Distribution-Aware Continual Test Time Adaptation for Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2309.13604v1)  

---


**ABSTRACT**  
Since autonomous driving systems usually face dynamic and ever-changing environments, continual test-time adaptation (CTTA) has been proposed as a strategy for transferring deployed models to continually changing target domains. However, the pursuit of long-term adaptation often introduces catastrophic forgetting and error accumulation problems, which impede the practical implementation of CTTA in the real world. Recently, existing CTTA methods mainly focus on utilizing a majority of parameters to fit target domain knowledge through self-training. Unfortunately, these approaches often amplify the challenge of error accumulation due to noisy pseudo-labels, and pose practical limitations stemming from the heavy computational costs associated with entire model updates. In this paper, we propose a distribution-aware tuning (DAT) method to make the semantic segmentation CTTA efficient and practical in real-world applications. DAT adaptively selects and updates two small groups of trainable parameters based on data distribution during the continual adaptation process, including domain-specific parameters (DSP) and task-relevant parameters (TRP). Specifically, DSP exhibits sensitivity to outputs with substantial distribution shifts, effectively mitigating the problem of error accumulation. In contrast, TRP are allocated to positions that are responsive to outputs with minor distribution shifts, which are fine-tuned to avoid the catastrophic forgetting problem. In addition, since CTTA is a temporal task, we introduce the Parameter Accumulation Update (PAU) strategy to collect the updated DSP and TRP in target domain sequences. We conduct extensive experiments on two widely-used semantic segmentation CTTA benchmarks, achieving promising performance compared to previous state-of-the-art methods.

{{</citation>}}


### (21/53) FaceAtt: Enhancing Image Captioning with Facial Attributes for Portrait Images (Naimul Haque et al., 2023)

{{<citation>}}

Naimul Haque, Iffat Labiba, Sadia Akter. (2023)  
**FaceAtt: Enhancing Image Captioning with Facial Attributes for Portrait Images**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: BLEU, Image Captioning  
[Paper Link](http://arxiv.org/abs/2309.13601v1)  

---


**ABSTRACT**  
Automated image caption generation is a critical area of research that enhances accessibility and understanding of visual content for diverse audiences. In this study, we propose the FaceAtt model, a novel approach to attribute-focused image captioning that emphasizes the accurate depiction of facial attributes within images. FaceAtt automatically detects and describes a wide range of attributes, including emotions, expressions, pointed noses, fair skin tones, hair textures, attractiveness, and approximate age ranges. Leveraging deep learning techniques, we explore the impact of different image feature extraction methods on caption quality and evaluate our model's performance using metrics such as BLEU and METEOR. Our FaceAtt model leverages annotated attributes of portraits as supplementary prior knowledge for our portrait images before captioning. This innovative addition yields a subtle yet discernible enhancement in the resulting scores, exemplifying the potency of incorporating additional attribute vectors during training. Furthermore, our research contributes to the broader discourse on ethical considerations in automated captioning. This study sets the stage for future research in refining attribute-focused captioning techniques, with a focus on enhancing linguistic coherence, addressing biases, and accommodating diverse user needs.

{{</citation>}}


### (22/53) Multi-Dimensional Hyena for Spatial Inductive Bias (Itamar Zimerman et al., 2023)

{{<citation>}}

Itamar Zimerman, Lior Wolf. (2023)  
**Multi-Dimensional Hyena for Spatial Inductive Bias**  

---
Primary Category: cs.CV  
Categories: F-2-2; I-2-7, cs-CV, cs-LG, cs.CV  
Keywords: Bias, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.13600v1)  

---


**ABSTRACT**  
In recent years, Vision Transformers have attracted increasing interest from computer vision researchers. However, the advantage of these transformers over CNNs is only fully manifested when trained over a large dataset, mainly due to the reduced inductive bias towards spatial locality within the transformer's self-attention mechanism. In this work, we present a data-efficient vision transformer that does not rely on self-attention. Instead, it employs a novel generalization to multiple axes of the very recent Hyena layer. We propose several alternative approaches for obtaining this generalization and delve into their unique distinctions and considerations from both empirical and theoretical perspectives.   Our empirical findings indicate that the proposed Hyena N-D layer boosts the performance of various Vision Transformer architectures, such as ViT, Swin, and DeiT across multiple datasets. Furthermore, in the small dataset regime, our Hyena-based ViT is favorable to ViT variants from the recent literature that are specifically designed for solving the same challenge, i.e., working with small datasets or incorporating image-specific inductive bias into the self-attention mechanism. Finally, we show that a hybrid approach that is based on Hyena N-D for the first layers in ViT, followed by layers that incorporate conventional attention, consistently boosts the performance of various vision transformer architectures.

{{</citation>}}


### (23/53) A SAM-based Solution for Hierarchical Panoptic Segmentation of Crops and Weeds Competition (Khoa Dang Nguyen et al., 2023)

{{<citation>}}

Khoa Dang Nguyen, Thanh-Hai Phung, Hoang-Giang Cao. (2023)  
**A SAM-based Solution for Hierarchical Panoptic Segmentation of Crops and Weeds Competition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Computer Vision  
[Paper Link](http://arxiv.org/abs/2309.13578v1)  

---


**ABSTRACT**  
Panoptic segmentation in agriculture is an advanced computer vision technique that provides a comprehensive understanding of field composition. It facilitates various tasks such as crop and weed segmentation, plant panoptic segmentation, and leaf instance segmentation, all aimed at addressing challenges in agriculture. Exploring the application of panoptic segmentation in agriculture, the 8th Workshop on Computer Vision in Plant Phenotyping and Agriculture (CVPPA) hosted the challenge of hierarchical panoptic segmentation of crops and weeds using the PhenoBench dataset. To tackle the tasks presented in this competition, we propose an approach that combines the effectiveness of the Segment AnyThing Model (SAM) for instance segmentation with prompt input from object detection models. Specifically, we integrated two notable approaches in object detection, namely DINO and YOLO-v8. Our best-performing model achieved a PQ+ score of 81.33 based on the evaluation metrics of the competition.

{{</citation>}}


### (24/53) Towards Subcentimeter Accuracy Digital-Twin Tracking via An RGBD-based Transformer Model and A Comprehensive Mobile Dataset (Zixun Huang et al., 2023)

{{<citation>}}

Zixun Huang, Keling Yao, Seth Z. Zhao, Chuanyu Pan, Tianjian Xu, Weiyu Feng, Allen Y. Yang. (2023)  
**Towards Subcentimeter Accuracy Digital-Twin Tracking via An RGBD-based Transformer Model and A Comprehensive Mobile Dataset**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.13570v1)  

---


**ABSTRACT**  
The potential of digital twin technology, involving the creation of precise digital replicas of physical objects, to reshape AR experiences in 3D object tracking and localization scenarios is significant. However, enabling 3D object tracking with subcentimeter accuracy in dynamic mobile AR environments remains a formidable challenge. These scenarios often require a more robust pose estimator capable of handling the inherent sensor-level measurement noise. In this paper, recognizing the absence of comprehensive solutions in existing literature, we build upon our previous work, the Digital Twin Tracking Dataset (DTTD), to address these challenges in mobile AR settings. Specifically, we propose a transformer-based 6DoF pose estimator designed to withstand the challenges posed by noisy depth data. Simultaneously, we introduce a novel RGBD dataset captured using a cutting-edge mobile sensor, the iPhone 14 Pro, expanding the applicability of our approach to iPhone sensor data. Through extensive experimentation and in-depth analysis, we illustrate the effectiveness of our methods in the face of significant depth data errors, surpassing the performance of existing baselines. Code will be made publicly available.

{{</citation>}}


### (25/53) LOGICSEG: Parsing Visual Semantics with Neural Logic Learning and Reasoning (Liulei Li et al., 2023)

{{<citation>}}

Liulei Li, Wenguan Wang, Yang Yi. (2023)  
**LOGICSEG: Parsing Visual Semantics with Neural Logic Learning and Reasoning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2309.13556v1)  

---


**ABSTRACT**  
Current high-performance semantic segmentation models are purely data-driven sub-symbolic approaches and blind to the structured nature of the visual world. This is in stark contrast to human cognition which abstracts visual perceptions at multiple levels and conducts symbolic reasoning with such structured abstraction. To fill these fundamental gaps, we devise LOGICSEG, a holistic visual semantic parser that integrates neural inductive learning and logic reasoning with both rich data and symbolic knowledge. In particular, the semantic concepts of interest are structured as a hierarchy, from which a set of constraints are derived for describing the symbolic relations and formalized as first-order logic rules. After fuzzy logic-based continuous relaxation, logical formulae are grounded onto data and neural computational graphs, hence enabling logic-induced network training. During inference, logical constraints are packaged into an iterative process and injected into the network in a form of several matrix multiplications, so as to achieve hierarchy-coherent prediction with logic reasoning. These designs together make LOGICSEG a general and compact neural-logic machine that is readily integrated into existing segmentation models. Extensive experiments over four datasets with various segmentation models and backbones verify the effectiveness and generality of LOGICSEG. We believe this study opens a new avenue for visual semantic parsing.

{{</citation>}}


### (26/53) Decoding Radiologists Intense Focus for Accurate CXR Diagnoses: A Controllable and Interpretable AI System (Trong Thang Pham et al., 2023)

{{<citation>}}

Trong Thang Pham, Jacob Brecheisen, Anh Nguyen, Hien Nguyen, Ngan Le. (2023)  
**Decoding Radiologists Intense Focus for Accurate CXR Diagnoses: A Controllable and Interpretable AI System**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.13550v1)  

---


**ABSTRACT**  
In the field of chest X-ray (CXR) diagnosis, existing works often focus solely on determining where a radiologist looks, typically through tasks such as detection, segmentation, or classification. However, these approaches are often designed as black-box models, lacking interpretability. In this paper, we introduce a novel and unified controllable interpretable pipeline for decoding the intense focus of radiologists in CXR diagnosis. Our approach addresses three key questions: where a radiologist looks, how long they focus on specific areas, and what findings they diagnose. By capturing the intensity of the radiologist's gaze, we provide a unified solution that offers insights into the cognitive process underlying radiological interpretation. Unlike current methods that rely on black-box machine learning models, which can be prone to extracting erroneous information from the entire input image during the diagnosis process, we tackle this issue by effectively masking out irrelevant information. Our approach leverages a vision-language model, allowing for precise control over the interpretation process while ensuring the exclusion of irrelevant features. To train our model, we utilize an eye gaze dataset to extract anatomical gaze information and generate ground truth heatmaps. Through extensive experimentation, we demonstrate the efficacy of our method. We showcase that the attention heatmaps, designed to mimic radiologists' focus, encode sufficient and relevant information, enabling accurate classification tasks using only a portion of CXR.

{{</citation>}}


### (27/53) Semi-Supervised Domain Generalization for Object Detection via Language-Guided Feature Alignment (Sina Malakouti et al., 2023)

{{<citation>}}

Sina Malakouti, Adriana Kovashka. (2023)  
**Semi-Supervised Domain Generalization for Object Detection via Language-Guided Feature Alignment**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2309.13525v1)  

---


**ABSTRACT**  
Existing domain adaptation (DA) and generalization (DG) methods in object detection enforce feature alignment in the visual space but face challenges like object appearance variability and scene complexity, which make it difficult to distinguish between objects and achieve accurate detection. In this paper, we are the first to address the problem of semi-supervised domain generalization by exploring vision-language pre-training and enforcing feature alignment through the language space. We employ a novel Cross-Domain Descriptive Multi-Scale Learning (CDDMSL) aiming to maximize the agreement between descriptions of an image presented with different domain-specific characteristics in the embedding space. CDDMSL significantly outperforms existing methods, achieving 11.7% and 7.5% improvement in DG and DA settings, respectively. Comprehensive analysis and ablation studies confirm the effectiveness of our method, positioning CDDMSL as a promising approach for domain generalization in object detection tasks.

{{</citation>}}


### (28/53) Global-correlated 3D-decoupling Transformer for Clothed Avatar Reconstruction (Zechuan Zhang et al., 2023)

{{<citation>}}

Zechuan Zhang, Li Sun, Zongxin Yang, Ling Chen, Yi Yang. (2023)  
**Global-correlated 3D-decoupling Transformer for Clothed Avatar Reconstruction**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.13524v1)  

---


**ABSTRACT**  
Reconstructing 3D clothed human avatars from single images is a challenging task, especially when encountering complex poses and loose clothing. Current methods exhibit limitations in performance, largely attributable to their dependence on insufficient 2D image features and inconsistent query methods. Owing to this, we present the Global-correlated 3D-decoupling Transformer for clothed Avatar reconstruction (GTA), a novel transformer-based architecture that reconstructs clothed human avatars from monocular images. Our approach leverages transformer architectures by utilizing a Vision Transformer model as an encoder for capturing global-correlated image features. Subsequently, our innovative 3D-decoupling decoder employs cross-attention to decouple tri-plane features, using learnable embeddings as queries for cross-plane generation. To effectively enhance feature fusion with the tri-plane 3D feature and human body prior, we propose a hybrid prior fusion strategy combining spatial and prior-enhanced queries, leveraging the benefits of spatial localization and human body prior knowledge. Comprehensive experiments on CAPE and THuman2.0 datasets illustrate that our method outperforms state-of-the-art approaches in both geometry and texture reconstruction, exhibiting high robustness to challenging poses and loose clothing, and producing higher-resolution textures. Codes will be available at https://github.com/River-Zhang/GTA.

{{</citation>}}


### (29/53) Bridging Semantic Gaps for Language-Supervised Semantic Segmentation (Yun Xing et al., 2023)

{{<citation>}}

Yun Xing, Jian Kang, Aoran Xiao, Jiahao Nie, Shao Ling, Shijian Lu. (2023)  
**Bridging Semantic Gaps for Language-Supervised Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2309.13505v1)  

---


**ABSTRACT**  
Vision-Language Pre-training has demonstrated its remarkable zero-shot recognition ability and potential to learn generalizable visual representations from language supervision. Taking a step ahead, language-supervised semantic segmentation enables spatial localization of textual inputs by learning pixel grouping solely from image-text pairs. Nevertheless, the state-of-the-art suffers from clear semantic gaps between visual and textual modality: plenty of visual concepts appeared in images are missing in their paired captions. Such semantic misalignment circulates in pre-training, leading to inferior zero-shot performance in dense predictions due to insufficient visual concepts captured in textual representations. To close such semantic gap, we propose Concept Curation (CoCu), a pipeline that leverages CLIP to compensate for the missing semantics. For each image-text pair, we establish a concept archive that maintains potential visually-matched concepts with our proposed vision-driven expansion and text-to-vision-guided ranking. Relevant concepts can thus be identified via cluster-guided sampling and fed into pre-training, thereby bridging the gap between visual and textual semantics. Extensive experiments over a broad suite of 8 segmentation benchmarks show that CoCu achieves superb zero-shot transfer performance and greatly boosts language-supervised segmentation baseline by a large margin, suggesting the value of bridging semantic gap in pre-training data.

{{</citation>}}


## cs.CL (11)



### (30/53) Text Classification: A Perspective of Deep Learning Methods (Zhongwei Wan, 2023)

{{<citation>}}

Zhongwei Wan. (2023)  
**Text Classification: A Perspective of Deep Learning Methods**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Text Classification  
[Paper Link](http://arxiv.org/abs/2309.13761v1)  

---


**ABSTRACT**  
In recent years, with the rapid development of information on the Internet, the number of complex texts and documents has increased exponentially, which requires a deeper understanding of deep learning methods in order to accurately classify texts using deep learning techniques, and thus deep learning methods have become increasingly important in text classification. Text classification is a class of tasks that automatically classifies a set of documents into multiple predefined categories based on their content and subject matter. Thus, the main goal of text classification is to enable users to extract information from textual resources and process processes such as retrieval, classification, and machine learning techniques together in order to classify different categories. Many new techniques of deep learning have already achieved excellent results in natural language processing. The success of these learning algorithms relies on their ability to understand complex models and non-linear relationships in data. However, finding the right structure, architecture, and techniques for text classification is a challenge for researchers. This paper introduces deep learning-based text classification algorithms, including important steps required for text classification tasks such as feature extraction, feature reduction, and evaluation strategies and methods. At the end of the article, different deep learning text classification methods are compared and summarized.

{{</citation>}}


### (31/53) Does the 'most sinfully decadent cake ever' taste good? Answering Yes/No Questions from Figurative Contexts (Geetanjali Rakshit et al., 2023)

{{<citation>}}

Geetanjali Rakshit, Jeffrey Flanigan. (2023)  
**Does the 'most sinfully decadent cake ever' taste good? Answering Yes/No Questions from Figurative Contexts**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, ChatGPT, GPT, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2309.13748v1)  

---


**ABSTRACT**  
Figurative language is commonplace in natural language, and while making communication memorable and creative, can be difficult to understand. In this work, we investigate the robustness of Question Answering (QA) models on figurative text. Yes/no questions, in particular, are a useful probe of figurative language understanding capabilities of large language models. We propose FigurativeQA, a set of 1000 yes/no questions with figurative and non-figurative contexts, extracted from the domains of restaurant and product reviews. We show that state-of-the-art BERT-based QA models exhibit an average performance drop of up to 15\% points when answering questions from figurative contexts, as compared to non-figurative ones. While models like GPT-3 and ChatGPT are better at handling figurative texts, we show that further performance gains can be achieved by automatically simplifying the figurative contexts into their non-figurative (literal) counterparts. We find that the best overall model is ChatGPT with chain-of-thought prompting to generate non-figurative contexts. Our work provides a promising direction for building more robust QA models with figurative language understanding capabilities.

{{</citation>}}


### (32/53) Use of Large Language Models for Stance Classification (Iain J. Cruickshank et al., 2023)

{{<citation>}}

Iain J. Cruickshank, Lynnette Hui Xian Ng. (2023)  
**Use of Large Language Models for Stance Classification**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.13734v1)  

---


**ABSTRACT**  
Stance detection, the task of predicting an author's viewpoint towards a subject of interest, has long been a focal point of research. Current stance detection methods predominantly rely on manual annotation of sentences, followed by training a supervised machine learning model. This manual annotation process, however, imposes limitations on the model's ability to fully comprehend the stances in the sentence and hampers its potential to generalize across different contexts. In this study, we investigate the use of Large Language Models (LLMs) for the task of stance classification, with an absolute minimum use of human labels. We scrutinize four distinct types of prompting schemes combined with LLMs, comparing their accuracies with manual stance determination. Our study reveals that while LLMs can match or sometimes even exceed the benchmark results in each dataset, their overall accuracy is not definitively better than what can be produced by supervised models. This suggests potential areas for improvement in the stance classification for LLMs. The application of LLMs, however, opens up promising avenues for unsupervised stance detection, thereby curtailing the need for manual collection and annotation of stances. This not only streamlines the process but also paves the way for expanding stance detection capabilities across languages. Through this paper, we shed light on the stance classification abilities of LLMs, thereby contributing valuable insights that can guide future advancements in this domain.

{{</citation>}}


### (33/53) Arabic Sentiment Analysis with Noisy Deep Explainable Model (Md. Atabuzzaman et al., 2023)

{{<citation>}}

Md. Atabuzzaman, Md Shajalal, Maksuda Bilkis Baby, Alexander Boden. (2023)  
**Arabic Sentiment Analysis with Noisy Deep Explainable Model**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, LSTM, Sentiment Analysis  
[Paper Link](http://arxiv.org/abs/2309.13731v1)  

---


**ABSTRACT**  
Sentiment Analysis (SA) is an indispensable task for many real-world applications. Compared to limited resourced languages (i.e., Arabic, Bengali), most of the research on SA are conducted for high resourced languages (i.e., English, Chinese). Moreover, the reasons behind any prediction of the Arabic sentiment analysis methods exploiting advanced artificial intelligence (AI)-based approaches are like black-box - quite difficult to understand. This paper proposes an explainable sentiment classification framework for the Arabic language by introducing a noise layer on Bi-Directional Long Short-Term Memory (BiLSTM) and Convolutional Neural Networks (CNN)-BiLSTM models that overcome over-fitting problem. The proposed framework can explain specific predictions by training a local surrogate explainable model to understand why a particular sentiment (positive or negative) is being predicted. We carried out experiments on public benchmark Arabic SA datasets. The results concluded that adding noise layers improves the performance in sentiment analysis for the Arabic language by reducing overfitting and our method outperformed some known state-of-the-art methods. In addition, the introduced explainability with noise layer could make the model more transparent and accountable and hence help adopting AI-enabled system in practice.

{{</citation>}}


### (34/53) Skill Check: Some Considerations on the Evaluation of Gamemastering Models for Role-playing Games (Santiago Góngora et al., 2023)

{{<citation>}}

Santiago Góngora, Luis Chiruzzo, Gonzalo Méndez, Pablo Gervás. (2023)  
**Skill Check: Some Considerations on the Evaluation of Gamemastering Models for Role-playing Games**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2309.13702v1)  

---


**ABSTRACT**  
In role-playing games a Game Master (GM) is the player in charge of the game, who must design the challenges the players face and narrate the outcomes of their actions. In this work we discuss some challenges to model GMs from an Interactive Storytelling and Natural Language Processing perspective. Following those challenges we propose three test categories to evaluate such dialogue systems, and we use them to test ChatGPT, Bard and OpenAssistant as out-of-the-box GMs.

{{</citation>}}


### (35/53) ALLURE: A Systematic Protocol for Auditing and Improving LLM-based Evaluation of Text using Iterative In-Context-Learning (Hosein Hasanbeig et al., 2023)

{{<citation>}}

Hosein Hasanbeig, Hiteshi Sharma, Leo Betthauser, Felipe Vieira Frujeri, Ida Momennejad. (2023)  
**ALLURE: A Systematic Protocol for Auditing and Improving LLM-based Evaluation of Text using Iterative In-Context-Learning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-HC, cs.CL  
Keywords: AI, Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2309.13701v1)  

---


**ABSTRACT**  
From grading papers to summarizing medical documents, large language models (LLMs) are evermore used for evaluation of text generated by humans and AI alike. However, despite their extensive utility, LLMs exhibit distinct failure modes, necessitating a thorough audit and improvement of their text evaluation capabilities. Here we introduce ALLURE, a systematic approach to Auditing Large Language Models Understanding and Reasoning Errors. ALLURE involves comparing LLM-generated evaluations with annotated data, and iteratively incorporating instances of significant deviation into the evaluator, which leverages in-context learning (ICL) to enhance and improve robust evaluation of text by LLMs. Through this iterative process, we aim to refine the performance of the evaluator LLM, ultimately reducing the reliance on human annotators in the evaluation process. We anticipate ALLURE to serve diverse applications of LLMs in various domains related to evaluation of textual data and productivity in these fields.

{{</citation>}}


### (36/53) Embers of Autoregression: Understanding Large Language Models Through the Problem They are Trained to Solve (R. Thomas McCoy et al., 2023)

{{<citation>}}

R. Thomas McCoy, Shunyu Yao, Dan Friedman, Matthew Hardy, Thomas L. Griffiths. (2023)  
**Embers of Autoregression: Understanding Large Language Models Through the Problem They are Trained to Solve**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, GPT, GPT-3.5, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2309.13638v1)  

---


**ABSTRACT**  
The widespread adoption of large language models (LLMs) makes it important to recognize their strengths and limitations. We argue that in order to develop a holistic understanding of these systems we need to consider the problem that they were trained to solve: next-word prediction over Internet text. By recognizing the pressures that this task exerts we can make predictions about the strategies that LLMs will adopt, allowing us to reason about when they will succeed or fail. This approach - which we call the teleological approach - leads us to identify three factors that we hypothesize will influence LLM accuracy: the probability of the task to be performed, the probability of the target output, and the probability of the provided input. We predict that LLMs will achieve higher accuracy when these probabilities are high than when they are low - even in deterministic settings where probability should not matter. To test our predictions, we evaluate two LLMs (GPT-3.5 and GPT-4) on eleven tasks, and we find robust evidence that LLMs are influenced by probability in the ways that we have hypothesized. In many cases, the experiments reveal surprising failure modes. For instance, GPT-4's accuracy at decoding a simple cipher is 51% when the output is a high-probability word sequence but only 13% when it is low-probability. These results show that AI practitioners should be careful about using LLMs in low-probability situations. More broadly, we conclude that we should not evaluate LLMs as if they are humans but should instead treat them as a distinct type of system - one that has been shaped by its own particular set of pressures.

{{</citation>}}


### (37/53) MentalLLaMA: Interpretable Mental Health Analysis on Social Media with Large Language Models (Kailai Yang et al., 2023)

{{<citation>}}

Kailai Yang, Tianlin Zhang, Ziyan Kuang, Qianqian Xie, Sophia Ananiadou. (2023)  
**MentalLLaMA: Interpretable Mental Health Analysis on Social Media with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, LLaMA, Language Model, Social Media  
[Paper Link](http://arxiv.org/abs/2309.13567v1)  

---


**ABSTRACT**  
With the development of web technology, social media texts are becoming a rich source for automatic mental health analysis. As traditional discriminative methods bear the problem of low interpretability, the recent large language models have been explored for interpretable mental health analysis on social media, which aims to provide detailed explanations along with predictions. The results show that ChatGPT can generate approaching-human explanations for its correct classifications. However, LLMs still achieve unsatisfactory classification performance in a zero-shot/few-shot manner. Domain-specific finetuning is an effective solution, but faces 2 challenges: 1) lack of high-quality training data. 2) no open-source LLMs for interpretable mental health analysis were released to lower the finetuning cost. To alleviate these problems, we build the first multi-task and multi-source interpretable mental health instruction (IMHI) dataset on social media, with 105K data samples. The raw social media data are collected from 10 existing sources covering 8 mental health analysis tasks. We use expert-written few-shot prompts and collected labels to prompt ChatGPT and obtain explanations from its responses. To ensure the reliability of the explanations, we perform strict automatic and human evaluations on the correctness, consistency, and quality of generated data. Based on the IMHI dataset and LLaMA2 foundation models, we train MentalLLaMA, the first open-source LLM series for interpretable mental health analysis with instruction-following capability. We also evaluate the performance of MentalLLaMA on the IMHI evaluation benchmark with 10 test sets, where their correctness for making predictions and the quality of explanations are examined. The results show that MentalLLaMA approaches state-of-the-art discriminative methods in correctness and generates high-quality explanations.

{{</citation>}}


### (38/53) Keeping in Time: Adding Temporal Context to Sentiment Analysis Models (Dean Ninalga, 2023)

{{<citation>}}

Dean Ninalga. (2023)  
**Keeping in Time: Adding Temporal Context to Sentiment Analysis Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Sentiment Analysis  
[Paper Link](http://arxiv.org/abs/2309.13562v1)  

---


**ABSTRACT**  
This paper presents a state-of-the-art solution to the LongEval CLEF 2023 Lab Task 2: LongEval-Classification. The goal of this task is to improve and preserve the performance of sentiment analysis models across shorter and longer time periods. Our framework feeds date-prefixed textual inputs to a pre-trained language model, where the timestamp is included in the text. We show date-prefixed samples better conditions model outputs on the temporal context of the respective texts. Moreover, we further boost performance by performing self-labeling on unlabeled data to train a student model. We augment the self-labeling process using a novel augmentation strategy leveraging the date-prefixed formatting of our samples. We demonstrate concrete performance gains on the LongEval-Classification evaluation set over non-augmented self-labeling. Our framework achieves a 2nd place ranking with an overall score of 0.6923 and reports the best Relative Performance Drop (RPD) of -0.0656 over the short evaluation set.

{{</citation>}}


### (39/53) Cordyceps@LT-EDI: Patching Language-Specific Homophobia/Transphobia Classifiers with a Multilingual Understanding (Dean Ninalga, 2023)

{{<citation>}}

Dean Ninalga. (2023)  
**Cordyceps@LT-EDI: Patching Language-Specific Homophobia/Transphobia Classifiers with a Multilingual Understanding**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Multilingual  
[Paper Link](http://arxiv.org/abs/2309.13561v1)  

---


**ABSTRACT**  
Detecting transphobia, homophobia, and various other forms of hate speech is difficult. Signals can vary depending on factors such as language, culture, geographical region, and the particular online platform. Here, we present a joint multilingual (M-L) and language-specific (L-S) approach to homophobia and transphobic hate speech detection (HSD). M-L models are needed to catch words, phrases, and concepts that are less common or missing in a particular language and subsequently overlooked by L-S models. Nonetheless, L-S models are better situated to understand the cultural and linguistic context of the users who typically write in a particular language. Here we construct a simple and successful way to merge the M-L and L-S approaches through simple weight interpolation in such a way that is interpretable and data-driven. We demonstrate our system on task A of the 'Shared Task on Homophobia/Transphobia Detection in social media comments' dataset for homophobia and transphobic HSD. Our system achieves the best results in three of five languages and achieves a 0.997 macro average F1-score on Malayalam texts.

{{</citation>}}


### (40/53) Substituting Data Annotation with Balanced Updates and Collective Loss in Multi-label Text Classification (Muberra Ozmen et al., 2023)

{{<citation>}}

Muberra Ozmen, Joseph Cotnareanu, Mark Coates. (2023)  
**Substituting Data Annotation with Balanced Updates and Collective Loss in Multi-label Text Classification**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Text Classification  
[Paper Link](http://arxiv.org/abs/2309.13543v1)  

---


**ABSTRACT**  
Multi-label text classification (MLTC) is the task of assigning multiple labels to a given text, and has a wide range of application domains. Most existing approaches require an enormous amount of annotated data to learn a classifier and/or a set of well-defined constraints on the label space structure, such as hierarchical relations which may be complicated to provide as the number of labels increases. In this paper, we study the MLTC problem in annotation-free and scarce-annotation settings in which the magnitude of available supervision signals is linear to the number of labels. Our method follows three steps, (1) mapping input text into a set of preliminary label likelihoods by natural language inference using a pre-trained language model, (2) calculating a signed label dependency graph by label descriptions, and (3) updating the preliminary label likelihoods with message passing along the label dependency graph, driven with a collective loss function that injects the information of expected label frequency and average multi-label cardinality of predictions. The experiments show that the proposed framework achieves effective performance under low supervision settings with almost imperceptible computational and memory overheads added to the usage of pre-trained language model outperforming its initial performance by 70\% in terms of example-based F1 score.

{{</citation>}}


## cs.RO (2)



### (41/53) Computer Vision Technology for Robotized Wire Harness Assembly (Hao Wang et al., 2023)

{{<citation>}}

Hao Wang, Omkar Salunkhe, Walter Quadrini, Dan Lämkull, Fredrik Ore, Björn Johansson, Johan Stahre. (2023)  
**Computer Vision Technology for Robotized Wire Harness Assembly**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-CV, cs-RO, cs.RO  
Keywords: Computer Vision  
[Paper Link](http://arxiv.org/abs/2309.13745v1)  

---


**ABSTRACT**  
Wire harnesses are essential hardware for electronic systems in modern automotive vehicles. With a shift in the automotive industry towards electrification and autonomous driving, more and more automotive electronics are responsible for energy transmission and safety-critical functions such as maneuvering, driver assistance, and safety system. This paradigm shift places more demand on automotive wiring harnesses from the safety perspective and stresses the greater importance of high-quality wire harness assembly in vehicles. However, most of the current operations of wire harness assembly are still performed manually by skilled workers, and some of the manual processes are problematic from different perspectives, such as quality control and ergonomics. There is also a persistent demand in the industry to increase competitiveness and gain market share. Hence, assuring assembly quality while improving ergonomics and optimizing labor costs is desired. Robotized assembly, accomplished by robots or in human-robot collaboration, is a key enabler for fulfilling the increasingly demanding quality and safety as it enables more replicable, transparent, and comprehensible processes than completely manual operations. However, robotized assembly of wire harnesses is challenging in real environments due to the flexibility of the deformable objects, though many preliminary automation solutions have been proposed under simplified industrial configurations. Previous research efforts have proposed the use of computer vision technology to facilitate robotized automation of wire harness assembly, enabling the robots to better perceive and manipulate the flexible wire harness. This article presents an overview on computer vision technology proposed for robotized wire harness assembly and derives research gaps that require further study to facilitate a more practical robotized assembly of wire harness.

{{</citation>}}


### (42/53) Boosting Offline Reinforcement Learning for Autonomous Driving with Hierarchical Latent Skills (Zenan Li et al., 2023)

{{<citation>}}

Zenan Li, Fan Nie, Qiao Sun, Fang Da, Hang Zhao. (2023)  
**Boosting Offline Reinforcement Learning for Autonomous Driving with Hierarchical Latent Skills**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.13614v1)  

---


**ABSTRACT**  
Learning-based vehicle planning is receiving increasing attention with the emergence of diverse driving simulators and large-scale driving datasets. While offline reinforcement learning (RL) is well suited for these safety-critical tasks, it still struggles to plan over extended periods. In this work, we present a skill-based framework that enhances offline RL to overcome the long-horizon vehicle planning challenge. Specifically, we design a variational autoencoder (VAE) to learn skills from offline demonstrations. To mitigate posterior collapse of common VAEs, we introduce a two-branch sequence encoder to capture both discrete options and continuous variations of the complex driving skills. The final policy treats learned skills as actions and can be trained by any off-the-shelf offline RL algorithms. This facilitates a shift in focus from per-step actions to temporally extended skills, thereby enabling long-term reasoning into the future. Extensive results on CARLA prove that our model consistently outperforms strong baselines at both training and new scenarios. Additional visualizations and experiments demonstrate the interpretability and transferability of extracted skills.

{{</citation>}}


## math.OC (1)



### (43/53) Data-Driven Superstabilization of Linear Systems under Quantization (Jared Miller et al., 2023)

{{<citation>}}

Jared Miller, Jian Zheng, Mario Sznaier, Chris Hixenbaugh. (2023)  
**Data-Driven Superstabilization of Linear Systems under Quantization**  

---
Primary Category: math.OC  
Categories: cs-SY, eess-SY, math-OC, math.OC  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2309.13712v1)  

---


**ABSTRACT**  
This paper focuses on the stabilization and regulation of linear systems affected by quantization in state-transition data and actuated input. The observed data are composed of tuples of current state, input, and the next state's interval ranges based on sensor quantization. Using an established characterization of input-logarithmically-quantized stabilization based on robustness to sector-bounded uncertainty, we formulate a nonconservative infinite-dimensional linear program that enforces superstabilization of all possible consistent systems under assumed priors. We solve this problem by posing a pair of exponentially-scaling linear programs, and demonstrate the success of our method on example quantized systems.

{{</citation>}}


## cs.HC (2)



### (44/53) 'Always Nice and Confident, Sometimes wrong': Developer's Experiences Engaging Generative AI Chatbots Versus Human-Powered Q&A Platforms (Jiachen Li et al., 2023)

{{<citation>}}

Jiachen Li, Elizabeth Mynatt, Varun Mishra, Jonathan Bell. (2023)  
**'Always Nice and Confident, Sometimes wrong': Developer's Experiences Engaging Generative AI Chatbots Versus Human-Powered Q&A Platforms**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs-SE, cs.HC  
Keywords: AI, ChatGPT, GPT, Generative AI  
[Paper Link](http://arxiv.org/abs/2309.13684v1)  

---


**ABSTRACT**  
Software engineers have historically relied on human-powered Q&A platforms, like Stack Overflow (SO), as coding aids. With the rise of generative AI, developers have adopted AI chatbots, such as ChatGPT, in their software development process. Recognizing the potential parallels between human-powered Q&A platforms and AI-powered question-based chatbots, we investigate and compare how developers integrate this assistance into their real-world coding experiences by conducting thematic analysis of Reddit posts. Through a comparative study of SO and ChatGPT, we identified each platform's strengths, use cases, and barriers. Our findings suggest that ChatGPT offers fast, clear, comprehensive responses and fosters a more respectful environment than SO. However, concerns about ChatGPT's reliability stem from its overly confident tone and the absence of validation mechanisms like SO's voting system. Based on these findings, we recommend leveraging each platform's unique features to improve developer experiences in the future.

{{</citation>}}


### (45/53) EvalLM: Interactive Evaluation of Large Language Model Prompts on User-Defined Criteria (Tae Soo Kim et al., 2023)

{{<citation>}}

Tae Soo Kim, Yoonjoo Lee, Jamin Shin, Young-Ho Kim, Juho Kim. (2023)  
**EvalLM: Interactive Evaluation of Large Language Model Prompts on User-Defined Criteria**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-CL, cs-HC, cs.HC  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.13633v1)  

---


**ABSTRACT**  
By simply composing prompts, developers can prototype novel generative applications with Large Language Models (LLMs). To refine prototypes into products, however, developers must iteratively revise prompts by evaluating outputs to diagnose weaknesses. Formative interviews (N=8) revealed that developers invest significant effort in manually evaluating outputs as they assess context-specific and subjective criteria. We present EvalLM, an interactive system for iteratively refining prompts by evaluating multiple outputs on user-defined criteria. By describing criteria in natural language, users can employ the system's LLM-based evaluator to get an overview of where prompts excel or fail, and improve these based on the evaluator's feedback. A comparative study (N=12) showed that EvalLM, when compared to manual evaluation, helped participants compose more diverse criteria, examine twice as many outputs, and reach satisfactory prompts with 59% fewer revisions. Beyond prompts, our work can be extended to augment model evaluation and alignment in specific application contexts.

{{</citation>}}


## eess.AS (3)



### (46/53) Cross-modal Alignment with Optimal Transport for CTC-based ASR (Xugang Lu et al., 2023)

{{<citation>}}

Xugang Lu, Peng Shen, Yu Tsao, Hisashi Kawai. (2023)  
**Cross-modal Alignment with Optimal Transport for CTC-based ASR**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.13650v1)  

---


**ABSTRACT**  
Temporal connectionist temporal classification (CTC)-based automatic speech recognition (ASR) is one of the most successful end to end (E2E) ASR frameworks. However, due to the token independence assumption in decoding, an external language model (LM) is required which destroys its fast parallel decoding property. Several studies have been proposed to transfer linguistic knowledge from a pretrained LM (PLM) to the CTC based ASR. Since the PLM is built from text while the acoustic model is trained with speech, a cross-modal alignment is required in order to transfer the context dependent linguistic knowledge from the PLM to acoustic encoding. In this study, we propose a novel cross-modal alignment algorithm based on optimal transport (OT). In the alignment process, a transport coupling matrix is obtained using OT, which is then utilized to transform a latent acoustic representation for matching the context-dependent linguistic features encoded by the PLM. Based on the alignment, the latent acoustic feature is forced to encode context dependent linguistic information. We integrate this latent acoustic feature to build conformer encoder-based CTC ASR system. On the AISHELL-1 data corpus, our system achieved 3.96% and 4.27% character error rate (CER) for dev and test sets, respectively, which corresponds to relative improvements of 28.39% and 29.42% compared to the baseline conformer CTC ASR system without cross-modal knowledge transfer.

{{</citation>}}


### (47/53) Efficient Black-Box Speaker Verification Model Adaptation with Reprogramming and Backend Learning (Jingyu Li et al., 2023)

{{<citation>}}

Jingyu Li, Tan Lee. (2023)  
**Efficient Black-Box Speaker Verification Model Adaptation with Reprogramming and Backend Learning**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Speaker Verification  
[Paper Link](http://arxiv.org/abs/2309.13605v1)  

---


**ABSTRACT**  
The development of deep neural networks (DNN) has significantly enhanced the performance of speaker verification (SV) systems in recent years. However, a critical issue that persists when applying DNN-based SV systems in practical applications is domain mismatch. To mitigate the performance degradation caused by the mismatch, domain adaptation becomes necessary. This paper introduces an approach to adapt DNN-based SV models by manipulating the learnable model inputs, inspired by the concept of adversarial reprogramming. The pre-trained SV model remains fixed and functions solely in the forward process, resembling a black-box model. A lightweight network is utilized to estimate the gradients for the learnable parameters at the input, which bypasses the gradient backpropagation through the black-box model. The reprogrammed output is processed by a two-layer backend learning module as the final adapted speaker embedding. The number of parameters involved in the gradient calculation is small in our design. With few additional parameters, the proposed method achieves both memory and parameter efficiency. The experiments are conducted in language mismatch scenarios. Using much less computation cost, the proposed method obtains close or superior performance to the fully finetuned models in our experiments, which demonstrates its effectiveness.

{{</citation>}}


### (48/53) Speech enhancement with frequency domain auto-regressive modeling (Anurenjan Purushothaman et al., 2023)

{{<citation>}}

Anurenjan Purushothaman, Debottam Dutta, Rohit Kumar, Sriram Ganapathy. (2023)  
**Speech enhancement with frequency domain auto-regressive modeling**  

---
Primary Category: eess.AS  
Categories: cs-AI, cs-SD, eess-AS, eess.AS  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2309.13537v1)  

---


**ABSTRACT**  
Speech applications in far-field real world settings often deal with signals that are corrupted by reverberation. The task of dereverberation constitutes an important step to improve the audible quality and to reduce the error rates in applications like automatic speech recognition (ASR). We propose a unified framework of speech dereverberation for improving the speech quality and the ASR performance using the approach of envelope-carrier decomposition provided by an autoregressive (AR) model. The AR model is applied in the frequency domain of the sub-band speech signals to separate the envelope and carrier parts. A novel neural architecture based on dual path long short term memory (DPLSTM) model is proposed, which jointly enhances the sub-band envelope and carrier components. The dereverberated envelope-carrier signals are modulated and the sub-band signals are synthesized to reconstruct the audio signal back. The DPLSTM model for dereverberation of envelope and carrier components also allows the joint learning of the network weights for the down stream ASR task. In the ASR tasks on the REVERB challenge dataset as well as on the VOiCES dataset, we illustrate that the joint learning of speech dereverberation network and the E2E ASR model yields significant performance improvements over the baseline ASR system trained on log-mel spectrogram as well as other benchmarks for dereverberation (average relative improvements of 10-24% over the baseline system). The speech quality improvements, evaluated using subjective listening tests, further highlight the improved quality of the reconstructed audio.

{{</citation>}}


## cs.CR (2)



### (49/53) Digital Twins and the Future of their Use Enabling Shift Left and Shift Right Cybersecurity Operations (Ahmad Mohsin et al., 2023)

{{<citation>}}

Ahmad Mohsin, Helge Janicke, Surya Nepal, David Holmes. (2023)  
**Digital Twins and the Future of their Use Enabling Shift Left and Shift Right Cybersecurity Operations**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-ET, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2309.13612v1)  

---


**ABSTRACT**  
Digital Twins (DTs), optimize operations and monitor performance in Smart Critical Systems (SCS) domains like smart grids and manufacturing. DT-based cybersecurity solutions are in their infancy, lacking a unified strategy to overcome challenges spanning next three to five decades. These challenges include reliable data accessibility from Cyber-Physical Systems (CPS), operating in unpredictable environments. Reliable data sources are pivotal for intelligent cybersecurity operations aided with underlying modeling capabilities across the SCS lifecycle, necessitating a DT. To address these challenges, we propose Security Digital Twins (SDTs) collecting realtime data from CPS, requiring the Shift Left and Shift Right (SLSR) design paradigm for SDT to implement both design time and runtime cybersecurity operations. Incorporating virtual CPS components (VC) in Cloud/Edge, data fusion to SDT models is enabled with high reliability, providing threat insights and enhancing cyber resilience. VC-enabled SDT ensures accurate data feeds for security monitoring for both design and runtime. This design paradigm shift propagates innovative SDT modeling and analytics for securing future critical systems. This vision paper outlines intelligent SDT design through innovative techniques, exploring hybrid intelligence with data-driven and rule-based semantic SDT models. Various operational use cases are discussed for securing smart critical systems through underlying modeling and analytics capabilities.

{{</citation>}}


### (50/53) Seeing Is Not Always Believing: Invisible Collision Attack and Defence on Pre-Trained Models (Minghang Deng et al., 2023)

{{<citation>}}

Minghang Deng, Zhong Zhang, Junming Shao. (2023)  
**Seeing Is Not Always Believing: Invisible Collision Attack and Defence on Pre-Trained Models**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: BERT, GPT, Pre-Trained Model  
[Paper Link](http://arxiv.org/abs/2309.13579v1)  

---


**ABSTRACT**  
Large-scale pre-trained models (PTMs) such as BERT and GPT have achieved great success in diverse fields. The typical paradigm is to pre-train a big deep learning model on large-scale data sets, and then fine-tune the model on small task-specific data sets for downstream tasks. Although PTMs have rapidly progressed with wide real-world applications, they also pose significant risks of potential attacks. Existing backdoor attacks or data poisoning methods often build up the assumption that the attacker invades the computers of victims or accesses the target data, which is challenging in real-world scenarios. In this paper, we propose a novel framework for an invisible attack on PTMs with enhanced MD5 collision. The key idea is to generate two equal-size models with the same MD5 checksum by leveraging the MD5 chosen-prefix collision. Afterwards, the two ``same" models will be deployed on public websites to induce victims to download the poisoned model. Unlike conventional attacks on deep learning models, this new attack is flexible, covert, and model-independent. Additionally, we propose a simple defensive strategy for recognizing the MD5 chosen-prefix collision and provide a theoretical justification for its feasibility. We extensively validate the effectiveness and stealthiness of our proposed attack and defensive method on different models and data sets.

{{</citation>}}


## math.NA (1)



### (51/53) Shape Optimization by Constrained First-Order Least Mean Approximation (Gerhard Starke, 2023)

{{<citation>}}

Gerhard Starke. (2023)  
**Shape Optimization by Constrained First-Order Least Mean Approximation**  

---
Primary Category: math.NA  
Categories: 65N30, 49M05, cs-NA, math-NA, math.NA  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.13595v1)  

---


**ABSTRACT**  
In this work, the problem of shape optimization, subject to PDE constraints, is reformulated as an $L^p$ best approximation problem under divergence constraints to the shape tensor introduced in Laurain and Sturm: ESAIM Math. Model. Numer. Anal. 50 (2016). More precisely, the main result of this paper states that the $L^p$ distance of the above approximation problem is equal to the dual norm of the shape derivative considered as a functional on $W^{1,p^\ast}$ (where $1/p + 1/p^\ast = 1$). This implies that for any given shape, one can evaluate its distance from being a stationary one with respect to the shape derivative by simply solving the associated $L^p$-type least mean approximation problem. Moreover, the Lagrange multiplier for the divergence constraint turns out to be the shape deformation of steepest descent. This provides a way, as an alternative to the approach by Deckelnick, Herbert and Hinze: ESAIM Control Optim. Calc. Var. 28 (2022), for computing shape gradients in $W^{1,p^\ast}$ for $p^\ast \in ( 2 , \infty )$. The discretization of the least mean approximation problem is done with (lowest-order) matrix-valued Raviart-Thomas finite element spaces leading to piecewise constant approximations of the shape deformation acting as Lagrange multiplier. Admissible deformations in $W^{1,p^\ast}$ to be used in a shape gradient iteration are reconstructed locally. Our computational results confirm that the $L^p$ distance of the best approximation does indeed measure the distance of the considered shape to optimality. Also confirmed by our computational tests are the observations that choosing $p^\ast$ (much) larger than 2 (which means that $p$ must be close to 1 in our best approximation problem) decreases the chance of encountering mesh degeneracy during the shape gradient iteration.

{{</citation>}}


## eess.IV (1)



### (52/53) Matrix Completion-Informed Deep Unfolded Equilibrium Models for Self-Supervised k-Space Interpolation in MRI (Chen Luo et al., 2023)

{{<citation>}}

Chen Luo, Huayu Wang, Taofeng Xie, Qiyu Jin, Guoqing Chen, Zhuo-Xu Cui, Dong Liang. (2023)  
**Matrix Completion-Informed Deep Unfolded Equilibrium Models for Self-Supervised k-Space Interpolation in MRI**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2309.13571v1)  

---


**ABSTRACT**  
Recently, regularization model-driven deep learning (DL) has gained significant attention due to its ability to leverage the potent representational capabilities of DL while retaining the theoretical guarantees of regularization models. However, most of these methods are tailored for supervised learning scenarios that necessitate fully sampled labels, which can pose challenges in practical MRI applications. To tackle this challenge, we propose a self-supervised DL approach for accelerated MRI that is theoretically guaranteed and does not rely on fully sampled labels. Specifically, we achieve neural network structure regularization by exploiting the inherent structural low-rankness of the $k$-space data. Simultaneously, we constrain the network structure to resemble a nonexpansive mapping, ensuring the network's convergence to a fixed point. Thanks to this well-defined network structure, this fixed point can completely reconstruct the missing $k$-space data based on matrix completion theory, even in situations where full-sampled labels are unavailable. Experiments validate the effectiveness of our proposed method and demonstrate its superiority over existing self-supervised approaches and traditional regularization methods, achieving performance comparable to that of supervised learning methods in certain scenarios.

{{</citation>}}


## cs.IT (1)



### (53/53) Integrated Sensing and Communications for IoT: Synergies with Key 6G Technology Enablers (Aryan Kaushik et al., 2023)

{{<citation>}}

Aryan Kaushik, Rohit Singh, Ming Li, Honghao Luo, Shalanika Dayarathna, Rajitha Senanayake, Xueli An, Richard A. Stirling-Gallacher, Wonjae Shin, Marco Di Renzo. (2023)  
**Integrated Sensing and Communications for IoT: Synergies with Key 6G Technology Enablers**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, math-IT  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.13542v1)  

---


**ABSTRACT**  
The Internet of Things (IoT) and wireless generations have been evolving simultaneously for the past few decades. Built upon wireless communication and sensing technologies, IoT networks are usually evaluated based on metrics that measure the device ability to sense information and effectively share it with the network, which makes Integrated Sensing and Communication (ISAC) a pivotal candidate for the sixth-generation (6G) IoT standards. This paper reveals several innovative aspects of ISAC from an IoT perspective in 6G, empowering various modern IoT use cases and key technology enablers. Moreover, we address the challenges and future potential of ISAC-enabled IoT, including synergies with Reconfigurable Intelligent Surfaces (RIS), Artificial Intelligence (AI), and key updates of ISAC-IoT in 6G standardization. Furthermore, several evolutionary concepts are introduced to open future research in 6G ISAC-IoT, including the interplay with Non-Terrestrial Networks (NTN) and Orthogonal Time-Frequency Space (OTFS) modulation.

{{</citation>}}
