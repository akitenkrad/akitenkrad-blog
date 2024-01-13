---
draft: false
title: "arXiv @ 2024.01.08"
date: 2024-01-08
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2024"]
menu:
  sidebar:
    name: "arXiv @ 2024.01.08"
    identifier: arxiv_20240108
    parent: 202401_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.LG (15)](#cslg-15)
- [cs.CL (9)](#cscl-9)
- [cs.CR (2)](#cscr-2)
- [cs.CV (10)](#cscv-10)
- [cs.NI (1)](#csni-1)
- [eess.IV (2)](#eessiv-2)
- [eess.AS (1)](#eessas-1)
- [math.OC (1)](#mathoc-1)
- [cs.HC (2)](#cshc-2)
- [cs.DC (1)](#csdc-1)
- [cs.CG (1)](#cscg-1)
- [cs.RO (2)](#csro-2)
- [cs.AI (3)](#csai-3)
- [cs.IR (1)](#csir-1)
- [cs.SE (1)](#csse-1)
- [cs.CY (1)](#cscy-1)

## cs.LG (15)



### (1/53) Attention and Autoencoder Hybrid Model for Unsupervised Online Anomaly Detection (Seyed Amirhossein Najafi et al., 2024)

{{<citation>}}

Seyed Amirhossein Najafi, Mohammad Hassan Asemani, Peyman Setoodeh. (2024)  
**Attention and Autoencoder Hybrid Model for Unsupervised Online Anomaly Detection**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, eess-SP  
Keywords: Anomaly Detection, Attention  
[Paper Link](http://arxiv.org/abs/2401.03322v1)  

---


**ABSTRACT**  
This paper introduces a hybrid attention and autoencoder (AE) model for unsupervised online anomaly detection in time series. The autoencoder captures local structural patterns in short embeddings, while the attention model learns long-term features, facilitating parallel computing with positional encoding. Unique in its approach, our proposed hybrid model combines attention and autoencoder for the first time in time series anomaly detection. It employs an attention-based mechanism, akin to the deep transformer model, with key architectural modifications for predicting the next time step window in the autoencoder's latent space. The model utilizes a threshold from the validation dataset for anomaly detection and introduces an alternative method based on analyzing the first statistical moment of error, improving accuracy without dependence on a validation dataset. Evaluation on diverse real-world benchmark datasets and comparing with other well-established models, confirms the effectiveness of our proposed model in anomaly detection.

{{</citation>}}


### (2/53) On Sample-Efficient Offline Reinforcement Learning: Data Diversity, Posterior Sampling, and Beyond (Thanh Nguyen-Tang et al., 2024)

{{<citation>}}

Thanh Nguyen-Tang, Raman Arora. (2024)  
**On Sample-Efficient Offline Reinforcement Learning: Data Diversity, Posterior Sampling, and Beyond**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, stat-ML  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.03301v1)  

---


**ABSTRACT**  
We seek to understand what facilitates sample-efficient learning from historical datasets for sequential decision-making, a problem that is popularly known as offline reinforcement learning (RL). Further, we are interested in algorithms that enjoy sample efficiency while leveraging (value) function approximation. In this paper, we address these fundamental questions by (i) proposing a notion of data diversity that subsumes the previous notions of coverage measures in offline RL and (ii) using this notion to {unify} three distinct classes of offline RL algorithms based on version spaces (VS), regularized optimization (RO), and posterior sampling (PS). We establish that VS-based, RO-based, and PS-based algorithms, under standard assumptions, achieve \emph{comparable} sample efficiency, which recovers the state-of-the-art sub-optimality bounds for finite and linear model classes with the standard assumptions. This result is surprising, given that the prior work suggested an unfavorable sample complexity of the RO-based algorithm compared to the VS-based algorithm, whereas posterior sampling is rarely considered in offline RL due to its explorative nature. Notably, our proposed model-free PS-based algorithm for offline RL is {novel}, with sub-optimality bounds that are {frequentist} (i.e., worst-case) in nature.

{{</citation>}}


### (3/53) FedTGP: Trainable Global Prototypes with Adaptive-Margin-Enhanced Contrastive Learning for Data and Model Heterogeneity in Federated Learning (Jianqing Zhang et al., 2024)

{{<citation>}}

Jianqing Zhang, Yang Liu, Yang Hua, Jian Cao. (2024)  
**FedTGP: Trainable Global Prototypes with Adaptive-Margin-Enhanced Contrastive Learning for Data and Model Heterogeneity in Federated Learning**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-DC, cs-LG, cs.LG  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2401.03230v1)  

---


**ABSTRACT**  
Recently, Heterogeneous Federated Learning (HtFL) has attracted attention due to its ability to support heterogeneous models and data. To reduce the high communication cost of transmitting model parameters, a major challenge in HtFL, prototype-based HtFL methods are proposed to solely share class representatives, a.k.a, prototypes, among heterogeneous clients while maintaining the privacy of clients' models. However, these prototypes are naively aggregated into global prototypes on the server using weighted averaging, resulting in suboptimal global knowledge which negatively impacts the performance of clients. To overcome this challenge, we introduce a novel HtFL approach called FedTGP, which leverages our Adaptive-margin-enhanced Contrastive Learning (ACL) to learn Trainable Global Prototypes (TGP) on the server. By incorporating ACL, our approach enhances prototype separability while preserving semantic meaning. Extensive experiments with twelve heterogeneous models demonstrate that our FedTGP surpasses state-of-the-art methods by up to 9.08% in accuracy while maintaining the communication and privacy advantages of prototype-based HtFL. Our code is available at https://github.com/TsingZ0/FedTGP.

{{</citation>}}


### (4/53) End-to-End Anti-Backdoor Learning on Images and Time Series (Yujing Jiang et al., 2024)

{{<citation>}}

Yujing Jiang, Xingjun Ma, Sarah Monazam Erfani, Yige Li, James Bailey. (2024)  
**End-to-End Anti-Backdoor Learning on Images and Time Series**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2401.03215v1)  

---


**ABSTRACT**  
Backdoor attacks present a substantial security concern for deep learning models, especially those utilized in applications critical to safety and security. These attacks manipulate model behavior by embedding a hidden trigger during the training phase, allowing unauthorized control over the model's output during inference time. Although numerous defenses exist for image classification models, there is a conspicuous absence of defenses tailored for time series data, as well as an end-to-end solution capable of training clean models on poisoned data. To address this gap, this paper builds upon Anti-Backdoor Learning (ABL) and introduces an innovative method, End-to-End Anti-Backdoor Learning (E2ABL), for robust training against backdoor attacks. Unlike the original ABL, which employs a two-stage training procedure, E2ABL accomplishes end-to-end training through an additional classification head linked to the shallow layers of a Deep Neural Network (DNN). This secondary head actively identifies potential backdoor triggers, allowing the model to dynamically cleanse these samples and their corresponding labels during training. Our experiments reveal that E2ABL significantly improves on existing defenses and is effective against a broad range of backdoor attacks in both image and time series domains.

{{</citation>}}


### (5/53) Understanding Representation Learnability of Nonlinear Self-Supervised Learning (Ruofeng Yang et al., 2024)

{{<citation>}}

Ruofeng Yang, Xiangyuan Li, Bo Jiang, Shuai Li. (2024)  
**Understanding Representation Learnability of Nonlinear Self-Supervised Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2401.03214v1)  

---


**ABSTRACT**  
Self-supervised learning (SSL) has empirically shown its data representation learnability in many downstream tasks. There are only a few theoretical works on data representation learnability, and many of those focus on final data representation, treating the nonlinear neural network as a ``black box". However, the accurate learning results of neural networks are crucial for describing the data distribution features learned by SSL models. Our paper is the first to analyze the learning results of the nonlinear SSL model accurately. We consider a toy data distribution that contains two features: the label-related feature and the hidden feature. Unlike previous linear setting work that depends on closed-form solutions, we use the gradient descent algorithm to train a 1-layer nonlinear SSL model with a certain initialization region and prove that the model converges to a local minimum. Furthermore, different from the complex iterative analysis, we propose a new analysis process which uses the exact version of Inverse Function Theorem to accurately describe the features learned by the local minimum. With this local minimum, we prove that the nonlinear SSL model can capture the label-related feature and hidden feature at the same time. In contrast, the nonlinear supervised learning (SL) model can only learn the label-related feature. We also present the learning processes and results of the nonlinear SSL and SL model via simulation experiments.

{{</citation>}}


### (6/53) Exploration of Adolescent Depression Risk Prediction Based on Census Surveys and General Life Issues (Qiang Li et al., 2024)

{{<citation>}}

Qiang Li, Yufeng Wu, Zhan Xu, Hefeng Zhou. (2024)  
**Exploration of Adolescent Depression Risk Prediction Based on Census Surveys and General Life Issues**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CY, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.03171v1)  

---


**ABSTRACT**  
In contemporary society, the escalating pressures of life and work have propelled psychological disorders to the forefront of modern health concerns, an issue that has been further accentuated by the COVID-19 pandemic. The prevalence of depression among adolescents is steadily increasing, and traditional diagnostic methods, which rely on scales or interviews, prove particularly inadequate for detecting depression in young people. Addressing these challenges, numerous AI-based methods for assisting in the diagnosis of mental health issues have emerged. However, most of these methods center around fundamental issues with scales or use multimodal approaches like facial expression recognition. Diagnosis of depression risk based on everyday habits and behaviors has been limited to small-scale qualitative studies. Our research leverages adolescent census data to predict depression risk, focusing on children's experiences with depression and their daily life situations. We introduced a method for managing severely imbalanced high-dimensional data and an adaptive predictive approach tailored to data structure characteristics. Furthermore, we proposed a cloud-based architecture for automatic online learning and data updates. This study utilized publicly available NSCH youth census data from 2020 to 2022, encompassing nearly 150,000 data entries. We conducted basic data analyses and predictive experiments, demonstrating significant performance improvements over standard machine learning and deep learning algorithms. This affirmed our data processing method's broad applicability in handling imbalanced medical data. Diverging from typical predictive method research, our study presents a comprehensive architectural solution, considering a wider array of user needs.

{{</citation>}}


### (7/53) An Empirical Investigation of Value-Based Multi-objective Reinforcement Learning for Stochastic Environments (Kewen Ding et al., 2024)

{{<citation>}}

Kewen Ding, Peter Vamplew, Cameron Foale, Richard Dazeley. (2024)  
**An Empirical Investigation of Value-Based Multi-objective Reinforcement Learning for Stochastic Environments**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.03163v1)  

---


**ABSTRACT**  
One common approach to solve multi-objective reinforcement learning (MORL) problems is to extend conventional Q-learning by using vector Q-values in combination with a utility function. However issues can arise with this approach in the context of stochastic environments, particularly when optimising for the Scalarised Expected Reward (SER) criterion. This paper extends prior research, providing a detailed examination of the factors influencing the frequency with which value-based MORL Q-learning algorithms learn the SER-optimal policy for an environment with stochastic state transitions. We empirically examine several variations of the core multi-objective Q-learning algorithm as well as reward engineering approaches, and demonstrate the limitations of these methods. In particular, we highlight the critical impact of the noisy Q-value estimates issue on the stability and convergence of these algorithms.

{{</citation>}}


### (8/53) Human as AI Mentor: Enhanced Human-in-the-loop Reinforcement Learning for Safe and Efficient Autonomous Driving (Zilin Huang et al., 2024)

{{<citation>}}

Zilin Huang, Zihao Sheng, Chengyuan Ma, Sikai Chen. (2024)  
**Human as AI Mentor: Enhanced Human-in-the-loop Reinforcement Learning for Safe and Efficient Autonomous Driving**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-RO, cs.LG  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.03160v2)  

---


**ABSTRACT**  
Despite significant progress in autonomous vehicles (AVs), the development of driving policies that ensure both the safety of AVs and traffic flow efficiency has not yet been fully explored. In this paper, we propose an enhanced human-in-the-loop reinforcement learning method, termed the Human as AI mentor-based deep reinforcement learning (HAIM-DRL) framework, which facilitates safe and efficient autonomous driving in mixed traffic platoon. Drawing inspiration from the human learning process, we first introduce an innovative learning paradigm that effectively injects human intelligence into AI, termed Human as AI mentor (HAIM). In this paradigm, the human expert serves as a mentor to the AI agent. While allowing the agent to sufficiently explore uncertain environments, the human expert can take control in dangerous situations and demonstrate correct actions to avoid potential accidents. On the other hand, the agent could be guided to minimize traffic flow disturbance, thereby optimizing traffic flow efficiency. In detail, HAIM-DRL leverages data collected from free exploration and partial human demonstrations as its two training sources. Remarkably, we circumvent the intricate process of manually designing reward functions; instead, we directly derive proxy state-action values from partial human demonstrations to guide the agents' policy learning. Additionally, we employ a minimal intervention technique to reduce the human mentor's cognitive load. Comparative results show that HAIM-DRL outperforms traditional methods in driving safety, sampling efficiency, mitigation of traffic flow disturbance, and generalizability to unseen traffic scenarios. The code and demo videos for this paper can be accessed at: https://zilin-huang.github.io/HAIM-DRL-website/

{{</citation>}}


### (9/53) Data-Dependent Stability Analysis of Adversarial Training (Yihan Wang et al., 2024)

{{<citation>}}

Yihan Wang, Shuang Liu, Xiao-Shan Gao. (2024)  
**Data-Dependent Stability Analysis of Adversarial Training**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Adversarial Training  
[Paper Link](http://arxiv.org/abs/2401.03156v1)  

---


**ABSTRACT**  
Stability analysis is an essential aspect of studying the generalization ability of deep learning, as it involves deriving generalization bounds for stochastic gradient descent-based training algorithms. Adversarial training is the most widely used defense against adversarial example attacks. However, previous generalization bounds for adversarial training have not included information regarding the data distribution. In this paper, we fill this gap by providing generalization bounds for stochastic gradient descent-based adversarial training that incorporate data distribution information. We utilize the concepts of on-average stability and high-order approximate Lipschitz conditions to examine how changes in data distribution and adversarial budget can affect robust generalization gaps. Our derived generalization bounds for both convex and non-convex losses are at least as good as the uniform stability-based counterparts which do not include data distribution information. Furthermore, our findings demonstrate how distribution shifts from data poisoning attacks can impact robust generalization.

{{</citation>}}


### (10/53) SPQR: Controlling Q-ensemble Independence with Spiked Random Model for Reinforcement Learning (Dohyeok Lee et al., 2024)

{{<citation>}}

Dohyeok Lee, Seungyub Han, Taehyun Cho, Jungwoo Lee. (2024)  
**SPQR: Controlling Q-ensemble Independence with Spiked Random Model for Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, stat-ML  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.03137v1)  

---


**ABSTRACT**  
Alleviating overestimation bias is a critical challenge for deep reinforcement learning to achieve successful performance on more complex tasks or offline datasets containing out-of-distribution data. In order to overcome overestimation bias, ensemble methods for Q-learning have been investigated to exploit the diversity of multiple Q-functions. Since network initialization has been the predominant approach to promote diversity in Q-functions, heuristically designed diversity injection methods have been studied in the literature. However, previous studies have not attempted to approach guaranteed independence over an ensemble from a theoretical perspective. By introducing a novel regularization loss for Q-ensemble independence based on random matrix theory, we propose spiked Wishart Q-ensemble independence regularization (SPQR) for reinforcement learning. Specifically, we modify the intractable hypothesis testing criterion for the Q-ensemble independence into a tractable KL divergence between the spectral distribution of the Q-ensemble and the target Wigner's semicircle distribution. We implement SPQR in several online and offline ensemble Q-learning algorithms. In the experiments, SPQR outperforms the baseline algorithms in both online and offline RL benchmarks.

{{</citation>}}


### (11/53) TimeGraphs: Graph-based Temporal Reasoning (Paridhi Maheshwari et al., 2024)

{{<citation>}}

Paridhi Maheshwari, Hongyu Ren, Yanan Wang, Rok Sosic, Jure Leskovec. (2024)  
**TimeGraphs: Graph-based Temporal Reasoning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2401.03134v1)  

---


**ABSTRACT**  
Many real-world systems exhibit temporal, dynamic behaviors, which are captured as time series of complex agent interactions. To perform temporal reasoning, current methods primarily encode temporal dynamics through simple sequence-based models. However, in general these models fail to efficiently capture the full spectrum of rich dynamics in the input, since the dynamics is not uniformly distributed. In particular, relevant information might be harder to extract and computing power is wasted for processing all individual timesteps, even if they contain no significant changes or no new information. Here we propose TimeGraphs, a novel approach that characterizes dynamic interactions as a hierarchical temporal graph, diverging from traditional sequential representations. Our approach models the interactions using a compact graph-based representation, enabling adaptive reasoning across diverse time scales. Adopting a self-supervised method, TimeGraphs constructs a multi-level event hierarchy from a temporal input, which is then used to efficiently reason about the unevenly distributed dynamics. This construction process is scalable and incremental to accommodate streaming data. We evaluate TimeGraphs on multiple datasets with complex, dynamic agent interactions, including a football simulator, the Resistance game, and the MOMA human activity dataset. The results demonstrate both robustness and efficiency of TimeGraphs on a range of temporal reasoning tasks. Our approach obtains state-of-the-art performance and leads to a performance increase of up to 12.2% on event prediction and recognition tasks over current approaches. Our experiments further demonstrate a wide array of capabilities including zero-shot generalization, robustness in case of data sparsity, and adaptability to streaming data flow.

{{</citation>}}


### (12/53) A Physics-guided Generative AI Toolkit for Geophysical Monitoring (Junhuan Yang et al., 2024)

{{<citation>}}

Junhuan Yang, Hanchen Wang, Yi Sheng, Youzuo Lin, Lei Yang. (2024)  
**A Physics-guided Generative AI Toolkit for Geophysical Monitoring**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG, eess-SP, physics-geo-ph  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2401.03131v1)  

---


**ABSTRACT**  
Full-waveform inversion (FWI) plays a vital role in geoscience to explore the subsurface. It utilizes the seismic wave to image the subsurface velocity map. As the machine learning (ML) technique evolves, the data-driven approaches using ML for FWI tasks have emerged, offering enhanced accuracy and reduced computational cost compared to traditional physics-based methods. However, a common challenge in geoscience, the unprivileged data, severely limits ML effectiveness. The issue becomes even worse during model pruning, a step essential in geoscience due to environmental complexities. To tackle this, we introduce the EdGeo toolkit, which employs a diffusion-based model guided by physics principles to generate high-fidelity velocity maps. The toolkit uses the acoustic wave equation to generate corresponding seismic waveform data, facilitating the fine-tuning of pruned ML models. Our results demonstrate significant improvements in SSIM scores and reduction in both MAE and MSE across various pruning ratios. Notably, the ML model fine-tuned using data generated by EdGeo yields superior quality of velocity maps, especially in representing unprivileged features, outperforming other existing methods.

{{</citation>}}


### (13/53) GLISP: A Scalable GNN Learning System by Exploiting Inherent Structural Properties of Graphs (Zhongshu Zhu et al., 2024)

{{<citation>}}

Zhongshu Zhu, Bin Jing, Xiaopei Wan, Zhizhen Liu, Lei Liang, Jun zhou. (2024)  
**GLISP: A Scalable GNN Learning System by Exploiting Inherent Structural Properties of Graphs**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2401.03114v1)  

---


**ABSTRACT**  
As a powerful tool for modeling graph data, Graph Neural Networks (GNNs) have received increasing attention in both academia and industry. Nevertheless, it is notoriously difficult to deploy GNNs on industrial scale graphs, due to their huge data size and complex topological structures. In this paper, we propose GLISP, a sampling based GNN learning system for industrial scale graphs. By exploiting the inherent structural properties of graphs, such as power law distribution and data locality, GLISP addresses the scalability and performance issues that arise at different stages of the graph learning process. GLISP consists of three core components: graph partitioner, graph sampling service and graph inference engine. The graph partitioner adopts the proposed vertex-cut graph partitioning algorithm AdaDNE to produce balanced partitioning for power law graphs, which is essential for sampling based GNN systems. The graph sampling service employs a load balancing design that allows the one hop sampling request of high degree vertices to be handled by multiple servers. In conjunction with the memory efficient data structure, the efficiency and scalability are effectively improved. The graph inference engine splits the $K$-layer GNN into $K$ slices and caches the vertex embeddings produced by each slice in the data locality aware hybrid caching system for reuse, thus completely eliminating redundant computation caused by the data dependency of graph. Extensive experiments show that GLISP achieves up to $6.53\times$ and $70.77\times$ speedups over existing GNN systems for training and inference tasks, respectively, and can scale to the graph with over 10 billion vertices and 40 billion edges with limited resources.

{{</citation>}}


### (14/53) When To Grow? A Fitting Risk-Aware Policy for Layer Growing in Deep Neural Networks (Haihang Wu et al., 2024)

{{<citation>}}

Haihang Wu, Wei Wang, Tamasha Malepathirana, Damith Senanayake, Denny Oetomo, Saman Halgamuge. (2024)  
**When To Grow? A Fitting Risk-Aware Policy for Layer Growing in Deep Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2401.03104v1)  

---


**ABSTRACT**  
Neural growth is the process of growing a small neural network to a large network and has been utilized to accelerate the training of deep neural networks. One crucial aspect of neural growth is determining the optimal growth timing. However, few studies investigate this systematically. Our study reveals that neural growth inherently exhibits a regularization effect, whose intensity is influenced by the chosen policy for growth timing. While this regularization effect may mitigate the overfitting risk of the model, it may lead to a notable accuracy drop when the model underfits. Yet, current approaches have not addressed this issue due to their lack of consideration of the regularization effect from neural growth. Motivated by these findings, we propose an under/over fitting risk-aware growth timing policy, which automatically adjusts the growth timing informed by the level of potential under/overfitting risks to address both risks. Comprehensive experiments conducted using CIFAR-10/100 and ImageNet datasets show that the proposed policy achieves accuracy improvements of up to 1.3% in models prone to underfitting while achieving similar accuracies in models suffering from overfitting compared to the existing methods.

{{</citation>}}


### (15/53) Plug-and-Play Transformer Modules for Test-Time Adaptation (Xiangyu Chang et al., 2024)

{{<citation>}}

Xiangyu Chang, Sk Miraj Ahmed, Srikanth V. Krishnamurthy, Basak Guler, Ananthram Swami, Samet Oymak, Amit K. Roy-Chowdhury. (2024)  
**Plug-and-Play Transformer Modules for Test-Time Adaptation**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.04130v2)  

---


**ABSTRACT**  
Parameter-efficient tuning (PET) methods such as LoRA, Adapter, and Visual Prompt Tuning (VPT) have found success in enabling adaptation to new domains by tuning small modules within a transformer model. However, the number of domains encountered during test time can be very large, and the data is usually unlabeled. Thus, adaptation to new domains is challenging; it is also impractical to generate customized tuned modules for each such domain. Toward addressing these challenges, this work introduces PLUTO: a Plug-and-pLay modUlar Test-time domain adaptatiOn strategy. We pre-train a large set of modules, each specialized for different source domains, effectively creating a ``module store''. Given a target domain with few-shot unlabeled data, we introduce an unsupervised test-time adaptation (TTA) method to (1) select a sparse subset of relevant modules from this store and (2) create a weighted combination of selected modules without tuning their weights. This plug-and-play nature enables us to harness multiple most-relevant source domains in a single inference call. Comprehensive evaluations demonstrate that PLUTO uniformly outperforms alternative TTA methods and that selecting $\leq$5 modules suffice to extract most of the benefit. At a high level, our method equips pre-trained transformers with the capability to dynamically adapt to new domains, motivating a new paradigm for efficient and scalable domain adaptation.

{{</citation>}}


## cs.CL (9)



### (16/53) PIXAR: Auto-Regressive Language Modeling in Pixel Space (Yintao Tai et al., 2024)

{{<citation>}}

Yintao Tai, Xiyang Liao, Alessandro Suglia, Antonio Vergari. (2024)  
**PIXAR: Auto-Regressive Language Modeling in Pixel Space**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2401.03321v1)  

---


**ABSTRACT**  
Recent works showed the possibility of building open-vocabulary large language models (LLMs) that directly operate on pixel representations and are implemented as encoder-decoder models that reconstruct masked image patches of rendered text. However, these pixel-based LLMs are limited to autoencoding tasks and cannot generate new text as images. As such, they cannot be used for open-answer or generative language tasks. In this work, we overcome this limitation and introduce PIXAR, the first pixel-based autoregressive LLM that does not rely on a pre-defined vocabulary for both input and output text. Consisting of only a decoder, PIXAR can answer free-form generative tasks while keeping the text representation learning performance on par with previous encoder-decoder models. Furthermore, we highlight the challenges to autoregressively generate non-blurred text as images and link this to the usual maximum likelihood objective. We propose a simple adversarial pretraining that significantly improves the readability and performance of PIXAR making it comparable to GPT2 on short text generation tasks. This paves the way to building open-vocabulary LLMs that are usable for free-form generative tasks and questions the necessity of the usual symbolic input representation -- text as tokens -- for these challenging tasks.

{{</citation>}}


### (17/53) Reflections on Inductive Thematic Saturation as a potential metric for measuring the validity of an inductive Thematic Analysis with LLMs (Stefano De Paoli et al., 2024)

{{<citation>}}

Stefano De Paoli, Walter Stan Mathis. (2024)  
**Reflections on Inductive Thematic Saturation as a potential metric for measuring the validity of an inductive Thematic Analysis with LLMs**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.03239v1)  

---


**ABSTRACT**  
This paper presents a set of reflections on saturation and the use of Large Language Models (LLMs) for performing Thematic Analysis (TA). The paper suggests that initial thematic saturation (ITS) could be used as a metric to assess part of the transactional validity of TA with LLM, focusing on the initial coding. The paper presents the initial coding of two datasets of different sizes, and it reflects on how the LLM reaches some form of analytical saturation during the coding. The procedure proposed in this work leads to the creation of two codebooks, one comprising the total cumulative initial codes and the other the total unique codes. The paper proposes a metric to synthetically measure ITS using a simple mathematical calculation employing the ratio between slopes of cumulative codes and unique codes. The paper contributes to the initial body of work exploring how to perform qualitative analysis with LLMs.

{{</citation>}}


### (18/53) The Dawn After the Dark: An Empirical Study on Factuality Hallucination in Large Language Models (Junyi Li et al., 2024)

{{<citation>}}

Junyi Li, Jie Chen, Ruiyang Ren, Xiaoxue Cheng, Wayne Xin Zhao, Jian-Yun Nie, Ji-Rong Wen. (2024)  
**The Dawn After the Dark: An Empirical Study on Factuality Hallucination in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2401.03205v1)  

---


**ABSTRACT**  
In the era of large language models (LLMs), hallucination (i.e., the tendency to generate factually incorrect content) poses great challenge to trustworthy and reliable deployment of LLMs in real-world applications. To tackle the LLM hallucination, three key questions should be well studied: how to detect hallucinations (detection), why do LLMs hallucinate (source), and what can be done to mitigate them (mitigation). To address these challenges, this work presents a systematic empirical study on LLM hallucination, focused on the the three aspects of hallucination detection, source and mitigation. Specially, we construct a new hallucination benchmark HaluEval 2.0, and designs a simple yet effective detection method for LLM hallucination. Furthermore, we zoom into the different training or utilization stages of LLMs and extensively analyze the potential factors that lead to the LLM hallucination. Finally, we implement and examine a series of widely used techniques to mitigate the hallucinations in LLMs. Our work has led to several important findings to understand the hallucination origin and mitigate the hallucinations in LLMs. Our code and data can be accessed at https://github.com/RUCAIBox/HaluEval-2.0.

{{</citation>}}


### (19/53) MPN: Leveraging Multilingual Patch Neuron for Cross-lingual Model Editing (Nianwen Si et al., 2024)

{{<citation>}}

Nianwen Si, Hao Zhang, Weiqiang Zhang. (2024)  
**MPN: Leveraging Multilingual Patch Neuron for Cross-lingual Model Editing**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CV, cs.CL  
Keywords: Multilingual, NLI  
[Paper Link](http://arxiv.org/abs/2401.03190v1)  

---


**ABSTRACT**  
Large language models are known for encoding a vast amount of factual knowledge, but they often becomes outdated due to the ever-changing nature of external information. A promising solution to this challenge is the utilization of model editing methods to update the knowledge in an efficient manner. However, the majority of existing model editing techniques are limited to monolingual frameworks, thus failing to address the crucial issue of cross-lingual knowledge synchronization for multilingual models. To tackle this problem, we propose a simple yet effective method that trains multilingual patch neuron to store cross-lingual knowledge. It can be easily adapted to existing approaches to enhance their cross-lingual editing capabilities. To evaluate our method, we conduct experiments using both the XNLI dataset and a self-constructed XFEVER dataset. Experimental results demonstrate that our proposed method achieves improved performance in cross-lingual editing tasks without requiring excessive modifications to the original methodology, thereby showcasing its user-friendly characteristics. Codes will be released soon.

{{</citation>}}


### (20/53) δ-CAUSAL: Exploring Defeasibility in Causal Reasoning (Shaobo Cui et al., 2024)

{{<citation>}}

Shaobo Cui, Lazar Milikic, Yiyang Feng, Mete Ismayilzada, Debjit Paul, Antoine Bosselut, Boi Faltings. (2024)  
**δ-CAUSAL: Exploring Defeasibility in Causal Reasoning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Attention, Embedding, GPT, GPT-3.5, Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2401.03183v1)  

---


**ABSTRACT**  
Defeasibility in causal reasoning implies that the causal relationship between cause and effect can be strengthened or weakened. Namely, the causal strength between cause and effect should increase or decrease with the incorporation of strengthening arguments (supporters) or weakening arguments (defeaters), respectively. However, existing works ignore defeasibility in causal reasoning and fail to evaluate existing causal strength metrics in defeasible settings. In this work, we present {\delta}-CAUSAL, the first benchmark dataset for studying defeasibility in causal reasoning. {\delta}-CAUSAL includes around 11K events spanning ten domains, featuring defeasible causality pairs, i.e., cause-effect pairs accompanied by supporters and defeaters. We further show current causal strength metrics fail to reflect the change of causal strength with the incorporation of supporters or defeaters in {\delta}-CAUSAL. To this end, we propose CESAR (Causal Embedding aSsociation with Attention Rating), a metric that measures causal strength based on token-level causal relationships. CESAR achieves a significant 69.7% relative improvement over existing metrics, increasing from 47.2% to 80.1% in capturing the causal strength change brought by supporters and defeaters. We further demonstrate even Large Language Models (LLMs) like GPT-3.5 still lag 4.5 and 10.7 points behind humans in generating supporters and defeaters, emphasizing the challenge posed by {\delta}-CAUSAL.

{{</citation>}}


### (21/53) A Joint-Reasoning based Disease Q&A System (Prakash Chandra Sukhwal et al., 2024)

{{<citation>}}

Prakash Chandra Sukhwal, Vaibhav Rajan, Atreyi Kankanhalli. (2024)  
**A Joint-Reasoning based Disease Q&A System**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, QA, Reasoning  
[Paper Link](http://arxiv.org/abs/2401.03181v1)  

---


**ABSTRACT**  
Medical question answer (QA) assistants respond to lay users' health-related queries by synthesizing information from multiple sources using natural language processing and related techniques. They can serve as vital tools to alleviate issues of misinformation, information overload, and complexity of medical language, thus addressing lay users' information needs while reducing the burden on healthcare professionals. QA systems, the engines of such assistants, have typically used either language models (LMs) or knowledge graphs (KG), though the approaches could be complementary. LM-based QA systems excel at understanding complex questions and providing well-formed answers, but are prone to factual mistakes. KG-based QA systems, which represent facts well, are mostly limited to answering short-answer questions with pre-created templates. While a few studies have jointly used LM and KG approaches for text-based QA, this was done to answer multiple-choice questions. Extant QA systems also have limitations in terms of automation and performance. We address these challenges by designing a novel, automated disease QA system which effectively utilizes both LM and KG techniques through a joint-reasoning approach to answer disease-related questions appropriate for lay users. Our evaluation of the system using a range of quality metrics demonstrates its efficacy over benchmark systems, including the popular ChatGPT.

{{</citation>}}


### (22/53) Part-of-Speech Tagger for Bodo Language using Deep Learning approach (Dhrubajyoti Pathak et al., 2024)

{{<citation>}}

Dhrubajyoti Pathak, Sanjib Narzary, Sukumar Nandi, Bidisha Som. (2024)  
**Part-of-Speech Tagger for Bodo Language using Deep Learning approach**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: BERT, Embedding, LSTM, NLP  
[Paper Link](http://arxiv.org/abs/2401.03175v1)  

---


**ABSTRACT**  
Language Processing systems such as Part-of-speech tagging, Named entity recognition, Machine translation, Speech recognition, and Language modeling (LM) are well-studied in high-resource languages. Nevertheless, research on these systems for several low-resource languages, including Bodo, Mizo, Nagamese, and others, is either yet to commence or is in its nascent stages. Language model plays a vital role in the downstream tasks of modern NLP. Extensive studies are carried out on LMs for high-resource languages. Nevertheless, languages such as Bodo, Rabha, and Mising continue to lack coverage. In this study, we first present BodoBERT, a language model for the Bodo language. To the best of our knowledge, this work is the first such effort to develop a language model for Bodo. Secondly, we present an ensemble DL-based POS tagging model for Bodo. The POS tagging model is based on combinations of BiLSTM with CRF and stacked embedding of BodoBERT with BytePairEmbeddings. We cover several language models in the experiment to see how well they work in POS tagging tasks. The best-performing model achieves an F1 score of 0.8041. A comparative experiment was also conducted on Assamese POS taggers, considering that the language is spoken in the same region as Bodo.

{{</citation>}}


### (23/53) Quartet Logic: A Four-Step Reasoning (QLFR) framework for advancing Short Text Classification (Hui Wu et al., 2024)

{{<citation>}}

Hui Wu, Yuanben Zhang, Zhonghe Han, Yingyan Hou, Lei Wang, Siye Liu, Qihang Gong, Yunping Ge. (2024)  
**Quartet Logic: A Four-Step Reasoning (QLFR) framework for advancing Short Text Classification**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Graph Convolutional Network, Language Model, NLP, Reasoning, Text Classification  
[Paper Link](http://arxiv.org/abs/2401.03158v1)  

---


**ABSTRACT**  
Short Text Classification (STC) is crucial for processing and comprehending the brief but substantial content prevalent on contemporary digital platforms. The STC encounters difficulties in grasping semantic and syntactic intricacies, an issue that is apparent in traditional pre-trained language models. Although Graph Convolutional Networks enhance performance by integrating external knowledge bases, these methods are limited by the quality and extent of the knowledge applied. Recently, the emergence of Large Language Models (LLMs) and Chain-of-Thought (CoT) has significantly improved the performance of complex reasoning tasks. However, some studies have highlighted the limitations of their application in fundamental NLP tasks. Consequently, this study sought to employ CoT to investigate the capabilities of LLMs in STC tasks. This study introduces Quartet Logic: A Four-Step Reasoning (QLFR) framework. This framework primarily incorporates Syntactic and Semantic Enrichment CoT, effectively decomposing the STC task into four distinct steps: (i) essential concept identification, (ii) common-sense knowledge retrieval, (iii) text rewriting, and (iv) classification. This elicits the inherent knowledge and abilities of LLMs to address the challenges in STC. Surprisingly, we found that QLFR can also improve the performance of smaller models. Therefore, we developed a CoT-Driven Multi-task learning (QLFR-CML) method to facilitate the knowledge transfer from LLMs to smaller models. Extensive experimentation across six short-text benchmarks validated the efficacy of the proposed methods. Notably, QLFR achieved state-of-the-art performance on all datasets, with significant improvements, particularly on the Ohsumed and TagMyNews datasets.

{{</citation>}}


### (24/53) Examining Forgetting in Continual Pre-training of Aligned Large Language Models (Chen-An Li et al., 2024)

{{<citation>}}

Chen-An Li, Hung-Yi Lee. (2024)  
**Examining Forgetting in Continual Pre-training of Aligned Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.03129v1)  

---


**ABSTRACT**  
Recent advances in Large Language Models (LLMs) have exhibited remarkable proficiency across various tasks. Given the potent applications of LLMs in numerous fields, there has been a surge in LLM development. In developing LLMs, a common practice involves continual pre-training on previously fine-tuned models. However, this can lead to catastrophic forgetting. In our work, we investigate the phenomenon of forgetting that occurs during continual pre-training on an existing fine-tuned LLM. We evaluate the impact of continuous pre-training on the fine-tuned LLM across various dimensions, including output format, knowledge, and reliability. Experiment results highlight the non-trivial challenge of addressing catastrophic forgetting during continual pre-training, especially the repetition issue.

{{</citation>}}


## cs.CR (2)



### (25/53) Malla: Demystifying Real-world Large Language Model Integrated Malicious Services (Zilong Lin et al., 2024)

{{<citation>}}

Zilong Lin, Jian Cui, Xiaojing Liao, XiaoFeng Wang. (2024)  
**Malla: Demystifying Real-world Large Language Model Integrated Malicious Services**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.03315v1)  

---


**ABSTRACT**  
The underground exploitation of large language models (LLMs) for malicious services (i.e., Malla) is witnessing an uptick, amplifying the cyber threat landscape and posing questions about the trustworthiness of LLM technologies. However, there has been little effort to understand this new cybercrime, in terms of its magnitude, impact, and techniques. In this paper, we conduct the first systematic study on 212 real-world Mallas, uncovering their proliferation in underground marketplaces and exposing their operational modalities. Our study discloses the Malla ecosystem, revealing its significant growth and impact on today's public LLM services. Through examining 212 Mallas, we uncovered eight backend LLMs used by Mallas, along with 182 prompts that circumvent the protective measures of public LLM APIs. We further demystify the tactics employed by Mallas, including the abuse of uncensored LLMs and the exploitation of public LLM APIs through jailbreak prompts. Our findings enable a better understanding of the real-world exploitation of LLMs by cybercriminals, offering insights into strategies to counteract this cybercrime.

{{</citation>}}


### (26/53) SecureReg: A Combined Framework for Proactively Exposing Malicious Domain Name Registrations (Furkan Çolhak et al., 2024)

{{<citation>}}

Furkan Çolhak, Mert İlhan Ecevit, Hasan Dağ, Reiner Creutzburg. (2024)  
**SecureReg: A Combined Framework for Proactively Exposing Malicious Domain Name Registrations**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2401.03196v1)  

---


**ABSTRACT**  
Rising cyber threats, with miscreants registering thousands of new domains daily for Internet-scale attacks like spam, phishing, and drive-by downloads, emphasize the need for innovative detection methods. This paper introduces a cutting-edge approach for identifying suspicious domains at the onset of the registration process. The accompanying data pipeline generates crucial features by comparing new domains to registered domains,emphasizing the crucial similarity score. Leveraging a novel combination of Natural Language Processing (NLP) techniques, including a pretrained Canine model, and Multilayer Perceptron (MLP) models, our system analyzes semantic and numerical attributes, providing a robust solution for early threat detection. This integrated approach significantly reduces the window of vulnerability, fortifying defenses against potential threats. The findings demonstrate the effectiveness of the integrated approach and contribute to the ongoing efforts in developing proactive strategies to mitigate the risks associated with illicit online activities through the early identification of suspicious domain registrations.

{{</citation>}}


## cs.CV (10)



### (27/53) Exploiting Data Hierarchy as a New Modality for Contrastive Learning (Arjun Bhalla et al., 2024)

{{<citation>}}

Arjun Bhalla, Daniel Levenson, Jan Bernhard, Anton Abilov. (2024)  
**Exploiting Data Hierarchy as a New Modality for Contrastive Learning**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2401.03312v1)  

---


**ABSTRACT**  
This work investigates how hierarchically structured data can help neural networks learn conceptual representations of cathedrals. The underlying WikiScenes dataset provides a spatially organized hierarchical structure of cathedral components. We propose a novel hierarchical contrastive training approach that leverages a triplet margin loss to represent the data's spatial hierarchy in the encoder's latent space. As such, the proposed approach investigates if the dataset structure provides valuable information for self-supervised learning. We apply t-SNE to visualize the resultant latent space and evaluate the proposed approach by comparing it with other dataset-specific contrastive learning methods using a common downstream classification task. The proposed method outperforms the comparable weakly-supervised and baseline methods. Our findings suggest that dataset structure is a valuable modality for weakly-supervised learning.

{{</citation>}}


### (28/53) Large Language Models as Visual Cross-Domain Learners (Shuhao Chen et al., 2024)

{{<citation>}}

Shuhao Chen, Yulong Zhang, Weisen Jiang, Jiangang Lu, Yu Zhang. (2024)  
**Large Language Models as Visual Cross-Domain Learners**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs-LG, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.03253v1)  

---


**ABSTRACT**  
Recent advances achieved by deep learning models rely on the independent and identically distributed assumption, hindering their applications in real-world scenarios with domain shifts. To address the above issues, cross-domain learning aims at extracting domain-invariant knowledge to reduce the domain shift between training and testing data. However, in visual cross-domain learning, traditional methods concentrate solely on the image modality, neglecting the use of the text modality to alleviate the domain shift. In this work, we propose Large Language models as Visual cross-dOmain learners (LLaVO). LLaVO uses vision-language models to convert images into detailed textual descriptions. A large language model is then finetuned on textual descriptions of the source/target domain generated by a designed instruction template. Extensive experimental results on various cross-domain tasks under the domain generalization and unsupervised domain adaptation settings have demonstrated the effectiveness of the proposed method.

{{</citation>}}


### (29/53) 3DMIT: 3D Multi-modal Instruction Tuning for Scene Understanding (Zeju Li et al., 2024)

{{<citation>}}

Zeju Li, Chao Zhang, Xiaoyan Wang, Ruilong Ren, Yifan Xu, Ruifei Ma, Xiangde Liu. (2024)  
**3DMIT: 3D Multi-modal Instruction Tuning for Scene Understanding**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2401.03201v1)  

---


**ABSTRACT**  
The remarkable potential of multi-modal large language models (MLLMs) in comprehending both vision and language information has been widely acknowledged. However, the scarcity of 3D scenes-language pairs in comparison to their 2D counterparts, coupled with the inadequacy of existing approaches in understanding of 3D scenes by LLMs, poses a significant challenge. In response, we collect and construct an extensive dataset comprising 75K instruction-response pairs tailored for 3D scenes. This dataset addresses tasks related to 3D VQA, 3D grounding, and 3D conversation. To further enhance the integration of 3D spatial information into LLMs, we introduce a novel and efficient prompt tuning paradigm, 3DMIT. This paradigm eliminates the alignment stage between 3D scenes and language and extends the instruction prompt with the 3D modality information including the entire scene and segmented objects. We evaluate the effectiveness of our method across diverse tasks in the 3D scene domain and find that our approach serves as a strategic means to enrich LLMs' comprehension of the 3D world. Our code is available at https://github.com/staymylove/3DMIT.

{{</citation>}}


### (30/53) Distribution-aware Interactive Attention Network and Large-scale Cloud Recognition Benchmark on FY-4A Satellite Image (Jiaqing Zhang et al., 2024)

{{<citation>}}

Jiaqing Zhang, Jie Lei, Weiying Xie, Kai Jiang, Mingxiang Cao, Yunsong Li. (2024)  
**Distribution-aware Interactive Attention Network and Large-scale Cloud Recognition Benchmark on FY-4A Satellite Image**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2401.03182v1)  

---


**ABSTRACT**  
Accurate cloud recognition and warning are crucial for various applications, including in-flight support, weather forecasting, and climate research. However, recent deep learning algorithms have predominantly focused on detecting cloud regions in satellite imagery, with insufficient attention to the specificity required for accurate cloud recognition. This limitation inspired us to develop the novel FY-4A-Himawari-8 (FYH) dataset, which includes nine distinct cloud categories and uses precise domain adaptation methods to align 70,419 image-label pairs in terms of projection, temporal resolution, and spatial resolution, thereby facilitating the training of supervised deep learning networks. Given the complexity and diversity of cloud formations, we have thoroughly analyzed the challenges inherent to cloud recognition tasks, examining the intricate characteristics and distribution of the data. To effectively address these challenges, we designed a Distribution-aware Interactive-Attention Network (DIAnet), which preserves pixel-level details through a high-resolution branch and a parallel multi-resolution cross-branch. We also integrated a distribution-aware loss (DAL) to mitigate the imbalance across cloud categories. An Interactive Attention Module (IAM) further enhances the robustness of feature extraction combined with spatial and channel information. Empirical evaluations on the FYH dataset demonstrate that our method outperforms other cloud recognition networks, achieving superior performance in terms of mean Intersection over Union (mIoU). The code for implementing DIAnet is available at https://github.com/icey-zhang/DIAnet.

{{</citation>}}


### (31/53) Multimodal Informative ViT: Information Aggregation and Distribution for Hyperspectral and LiDAR Classification (Jiaqing Zhang et al., 2024)

{{<citation>}}

Jiaqing Zhang, Jie Lei, Weiying Xie, Geng Yang, Daixun Li, Yunsong Li, Karim Seghouane. (2024)  
**Multimodal Informative ViT: Information Aggregation and Distribution for Hyperspectral and LiDAR Classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.03179v1)  

---


**ABSTRACT**  
In multimodal land cover classification (MLCC), a common challenge is the redundancy in data distribution, where irrelevant information from multiple modalities can hinder the effective integration of their unique features. To tackle this, we introduce the Multimodal Informative Vit (MIVit), a system with an innovative information aggregate-distributing mechanism. This approach redefines redundancy levels and integrates performance-aware elements into the fused representation, facilitating the learning of semantics in both forward and backward directions. MIVit stands out by significantly reducing redundancy in the empirical distribution of each modality's separate and fused features. It employs oriented attention fusion (OAF) for extracting shallow local features across modalities in horizontal and vertical dimensions, and a Transformer feature extractor for extracting deep global features through long-range attention. We also propose an information aggregation constraint (IAC) based on mutual information, designed to remove redundant information and preserve complementary information within embedded features. Additionally, the information distribution flow (IDF) in MIVit enhances performance-awareness by distributing global classification information across different modalities' feature maps. This architecture also addresses missing modality challenges with lightweight independent modality classifiers, reducing the computational load typically associated with Transformers. Our results show that MIVit's bidirectional aggregate-distributing mechanism between modalities is highly effective, achieving an average overall accuracy of 95.56% across three multimodal datasets. This performance surpasses current state-of-the-art methods in MLCC. The code for MIVit is accessible at https://github.com/icey-zhang/MIViT.

{{</citation>}}


### (32/53) PosDiffNet: Positional Neural Diffusion for Point Cloud Registration in a Large Field of View with Perturbations (Rui She et al., 2024)

{{<citation>}}

Rui She, Sijie Wang, Qiyu Kang, Kai Zhao, Yang Song, Wee Peng Tay, Tianyu Geng, Xingchao Jian. (2024)  
**PosDiffNet: Positional Neural Diffusion for Point Cloud Registration in a Large Field of View with Perturbations**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI, Transformer  
[Paper Link](http://arxiv.org/abs/2401.03167v1)  

---


**ABSTRACT**  
Point cloud registration is a crucial technique in 3D computer vision with a wide range of applications. However, this task can be challenging, particularly in large fields of view with dynamic objects, environmental noise, or other perturbations. To address this challenge, we propose a model called PosDiffNet. Our approach performs hierarchical registration based on window-level, patch-level, and point-level correspondence. We leverage a graph neural partial differential equation (PDE) based on Beltrami flow to obtain high-dimensional features and position embeddings for point clouds. We incorporate position embeddings into a Transformer module based on a neural ordinary differential equation (ODE) to efficiently represent patches within points. We employ the multi-level correspondence derived from the high feature similarity scores to facilitate alignment between point clouds. Subsequently, we use registration methods such as SVD-based algorithms to predict the transformation using corresponding point pairs. We evaluate PosDiffNet on several 3D point cloud datasets, verifying that it achieves state-of-the-art (SOTA) performance for point cloud registration in large fields of view with perturbations. The implementation code of experiments is available at https://github.com/AI-IT-AVs/PosDiffNet.

{{</citation>}}


### (33/53) Controllable Image Synthesis of Industrial Data Using Stable Diffusion (Gabriele Valvano et al., 2024)

{{<citation>}}

Gabriele Valvano, Antonino Agostino, Giovanni De Magistris, Antonino Graziano, Giacomo Veneri. (2024)  
**Controllable Image Synthesis of Industrial Data Using Stable Diffusion**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2401.03152v1)  

---


**ABSTRACT**  
Training supervised deep neural networks that perform defect detection and segmentation requires large-scale fully-annotated datasets, which can be hard or even impossible to obtain in industrial environments. Generative AI offers opportunities to enlarge small industrial datasets artificially, thus enabling the usage of state-of-the-art supervised approaches in the industry. Unfortunately, also good generative models need a lot of data to train, while industrial datasets are often tiny. Here, we propose a new approach for reusing general-purpose pre-trained generative models on industrial data, ultimately allowing the generation of self-labelled defective images. First, we let the model learn the new concept, entailing the novel data distribution. Then, we force it to learn to condition the generative process, producing industrial images that satisfy well-defined topological characteristics and show defects with a given geometry and location. To highlight the advantage of our approach, we use the synthetic dataset to optimise a crack segmentor for a real industrial use case. When the available data is small, we observe considerable performance increase under several metrics, showing the method's potential in production environments.

{{</citation>}}


### (34/53) Self-supervised Feature Adaptation for 3D Industrial Anomaly Detection (Yuanpeng Tu et al., 2024)

{{<citation>}}

Yuanpeng Tu, Boshen Zhang, Liang Liu, Yuxi Li, Chenhai Xu, Jiangning Zhang, Yabiao Wang, Chengjie Wang, Cai Rong Zhao. (2024)  
**Self-supervised Feature Adaptation for 3D Industrial Anomaly Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Anomaly Detection, ImageNet  
[Paper Link](http://arxiv.org/abs/2401.03145v1)  

---


**ABSTRACT**  
Industrial anomaly detection is generally addressed as an unsupervised task that aims at locating defects with only normal training samples. Recently, numerous 2D anomaly detection methods have been proposed and have achieved promising results, however, using only the 2D RGB data as input is not sufficient to identify imperceptible geometric surface anomalies. Hence, in this work, we focus on multi-modal anomaly detection. Specifically, we investigate early multi-modal approaches that attempted to utilize models pre-trained on large-scale visual datasets, i.e., ImageNet, to construct feature databases. And we empirically find that directly using these pre-trained models is not optimal, it can either fail to detect subtle defects or mistake abnormal features as normal ones. This may be attributed to the domain gap between target industrial data and source data.Towards this problem, we propose a Local-to-global Self-supervised Feature Adaptation (LSFA) method to finetune the adaptors and learn task-oriented representation toward anomaly detection.Both intra-modal adaptation and cross-modal alignment are optimized from a local-to-global perspective in LSFA to ensure the representation quality and consistency in the inference stage.Extensive experiments demonstrate that our method not only brings a significant performance boost to feature embedding based approaches, but also outperforms previous State-of-The-Art (SoTA) methods prominently on both MVTec-3D AD and Eyecandies datasets, e.g., LSFA achieves 97.1% I-AUROC on MVTec-3D, surpass previous SoTA by +3.4%.

{{</citation>}}


### (35/53) Dress-Me-Up: A Dataset & Method for Self-Supervised 3D Garment Retargeting (Shanthika Naik et al., 2024)

{{<citation>}}

Shanthika Naik, Kunwar Singh, Astitva Srivastava, Dhawal Sirikonda, Amit Raj, Varun Jampani, Avinash Sharma. (2024)  
**Dress-Me-Up: A Dataset & Method for Self-Supervised 3D Garment Retargeting**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Embedding, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2401.03108v1)  

---


**ABSTRACT**  
We propose a novel self-supervised framework for retargeting non-parameterized 3D garments onto 3D human avatars of arbitrary shapes and poses, enabling 3D virtual try-on (VTON). Existing self-supervised 3D retargeting methods only support parametric and canonical garments, which can only be draped over parametric body, e.g. SMPL. To facilitate the non-parametric garments and body, we propose a novel method that introduces Isomap Embedding based correspondences matching between the garment and the human body to get a coarse alignment between the two meshes. We perform neural refinement of the coarse alignment in a self-supervised setting. Further, we leverage a Laplacian detail integration method for preserving the inherent details of the input garment. For evaluating our 3D non-parametric garment retargeting framework, we propose a dataset of 255 real-world garments with realistic noise and topological deformations. The dataset contains $44$ unique garments worn by 15 different subjects in 5 distinctive poses, captured using a multi-view RGBD capture setup. We show superior retargeting quality on non-parametric garments and human avatars over existing state-of-the-art methods, acting as the first-ever baseline on the proposed dataset for non-parametric 3D garment retargeting.

{{</citation>}}


### (36/53) Incorporating Visual Experts to Resolve the Information Loss in Multimodal Large Language Models (Xin He et al., 2024)

{{<citation>}}

Xin He, Longhui Wei, Lingxi Xie, Qi Tian. (2024)  
**Incorporating Visual Experts to Resolve the Information Loss in Multimodal Large Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.03105v1)  

---


**ABSTRACT**  
Multimodal Large Language Models (MLLMs) are experiencing rapid growth, yielding a plethora of noteworthy contributions in recent months. The prevailing trend involves adopting data-driven methodologies, wherein diverse instruction-following datasets are collected. However, a prevailing challenge persists in these approaches, specifically in relation to the limited visual perception ability, as CLIP-like encoders employed for extracting visual information from inputs. Though these encoders are pre-trained on billions of image-text pairs, they still grapple with the information loss dilemma, given that textual captions only partially capture the contents depicted in images. To address this limitation, this paper proposes to improve the visual perception ability of MLLMs through a mixture-of-experts knowledge enhancement mechanism. Specifically, we introduce a novel method that incorporates multi-task encoders and visual tools into the existing MLLMs training and inference pipeline, aiming to provide a more comprehensive and accurate summarization of visual inputs. Extensive experiments have evaluated its effectiveness of advancing MLLMs, showcasing improved visual perception achieved through the integration of visual experts.

{{</citation>}}


## cs.NI (1)



### (37/53) CAVIAR: Co-simulation of 6G Communications, 3D Scenarios and AI for Digital Twins (João Borges et al., 2024)

{{<citation>}}

João Borges, Felipe Bastos, Ilan Correa, Pedro Batista, Aldebaro Klautau. (2024)  
**CAVIAR: Co-simulation of 6G Communications, 3D Scenarios and AI for Digital Twins**  

---
Primary Category: cs.NI  
Categories: cs-AI, cs-NI, cs.NI, eess-SP  
Keywords: AI, Yolo  
[Paper Link](http://arxiv.org/abs/2401.03310v1)  

---


**ABSTRACT**  
Digital twins are an important technology for advancing mobile communications, specially in use cases that require simultaneously simulating the wireless channel, 3D scenes and machine learning. Aiming at providing a solution to this demand, this work describes a modular co-simulation methodology called CAVIAR. Here, CAVIAR is upgraded to support a message passing library and enable the virtual counterpart of a digital twin system using different 6G-related simulators. The main contributions of this work are the detailed description of different CAVIAR architectures, the implementation of this methodology to assess a 6G use case of UAV-based search and rescue mission (SAR), and the generation of benchmarking data about the computational resource usage. For executing the SAR co-simulation we adopt five open-source solutions: the physical and link level network simulator Sionna, the simulator for autonomous vehicles AirSim, scikit-learn for training a decision tree for MIMO beam selection, Yolov8 for the detection of rescue targets and NATS for message passing. Results for the implemented SAR use case suggest that the methodology can run in a single machine, with the main demanded resources being the CPU processing and the GPU memory.

{{</citation>}}


## eess.IV (2)



### (38/53) Realism in Action: Anomaly-Aware Diagnosis of Brain Tumors from Medical Images Using YOLOv8 and DeiT (Seyed Mohammad Hossein Hashemi et al., 2024)

{{<citation>}}

Seyed Mohammad Hossein Hashemi, Leila Safari, Amirhossein Dadashzade Taromi. (2024)  
**Realism in Action: Anomaly-Aware Diagnosis of Brain Tumors from Medical Images Using YOLOv8 and DeiT**  

---
Primary Category: eess.IV  
Categories: cs-AI, cs-CV, cs-LG, eess-IV, eess.IV, stat-ML  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.03302v2)  

---


**ABSTRACT**  
In the field of medical sciences, reliable detection and classification of brain tumors from images remains a formidable challenge due to the rarity of tumors within the population of patients. Therefore, the ability to detect tumors in anomaly scenarios is paramount for ensuring timely interventions and improved patient outcomes. This study addresses the issue by leveraging deep learning (DL) techniques to detect and classify brain tumors in challenging situations. The curated data set from the National Brain Mapping Lab (NBML) comprises 81 patients, including 30 Tumor cases and 51 Normal cases. The detection and classification pipelines are separated into two consecutive tasks. The detection phase involved comprehensive data analysis and pre-processing to modify the number of image samples and the number of patients of each class to anomaly distribution (9 Normal per 1 Tumor) to comply with real world scenarios. Next, in addition to common evaluation metrics for the testing, we employed a novel performance evaluation method called Patient to Patient (PTP), focusing on the realistic evaluation of the model. In the detection phase, we fine-tuned a YOLOv8n detection model to detect the tumor region. Subsequent testing and evaluation yielded competitive performance both in Common Evaluation Metrics and PTP metrics. Furthermore, using the Data Efficient Image Transformer (DeiT) module, we distilled a Vision Transformer (ViT) model from a fine-tuned ResNet152 as a teacher in the classification phase. This approach demonstrates promising strides in reliable tumor detection and classification, offering potential advancements in tumor diagnosis for real-world medical imaging scenarios.

{{</citation>}}


### (39/53) Vision Transformers and Bi-LSTM for Alzheimer's Disease Diagnosis from 3D MRI (Taymaz Akan et al., 2024)

{{<citation>}}

Taymaz Akan, Sait Alp, Mohammad A. N Bhuiyanb. (2024)  
**Vision Transformers and Bi-LSTM for Alzheimer's Disease Diagnosis from 3D MRI**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: LSTM, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.03132v1)  

---


**ABSTRACT**  
Alzheimer's is a brain disease that gets worse over time and affects memory, thinking, and behavior. Alzheimer's disease (AD) can be treated and managed if it is diagnosed early, which can slow the progression of symptoms and improve quality of life. In this study, we suggested using the Visual Transformer (ViT) and bi-LSTM to process MRI images for diagnosing Alzheimer's disease. We used ViT to extract features from the MRI and then map them to a feature sequence. Then, we used Bi-LSTM sequence modeling to keep the interdependencies between related features. In addition, we evaluated the performance of the proposed model for the binary classification of AD patients using data from the Alzheimer's Disease Neuroimaging Initiative (ADNI). Finally, we evaluated our method against other deep learning models in the literature. The proposed method performs well in terms of accuracy, precision, F-score, and recall for the diagnosis of AD.

{{</citation>}}


## eess.AS (1)



### (40/53) TeLeS: Temporal Lexeme Similarity Score to Estimate Confidence in End-to-End ASR (Nagarathna Ravi et al., 2024)

{{<citation>}}

Nagarathna Ravi, Thishyan Raj T, Vipul Arora. (2024)  
**TeLeS: Temporal Lexeme Similarity Score to Estimate Confidence in End-to-End ASR**  

---
Primary Category: eess.AS  
Categories: cs-LG, cs-SD, eess-AS, eess.AS, stat-ML  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2401.03251v1)  

---


**ABSTRACT**  
Confidence estimation of predictions from an End-to-End (E2E) Automatic Speech Recognition (ASR) model benefits ASR's downstream and upstream tasks. Class-probability-based confidence scores do not accurately represent the quality of overconfident ASR predictions. An ancillary Confidence Estimation Model (CEM) calibrates the predictions. State-of-the-art (SOTA) solutions use binary target scores for CEM training. However, the binary labels do not reveal the granular information of predicted words, such as temporal alignment between reference and hypothesis and whether the predicted word is entirely incorrect or contains spelling errors. Addressing this issue, we propose a novel Temporal-Lexeme Similarity (TeLeS) confidence score to train CEM. To address the data imbalance of target scores while training CEM, we use shrinkage loss to focus on hard-to-learn data points and minimise the impact of easily learned data points. We conduct experiments with ASR models trained in three languages, namely Hindi, Tamil, and Kannada, with varying training data sizes. Experiments show that TeLeS generalises well across domains. To demonstrate the applicability of the proposed method, we formulate a TeLeS-based Acquisition (TeLeS-A) function for sampling uncertainty in active learning. We observe a significant reduction in the Word Error Rate (WER) as compared to SOTA methods.

{{</citation>}}


## math.OC (1)



### (41/53) Artificial Intelligence for Operations Research: Revolutionizing the Operations Research Process (Zhenan Fan et al., 2024)

{{<citation>}}

Zhenan Fan, Bissan Ghaddar, Xinglu Wang, Linzi Xing, Yong Zhang, Zirui Zhou. (2024)  
**Artificial Intelligence for Operations Research: Revolutionizing the Operations Research Process**  

---
Primary Category: math.OC  
Categories: cs-AI, math-OC, math.OC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.03244v1)  

---


**ABSTRACT**  
The rapid advancement of artificial intelligence (AI) techniques has opened up new opportunities to revolutionize various fields, including operations research (OR). This survey paper explores the integration of AI within the OR process (AI4OR) to enhance its effectiveness and efficiency across multiple stages, such as parameter generation, model formulation, and model optimization. By providing a comprehensive overview of the state-of-the-art and examining the potential of AI to transform OR, this paper aims to inspire further research and innovation in the development of AI-enhanced OR methods and tools. The synergy between AI and OR is poised to drive significant advancements and novel solutions in a multitude of domains, ultimately leading to more effective and efficient decision-making.

{{</citation>}}


## cs.HC (2)



### (42/53) Using Large Language Models to Assess Tutors' Performance in Reacting to Students Making Math Errors (Sanjit Kakarla et al., 2024)

{{<citation>}}

Sanjit Kakarla, Danielle Thomas, Jionghao Lin, Shivang Gupta, Kenneth R. Koedinger. (2024)  
**Using Large Language Models to Assess Tutors' Performance in Reacting to Students Making Math Errors**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-CY, cs-HC, cs.HC  
Keywords: AI, GPT, GPT-3.5, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2401.03238v1)  

---


**ABSTRACT**  
Research suggests that tutors should adopt a strategic approach when addressing math errors made by low-efficacy students. Rather than drawing direct attention to the error, tutors should guide the students to identify and correct their mistakes on their own. While tutor lessons have introduced this pedagogical skill, human evaluation of tutors applying this strategy is arduous and time-consuming. Large language models (LLMs) show promise in providing real-time assessment to tutors during their actual tutoring sessions, yet little is known regarding their accuracy in this context. In this study, we investigate the capacity of generative AI to evaluate real-life tutors' performance in responding to students making math errors. By analyzing 50 real-life tutoring dialogues, we find both GPT-3.5-Turbo and GPT-4 demonstrate proficiency in assessing the criteria related to reacting to students making errors. However, both models exhibit limitations in recognizing instances where the student made an error. Notably, GPT-4 tends to overidentify instances of students making errors, often attributing student uncertainty or inferring potential errors where human evaluators did not. Future work will focus on enhancing generalizability by assessing a larger dataset of dialogues and evaluating learning transfer. Specifically, we will analyze the performance of tutors in real-life scenarios when responding to students' math errors before and after lesson completion on this crucial tutoring skill.

{{</citation>}}


### (43/53) An intelligent sociotechnical systems (iSTS) framework: Toward a sociotechnically-based hierarchical human-centered AI approach (Wei Xu et al., 2024)

{{<citation>}}

Wei Xu, Zaifeng Gao. (2024)  
**An intelligent sociotechnical systems (iSTS) framework: Toward a sociotechnically-based hierarchical human-centered AI approach**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.03223v1)  

---


**ABSTRACT**  
Insights: - The human-centered AI (HCAI) approach and the sociotechnical systems (STS) theory share the same goal: ensuring that new technologies such as AI best serve humans in a sociotechnical environment. - HCAI practice needs to fully embrace sociotechnical systems thinking, while traditional STS needs to evolve to address the emerging characteristics of AI technology. - We propose a conceptual framework for intelligent sociotechnical systems (iSTS) to enhance traditional STS theory in the AI era. - Based on iSTS, we further propose a sociotechnical-based hierarchical HCAI approach as a paradigmatic extension to existing HCAI practice, further advancing HCAI practice.

{{</citation>}}


## cs.DC (1)



### (44/53) RAID Organizations for Improved Reliability and Performance: A Not Entirely Unbiased Tutorial (1st revision) (Alexander Thomasian, 2024)

{{<citation>}}

Alexander Thomasian. (2024)  
**RAID Organizations for Improved Reliability and Performance: A Not Entirely Unbiased Tutorial (1st revision)**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs-OS, cs-PF, cs.DC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.03235v1)  

---


**ABSTRACT**  
RAID proposal advocated replacing large disks with arrays of PC disks, but as the capacity of small disks increased 100-fold in 1990s the production of large disks was discontinued. Storage dependability is increased via replication or erasure coding. Cloud storage providers store multiple copies of data obviating for need for further redundancy. Varitaions of RAID based on local recovery codes, partial MDS reduce recovery cost. NAND flash Solid State Disks - SSDs have low latency and high bandwidth, are more reliable, consume less power and have a lower TCO than Hard Disk Drives, which are more viable for hyperscalers.

{{</citation>}}


## cs.CG (1)



### (45/53) On 1-bend Upward Point-set Embeddings of $st$-digraphs (Emilio Di Giacomo et al., 2024)

{{<citation>}}

Emilio Di Giacomo, Henry Förster, Daria Kokhovich, Tamara Mchedlidze, Fabrizio Montecchiani, Antonios Symvonis, Anaïs Villedieu. (2024)  
**On 1-bend Upward Point-set Embeddings of $st$-digraphs**  

---
Primary Category: cs.CG  
Categories: cs-CG, cs.CG  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2401.03226v1)  

---


**ABSTRACT**  
We study the upward point-set embeddability of digraphs on one-sided convex point sets with at most 1 bend per edge. We provide an algorithm to compute a 1-bend upward point-set embedding of outerplanar $st$-digraphs on arbitrary one-sided convex point sets. We complement this result by proving that for every $n \geq 18$ there exists a $2$-outerplanar $st$-digraph $G$ with $n$ vertices and a one-sided convex point set $S$ so that $G$ does not admit a 1-bend upward point-set embedding on $S$.

{{</citation>}}


## cs.RO (2)



### (46/53) Understanding Large-Language Model (LLM)-powered Human-Robot Interaction (Callie Y. Kim et al., 2024)

{{<citation>}}

Callie Y. Kim, Christine P. Lee, Bilge Mutlu. (2024)  
**Understanding Large-Language Model (LLM)-powered Human-Robot Interaction**  

---
Primary Category: cs.RO  
Categories: cs-HC, cs-RO, cs.RO  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.03217v1)  

---


**ABSTRACT**  
Large-language models (LLMs) hold significant promise in improving human-robot interaction, offering advanced conversational skills and versatility in managing diverse, open-ended user requests in various tasks and domains. Despite the potential to transform human-robot interaction, very little is known about the distinctive design requirements for utilizing LLMs in robots, which may differ from text and voice interaction and vary by task and context. To better understand these requirements, we conducted a user study (n = 32) comparing an LLM-powered social robot against text- and voice-based agents, analyzing task-based requirements in conversational tasks, including choose, generate, execute, and negotiate. Our findings show that LLM-powered robots elevate expectations for sophisticated non-verbal cues and excel in connection-building and deliberation, but fall short in logical communication and may induce anxiety. We provide design implications both for robots integrating LLMs and for fine-tuning LLMs for use with robots.

{{</citation>}}


### (47/53) Estimating the Lateral Motion States of an Underwater Robot by Propeller Wake Sensing Using an Artificial Lateral Line (Jun Wang et al., 2024)

{{<citation>}}

Jun Wang, Dexin Zhao, Youxi Zhao, Feitian Zhang, Tongsheng Shen. (2024)  
**Estimating the Lateral Motion States of an Underwater Robot by Propeller Wake Sensing Using an Artificial Lateral Line**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2401.03141v1)  

---


**ABSTRACT**  
An artificial lateral line (ALL) is a bioinspired flow sensing system of an underwater robot that consists of distributed flow sensors. The ALL has achieved great success in sensing the motion states of bioinspired underwater robots, e.g., robotic fish, that are driven by body undulation and/or tail flapping. However, the ALL has not been systematically tested and studied in the sensing of underwater robots driven by rotating propellers due to the highly dynamic and complex flow field therein. This paper makes a bold hypothesis that the distributed flow measurements sampled from the propeller wake flow, although infeasible to represent the entire flow dynamics, provides sufficient information for estimating the lateral motion states of the leader underwater robot. An experimental testbed is constructed to investigate the feasibility of such a state estimator which comprises a cylindrical ALL sensory system, a rotating leader propeller, and a water tank with a planar sliding guide. Specifically, a hybrid network that consists of a one-dimensional convolution network (1DCNN) and a bidirectional long short-term memory network (BiLSTM) is designed to extract the spatiotemporal features of the time series of distributed pressure measurements. A multi-output deep learning network is adopted to estimate the lateral motion states of the leader propeller. In addition, the state estimator is optimized using the whale optimization algorithm (WOA) considering the comprehensive estimation performance. Extensive experiments are conducted the results of which validate the proposed data-driven algorithm in estimating the motion states of the leader underwater robot by propeller wake sensing.

{{</citation>}}


## cs.AI (3)



### (48/53) Decision Making in Non-Stationary Environments with Policy-Augmented Search (Ava Pettet et al., 2024)

{{<citation>}}

Ava Pettet, Yunuo Zhang, Baiting Luo, Kyle Wray, Hendrik Baier, Aron Laszka, Abhishek Dubey, Ayan Mukhopadhyay. (2024)  
**Decision Making in Non-Stationary Environments with Policy-Augmented Search**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.03197v1)  

---


**ABSTRACT**  
Sequential decision-making under uncertainty is present in many important problems. Two popular approaches for tackling such problems are reinforcement learning and online search (e.g., Monte Carlo tree search). While the former learns a policy by interacting with the environment (typically done before execution), the latter uses a generative model of the environment to sample promising action trajectories at decision time. Decision-making is particularly challenging in non-stationary environments, where the environment in which an agent operates can change over time. Both approaches have shortcomings in such settings -- on the one hand, policies learned before execution become stale when the environment changes and relearning takes both time and computational effort. Online search, on the other hand, can return sub-optimal actions when there are limitations on allowed runtime. In this paper, we introduce \textit{Policy-Augmented Monte Carlo tree search} (PA-MCTS), which combines action-value estimates from an out-of-date policy with an online search using an up-to-date model of the environment. We prove theoretical results showing conditions under which PA-MCTS selects the one-step optimal action and also bound the error accrued while following PA-MCTS as a policy. We compare and contrast our approach with AlphaZero, another hybrid planning approach, and Deep Q Learning on several OpenAI Gym environments. Through extensive experiments, we show that under non-stationary settings with limited time constraints, PA-MCTS outperforms these baselines.

{{</citation>}}


### (49/53) A Survey on Verification and Validation, Testing and Evaluations of Neurosymbolic Artificial Intelligence (Justus Renkhoff et al., 2024)

{{<citation>}}

Justus Renkhoff, Ke Feng, Marc Meier-Doernberg, Alvaro Velasquez, Houbing Herbert Song. (2024)  
**A Survey on Verification and Validation, Testing and Evaluations of Neurosymbolic Artificial Intelligence**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.03188v2)  

---


**ABSTRACT**  
Neurosymbolic artificial intelligence (AI) is an emerging branch of AI that combines the strengths of symbolic AI and sub-symbolic AI. A major drawback of sub-symbolic AI is that it acts as a "black box", meaning that predictions are difficult to explain, making the testing & evaluation (T&E) and validation & verification (V&V) processes of a system that uses sub-symbolic AI a challenge. Since neurosymbolic AI combines the advantages of both symbolic and sub-symbolic AI, this survey explores how neurosymbolic applications can ease the V&V process. This survey considers two taxonomies of neurosymbolic AI, evaluates them, and analyzes which algorithms are commonly used as the symbolic and sub-symbolic components in current applications. Additionally, an overview of current techniques for the T&E and V&V processes of these components is provided. Furthermore, it is investigated how the symbolic part is used for T&E and V&V purposes in current neurosymbolic applications. Our research shows that neurosymbolic AI as great potential to ease the T&E and V&V processes of sub-symbolic AI by leveraging the possibilities of symbolic AI. Additionally, the applicability of current T&E and V&V methods to neurosymbolic AI is assessed, and how different neurosymbolic architectures can impact these methods is explored. It is found that current T&E and V&V techniques are partly sufficient to test, evaluate, verify, or validate the symbolic and sub-symbolic part of neurosymbolic applications independently, while some of them use approaches where current T&E and V&V methods are not applicable by default, and adjustments or even new approaches are needed. Our research shows that there is great potential in using symbolic AI to test, evaluate, verify, or validate the predictions of a sub-symbolic model, making neurosymbolic AI an interesting research direction for safe, secure, and trustworthy AI.

{{</citation>}}


### (50/53) Manifold-based Shapley for SAR Recognization Network Explanation (Xuran Hu et al., 2024)

{{<citation>}}

Xuran Hu, Mingzhe Zhu, Yuanjing Liu, Zhenpeng Feng, LJubisa Stankovic. (2024)  
**Manifold-based Shapley for SAR Recognization Network Explanation**  

---
Primary Category: cs.AI  
Categories: H-1-m, cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.03128v1)  

---


**ABSTRACT**  
Explainable artificial intelligence (XAI) holds immense significance in enhancing the deep neural network's transparency and credibility, particularly in some risky and high-cost scenarios, like synthetic aperture radar (SAR). Shapley is a game-based explanation technique with robust mathematical foundations. However, Shapley assumes that model's features are independent, rendering Shapley explanation invalid for high dimensional models. This study introduces a manifold-based Shapley method by projecting high-dimensional features into low-dimensional manifold features and subsequently obtaining Fusion-Shap, which aims at (1) addressing the issue of erroneous explanations encountered by traditional Shap; (2) resolving the challenge of interpretability that traditional Shap faces in complex scenarios.

{{</citation>}}


## cs.IR (1)



### (51/53) QoS-Aware Graph Contrastive Learning for Web Service Recommendation (Jeongwhan Choi et al., 2024)

{{<citation>}}

Jeongwhan Choi, Duksan Ryu. (2024)  
**QoS-Aware Graph Contrastive Learning for Web Service Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs-LG, cs-SE, cs.IR  
Keywords: Contrastive Learning, QA  
[Paper Link](http://arxiv.org/abs/2401.03162v1)  

---


**ABSTRACT**  
With the rapid growth of cloud services driven by advancements in web service technology, selecting a high-quality service from a wide range of options has become a complex task. This study aims to address the challenges of data sparsity and the cold-start problem in web service recommendation using Quality of Service (QoS). We propose a novel approach called QoS-aware graph contrastive learning (QAGCL) for web service recommendation. Our model harnesses the power of graph contrastive learning to handle cold-start problems and improve recommendation accuracy effectively. By constructing contextually augmented graphs with geolocation information and randomness, our model provides diverse views. Through the use of graph convolutional networks and graph contrastive learning techniques, we learn user and service embeddings from these augmented graphs. The learned embeddings are then utilized to seamlessly integrate QoS considerations into the recommendation process. Experimental results demonstrate the superiority of our QAGCL model over several existing models, highlighting its effectiveness in addressing data sparsity and the cold-start problem in QoS-aware service recommendations. Our research contributes to the potential for more accurate recommendations in real-world scenarios, even with limited user-service interaction data.

{{</citation>}}


## cs.SE (1)



### (52/53) Semi-supervised learning via DQN for log anomaly detection (Yingying He et al., 2024)

{{<citation>}}

Yingying He, Xiaobing Pei, Lihong Shen. (2024)  
**Semi-supervised learning via DQN for log anomaly detection**  

---
Primary Category: cs.SE  
Categories: cs-LG, cs-SE, cs.SE  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.03151v1)  

---


**ABSTRACT**  
Log anomaly detection plays a critical role in ensuring the security and maintenance of modern software systems. At present, the primary approach for detecting anomalies in log data is through supervised anomaly detection. Nonetheless, existing supervised methods heavily rely on labeled data, which can be frequently limited in real-world scenarios. In this paper, we propose a semi-supervised log anomaly detection method that combines the DQN algorithm from deep reinforcement learning, which is called DQNLog. DQNLog leverages a small amount of labeled data and a large-scale unlabeled dataset, effectively addressing the challenges of imbalanced data and limited labeling. This approach not only learns known anomalies by interacting with an environment biased towards anomalies but also discovers unknown anomalies by actively exploring the unlabeled dataset. Additionally, DQNLog incorporates a cross-entropy loss term to prevent model overestimation during Deep Reinforcement Learning (DRL). Our evaluation on three widely-used datasets demonstrates that DQNLog significantly improves recall rate and F1-score while maintaining precision, validating its practicality.

{{</citation>}}


## cs.CY (1)



### (53/53) Integrating Personalized Parsons Problems with Multi-Level Textual Explanations to Scaffold Code Writing (Xinying Hou et al., 2024)

{{<citation>}}

Xinying Hou, Barbara J. Ericson, Xu Wang. (2024)  
**Integrating Personalized Parsons Problems with Multi-Level Textual Explanations to Scaffold Code Writing**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs-HC, cs.CY  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2401.03144v2)  

---


**ABSTRACT**  
Novice programmers need to write basic code as part of the learning process, but they often face difficulties. To assist struggling students, we recently implemented personalized Parsons problems, which are code puzzles where students arrange blocks of code to solve them, as pop-up scaffolding. Students found them to be more engaging and preferred them for learning, instead of simply receiving the correct answer, such as the response they might get from generative AI tools like ChatGPT. However, a drawback of using Parsons problems as scaffolding is that students may be able to put the code blocks in the correct order without fully understanding the rationale of the correct solution. As a result, the learning benefits of scaffolding are compromised. Can we improve the understanding of personalized Parsons scaffolding by providing textual code explanations? In this poster, we propose a design that incorporates multiple levels of textual explanations for the Parsons problems. This design will be used for future technical evaluations and classroom experiments. These experiments will explore the effectiveness of adding textual explanations to Parsons problems to improve instructional benefits.

{{</citation>}}
