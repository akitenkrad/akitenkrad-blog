---
draft: false
title: "arXiv @ 2023.10.20"
date: 2023-10-20
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.10.20"
    identifier: arxiv_20231020
    parent: 202310_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.LG (30)](#cslg-30)
- [cs.AI (7)](#csai-7)
- [cs.CL (47)](#cscl-47)
- [eess.AS (2)](#eessas-2)
- [cs.CR (7)](#cscr-7)
- [cs.IR (6)](#csir-6)
- [cs.SE (3)](#csse-3)
- [physics.soc-ph (1)](#physicssoc-ph-1)
- [cs.CV (19)](#cscv-19)
- [cs.MM (1)](#csmm-1)
- [stat.ML (1)](#statml-1)
- [cs.RO (4)](#csro-4)
- [cs.DC (1)](#csdc-1)
- [astro-ph.IM (1)](#astro-phim-1)
- [cs.SD (3)](#cssd-3)
- [eess.IV (2)](#eessiv-2)
- [cs.NI (1)](#csni-1)
- [cs.HC (2)](#cshc-2)
- [cs.CY (1)](#cscy-1)
- [cs.OH (1)](#csoh-1)
- [eess.SY (2)](#eesssy-2)
- [cs.LO (1)](#cslo-1)

## cs.LG (30)



### (1/143) Learning to Solve Climate Sensor Placement Problems with a Transformer (Chen Wang et al., 2023)

{{<citation>}}

Chen Wang, Victoria Huang, Gang Chen, Hui Ma, Bryce Chen, Jochen Schmidt. (2023)  
**Learning to Solve Climate Sensor Placement Problems with a Transformer**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.12387v1)  

---


**ABSTRACT**  
The optimal placement of sensors for environmental monitoring and disaster management is a challenging problem due to its NP-hard nature. Traditional methods for sensor placement involve exact, approximation, or heuristic approaches, with the latter being the most widely used. However, heuristic methods are limited by expert intuition and experience. Deep learning (DL) has emerged as a promising approach for generating heuristic algorithms automatically. In this paper, we introduce a novel sensor placement approach focused on learning improvement heuristics using deep reinforcement learning (RL) methods. Our approach leverages an RL formulation for learning improvement heuristics, driven by an actor-critic algorithm for training the policy network. We compare our method with several state-of-the-art approaches by conducting comprehensive experiments, demonstrating the effectiveness and superiority of our proposed approach in producing high-quality solutions. Our work presents a promising direction for applying advanced DL and RL techniques to challenging climate sensor placement problems.

{{</citation>}}


### (2/143) Networkwide Traffic State Forecasting Using Exogenous Information: A Multi-Dimensional Graph Attention-Based Approach (Syed Islam et al., 2023)

{{<citation>}}

Syed Islam, Monika Filipovska. (2023)  
**Networkwide Traffic State Forecasting Using Exogenous Information: A Multi-Dimensional Graph Attention-Based Approach**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2310.12353v1)  

---


**ABSTRACT**  
Traffic state forecasting is crucial for traffic management and control strategies, as well as user- and system-level decision making in the transportation network. While traffic forecasting has been approached with a variety of techniques over the last couple of decades, most approaches simply rely on endogenous traffic variables for state prediction, despite the evidence that exogenous factors can significantly impact traffic conditions. This paper proposes a multi-dimensional spatio-temporal graph attention-based traffic prediction approach (M-STGAT), which predicts traffic based on past observations of speed, along with lane closure events, temperature, and visibility across the transportation network. The approach is based on a graph attention network architecture, which also learns based on the structure of the transportation network on which these variables are observed. Numerical experiments are performed using traffic speed and lane closure data from the California Department of Transportation (Caltrans) Performance Measurement System (PeMS). The corresponding weather data were downloaded from the National Oceanic and Atmospheric Administration (NOOA) Automated Surface Observing Systems (ASOS). For comparison, the numerical experiments implement three alternative models which do not allow for the multi-dimensional input. The M-STGAT is shown to outperform the three alternative models, when performing tests using our primary data set for prediction with a 30-, 45-, and 60-minute prediction horizon, in terms of three error measures: Mean Absolute Error (MAE), Root Mean Square Error (RMSE) and Mean Absolute Percentage Error (MAPE). However, the model's transferability can vary for different transfer data sets and this aspect may require further investigation.

{{</citation>}}


### (3/143) Equipping Federated Graph Neural Networks with Structure-aware Group Fairness (Nan Cui et al., 2023)

{{<citation>}}

Nan Cui, Xiuling Wang, Wendy Hui Wang, Violet Chen, Yue Ning. (2023)  
**Equipping Federated Graph Neural Networks with Structure-aware Group Fairness**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.12350v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) have been widely used for various types of graph data processing and analytical tasks in different domains. Training GNNs over centralized graph data can be infeasible due to privacy concerns and regulatory restrictions. Thus, federated learning (FL) becomes a trending solution to address this challenge in a distributed learning paradigm. However, as GNNs may inherit historical bias from training data and lead to discriminatory predictions, the bias of local models can be easily propagated to the global model in distributed settings. This poses a new challenge in mitigating bias in federated GNNs. To address this challenge, we propose $\text{F}^2$GNN, a Fair Federated Graph Neural Network, that enhances group fairness of federated GNNs. As bias can be sourced from both data and learning algorithms, $\text{F}^2$GNN aims to mitigate both types of bias under federated settings. First, we provide theoretical insights on the connection between data bias in a training graph and statistical fairness metrics of the trained GNN models. Based on the theoretical analysis, we design $\text{F}^2$GNN which contains two key components: a fairness-aware local model update scheme that enhances group fairness of the local models on the client side, and a fairness-weighted global model update scheme that takes both data bias and fairness metrics of local models into consideration in the aggregation process. We evaluate $\text{F}^2$GNN empirically versus a number of baseline methods, and demonstrate that $\text{F}^2$GNN outperforms these baselines in terms of both fairness and model accuracy.

{{</citation>}}


### (4/143) Open-Set Multivariate Time-Series Anomaly Detection (Thomas Lai et al., 2023)

{{<citation>}}

Thomas Lai, Thi Kieu Khanh Ho, Narges Armanfard. (2023)  
**Open-Set Multivariate Time-Series Anomaly Detection**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2310.12294v1)  

---


**ABSTRACT**  
Numerous methods for time series anomaly detection (TSAD) methods have emerged in recent years. Most existing methods are unsupervised and assume the availability of normal training samples only, while few supervised methods have shown superior performance by incorporating labeled anomalous samples in the training phase. However, certain anomaly types are inherently challenging for unsupervised methods to differentiate from normal data, while supervised methods are constrained to detecting anomalies resembling those present during training, failing to generalize to unseen anomaly classes. This paper is the first attempt in providing a novel approach for the open-set TSAD problem, in which a small number of labeled anomalies from a limited class of anomalies are visible in the training phase, with the objective of detecting both seen and unseen anomaly classes in the test phase. The proposed method, called Multivariate Open-Set timeseries Anomaly Detection (MOSAD) consists of three primary modules: a Feature Extractor to extract meaningful time-series features; a Multi-head Network consisting of Generative-, Deviation-, and Contrastive heads for capturing both seen and unseen anomaly classes; and an Anomaly Scoring module leveraging the insights of the three heads to detect anomalies. Extensive experiments on three real-world datasets consistently show that our approach surpasses existing methods under various experimental settings, thus establishing a new state-of-the-art performance in the TSAD field.

{{</citation>}}


### (5/143) Enhancing the Performance of Automated Grade Prediction in MOOC using Graph Representation Learning (Soheila Farokhi et al., 2023)

{{<citation>}}

Soheila Farokhi, Aswani Yaramala, Jiangtao Huang, Muhammad F. A. Khan, Xiaojun Qi, Hamid Karimi. (2023)  
**Enhancing the Performance of Automated Grade Prediction in MOOC using Graph Representation Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2310.12281v1)  

---


**ABSTRACT**  
In recent years, Massive Open Online Courses (MOOCs) have gained significant traction as a rapidly growing phenomenon in online learning. Unlike traditional classrooms, MOOCs offer a unique opportunity to cater to a diverse audience from different backgrounds and geographical locations. Renowned universities and MOOC-specific providers, such as Coursera, offer MOOC courses on various subjects. Automated assessment tasks like grade and early dropout predictions are necessary due to the high enrollment and limited direct interaction between teachers and learners. However, current automated assessment approaches overlook the structural links between different entities involved in the downstream tasks, such as the students and courses. Our hypothesis suggests that these structural relationships, manifested through an interaction graph, contain valuable information that can enhance the performance of the task at hand. To validate this, we construct a unique knowledge graph for a large MOOC dataset, which will be publicly available to the research community. Furthermore, we utilize graph embedding techniques to extract latent structural information encoded in the interactions between entities in the dataset. These techniques do not require ground truth labels and can be utilized for various tasks. Finally, by combining entity-specific features, behavioral features, and extracted structural features, we enhance the performance of predictive machine learning models in student assignment grade prediction. Our experiments demonstrate that structural features can significantly improve the predictive performance of downstream assessment tasks. The code and data are available in \url{https://github.com/DSAatUSU/MOOPer_grade_prediction}

{{</citation>}}


### (6/143) REVAMP: Automated Simulations of Adversarial Attacks on Arbitrary Objects in Realistic Scenes (Matthew Hull et al., 2023)

{{<citation>}}

Matthew Hull, Zijie J. Wang, Duen Horng Chau. (2023)  
**REVAMP: Automated Simulations of Adversarial Attacks on Arbitrary Objects in Realistic Scenes**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2310.12243v1)  

---


**ABSTRACT**  
Deep Learning models, such as those used in an autonomous vehicle are vulnerable to adversarial attacks where an attacker could place an adversarial object in the environment, leading to mis-classification. Generating these adversarial objects in the digital space has been extensively studied, however successfully transferring these attacks from the digital realm to the physical realm has proven challenging when controlling for real-world environmental factors. In response to these limitations, we introduce REVAMP, an easy-to-use Python library that is the first-of-its-kind tool for creating attack scenarios with arbitrary objects and simulating realistic environmental factors, lighting, reflection, and refraction. REVAMP enables researchers and practitioners to swiftly explore various scenarios within the digital realm by offering a wide range of configurable options for designing experiments and using differentiable rendering to reproduce physically plausible adversarial objects. We will demonstrate and invite the audience to try REVAMP to produce an adversarial texture on a chosen object while having control over various scene parameters. The audience will choose a scene, an object to attack, the desired attack class, and the number of camera positions to use. Then, in real time, we show how this altered texture causes the chosen object to be mis-classified, showcasing the potential of REVAMP in real-world scenarios. REVAMP is open-source and available at https://github.com/poloclub/revamp.

{{</citation>}}


### (7/143) Fairer and More Accurate Tabular Models Through NAS (Richeek Das et al., 2023)

{{<citation>}}

Richeek Das, Samuel Dooley. (2023)  
**Fairer and More Accurate Tabular Models Through NAS**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CY, cs-LG, cs.LG, stat-ML  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.12145v1)  

---


**ABSTRACT**  
Making models algorithmically fairer in tabular data has been long studied, with techniques typically oriented towards fixes which usually take a neural model with an undesirable outcome and make changes to how the data are ingested, what the model weights are, or how outputs are processed. We employ an emergent and different strategy where we consider updating the model's architecture and training hyperparameters to find an entirely new model with better outcomes from the beginning of the debiasing procedure. In this work, we propose using multi-objective Neural Architecture Search (NAS) and Hyperparameter Optimization (HPO) in the first application to the very challenging domain of tabular data. We conduct extensive exploration of architectural and hyperparameter spaces (MLP, ResNet, and FT-Transformer) across diverse datasets, demonstrating the dependence of accuracy and fairness metrics of model predictions on hyperparameter combinations. We show that models optimized solely for accuracy with NAS often fail to inherently address fairness concerns. We propose a novel approach that jointly optimizes architectural and training hyperparameters in a multi-objective constraint of both accuracy and fairness. We produce architectures that consistently Pareto dominate state-of-the-art bias mitigation methods either in fairness, accuracy or both, all of this while being Pareto-optimal over hyperparameters achieved through single-objective (accuracy) optimization runs. This research underscores the promise of automating fairness and accuracy optimization in deep learning models.

{{</citation>}}


### (8/143) SHARCS: Efficient Transformers through Routing with Dynamic Width Sub-networks (Mohammadreza Salehi et al., 2023)

{{<citation>}}

Mohammadreza Salehi, Sachin Mehta, Aditya Kusupati, Ali Farhadi, Hannaneh Hajishirzi. (2023)  
**SHARCS: Efficient Transformers through Routing with Dynamic Width Sub-networks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.12126v1)  

---


**ABSTRACT**  
We introduce SHARCS for adaptive inference that takes into account the hardness of input samples. SHARCS can train a router on any transformer network, enabling the model to direct different samples to sub-networks with varying widths. Our experiments demonstrate that: (1) SHARCS outperforms or complements existing per-sample adaptive inference methods across various classification tasks in terms of accuracy vs. FLOPs; (2) SHARCS generalizes across different architectures and can be even applied to compressed and efficient transformer encoders to further improve their efficiency; (3) SHARCS can provide a 2 times inference speed up at an insignificant drop in accuracy.

{{</citation>}}


### (9/143) Monarch Mixer: A Simple Sub-Quadratic GEMM-Based Architecture (Daniel Y. Fu et al., 2023)

{{<citation>}}

Daniel Y. Fu, Simran Arora, Jessica Grogan, Isys Johnson, Sabri Eyuboglu, Armin W. Thomas, Benjamin Spector, Michael Poli, Atri Rudra, Christopher Ré. (2023)  
**Monarch Mixer: A Simple Sub-Quadratic GEMM-Based Architecture**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: BERT, GLUE, GPT, ImageNet, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.12109v1)  

---


**ABSTRACT**  
Machine learning models are increasingly being scaled in both sequence length and model dimension to reach longer contexts and better performance. However, existing architectures such as Transformers scale quadratically along both these axes. We ask: are there performant architectures that can scale sub-quadratically along sequence length and model dimension? We introduce Monarch Mixer (M2), a new architecture that uses the same sub-quadratic primitive along both sequence length and model dimension: Monarch matrices, a simple class of expressive structured matrices that captures many linear transforms, achieves high hardware efficiency on GPUs, and scales sub-quadratically. As a proof of concept, we explore the performance of M2 in three domains: non-causal BERT-style language modeling, ViT-style image classification, and causal GPT-style language modeling. For non-causal BERT-style modeling, M2 matches BERT-base and BERT-large in downstream GLUE quality with up to 27% fewer parameters, and achieves up to 9.1$\times$ higher throughput at sequence length 4K. On ImageNet, M2 outperforms ViT-b by 1% in accuracy, with only half the parameters. Causal GPT-style models introduce a technical challenge: enforcing causality via masking introduces a quadratic bottleneck. To alleviate this bottleneck, we develop a novel theoretical view of Monarch matrices based on multivariate polynomial evaluation and interpolation, which lets us parameterize M2 to be causal while remaining sub-quadratic. Using this parameterization, M2 matches GPT-style Transformers at 360M parameters in pretraining perplexity on The PILE--showing for the first time that it may be possible to match Transformer quality without attention or MLPs.

{{</citation>}}


### (10/143) Understanding Reward Ambiguity Through Optimal Transport Theory in Inverse Reinforcement Learning (Ali Baheri, 2023)

{{<citation>}}

Ali Baheri. (2023)  
**Understanding Reward Ambiguity Through Optimal Transport Theory in Inverse Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SY, cs.LG, eess-SY, math-OC  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.12055v1)  

---


**ABSTRACT**  
In inverse reinforcement learning (IRL), the central objective is to infer underlying reward functions from observed expert behaviors in a way that not only explains the given data but also generalizes to unseen scenarios. This ensures robustness against reward ambiguity where multiple reward functions can equally explain the same expert behaviors. While significant efforts have been made in addressing this issue, current methods often face challenges with high-dimensional problems and lack a geometric foundation. This paper harnesses the optimal transport (OT) theory to provide a fresh perspective on these challenges. By utilizing the Wasserstein distance from OT, we establish a geometric framework that allows for quantifying reward ambiguity and identifying a central representation or centroid of reward functions. These insights pave the way for robust IRL methodologies anchored in geometric interpretations, offering a structured approach to tackle reward ambiguity in high-dimensional settings.

{{</citation>}}


### (11/143) Removing Spurious Concepts from Neural Network Representations via Joint Subspace Estimation (Floris Holstege et al., 2023)

{{<citation>}}

Floris Holstege, Bram Wouters, Noud van Giersbergen, Cees Diks. (2023)  
**Removing Spurious Concepts from Neural Network Representations via Joint Subspace Estimation**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: NLI  
[Paper Link](http://arxiv.org/abs/2310.11991v1)  

---


**ABSTRACT**  
Out-of-distribution generalization in neural networks is often hampered by spurious correlations. A common strategy is to mitigate this by removing spurious concepts from the neural network representation of the data. Existing concept-removal methods tend to be overzealous by inadvertently eliminating features associated with the main task of the model, thereby harming model performance. We propose an iterative algorithm that separates spurious from main-task concepts by jointly identifying two low-dimensional orthogonal subspaces in the neural network representation. We evaluate the algorithm on benchmark datasets for computer vision (Waterbirds, CelebA) and natural language processing (MultiNLI), and show that it outperforms existing concept removal methods

{{</citation>}}


### (12/143) Image Clustering with External Guidance (Yunfan Li et al., 2023)

{{<citation>}}

Yunfan Li, Peng Hu, Dezhong Peng, Jiancheng Lv, Jianping Fan, Xi Peng. (2023)  
**Image Clustering with External Guidance**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2310.11989v1)  

---


**ABSTRACT**  
The core of clustering is incorporating prior knowledge to construct supervision signals. From classic k-means based on data compactness to recent contrastive clustering guided by self-supervision, the evolution of clustering methods intrinsically corresponds to the progression of supervision signals. At present, substantial efforts have been devoted to mining internal supervision signals from data. Nevertheless, the abundant external knowledge such as semantic descriptions, which naturally conduces to clustering, is regrettably overlooked. In this work, we propose leveraging external knowledge as a new supervision signal to guide clustering, even though it seems irrelevant to the given data. To implement and validate our idea, we design an externally guided clustering method (Text-Aided Clustering, TAC), which leverages the textual semantics of WordNet to facilitate image clustering. Specifically, TAC first selects and retrieves WordNet nouns that best distinguish images to enhance the feature discriminability. Then, to improve image clustering performance, TAC collaborates text and image modalities by mutually distilling cross-modal neighborhood information. Experiments demonstrate that TAC achieves state-of-the-art performance on five widely used and three more challenging image clustering benchmarks, including the full ImageNet-1K dataset.

{{</citation>}}


### (13/143) From Interpolation to Extrapolation: Complete Length Generalization for Arithmetic Transformers (Shaoxiong Duan et al., 2023)

{{<citation>}}

Shaoxiong Duan, Yining Shi. (2023)  
**From Interpolation to Extrapolation: Complete Length Generalization for Arithmetic Transformers**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Attention, Bias, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.11984v1)  

---


**ABSTRACT**  
Since its introduction, the transformer model has demonstrated outstanding performance across various tasks. However, there are still unresolved issues regarding length generalization, particularly in algorithmic tasks. In this paper, we investigate the inherent capabilities of transformer models in learning arithmetic algorithms, such as addition and multiplication. Through experiments and attention analysis, we identify a number of crucial factors for achieving optimal length generalization. We show that transformer models are able to generalize to long lengths with the help of targeted attention biasing. We then introduce Attention Bias Calibration (ABC), a calibration stage that enables the model to automatically learn the proper attention biases, which we link to mechanisms in relative position encoding. We demonstrate that using ABC, the transformer model can achieve unprecedented perfect length generalization on certain arithmetic tasks.

{{</citation>}}


### (14/143) Improving Generalization of Alignment with Human Preferences through Group Invariant Learning (Rui Zheng et al., 2023)

{{<citation>}}

Rui Zheng, Wei Shen, Yuan Hua, Wenbin Lai, Shihan Dou, Yuhao Zhou, Zhiheng Xi, Xiao Wang, Haoran Huang, Tao Gui, Qi Zhang, Xuanjing Huang. (2023)  
**Improving Generalization of Alignment with Human Preferences through Group Invariant Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.11971v2)  

---


**ABSTRACT**  
The success of AI assistants based on language models (LLMs) hinges crucially on Reinforcement Learning from Human Feedback (RLHF), which enables the generation of responses more aligned with human preferences. As universal AI assistants, there's a growing expectation for them to perform consistently across various domains. However, previous work shows that Reinforcement Learning (RL) often exploits shortcuts to attain high rewards and overlooks challenging samples. This focus on quick reward gains undermines both the stability in training and the model's ability to generalize to new, unseen data. In this work, we propose a novel approach that can learn a consistent policy via RL across various data groups or domains. Given the challenges associated with acquiring group annotations, our method automatically classifies data into different groups, deliberately maximizing performance variance. Then, we optimize the policy to perform well on challenging groups. Lastly, leveraging the established groups, our approach adaptively adjusts the exploration space, allocating more learning capacity to more challenging data and preventing the model from over-optimizing on simpler data. Experimental results indicate that our approach significantly enhances training stability and model generalization.

{{</citation>}}


### (15/143) A Multi-Scale Decomposition MLP-Mixer for Time Series Analysis (Shuhan Zhong et al., 2023)

{{<citation>}}

Shuhan Zhong, Sizhe Song, Guanyao Li, Weipeng Zhuo, Yang Liu, S. -H. Gary Chan. (2023)  
**A Multi-Scale Decomposition MLP-Mixer for Time Series Analysis**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2310.11959v1)  

---


**ABSTRACT**  
Time series data, often characterized by unique composition and complex multi-scale temporal variations, requires special consideration of decomposition and multi-scale modeling in its analysis. Existing deep learning methods on this best fit to only univariate time series, and have not sufficiently accounted for sub-series level modeling and decomposition completeness. To address this, we propose MSD-Mixer, a Multi-Scale Decomposition MLP-Mixer which learns to explicitly decompose the input time series into different components, and represents the components in different layers. To handle multi-scale temporal patterns and inter-channel dependencies, we propose a novel temporal patching approach to model the time series as multi-scale sub-series, i.e., patches, and employ MLPs to mix intra- and inter-patch variations and channel-wise correlations. In addition, we propose a loss function to constrain both the magnitude and autocorrelation of the decomposition residual for decomposition completeness. Through extensive experiments on various real-world datasets for five common time series analysis tasks (long- and short-term forecasting, imputation, anomaly detection, and classification), we demonstrate that MSD-Mixer consistently achieves significantly better performance in comparison with other state-of-the-art task-general and task-specific approaches.

{{</citation>}}


### (16/143) Recasting Continual Learning as Sequence Modeling (Soochan Lee et al., 2023)

{{<citation>}}

Soochan Lee, Jaehyeon Son, Gunhee Kim. (2023)  
**Recasting Continual Learning as Sequence Modeling**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.11952v1)  

---


**ABSTRACT**  
In this work, we aim to establish a strong connection between two significant bodies of machine learning research: continual learning and sequence modeling. That is, we propose to formulate continual learning as a sequence modeling problem, allowing advanced sequence models to be utilized for continual learning. Under this formulation, the continual learning process becomes the forward pass of a sequence model. By adopting the meta-continual learning (MCL) framework, we can train the sequence model at the meta-level, on multiple continual learning episodes. As a specific example of our new formulation, we demonstrate the application of Transformers and their efficient variants as MCL methods. Our experiments on seven benchmarks, covering both classification and regression, show that sequence models can be an attractive solution for general MCL.

{{</citation>}}


### (17/143) Accelerated Policy Gradient: On the Nesterov Momentum for Reinforcement Learning (Yen-Ju Chen et al., 2023)

{{<citation>}}

Yen-Ju Chen, Nai-Chieh Huang, Ping-Chun Hsieh. (2023)  
**Accelerated Policy Gradient: On the Nesterov Momentum for Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.11897v1)  

---


**ABSTRACT**  
Policy gradient methods have recently been shown to enjoy global convergence at a $\Theta(1/t)$ rate in the non-regularized tabular softmax setting. Accordingly, one important research question is whether this convergence rate can be further improved, with only first-order updates. In this paper, we answer the above question from the perspective of momentum by adapting the celebrated Nesterov's accelerated gradient (NAG) method to reinforcement learning (RL), termed \textit{Accelerated Policy Gradient} (APG). To demonstrate the potential of APG in achieving faster global convergence, we formally show that with the true gradient, APG with softmax policy parametrization converges to an optimal policy at a $\tilde{O}(1/t^2)$ rate. To the best of our knowledge, this is the first characterization of the global convergence rate of NAG in the context of RL. Notably, our analysis relies on one interesting finding: Regardless of the initialization, APG could end up reaching a locally nearly-concave regime, where APG could benefit significantly from the momentum, within finite iterations. By means of numerical validation, we confirm that APG exhibits $\tilde{O}(1/t^2)$ rate as well as show that APG could significantly improve the convergence behavior over the standard policy gradient.

{{</citation>}}


### (18/143) Accelerate Presolve in Large-Scale Linear Programming via Reinforcement Learning (Yufei Kuang et al., 2023)

{{<citation>}}

Yufei Kuang, Xijun Li, Jie Wang, Fangzhou Zhu, Meng Lu, Zhihai Wang, Jia Zeng, Houqiang Li, Yongdong Zhang, Feng Wu. (2023)  
**Accelerate Presolve in Large-Scale Linear Programming via Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.11845v1)  

---


**ABSTRACT**  
Large-scale LP problems from industry usually contain much redundancy that severely hurts the efficiency and reliability of solving LPs, making presolve (i.e., the problem simplification module) one of the most critical components in modern LP solvers. However, how to design high-quality presolve routines -- that is, the program determining (P1) which presolvers to select, (P2) in what order to execute, and (P3) when to stop -- remains a highly challenging task due to the extensive requirements on expert knowledge and the large search space. Due to the sequential decision property of the task and the lack of expert demonstrations, we propose a simple and efficient reinforcement learning (RL) framework -- namely, reinforcement learning for presolve (RL4Presolve) -- to tackle (P1)-(P3) simultaneously. Specifically, we formulate the routine design task as a Markov decision process and propose an RL framework with adaptive action sequences to generate high-quality presolve routines efficiently. Note that adaptive action sequences help learn complex behaviors efficiently and adapt to various benchmarks. Experiments on two solvers (open-source and commercial) and eight benchmarks (real-world and synthetic) demonstrate that RL4Presolve significantly and consistently improves the efficiency of solving large-scale LPs, especially on benchmarks from industry. Furthermore, we optimize the hard-coded presolve routines in LP solvers by extracting rules from learned policies for simple and efficient deployment to Huawei's supply chain. The results show encouraging economic and academic potential for incorporating machine learning to modern solvers.

{{</citation>}}


### (19/143) On The Expressivity of Objective-Specification Formalisms in Reinforcement Learning (Rohan Subramani et al., 2023)

{{<citation>}}

Rohan Subramani, Marcus Williams, Max Heitmann, Halfdan Holm, Charlie Griffin, Joar Skalse. (2023)  
**On The Expressivity of Objective-Specification Formalisms in Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.11840v1)  

---


**ABSTRACT**  
To solve a task with reinforcement learning (RL), it is necessary to formally specify the goal of that task. Although most RL algorithms require that the goal is formalised as a Markovian reward function, alternatives have been developed (such as Linear Temporal Logic and Multi-Objective Reinforcement Learning). Moreover, it is well known that some of these formalisms are able to express certain tasks that other formalisms cannot express. However, there has not yet been any thorough analysis of how these formalisms relate to each other in terms of expressivity. In this work, we fill this gap in the existing literature by providing a comprehensive comparison of the expressivities of 17 objective-specification formalisms in RL. We place these formalisms in a preorder based on their expressive power, and present this preorder as a Hasse diagram. We find a variety of limitations for the different formalisms, and that no formalism is both dominantly expressive and straightforward to optimise with current techniques. For example, we prove that each of Regularised RL, Outer Nonlinear Markov Rewards, Reward Machines, Linear Temporal Logic, and Limit Average Rewards can express an objective that the others cannot. Our findings have implications for both policy optimisation and reward learning. Firstly, we identify expressivity limitations which are important to consider when specifying objectives in practice. Secondly, our results highlight the need for future research which adapts reward learning to work with a variety of formalisms, since many existing reward learning methods implicitly assume that desired objectives can be expressed with Markovian rewards. Our work contributes towards a more cohesive understanding of the costs and benefits of different RL objective-specification formalisms.

{{</citation>}}


### (20/143) Conservative Predictions on Noisy Financial Data (Omkar Nabar et al., 2023)

{{<citation>}}

Omkar Nabar, Gautam Shroff. (2023)  
**Conservative Predictions on Noisy Financial Data**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CE, cs-LG, cs.LG  
Keywords: Financial  
[Paper Link](http://arxiv.org/abs/2310.11815v1)  

---


**ABSTRACT**  
Price movements in financial markets are well known to be very noisy. As a result, even if there are, on occasion, exploitable patterns that could be picked up by machine-learning algorithms, these are obscured by feature and label noise rendering the predictions less useful, and risky in practice. Traditional rule-learning techniques developed for noisy data, such as CN2, would seek only high precision rules and refrain from making predictions where their antecedents did not apply. We apply a similar approach, where a model abstains from making a prediction on data points that it is uncertain on. During training, a cascade of such models are learned in sequence, similar to rule lists, with each model being trained only on data on which the previous model(s) were uncertain. Similar pruning of data takes place at test-time, with (higher accuracy) predictions being made albeit only on a fraction (support) of test-time data. In a financial prediction setting, such an approach allows decisions to be taken only when the ensemble model is confident, thereby reducing risk. We present results using traditional MLPs as well as differentiable decision trees, on synthetic data as well as real financial market data, to predict fixed-term returns using commonly used features. We submit that our approach is likely to result in better overall returns at a lower level of risk. In this context we introduce an utility metric to measure the average gain per trade, as well as the return adjusted for downside risk, both of which are improved significantly by our approach.

{{</citation>}}


### (21/143) Adversarial Training for Physics-Informed Neural Networks (Yao Li et al., 2023)

{{<citation>}}

Yao Li, Shengzhu Shi, Zhichang Guo, Boying Wu. (2023)  
**Adversarial Training for Physics-Informed Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-NA, cs.LG, math-NA  
Keywords: Adversarial Training  
[Paper Link](http://arxiv.org/abs/2310.11789v1)  

---


**ABSTRACT**  
Physics-informed neural networks have shown great promise in solving partial differential equations. However, due to insufficient robustness, vanilla PINNs often face challenges when solving complex PDEs, especially those involving multi-scale behaviors or solutions with sharp or oscillatory characteristics. To address these issues, based on the projected gradient descent adversarial attack, we proposed an adversarial training strategy for PINNs termed by AT-PINNs. AT-PINNs enhance the robustness of PINNs by fine-tuning the model with adversarial samples, which can accurately identify model failure locations and drive the model to focus on those regions during training. AT-PINNs can also perform inference with temporal causality by selecting the initial collocation points around temporal initial values. We implement AT-PINNs to the elliptic equation with multi-scale coefficients, Poisson equation with multi-peak solutions, Burgers equation with sharp solutions and the Allen-Cahn equation. The results demonstrate that AT-PINNs can effectively locate and reduce failure regions. Moreover, AT-PINNs are suitable for solving complex PDEs, since locating failure regions through adversarial attacks is independent of the size of failure regions or the complexity of the distribution.

{{</citation>}}


### (22/143) A Quasi-Wasserstein Loss for Learning Graph Neural Networks (Minjie Cheng et al., 2023)

{{<citation>}}

Minjie Cheng, Hongteng Xu. (2023)  
**A Quasi-Wasserstein Loss for Learning Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.11762v2)  

---


**ABSTRACT**  
When learning graph neural networks (GNNs) in node-level prediction tasks, most existing loss functions are applied for each node independently, even if node embeddings and their labels are non-i.i.d. because of their graph structures. To eliminate such inconsistency, in this study we propose a novel Quasi-Wasserstein (QW) loss with the help of the optimal transport defined on graphs, leading to new learning and prediction paradigms of GNNs. In particular, we design a "Quasi-Wasserstein" distance between the observed multi-dimensional node labels and their estimations, optimizing the label transport defined on graph edges. The estimations are parameterized by a GNN in which the optimal label transport may determine the graph edge weights optionally. By reformulating the strict constraint of the label transport to a Bregman divergence-based regularizer, we obtain the proposed Quasi-Wasserstein loss associated with two efficient solvers learning the GNN together with optimal label transport. When predicting node labels, our model combines the output of the GNN with the residual component provided by the optimal label transport, leading to a new transductive prediction paradigm. Experiments show that the proposed QW loss applies to various GNNs and helps to improve their performance in node-level classification and regression tasks.

{{</citation>}}


### (23/143) Investigating Uncertainty Calibration of Aligned Language Models under the Multiple-Choice Setting (Guande He et al., 2023)

{{<citation>}}

Guande He, Peng Cui, Jianfei Chen, Wenbo Hu, Jun Zhu. (2023)  
**Investigating Uncertainty Calibration of Aligned Language Models under the Multiple-Choice Setting**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.11732v1)  

---


**ABSTRACT**  
Despite the significant progress made in practical applications of aligned language models (LMs), they tend to be overconfident in output answers compared to the corresponding pre-trained LMs. In this work, we systematically evaluate the impact of the alignment process on logit-based uncertainty calibration of LMs under the multiple-choice setting. We first conduct a thoughtful empirical study on how aligned LMs differ in calibration from their pre-trained counterparts. Experimental results reveal that there are two distinct uncertainties in LMs under the multiple-choice setting, which are responsible for the answer decision and the format preference of the LMs, respectively. Then, we investigate the role of these two uncertainties on aligned LM's calibration through fine-tuning in simple synthetic alignment schemes and conclude that one reason for aligned LMs' overconfidence is the conflation of these two types of uncertainty. Furthermore, we examine the utility of common post-hoc calibration methods for aligned LMs and propose an easy-to-implement and sample-efficient method to calibrate aligned LMs. We hope our findings could provide insights into the design of more reliable alignment processes for LMs.

{{</citation>}}


### (24/143) Federated Heterogeneous Graph Neural Network for Privacy-preserving Recommendation (Bo Yan et al., 2023)

{{<citation>}}

Bo Yan, Yang Cao, Haoyu Wang, Wenchuan Yang, Junping Du, Chuan Shi. (2023)  
**Federated Heterogeneous Graph Neural Network for Privacy-preserving Recommendation**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CR, cs-DC, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2310.11730v1)  

---


**ABSTRACT**  
Heterogeneous information network (HIN), which contains rich semantics depicted by meta-paths, has become a powerful tool to alleviate data sparsity in recommender systems. Existing HIN-based recommendations hold the data centralized storage assumption and conduct centralized model training. However, the real-world data is often stored in a distributed manner for privacy concerns, resulting in the failure of centralized HIN-based recommendations. In this paper, we suggest the HIN is partitioned into private HINs stored in the client side and shared HINs in the server. Following this setting, we propose a federated heterogeneous graph neural network (FedHGNN) based framework, which can collaboratively train a recommendation model on distributed HINs without leaking user privacy. Specifically, we first formalize the privacy definition in the light of differential privacy for HIN-based federated recommendation, which aims to protect user-item interactions of private HIN as well as user's high-order patterns from shared HINs. To recover the broken meta-path based semantics caused by distributed data storage and satisfy the proposed privacy, we elaborately design a semantic-preserving user interactions publishing method, which locally perturbs user's high-order patterns as well as related user-item interactions for publishing. After that, we propose a HGNN model for recommendation, which conducts node- and semantic-level aggregations to capture recovered semantics. Extensive experiments on three datasets demonstrate our model outperforms existing methods by a large margin (up to 34% in HR@10 and 42% in NDCG@10) under an acceptable privacy budget.

{{</citation>}}


### (25/143) Learning under Label Proportions for Text Classification (Jatin Chauhan et al., 2023)

{{<citation>}}

Jatin Chauhan, Xiaoxuan Wang, Wei Wang. (2023)  
**Learning under Label Proportions for Text Classification**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: NLP, Text Classification  
[Paper Link](http://arxiv.org/abs/2310.11707v1)  

---


**ABSTRACT**  
We present one of the preliminary NLP works under the challenging setup of Learning from Label Proportions (LLP), where the data is provided in an aggregate form called bags and only the proportion of samples in each class as the ground truth. This setup is inline with the desired characteristics of training models under Privacy settings and Weakly supervision. By characterizing some irregularities of the most widely used baseline technique DLLP, we propose a novel formulation that is also robust. This is accompanied with a learnability result that provides a generalization bound under LLP. Combining this formulation with a self-supervised objective, our method achieves better results as compared to the baselines in almost 87% of the experimental configurations which include large scale models for both long and short range texts across multiple metrics.

{{</citation>}}


### (26/143) Architectural Implications of GNN Aggregation Programming Abstractions (Yingjie Qi et al., 2023)

{{<citation>}}

Yingjie Qi, Jianlei Yang, Ao Zhou, Tong Qiao, Chunming Hu. (2023)  
**Architectural Implications of GNN Aggregation Programming Abstractions**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-PF, cs.LG  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2310.12184v2)  

---


**ABSTRACT**  
Graph neural networks (GNNs) have gained significant popularity due to the powerful capability to extract useful representations from graph data. As the need for efficient GNN computation intensifies, a variety of programming abstractions designed for optimizing GNN Aggregation have emerged to facilitate acceleration. However, there is no comprehensive evaluation and analysis upon existing abstractions, thus no clear consensus on which approach is better. In this letter, we classify existing programming abstractions for GNN Aggregation by the dimension of data organization and propagation method. By constructing these abstractions on a state-of-the-art GNN library, we perform a thorough and detailed characterization study to compare their performance and efficiency, and provide several insights on future GNN acceleration based on our analysis.

{{</citation>}}


### (27/143) Quantum Acceleration of Infinite Horizon Average-Reward Reinforcement Learning (Bhargav Ganguly et al., 2023)

{{<citation>}}

Bhargav Ganguly, Vaneet Aggarwal. (2023)  
**Quantum Acceleration of Infinite Horizon Average-Reward Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, quant-ph  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.11684v1)  

---


**ABSTRACT**  
This paper investigates the potential of quantum acceleration in addressing infinite horizon Markov Decision Processes (MDPs) to enhance average reward outcomes. We introduce an innovative quantum framework for the agent's engagement with an unknown MDP, extending the conventional interaction paradigm. Our approach involves the design of an optimism-driven tabular Reinforcement Learning algorithm that harnesses quantum signals acquired by the agent through efficient quantum mean estimation techniques. Through thorough theoretical analysis, we demonstrate that the quantum advantage in mean estimation leads to exponential advancements in regret guarantees for infinite horizon Reinforcement Learning. Specifically, the proposed Quantum algorithm achieves a regret bound of $\tilde{\mathcal{O}}(1)$, a significant improvement over the $\tilde{\mathcal{O}}(\sqrt{T})$ bound exhibited by classical counterparts.

{{</citation>}}


### (28/143) Using Experience Classification for Training Non-Markovian Tasks (Ruixuan Miao et al., 2023)

{{<citation>}}

Ruixuan Miao, Xu Lu, Cong Tian, Bin Yu, Zhenhua Duan. (2023)  
**Using Experience Classification for Training Non-Markovian Tasks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-FL, cs-LG, cs-LO, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.11678v1)  

---


**ABSTRACT**  
Unlike the standard Reinforcement Learning (RL) model, many real-world tasks are non-Markovian, whose rewards are predicated on state history rather than solely on the current state. Solving a non-Markovian task, frequently applied in practical applications such as autonomous driving, financial trading, and medical diagnosis, can be quite challenging. We propose a novel RL approach to achieve non-Markovian rewards expressed in temporal logic LTL$_f$ (Linear Temporal Logic over Finite Traces). To this end, an encoding of linear complexity from LTL$_f$ into MDPs (Markov Decision Processes) is introduced to take advantage of advanced RL algorithms. Then, a prioritized experience replay technique based on the automata structure (semantics equivalent to LTL$_f$ specification) is utilized to improve the training process. We empirically evaluate several benchmark problems augmented with non-Markovian tasks to demonstrate the feasibility and effectiveness of our approach.

{{</citation>}}


### (29/143) PREM: A Simple Yet Effective Approach for Node-Level Graph Anomaly Detection (Junjun Pan et al., 2023)

{{<citation>}}

Junjun Pan, Yixin Liu, Yizhen Zheng, Shirui Pan. (2023)  
**PREM: A Simple Yet Effective Approach for Node-Level Graph Anomaly Detection**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2310.11676v1)  

---


**ABSTRACT**  
Node-level graph anomaly detection (GAD) plays a critical role in identifying anomalous nodes from graph-structured data in various domains such as medicine, social networks, and e-commerce. However, challenges have arisen due to the diversity of anomalies and the dearth of labeled data. Existing methodologies - reconstruction-based and contrastive learning - while effective, often suffer from efficiency issues, stemming from their complex objectives and elaborate modules. To improve the efficiency of GAD, we introduce a simple method termed PREprocessing and Matching (PREM for short). Our approach streamlines GAD, reducing time and memory consumption while maintaining powerful anomaly detection capabilities. Comprising two modules - a pre-processing module and an ego-neighbor matching module - PREM eliminates the necessity for message-passing propagation during training, and employs a simple contrastive loss, leading to considerable reductions in training time and memory usage. Moreover, through rigorous evaluations of five real-world datasets, our method demonstrated robustness and effectiveness. Notably, when validated on the ACM dataset, PREM achieved a 5% improvement in AUC, a 9-fold increase in training speed, and sharply reduce memory usage compared to the most efficient baseline.

{{</citation>}}


### (30/143) Hetero$^2$Net: Heterophily-aware Representation Learning on Heterogenerous Graphs (Jintang Li et al., 2023)

{{<citation>}}

Jintang Li, Zheng Wei, Jiawang Dan, Jing Zhou, Yuchang Zhu, Ruofan Wu, Baokun Wang, Zhang Zhen, Changhua Meng, Hong Jin, Zibin Zheng, Liang Chen. (2023)  
**Hetero$^2$Net: Heterophily-aware Representation Learning on Heterogenerous Graphs**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Representation Learning  
[Paper Link](http://arxiv.org/abs/2310.11664v1)  

---


**ABSTRACT**  
Real-world graphs are typically complex, exhibiting heterogeneity in the global structure, as well as strong heterophily within local neighborhoods. While a growing body of literature has revealed the limitations of common graph neural networks (GNNs) in handling homogeneous graphs with heterophily, little work has been conducted on investigating the heterophily properties in the context of heterogeneous graphs. To bridge this research gap, we identify the heterophily in heterogeneous graphs using metapaths and propose two practical metrics to quantitatively describe the levels of heterophily. Through in-depth investigations on several real-world heterogeneous graphs exhibiting varying levels of heterophily, we have observed that heterogeneous graph neural networks (HGNNs), which inherit many mechanisms from GNNs designed for homogeneous graphs, fail to generalize to heterogeneous graphs with heterophily or low level of homophily. To address the challenge, we present Hetero$^2$Net, a heterophily-aware HGNN that incorporates both masked metapath prediction and masked label prediction tasks to effectively and flexibly handle both homophilic and heterophilic heterogeneous graphs. We evaluate the performance of Hetero$^2$Net on five real-world heterogeneous graph benchmarks with varying levels of heterophily. The results demonstrate that Hetero$^2$Net outperforms strong baselines in the semi-supervised node classification task, providing valuable insights into effectively handling more complex heterogeneous graphs.

{{</citation>}}


## cs.AI (7)



### (31/143) Online Learning and Planning in Cognitive Hierarchies (Bernhard Hengst et al., 2023)

{{<citation>}}

Bernhard Hengst, Maurice Pagnucco, David Rajaratnam, Claude Sammut, Michael Thielscher. (2023)  
**Online Learning and Planning in Cognitive Hierarchies**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-RO, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.12386v1)  

---


**ABSTRACT**  
Complex robot behaviour typically requires the integration of multiple robotic and Artificial Intelligence (AI) techniques and components. Integrating such disparate components into a coherent system, while also ensuring global properties and behaviours, is a significant challenge for cognitive robotics. Using a formal framework to model the interactions between components can be an important step in dealing with this challenge. In this paper we extend an existing formal framework [Clark et al., 2016] to model complex integrated reasoning behaviours of robotic systems; from symbolic planning through to online learning of policies and transition systems. Furthermore the new framework allows for a more flexible modelling of the interactions between different reasoning components.

{{</citation>}}


### (32/143) Fact-based Agent modeling for Multi-Agent Reinforcement Learning (Baofu Fang et al., 2023)

{{<citation>}}

Baofu Fang, Caiming Zheng, Hao Wang. (2023)  
**Fact-based Agent modeling for Multi-Agent Reinforcement Learning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.12290v1)  

---


**ABSTRACT**  
In multi-agent systems, agents need to interact and collaborate with other agents in environments. Agent modeling is crucial to facilitate agent interactions and make adaptive cooperation strategies. However, it is challenging for agents to model the beliefs, behaviors, and intentions of other agents in non-stationary environment where all agent policies are learned simultaneously. In addition, the existing methods realize agent modeling through behavior cloning which assume that the local information of other agents can be accessed during execution or training. However, this assumption is infeasible in unknown scenarios characterized by unknown agents, such as competition teams, unreliable communication and federated learning due to privacy concerns. To eliminate this assumption and achieve agent modeling in unknown scenarios, Fact-based Agent modeling (FAM) method is proposed in which fact-based belief inference (FBI) network models other agents in partially observable environment only based on its local information. The reward and observation obtained by agents after taking actions are called facts, and FAM uses facts as reconstruction target to learn the policy representation of other agents through a variational autoencoder. We evaluate FAM on various Multiagent Particle Environment (MPE) and compare the results with several state-of-the-art MARL algorithms. Experimental results show that compared with baseline methods, FAM can effectively improve the efficiency of agent policy learning by making adaptive cooperation strategies in multi-agent reinforcement learning tasks, while achieving higher returns in complex competitive-cooperative mixed scenarios.

{{</citation>}}


### (33/143) Sociotechnical Safety Evaluation of Generative AI Systems (Laura Weidinger et al., 2023)

{{<citation>}}

Laura Weidinger, Maribeth Rauh, Nahema Marchal, Arianna Manzini, Lisa Anne Hendricks, Juan Mateos-Garcia, Stevie Bergman, Jackie Kay, Conor Griffin, Ben Bariach, Iason Gabriel, Verena Rieser, William Isaac. (2023)  
**Sociotechnical Safety Evaluation of Generative AI Systems**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-CY, cs.AI  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2310.11986v1)  

---


**ABSTRACT**  
Generative AI systems produce a range of risks. To ensure the safety of generative AI systems, these risks must be evaluated. In this paper, we make two main contributions toward establishing such evaluations. First, we propose a three-layered framework that takes a structured, sociotechnical approach to evaluating these risks. This framework encompasses capability evaluations, which are the main current approach to safety evaluation. It then reaches further by building on system safety principles, particularly the insight that context determines whether a given capability may cause harm. To account for relevant context, our framework adds human interaction and systemic impacts as additional layers of evaluation. Second, we survey the current state of safety evaluation of generative AI systems and create a repository of existing evaluations. Three salient evaluation gaps emerge from this analysis. We propose ways forward to closing these gaps, outlining practical steps as well as roles and responsibilities for different actors. Sociotechnical safety evaluation is a tractable approach to the robust and comprehensive safety evaluation of generative AI systems.

{{</citation>}}


### (34/143) From Neural Activations to Concepts: A Survey on Explaining Concepts in Neural Networks (Jae Hee Lee et al., 2023)

{{<citation>}}

Jae Hee Lee, Sergio Lanza, Stefan Wermter. (2023)  
**From Neural Activations to Concepts: A Survey on Explaining Concepts in Neural Networks**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs-NE, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.11884v1)  

---


**ABSTRACT**  
In this paper, we review recent approaches for explaining concepts in neural networks. Concepts can act as a natural link between learning and reasoning: once the concepts are identified that a neural learning system uses, one can integrate those concepts with a reasoning system for inference or use a reasoning system to act upon them to improve or enhance the learning system. On the other hand, knowledge can not only be extracted from neural networks but concept knowledge can also be inserted into neural network architectures. Since integrating learning and reasoning is at the core of neuro-symbolic AI, the insights gained from this survey can serve as an important step towards realizing neuro-symbolic AI based on explainable concepts.

{{</citation>}}


### (35/143) IntentDial: An Intent Graph based Multi-Turn Dialogue System with Reasoning Path Visualization (Zengguang Hao et al., 2023)

{{<citation>}}

Zengguang Hao, Jie Zhang, Binxia Xu, Yafang Wang, Gerard de Melo, Xiaolong Li. (2023)  
**IntentDial: An Intent Graph based Multi-Turn Dialogue System with Reasoning Path Visualization**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Dialog, Dialogue, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.11818v1)  

---


**ABSTRACT**  
Intent detection and identification from multi-turn dialogue has become a widely explored technique in conversational agents, for example, voice assistants and intelligent customer services. The conventional approaches typically cast the intent mining process as a classification task. Although neural classifiers have proven adept at such classification tasks, the issue of neural network models often impedes their practical deployment in real-world settings. We present a novel graph-based multi-turn dialogue system called , which identifies a user's intent by identifying intent elements and a standard query from a dynamically constructed and extensible intent graph using reinforcement learning. In addition, we provide visualization components to monitor the immediate reasoning path for each turn of a dialogue, which greatly facilitates further improvement of the system.

{{</citation>}}


### (36/143) Action-Quantized Offline Reinforcement Learning for Robotic Skill Learning (Jianlan Luo et al., 2023)

{{<citation>}}

Jianlan Luo, Perry Dong, Jeffrey Wu, Aviral Kumar, Xinyang Geng, Sergey Levine. (2023)  
**Action-Quantized Offline Reinforcement Learning for Robotic Skill Learning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.11731v1)  

---


**ABSTRACT**  
The offline reinforcement learning (RL) paradigm provides a general recipe to convert static behavior datasets into policies that can perform better than the policy that collected the data. While policy constraints, conservatism, and other methods for mitigating distributional shifts have made offline reinforcement learning more effective, the continuous action setting often necessitates various approximations for applying these techniques. Many of these challenges are greatly alleviated in discrete action settings, where offline RL constraints and regularizers can often be computed more precisely or even exactly. In this paper, we propose an adaptive scheme for action quantization. We use a VQ-VAE to learn state-conditioned action quantization, avoiding the exponential blowup that comes with na\"ive discretization of the action space. We show that several state-of-the-art offline RL methods such as IQL, CQL, and BRAC improve in performance on benchmarks when combined with our proposed discretization scheme. We further validate our approach on a set of challenging long-horizon complex robotic manipulation tasks in the Robomimic environment, where our discretized offline RL algorithms are able to improve upon their continuous counterparts by 2-3x. Our project page is at https://saqrl.github.io/

{{</citation>}}


### (37/143) SOTOPIA: Interactive Evaluation for Social Intelligence in Language Agents (Xuhui Zhou et al., 2023)

{{<citation>}}

Xuhui Zhou, Hao Zhu, Leena Mathur, Ruohong Zhang, Haofei Yu, Zhengyang Qi, Louis-Philippe Morency, Yonatan Bisk, Daniel Fried, Graham Neubig, Maarten Sap. (2023)  
**SOTOPIA: Interactive Evaluation for Social Intelligence in Language Agents**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-LG, cs.AI  
Keywords: AI, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2310.11667v1)  

---


**ABSTRACT**  
Humans are social beings; we pursue social goals in our daily interactions, which is a crucial aspect of social intelligence. Yet, AI systems' abilities in this realm remain elusive. We present SOTOPIA, an open-ended environment to simulate complex social interactions between artificial agents and evaluate their social intelligence. In our environment, agents role-play and interact under a wide variety of scenarios; they coordinate, collaborate, exchange, and compete with each other to achieve complex social goals. We simulate the role-play interaction between LLM-based agents and humans within this task space and evaluate their performance with a holistic evaluation framework called SOTOPIA-Eval. With SOTOPIA, we find significant differences between these models in terms of their social intelligence, and we identify a subset of SOTOPIA scenarios, SOTOPIA-hard, that is generally challenging for all models. We find that on this subset, GPT-4 achieves a significantly lower goal completion rate than humans and struggles to exhibit social commonsense reasoning and strategic communication skills. These findings demonstrate SOTOPIA's promise as a general platform for research on evaluating and improving social intelligence in artificial agents.

{{</citation>}}


## cs.CL (47)



### (38/143) Solving Hard Analogy Questions with Relation Embedding Chains (Nitesh Kumar et al., 2023)

{{<citation>}}

Nitesh Kumar, Steven Schockaert. (2023)  
**Solving Hard Analogy Questions with Relation Embedding Chains**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2310.12379v1)  

---


**ABSTRACT**  
Modelling how concepts are related is a central topic in Lexical Semantics. A common strategy is to rely on knowledge graphs (KGs) such as ConceptNet, and to model the relation between two concepts as a set of paths. However, KGs are limited to a fixed set of relation types, and they are incomplete and often noisy. Another strategy is to distill relation embeddings from a fine-tuned language model. However, this is less suitable for words that are only indirectly related and it does not readily allow us to incorporate structured domain knowledge. In this paper, we aim to combine the best of both worlds. We model relations as paths but associate their edges with relation embeddings. The paths are obtained by first identifying suitable intermediate words and then selecting those words for which informative relation embeddings can be obtained. We empirically show that our proposed representations are useful for solving hard analogy questions.

{{</citation>}}


### (39/143) GRI: Graph-based Relative Isomorphism of Word Embedding Spaces (Muhammad Asif Ali et al., 2023)

{{<citation>}}

Muhammad Asif Ali, Yan Hu, Jianbin Qin, Di Wang. (2023)  
**GRI: Graph-based Relative Isomorphism of Word Embedding Spaces**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Embedding, Word Embedding  
[Paper Link](http://arxiv.org/abs/2310.12360v1)  

---


**ABSTRACT**  
Automated construction of bilingual dictionaries using monolingual embedding spaces is a core challenge in machine translation. The end performance of these dictionaries relies upon the geometric similarity of individual spaces, i.e., their degree of isomorphism. Existing attempts aimed at controlling the relative isomorphism of different spaces fail to incorporate the impact of semantically related words in the training objective. To address this, we propose GRI that combines the distributional training objectives with attentive graph convolutions to unanimously consider the impact of semantically similar words required to define/compute the relative isomorphism of multiple spaces. Experimental evaluation shows that GRI outperforms the existing research by improving the average P@1 by a relative score of up to 63.6%. We release the codes for GRI at https://github.com/asif6827/GRI.

{{</citation>}}


### (40/143) LACMA: Language-Aligning Contrastive Learning with Meta-Actions for Embodied Instruction Following (Cheng-Fu Yang et al., 2023)

{{<citation>}}

Cheng-Fu Yang, Yen-Chun Chen, Jianwei Yang, Xiyang Dai, Lu Yuan, Yu-Chiang Frank Wang, Kai-Wei Chang. (2023)  
**LACMA: Language-Aligning Contrastive Learning with Meta-Actions for Embodied Instruction Following**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CV, cs.CL  
Keywords: Contrastive Learning, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.12344v1)  

---


**ABSTRACT**  
End-to-end Transformers have demonstrated an impressive success rate for Embodied Instruction Following when the environment has been seen in training. However, they tend to struggle when deployed in an unseen environment. This lack of generalizability is due to the agent's insensitivity to subtle changes in natural language instructions. To mitigate this issue, we propose explicitly aligning the agent's hidden states with the instructions via contrastive learning. Nevertheless, the semantic gap between high-level language instructions and the agent's low-level action space remains an obstacle. Therefore, we further introduce a novel concept of meta-actions to bridge the gap. Meta-actions are ubiquitous action patterns that can be parsed from the original action sequence. These patterns represent higher-level semantics that are intuitively aligned closer to the instructions. When meta-actions are applied as additional training signals, the agent generalizes better to unseen environments. Compared to a strong multi-modal Transformer baseline, we achieve a significant 4.5% absolute gain in success rate in unseen environments of ALFRED Embodied Instruction Following. Additional analysis shows that the contrastive objective and meta-actions are complementary in achieving the best results, and the resulting agent better aligns its states with corresponding instructions, making it more suitable for real-world embodied agents. The code is available at: https://github.com/joeyy5588/LACMA.

{{</citation>}}


### (41/143) Eliminating Reasoning via Inferring with Planning: A New Framework to Guide LLMs' Non-linear Thinking (Yongqi Tong et al., 2023)

{{<citation>}}

Yongqi Tong, Yifan Wang, Dawei Li, Sizhe Wang, Zi Lin, Simeng Han, Jingbo Shang. (2023)  
**Eliminating Reasoning via Inferring with Planning: A New Framework to Guide LLMs' Non-linear Thinking**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: NLI, Natural Language Inference, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.12342v1)  

---


**ABSTRACT**  
Chain-of-Thought(CoT) prompting and its variants explore equipping large language models (LLMs) with high-level reasoning abilities by emulating human-like linear cognition and logic. However, the human mind is complicated and mixed with both linear and nonlinear thinking. In this work, we propose \textbf{I}nferential \textbf{E}xclusion \textbf{P}rompting (IEP), a novel prompting that combines the principles of elimination and inference in order to guide LLMs to think non-linearly. IEP guides LLMs to plan and then utilize Natural Language Inference (NLI) to deduce each possible solution's entailment relation with context, commonsense, or facts, therefore yielding a broader perspective by thinking back for inferring. This forward planning and backward eliminating process allows IEP to better simulate the complex human thinking processes compared to other CoT-based methods, which only reflect linear cognitive processes. We conducted a series of empirical studies and have corroborated that IEP consistently outperforms CoT across various tasks. Additionally, we observe that integrating IEP and CoT further improves the LLMs' performance on certain tasks, highlighting the necessity of equipping LLMs with mixed logic processes. Moreover, to better evaluate comprehensive features inherent in human logic, we introduce \textbf{M}ental-\textbf{A}bility \textbf{R}easoning \textbf{B}enchmark (MARB). The benchmark comprises six novel subtasks with a total of 9,115 questions, among which 1,685 are developed with hand-crafted rationale references. We believe both \textsc{IEP} and \textsc{MARB} can serve as a promising direction for unveiling LLMs' logic and verbal reasoning abilities and drive further advancements. \textsc{MARB} will be available at ~\texttt{anonymity link} soon.

{{</citation>}}


### (42/143) The Sentiment Problem: A Critical Survey towards Deconstructing Sentiment Analysis (Pranav Narayanan Venkit et al., 2023)

{{<citation>}}

Pranav Narayanan Venkit, Mukund Srinath, Sanjana Gautam, Saranya Venkatraman, Vipul Gupta, Rebecca J. Passonneau, Shomir Wilson. (2023)  
**The Sentiment Problem: A Critical Survey towards Deconstructing Sentiment Analysis**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CY, cs-HC, cs.CL  
Keywords: Sentiment Analysis  
[Paper Link](http://arxiv.org/abs/2310.12318v1)  

---


**ABSTRACT**  
We conduct an inquiry into the sociotechnical aspects of sentiment analysis (SA) by critically examining 189 peer-reviewed papers on their applications, models, and datasets. Our investigation stems from the recognition that SA has become an integral component of diverse sociotechnical systems, exerting influence on both social and technical users. By delving into sociological and technological literature on sentiment, we unveil distinct conceptualizations of this term in domains such as finance, government, and medicine. Our study exposes a lack of explicit definitions and frameworks for characterizing sentiment, resulting in potential challenges and biases. To tackle this issue, we propose an ethics sheet encompassing critical inquiries to guide practitioners in ensuring equitable utilization of SA. Our findings underscore the significance of adopting an interdisciplinary approach to defining sentiment in SA and offer a pragmatic solution for its implementation.

{{</citation>}}


### (43/143) Document-Level Language Models for Machine Translation (Frithjof Petrick et al., 2023)

{{<citation>}}

Frithjof Petrick, Christian Herold, Pavel Petrushkov, Shahram Khadivi, Hermann Ney. (2023)  
**Document-Level Language Models for Machine Translation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model, Machine Translation  
[Paper Link](http://arxiv.org/abs/2310.12303v1)  

---


**ABSTRACT**  
Despite the known limitations, most machine translation systems today still operate on the sentence-level. One reason for this is, that most parallel training data is only sentence-level aligned, without document-level meta information available. In this work, we set out to build context-aware translation systems utilizing document-level monolingual data instead. This can be achieved by combining any existing sentence-level translation model with a document-level language model. We improve existing approaches by leveraging recent advancements in model combination. Additionally, we propose novel weighting techniques that make the system combination more flexible and significantly reduce computational overhead. In a comprehensive evaluation on four diverse translation tasks, we show that our extensions improve document-targeted scores substantially and are also computationally more efficient. However, we also find that in most scenarios, back-translation gives even better results, at the cost of having to re-train the translation system. Finally, we explore language model fusion in the light of recent advancements in large language models. Our findings suggest that there might be strong potential in utilizing large language models via model combination.

{{</citation>}}


### (44/143) Direct Neural Machine Translation with Task-level Mixture of Experts models (Isidora Chara Tourni et al., 2023)

{{<citation>}}

Isidora Chara Tourni, Subhajit Naskar. (2023)  
**Direct Neural Machine Translation with Task-level Mixture of Experts models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Machine Translation, Transformer  
[Paper Link](http://arxiv.org/abs/2310.12236v1)  

---


**ABSTRACT**  
Direct neural machine translation (direct NMT) is a type of NMT system that translates text between two non-English languages. Direct NMT systems often face limitations due to the scarcity of parallel data between non-English language pairs. Several approaches have been proposed to address this limitation, such as multilingual NMT and pivot NMT (translation between two languages via English). Task-level Mixture of expert models (Task-level MoE), an inference-efficient variation of Transformer-based models, has shown promising NMT performance for a large number of language pairs. In Task-level MoE, different language groups can use different routing strategies to optimize cross-lingual learning and inference speed. In this work, we examine Task-level MoE's applicability in direct NMT and propose a series of high-performing training and evaluation configurations, through which Task-level MoE-based direct NMT systems outperform bilingual and pivot-based models for a large number of low and high-resource direct pairs, and translation directions. Our Task-level MoE with 16 experts outperforms bilingual NMT, Pivot NMT models for 7 language pairs, while pivot-based models still performed better in 9 pairs and directions.

{{</citation>}}


### (45/143) Understanding Retrieval Augmentation for Long-Form Question Answering (Hung-Ting Chen et al., 2023)

{{<citation>}}

Hung-Ting Chen, Fangyuan Xu, Shane A. Arora, Eunsol Choi. (2023)  
**Understanding Retrieval Augmentation for Long-Form Question Answering**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Augmentation, Question Answering  
[Paper Link](http://arxiv.org/abs/2310.12150v1)  

---


**ABSTRACT**  
We present a study of retrieval-augmented language models (LMs) on long-form question answering. We analyze how retrieval augmentation impacts different LMs, by comparing answers generated from models while using the same evidence documents, and how differing quality of retrieval document set impacts the answers generated from the same LM. We study various attributes of generated answers (e.g., fluency, length, variance) with an emphasis on the attribution of generated long-form answers to in-context evidence documents. We collect human annotations of answer attribution and evaluate methods for automatically judging attribution. Our study provides new insights on how retrieval augmentation impacts long, knowledge-rich text generation of LMs. We further identify attribution patterns for long text generation and analyze the main culprits of attribution errors. Together, our analysis reveals how retrieval augmentation impacts long knowledge-rich text generation and provide directions for future work.

{{</citation>}}


### (46/143) Pseudointelligence: A Unifying Framework for Language Model Evaluation (Shikhar Murty et al., 2023)

{{<citation>}}

Shikhar Murty, Orr Paradise, Pratyusha Sharma. (2023)  
**Pseudointelligence: A Unifying Framework for Language Model Evaluation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.12135v1)  

---


**ABSTRACT**  
With large language models surpassing human performance on an increasing number of benchmarks, we must take a principled approach for targeted evaluation of model capabilities. Inspired by pseudorandomness, we propose pseudointelligence, which captures the maxim that "(perceived) intelligence lies in the eye of the beholder". That is, that claims of intelligence are meaningful only when their evaluator is taken into account. Concretely, we propose a complexity-theoretic framework of model evaluation cast as a dynamic interaction between a model and a learned evaluator. We demonstrate that this framework can be used to reason about two case studies in language model evaluation, as well as analyze existing evaluation methods.

{{</citation>}}


### (47/143) A Tale of Pronouns: Interpretability Informs Gender Bias Mitigation for Fairer Instruction-Tuned Machine Translation (Giuseppe Attanasio et al., 2023)

{{<citation>}}

Giuseppe Attanasio, Flor Miriam Plaza-del-Arco, Debora Nozza, Anne Lauscher. (2023)  
**A Tale of Pronouns: Interpretability Informs Gender Bias Mitigation for Fairer Instruction-Tuned Machine Translation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Bias, Machine Translation, NLP  
[Paper Link](http://arxiv.org/abs/2310.12127v1)  

---


**ABSTRACT**  
Recent instruction fine-tuned models can solve multiple NLP tasks when prompted to do so, with machine translation (MT) being a prominent use case. However, current research often focuses on standard performance benchmarks, leaving compelling fairness and ethical considerations behind. In MT, this might lead to misgendered translations, resulting, among other harms, in the perpetuation of stereotypes and prejudices. In this work, we address this gap by investigating whether and to what extent such models exhibit gender bias in machine translation and how we can mitigate it. Concretely, we compute established gender bias metrics on the WinoMT corpus from English to German and Spanish. We discover that IFT models default to male-inflected translations, even disregarding female occupational stereotypes. Next, using interpretability methods, we unveil that models systematically overlook the pronoun indicating the gender of a target occupation in misgendered translations. Finally, based on this finding, we propose an easy-to-implement and effective bias mitigation solution based on few-shot learning that leads to significantly fairer translations.

{{</citation>}}


### (48/143) Harnessing Dataset Cartography for Improved Compositional Generalization in Transformers (Osman Batur İnce et al., 2023)

{{<citation>}}

Osman Batur İnce, Tanin Zeraati, Semih Yagcioglu, Yadollah Yaghoobzadeh, Erkut Erdem, Aykut Erdem. (2023)  
**Harnessing Dataset Cartography for Improved Compositional Generalization in Transformers**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.12118v1)  

---


**ABSTRACT**  
Neural networks have revolutionized language modeling and excelled in various downstream tasks. However, the extent to which these models achieve compositional generalization comparable to human cognitive abilities remains a topic of debate. While existing approaches in the field have mainly focused on novel architectures and alternative learning paradigms, we introduce a pioneering method harnessing the power of dataset cartography (Swayamdipta et al., 2020). By strategically identifying a subset of compositional generalization data using this approach, we achieve a remarkable improvement in model accuracy, yielding enhancements of up to 10% on CFQ and COGS datasets. Notably, our technique incorporates dataset cartography as a curriculum learning criterion, eliminating the need for hyperparameter tuning while consistently achieving superior performance. Our findings highlight the untapped potential of dataset cartography in unleashing the full capabilities of compositional generalization within Transformer models. Our code is available at https://github.com/cyberiada/cartography-for-compositionality.

{{</citation>}}


### (49/143) Position Interpolation Improves ALiBi Extrapolation (Faisal Al-Khateeb et al., 2023)

{{<citation>}}

Faisal Al-Khateeb, Nolan Dey, Daria Soboleva, Joel Hestness. (2023)  
**Position Interpolation Improves ALiBi Extrapolation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Attention, Bias  
[Paper Link](http://arxiv.org/abs/2310.13017v1)  

---


**ABSTRACT**  
Linear position interpolation helps pre-trained models using rotary position embeddings (RoPE) to extrapolate to longer sequence lengths. We propose using linear position interpolation to extend the extrapolation range of models using Attention with Linear Biases (ALiBi). We find position interpolation significantly improves extrapolation capability on upstream language modelling and downstream summarization and retrieval tasks.

{{</citation>}}


### (50/143) Unveiling the Siren's Song: Towards Reliable Fact-Conflicting Hallucination Detection (Xiang Chen et al., 2023)

{{<citation>}}

Xiang Chen, Duanzheng Song, Honghao Gui, Chengxi Wang, Ningyu Zhang, Fei Huang, Chengfei Lv, Dan Zhang, Huajun Chen. (2023)  
**Unveiling the Siren's Song: Towards Reliable Fact-Conflicting Hallucination Detection**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CV, cs-IR, cs-LG, cs.CL  
Keywords: ChatGPT, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2310.12086v1)  

---


**ABSTRACT**  
Large Language Models (LLMs), such as ChatGPT/GPT-4, have garnered widespread attention owing to their myriad of practical applications, yet their adoption has been constrained by issues of fact-conflicting hallucinations across web platforms. The assessment of factuality in text, produced by LLMs, remains inadequately explored, extending not only to the judgment of vanilla facts but also encompassing the evaluation of factual errors emerging in complex inferential tasks like multi-hop, and etc. In response, we introduce FactCHD, a fact-conflicting hallucination detection benchmark meticulously designed for LLMs. Functioning as a pivotal tool in evaluating factuality within "Query-Respons" contexts, our benchmark assimilates a large-scale dataset, encapsulating a broad spectrum of factuality patterns, such as vanilla, multi-hops, comparison, and set-operation patterns. A distinctive feature of our benchmark is its incorporation of fact-based chains of evidence, thereby facilitating comprehensive and conducive factual reasoning throughout the assessment process. We evaluate multiple LLMs, demonstrating the effectiveness of the benchmark and current methods fall short of faithfully detecting factual errors. Furthermore, we present TRUTH-TRIANGULATOR that synthesizes reflective considerations by tool-enhanced ChatGPT and LoRA-tuning based on Llama2, aiming to yield more credible detection through the amalgamation of predictive results and evidence. The benchmark dataset and source code will be made available in https://github.com/zjunlp/FactCHD.

{{</citation>}}


### (51/143) Towards Safer Operations: An Expert-involved Dataset of High-Pressure Gas Incidents for Preventing Future Failures (Shumpei Inoue et al., 2023)

{{<citation>}}

Shumpei Inoue, Minh-Tien Nguyen, Hiroki Mizokuchi, Tuan-Anh D. Nguyen, Huu-Hiep Nguyen, Dung Tien Le. (2023)  
**Towards Safer Operations: An Expert-involved Dataset of High-Pressure Gas Incidents for Preventing Future Failures**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, NLP  
[Paper Link](http://arxiv.org/abs/2310.12074v2)  

---


**ABSTRACT**  
This paper introduces a new IncidentAI dataset for safety prevention. Different from prior corpora that usually contain a single task, our dataset comprises three tasks: named entity recognition, cause-effect extraction, and information retrieval. The dataset is annotated by domain experts who have at least six years of practical experience as high-pressure gas conservation managers. We validate the contribution of the dataset in the scenario of safety prevention. Preliminary results on the three tasks show that NLP techniques are beneficial for analyzing incident reports to prevent future failures. The dataset facilitates future research in NLP and incident management communities. The access to the dataset is also provided (the IncidentAI dataset is available at: https://github.com/Cinnamon/incident-ai-dataset).

{{</citation>}}


### (52/143) SPEED: Speculative Pipelined Execution for Efficient Decoding (Coleman Hooper et al., 2023)

{{<citation>}}

Coleman Hooper, Sehoon Kim, Hiva Mohammadzadeh, Hasan Genc, Kurt Keutzer, Amir Gholami, Sophia Shao. (2023)  
**SPEED: Speculative Pipelined Execution for Efficient Decoding**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Natural Language Processing, Transformer  
[Paper Link](http://arxiv.org/abs/2310.12072v1)  

---


**ABSTRACT**  
Generative Large Language Models (LLMs) based on the Transformer architecture have recently emerged as a dominant foundation model for a wide range of Natural Language Processing tasks. Nevertheless, their application in real-time scenarios has been highly restricted due to the significant inference latency associated with these models. This is particularly pronounced due to the autoregressive nature of generative LLM inference, where tokens are generated sequentially since each token depends on all previous output tokens. It is therefore challenging to achieve any token-level parallelism, making inference extremely memory-bound. In this work, we propose SPEED, which improves inference efficiency by speculatively executing multiple future tokens in parallel with the current token using predicted values based on early-layer hidden states. For Transformer decoders that employ parameter sharing, the memory operations for the tokens executing in parallel can be amortized, which allows us to accelerate generative LLM inference. We demonstrate the efficiency of our method in terms of latency reduction relative to model accuracy and demonstrate how speculation allows for training deeper decoders with parameter sharing with minimal runtime overhead.

{{</citation>}}


### (53/143) Evaluating the Symbol Binding Ability of Large Language Models for Multiple-Choice Questions in Vietnamese General Education (Duc-Vu Nguyen et al., 2023)

{{<citation>}}

Duc-Vu Nguyen, Quoc-Nam Nguyen. (2023)  
**Evaluating the Symbol Binding Ability of Large Language Models for Multiple-Choice Questions in Vietnamese General Education**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BLOOM, ChatGPT, GPT, GPT-3.5, GPT-4, LLaMA, Language Model, NLP, QA  
[Paper Link](http://arxiv.org/abs/2310.12059v2)  

---


**ABSTRACT**  
In this paper, we evaluate the ability of large language models (LLMs) to perform multiple choice symbol binding (MCSB) for multiple choice question answering (MCQA) tasks in zero-shot, one-shot, and few-shot settings. We focus on Vietnamese, with fewer challenging MCQA datasets than in English. The two existing datasets, ViMMRC 1.0 and ViMMRC 2.0, focus on literature. Recent research in Vietnamese natural language processing (NLP) has focused on the Vietnamese National High School Graduation Examination (VNHSGE) from 2019 to 2023 to evaluate ChatGPT. However, these studies have mainly focused on how ChatGPT solves the VNHSGE step by step. We aim to create a novel and high-quality dataset by providing structured guidelines for typing LaTeX formulas for mathematics, physics, chemistry, and biology. This dataset can be used to evaluate the MCSB ability of LLMs and smaller language models (LMs) because it is typed in a strict LaTeX style. We focus on predicting the character (A, B, C, or D) that is the most likely answer to a question, given the context of the question. Our evaluation of six well-known LLMs, namely BLOOMZ-7.1B-MT, LLaMA-2-7B, LLaMA-2-70B, GPT-3, GPT-3.5, and GPT-4.0, on the ViMMRC 1.0 and ViMMRC 2.0 benchmarks and our proposed dataset shows promising results on the MCSB ability of LLMs for Vietnamese. The dataset is available for research purposes only.

{{</citation>}}


### (54/143) Concept-Guided Chain-of-Thought Prompting for Pairwise Comparison Scaling of Texts with Large Language Models (Patrick Y. Wu et al., 2023)

{{<citation>}}

Patrick Y. Wu, Jonathan Nagler, Joshua A. Tucker, Solomon Messing. (2023)  
**Concept-Guided Chain-of-Thought Prompting for Pairwise Comparison Scaling of Texts with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs.CL  
Keywords: BERT, Language Model, Twitter  
[Paper Link](http://arxiv.org/abs/2310.12049v1)  

---


**ABSTRACT**  
Existing text scaling methods often require a large corpus, struggle with short texts, or require labeled data. We develop a text scaling method that leverages the pattern recognition capabilities of generative large language models (LLMs). Specifically, we propose concept-guided chain-of-thought (CGCoT), which uses prompts designed to summarize ideas and identify target parties in texts to generate concept-specific breakdowns, in many ways similar to guidance for human coder content analysis. CGCoT effectively shifts pairwise text comparisons from a reasoning problem to a pattern recognition problem. We then pairwise compare concept-specific breakdowns using an LLM. We use the results of these pairwise comparisons to estimate a scale using the Bradley-Terry model. We use this approach to scale affective speech on Twitter. Our measures correlate more strongly with human judgments than alternative approaches like Wordfish. Besides a small set of pilot data to develop the CGCoT prompts, our measures require no additional labeled data and produce binary predictions comparable to a RoBERTa-Large model fine-tuned on thousands of human-labeled tweets. We demonstrate how combining substantive knowledge with LLMs can create state-of-the-art measures of abstract concepts.

{{</citation>}}


### (55/143) CORE: A Few-Shot Company Relation Classification Dataset for Robust Domain Adaptation (Philipp Borchert et al., 2023)

{{<citation>}}

Philipp Borchert, Jochen De Weerdt, Kristof Coussement, Arno De Caigny, Marie-Francine Moens. (2023)  
**CORE: A Few-Shot Company Relation Classification Dataset for Robust Domain Adaptation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2310.12024v1)  

---


**ABSTRACT**  
We introduce CORE, a dataset for few-shot relation classification (RC) focused on company relations and business entities. CORE includes 4,708 instances of 12 relation types with corresponding textual evidence extracted from company Wikipedia pages. Company names and business entities pose a challenge for few-shot RC models due to the rich and diverse information associated with them. For example, a company name may represent the legal entity, products, people, or business divisions depending on the context. Therefore, deriving the relation type between entities is highly dependent on textual context. To evaluate the performance of state-of-the-art RC models on the CORE dataset, we conduct experiments in the few-shot domain adaptation setting. Our results reveal substantial performance gaps, confirming that models trained on different domains struggle to adapt to CORE. Interestingly, we find that models trained on CORE showcase improved out-of-domain performance, which highlights the importance of high-quality data for robust domain adaptation. Specifically, the information richness embedded in business entities allows models to focus on contextual nuances, reducing their reliance on superficial clues such as relation-specific verbs. In addition to the dataset, we provide relevant code snippets to facilitate reproducibility and encourage further research in the field.

{{</citation>}}


### (56/143) Gold: A Global and Local-aware Denoising Framework for Commonsense Knowledge Graph Noise Detection (Zheye Deng et al., 2023)

{{<citation>}}

Zheye Deng, Weiqi Wang, Zhaowei Wang, Xin Liu, Yangqiu Song. (2023)  
**Gold: A Global and Local-aware Denoising Framework for Commonsense Knowledge Graph Noise Detection**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Commonsense Knowledge, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2310.12011v1)  

---


**ABSTRACT**  
Commonsense Knowledge Graphs (CSKGs) are crucial for commonsense reasoning, yet constructing them through human annotations can be costly. As a result, various automatic methods have been proposed to construct CSKG with larger semantic coverage. However, these unsupervised approaches introduce spurious noise that can lower the quality of the resulting CSKG, which cannot be tackled easily by existing denoising algorithms due to the unique characteristics of nodes and structures in CSKGs. To address this issue, we propose Gold (Global and Local-aware Denoising), a denoising framework for CSKGs that incorporates entity semantic information, global rules, and local structural information from the CSKG. Experiment results demonstrate that Gold outperforms all baseline methods in noise detection tasks on synthetic noisy CSKG benchmarks. Furthermore, we show that denoising a real-world CSKG is effective and even benefits the downstream zero-shot commonsense question-answering task.

{{</citation>}}


### (57/143) Multi-view Contrastive Learning for Entity Typing over Knowledge Graphs (Zhiwei Hu et al., 2023)

{{<citation>}}

Zhiwei Hu, Víctor Gutiérrez-Basulto, Zhiliang Xiang, Ru Li, Jeff Z. Pan. (2023)  
**Multi-view Contrastive Learning for Entity Typing over Knowledge Graphs**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Contrastive Learning, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2310.12008v1)  

---


**ABSTRACT**  
Knowledge graph entity typing (KGET) aims at inferring plausible types of entities in knowledge graphs. Existing approaches to KGET focus on how to better encode the knowledge provided by the neighbors and types of an entity into its representation. However, they ignore the semantic knowledge provided by the way in which types can be clustered together. In this paper, we propose a novel method called Multi-view Contrastive Learning for knowledge graph Entity Typing (MCLET), which effectively encodes the coarse-grained knowledge provided by clusters into entity and type embeddings. MCLET is composed of three modules: i) Multi-view Generation and Encoder module, which encodes structured information from entity-type, entity-cluster and cluster-type views; ii) Cross-view Contrastive Learning module, which encourages different views to collaboratively improve view-specific representations of entities and types; iii) Entity Typing Prediction module, which integrates multi-head attention and a Mixture-of-Experts strategy to infer missing entity types. Extensive experiments show the strong performance of MCLET compared to the state-of-the-art

{{</citation>}}


### (58/143) InfoDiffusion: Information Entropy Aware Diffusion Process for Non-Autoregressive Text Generation (Renzhi Wang et al., 2023)

{{<citation>}}

Renzhi Wang, Jing Li, Piji Li. (2023)  
**InfoDiffusion: Information Entropy Aware Diffusion Process for Non-Autoregressive Text Generation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Text Generation  
[Paper Link](http://arxiv.org/abs/2310.11976v1)  

---


**ABSTRACT**  
Diffusion models have garnered considerable interest in the field of text generation. Several studies have explored text diffusion models with different structures and applied them to various tasks, including named entity recognition and summarization. However, there exists a notable disparity between the "easy-first" text generation process of current diffusion models and the "keyword-first" natural text generation process of humans, which has received limited attention. To bridge this gap, we propose InfoDiffusion, a non-autoregressive text diffusion model. Our approach introduces a "keyinfo-first" generation strategy and incorporates a noise schedule based on the amount of text information. In addition, InfoDiffusion combines self-conditioning with a newly proposed partially noising model structure. Experimental results show that InfoDiffusion outperforms the baseline model in terms of generation quality and diversity, as well as exhibiting higher sampling efficiency.

{{</citation>}}


### (59/143) AMR Parsing with Causal Hierarchical Attention and Pointers (Chao Lou et al., 2023)

{{<citation>}}

Chao Lou, Kewei Tu. (2023)  
**AMR Parsing with Causal Hierarchical Attention and Pointers**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2310.11964v1)  

---


**ABSTRACT**  
Translation-based AMR parsers have recently gained popularity due to their simplicity and effectiveness. They predict linearized graphs as free texts, avoiding explicit structure modeling. However, this simplicity neglects structural locality in AMR graphs and introduces unnecessary tokens to represent coreferences. In this paper, we introduce new target forms of AMR parsing and a novel model, CHAP, which is equipped with causal hierarchical attention and the pointer mechanism, enabling the integration of structures into the Transformer decoder. We empirically explore various alternative modeling options. Experiments show that our model outperforms baseline models on four out of five benchmarks in the setting of no additional data.

{{</citation>}}


### (60/143) Fast Multipole Attention: A Divide-and-Conquer Attention Mechanism for Long Sequences (Yanming Kang et al., 2023)

{{<citation>}}

Yanming Kang, Giang Tran, Hans De Sterck. (2023)  
**Fast Multipole Attention: A Divide-and-Conquer Attention Mechanism for Long Sequences**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2310.11960v2)  

---


**ABSTRACT**  
Transformer-based models have achieved state-of-the-art performance in many areas. However, the quadratic complexity of self-attention with respect to the input length hinders the applicability of Transformer-based models to long sequences. To address this, we present Fast Multipole Attention, a new attention mechanism that uses a divide-and-conquer strategy to reduce the time and memory complexity of attention for sequences of length $n$ from $\mathcal{O}(n^2)$ to $\mathcal{O}(n \log n)$ or $O(n)$, while retaining a global receptive field. The hierarchical approach groups queries, keys, and values into $\mathcal{O}( \log n)$ levels of resolution, where groups at greater distances are increasingly larger in size and the weights to compute group quantities are learned. As such, the interaction between tokens far from each other is considered in lower resolution in an efficient hierarchical manner. The overall complexity of Fast Multipole Attention is $\mathcal{O}(n)$ or $\mathcal{O}(n \log n)$, depending on whether the queries are down-sampled or not. This multi-level divide-and-conquer strategy is inspired by fast summation methods from $n$-body physics and the Fast Multipole Method. We perform evaluation on autoregressive and bidirectional language modeling tasks and compare our Fast Multipole Attention model with other efficient attention variants on medium-size datasets. We find empirically that the Fast Multipole Transformer performs much better than other efficient transformers in terms of memory size and accuracy. The Fast Multipole Attention mechanism has the potential to empower large language models with much greater sequence lengths, taking the full context into account in an efficient, naturally hierarchical manner during training and when generating long sequences.

{{</citation>}}


### (61/143) MusicAgent: An AI Agent for Music Understanding and Generation with Large Language Models (Dingyao Yu et al., 2023)

{{<citation>}}

Dingyao Yu, Kaitao Song, Peiling Lu, Tianyu He, Xu Tan, Wei Ye, Shikun Zhang, Jiang Bian. (2023)  
**MusicAgent: An AI Agent for Music Understanding and Generation with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-MM, cs.CL, eess-AS  
Keywords: AI, ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.11954v1)  

---


**ABSTRACT**  
AI-empowered music processing is a diverse field that encompasses dozens of tasks, ranging from generation tasks (e.g., timbre synthesis) to comprehension tasks (e.g., music classification). For developers and amateurs, it is very difficult to grasp all of these task to satisfy their requirements in music processing, especially considering the huge differences in the representations of music data and the model applicability across platforms among various tasks. Consequently, it is necessary to build a system to organize and integrate these tasks, and thus help practitioners to automatically analyze their demand and call suitable tools as solutions to fulfill their requirements. Inspired by the recent success of large language models (LLMs) in task automation, we develop a system, named MusicAgent, which integrates numerous music-related tools and an autonomous workflow to address user requirements. More specifically, we build 1) toolset that collects tools from diverse sources, including Hugging Face, GitHub, and Web API, etc. 2) an autonomous workflow empowered by LLMs (e.g., ChatGPT) to organize these tools and automatically decompose user requests into multiple sub-tasks and invoke corresponding music tools. The primary goal of this system is to free users from the intricacies of AI-music tools, enabling them to concentrate on the creative aspect. By granting users the freedom to effortlessly combine tools, the system offers a seamless and enriching music experience.

{{</citation>}}


### (62/143) Grounded and Well-rounded: A Methodological Approach to the Study of Cross-modal and Cross-lingual Grounding (Timothee Mickus et al., 2023)

{{<citation>}}

Timothee Mickus, Elaine Zosa, Denis Paperno. (2023)  
**Grounded and Well-rounded: A Methodological Approach to the Study of Cross-modal and Cross-lingual Grounding**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2310.11938v1)  

---


**ABSTRACT**  
Grounding has been argued to be a crucial component towards the development of more complete and truly semantically competent artificial intelligence systems. Literature has divided into two camps: While some argue that grounding allows for qualitatively different generalizations, others believe it can be compensated by mono-modal data quantity. Limited empirical evidence has emerged for or against either position, which we argue is due to the methodological challenges that come with studying grounding and its effects on NLP systems.   In this paper, we establish a methodological framework for studying what the effects are - if any - of providing models with richer input sources than text-only. The crux of it lies in the construction of comparable samples of populations of models trained on different input modalities, so that we can tease apart the qualitative effects of different input sources from quantifiable model performances. Experiments using this framework reveal qualitative differences in model behavior between cross-modally grounded, cross-lingually grounded, and ungrounded models, which we measure both at a global dataset level as well as for specific word representations, depending on how concrete their semantics is.

{{</citation>}}


### (63/143) Investigating semantic subspaces of Transformer sentence embeddings through linear structural probing (Dmitry Nikolaev et al., 2023)

{{<citation>}}

Dmitry Nikolaev, Sebastian Padó. (2023)  
**Investigating semantic subspaces of Transformer sentence embeddings through linear structural probing**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP, Transformer  
[Paper Link](http://arxiv.org/abs/2310.11923v1)  

---


**ABSTRACT**  
The question of what kinds of linguistic information are encoded in different layers of Transformer-based language models is of considerable interest for the NLP community. Existing work, however, has overwhelmingly focused on word-level representations and encoder-only language models with the masked-token training objective. In this paper, we present experiments with semantic structural probing, a method for studying sentence-level representations via finding a subspace of the embedding space that provides suitable task-specific pairwise distances between data-points. We apply our method to language models from different families (encoder-only, decoder-only, encoder-decoder) and of different sizes in the context of two tasks, semantic textual similarity and natural-language inference. We find that model families differ substantially in their performance and layer dynamics, but that the results are largely model-size invariant.

{{</citation>}}


### (64/143) A Benchmark for Semi-Inductive Link Prediction in Knowledge Graphs (Adrian Kochsiek et al., 2023)

{{<citation>}}

Adrian Kochsiek, Rainer Gemulla. (2023)  
**A Benchmark for Semi-Inductive Link Prediction in Knowledge Graphs**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2310.11917v1)  

---


**ABSTRACT**  
Semi-inductive link prediction (LP) in knowledge graphs (KG) is the task of predicting facts for new, previously unseen entities based on context information. Although new entities can be integrated by retraining the model from scratch in principle, such an approach is infeasible for large-scale KGs, where retraining is expensive and new entities may arise frequently. In this paper, we propose and describe a large-scale benchmark to evaluate semi-inductive LP models. The benchmark is based on and extends Wikidata5M: It provides transductive, k-shot, and 0-shot LP tasks, each varying the available information from (i) only KG structure, to (ii) including textual mentions, and (iii) detailed descriptions of the entities. We report on a small study of recent approaches and found that semi-inductive LP performance is far from transductive performance on long-tail entities throughout all experiments. The benchmark provides a test bed for further research into integrating context and textual information in semi-inductive LP models.

{{</citation>}}


### (65/143) Rather a Nurse than a Physician -- Contrastive Explanations under Investigation (Oliver Eberle et al., 2023)

{{<citation>}}

Oliver Eberle, Ilias Chalkidis, Laura Cabello, Stephanie Brandl. (2023)  
**Rather a Nurse than a Physician -- Contrastive Explanations under Investigation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, T5  
[Paper Link](http://arxiv.org/abs/2310.11906v1)  

---


**ABSTRACT**  
Contrastive explanations, where one decision is explained in contrast to another, are supposed to be closer to how humans explain a decision than non-contrastive explanations, where the decision is not necessarily referenced to an alternative. This claim has never been empirically validated. We analyze four English text-classification datasets (SST2, DynaSent, BIOS and DBpedia-Animals). We fine-tune and extract explanations from three different models (RoBERTa, GTP-2, and T5), each in three different sizes and apply three post-hoc explainability methods (LRP, GradientxInput, GradNorm). We furthermore collect and release human rationale annotations for a subset of 100 samples from the BIOS dataset for contrastive and non-contrastive settings. A cross-comparison between model-based rationales and human annotations, both in contrastive and non-contrastive settings, yields a high agreement between the two settings for models as well as for humans. Moreover, model-based explanations computed in both settings align equally well with human rationales. Thus, we empirically find that humans do not necessarily explain in a contrastive manner.9 pages, long paper at ACL 2022 proceedings.

{{</citation>}}


### (66/143) From Dissonance to Insights: Dissecting Disagreements in Rationale Construction for Case Outcome Classification (Shanshan Xu et al., 2023)

{{<citation>}}

Shanshan Xu, Santosh T. Y. S. S, Oana Ichim, Isabella Risini, Barbara Plank, Matthias Grabmair. (2023)  
**From Dissonance to Insights: Dissecting Disagreements in Rationale Construction for Case Outcome Classification**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2310.11878v3)  

---


**ABSTRACT**  
In legal NLP, Case Outcome Classification (COC) must not only be accurate but also trustworthy and explainable. Existing work in explainable COC has been limited to annotations by a single expert. However, it is well-known that lawyers may disagree in their assessment of case facts. We hence collect a novel dataset RAVE: Rationale Variation in ECHR1, which is obtained from two experts in the domain of international human rights law, for whom we observe weak agreement. We study their disagreements and build a two-level task-independent taxonomy, supplemented with COC-specific subcategories. To our knowledge, this is the first work in the legal NLP that focuses on human label variation. We quantitatively assess different taxonomy categories and find that disagreements mainly stem from underspecification of the legal context, which poses challenges given the typically limited granularity and noise in COC metadata. We further assess the explainablility of SOTA COC models on RAVE and observe limited agreement between models and experts. Overall, our case study reveals hitherto underappreciated complexities in creating benchmark datasets in legal NLP that revolve around identifying aspects of a case's facts supposedly relevant to its outcome.

{{</citation>}}


### (67/143) The Curious Case of Hallucinatory Unanswerablity: Finding Truths in the Hidden States of Over-Confident Large Language Models (Aviv Slobodkin et al., 2023)

{{<citation>}}

Aviv Slobodkin, Omer Goldman, Avi Caciularu, Ido Dagan, Shauli Ravfogel. (2023)  
**The Curious Case of Hallucinatory Unanswerablity: Finding Truths in the Hidden States of Over-Confident Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.11877v1)  

---


**ABSTRACT**  
Large language models (LLMs) have been shown to possess impressive capabilities, while also raising crucial concerns about the faithfulness of their responses. A primary issue arising in this context is the management of unanswerable queries by LLMs, which often results in hallucinatory behavior, due to overconfidence. In this paper, we explore the behavior of LLMs when presented with unanswerable queries. We ask: do models \textbf{represent} the fact that the question is unanswerable when generating a hallucinatory answer? Our results show strong indications that such models encode the answerability of an input query, with the representation of the first decoded token often being a strong indicator. These findings shed new light on the spatial organization within the latent representations of LLMs, unveiling previously unexplored facets of these models. Moreover, they pave the way for the development of improved decoding techniques with better adherence to factual generation, particularly in scenarios where query unanswerability is a concern.

{{</citation>}}


### (68/143) AI Nushu: An Exploration of Language Emergence in Sisterhood -Through the Lens of Computational Linguistics (Yuqian Sun et al., 2023)

{{<citation>}}

Yuqian Sun, Yuying Tang, Ze Gao, Zhijun Pan, Chuyan Xu, Yurou Chen, Kejiang Qian, Zhigang Wang, Tristan Braud, Chang Hee Lee, Ali Asadipour. (2023)  
**AI Nushu: An Exploration of Language Emergence in Sisterhood -Through the Lens of Computational Linguistics**  

---
Primary Category: cs.CL  
Categories: 14J60 (Primary) 14F05, 14J26 (Secondary), F-2-2; I-2-7, cs-AI, cs-CL, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.11870v1)  

---


**ABSTRACT**  
This paper presents "AI Nushu," an emerging language system inspired by Nushu (women's scripts), the unique language created and used exclusively by ancient Chinese women who were thought to be illiterate under a patriarchal society. In this interactive installation, two artificial intelligence (AI) agents are trained in the Chinese dictionary and the Nushu corpus. By continually observing their environment and communicating, these agents collaborate towards creating a standard writing system to encode Chinese. It offers an artistic interpretation of the creation of a non-western script from a computational linguistics perspective, integrating AI technology with Chinese cultural heritage and a feminist viewpoint.

{{</citation>}}


### (69/143) Improving Long Document Topic Segmentation Models With Enhanced Coherence Modeling (Hai Yu et al., 2023)

{{<citation>}}

Hai Yu, Chong Deng, Qinglin Zhang, Jiaqing Liu, Qian Chen, Wen Wang. (2023)  
**Improving Long Document Topic Segmentation Models With Enhanced Coherence Modeling**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Semantic Similarity  
[Paper Link](http://arxiv.org/abs/2310.11772v3)  

---


**ABSTRACT**  
Topic segmentation is critical for obtaining structured documents and improving downstream tasks such as information retrieval. Due to its ability of automatically exploring clues of topic shift from abundant labeled data, recent supervised neural models have greatly promoted the development of long document topic segmentation, but leaving the deeper relationship between coherence and topic segmentation underexplored. Therefore, this paper enhances the ability of supervised models to capture coherence from both logical structure and semantic similarity perspectives to further improve the topic segmentation performance, proposing Topic-aware Sentence Structure Prediction (TSSP) and Contrastive Semantic Similarity Learning (CSSL). Specifically, the TSSP task is proposed to force the model to comprehend structural information by learning the original relations between adjacent sentences in a disarrayed document, which is constructed by jointly disrupting the original document at topic and sentence levels. Moreover, we utilize inter- and intra-topic information to construct contrastive samples and design the CSSL objective to ensure that the sentences representations in the same topic have higher similarity, while those in different topics are less similar. Extensive experiments show that the Longformer with our approach significantly outperforms old state-of-the-art (SOTA) methods. Our approach improve $F_1$ of old SOTA by 3.42 (73.74 -> 77.16) and reduces $P_k$ by 1.11 points (15.0 -> 13.89) on WIKI-727K and achieves an average relative reduction of 4.3% on $P_k$ on WikiSection. The average relative $P_k$ drop of 8.38% on two out-of-domain datasets also demonstrates the robustness of our approach.

{{</citation>}}


### (70/143) Annotated Job Ads with Named Entity Recognition (Felix Stollenwerk et al., 2023)

{{<citation>}}

Felix Stollenwerk, Niklas Fastlund, Anna Nyqvist, Joey Öhman. (2023)  
**Annotated Job Ads with Named Entity Recognition**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, NER, Named Entity Recognition  
[Paper Link](http://arxiv.org/abs/2310.11769v1)  

---


**ABSTRACT**  
We have trained a named entity recognition (NER) model that screens Swedish job ads for different kinds of useful information (e.g. skills required from a job seeker). It was obtained by fine-tuning KB-BERT. The biggest challenge we faced was the creation of a labelled dataset, which required manual annotation. This paper gives an overview of the methods we employed to make the annotation process more efficient and to ensure high quality data. We also report on the performance of the resulting model.

{{</citation>}}


### (71/143) A Comprehensive Evaluation of Large Language Models on Legal Judgment Prediction (Ruihao Shui et al., 2023)

{{<citation>}}

Ruihao Shui, Yixin Cao, Xiang Wang, Tat-Seng Chua. (2023)  
**A Comprehensive Evaluation of Large Language Models on Legal Judgment Prediction**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model, Legal  
[Paper Link](http://arxiv.org/abs/2310.11761v1)  

---


**ABSTRACT**  
Large language models (LLMs) have demonstrated great potential for domain-specific applications, such as the law domain. However, recent disputes over GPT-4's law evaluation raise questions concerning their performance in real-world legal tasks. To systematically investigate their competency in the law, we design practical baseline solutions based on LLMs and test on the task of legal judgment prediction. In our solutions, LLMs can work alone to answer open questions or coordinate with an information retrieval (IR) system to learn from similar cases or solve simplified multi-choice questions. We show that similar cases and multi-choice options, namely label candidates, included in prompts can help LLMs recall domain knowledge that is critical for expertise legal reasoning. We additionally present an intriguing paradox wherein an IR system surpasses the performance of LLM+IR due to limited gains acquired by weaker LLMs from powerful IR systems. In such cases, the role of LLMs becomes redundant. Our evaluation pipeline can be easily extended into other tasks to facilitate evaluations in other domains. Code is available at https://github.com/srhthu/LM-CompEval-Legal

{{</citation>}}


### (72/143) Quantify Health-Related Atomic Knowledge in Chinese Medical Large Language Models: A Computational Analysis (Yaxin Fan et al., 2023)

{{<citation>}}

Yaxin Fan, Feng Jiang, Peifeng Li, Haizhou Li. (2023)  
**Quantify Health-Related Atomic Knowledge in Chinese Medical Large Language Models: A Computational Analysis**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2310.11722v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have the potential to revolutionize the way users self-diagnose through search engines by offering direct and efficient suggestions. Recent studies primarily focused on the quality of LLMs evaluated by GPT-4 or their ability to pass medical exams, no studies have quantified the extent of health-related atomic knowledge stored in LLMs' memory, which is the basis of LLMs to provide more factual suggestions. In this paper, we first constructed a benchmark, including the most common types of atomic knowledge in user self-diagnosis queries, with 17 atomic types and a total of 14, 048 pieces of atomic knowledge. Then, we evaluated both generic and specialized LLMs on the benchmark. The experimental results showcased that generic LLMs perform better than specialized LLMs in terms of atomic knowledge and instruction-following ability. Error analysis revealed that both generic and specialized LLMs are sycophantic, e.g., always catering to users' claims when it comes to unknown knowledge. Besides, generic LLMs showed stronger safety, which can be learned by specialized LLMs through distilled data. We further explored different types of data commonly adopted for fine-tuning specialized LLMs, i.e., real-world, semi-distilled, and distilled data, and found that distilled data can benefit LLMs most.

{{</citation>}}


### (73/143) Chain-of-Thought Tuning: Masked Language Models can also Think Step By Step in Natural Language Understanding (Caoyun Fan et al., 2023)

{{<citation>}}

Caoyun Fan, Jidong Tian, Yitian Li, Wenqing Chen, Hao He, Yaohui Jin. (2023)  
**Chain-of-Thought Tuning: Masked Language Models can also Think Step By Step in Natural Language Understanding**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Language Model, NLU, Natural Language Understanding  
[Paper Link](http://arxiv.org/abs/2310.11721v1)  

---


**ABSTRACT**  
Chain-of-Thought (CoT) is a technique that guides Large Language Models (LLMs) to decompose complex tasks into multi-step reasoning through intermediate steps in natural language form. Briefly, CoT enables LLMs to think step by step. However, although many Natural Language Understanding (NLU) tasks also require thinking step by step, LLMs perform less well than small-scale Masked Language Models (MLMs). To migrate CoT from LLMs to MLMs, we propose Chain-of-Thought Tuning (CoTT), a two-step reasoning framework based on prompt tuning, to implement step-by-step thinking for MLMs on NLU tasks. From the perspective of CoT, CoTT's two-step framework enables MLMs to implement task decomposition; CoTT's prompt tuning allows intermediate steps to be used in natural language form. Thereby, the success of CoT can be extended to NLU tasks through MLMs. To verify the effectiveness of CoTT, we conduct experiments on two NLU tasks: hierarchical classification and relation extraction, and the results show that CoTT outperforms baselines and achieves state-of-the-art performance.

{{</citation>}}


### (74/143) Reflection-Tuning: Data Recycling Improves LLM Instruction-Tuning (Ming Li et al., 2023)

{{<citation>}}

Ming Li, Lichang Chen, Jiuhai Chen, Shwai He, Heng Huang, Jiuxiang Gu, Tianyi Zhou. (2023)  
**Reflection-Tuning: Data Recycling Improves LLM Instruction-Tuning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.11716v1)  

---


**ABSTRACT**  
Recent advancements in Large Language Models (LLMs) have expanded the horizons of natural language understanding and generation. Notably, the output control and alignment with the input of LLMs can be refined through instruction tuning. However, as highlighted in several studies, low-quality data in the training set are usually detrimental to instruction tuning, resulting in inconsistent or even misleading LLM outputs. We propose a novel method, termed "reflection-tuning," which addresses the problem by self-improvement and judging capabilities of LLMs. This approach utilizes an oracle LLM to recycle the original training data by introspecting and enhancing the quality of instructions and responses in the data. Extensive experiments on widely used evaluation benchmarks show that LLMs trained with our recycled data outperform those trained with existing datasets in various benchmarks.

{{</citation>}}


### (75/143) Enhancing Low-resource Fine-grained Named Entity Recognition by Leveraging Coarse-grained Datasets (Su Ah Lee et al., 2023)

{{<citation>}}

Su Ah Lee, Seokjin Oh, Woohwan Jung. (2023)  
**Enhancing Low-resource Fine-grained Named Entity Recognition by Leveraging Coarse-grained Datasets**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: NER, Named Entity Recognition  
[Paper Link](http://arxiv.org/abs/2310.11715v1)  

---


**ABSTRACT**  
Named Entity Recognition (NER) frequently suffers from the problem of insufficient labeled data, particularly in fine-grained NER scenarios. Although $K$-shot learning techniques can be applied, their performance tends to saturate when the number of annotations exceeds several tens of labels. To overcome this problem, we utilize existing coarse-grained datasets that offer a large number of annotations. A straightforward approach to address this problem is pre-finetuning, which employs coarse-grained data for representation learning. However, it cannot directly utilize the relationships between fine-grained and coarse-grained entities, although a fine-grained entity type is likely to be a subcategory of a coarse-grained entity type. We propose a fine-grained NER model with a Fine-to-Coarse(F2C) mapping matrix to leverage the hierarchical structure explicitly. In addition, we present an inconsistency filtering method to eliminate coarse-grained entities that are inconsistent with fine-grained entity types to avoid performance degradation. Our experimental results show that our method outperforms both $K$-shot learning and supervised learning methods when dealing with a small number of fine-grained annotations.

{{</citation>}}


### (76/143) Adaptation with Self-Evaluation to Improve Selective Prediction in LLMs (Jiefeng Chen et al., 2023)

{{<citation>}}

Jiefeng Chen, Jinsung Yoon, Sayna Ebrahimi, Sercan O Arik, Tomas Pfister, Somesh Jha. (2023)  
**Adaptation with Self-Evaluation to Improve Selective Prediction in LLMs**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2310.11689v1)  

---


**ABSTRACT**  
Large language models (LLMs) have recently shown great advances in a variety of tasks, including natural language understanding and generation. However, their use in high-stakes decision-making scenarios is still limited due to the potential for errors. Selective prediction is a technique that can be used to improve the reliability of the LLMs by allowing them to abstain from making predictions when they are unsure of the answer. In this work, we propose a novel framework for adaptation with self-evaluation to improve the selective prediction performance of LLMs. Our framework is based on the idea of using parameter-efficient tuning to adapt the LLM to the specific task at hand while improving its ability to perform self-evaluation. We evaluate our method on a variety of question-answering (QA) datasets and show that it outperforms state-of-the-art selective prediction methods. For example, on the CoQA benchmark, our method improves the AUACC from 91.23% to 92.63% and improves the AUROC from 74.61% to 80.25%.

{{</citation>}}


### (77/143) Superiority of Softmax: Unveiling the Performance Edge Over Linear Attention (Yichuan Deng et al., 2023)

{{<citation>}}

Yichuan Deng, Zhao Song, Tianyi Zhou. (2023)  
**Superiority of Softmax: Unveiling the Performance Edge Over Linear Attention**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2310.11685v1)  

---


**ABSTRACT**  
Large transformer models have achieved state-of-the-art results in numerous natural language processing tasks. Among the pivotal components of the transformer architecture, the attention mechanism plays a crucial role in capturing token interactions within sequences through the utilization of softmax function.   Conversely, linear attention presents a more computationally efficient alternative by approximating the softmax operation with linear complexity. However, it exhibits substantial performance degradation when compared to the traditional softmax attention mechanism.   In this paper, we bridge the gap in our theoretical understanding of the reasons behind the practical performance gap between softmax and linear attention. By conducting a comprehensive comparative analysis of these two attention mechanisms, we shed light on the underlying reasons for why softmax attention outperforms linear attention in most scenarios.

{{</citation>}}


### (78/143) Descriptive Knowledge Graph in Biomedical Domain (Kerui Zhu et al., 2023)

{{<citation>}}

Kerui Zhu, Jie Huang, Kevin Chen-Chuan Chang. (2023)  
**Descriptive Knowledge Graph in Biomedical Domain**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2310.11681v1)  

---


**ABSTRACT**  
We present a novel system that automatically extracts and generates informative and descriptive sentences from the biomedical corpus and facilitates the efficient search for relational knowledge. Unlike previous search engines or exploration systems that retrieve unconnected passages, our system organizes descriptive sentences as a relational graph, enabling researchers to explore closely related biomedical entities (e.g., diseases treated by a chemical) or indirectly connected entities (e.g., potential drugs for treating a disease). Our system also uses ChatGPT and a fine-tuned relation synthesis model to generate concise and reliable descriptive sentences from retrieved information, reducing the need for extensive human reading effort. With our system, researchers can easily obtain both high-level knowledge and detailed references and interactively steer to the information of interest. We spotlight the application of our system in COVID-19 research, illustrating its utility in areas such as drug repurposing and literature curation.

{{</citation>}}


### (79/143) Open-ended Commonsense Reasoning with Unrestricted Answer Scope (Chen Ling et al., 2023)

{{<citation>}}

Chen Ling, Xuchao Zhang, Xujiang Zhao, Yanchi Liu, Wei Cheng, Takao Osaki, Katsushi Matsuda, Haifeng Chen, Liang Zhao. (2023)  
**Open-ended Commonsense Reasoning with Unrestricted Answer Scope**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2310.11672v1)  

---


**ABSTRACT**  
Open-ended Commonsense Reasoning is defined as solving a commonsense question without providing 1) a short list of answer candidates and 2) a pre-defined answer scope. Conventional ways of formulating the commonsense question into a question-answering form or utilizing external knowledge to learn retrieval-based methods are less applicable in the open-ended setting due to an inherent challenge. Without pre-defining an answer scope or a few candidates, open-ended commonsense reasoning entails predicting answers by searching over an extremely large searching space. Moreover, most questions require implicit multi-hop reasoning, which presents even more challenges to our problem. In this work, we leverage pre-trained language models to iteratively retrieve reasoning paths on the external knowledge base, which does not require task-specific supervision. The reasoning paths can help to identify the most precise answer to the commonsense question. We conduct experiments on two commonsense benchmark datasets. Compared to other approaches, our proposed method achieves better performance both quantitatively and qualitatively.

{{</citation>}}


### (80/143) MixEdit: Revisiting Data Augmentation and Beyond for Grammatical Error Correction (Jingheng Ye et al., 2023)

{{<citation>}}

Jingheng Ye, Yinghui Li, Yangning Li, Hai-Tao Zheng. (2023)  
**MixEdit: Revisiting Data Augmentation and Beyond for Grammatical Error Correction**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2310.11671v1)  

---


**ABSTRACT**  
Data Augmentation through generating pseudo data has been proven effective in mitigating the challenge of data scarcity in the field of Grammatical Error Correction (GEC). Various augmentation strategies have been widely explored, most of which are motivated by two heuristics, i.e., increasing the distribution similarity and diversity of pseudo data. However, the underlying mechanism responsible for the effectiveness of these strategies remains poorly understood. In this paper, we aim to clarify how data augmentation improves GEC models. To this end, we introduce two interpretable and computationally efficient measures: Affinity and Diversity. Our findings indicate that an excellent GEC data augmentation strategy characterized by high Affinity and appropriate Diversity can better improve the performance of GEC models. Based on this observation, we propose MixEdit, a data augmentation approach that strategically and dynamically augments realistic data, without requiring extra monolingual corpora. To verify the correctness of our findings and the effectiveness of the proposed MixEdit, we conduct experiments on mainstream English and Chinese GEC datasets. The results show that MixEdit substantially improves GEC models and is complementary to traditional data augmentation methods.

{{</citation>}}


### (81/143) Field-testing items using artificial intelligence: Natural language processing with transformers (Hotaka Maeda, 2023)

{{<citation>}}

Hotaka Maeda. (2023)  
**Field-testing items using artificial intelligence: Natural language processing with transformers**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2310.11655v1)  

---


**ABSTRACT**  
Five thousand variations of the RoBERTa model, an artificially intelligent "transformer" that can understand text language, completed an English literacy exam with 29 multiple-choice questions. Data were used to calculate the psychometric properties of the items, which showed some degree of agreement to those obtained from human examinee data.

{{</citation>}}


### (82/143) Zero-shot Faithfulness Evaluation for Text Summarization with Foundation Language Model (Qi Jia et al., 2023)

{{<citation>}}

Qi Jia, Siyu Ren, Yizhu Liu, Kenny Q. Zhu. (2023)  
**Zero-shot Faithfulness Evaluation for Text Summarization with Foundation Language Model**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model, Summarization, Text Summarization  
[Paper Link](http://arxiv.org/abs/2310.11648v1)  

---


**ABSTRACT**  
Despite tremendous improvements in natural language generation, summarization models still suffer from the unfaithfulness issue. Previous work evaluates faithfulness either using models trained on the other tasks or in-domain synthetic data, or prompting a large model such as ChatGPT. This paper proposes to do zero-shot faithfulness evaluation simply with a moderately-sized foundation language model. We introduce a new metric FFLM, which is a combination of probability changes based on the intuition that prefixing a piece of text that is consistent with the output will increase the probability of predicting the output. Experiments show that FFLM performs competitively with or even outperforms ChatGPT on both inconsistency detection and faithfulness rating with 24x fewer parameters. FFLM also achieves improvements over other strong baselines.

{{</citation>}}


### (83/143) Systematic Assessment of Factual Knowledge in Large Language Models (Linhao Luo et al., 2023)

{{<citation>}}

Linhao Luo, Thuy-Trang Vu, Dinh Phung, Gholamreza Haffari. (2023)  
**Systematic Assessment of Factual Knowledge in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.11638v2)  

---


**ABSTRACT**  
Previous studies have relied on existing question-answering benchmarks to evaluate the knowledge stored in large language models (LLMs). However, this approach has limitations regarding factual knowledge coverage, as it mostly focuses on generic domains which may overlap with the pretraining data. This paper proposes a framework to systematically assess the factual knowledge of LLMs by leveraging knowledge graphs (KGs). Our framework automatically generates a set of questions and expected answers from the facts stored in a given KG, and then evaluates the accuracy of LLMs in answering these questions. We systematically evaluate the state-of-the-art LLMs with KGs in generic and specific domains. The experiment shows that ChatGPT is consistently the top performer across all domains. We also find that LLMs performance depends on the instruction finetuning, domain and question complexity and is prone to adversarial context.

{{</citation>}}


### (84/143) MAGNIFICo: Evaluating the In-Context Learning Ability of Large Language Models to Generalize to Novel Interpretations (Arkil Patel et al., 2023)

{{<citation>}}

Arkil Patel, Satwik Bhattamishra, Siva Reddy, Dzmitry Bahdanau. (2023)  
**MAGNIFICo: Evaluating the In-Context Learning Ability of Large Language Models to Generalize to Novel Interpretations**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.11634v1)  

---


**ABSTRACT**  
Humans possess a remarkable ability to assign novel interpretations to linguistic expressions, enabling them to learn new words and understand community-specific connotations. However, Large Language Models (LLMs) have a knowledge cutoff and are costly to finetune repeatedly. Therefore, it is crucial for LLMs to learn novel interpretations in-context. In this paper, we systematically analyse the ability of LLMs to acquire novel interpretations using in-context learning. To facilitate our study, we introduce MAGNIFICo, an evaluation suite implemented within a text-to-SQL semantic parsing framework that incorporates diverse tokens and prompt settings to simulate real-world complexity. Experimental results on MAGNIFICo demonstrate that LLMs exhibit a surprisingly robust capacity for comprehending novel interpretations from natural language descriptions as well as from discussions within long conversations. Nevertheless, our findings also highlight the need for further improvements, particularly when interpreting unfamiliar words or when composing multiple novel interpretations simultaneously in the same example. Additionally, our analysis uncovers the semantic predispositions in LLMs and reveals the impact of recency bias for information presented in long contexts.

{{</citation>}}


## eess.AS (2)



### (85/143) The CHiME-7 Challenge: System Description and Performance of NeMo Team's DASR System (Tae Jin Park et al., 2023)

{{<citation>}}

Tae Jin Park, He Huang, Ante Jukic, Kunal Dhawan, Krishna C. Puvvada, Nithin Koluguri, Nikolay Karpov, Aleksandr Laptev, Jagadeesh Balam, Boris Ginsburg. (2023)  
**The CHiME-7 Challenge: System Description and Performance of NeMo Team's DASR System**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2310.12378v1)  

---


**ABSTRACT**  
We present the NVIDIA NeMo team's multi-channel speech recognition system for the 7th CHiME Challenge Distant Automatic Speech Recognition (DASR) Task, focusing on the development of a multi-channel, multi-speaker speech recognition system tailored to transcribe speech from distributed microphones and microphone arrays. The system predominantly comprises of the following integral modules: the Speaker Diarization Module, Multi-channel Audio Front-End Processing Module, and the ASR Module. These components collectively establish a cascading system, meticulously processing multi-channel and multi-speaker audio input. Moreover, this paper highlights the comprehensive optimization process that significantly enhanced our system's performance. Our team's submission is largely based on NeMo toolkits and will be publicly available.

{{</citation>}}


### (86/143) DASA: Difficulty-Aware Semantic Augmentation for Speaker Verification (Yuanyuan Wang et al., 2023)

{{<citation>}}

Yuanyuan Wang, Yang Zhang, Zhiyong Wu, Zhihan Yang, Tao Wei, Kun Zou, Helen Meng. (2023)  
**DASA: Difficulty-Aware Semantic Augmentation for Speaker Verification**  

---
Primary Category: eess.AS  
Categories: cs-AI, eess-AS, eess.AS  
Keywords: Augmentation, Speaker Verification  
[Paper Link](http://arxiv.org/abs/2310.12111v1)  

---


**ABSTRACT**  
Data augmentation is vital to the generalization ability and robustness of deep neural networks (DNNs) models. Existing augmentation methods for speaker verification manipulate the raw signal, which are time-consuming and the augmented samples lack diversity. In this paper, we present a novel difficulty-aware semantic augmentation (DASA) approach for speaker verification, which can generate diversified training samples in speaker embedding space with negligible extra computing cost. Firstly, we augment training samples by perturbing speaker embeddings along semantic directions, which are obtained from speaker-wise covariance matrices. Secondly, accurate covariance matrices are estimated from robust speaker embeddings during training, so we introduce difficultyaware additive margin softmax (DAAM-Softmax) to obtain optimal speaker embeddings. Finally, we assume the number of augmented samples goes to infinity and derive a closed-form upper bound of the expected loss with DASA, which achieves compatibility and efficiency. Extensive experiments demonstrate the proposed approach can achieve a remarkable performance improvement. The best result achieves a 14.6% relative reduction in EER metric on CN-Celeb evaluation set.

{{</citation>}}


## cs.CR (7)



### (87/143) REMARK-LLM: A Robust and Efficient Watermarking Framework for Generative Large Language Models (Ruisi Zhang et al., 2023)

{{<citation>}}

Ruisi Zhang, Shehzeen Samarah Hussain, Paarth Neekhara, Farinaz Koushanfar. (2023)  
**REMARK-LLM: A Robust and Efficient Watermarking Framework for Generative Large Language Models**  

---
Primary Category: cs.CR  
Categories: cs-CL, cs-CR, cs.CR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.12362v1)  

---


**ABSTRACT**  
We present REMARK-LLM, a novel efficient, and robust watermarking framework designed for texts generated by large language models (LLMs). Synthesizing human-like content using LLMs necessitates vast computational resources and extensive datasets, encapsulating critical intellectual property (IP). However, the generated content is prone to malicious exploitation, including spamming and plagiarism. To address the challenges, REMARK-LLM proposes three new components: (i) a learning-based message encoding module to infuse binary signatures into LLM-generated texts; (ii) a reparameterization module to transform the dense distributions from the message encoding to the sparse distribution of the watermarked textual tokens; (iii) a decoding module dedicated for signature extraction; Furthermore, we introduce an optimized beam search algorithm to guarantee the coherence and consistency of the generated content. REMARK-LLM is rigorously trained to encourage the preservation of semantic integrity in watermarked content, while ensuring effective watermark retrieval. Extensive evaluations on multiple unseen datasets highlight REMARK-LLM proficiency and transferability in inserting 2 times more signature bits into the same texts when compared to prior art, all while maintaining semantic integrity. Furthermore, REMARK-LLM exhibits better resilience against a spectrum of watermark detection and removal attacks.

{{</citation>}}


### (88/143) PrivInfer: Privacy-Preserving Inference for Black-box Large Language Model (Meng Tong et al., 2023)

{{<citation>}}

Meng Tong, Kejiang Chen, Yuang Qi, Jie Zhang, Weiming Zhang, Nenghai Yu. (2023)  
**PrivInfer: Privacy-Preserving Inference for Black-box Large Language Model**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: ChatGPT, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2310.12214v2)  

---


**ABSTRACT**  
Large language models (LLMs), such as ChatGPT, have simplified text generation tasks, yet their inherent privacy risks are increasingly garnering attention. While differential privacy techniques have been successfully applied to text classification tasks, the resultant semantic bias makes them unsuitable for text generation. Homomorphic encryption inference methods have also been introduced. However, the significant computational and communication costs limit their viability. Furthermore, closed-source, black-box models such as GPT-4 withhold their architecture, thwarting certain privacy-enhancing strategies such as splitting inference into local and remote and then adding noise when communicating. To overcome these challenges, we introduce PrivInfer, the first practical privacy-preserving inference framework for black-box LLMs in text generation. PrivInfer employs differential privacy methods to generate perturbed prompts for remote LLMs inference and extracts the meaningful response from the remote perturbed results. We also introduce RANTEXT, a differential privacy mechanism within the perturbation module of PrivInfer specifically for LLMs that leverages random adjacency in text perturbations. Experimental results indicate that PrivInfer is comparable to GPT-4 in terms of text generation quality while protecting privacy, and RANTEXT provides enhanced privacy protection against three types of differential privacy attacks, including our newly introduced GPT inference attack, compared to baseline methods.

{{</citation>}}


### (89/143) Envisioning the Future of Cyber Security in Post-Quantum Era: A Survey on PQ Standardization, Applications, Challenges and Opportunities (Saleh Darzi et al., 2023)

{{<citation>}}

Saleh Darzi, Kasra Ahmadi, Saeed Aghapour, Attila Altay Yavuz, Mehran Mozaffari Kermani. (2023)  
**Envisioning the Future of Cyber Security in Post-Quantum Era: A Survey on PQ Standardization, Applications, Challenges and Opportunities**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Cyber Security, Security  
[Paper Link](http://arxiv.org/abs/2310.12037v1)  

---


**ABSTRACT**  
The rise of quantum computers exposes vulnerabilities in current public key cryptographic protocols, necessitating the development of secure post-quantum (PQ) schemes. Hence, we conduct a comprehensive study on various PQ approaches, covering the constructional design, structural vulnerabilities, and offer security assessments, implementation evaluations, and a particular focus on side-channel attacks. We analyze global standardization processes, evaluate their metrics in relation to real-world applications, and primarily focus on standardized PQ schemes, selected additional signature competition candidates, and PQ-secure cutting-edge schemes beyond standardization. Finally, we present visions and potential future directions for a seamless transition to the PQ era.

{{</citation>}}


### (90/143) Malicious Agent Detection for Robust Multi-Agent Collaborative Perception (Yangheng Zhao et al., 2023)

{{<citation>}}

Yangheng Zhao, Zhen Xiang, Sheng Yin, Xianghe Pang, Siheng Chen, Yanfeng Wang. (2023)  
**Malicious Agent Detection for Robust Multi-Agent Collaborative Perception**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.11901v1)  

---


**ABSTRACT**  
Recently, multi-agent collaborative (MAC) perception has been proposed and outperformed the traditional single-agent perception in many applications, such as autonomous driving. However, MAC perception is more vulnerable to adversarial attacks than single-agent perception due to the information exchange. The attacker can easily degrade the performance of a victim agent by sending harmful information from a malicious agent nearby. In this paper, we extend adversarial attacks to an important perception task -- MAC object detection, where generic defenses such as adversarial training are no longer effective against these attacks. More importantly, we propose Malicious Agent Detection (MADE), a reactive defense specific to MAC perception that can be deployed by each agent to accurately detect and then remove any potential malicious agent in its local collaboration network. In particular, MADE inspects each agent in the network independently using a semi-supervised anomaly detector based on a double-hypothesis test with the Benjamini-Hochberg procedure to control the false positive rate of the inference. For the two hypothesis tests, we propose a match loss statistic and a collaborative reconstruction loss statistic, respectively, both based on the consistency between the agent to be inspected and the ego agent where our detector is deployed. We conduct comprehensive evaluations on a benchmark 3D dataset V2X-sim and a real-road dataset DAIR-V2X and show that with the protection of MADE, the drops in the average precision compared with the best-case "oracle" defender against our attack are merely 1.28% and 0.34%, respectively, much lower than 8.92% and 10.00% for adversarial training, respectively.

{{</citation>}}


### (91/143) Revisiting Transferable Adversarial Image Examples: Attack Categorization, Evaluation Guidelines, and New Insights (Zhengyu Zhao et al., 2023)

{{<citation>}}

Zhengyu Zhao, Hanwei Zhang, Renjue Li, Ronan Sicre, Laurent Amsaleg, Michael Backes, Qi Li, Chao Shen. (2023)  
**Revisiting Transferable Adversarial Image Examples: Attack Categorization, Evaluation Guidelines, and New Insights**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-CV, cs-LG, cs.CR  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2310.11850v1)  

---


**ABSTRACT**  
Transferable adversarial examples raise critical security concerns in real-world, black-box attack scenarios. However, in this work, we identify two main problems in common evaluation practices: (1) For attack transferability, lack of systematic, one-to-one attack comparison and fair hyperparameter settings. (2) For attack stealthiness, simply no comparisons. To address these problems, we establish new evaluation guidelines by (1) proposing a novel attack categorization strategy and conducting systematic and fair intra-category analyses on transferability, and (2) considering diverse imperceptibility metrics and finer-grained stealthiness characteristics from the perspective of attack traceback. To this end, we provide the first large-scale evaluation of transferable adversarial examples on ImageNet, involving 23 representative attacks against 9 representative defenses. Our evaluation leads to a number of new insights, including consensus-challenging ones: (1) Under a fair attack hyperparameter setting, one early attack method, DI, actually outperforms all the follow-up methods. (2) A state-of-the-art defense, DiffPure, actually gives a false sense of (white-box) security since it is indeed largely bypassed by our (black-box) transferable attacks. (3) Even when all attacks are bounded by the same $L_p$ norm, they lead to dramatically different stealthiness performance, which negatively correlates with their transferability performance. Overall, our work demonstrates that existing problematic evaluations have indeed caused misleading conclusions and missing points, and as a result, hindered the assessment of the actual progress in this field.

{{</citation>}}


### (92/143) PhishReplicant: A Language Model-based Approach to Detect Generated Squatting Domain Names (Takashi Koide et al., 2023)

{{<citation>}}

Takashi Koide, Naoki Fukushi, Hiroki Nakano, Daiki Chiba. (2023)  
**PhishReplicant: A Language Model-based Approach to Detect Generated Squatting Domain Names**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.11763v1)  

---


**ABSTRACT**  
Domain squatting is a technique used by attackers to create domain names for phishing sites. In recent phishing attempts, we have observed many domain names that use multiple techniques to evade existing methods for domain squatting. These domain names, which we call generated squatting domains (GSDs), are quite different in appearance from legitimate domain names and do not contain brand names, making them difficult to associate with phishing. In this paper, we propose a system called PhishReplicant that detects GSDs by focusing on the linguistic similarity of domain names. We analyzed newly registered and observed domain names extracted from certificate transparency logs, passive DNS, and DNS zone files. We detected 3,498 domain names acquired by attackers in a four-week experiment, of which 2,821 were used for phishing sites within a month of detection. We also confirmed that our proposed system outperformed existing systems in both detection accuracy and number of domain names detected. As an in-depth analysis, we examined 205k GSDs collected over 150 days and found that phishing using GSDs was distributed globally. However, attackers intensively targeted brands in specific regions and industries. By analyzing GSDs in real time, we can block phishing sites before or immediately after they appear.

{{</citation>}}


### (93/143) Free-text Keystroke Authentication using Transformers: A Comparative Study of Architectures and Loss Functions (Saleh Momeni et al., 2023)

{{<citation>}}

Saleh Momeni, Bagher BabaAli. (2023)  
**Free-text Keystroke Authentication using Transformers: A Comparative Study of Architectures and Loss Functions**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs.CR  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.11640v1)  

---


**ABSTRACT**  
Keystroke biometrics is a promising approach for user identification and verification, leveraging the unique patterns in individuals' typing behavior. In this paper, we propose a Transformer-based network that employs self-attention to extract informative features from keystroke sequences, surpassing the performance of traditional Recurrent Neural Networks. We explore two distinct architectures, namely bi-encoder and cross-encoder, and compare their effectiveness in keystroke authentication. Furthermore, we investigate different loss functions, including triplet, batch-all triplet, and WDCL loss, along with various distance metrics such as Euclidean, Manhattan, and cosine distances. These experiments allow us to optimize the training process and enhance the performance of our model. To evaluate our proposed model, we employ the Aalto desktop keystroke dataset. The results demonstrate that the bi-encoder architecture with batch-all triplet loss and cosine distance achieves the best performance, yielding an exceptional Equal Error Rate of 0.0186%. Furthermore, alternative algorithms for calculating similarity scores are explored to enhance accuracy. Notably, the utilization of a one-class Support Vector Machine reduces the Equal Error Rate to an impressive 0.0163%. The outcomes of this study indicate that our model surpasses the previous state-of-the-art in free-text keystroke authentication. These findings contribute to advancing the field of keystroke authentication and offer practical implications for secure user verification systems.

{{</citation>}}


## cs.IR (6)



### (94/143) Retrieve-Cluster-Summarize: An Alternative to End-to-End Training for Query-specific Article Generation (Connor Lennox et al., 2023)

{{<citation>}}

Connor Lennox, Sumanta Kashyapi, Laura Dietz. (2023)  
**Retrieve-Cluster-Summarize: An Alternative to End-to-End Training for Query-specific Article Generation**  

---
Primary Category: cs.IR  
Categories: H-3, cs-IR, cs.IR  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.12361v1)  

---


**ABSTRACT**  
Query-specific article generation is the task of, given a search query, generate a single article that gives an overview of the topic. We envision such articles as an alternative to presenting a ranking of search results. While generative Large Language Models (LLMs) like chatGPT also address this task, they are known to hallucinate new information, their models are secret, hard to analyze and control. Some generative LLMs provide supporting references, yet these are often unrelated to the generated content. As an alternative, we propose to study article generation systems that integrate document retrieval, query-specific clustering, and summarization. By design, such models can provide actual citations as provenance for their generated text. In particular, we contribute an evaluation framework that allows to separately trains and evaluate each of these three components before combining them into one system. We experimentally demonstrate that a system comprised of the best-performing individual components also obtains the best F-1 overall system quality.

{{</citation>}}


### (95/143) Automated Attribute Extraction from Legal Proceedings (Subinay Adhikary et al., 2023)

{{<citation>}}

Subinay Adhikary, Sagnik Das, Sagnik Saha, Procheta Sen, Dwaipayan Roy, Kripabandhu Ghosh. (2023)  
**Automated Attribute Extraction from Legal Proceedings**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: AI, Legal  
[Paper Link](http://arxiv.org/abs/2310.12131v1)  

---


**ABSTRACT**  
The escalating number of pending cases is a growing concern world-wide. Recent advancements in digitization have opened up possibilities for leveraging artificial intelligence (AI) tools in the processing of legal documents. Adopting a structured representation for legal documents, as opposed to a mere bag-of-words flat text representation, can significantly enhance processing capabilities. With the aim of achieving this objective, we put forward a set of diverse attributes for criminal case proceedings. We use a state-of-the-art sequence labeling framework to automatically extract attributes from the legal documents. Moreover, we demonstrate the efficacy of the extracted attributes in a downstream task, namely legal judgment prediction.

{{</citation>}}


### (96/143) CIR at the NTCIR-17 ULTRE-2 Task (Lulu Yu et al., 2023)

{{<citation>}}

Lulu Yu, Keping Bi, Jiafeng Guo, Xueqi Cheng. (2023)  
**CIR at the NTCIR-17 ULTRE-2 Task**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Information Retrieval  
[Paper Link](http://arxiv.org/abs/2310.11852v1)  

---


**ABSTRACT**  
The Chinese academy of sciences Information Retrieval team (CIR) has participated in the NTCIR-17 ULTRE-2 task. This paper describes our approaches and reports our results on the ULTRE-2 task. We recognize the issue of false negatives in the Baidu search data in this competition is very severe, much more severe than position bias. Hence, we adopt the Dual Learning Algorithm (DLA) to address the position bias and use it as an auxiliary model to study how to alleviate the false negative issue. We approach the problem from two perspectives: 1) correcting the labels for non-clicked items by a relevance judgment model trained from DLA, and learn a new ranker that is initialized from DLA; 2) including random documents as true negatives and documents that have partial matching as hard negatives. Both methods can enhance the model performance and our best method has achieved nDCG@10 of 0.5355, which is 2.66% better than the best score from the organizer.

{{</citation>}}


### (97/143) From Relevance to Utility: Evidence Retrieval with Feedback for Fact Verification (Hengran Zhang et al., 2023)

{{<citation>}}

Hengran Zhang, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Yixing Fan, Xueqi Cheng. (2023)  
**From Relevance to Utility: Evidence Retrieval with Feedback for Fact Verification**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Fact Verification  
[Paper Link](http://arxiv.org/abs/2310.11675v2)  

---


**ABSTRACT**  
Retrieval-enhanced methods have become a primary approach in fact verification (FV); it requires reasoning over multiple retrieved pieces of evidence to verify the integrity of a claim. To retrieve evidence, existing work often employs off-the-shelf retrieval models whose design is based on the probability ranking principle. We argue that, rather than relevance, for FV we need to focus on the utility that a claim verifier derives from the retrieved evidence. We introduce the feedback-based evidence retriever(FER) that optimizes the evidence retrieval process by incorporating feedback from the claim verifier. As a feedback signal we use the divergence in utility between how effectively the verifier utilizes the retrieved evidence and the ground-truth evidence to produce the final claim label. Empirical studies demonstrate the superiority of FER over prevailing baselines.

{{</citation>}}


### (98/143) VKIE: The Application of Key Information Extraction on Video Text (Siyu An et al., 2023)

{{<citation>}}

Siyu An, Ye Liu, Haoyuan Peng, Di Yin. (2023)  
**VKIE: The Application of Key Information Extraction on Video Text**  

---
Primary Category: cs.IR  
Categories: cs-CV, cs-IR, cs-MM, cs.IR  
Keywords: Information Extraction  
[Paper Link](http://arxiv.org/abs/2310.11650v1)  

---


**ABSTRACT**  
Extracting structured information from videos is critical for numerous downstream applications in the industry. In this paper, we define a significant task of extracting hierarchical key information from visual texts on videos. To fulfill this task, we decouples it into four subtasks and introduce two implementation solutions called PipVKIE and UniVKIE. PipVKIE sequentially completes the four subtasks in continuous stages, while UniVKIE is improved by unifying all the subtasks into one backbone. Both PipVKIE and UniVKIE leverage multimodal information from vision, text, and coordinates for feature representation. Extensive experiments on one well-defined dataset demonstrate that our solutions can achieve remarkable performance and efficient inference speed. The code and dataset will be publicly available.

{{</citation>}}


### (99/143) Open Information Extraction: A Review of Baseline Techniques, Approaches, and Applications (Serafina Kamp et al., 2023)

{{<citation>}}

Serafina Kamp, Morteza Fayazi, Zineb Benameur-El, Shuyan Yu, Ronald Dreslinski. (2023)  
**Open Information Extraction: A Review of Baseline Techniques, Approaches, and Applications**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Information Extraction, Knowledge Graph, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2310.11644v1)  

---


**ABSTRACT**  
With the abundant amount of available online and offline text data, there arises a crucial need to extract the relation between phrases and summarize the main content of each document in a few words. For this purpose, there have been many studies recently in Open Information Extraction (OIE). OIE improves upon relation extraction techniques by analyzing relations across different domains and avoids requiring hand-labeling pre-specified relations in sentences. This paper surveys recent approaches of OIE and its applications on Knowledge Graph (KG), text summarization, and Question Answering (QA). Moreover, the paper describes OIE basis methods in relation extraction. It briefly discusses the main approaches and the pros and cons of each method. Finally, it gives an overview about challenges, open issues, and future work opportunities for OIE, relation extraction, and OIE applications.

{{</citation>}}


## cs.SE (3)



### (100/143) Large Language Models for Code Analysis: Do LLMs Really Do Their Job? (Chongzhou Fang et al., 2023)

{{<citation>}}

Chongzhou Fang, Ning Miao, Shaurya Srivastav, Jialin Liu, Ruoyu Zhang, Ruijie Fang, Asmita Asmita, Ryan Tsang, Najmeh Nazari, Han Wang, Houman Homayoun. (2023)  
**Large Language Models for Code Analysis: Do LLMs Really Do Their Job?**  

---
Primary Category: cs.SE  
Categories: cs-CR, cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.12357v1)  

---


**ABSTRACT**  
Large language models (LLMs) have demonstrated significant potential in the realm of natural language understanding and programming code processing tasks. Their capacity to comprehend and generate human-like code has spurred research into harnessing LLMs for code analysis purposes. However, the existing body of literature falls short in delivering a systematic evaluation and assessment of LLMs' effectiveness in code analysis, particularly in the context of obfuscated code.   This paper seeks to bridge this gap by offering a comprehensive evaluation of LLMs' capabilities in performing code analysis tasks. Additionally, it presents real-world case studies that employ LLMs for the analysis of malicious code. Our findings indicate that LLMs can indeed serve as valuable tools for automating code analysis, albeit with certain limitations. Through meticulous exploration, this research contributes to a deeper understanding of the potential and constraints associated with utilizing LLMs in code analysis, paving the way for enhanced applications in this critical domain.

{{</citation>}}


### (101/143) A comprehensible analysis of the efficacy of Ensemble Models for Bug Prediction (Ingrid Marçal et al., 2023)

{{<citation>}}

Ingrid Marçal, Rogério Eduardo Garcia. (2023)  
**A comprehensible analysis of the efficacy of Ensemble Models for Bug Prediction**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.12133v1)  

---


**ABSTRACT**  
The correctness of software systems is vital for their effective operation. It makes discovering and fixing software bugs an important development task. The increasing use of Artificial Intelligence (AI) techniques in Software Engineering led to the development of a number of techniques that can assist software developers in identifying potential bugs in code. In this paper, we present a comprehensible comparison and analysis of the efficacy of two AI-based approaches, namely single AI models and ensemble AI models, for predicting the probability of a Java class being buggy. We used two open-source Apache Commons Project's Java components for training and evaluating the models. Our experimental findings indicate that the ensemble of AI models can outperform the results of applying individual AI models. We also offer insight into the factors that contribute to the enhanced performance of the ensemble AI model. The presented results demonstrate the potential of using ensemble AI models to enhance bug prediction results, which could ultimately result in more reliable software systems.

{{</citation>}}


### (102/143) Telecom AI Native Systems in the Age of Generative AI -- An Engineering Perspective (Ricardo Britto et al., 2023)

{{<citation>}}

Ricardo Britto, Timothy Murphy, Massimo Iovene, Leif Jonsson, Melike Erol-Kantarci, Benedek Kovács. (2023)  
**Telecom AI Native Systems in the Age of Generative AI -- An Engineering Perspective**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2310.11770v1)  

---


**ABSTRACT**  
The rapid advancements in Artificial Intelligence (AI), particularly in generative AI and foundational models (FMs), have ushered in transformative changes across various industries. Large language models (LLMs), a type of FM, have demonstrated their prowess in natural language processing tasks and content generation, revolutionizing how we interact with software products and services. This article explores the integration of FMs in the telecommunications industry, shedding light on the concept of AI native telco, where AI is seamlessly woven into the fabric of telecom products. It delves into the engineering considerations and unique challenges associated with implementing FMs into the software life cycle, emphasizing the need for AI native-first approaches. Despite the enormous potential of FMs, ethical, regulatory, and operational challenges require careful consideration, especially in mission-critical telecom contexts. As the telecom industry seeks to harness the power of AI, a comprehensive understanding of these challenges is vital to thrive in a fiercely competitive market.

{{</citation>}}


## physics.soc-ph (1)



### (103/143) Tracking electricity losses and their perceived causes using nighttime light and social media (Samuel W Kerber et al., 2023)

{{<citation>}}

Samuel W Kerber, Nicholas A Duncan, Guillaume F LHer, Morgan Bazilian, Chris Elvidge, Mark R Deinert. (2023)  
**Tracking electricity losses and their perceived causes using nighttime light and social media**  

---
Primary Category: physics.soc-ph  
Categories: cs-LG, cs-SI, physics-soc-ph, physics.soc-ph  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2310.12346v1)  

---


**ABSTRACT**  
Urban environments are intricate systems where the breakdown of critical infrastructure can impact both the economic and social well-being of communities. Electricity systems hold particular significance, as they are essential for other infrastructure, and disruptions can trigger widespread consequences. Typically, assessing electricity availability requires ground-level data, a challenge in conflict zones and regions with limited access. This study shows how satellite imagery, social media, and information extraction can monitor blackouts and their perceived causes. Night-time light data (in March 2019 for Caracas, Venezuela) is used to indicate blackout regions. Twitter data is used to determine sentiment and topic trends, while statistical analysis and topic modeling delved into public perceptions regarding blackout causes. The findings show an inverse relationship between nighttime light intensity. Tweets mentioning the Venezuelan President displayed heightened negativity and a greater prevalence of blame-related terms, suggesting a perception of government accountability for the outages.

{{</citation>}}


## cs.CV (19)



### (104/143) Improving Representation Learning for Histopathologic Images with Cluster Constraints (Weiyi Wu et al., 2023)

{{<citation>}}

Weiyi Wu, Chongyang Gao, Joseph DiPalma, Soroush Vosoughi, Saeed Hassanpour. (2023)  
**Improving Representation Learning for Histopathologic Images with Cluster Constraints**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2310.12334v1)  

---


**ABSTRACT**  
Recent advances in whole-slide image (WSI) scanners and computational capabilities have significantly propelled the application of artificial intelligence in histopathology slide analysis. While these strides are promising, current supervised learning approaches for WSI analysis come with the challenge of exhaustively labeling high-resolution slides - a process that is both labor-intensive and time-consuming. In contrast, self-supervised learning (SSL) pretraining strategies are emerging as a viable alternative, given that they don't rely on explicit data annotations. These SSL strategies are quickly bridging the performance disparity with their supervised counterparts. In this context, we introduce an SSL framework. This framework aims for transferable representation learning and semantically meaningful clustering by synergizing invariance loss and clustering loss in WSI analysis. Notably, our approach outperforms common SSL methods in downstream classification and clustering tasks, as evidenced by tests on the Camelyon16 and a pancreatic cancer dataset. The code and additional details are accessible at: https://github.com/wwyi1828/CluSiam.

{{</citation>}}


### (105/143) Understanding Video Transformers for Segmentation: A Survey of Application and Interpretability (Rezaul Karim et al., 2023)

{{<citation>}}

Rezaul Karim, Richard P. Wildes. (2023)  
**Understanding Video Transformers for Segmentation: A Survey of Application and Interpretability**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.12296v1)  

---


**ABSTRACT**  
Video segmentation encompasses a wide range of categories of problem formulation, e.g., object, scene, actor-action and multimodal video segmentation, for delineating task-specific scene components with pixel-level masks. Recently, approaches in this research area shifted from concentrating on ConvNet-based to transformer-based models. In addition, various interpretability approaches have appeared for transformer models and video temporal dynamics, motivated by the growing interest in basic scientific understanding, model diagnostics and societal implications of real-world deployment. Previous surveys mainly focused on ConvNet models on a subset of video segmentation tasks or transformers for classification tasks. Moreover, component-wise discussion of transformer-based video segmentation models has not yet received due focus. In addition, previous reviews of interpretability methods focused on transformers for classification, while analysis of video temporal dynamics modelling capabilities of video models received less attention. In this survey, we address the above with a thorough discussion of various categories of video segmentation, a component-wise discussion of the state-of-the-art transformer-based models, and a review of related interpretability methods. We first present an introduction to the different video segmentation task categories, their objectives, specific challenges and benchmark datasets. Next, we provide a component-wise review of recent transformer-based models and document the state of the art on different video segmentation tasks. Subsequently, we discuss post-hoc and ante-hoc interpretability methods for transformer models and interpretability methods for understanding the role of the temporal dimension in video models. Finally, we conclude our discussion with future research directions.

{{</citation>}}


### (106/143) An Image is Worth Multiple Words: Learning Object Level Concepts using Multi-Concept Prompt Learning (Chen Jin et al., 2023)

{{<citation>}}

Chen Jin, Ryutaro Tanno, Amrutha Saseendran, Tom Diethe, Philip Teare. (2023)  
**An Image is Worth Multiple Words: Learning Object Level Concepts using Multi-Concept Prompt Learning**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs-GR, cs-LG, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2310.12274v1)  

---


**ABSTRACT**  
Textural Inversion, a prompt learning method, learns a singular embedding for a new "word" to represent image style and appearance, allowing it to be integrated into natural language sentences to generate novel synthesised images. However, identifying and integrating multiple object-level concepts within one scene poses significant challenges even when embeddings for individual concepts are attainable. This is further confirmed by our empirical tests. To address this challenge, we introduce a framework for Multi-Concept Prompt Learning (MCPL), where multiple new "words" are simultaneously learned from a single sentence-image pair. To enhance the accuracy of word-concept correlation, we propose three regularisation techniques: Attention Masking (AttnMask) to concentrate learning on relevant areas; Prompts Contrastive Loss (PromptCL) to separate the embeddings of different concepts; and Bind adjective (Bind adj.) to associate new "words" with known words. We evaluate via image generation, editing, and attention visualisation with diverse images. Extensive quantitative comparisons demonstrate that our method can learn more semantically disentangled concepts with enhanced word-concept correlation. Additionally, we introduce a novel dataset and evaluation protocol tailored for this new task of learning object-level concepts.

{{</citation>}}


### (107/143) Improving SCGAN's Similarity Constraint and Learning a Better Disentangled Representation (Iman Yazdanpanah, 2023)

{{<citation>}}

Iman Yazdanpanah. (2023)  
**Improving SCGAN's Similarity Constraint and Learning a Better Disentangled Representation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV, eess-IV  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2310.12262v1)  

---


**ABSTRACT**  
SCGAN adds a similarity constraint between generated images and conditions as a regularization term on generative adversarial networks. Similarity constraint works as a tutor to instruct the generator network to comprehend the difference of representations based on conditions. We understand how SCGAN works on a deeper level. This understanding makes us realize that the similarity constraint functions like the contrastive loss function. We believe that a model with high understanding and intelligence measures the similarity between images based on their structure and high level features, just like humans do. Two major changes we applied to SCGAN in order to make a modified model are using SSIM to measure similarity between images and applying contrastive loss principles to the similarity constraint. The modified model performs better using FID and FactorVAE metrics. The modified model also has better generalisability compared to other models. Keywords Generative Adversarial Nets, Unsupervised Learning, Disentangled Representation Learning, Contrastive Disentanglement, SSIM

{{</citation>}}


### (108/143) Tailoring Adversarial Attacks on Deep Neural Networks for Targeted Class Manipulation Using DeepFool Algorithm (S. M. Fazle Rabby Labib et al., 2023)

{{<citation>}}

S. M. Fazle Rabby Labib, Joyanta Jyoti Mondal, Meem Arafat Manab. (2023)  
**Tailoring Adversarial Attacks on Deep Neural Networks for Targeted Class Manipulation Using DeepFool Algorithm**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Adversarial Attack, Transformer  
[Paper Link](http://arxiv.org/abs/2310.13019v1)  

---


**ABSTRACT**  
Deep neural networks (DNNs) have significantly advanced various domains, but their vulnerability to adversarial attacks poses serious concerns. Understanding these vulnerabilities and developing effective defense mechanisms is crucial. DeepFool, an algorithm proposed by Moosavi-Dezfooli et al. (2016), finds minimal perturbations to misclassify input images. However, DeepFool lacks a targeted approach, making it less effective in specific attack scenarios. Also, in previous related works, researchers primarily focus on success, not considering how much an image is getting distorted; the integrity of the image quality, and the confidence level to misclassifying. So, in this paper, we propose Targeted DeepFool, an augmented version of DeepFool that allows targeting specific classes for misclassification. We also introduce a minimum confidence score requirement hyperparameter to enhance flexibility. Our experiments demonstrate the effectiveness and efficiency of the proposed method across different deep neural network architectures while preserving image integrity as much as possible. Results show that one of the deep convolutional neural network architectures, AlexNet, and one of the state-of-the-art model Vision Transformer exhibit high robustness to getting fooled. Our code will be made public when publishing the paper.

{{</citation>}}


### (109/143) Learning from Rich Semantics and Coarse Locations for Long-tailed Object Detection (Lingchen Meng et al., 2023)

{{<citation>}}

Lingchen Meng, Xiyang Dai, Jianwei Yang, Dongdong Chen, Yinpeng Chen, Mengchen Liu, Yi-Ling Chen, Zuxuan Wu, Lu Yuan, Yu-Gang Jiang. (2023)  
**Learning from Rich Semantics and Coarse Locations for Long-tailed Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2310.12152v1)  

---


**ABSTRACT**  
Long-tailed object detection (LTOD) aims to handle the extreme data imbalance in real-world datasets, where many tail classes have scarce instances. One popular strategy is to explore extra data with image-level labels, yet it produces limited results due to (1) semantic ambiguity -- an image-level label only captures a salient part of the image, ignoring the remaining rich semantics within the image; and (2) location sensitivity -- the label highly depends on the locations and crops of the original image, which may change after data transformations like random cropping. To remedy this, we propose RichSem, a simple but effective method, which is robust to learn rich semantics from coarse locations without the need of accurate bounding boxes. RichSem leverages rich semantics from images, which are then served as additional soft supervision for training detectors. Specifically, we add a semantic branch to our detector to learn these soft semantics and enhance feature representations for long-tailed object detection. The semantic branch is only used for training and is removed during inference. RichSem achieves consistent improvements on both overall and rare-category of LVIS under different backbones and detectors. Our method achieves state-of-the-art performance without requiring complex training and testing procedures. Moreover, we show the effectiveness of our method on other long-tailed datasets with additional experiments. Code is available at \url{https://github.com/MengLcool/RichSem}.

{{</citation>}}


### (110/143) DiagrammerGPT: Generating Open-Domain, Open-Platform Diagrams via LLM Planning (Abhay Zala et al., 2023)

{{<citation>}}

Abhay Zala, Han Lin, Jaemin Cho, Mohit Bansal. (2023)  
**DiagrammerGPT: Generating Open-Domain, Open-Platform Diagrams via LLM Planning**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs.CV  
Keywords: AI, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2310.12128v1)  

---


**ABSTRACT**  
Text-to-image (T2I) generation has seen significant growth over the past few years. Despite this, there has been little work on generating diagrams with T2I models. A diagram is a symbolic/schematic representation that explains information using structurally rich and spatially complex visualizations (e.g., a dense combination of related objects, text labels, directional arrows, connection lines, etc.). Existing state-of-the-art T2I models often fail at diagram generation because they lack fine-grained object layout control when many objects are densely connected via complex relations such as arrows/lines and also often fail to render comprehensible text labels. To address this gap, we present DiagrammerGPT, a novel two-stage text-to-diagram generation framework that leverages the layout guidance capabilities of LLMs (e.g., GPT-4) to generate more accurate open-domain, open-platform diagrams. In the first stage, we use LLMs to generate and iteratively refine 'diagram plans' (in a planner-auditor feedback loop) which describe all the entities (objects and text labels), their relationships (arrows or lines), and their bounding box layouts. In the second stage, we use a diagram generator, DiagramGLIGEN, and a text label rendering module to generate diagrams following the diagram plans. To benchmark the text-to-diagram generation task, we introduce AI2D-Caption, a densely annotated diagram dataset built on top of the AI2D dataset. We show quantitatively and qualitatively that our DiagrammerGPT framework produces more accurate diagrams, outperforming existing T2I models. We also provide comprehensive analysis including open-domain diagram generation, vector graphic diagram generation in different platforms, human-in-the-loop diagram plan editing, and multimodal planner/auditor LLMs (e.g., GPT-4Vision). We hope our work can inspire further research on diagram generation via T2I models and LLMs.

{{</citation>}}


### (111/143) On the Benefit of Generative Foundation Models for Human Activity Recognition (Zikang Leng et al., 2023)

{{<citation>}}

Zikang Leng, Hyeokhyen Kwon, Thomas Plötz. (2023)  
**On the Benefit of Generative Foundation Models for Human Activity Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2310.12085v1)  

---


**ABSTRACT**  
In human activity recognition (HAR), the limited availability of annotated data presents a significant challenge. Drawing inspiration from the latest advancements in generative AI, including Large Language Models (LLMs) and motion synthesis models, we believe that generative AI can address this data scarcity by autonomously generating virtual IMU data from text descriptions. Beyond this, we spotlight several promising research pathways that could benefit from generative AI for the community, including the generating benchmark datasets, the development of foundational models specific to HAR, the exploration of hierarchical structures within HAR, breaking down complex activities, and applications in health sensing and activity summarization.

{{</citation>}}


### (112/143) Exploring Fairness in Pre-trained Visual Transformer based Natural and GAN Generated Image Detection Systems and Understanding the Impact of Image Compression in Fairness (Manjary P. Gangan et al., 2023)

{{<citation>}}

Manjary P. Gangan, Anoop Kadan, Lajish V L. (2023)  
**Exploring Fairness in Pre-trained Visual Transformer based Natural and GAN Generated Image Detection Systems and Understanding the Impact of Image Compression in Fairness**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.12076v1)  

---


**ABSTRACT**  
It is not only sufficient to construct computational models that can accurately classify or detect fake images from real images taken from a camera, but it is also important to ensure whether these computational models are fair enough or produce biased outcomes that can eventually harm certain social groups or cause serious security threats. Exploring fairness in forensic algorithms is an initial step towards correcting these biases. Since visual transformers are recently being widely used in most image classification based tasks due to their capability to produce high accuracies, this study tries to explore bias in the transformer based image forensic algorithms that classify natural and GAN generated images. By procuring a bias evaluation corpora, this study analyzes bias in gender, racial, affective, and intersectional domains using a wide set of individual and pairwise bias evaluation measures. As the generalizability of the algorithms against image compression is an important factor to be considered in forensic tasks, this study also analyzes the role of image compression on model bias. Hence to study the impact of image compression on model bias, a two phase evaluation setting is followed, where a set of experiments is carried out in the uncompressed evaluation setting and the other in the compressed evaluation setting.

{{</citation>}}


### (113/143) On the use of Vision-Language models for Visual Sentiment Analysis: a study on CLIP (Cristina Bustos et al., 2023)

{{<citation>}}

Cristina Bustos, Carles Civit, Brian Du, Albert Sole-Ribalta, Agata Lapedriza. (2023)  
**On the use of Vision-Language models for Visual Sentiment Analysis: a study on CLIP**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Sentiment Analysis, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2310.12062v1)  

---


**ABSTRACT**  
This work presents a study on how to exploit the CLIP embedding space to perform Visual Sentiment Analysis. We experiment with two architectures built on top of the CLIP embedding space, which we denote by CLIP-E. We train the CLIP-E models with WEBEmo, the largest publicly available and manually labeled benchmark for Visual Sentiment Analysis, and perform two sets of experiments. First, we test on WEBEmo and compare the CLIP-E architectures with state-of-the-art (SOTA) models and with CLIP Zero-Shot. Second, we perform cross dataset evaluation, and test the CLIP-E architectures trained with WEBEmo on other Visual Sentiment Analysis benchmarks. Our results show that the CLIP-E approaches outperform SOTA models in WEBEmo fine grained categorization, and they also generalize better when tested on datasets that have not been seen during training. Interestingly, we observed that for the FI dataset, CLIP Zero-Shot produces better accuracies than SOTA models and CLIP-E trained on WEBEmo. These results motivate several questions that we discuss in this paper, such as how we should design new benchmarks and evaluate Visual Sentiment Analysis, and whether we should keep designing tailored Deep Learning models for Visual Sentiment Analysis or focus our efforts on better using the knowledge encoded in large vision-language models such as CLIP for this task.

{{</citation>}}


### (114/143) SegmATRon: Embodied Adaptive Semantic Segmentation for Indoor Environment (Tatiana Zemskova et al., 2023)

{{<citation>}}

Tatiana Zemskova, Margarita Kichik, Dmitry Yudin, Aleksei Staroverov, Aleksandr Panov. (2023)  
**SegmATRon: Embodied Adaptive Semantic Segmentation for Indoor Environment**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: AI, Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2310.12031v1)  

---


**ABSTRACT**  
This paper presents an adaptive transformer model named SegmATRon for embodied image semantic segmentation. Its distinctive feature is the adaptation of model weights during inference on several images using a hybrid multicomponent loss function. We studied this model on datasets collected in the photorealistic Habitat and the synthetic AI2-THOR Simulators. We showed that obtaining additional images using the agent's actions in an indoor environment can improve the quality of semantic segmentation. The code of the proposed approach and datasets are publicly available at https://github.com/wingrune/SegmATRon.

{{</citation>}}


### (115/143) IRAD: Implicit Representation-driven Image Resampling against Adversarial Attacks (Yue Cao et al., 2023)

{{<citation>}}

Yue Cao, Tianlin Li, Xiaofeng Cao, Ivor Tsang, Yang Liu, Qing Guo. (2023)  
**IRAD: Implicit Representation-driven Image Resampling against Adversarial Attacks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2310.11890v1)  

---


**ABSTRACT**  
We introduce a novel approach to counter adversarial attacks, namely, image resampling. Image resampling transforms a discrete image into a new one, simulating the process of scene recapturing or rerendering as specified by a geometrical transformation. The underlying rationale behind our idea is that image resampling can alleviate the influence of adversarial perturbations while preserving essential semantic information, thereby conferring an inherent advantage in defending against adversarial attacks. To validate this concept, we present a comprehensive study on leveraging image resampling to defend against adversarial attacks. We have developed basic resampling methods that employ interpolation strategies and coordinate shifting magnitudes. Our analysis reveals that these basic methods can partially mitigate adversarial attacks. However, they come with apparent limitations: the accuracy of clean images noticeably decreases, while the improvement in accuracy on adversarial examples is not substantial. We propose implicit representation-driven image resampling (IRAD) to overcome these limitations. First, we construct an implicit continuous representation that enables us to represent any input image within a continuous coordinate space. Second, we introduce SampleNet, which automatically generates pixel-wise shifts for resampling in response to different inputs. Furthermore, we can extend our approach to the state-of-the-art diffusion-based method, accelerating it with fewer time steps while preserving its defense capability. Extensive experiments demonstrate that our method significantly enhances the adversarial robustness of diverse deep models against various attacks while maintaining high accuracy on clean images.

{{</citation>}}


### (116/143) Evaluating the Fairness of Discriminative Foundation Models in Computer Vision (Junaid Ali et al., 2023)

{{<citation>}}

Junaid Ali, Matthaeus Kleindessner, Florian Wenzel, Kailash Budhathoki, Volkan Cevher, Chris Russell. (2023)  
**Evaluating the Fairness of Discriminative Foundation Models in Computer Vision**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-CY, cs-LG, cs.CV  
Keywords: AI, Computer Vision  
[Paper Link](http://arxiv.org/abs/2310.11867v1)  

---


**ABSTRACT**  
We propose a novel taxonomy for bias evaluation of discriminative foundation models, such as Contrastive Language-Pretraining (CLIP), that are used for labeling tasks. We then systematically evaluate existing methods for mitigating bias in these models with respect to our taxonomy. Specifically, we evaluate OpenAI's CLIP and OpenCLIP models for key applications, such as zero-shot classification, image retrieval and image captioning. We categorize desired behaviors based around three axes: (i) if the task concerns humans; (ii) how subjective the task is (i.e., how likely it is that people from a diverse range of backgrounds would agree on a labeling); and (iii) the intended purpose of the task and if fairness is better served by impartiality (i.e., making decisions independent of the protected attributes) or representation (i.e., making decisions to maximize diversity). Finally, we provide quantitative fairness evaluations for both binary-valued and multi-valued protected attributes over ten diverse datasets. We find that fair PCA, a post-processing method for fair representations, works very well for debiasing in most of the aforementioned tasks while incurring only minor loss of performance. However, different debiasing approaches vary in their effectiveness depending on the task. Hence, one should choose the debiasing approach depending on the specific use case.

{{</citation>}}


### (117/143) VQ-NeRF: Neural Reflectance Decomposition and Editing with Vector Quantization (Hongliang Zhong et al., 2023)

{{<citation>}}

Hongliang Zhong, Jingbo Zhang, Jing Liao. (2023)  
**VQ-NeRF: Neural Reflectance Decomposition and Editing with Vector Quantization**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-GR, cs-LG, cs.CV  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2310.11864v1)  

---


**ABSTRACT**  
We propose VQ-NeRF, a two-branch neural network model that incorporates Vector Quantization (VQ) to decompose and edit reflectance fields in 3D scenes. Conventional neural reflectance fields use only continuous representations to model 3D scenes, despite the fact that objects are typically composed of discrete materials in reality. This lack of discretization can result in noisy material decomposition and complicated material editing. To address these limitations, our model consists of a continuous branch and a discrete branch. The continuous branch follows the conventional pipeline to predict decomposed materials, while the discrete branch uses the VQ mechanism to quantize continuous materials into individual ones. By discretizing the materials, our model can reduce noise in the decomposition process and generate a segmentation map of discrete materials. Specific materials can be easily selected for further editing by clicking on the corresponding area of the segmentation outcomes. Additionally, we propose a dropout-based VQ codeword ranking strategy to predict the number of materials in a scene, which reduces redundancy in the material segmentation process. To improve usability, we also develop an interactive interface to further assist material editing. We evaluate our model on both computer-generated and real-world scenes, demonstrating its superior performance. To the best of our knowledge, our model is the first to enable discrete material editing in 3D scenes.

{{</citation>}}


### (118/143) Learning to Generate Parameters of ConvNets for Unseen Image Data (Shiye Wang et al., 2023)

{{<citation>}}

Shiye Wang, Kaituo Feng, Changsheng Li, Ye Yuan, Guoren Wang. (2023)  
**Learning to Generate Parameters of ConvNets for Unseen Image Data**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2310.11862v1)  

---


**ABSTRACT**  
Typical Convolutional Neural Networks (ConvNets) depend heavily on large amounts of image data and resort to an iterative optimization algorithm (e.g., SGD or Adam) to learn network parameters, which makes training very time- and resource-intensive. In this paper, we propose a new training paradigm and formulate the parameter learning of ConvNets into a prediction task: given a ConvNet architecture, we observe there exists correlations between image datasets and their corresponding optimal network parameters, and explore if we can learn a hyper-mapping between them to capture the relations, such that we can directly predict the parameters of the network for an image dataset never seen during the training phase. To do this, we put forward a new hypernetwork based model, called PudNet, which intends to learn a mapping between datasets and their corresponding network parameters, and then predicts parameters for unseen data with only a single forward propagation. Moreover, our model benefits from a series of adaptive hyper recurrent units sharing weights to capture the dependencies of parameters among different network layers. Extensive experiments demonstrate that our proposed method achieves good efficacy for unseen image datasets on two kinds of settings: Intra-dataset prediction and Inter-dataset prediction. Our PudNet can also well scale up to large-scale datasets, e.g., ImageNet-1K. It takes 8967 GPU seconds to train ResNet-18 on the ImageNet-1K using GC from scratch and obtain a top-5 accuracy of 44.65 %. However, our PudNet costs only 3.89 GPU seconds to predict the network parameters of ResNet-18 achieving comparable performance (44.92 %), more than 2,300 times faster than the traditional training paradigm.

{{</citation>}}


### (119/143) Domain-Generalized Face Anti-Spoofing with Unknown Attacks (Zong-Wei Hong et al., 2023)

{{<citation>}}

Zong-Wei Hong, Yu-Chen Lin, Hsuan-Tung Liu, Yi-Ren Yeh, Chu-Song Chen. (2023)  
**Domain-Generalized Face Anti-Spoofing with Unknown Attacks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.11758v1)  

---


**ABSTRACT**  
Although face anti-spoofing (FAS) methods have achieved remarkable performance on specific domains or attack types, few studies have focused on the simultaneous presence of domain changes and unknown attacks, which is closer to real application scenarios. To handle domain-generalized unknown attacks, we introduce a new method, DGUA-FAS, which consists of a Transformer-based feature extractor and a synthetic unknown attack sample generator (SUASG). The SUASG network simulates unknown attack samples to assist the training of the feature extractor. Experimental results show that our method achieves superior performance on domain generalization FAS with known or unknown attacks.

{{</citation>}}


### (120/143) BanglaAbuseMeme: A Dataset for Bengali Abusive Meme Classification (Mithun Das et al., 2023)

{{<citation>}}

Mithun Das, Animesh Mukherjee. (2023)  
**BanglaAbuseMeme: A Dataset for Bengali Abusive Meme Classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.11748v1)  

---


**ABSTRACT**  
The dramatic increase in the use of social media platforms for information sharing has also fueled a steep growth in online abuse. A simple yet effective way of abusing individuals or communities is by creating memes, which often integrate an image with a short piece of text layered on top of it. Such harmful elements are in rampant use and are a threat to online safety. Hence it is necessary to develop efficient models to detect and flag abusive memes. The problem becomes more challenging in a low-resource setting (e.g., Bengali memes, i.e., images with Bengali text embedded on it) because of the absence of benchmark datasets on which AI models could be trained. In this paper we bridge this gap by building a Bengali meme dataset. To setup an effective benchmark we implement several baseline models for classifying abusive memes using this dataset. We observe that multimodal models that use both textual and visual information outperform unimodal models. Our best-performing model achieves a macro F1 score of 70.51. Finally, we perform a qualitative error analysis of the misclassified memes of the best-performing text-based, image-based and multimodal models.

{{</citation>}}


### (121/143) VST++: Efficient and Stronger Visual Saliency Transformer (Nian Liu et al., 2023)

{{<citation>}}

Nian Liu, Ziyang Luo, Ni Zhang, Junwei Han. (2023)  
**VST++: Efficient and Stronger Visual Saliency Transformer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2310.11725v1)  

---


**ABSTRACT**  
While previous CNN-based models have exhibited promising results for salient object detection (SOD), their ability to explore global long-range dependencies is restricted. Our previous work, the Visual Saliency Transformer (VST), addressed this constraint from a transformer-based sequence-to-sequence perspective, to unify RGB and RGB-D SOD. In VST, we developed a multi-task transformer decoder that concurrently predicts saliency and boundary outcomes in a pure transformer architecture. Moreover, we introduced a novel token upsampling method called reverse T2T for predicting a high-resolution saliency map effortlessly within transformer-based structures. Building upon the VST model, we further propose an efficient and stronger VST version in this work, i.e. VST++. To mitigate the computational costs of the VST model, we propose a Select-Integrate Attention (SIA) module, partitioning foreground into fine-grained segments and aggregating background information into a single coarse-grained token. To incorporate 3D depth information with low cost, we design a novel depth position encoding method tailored for depth maps. Furthermore, we introduce a token-supervised prediction loss to provide straightforward guidance for the task-related tokens. We evaluate our VST++ model across various transformer-based backbones on RGB, RGB-D, and RGB-T SOD benchmark datasets. Experimental results show that our model outperforms existing methods while achieving a 25% reduction in computational costs without significant performance compromise. The demonstrated strong ability for generalization, enhanced performance, and heightened efficiency of our VST++ model highlight its potential.

{{</citation>}}


### (122/143) ChatGPT-guided Semantics for Zero-shot Learning (Fahimul Hoque Shubho et al., 2023)

{{<citation>}}

Fahimul Hoque Shubho, Townim Faisal Chowdhury, Ali Cheraghian, Morteza Saberi, Nabeel Mohammed, Shafin Rahman. (2023)  
**ChatGPT-guided Semantics for Zero-shot Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2310.11657v1)  

---


**ABSTRACT**  
Zero-shot learning (ZSL) aims to classify objects that are not observed or seen during training. It relies on class semantic description to transfer knowledge from the seen classes to the unseen classes. Existing methods of obtaining class semantics include manual attributes or automatic word vectors from language models (like word2vec). We know attribute annotation is costly, whereas automatic word-vectors are relatively noisy. To address this problem, we explore how ChatGPT, a large language model, can enhance class semantics for ZSL tasks. ChatGPT can be a helpful source to obtain text descriptions for each class containing related attributes and semantics. We use the word2vec model to get a word vector using the texts from ChatGPT. Then, we enrich word vectors by combining the word embeddings from class names and descriptions generated by ChatGPT. More specifically, we leverage ChatGPT to provide extra supervision for the class description, eventually benefiting ZSL models. We evaluate our approach on various 2D image (CUB and AwA) and 3D point cloud (ModelNet10, ModelNet40, and ScanObjectNN) datasets and show that it improves ZSL performance. Our work contributes to the ZSL literature by applying ChatGPT for class semantics enhancement and proposing a novel word vector fusion method.

{{</citation>}}


## cs.MM (1)



### (123/143) Soccer on Social Media (Mehdi Houshmand Sarkhoosh et al., 2023)

{{<citation>}}

Mehdi Houshmand Sarkhoosh, Sayed Mohammad Majidi Dorcheh, Sushant Gautam, Cise Midoglu, Saeed Shafiee Sabet, Pål Halvorsen. (2023)  
**Soccer on Social Media**  

---
Primary Category: cs.MM  
Categories: cs-MM, cs-SI, cs.MM  
Keywords: AI, Social Media  
[Paper Link](http://arxiv.org/abs/2310.12328v1)  

---


**ABSTRACT**  
In the era of digitalization, social media has become an integral part of our lives, serving as a significant hub for individuals and businesses to share information, communicate, and engage. This is also the case for professional sports, where leagues, clubs and players are using social media to reach out to their fans. In this respect, a huge amount of time is spent curating multimedia content for various social media platforms and their target users. With the emergence of Artificial Intelligence (AI), AI-based tools for automating content generation and enhancing user experiences on social media have become widely popular. However, to effectively utilize such tools, it is imperative to comprehend the demographics and preferences of users on different platforms, understand how content providers post information in these channels, and how different types of multimedia are consumed by audiences. This report presents an analysis of social media platforms, in terms of demographics, supported multimedia modalities, and distinct features and specifications for different modalities, followed by a comparative case study of select European soccer leagues and teams, in terms of their social media practices. Through this analysis, we demonstrate that social media, while being very important for and widely used by supporters from all ages, also requires a fine-tuned effort on the part of soccer professionals, in order to elevate fan experiences and foster engagement.

{{</citation>}}


## stat.ML (1)



### (124/143) Preference Optimization for Molecular Language Models (Ryan Park et al., 2023)

{{<citation>}}

Ryan Park, Ryan Theisen, Navriti Sahni, Marcel Patek, Anna Cichońska, Rayees Rahman. (2023)  
**Preference Optimization for Molecular Language Models**  

---
Primary Category: stat.ML  
Categories: cs-AI, cs-LG, stat-ML, stat.ML  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.12304v1)  

---


**ABSTRACT**  
Molecular language modeling is an effective approach to generating novel chemical structures. However, these models do not \emph{a priori} encode certain preferences a chemist may desire. We investigate the use of fine-tuning using Direct Preference Optimization to better align generated molecules with chemist preferences. Our findings suggest that this approach is simple, efficient, and highly effective.

{{</citation>}}


## cs.RO (4)



### (125/143) Plan-Guided Reinforcement Learning for Whole-Body Manipulation (Mengchao Zhang et al., 2023)

{{<citation>}}

Mengchao Zhang, Jose Barreiros, Aykut Ozgun Onol. (2023)  
**Plan-Guided Reinforcement Learning for Whole-Body Manipulation**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.12263v1)  

---


**ABSTRACT**  
Synthesizing complex whole-body manipulation behaviors has fundamental challenges due to the rapidly growing combinatorics inherent to contact interaction planning. While model-based methods have shown promising results in solving long-horizon manipulation tasks, they often work under strict assumptions, such as known model parameters, oracular observation of the environment state, and simplified dynamics, resulting in plans that cannot easily transfer to hardware. Learning-based approaches, such as imitation learning (IL) and reinforcement learning (RL), have been shown to be robust when operating over in-distribution states; however, they need heavy human supervision. Specifically, model-free RL requires a tedious reward-shaping process. IL methods, on the other hand, rely on human demonstrations that involve advanced teleoperation methods. In this work, we propose a plan-guided reinforcement learning (PGRL) framework to combine the advantages of model-based planning and reinforcement learning. Our method requires minimal human supervision because it relies on plans generated by model-based planners to guide the exploration in RL. In exchange, RL derives a more robust policy thanks to domain randomization. We test this approach on a whole-body manipulation task on Punyo, an upper-body humanoid robot with compliant, air-filled arm coverings, to pivot and lift a large box. Our preliminary results indicate that the proposed methodology is promising to address challenges that remain difficult for either model- or learning-based strategies alone.

{{</citation>}}


### (126/143) Few-Shot In-Context Imitation Learning via Implicit Graph Alignment (Vitalis Vosylius et al., 2023)

{{<citation>}}

Vitalis Vosylius, Edward Johns. (2023)  
**Few-Shot In-Context Imitation Learning via Implicit Graph Alignment**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-LG, cs-RO, cs.RO  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2310.12238v1)  

---


**ABSTRACT**  
Consider the following problem: given a few demonstrations of a task across a few different objects, how can a robot learn to perform that same task on new, previously unseen objects? This is challenging because the large variety of objects within a class makes it difficult to infer the task-relevant relationship between the new objects and the objects in the demonstrations. We address this by formulating imitation learning as a conditional alignment problem between graph representations of objects. Consequently, we show that this conditioning allows for in-context learning, where a robot can perform a task on a set of new objects immediately after the demonstrations, without any prior knowledge about the object class or any further training. In our experiments, we explore and validate our design choices, and we show that our method is highly effective for few-shot learning of several real-world, everyday tasks, whilst outperforming baselines. Videos are available on our project webpage at https://www.robot-learning.uk/implicit-graph-alignment.

{{</citation>}}


### (127/143) Flexible Computation Offloading at the Edge for Autonomous Drones with Uncertain Flight Times (Giorgos Polychronis et al., 2023)

{{<citation>}}

Giorgos Polychronis, Spyros Lalis. (2023)  
**Flexible Computation Offloading at the Edge for Autonomous Drones with Uncertain Flight Times**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs-SY, cs.RO, eess-SY  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2310.11895v1)  

---


**ABSTRACT**  
An ever increasing number of applications can employ aerial unmanned vehicles, or so-called drones, to perform different sensing and possibly also actuation tasks from the air. In some cases, the data that is captured at a given point has to be processed before moving to the next one. Drones can exploit nearby edge servers to offload the computation instead of performing it locally. However, doing this in a naive way can be suboptimal if servers have limited computing resources and drones have limited energy resources. In this paper, we propose a protocol and resource reservation scheme for each drone and edge server to decide, in a dynamic and fully decentralized way, whether to offload the computation and respectively whether to accept such an offloading requests, with the objective to evenly reduce the drones' mission times. We evaluate our approach through extensive simulation experiments, showing that it can significantly reduce the mission times compared to a no-offloading scenario by up to 26.2%, while outperforming an offloading schedule that has been computed offline by up to 7.4% as well as a purely opportunistic approach by up to 23.9%.

{{</citation>}}


### (128/143) Bias in Emotion Recognition with ChatGPT (Naoki Wake et al., 2023)

{{<citation>}}

Naoki Wake, Atsushi Kanehira, Kazuhiro Sasabuchi, Jun Takamatsu, Katsushi Ikeuchi. (2023)  
**Bias in Emotion Recognition with ChatGPT**  

---
Primary Category: cs.RO  
Categories: cs-CL, cs-RO, cs.RO  
Keywords: Bias, ChatGPT, Emotion Recognition, GPT  
[Paper Link](http://arxiv.org/abs/2310.11753v1)  

---


**ABSTRACT**  
This technical report explores the ability of ChatGPT in recognizing emotions from text, which can be the basis of various applications like interactive chatbots, data annotation, and mental health analysis. While prior research has shown ChatGPT's basic ability in sentiment analysis, its performance in more nuanced emotion recognition is not yet explored. Here, we conducted experiments to evaluate its performance of emotion recognition across different datasets and emotion labels. Our findings indicate a reasonable level of reproducibility in its performance, with noticeable improvement through fine-tuning. However, the performance varies with different emotion labels and datasets, highlighting an inherent instability and possible bias. The choice of dataset and emotion labels significantly impacts ChatGPT's emotion recognition performance. This paper sheds light on the importance of dataset and label selection, and the potential of fine-tuning in enhancing ChatGPT's emotion recognition capabilities, providing a groundwork for better integration of emotion analysis in applications using ChatGPT.

{{</citation>}}


## cs.DC (1)



### (129/143) Distributed Indexing Schemes for k-Dominant Skyline Analytics on Uncertain Edge-IoT Data (Chuan-Chi Lai et al., 2023)

{{<citation>}}

Chuan-Chi Lai, Hsuan-Yu Lin, Chuan-Ming Liu. (2023)  
**Distributed Indexing Schemes for k-Dominant Skyline Analytics on Uncertain Edge-IoT Data**  

---
Primary Category: cs.DC  
Categories: cs-DB, cs-DC, cs.DC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.12116v1)  

---


**ABSTRACT**  
Skyline queries typically search a Pareto-optimal set from a given data set to solve the corresponding multiobjective optimization problem. As the number of criteria increases, the skyline presumes excessive data items, which yield a meaningless result. To address this curse of dimensionality, we proposed a k-dominant skyline in which the number of skyline members was reduced by relaxing the restriction on the number of dimensions, considering the uncertainty of data. Specifically, each data item was associated with a probability of appearance, which represented the probability of becoming a member of the k-dominant skyline. As data items appear continuously in data streams, the corresponding k-dominant skyline may vary with time. Therefore, an effective and rapid mechanism of updating the k-dominant skyline becomes crucial. Herein, we proposed two time-efficient schemes, Middle Indexing (MI) and All Indexing (AI), for k-dominant skyline in distributed edge-computing environments, where irrelevant data items can be effectively excluded from the compute to reduce the processing duration. Furthermore, the proposed schemes were validated with extensive experimental simulations. The experimental results demonstrated that the proposed MI and AI schemes reduced the computation time by approximately 13% and 56%, respectively, compared with the existing method.

{{</citation>}}


## astro-ph.IM (1)



### (130/143) Transformers for scientific data: a pedagogical review for astronomers (Dimitrios Tanoglidis et al., 2023)

{{<citation>}}

Dimitrios Tanoglidis, Bhuvnesh Jain, Helen Qu. (2023)  
**Transformers for scientific data: a pedagogical review for astronomers**  

---
Primary Category: astro-ph.IM  
Categories: astro-ph-IM, astro-ph.IM, cs-LG  
Keywords: AI, ChatGPT, GPT, Natural Language Processing, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.12069v2)  

---


**ABSTRACT**  
The deep learning architecture associated with ChatGPT and related generative AI products is known as transformers. Initially applied to Natural Language Processing, transformers and the self-attention mechanism they exploit have gained widespread interest across the natural sciences. The goal of this pedagogical and informal review is to introduce transformers to scientists. The review includes the mathematics underlying the attention mechanism, a description of the original transformer architecture, and a section on applications to time series and imaging data in astronomy. We include a Frequently Asked Questions section for readers who are curious about generative AI or interested in getting started with transformers for their research problem.

{{</citation>}}


## cs.SD (3)



### (131/143) Take the aTrain. Introducing an Interface for the Accessible Transcription of Interviews (Armin Haberl et al., 2023)

{{<citation>}}

Armin Haberl, Jürgen Fleiß, Dominik Kowald, Stefan Thalmann. (2023)  
**Take the aTrain. Introducing an Interface for the Accessible Transcription of Interviews**  

---
Primary Category: cs.SD  
Categories: cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: AI, Microsoft  
[Paper Link](http://arxiv.org/abs/2310.11967v1)  

---


**ABSTRACT**  
aTrain is an open-source and offline tool for transcribing audio data in multiple languages with CPU and NVIDIA GPU support. It is specifically designed for researchers using qualitative data generated from various forms of speech interactions with research participants. aTrain requires no programming skills, runs on most computers, does not require an internet connection, and was verified not to upload data to any server. aTrain combines OpenAI's Whisper model with speaker recognition to provide output that integrates with the popular qualitative data analysis software tools MAXQDA and ATLAS.ti. It has an easy-to-use graphical interface and is provided as a Windows-App through the Microsoft Store allowing for simple installation by researchers. The source code is freely available on GitHub. Having developed aTrain with a focus on speed on local computers, we show that the transcription time on current mobile CPUs is around 2 to 3 times the duration of the audio file using the highest-accuracy transcription models. If an entry-level graphics card is available, the transcription speed increases to 20% of the audio duration.

{{</citation>}}


### (132/143) BUT CHiME-7 system description (Martin Karafiát et al., 2023)

{{<citation>}}

Martin Karafiát, Karel Veselý, Igor Szöke, Ladislav Mošner, Karel Beneš, Marcin Witkowski, Germán Barchi, Leonardo Pepino. (2023)  
**BUT CHiME-7 system description**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2310.11921v1)  

---


**ABSTRACT**  
This paper describes the joint effort of Brno University of Technology (BUT), AGH University of Krakow and University of Buenos Aires on the development of Automatic Speech Recognition systems for the CHiME-7 Challenge. We train and evaluate various end-to-end models with several toolkits. We heavily relied on Guided Source Separation (GSS) to convert multi-channel audio to single channel. The ASR is leveraging speech representations from models pre-trained by self-supervised learning, and we do a fusion of several ASR systems. In addition, we modified external data from the LibriSpeech corpus to become a close domain and added it to the training. Our efforts were focused on the far-field acoustic robustness sub-track of Task 1 - Distant Automatic Speech Recognition (DASR), our systems use oracle segmentation.

{{</citation>}}


### (133/143) CLARA: Multilingual Contrastive Learning for Audio Representation Acquisition (Kari A Noriy et al., 2023)

{{<citation>}}

Kari A Noriy, Xiaosong Yang, Marcin Budka, Jian Jun Zhang. (2023)  
**CLARA: Multilingual Contrastive Learning for Audio Representation Acquisition**  

---
Primary Category: cs.SD  
Categories: cs-LG, cs-MM, cs-SD, cs.SD, eess-AS  
Keywords: Contrastive Learning, Multilingual  
[Paper Link](http://arxiv.org/abs/2310.11830v1)  

---


**ABSTRACT**  
This paper proposes a novel framework for multilingual speech and sound representation learning using contrastive learning. The lack of sizeable labelled datasets hinders speech-processing research across languages. Recent advances in contrastive learning provide self-supervised techniques to learn from unlabelled data. Motivated by reducing data dependence and improving generalisation across diverse languages and conditions, we develop a multilingual contrastive framework. This framework enables models to acquire shared representations across languages, facilitating cross-lingual transfer with limited target language data.   Additionally, capturing emotional cues within speech is challenging due to subjective perceptual assessments. By learning expressive representations from diverse, multilingual data in a self-supervised manner, our approach aims to develop speech representations that encode emotive dimensions.   Our method trains encoders on a large corpus of multi-lingual audio data. Data augmentation techniques are employed to expand the dataset. The contrastive learning approach trains the model to maximise agreement between positive pairs and minimise agreement between negative pairs. Extensive experiments demonstrate state-of-the-art performance of the proposed model on emotion recognition, audio classification, and retrieval benchmarks under zero-shot and few-shot conditions. This provides an effective approach for acquiring shared and generalised speech representations across languages and acoustic conditions while encoding latent emotional dimensions.

{{</citation>}}


## eess.IV (2)



### (134/143) A New Multimodal Medical Image Fusion based on Laplacian Autoencoder with Channel Attention (Payal Wankhede et al., 2023)

{{<citation>}}

Payal Wankhede, Manisha Das, Deep Gupta, Petia Radeva, Ashwini M Bakde. (2023)  
**A New Multimodal Medical Image Fusion based on Laplacian Autoencoder with Channel Attention**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2310.11896v1)  

---


**ABSTRACT**  
Medical image fusion combines the complementary information of multimodal medical images to assist medical professionals in the clinical diagnosis of patients' disorders and provide guidance during preoperative and intra-operative procedures. Deep learning (DL) models have achieved end-to-end image fusion with highly robust and accurate fusion performance. However, most DL-based fusion models perform down-sampling on the input images to minimize the number of learnable parameters and computations. During this process, salient features of the source images become irretrievable leading to the loss of crucial diagnostic edge details and contrast of various brain tissues. In this paper, we propose a new multimodal medical image fusion model is proposed that is based on integrated Laplacian-Gaussian concatenation with attention pooling (LGCA). We prove that our model preserves effectively complementary information and important tissue structures.

{{</citation>}}


### (135/143) Cloud-Magnetic Resonance Imaging System: In the Era of 6G and Artificial Intelligence (Yirong Zhou et al., 2023)

{{<citation>}}

Yirong Zhou, Yanhuang Wu, Yuhan Su, Jing Li, Jianyun Cai, Yongfu You, Di Guo, Xiaobo Qu. (2023)  
**Cloud-Magnetic Resonance Imaging System: In the Era of 6G and Artificial Intelligence**  

---
Primary Category: eess.IV  
Categories: cs-AI, eess-IV, eess.IV, physics-med-ph  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.11641v1)  

---


**ABSTRACT**  
Magnetic Resonance Imaging (MRI) plays an important role in medical diagnosis, generating petabytes of image data annually in large hospitals. This voluminous data stream requires a significant amount of network bandwidth and extensive storage infrastructure. Additionally, local data processing demands substantial manpower and hardware investments. Data isolation across different healthcare institutions hinders cross-institutional collaboration in clinics and research. In this work, we anticipate an innovative MRI system and its four generations that integrate emerging distributed cloud computing, 6G bandwidth, edge computing, federated learning, and blockchain technology. This system is called Cloud-MRI, aiming at solving the problems of MRI data storage security, transmission speed, AI algorithm maintenance, hardware upgrading, and collaborative work. The workflow commences with the transformation of k-space raw data into the standardized Imaging Society for Magnetic Resonance in Medicine Raw Data (ISMRMRD) format. Then, the data are uploaded to the cloud or edge nodes for fast image reconstruction, neural network training, and automatic analysis. Then, the outcomes are seamlessly transmitted to clinics or research institutes for diagnosis and other services. The Cloud-MRI system will save the raw imaging data, reduce the risk of data loss, facilitate inter-institutional medical collaboration, and finally improve diagnostic accuracy and work efficiency.

{{</citation>}}


## cs.NI (1)



### (136/143) Building a Graph-based Deep Learning network model from captured traffic traces (Carlos Güemes-Palau et al., 2023)

{{<citation>}}

Carlos Güemes-Palau, Miquel Ferriol Galmés, Albert Cabellos-Aparicio, Pere Barlet-Ros. (2023)  
**Building a Graph-based Deep Learning network model from captured traffic traces**  

---
Primary Category: cs.NI  
Categories: cs-LG, cs-NI, cs.NI  
Keywords: GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2310.11889v1)  

---


**ABSTRACT**  
Currently the state of the art network models are based or depend on Discrete Event Simulation (DES). While DES is highly accurate, it is also computationally costly and cumbersome to parallelize, making it unpractical to simulate high performance networks. Additionally, simulated scenarios fail to capture all of the complexities present in real network scenarios. While there exists network models based on Machine Learning (ML) techniques to minimize these issues, these models are also trained with simulated data and hence vulnerable to the same pitfalls. Consequently, the Graph Neural Networking Challenge 2023 introduces a dataset of captured traffic traces that can be used to build a ML-based network model without these limitations. In this paper we propose a Graph Neural Network (GNN)-based solution specifically designed to better capture the complexities of real network scenarios. This is done through a novel encoding method to capture information from the sequence of captured packets, and an improved message passing algorithm to better represent the dependencies present in physical networks. We show that the proposed solution it is able to learn and generalize to unseen captured network scenarios.

{{</citation>}}


## cs.HC (2)



### (137/143) The Value-Sensitive Conversational Agent Co-Design Framework (Malak Sadek et al., 2023)

{{<citation>}}

Malak Sadek, Rafael A. Calvo, Celine Mougenot. (2023)  
**The Value-Sensitive Conversational Agent Co-Design Framework**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.11848v1)  

---


**ABSTRACT**  
Conversational agents (CAs) are gaining traction in both industry and academia, especially with the advent of generative AI and large language models. As these agents are used more broadly by members of the general public and take on a number of critical use cases and social roles, it becomes important to consider the values embedded in these systems. This consideration includes answering questions such as 'whose values get embedded in these agents?' and 'how do those values manifest in the agents being designed?' Accordingly, the aim of this paper is to present the Value-Sensitive Conversational Agent (VSCA) Framework for enabling the collaborative design (co-design) of value-sensitive CAs with relevant stakeholders. Firstly, requirements for co-designing value-sensitive CAs which were identified in previous works are summarised here. Secondly, the practical framework is presented and discussed, including its operationalisation into a design toolkit. The framework facilitates the co-design of three artefacts that elicit stakeholder values and have a technical utility to CA teams to guide CA implementation, enabling the creation of value-embodied CA prototypes. Finally, an evaluation protocol for the framework is proposed where the effects of the framework and toolkit are explored in a design workshop setting to evaluate both the process followed and the outcomes produced.

{{</citation>}}


### (138/143) AdaVis: Adaptive and Explainable Visualization Recommendation for Tabular Data (Songheng Zhang et al., 2023)

{{<citation>}}

Songheng Zhang, Haotian Li, Huamin Qu, Yong Wang. (2023)  
**AdaVis: Adaptive and Explainable Visualization Recommendation for Tabular Data**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2310.11742v1)  

---


**ABSTRACT**  
Automated visualization recommendation facilitates the rapid creation of effective visualizations, which is especially beneficial for users with limited time and limited knowledge of data visualization. There is an increasing trend in leveraging machine learning (ML) techniques to achieve an end-to-end visualization recommendation. However, existing ML-based approaches implicitly assume that there is only one appropriate visualization for a specific dataset, which is often not true for real applications. Also, they often work like a black box, and are difficult for users to understand the reasons for recommending specific visualizations. To fill the research gap, we propose AdaVis, an adaptive and explainable approach to recommend one or multiple appropriate visualizations for a tabular dataset. It leverages a box embedding-based knowledge graph to well model the possible one-to-many mapping relations among different entities (i.e., data features, dataset columns, datasets, and visualization choices). The embeddings of the entities and relations can be learned from dataset-visualization pairs. Also, AdaVis incorporates the attention mechanism into the inference framework. Attention can indicate the relative importance of data features for a dataset and provide fine-grained explainability. Our extensive evaluations through quantitative metric evaluations, case studies, and user interviews demonstrate the effectiveness of AdaVis.

{{</citation>}}


## cs.CY (1)



### (139/143) Language Agents for Detecting Implicit Stereotypes in Text-to-image Models at Scale (Qichao Wang et al., 2023)

{{<citation>}}

Qichao Wang, Tian Bian, Yian Yin, Tingyang Xu, Hong Cheng, Helen M. Meng, Zibin Zheng, Liang Chen, Bingzhe Wu. (2023)  
**Language Agents for Detecting Implicit Stereotypes in Text-to-image Models at Scale**  

---
Primary Category: cs.CY  
Categories: cs-CL, cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.11778v1)  

---


**ABSTRACT**  
The recent surge in the research of diffusion models has accelerated the adoption of text-to-image models in various Artificial Intelligence Generated Content (AIGC) commercial products. While these exceptional AIGC products are gaining increasing recognition and sparking enthusiasm among consumers, the questions regarding whether, when, and how these models might unintentionally reinforce existing societal stereotypes remain largely unaddressed. Motivated by recent advancements in language agents, here we introduce a novel agent architecture tailored for stereotype detection in text-to-image models. This versatile agent architecture is capable of accommodating free-form detection tasks and can autonomously invoke various tools to facilitate the entire process, from generating corresponding instructions and images, to detecting stereotypes. We build the stereotype-relevant benchmark based on multiple open-text datasets, and apply this architecture to commercial products and popular open source text-to-image models. We find that these models often display serious stereotypes when it comes to certain prompts about personal characteristics, social cultural context and crime-related aspects. In summary, these empirical findings underscore the pervasive existence of stereotypes across social dimensions, including gender, race, and religion, which not only validate the effectiveness of our proposed approach, but also emphasize the critical necessity of addressing potential ethical risks in the burgeoning realm of AIGC. As AIGC continues its rapid expansion trajectory, with new models and plugins emerging daily in staggering numbers, the challenge lies in the timely detection and mitigation of potential biases within these models.

{{</citation>}}


## cs.OH (1)



### (140/143) Solving the multiplication problem of a large language model system using a graph-based method (Turker Tuncer et al., 2023)

{{<citation>}}

Turker Tuncer, Sengul Dogan, Mehmet Baygin, Prabal Datta Barua, Abdul Hafeez-Baig, Ru-San Tan, Subrata Chakraborty, U. Rajendra Acharya. (2023)  
**Solving the multiplication problem of a large language model system using a graph-based method**  

---
Primary Category: cs.OH  
Categories: cs-AI, cs-OH, cs.OH  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2310.13016v1)  

---


**ABSTRACT**  
The generative pre-trained transformer (GPT)-based chatbot software ChatGPT possesses excellent natural language processing capabilities but is inadequate for solving arithmetic problems, especially multiplication. Its GPT structure uses a computational graph for multiplication, which has limited accuracy beyond simple multiplication operations. We developed a graph-based multiplication algorithm that emulated human-like numerical operations by incorporating a 10k operator, where k represents the maximum power to base 10 of the larger of two input numbers. Our proposed algorithm attained 100% accuracy for 1,000,000 large number multiplication tasks, effectively solving the multiplication challenge of GPT-based and other large language models. Our work highlights the importance of blending simple human insights into the design of artificial intelligence algorithms. Keywords: Graph-based multiplication; ChatGPT; Multiplication problem

{{</citation>}}


## eess.SY (2)



### (141/143) A Security-Constrained Optimal Power Management Algorithm for Shipboard Microgrids with Battery Energy Storage System and Fuel Cell (Fabio D'Agostino et al., 2023)

{{<citation>}}

Fabio D'Agostino, Marco Gallo, Matteo Saviozzi, Federico Silvestro. (2023)  
**A Security-Constrained Optimal Power Management Algorithm for Shipboard Microgrids with Battery Energy Storage System and Fuel Cell**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2310.11760v1)  

---


**ABSTRACT**  
This work proposes an optimal power management strategy for shipboard microgrids equipped with diesel generators, a fuel cell and a battery energy storage system. The optimization aims to determine both the unit commitment and the optimal power dispatch for all resources to ensure a reliable power supply at minimum cost and with minimal environmental impact. This strategy takes into account the zero-emission capability of the ship and incorporates a soft constraint related to the ship's speed. The optimization is performed solving a mixed integer linear programming problem, where the constraints are defined according to the operational limits of the resources when a contingency occurs. The algorithm is tested on a notional all-electric ship where the electrical load is generated through a Markov chain, modelled on real measurement data. The results show that the proposed power management strategy successfully maximizes fuel and emission savings while ensuring blackout prevention capability.

{{</citation>}}


### (142/143) Deep learning based on Transformer architecture for power system short-term voltage stability assessment with class imbalance (Yang Li et al., 2023)

{{<citation>}}

Yang Li, Jiting Cao, Yan Xu, Lipeng Zhu, Zhao Yang Dong. (2023)  
**Deep learning based on Transformer architecture for power system short-term voltage stability assessment with class imbalance**  

---
Primary Category: eess.SY  
Categories: cs-LG, cs-SY, eess-SY, eess.SY  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.11690v1)  

---


**ABSTRACT**  
Most existing data-driven power system short-term voltage stability assessment (STVSA) approaches presume class-balanced input data. However, in practical applications, the occurrence of short-term voltage instability following a disturbance is minimal, leading to a significant class imbalance problem and a consequent decline in classifier performance. This work proposes a Transformer-based STVSA method to address this challenge. By utilizing the basic Transformer architecture, a stability assessment Transformer (StaaT) is developed {as a classification model to reflect the correlation between the operational states of the system and the resulting stability outcomes}. To combat the negative impact of imbalanced datasets, this work employs a conditional Wasserstein generative adversarial network with gradient penalty (CWGAN-GP) for synthetic data generation, aiding in the creation of a balanced, representative training set for the classifier. Semi-supervised clustering learning is implemented to enhance clustering quality, addressing the lack of a unified quantitative criterion for short-term voltage stability. {Numerical tests on the IEEE 39-bus test system extensively demonstrate that the proposed method exhibits robust performance under class imbalances up to 100:1 and noisy environments, and maintains consistent effectiveness even with an increased penetration of renewable energy}. Comparative results reveal that the CWGAN-GP generates more balanced datasets than traditional oversampling methods and that the StaaT outperforms other deep learning algorithms. This study presents a compelling solution for real-world STVSA applications that often face class imbalance and data noise challenges.

{{</citation>}}


## cs.LO (1)



### (143/143) A Symbolic Language for Interpreting Decision Trees (Marcelo Arenas et al., 2023)

{{<citation>}}

Marcelo Arenas, Pablo Barcelo, Diego Bustamente, Jose Caraball, Bernardo Subercaseaux. (2023)  
**A Symbolic Language for Interpreting Decision Trees**  

---
Primary Category: cs.LO  
Categories: cs-AI, cs-LO, cs.LO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.11636v1)  

---


**ABSTRACT**  
The recent development of formal explainable AI has disputed the folklore claim that "decision trees are readily interpretable models", showing different interpretability queries that are computationally hard on decision trees, as well as proposing different methods to deal with them in practice. Nonetheless, no single explainability query or score works as a "silver bullet" that is appropriate for every context and end-user. This naturally suggests the possibility of "interpretability languages" in which a wide variety of queries can be expressed, giving control to the end-user to tailor queries to their particular needs. In this context, our work presents ExplainDT, a symbolic language for interpreting decision trees. ExplainDT is rooted in a carefully constructed fragment of first-ordered logic that we call StratiFOILed. StratiFOILed balances expressiveness and complexity of evaluation, allowing for the computation of many post-hoc explanations--both local (e.g., abductive and contrastive explanations) and global ones (e.g., feature relevancy)--while remaining in the Boolean Hierarchy over NP. Furthermore, StratiFOILed queries can be written as a Boolean combination of NP-problems, thus allowing us to evaluate them in practice with a constant number of calls to a SAT solver. On the theoretical side, our main contribution is an in-depth analysis of the expressiveness and complexity of StratiFOILed, while on the practical side, we provide an optimized implementation for encoding StratiFOILed queries as propositional formulas, together with an experimental study on its efficiency.

{{</citation>}}
