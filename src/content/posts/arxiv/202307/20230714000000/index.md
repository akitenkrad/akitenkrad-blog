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

- [cs.LG (15)](#cslg-15)
- [cs.DB (1)](#csdb-1)
- [cs.SE (3)](#csse-3)
- [cs.CR (3)](#cscr-3)
- [math.OC (1)](#mathoc-1)
- [eess.IV (3)](#eessiv-3)
- [cs.CV (23)](#cscv-23)
- [cs.CL (10)](#cscl-10)
- [cs.RO (5)](#csro-5)
- [stat.ML (1)](#statml-1)
- [eess.AS (1)](#eessas-1)
- [cs.MA (1)](#csma-1)
- [cs.AI (5)](#csai-5)
- [cs.SD (2)](#cssd-2)
- [cs.MM (1)](#csmm-1)
- [cs.IR (1)](#csir-1)
- [cs.AR (1)](#csar-1)
- [cs.ET (1)](#cset-1)
- [cs.NI (1)](#csni-1)
- [cs.HC (1)](#cshc-1)
- [eess.SY (1)](#eesssy-1)

## cs.LG (15)



### (1/81) Misclassification in Automated Content Analysis Causes Bias in Regression. Can We Fix It? Yes We Can! (Nathan TeBlunthuis et al., 2023)

{{<citation>}}

Nathan TeBlunthuis, Valerie Hase, Chung-Hong Chan. (2023)  
**Misclassification in Automated Content Analysis Causes Bias in Regression. Can We Fix It? Yes We Can!**  

---
Primary Category: cs.LG  
Categories: G-3; K-4-0; I-2-6, cs-AI, cs-CL, cs-CY, cs-LG, cs.LG  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2307.06483v1)  

---


**ABSTRACT**  
Automated classifiers (ACs), often built via supervised machine learning (SML), can categorize large, statistically powerful samples of data ranging from text to images and video, and have become widely popular measurement devices in communication science and related fields. Despite this popularity, even highly accurate classifiers make errors that cause misclassification bias and misleading results in downstream analyses-unless such analyses account for these errors. As we show in a systematic literature review of SML applications, communication scholars largely ignore misclassification bias. In principle, existing statistical methods can use "gold standard" validation data, such as that created by human annotators, to correct misclassification bias and produce consistent estimates. We introduce and test such methods, including a new method we design and implement in the R package misclassificationmodels, via Monte Carlo simulations designed to reveal each method's limitations, which we also release. Based on our results, we recommend our new error correction method as it is versatile and efficient. In sum, automated classifiers, even those below common accuracy standards or making systematic misclassifications, can be useful for measurement with careful study design and appropriate error correction methods.

{{</citation>}}


### (2/81) No Train No Gain: Revisiting Efficient Training Algorithms For Transformer-based Language Models (Jean Kaddour et al., 2023)

{{<citation>}}

Jean Kaddour, Oscar Key, Piotr Nawrot, Pasquale Minervini, Matt J. Kusner. (2023)  
**No Train No Gain: Revisiting Efficient Training Algorithms For Transformer-based Language Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs-NE, cs-PF, cs.LG  
Keywords: BERT, Language Model, T5, Transformer  
[Paper Link](http://arxiv.org/abs/2307.06440v1)  

---


**ABSTRACT**  
The computation necessary for training Transformer-based language models has skyrocketed in recent years. This trend has motivated research on efficient training algorithms designed to improve training, validation, and downstream performance faster than standard training. In this work, we revisit three categories of such algorithms: dynamic architectures (layer stacking, layer dropping), batch selection (selective backprop, RHO loss), and efficient optimizers (Lion, Sophia). When pre-training BERT and T5 with a fixed computation budget using such methods, we find that their training, validation, and downstream gains vanish compared to a baseline with a fully-decayed learning rate. We define an evaluation protocol that enables computation to be done on arbitrary machines by mapping all computation time to a reference machine which we call reference system time. We discuss the limitations of our proposed protocol and release our code to encourage rigorous research in efficient training procedures: https://github.com/JeanKaddour/NoTrainNoGain.

{{</citation>}}


### (3/81) Differentially Private Decoupled Graph Convolutions for Multigranular Topology Protection (Eli Chien et al., 2023)

{{<citation>}}

Eli Chien, Wei-Ning Chen, Chao Pan, Pan Li, Ayfer Özgür, Olgica Milenkovic. (2023)  
**Differentially Private Decoupled Graph Convolutions for Multigranular Topology Protection**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2307.06422v1)  

---


**ABSTRACT**  
Graph learning methods, such as Graph Neural Networks (GNNs) based on graph convolutions, are highly successful in solving real-world learning problems involving graph-structured data. However, graph learning methods expose sensitive user information and interactions not only through their model parameters but also through their model predictions. Consequently, standard Differential Privacy (DP) techniques that merely offer model weight privacy are inadequate. This is especially the case for node predictions that leverage neighboring node attributes directly via graph convolutions that create additional risks of privacy leakage. To address this problem, we introduce Graph Differential Privacy (GDP), a new formal DP framework tailored to graph learning settings that ensures both provably private model parameters and predictions. Furthermore, since there may be different privacy requirements for the node attributes and graph structure, we introduce a novel notion of relaxed node-level data adjacency. This relaxation can be used for establishing guarantees for different degrees of graph topology privacy while maintaining node attribute privacy. Importantly, this relaxation reveals a useful trade-off between utility and topology privacy for graph learning methods. In addition, our analysis of GDP reveals that existing DP-GNNs fail to exploit this trade-off due to the complex interplay between graph topology and attribute data in standard graph convolution designs. To mitigate this problem, we introduce the Differentially Private Decoupled Graph Convolution (DPDGC) model, which benefits from decoupled graph convolution while providing GDP guarantees. Extensive experiments on seven node classification benchmarking datasets demonstrate the superior privacy-utility trade-off of DPDGC over existing DP-GNNs based on standard graph convolution design.

{{</citation>}}


### (4/81) Personalized Anomaly Detection in PPG Data using Representation Learning and Biometric Identification (Ramin Ghorbani et al., 2023)

{{<citation>}}

Ramin Ghorbani, Marcel J. T. Reinders, David M. J. Tax. (2023)  
**Personalized Anomaly Detection in PPG Data using Representation Learning and Biometric Identification**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, eess-SP  
Keywords: Anomaly Detection, Representation Learning  
[Paper Link](http://arxiv.org/abs/2307.06380v1)  

---


**ABSTRACT**  
Photoplethysmography (PPG) signals, typically acquired from wearable devices, hold significant potential for continuous fitness-health monitoring. In particular, heart conditions that manifest in rare and subtle deviating heart patterns may be interesting. However, robust and reliable anomaly detection within these data remains a challenge due to the scarcity of labeled data and high inter-subject variability. This paper introduces a two-stage framework leveraging representation learning and personalization to improve anomaly detection performance in PPG data. The proposed framework first employs representation learning to transform the original PPG signals into a more discriminative and compact representation. We then apply three different unsupervised anomaly detection methods for movement detection and biometric identification. We validate our approach using two different datasets in both generalized and personalized scenarios. The results show that representation learning significantly improves anomaly detection performance while reducing the high inter-subject variability. Personalized models further enhance anomaly detection performance, underscoring the role of personalization in PPG-based fitness-health monitoring systems. The results from biometric identification show that it's easier to distinguish a new user from one intended authorized user than from a group of users. Overall, this study provides evidence of the effectiveness of representation learning and personalization for anomaly detection in PPG data.

{{</citation>}}


### (5/81) FDAPT: Federated Domain-adaptive Pre-training for Language Models (Lekang Jiang et al., 2023)

{{<citation>}}

Lekang Jiang, Filip Svoboda, Nicholas D. Lane. (2023)  
**FDAPT: Federated Domain-adaptive Pre-training for Language Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-DC, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.06933v1)  

---


**ABSTRACT**  
Combining Domain-adaptive Pre-training (DAPT) with Federated Learning (FL) can enhance model adaptation by leveraging more sensitive and distributed data while preserving data privacy. However, few studies have focused on this method. Therefore, we conduct the first comprehensive empirical study to evaluate the performance of Federated Domain-adaptive Pre-training (FDAPT). We demonstrate that FDAPT can maintain competitive downstream task performance to the centralized baseline in both IID and non-IID situations. Furthermore, we propose a novel algorithm, Frozen Federated Domain-adaptive Pre-training (FFDAPT). FFDAPT improves the computational efficiency by 12.1% on average and exhibits similar downstream task performance to standard FDAPT, with general performance fluctuations remaining less than 1%. Finally, through a critical evaluation of our work, we identify promising future research directions for this new research area.

{{</citation>}}


### (6/81) DSSE: a drone swarm search environment (Manuel Castanares et al., 2023)

{{<citation>}}

Manuel Castanares, Luis F. S. Carrete, Enrico F. Damiani, Leonardo D. M. de Abreu, José Fernando B. Brancalion, Fabrício J. Barth. (2023)  
**DSSE: a drone swarm search environment**  

---
Primary Category: cs.LG  
Categories: I-2-6; I-6-7, cs-AI, cs-LG, cs-RO, cs-SY, cs.LG, eess-SY  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2307.06240v1)  

---


**ABSTRACT**  
The Drone Swarm Search project is an environment, based on PettingZoo, that is to be used in conjunction with multi-agent (or single-agent) reinforcement learning algorithms. It is an environment in which the agents (drones), have to find the targets (shipwrecked people). The agents do not know the position of the target and do not receive rewards related to their own distance to the target(s). However, the agents receive the probabilities of the target(s) being in a certain cell of the map. The aim of this project is to aid in the study of reinforcement learning algorithms that require dynamic probabilities as inputs.

{{</citation>}}


### (7/81) Unified Molecular Modeling via Modality Blending (Qiying Yu et al., 2023)

{{<citation>}}

Qiying Yu, Yudi Zhang, Yuyan Ni, Shikun Feng, Yanyan Lan, Hao Zhou, Jingjing Liu. (2023)  
**Unified Molecular Modeling via Modality Blending**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, q-bio-BM  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.06235v1)  

---


**ABSTRACT**  
Self-supervised molecular representation learning is critical for molecule-based tasks such as AI-assisted drug discovery. Recent studies consider leveraging both 2D and 3D information for representation learning, with straightforward alignment strategies that treat each modality separately. In this work, we introduce a novel "blend-then-predict" self-supervised learning method (MoleBLEND), which blends atom relations from different modalities into one unified relation matrix for encoding, then recovers modality-specific information for both 2D and 3D structures. By treating atom relationships as anchors, seemingly dissimilar 2D and 3D manifolds are aligned and integrated at fine-grained relation-level organically. Extensive experiments show that MoleBLEND achieves state-of-the-art performance across major 2D/3D benchmarks. We further provide theoretical insights from the perspective of mutual-information maximization, demonstrating that our method unifies contrastive, generative (inter-modal prediction) and mask-then-predict (intra-modal prediction) objectives into a single cohesive blend-then-predict framework.

{{</citation>}}


### (8/81) NetGPT: A Native-AI Network Architecture Beyond Provisioning Personalized Generative Services (Yuxuan Chen et al., 2023)

{{<citation>}}

Yuxuan Chen, Rongpeng Li, Zhifeng Zhao, Chenghui Peng, Jianjun Wu, Ekram Hossain, Honggang Zhang. (2023)  
**NetGPT: A Native-AI Network Architecture Beyond Provisioning Personalized Generative Services**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI, GPT, LLaMA  
[Paper Link](http://arxiv.org/abs/2307.06148v1)  

---


**ABSTRACT**  
Large language models (LLMs) have triggered tremendous success to empower daily life by generative information, and the personalization of LLMs could further contribute to their applications due to better alignment with human intents. Towards personalized generative services, a collaborative cloud-edge methodology sounds promising, as it facilitates the effective orchestration of heterogeneous distributed communication and computing resources. In this article, after discussing the pros and cons of several candidate cloud-edge collaboration techniques, we put forward NetGPT to capably deploy appropriate LLMs at the edge and the cloud in accordance with their computing capacity. In addition, edge LLMs could efficiently leverage location-based information for personalized prompt completion, thus benefiting the interaction with cloud LLMs. After deploying representative open-source LLMs (e.g., GPT-2-base and LLaMA model) at the edge and the cloud, we present the feasibility of NetGPT on the basis of low-rank adaptation-based light-weight fine-tuning. Subsequently, we highlight substantial essential changes required for a native artificial intelligence (AI) network architecture towards NetGPT, with special emphasis on deeper integration of communications and computing resources and careful calibration of logical AI workflow. Furthermore, we demonstrate several by-product benefits of NetGPT, given edge LLM's astonishing capability to predict trends and infer intents, which possibly leads to a unified solution for intelligent network management \& orchestration. In a nutshell, we argue that NetGPT is a promising native-AI network architecture beyond provisioning personalized generative services.

{{</citation>}}


### (9/81) Learning Stochastic Dynamical Systems as an Implicit Regularization with Graph Neural Networks (Jin Guo et al., 2023)

{{<citation>}}

Jin Guo, Ting Gao, Yufu Lan, Peng Zhang, Sikun Yang, Jinqiao Duan. (2023)  
**Learning Stochastic Dynamical Systems as an Implicit Regularization with Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, math-DS  
Keywords: Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2307.06097v1)  

---


**ABSTRACT**  
Stochastic Gumbel graph networks are proposed to learn high-dimensional time series, where the observed dimensions are often spatially correlated. To that end, the observed randomness and spatial-correlations are captured by learning the drift and diffusion terms of the stochastic differential equation with a Gumble matrix embedding, respectively. In particular, this novel framework enables us to investigate the implicit regularization effect of the noise terms in S-GGNs. We provide a theoretical guarantee for the proposed S-GGNs by deriving the difference between the two corresponding loss functions in a small neighborhood of weight. Then, we employ Kuramoto's model to generate data for comparing the spectral density from the Hessian Matrix of the two loss functions. Experimental results on real-world data, demonstrate that S-GGNs exhibit superior convergence, robustness, and generalization, compared with state-of-the-arts.

{{</citation>}}


### (10/81) An OOD Multi-Task Perspective for Link Prediction with New Relation Types and Nodes (Jincheng Zhou et al., 2023)

{{<citation>}}

Jincheng Zhou, Beatrice Bevilacqua, Bruno Ribeiro. (2023)  
**An OOD Multi-Task Perspective for Link Prediction with New Relation Types and Nodes**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2307.06046v1)  

---


**ABSTRACT**  
The task of inductive link prediction in (discrete) attributed multigraphs infers missing attributed links (relations) between nodes in new test multigraphs. Traditional relational learning methods face the challenge of limited generalization to OOD test multigraphs containing both novel nodes and novel relation types not seen in training. Recently, under the only assumption that all relation types share the same structural predictive patterns (single task), Gao et al. (2023) proposed an OOD link prediction method using the theoretical concept of double exchangeability (for nodes & relation types), in contrast to the (single) exchangeability (only for nodes) used to design Graph Neural Networks (GNNs). In this work we further extend the double exchangeability concept to multi-task double exchangeability, where we define link prediction in attributed multigraphs that can have distinct and potentially conflicting predictive patterns for different sets of relation types (multiple tasks). Our empirical results on real-world datasets demonstrate that our approach can effectively generalize to entirely new relation types in test, without access to additional information, yielding significant performance improvements over existing methods.

{{</citation>}}


### (11/81) Transformers in Reinforcement Learning: A Survey (Pranav Agarwal et al., 2023)

{{<citation>}}

Pranav Agarwal, Aamer Abdul Rahman, Pierre-Luc St-Charles, Simon J. D. Prince, Samira Ebrahimi Kahou. (2023)  
**Transformers in Reinforcement Learning: A Survey**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: Reinforcement Learning, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.05979v1)  

---


**ABSTRACT**  
Transformers have significantly impacted domains like natural language processing, computer vision, and robotics, where they improve performance compared to other neural networks. This survey explores how transformers are used in reinforcement learning (RL), where they are seen as a promising solution for addressing challenges such as unstable training, credit assignment, lack of interpretability, and partial observability. We begin by providing a brief domain overview of RL, followed by a discussion on the challenges of classical RL algorithms. Next, we delve into the properties of the transformer and its variants and discuss the characteristics that make them well-suited to address the challenges inherent in RL. We examine the application of transformers to various aspects of RL, including representation learning, transition and reward function modeling, and policy optimization. We also discuss recent research that aims to enhance the interpretability and efficiency of transformers in RL, using visualization techniques and efficient training strategies. Often, the transformer architecture must be tailored to the specific needs of a given application. We present a broad overview of how transformers have been adapted for several applications, including robotics, medicine, language modeling, cloud computing, and combinatorial optimization. We conclude by discussing the limitations of using transformers in RL and assess their potential for catalyzing future breakthroughs in this field.

{{</citation>}}


### (12/81) Newell's theory based feature transformations for spatio-temporal traffic prediction (Agnimitra Sengupta et al., 2023)

{{<citation>}}

Agnimitra Sengupta, S. Ilgin Guler. (2023)  
**Newell's theory based feature transformations for spatio-temporal traffic prediction**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2307.05949v2)  

---


**ABSTRACT**  
Deep learning (DL) models for spatio-temporal traffic flow forecasting employ convolutional or graph-convolutional filters along with recurrent neural networks to capture spatial and temporal dependencies in traffic data. These models, such as CNN-LSTM, utilize traffic flows from neighboring detector stations to predict flows at a specific location of interest. However, these models are limited in their ability to capture the broader dynamics of the traffic system, as they primarily learn features specific to the detector configuration and traffic characteristics at the target location. Hence, the transferability of these models to different locations becomes challenging, particularly when data is unavailable at the new location for model training. To address this limitation, we propose a traffic flow physics-based feature transformation for spatio-temporal DL models. This transformation incorporates Newell's uncongested and congested-state estimators of traffic flows at the target locations, enabling the models to learn broader dynamics of the system. Our methodology is empirically validated using traffic data from two different locations. The results demonstrate that the proposed feature transformation improves the models' performance in predicting traffic flows over different prediction horizons, as indicated by better goodness-of-fit statistics. An important advantage of our framework is its ability to be transferred to new locations where data is unavailable. This is achieved by appropriately accounting for spatial dependencies based on station distances and various traffic parameters. In contrast, regular DL models are not easily transferable as their inputs remain fixed. It should be noted that due to data limitations, we were unable to perform spatial sensitivity analysis, which calls for further research using simulated data.

{{</citation>}}


### (13/81) Prompt Generate Train (PGT): A framework for few-shot domain adaptation, alignment, and uncertainty calibration of a retriever augmented generation (RAG) model for domain specific open book question-answering (C. S. Krishna, 2023)

{{<citation>}}

C. S. Krishna. (2023)  
**Prompt Generate Train (PGT): A framework for few-shot domain adaptation, alignment, and uncertainty calibration of a retriever augmented generation (RAG) model for domain specific open book question-answering**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GPT, GPT-4, T5  
[Paper Link](http://arxiv.org/abs/2307.05915v1)  

---


**ABSTRACT**  
We present a framework - Prompt, Generate, Train (PGT) - to efficiently develop a generative question-answering model for open-book question-answering over a proprietary collection of text documents. The framework adapts a retriever augmented generation model to the target domain using supervised finetuning and reinforcement learning with synthetic feedback in a few-shot setting. This yields an aligned, uncertainty calibrated model that is competitive with GPT-4 based in-context retrieval augmented generation in generating relevant answers at lower serving costs. The synthetic generation pipeline generates high quality synthetic training data musing a medium sized LLM, Flan-T5 XXL, and a novel consistency filtering scheme. The pipeline is designed to generate both abstractive and extractive questions that span the entire corpus. Using samples from this dataset, the framework fine-tunes a smaller RAG model comprising a dense retriever and a smaller sized LLM on samples from the dataset. In parallel, the framework trains a Reward model to score domain grounded answers higher than hallucinated answers. In the next phase, the framework aligns to the RAG model with the target domain using reinforcement learning. This step improves the RAG model's ability to generate grounded answers and ignore out of domain questions. In the final phase, the framework calibrates the model uncertainty for extractive question-answers. This is a desirable feature since the model can be integrated into a cascading system where the RAG model's answer is surfaced only when the model is confident of its answer.

{{</citation>}}


### (14/81) PID-Inspired Inductive Biases for Deep Reinforcement Learning in Partially Observable Control Tasks (Ian Char et al., 2023)

{{<citation>}}

Ian Char, Jeff Schneider. (2023)  
**PID-Inspired Inductive Biases for Deep Reinforcement Learning in Partially Observable Control Tasks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Bias, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.05891v1)  

---


**ABSTRACT**  
Deep reinforcement learning (RL) has shown immense potential for learning to control systems through data alone. However, one challenge deep RL faces is that the full state of the system is often not observable. When this is the case, the policy needs to leverage the history of observations to infer the current state. At the same time, differences between the training and testing environments makes it critical for the policy not to overfit to the sequence of observations it sees at training time. As such, there is an important balancing act between having the history encoder be flexible enough to extract relevant information, yet be robust to changes in the environment. To strike this balance, we look to the PID controller for inspiration. We assert the PID controller's success shows that only summing and differencing are needed to accumulate information over time for many control tasks. Following this principle, we propose two architectures for encoding history: one that directly uses PID features and another that extends these core ideas and can be used in arbitrary control tasks. When compared with prior approaches, our encoders produce policies that are often more robust and achieve better performance on a variety of tracking tasks. Going beyond tracking tasks, our policies achieve 1.7x better performance on average over previous state-of-the-art methods on a suite of high dimensional control tasks.

{{</citation>}}


### (15/81) FAIRO: Fairness-aware Adaptation in Sequential-Decision Making for Human-in-the-Loop Systems (Tianyu Zhao et al., 2023)

{{<citation>}}

Tianyu Zhao, Mojtaba Taherisadr, Salma Elmalaki. (2023)  
**FAIRO: Fairness-aware Adaptation in Sequential-Decision Making for Human-in-the-Loop Systems**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CY, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.05857v1)  

---


**ABSTRACT**  
Achieving fairness in sequential-decision making systems within Human-in-the-Loop (HITL) environments is a critical concern, especially when multiple humans with different behavior and expectations are affected by the same adaptation decisions in the system. This human variability factor adds more complexity since policies deemed fair at one point in time may become discriminatory over time due to variations in human preferences resulting from inter- and intra-human variability. This paper addresses the fairness problem from an equity lens, considering human behavior variability, and the changes in human preferences over time. We propose FAIRO, a novel algorithm for fairness-aware sequential-decision making in HITL adaptation, which incorporates these notions into the decision-making process. In particular, FAIRO decomposes this complex fairness task into adaptive sub-tasks based on individual human preferences through leveraging the Options reinforcement learning framework. We design FAIRO to generalize to three types of HITL application setups that have the shared adaptation decision problem. Furthermore, we recognize that fairness-aware policies can sometimes conflict with the application's utility. To address this challenge, we provide a fairness-utility tradeoff in FAIRO, allowing system designers to balance the objectives of fairness and utility based on specific application requirements. Extensive evaluations of FAIRO on the three HITL applications demonstrate its generalizability and effectiveness in promoting fairness while accounting for human variability. On average, FAIRO can improve fairness compared with other methods across all three applications by 35.36%.

{{</citation>}}


## cs.DB (1)



### (16/81) WiscSort: External Sorting For Byte-Addressable Storage (Vinay Banakar et al., 2023)

{{<citation>}}

Vinay Banakar, Kan Wu, Yuvraj Patel, Kimberly Keeton, Andrea C. Arpaci-Dusseau, Remzi H. Arpaci-Dusseau. (2023)  
**WiscSort: External Sorting For Byte-Addressable Storage**  

---
Primary Category: cs.DB  
Categories: cs-DB, cs-PF, cs.DB  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.06476v1)  

---


**ABSTRACT**  
We present WiscSort, a new approach to high-performance concurrent sorting for existing and future byte-addressable storage (BAS) devices. WiscSort carefully reduces writes, exploits random reads by splitting keys and values during sorting, and performs interference-aware scheduling with thread pool sizing to avoid I/O bandwidth degradation. We introduce the BRAID model which encompasses the unique characteristics of BAS devices. Many state-of-the-art sorting systems do not comply with the BRAID model and deliver sub-optimal performance, whereas WiscSort demonstrates the effectiveness of complying with BRAID. We show that WiscSort is 2-7x faster than competing approaches on a standard sort benchmark. We evaluate the effectiveness of key-value separation on different key-value sizes and compare our concurrency optimizations with various other concurrency models. Finally, we emulate generic BAS devices and show how our techniques perform well with various combinations of hardware properties.

{{</citation>}}


## cs.SE (3)



### (17/81) Assessing the Ability of ChatGPT to Screen Articles for Systematic Reviews (Eugene Syriani et al., 2023)

{{<citation>}}

Eugene Syriani, Istvan David, Gauransh Kumar. (2023)  
**Assessing the Ability of ChatGPT to Screen Articles for Systematic Reviews**  

---
Primary Category: cs.SE  
Categories: cs-CL, cs-IR, cs-SE, cs.SE  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2307.06464v1)  

---


**ABSTRACT**  
By organizing knowledge within a research field, Systematic Reviews (SR) provide valuable leads to steer research. Evidence suggests that SRs have become first-class artifacts in software engineering. However, the tedious manual effort associated with the screening phase of SRs renders these studies a costly and error-prone endeavor. While screening has traditionally been considered not amenable to automation, the advent of generative AI-driven chatbots, backed with large language models is set to disrupt the field. In this report, we propose an approach to leverage these novel technological developments for automating the screening of SRs. We assess the consistency, classification performance, and generalizability of ChatGPT in screening articles for SRs and compare these figures with those of traditional classifiers used in SR automation. Our results indicate that ChatGPT is a viable option to automate the SR processes, but requires careful considerations from developers when integrating ChatGPT into their SR tools.

{{</citation>}}


### (18/81) Navigating the Complexity of Generative AI Adoption in Software Engineering (Daniel Russo, 2023)

{{<citation>}}

Daniel Russo. (2023)  
**Navigating the Complexity of Generative AI Adoption in Software Engineering**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2307.06081v1)  

---


**ABSTRACT**  
In this paper, the adoption patterns of Generative Artificial Intelligence (AI) tools within software engineering are investigated. Influencing factors at the individual, technological, and societal levels are analyzed using a mixed-methods approach for an extensive comprehension of AI adoption. An initial structured interview was conducted with 100 software engineers, employing the Technology Acceptance Model (TAM), the Diffusion of Innovations theory (DOI), and the Social Cognitive Theory (SCT) as guiding theories.   A theoretical model named the Human-AI Collaboration and Adaptation Framework (HACAF) was deduced using the Gioia Methodology, characterizing AI adoption in software engineering. This model's validity was subsequently tested through Partial Least Squares - Structural Equation Modeling (PLS-SEM), using data collected from 183 software professionals.   The results indicate that the adoption of AI tools in these early integration stages is primarily driven by their compatibility with existing development workflows. This finding counters the traditional theories of technology acceptance. Contrary to expectations, the influence of perceived usefulness, social aspects, and personal innovativeness on adoption appeared to be less significant. This paper yields significant insights for the design of future AI tools and supplies a structure for devising effective strategies for organizational implementation.

{{</citation>}}


### (19/81) Securely extending and running low-code applications with C# (Lennart Brüggemann, 2023)

{{<citation>}}

Lennart Brüggemann. (2023)  
**Securely extending and running low-code applications with C#**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2307.06340v1)  

---


**ABSTRACT**  
Low-code development platforms provide an accessible infrastructure for the creation of software by domain experts, also called "citizen developers", without the need for formal programming education. Development is facilitated through graphical user interfaces, although traditional programming can still be used to extend low-code applications, for example when external services or complex business logic needs to be implemented that cannot be realized with the features available on a platform. Since citizen developers are usually not specifically trained in software development, they require additional support when writing code, particularly with regard to security and advanced techniques like debugging or versioning. In this thesis, several options to assist developers of low-code applications are investigated and implemented. A framework to quickly build code editor extensions is developed, and an approach to leverage the Roslyn compiler platform to implement custom static code analysis rules for low-code development platforms using the .NET platform is demonstrated. Furthermore, a sample application showing how Roslyn can be used to build a simple, integrated debugging tool, as well as an abstraction of the version control system Git for easier usage by citizen developers, is implemented. Security is a critical aspect when low-code applications are deployed. To provide an overview over possible options to ensure the secure and isolated execution of low-code applications, a threat model is developed and used as the basis for a comparison between OS-level virtualization, sandboxing, and runtime code security implementations.

{{</citation>}}


## cs.CR (3)



### (20/81) Benchmarking the Security Protocol and Data Model (SPDM) for component authentication (Renan C. A. Alves et al., 2023)

{{<citation>}}

Renan C. A. Alves, Bruno C. Albertini, Marcos A. Simplicio Jr. (2023)  
**Benchmarking the Security Protocol and Data Model (SPDM) for component authentication**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2307.06456v1)  

---


**ABSTRACT**  
Efforts to secure computing systems via software traditionally focus on the operating system and application levels. In contrast, the Security Protocol and Data Model (SPDM) tackles firmware level security challenges, which are much harder (if at all possible) to detect with regular protection software. SPDM includes key features like enabling peripheral authentication, authenticated hardware measurements retrieval, and secure session establishment. Since SPDM is a relatively recent proposal, there is a lack of studies evaluating its performance impact on real-world applications. In this article, we address this gap by: (1) implementing the protocol on a simple virtual device, and then investigating the overhead introduced by each SDPM message; and (2) creating an SPDM-capable virtual hard drive based on VirtIO, and comparing the resulting read/write performance with a regular, unsecured implementation. Our results suggest that SPDM bootstrap time takes the order of tens of milliseconds, while the toll of introducing SPDM on hard drive communication highly depends on specific workload patterns. For example, for mixed random read/write operations, the slowdown is negligible in comparison to the baseline unsecured setup. Conversely, for sequential read or write operations, the data encryption process becomes the bottleneck, reducing the performance indicators by several orders of magnitude.

{{</citation>}}


### (21/81) Security in Online Freelance Software Development: A case for Distributed Security Responsibility (Irum Rauf et al., 2023)

{{<citation>}}

Irum Rauf, Tamara Lopez, Thein Tun, Marian Petre, Bashar Nuseibeh. (2023)  
**Security in Online Freelance Software Development: A case for Distributed Security Responsibility**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-CY, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2307.06066v1)  

---


**ABSTRACT**  
Secure software is a cornerstone to safe and resilient digital ecosystems. It offers strong foundation to protect users' sensitive data and guard against cyber-threats. The rapidly increasing landscape of digital economy has encouraged developers from different socio-technical and socio-economic backgrounds to join online freelance marketplaces. While, secure software practices facilitate software developers in developing secure software, there is paucity of research on how freelance developers adhere to security practices and how they can be facilitated to improve their security behavior in under-resourced environments. Moreover, freelance developers are often held responsible for producing insecure code. In this position paper, we review existing literature and argue for the case of distributed security responsibilities in online freelance environment. We propose a research agenda aimed at offering an organized and systematic effort by researchers to address security needs and challenges of online freelance marketplaces. These include: characterising software security and defining separation of responsibilities, building trust in online freelance development communities, leveraging the potential of online freelancing platforms in the promotion of secure software development and building adaptive security interventions for online freelance software development. The research has the potential to bring forth existing security solutions to wider developer community and deliver substantial benefits to the broader security ecosystem.

{{</citation>}}


### (22/81) Introducing Packet-Level Analysis in Programmable Data Planes to Advance Network Intrusion Detection (Roberto Doriguzzi-Corin et al., 2023)

{{<citation>}}

Roberto Doriguzzi-Corin, Luis Augusto Dias Knob, Luca Mendozzi, Domenico Siracusa, Marco Savi. (2023)  
**Introducing Packet-Level Analysis in Programmable Data Planes to Advance Network Intrusion Detection**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-NI, cs.CR  
Keywords: Intrusion Detection  
[Paper Link](http://arxiv.org/abs/2307.05936v1)  

---


**ABSTRACT**  
Programmable data planes offer precise control over the low-level processing steps applied to network packets, serving as a valuable tool for analysing malicious flows in the field of intrusion detection. Albeit with limitations on physical resources and capabilities, they allow for the efficient extraction of detailed traffic information, which can then be utilised by Machine Learning (ML) algorithms responsible for identifying security threats. In addressing resource constraints, existing solutions in the literature rely on compressing network data through the collection of statistical traffic features in the data plane. While this compression saves memory resources in switches and minimises the burden on the control channel between the data and the control plane, it also results in a loss of information available to the Network Intrusion Detection System (NIDS), limiting access to packet payload, categorical features, and the semantic understanding of network communications, such as the behaviour of packets within traffic flows. This paper proposes P4DDLe, a framework that exploits the flexibility of P4-based programmable data planes for packet-level feature extraction and pre-processing. P4DDLe leverages the programmable data plane to extract raw packet features from the network traffic, categorical features included, and to organise them in a way that the semantics of traffic flows is preserved. To minimise memory and control channel overheads, P4DDLe selectively processes and filters packet-level data, so that all and only the relevant features required by the NIDS are collected. The experimental evaluation with recent Distributed Denial of Service (DDoS) attack data demonstrates that the proposed approach is very efficient in collecting compact and high-quality representations of network flows, ensuring precise detection of DDoS attacks.

{{</citation>}}


## math.OC (1)



### (23/81) Stochastic Delay Differential Games: Financial Modeling and Machine Learning Algorithms (Robert Balkin et al., 2023)

{{<citation>}}

Robert Balkin, Hector D. Ceniceros, Ruimeng Hu. (2023)  
**Stochastic Delay Differential Games: Financial Modeling and Machine Learning Algorithms**  

---
Primary Category: math.OC  
Categories: cs-LG, math-OC, math.OC, q-fin-CP  
Keywords: Financial  
[Paper Link](http://arxiv.org/abs/2307.06450v1)  

---


**ABSTRACT**  
In this paper, we propose a numerical methodology for finding the closed-loop Nash equilibrium of stochastic delay differential games through deep learning. These games are prevalent in finance and economics where multi-agent interaction and delayed effects are often desired features in a model, but are introduced at the expense of increased dimensionality of the problem. This increased dimensionality is especially significant as that arising from the number of players is coupled with the potential infinite dimensionality caused by the delay. Our approach involves parameterizing the controls of each player using distinct recurrent neural networks. These recurrent neural network-based controls are then trained using a modified version of Brown's fictitious play, incorporating deep learning techniques. To evaluate the effectiveness of our methodology, we test it on finance-related problems with known solutions. Furthermore, we also develop new problems and derive their analytical Nash equilibrium solutions, which serve as additional benchmarks for assessing the performance of our proposed deep learning approach.

{{</citation>}}


## eess.IV (3)



### (24/81) SAM-Path: A Segment Anything Model for Semantic Segmentation in Digital Pathology (Jingwei Zhang et al., 2023)

{{<citation>}}

Jingwei Zhang, Ke Ma, Saarthak Kapse, Joel Saltz, Maria Vakalopoulou, Prateek Prasanna, Dimitris Samaras. (2023)  
**SAM-Path: A Segment Anything Model for Semantic Segmentation in Digital Pathology**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2307.09570v1)  

---


**ABSTRACT**  
Semantic segmentations of pathological entities have crucial clinical value in computational pathology workflows. Foundation models, such as the Segment Anything Model (SAM), have been recently proposed for universal use in segmentation tasks. SAM shows remarkable promise in instance segmentation on natural images. However, the applicability of SAM to computational pathology tasks is limited due to the following factors: (1) lack of comprehensive pathology datasets used in SAM training and (2) the design of SAM is not inherently optimized for semantic segmentation tasks. In this work, we adapt SAM for semantic segmentation by introducing trainable class prompts, followed by further enhancements through the incorporation of a pathology encoder, specifically a pathology foundation model. Our framework, SAM-Path enhances SAM's ability to conduct semantic segmentation in digital pathology without human input prompts. Through experiments on two public pathology datasets, the BCSS and the CRAG datasets, we demonstrate that the fine-tuning with trainable class prompts outperforms vanilla SAM with manual prompts and post-processing by 27.52% in Dice score and 71.63% in IOU. On these two datasets, the proposed additional pathology foundation model further achieves a relative improvement of 5.07% to 5.12% in Dice score and 4.50% to 8.48% in IOU.

{{</citation>}}


### (25/81) Sequential Experimental Design for X-Ray CT Using Deep Reinforcement Learning (Tianyuan Wang et al., 2023)

{{<citation>}}

Tianyuan Wang, Felix Lucka, Tristan van Leeuwen. (2023)  
**Sequential Experimental Design for X-Ray CT Using Deep Reinforcement Learning**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.06343v1)  

---


**ABSTRACT**  
In X-ray Computed Tomography (CT), projections from many angles are acquired and used for 3D reconstruction. To make CT suitable for in-line quality control, reducing the number of angles while maintaining reconstruction quality is necessary. Sparse-angle tomography is a popular approach for obtaining 3D reconstructions from limited data. To optimize its performance, one can adapt scan angles sequentially to select the most informative angles for each scanned object. Mathematically, this corresponds to solving and optimal experimental design (OED) problem. OED problems are high-dimensional, non-convex, bi-level optimization problems that cannot be solved online, i.e., during the scan. To address these challenges, we pose the OED problem as a partially observable Markov decision process in a Bayesian framework, and solve it through deep reinforcement learning. The approach learns efficient non-greedy policies to solve a given class of OED problems through extensive offline training rather than solving a given OED problem directly via numerical optimization. As such, the trained policy can successfully find the most informative scan angles online. We use a policy training method based on the Actor-Critic approach and evaluate its performance on 2D tomography with synthetic data.

{{</citation>}}


### (26/81) Unified Medical Image-Text-Label Contrastive Learning With Continuous Prompt (Yuhao Wang, 2023)

{{<citation>}}

Yuhao Wang. (2023)  
**Unified Medical Image-Text-Label Contrastive Learning With Continuous Prompt**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2307.05920v1)  

---


**ABSTRACT**  
Contrastive language-image Pre-training (CLIP) [13] can leverage large datasets of unlabeled Image-Text pairs, which have demonstrated impressive performance in various downstream tasks. Given that annotating medical data is time-consuming and laborious, Image-Text Pre-training has promising applications in exploiting large-scale medical image and radiology report datasets. However, medical Image-Text Pre-training faces several challenges, as follows: (1) Due to privacy concerns, the amount of available medical data is relatively small compared to natural data, leading to weaker generalization ability of the model. (2) Medical images are highly similar with only fine-grained differences in subtleties, resulting in a large number of false-negative sample pairs in comparison learning. (3) The hand-crafted Prompt usually differs from the natural medical image report, Subtle changes in wording can lead to significant differences in performance. In this paper, we propose a unified Image-Text-Label contrastive learning framework based on continuous prompts, with three main contributions. First, We unified the data of images, text, and labels, which greatly expanded the training data that the model could utilize. Second, we address the issue of data diversity and the impact of hand-crafted prompts on model performance by introducing continuous implicit prompts. Lastly, we propose a ImageText-Label contrastive Training to mitigate the problem of too many false-negative samples. We demonstrate through sufficient experiments that the Unified Medical Contrastive Learning (UMCL) framework exhibits excellent performance on several downstream tasks.

{{</citation>}}


## cs.CV (23)



### (27/81) Efficient Convolution and Transformer-Based Network for Video Frame Interpolation (Issa Khalifeh et al., 2023)

{{<citation>}}

Issa Khalifeh, Luka Murn, Marta Mrak, Ebroul Izquierdo. (2023)  
**Efficient Convolution and Transformer-Based Network for Video Frame Interpolation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.06443v1)  

---


**ABSTRACT**  
Video frame interpolation is an increasingly important research task with several key industrial applications in the video coding, broadcast and production sectors. Recently, transformers have been introduced to the field resulting in substantial performance gains. However, this comes at a cost of greatly increased memory usage, training and inference time. In this paper, a novel method integrating a transformer encoder and convolutional features is proposed. This network reduces the memory burden by close to 50% and runs up to four times faster during inference time compared to existing transformer-based interpolation methods. A dual-encoder architecture is introduced which combines the strength of convolutions in modelling local correlations with those of the transformer for long-range dependencies. Quantitative evaluations are conducted on various benchmarks with complex motion to showcase the robustness of the proposed method, achieving competitive performance compared to state-of-the-art interpolation networks.

{{</citation>}}


### (28/81) RaBiT: An Efficient Transformer using Bidirectional Feature Pyramid Network with Reverse Attention for Colon Polyp Segmentation (Nguyen Hoang Thuan et al., 2023)

{{<citation>}}

Nguyen Hoang Thuan, Nguyen Thi Oanh, Nguyen Thi Thuy, Stuart Perry, Dinh Viet Sang. (2023)  
**RaBiT: An Efficient Transformer using Bidirectional Feature Pyramid Network with Reverse Attention for Colon Polyp Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2307.06420v1)  

---


**ABSTRACT**  
Automatic and accurate segmentation of colon polyps is essential for early diagnosis of colorectal cancer. Advanced deep learning models have shown promising results in polyp segmentation. However, they still have limitations in representing multi-scale features and generalization capability. To address these issues, this paper introduces RaBiT, an encoder-decoder model that incorporates a lightweight Transformer-based architecture in the encoder to model multiple-level global semantic relationships. The decoder consists of several bidirectional feature pyramid layers with reverse attention modules to better fuse feature maps at various levels and incrementally refine polyp boundaries. We also propose ideas to lighten the reverse attention module and make it more suitable for multi-class segmentation. Extensive experiments on several benchmark datasets show that our method outperforms existing methods across all datasets while maintaining low computational complexity. Moreover, our method demonstrates high generalization capability in cross-dataset experiments, even when the training and test sets have different characteristics.

{{</citation>}}


### (29/81) Data Augmentation in Training CNNs: Injecting Noise to Images (M. Eren Akbiyik, 2023)

{{<citation>}}

M. Eren Akbiyik. (2023)  
**Data Augmentation in Training CNNs: Injecting Noise to Images**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2307.06855v1)  

---


**ABSTRACT**  
Noise injection is a fundamental tool for data augmentation, and yet there is no widely accepted procedure to incorporate it with learning frameworks. This study analyzes the effects of adding or applying different noise models of varying magnitudes to Convolutional Neural Network (CNN) architectures. Noise models that are distributed with different density functions are given common magnitude levels via Structural Similarity (SSIM) metric in order to create an appropriate ground for comparison. The basic results are conforming with the most of the common notions in machine learning, and also introduce some novel heuristics and recommendations on noise injection. The new approaches will provide better understanding on optimal learning procedures for image classification.

{{</citation>}}


### (30/81) Correlation-Aware Mutual Learning for Semi-supervised Medical Image Segmentation (Shengbo Gao et al., 2023)

{{<citation>}}

Shengbo Gao, Ziji Zhang, Jiechao Ma, Zihao Li, Shu Zhang. (2023)  
**Correlation-Aware Mutual Learning for Semi-supervised Medical Image Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2307.06312v1)  

---


**ABSTRACT**  
Semi-supervised learning has become increasingly popular in medical image segmentation due to its ability to leverage large amounts of unlabeled data to extract additional information. However, most existing semi-supervised segmentation methods only focus on extracting information from unlabeled data, disregarding the potential of labeled data to further improve the performance of the model. In this paper, we propose a novel Correlation Aware Mutual Learning (CAML) framework that leverages labeled data to guide the extraction of information from unlabeled data. Our approach is based on a mutual learning strategy that incorporates two modules: the Cross-sample Mutual Attention Module (CMA) and the Omni-Correlation Consistency Module (OCC). The CMA module establishes dense cross-sample correlations among a group of samples, enabling the transfer of label prior knowledge to unlabeled data. The OCC module constructs omni-correlations between the unlabeled and labeled datasets and regularizes dual models by constraining the omni-correlation matrix of each sub-model to be consistent. Experiments on the Atrial Segmentation Challenge dataset demonstrate that our proposed approach outperforms state-of-the-art methods, highlighting the effectiveness of our framework in medical image segmentation tasks. The codes, pre-trained weights, and data are publicly available.

{{</citation>}}


### (31/81) Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution (Mostafa Dehghani et al., 2023)

{{<citation>}}

Mostafa Dehghani, Basil Mustafa, Josip Djolonga, Jonathan Heek, Matthias Minderer, Mathilde Caron, Andreas Steiner, Joan Puigcerver, Robert Geirhos, Ibrahim Alabdulmohsin, Avital Oliver, Piotr Padlewski, Alexey Gritsenko, Mario Lučić, Neil Houlsby. (2023)  
**Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.06304v1)  

---


**ABSTRACT**  
The ubiquitous and demonstrably suboptimal choice of resizing images to a fixed resolution before processing them with computer vision models has not yet been successfully challenged. However, models such as the Vision Transformer (ViT) offer flexible sequence-based modeling, and hence varying input sequence lengths. We take advantage of this with NaViT (Native Resolution ViT) which uses sequence packing during training to process inputs of arbitrary resolutions and aspect ratios. Alongside flexible model usage, we demonstrate improved training efficiency for large-scale supervised and contrastive image-text pretraining. NaViT can be efficiently transferred to standard tasks such as image and video classification, object detection, and semantic segmentation and leads to improved results on robustness and fairness benchmarks. At inference time, the input resolution flexibility can be used to smoothly navigate the test-time cost-performance trade-off. We believe that NaViT marks a departure from the standard, CNN-designed, input and modelling pipeline used by most computer vision models, and represents a promising direction for ViTs.

{{</citation>}}


### (32/81) MMBench: Is Your Multi-modal Model an All-around Player? (Yuan Liu et al., 2023)

{{<citation>}}

Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, Dahua Lin. (2023)  
**MMBench: Is Your Multi-modal Model an All-around Player?**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: ChatGPT, GPT, QA  
[Paper Link](http://arxiv.org/abs/2307.06281v1)  

---


**ABSTRACT**  
Large vision-language models have recently achieved remarkable progress, exhibiting great perception and reasoning abilities concerning visual information. However, how to effectively evaluate these large vision-language models remains a major obstacle, hindering future model development. Traditional benchmarks like VQAv2 or COCO Caption provide quantitative performance measurements but suffer from a lack of fine-grained ability assessment and non-robust evaluation metrics. Recent subjective benchmarks, such as OwlEval, offer comprehensive evaluations of a model's abilities by incorporating human labor, but they are not scalable and display significant bias. In response to these challenges, we propose MMBench, a novel multi-modality benchmark. MMBench methodically develops a comprehensive evaluation pipeline, primarily comprised of two elements. The first element is a meticulously curated dataset that surpasses existing similar benchmarks in terms of the number and variety of evaluation questions and abilities. The second element introduces a novel CircularEval strategy and incorporates the use of ChatGPT. This implementation is designed to convert free-form predictions into pre-defined choices, thereby facilitating a more robust evaluation of the model's predictions. MMBench is a systematically-designed objective benchmark for robustly evaluating the various abilities of vision-language models. We hope MMBench will assist the research community in better evaluating their models and encourage future advancements in this domain. Project page: https://opencompass.org.cn/mmbench.

{{</citation>}}


### (33/81) UGCANet: A Unified Global Context-Aware Transformer-based Network with Feature Alignment for Endoscopic Image Analysis (Pham Vu Hung et al., 2023)

{{<citation>}}

Pham Vu Hung, Nguyen Duy Manh, Nguyen Thi Oanh, Nguyen Thi Thuy, Dinh Viet Sang. (2023)  
**UGCANet: A Unified Global Context-Aware Transformer-based Network with Feature Alignment for Endoscopic Image Analysis**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.06260v1)  

---


**ABSTRACT**  
Gastrointestinal endoscopy is a medical procedure that utilizes a flexible tube equipped with a camera and other instruments to examine the digestive tract. This minimally invasive technique allows for diagnosing and managing various gastrointestinal conditions, including inflammatory bowel disease, gastrointestinal bleeding, and colon cancer. The early detection and identification of lesions in the upper gastrointestinal tract and the identification of malignant polyps that may pose a risk of cancer development are critical components of gastrointestinal endoscopy's diagnostic and therapeutic applications. Therefore, enhancing the detection rates of gastrointestinal disorders can significantly improve a patient's prognosis by increasing the likelihood of timely medical intervention, which may prolong the patient's lifespan and improve overall health outcomes. This paper presents a novel Transformer-based deep neural network designed to perform multiple tasks simultaneously, thereby enabling accurate identification of both upper gastrointestinal tract lesions and colon polyps. Our approach proposes a unique global context-aware module and leverages the powerful MiT backbone, along with a feature alignment block, to enhance the network's representation capability. This novel design leads to a significant improvement in performance across various endoscopic diagnosis tasks. Extensive experiments demonstrate the superior performance of our method compared to other state-of-the-art approaches.

{{</citation>}}


### (34/81) CellGAN: Conditional Cervical Cell Synthesis for Augmenting Cytopathological Image Classification (Zhenrong Shen et al., 2023)

{{<citation>}}

Zhenrong Shen, Maosong Cao, Sheng Wang, Lichi Zhang, Qian Wang. (2023)  
**CellGAN: Conditional Cervical Cell Synthesis for Augmenting Cytopathological Image Classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2307.06182v1)  

---


**ABSTRACT**  
Automatic examination of thin-prep cytologic test (TCT) slides can assist pathologists in finding cervical abnormality for accurate and efficient cancer screening. Current solutions mostly need to localize suspicious cells and classify abnormality based on local patches, concerning the fact that whole slide images of TCT are extremely large. It thus requires many annotations of normal and abnormal cervical cells, to supervise the training of the patch-level classifier for promising performance. In this paper, we propose CellGAN to synthesize cytopathological images of various cervical cell types for augmenting patch-level cell classification. Built upon a lightweight backbone, CellGAN is equipped with a non-linear class mapping network to effectively incorporate cell type information into image generation. We also propose the Skip-layer Global Context module to model the complex spatial relationship of the cells, and attain high fidelity of the synthesized images through adversarial learning. Our experiments demonstrate that CellGAN can produce visually plausible TCT cytopathological images for different cell types. We also validate the effectiveness of using CellGAN to greatly augment patch-level cell classification performance.

{{</citation>}}


### (35/81) Large Class Separation is not what you need for Relational Reasoning-based OOD Detection (Lorenzo Li Lu et al., 2023)

{{<citation>}}

Lorenzo Li Lu, Giulia D'Ascenzi, Francesco Cappio Borlino, Tatiana Tommasi. (2023)  
**Large Class Separation is not what you need for Relational Reasoning-based OOD Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2307.06179v1)  

---


**ABSTRACT**  
Standard recognition approaches are unable to deal with novel categories at test time. Their overconfidence on the known classes makes the predictions unreliable for safety-critical applications such as healthcare or autonomous driving. Out-Of-Distribution (OOD) detection methods provide a solution by identifying semantic novelty. Most of these methods leverage a learning stage on the known data, which means training (or fine-tuning) a model to capture the concept of normality. This process is clearly sensitive to the amount of available samples and might be computationally expensive for on-board systems. A viable alternative is that of evaluating similarities in the embedding space produced by large pre-trained models without any further learning effort. We focus exactly on such a fine-tuning-free OOD detection setting. This works presents an in-depth analysis of the recently introduced relational reasoning pre-training and investigates the properties of the learned embedding, highlighting the existence of a correlation between the inter-class feature distance and the OOD detection accuracy. As the class separation depends on the chosen pre-training objective, we propose an alternative loss function to control the inter-class margin, and we show its advantage with thorough experiments.

{{</citation>}}


### (36/81) Smart Infrastructure: A Research Junction (Manuel Hetzel et al., 2023)

{{<citation>}}

Manuel Hetzel, Hannes Reichert, Konrad Doll, Bernhard Sick. (2023)  
**Smart Infrastructure: A Research Junction**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.06177v1)  

---


**ABSTRACT**  
Complex inner-city junctions are among the most critical traffic areas for injury and fatal accidents. The development of highly automated driving (HAD) systems struggles with the complex and hectic everyday life within those areas. Sensor-equipped smart infrastructures, which can communicate and cooperate with vehicles, are essential to enable a holistic scene understanding to resolve occlusions drivers and vehicle perception systems for themselves can not cover. We introduce an intelligent research infrastructure equipped with visual sensor technology, located at a public inner-city junction in Aschaffenburg, Germany. A multiple-view camera system monitors the traffic situation to perceive road users' behavior. Both motorized and non-motorized traffic is considered. The system is used for research in data generation, evaluating new HAD sensors systems, algorithms, and Artificial Intelligence (AI) training strategies using real-, synthetic- and augmented data. In addition, the junction features a highly accurate digital twin. Real-world data can be taken into the digital twin for simulation purposes and synthetic data generation.

{{</citation>}}


### (37/81) Can Vision-Language Models be a Good Guesser? Exploring VLMs for Times and Location Reasoning (Gengyuan Zhang et al., 2023)

{{<citation>}}

Gengyuan Zhang, Yurui Zhang, Kerui Zhang, Volker Tresp. (2023)  
**Can Vision-Language Models be a Good Guesser? Exploring VLMs for Times and Location Reasoning**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2307.06166v1)  

---


**ABSTRACT**  
Vision-Language Models (VLMs) are expected to be capable of reasoning with commonsense knowledge as human beings. One example is that humans can reason where and when an image is taken based on their knowledge. This makes us wonder if, based on visual cues, Vision-Language Models that are pre-trained with large-scale image-text resources can achieve and even outperform human's capability in reasoning times and location. To address this question, we propose a two-stage \recognition\space and \reasoning\space probing task, applied to discriminative and generative VLMs to uncover whether VLMs can recognize times and location-relevant features and further reason about it. To facilitate the investigation, we introduce WikiTiLo, a well-curated image dataset compromising images with rich socio-cultural cues. In the extensive experimental studies, we find that although VLMs can effectively retain relevant features in visual encoders, they still fail to make perfect reasoning. We will release our dataset and codes to facilitate future studies.

{{</citation>}}


### (38/81) Learning Kernel-Modulated Neural Representation for Efficient Light Field Compression (Jinglei Shi et al., 2023)

{{<citation>}}

Jinglei Shi, Yihong Xu, Christine Guillemot. (2023)  
**Learning Kernel-Modulated Neural Representation for Efficient Light Field Compression**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.06143v1)  

---


**ABSTRACT**  
Light field is a type of image data that captures the 3D scene information by recording light rays emitted from a scene at various orientations. It offers a more immersive perception than classic 2D images but at the cost of huge data volume. In this paper, we draw inspiration from the visual characteristics of Sub-Aperture Images (SAIs) of light field and design a compact neural network representation for the light field compression task. The network backbone takes randomly initialized noise as input and is supervised on the SAIs of the target light field. It is composed of two types of complementary kernels: descriptive kernels (descriptors) that store scene description information learned during training, and modulatory kernels (modulators) that control the rendering of different SAIs from the queried perspectives. To further enhance compactness of the network meanwhile retain high quality of the decoded light field, we accordingly introduce modulator allocation and kernel tensor decomposition mechanisms, followed by non-uniform quantization and lossless entropy coding techniques, to finally form an efficient compression pipeline. Extensive experiments demonstrate that our method outperforms other state-of-the-art (SOTA) methods by a significant margin in the light field compression task. Moreover, after aligning descriptors, the modulators learned from one light field can be transferred to new light fields for rendering dense views, indicating a potential solution for view synthesis task.

{{</citation>}}


### (39/81) TreeFormer: a Semi-Supervised Transformer-based Framework for Tree Counting from a Single High Resolution Image (Hamed Amini Amirkolaee et al., 2023)

{{<citation>}}

Hamed Amini Amirkolaee, Miaojing Shi, Mark Mulligan. (2023)  
**TreeFormer: a Semi-Supervised Transformer-based Framework for Tree Counting from a Single High Resolution Image**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Semi-Supervised, Transformer  
[Paper Link](http://arxiv.org/abs/2307.06118v1)  

---


**ABSTRACT**  
Automatic tree density estimation and counting using single aerial and satellite images is a challenging task in photogrammetry and remote sensing, yet has an important role in forest management. In this paper, we propose the first semisupervised transformer-based framework for tree counting which reduces the expensive tree annotations for remote sensing images. Our method, termed as TreeFormer, first develops a pyramid tree representation module based on transformer blocks to extract multi-scale features during the encoding stage. Contextual attention-based feature fusion and tree density regressor modules are further designed to utilize the robust features from the encoder to estimate tree density maps in the decoder. Moreover, we propose a pyramid learning strategy that includes local tree density consistency and local tree count ranking losses to utilize unlabeled images into the training process. Finally, the tree counter token is introduced to regulate the network by computing the global tree counts for both labeled and unlabeled images. Our model was evaluated on two benchmark tree counting datasets, Jiangsu, and Yosemite, as well as a new dataset, KCL-London, created by ourselves. Our TreeFormer outperforms the state of the art semi-supervised methods under the same setting and exceeds the fully-supervised methods using the same number of labeled images. The codes and datasets are available at https://github.com/HAAClassic/TreeFormer.

{{</citation>}}


### (40/81) ConvNeXt-ChARM: ConvNeXt-based Transform for Efficient Neural Image Compression (Ahmed Ghorbel et al., 2023)

{{<citation>}}

Ahmed Ghorbel, Wassim Hamidouche, Luce Morin. (2023)  
**ConvNeXt-ChARM: ConvNeXt-based Transform for Efficient Neural Image Compression**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.06342v1)  

---


**ABSTRACT**  
Over the last few years, neural image compression has gained wide attention from research and industry, yielding promising end-to-end deep neural codecs outperforming their conventional counterparts in rate-distortion performance. Despite significant advancement, current methods, including attention-based transform coding, still need to be improved in reducing the coding rate while preserving the reconstruction fidelity, especially in non-homogeneous textured image areas. Those models also require more parameters and a higher decoding time. To tackle the above challenges, we propose ConvNeXt-ChARM, an efficient ConvNeXt-based transform coding framework, paired with a compute-efficient channel-wise auto-regressive prior to capturing both global and local contexts from the hyper and quantized latent representations. The proposed architecture can be optimized end-to-end to fully exploit the context information and extract compact latent representation while reconstructing higher-quality images. Experimental results on four widely-used datasets showed that ConvNeXt-ChARM brings consistent and significant BD-rate (PSNR) reductions estimated on average to 5.24% and 1.22% over the versatile video coding (VVC) reference encoder (VTM-18.0) and the state-of-the-art learned image compression method SwinT-ChARM, respectively. Moreover, we provide model scaling studies to verify the computational efficiency of our approach and conduct several objective and subjective analyses to bring to the fore the performance gap between the next generation ConvNet, namely ConvNeXt, and Swin Transformer.

{{</citation>}}


### (41/81) AICT: An Adaptive Image Compression Transformer (Ahmed Ghorbel et al., 2023)

{{<citation>}}

Ahmed Ghorbel, Wassim Hamidouche, Luce Morin. (2023)  
**AICT: An Adaptive Image Compression Transformer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: AI, Transformer  
[Paper Link](http://arxiv.org/abs/2307.06091v1)  

---


**ABSTRACT**  
Motivated by the efficiency investigation of the Tranformer-based transform coding framework, namely SwinT-ChARM, we propose to enhance the latter, as first, with a more straightforward yet effective Tranformer-based channel-wise auto-regressive prior model, resulting in an absolute image compression transformer (ICT). Current methods that still rely on ConvNet-based entropy coding are limited in long-range modeling dependencies due to their local connectivity and an increasing number of architectural biases and priors. On the contrary, the proposed ICT can capture both global and local contexts from the latent representations and better parameterize the distribution of the quantized latents. Further, we leverage a learnable scaling module with a sandwich ConvNeXt-based pre/post-processor to accurately extract more compact latent representation while reconstructing higher-quality images. Extensive experimental results on benchmark datasets showed that the proposed adaptive image compression transformer (AICT) framework significantly improves the trade-off between coding efficiency and decoder complexity over the versatile video coding (VVC) reference encoder (VTM-18.0) and the neural codec SwinT-ChARM.

{{</citation>}}


### (42/81) Visualization for Multivariate Gaussian Anomaly Detection in Images (Joao P C Bertoldo et al., 2023)

{{<citation>}}

Joao P C Bertoldo, David Arrustico. (2023)  
**Visualization for Multivariate Gaussian Anomaly Detection in Images**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2307.06052v1)  

---


**ABSTRACT**  
This paper introduces a simplified variation of the PaDiM (Pixel-Wise Anomaly Detection through Instance Modeling) method for anomaly detection in images, fitting a single multivariate Gaussian (MVG) distribution to the feature vectors extracted from a backbone convolutional neural network (CNN) and using their Mahalanobis distance as the anomaly score. We introduce an intermediate step in this framework by applying a whitening transformation to the feature vectors, which enables the generation of heatmaps capable of visually explaining the features learned by the MVG. The proposed technique is evaluated on the MVTec-AD dataset, and the results show the importance of visual model validation, providing insights into issues in this framework that were otherwise invisible. The visualizations generated for this paper are publicly available at https://doi.org/10.5281/zenodo.7937978.

{{</citation>}}


### (43/81) What Happens During Finetuning of Vision Transformers: An Invariance Based Investigation (Gabriele Merlin et al., 2023)

{{<citation>}}

Gabriele Merlin, Vedant Nanda, Ruchit Rawal, Mariya Toneva. (2023)  
**What Happens During Finetuning of Vision Transformers: An Invariance Based Investigation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.06006v1)  

---


**ABSTRACT**  
The pretrain-finetune paradigm usually improves downstream performance over training a model from scratch on the same task, becoming commonplace across many areas of machine learning. While pretraining is empirically observed to be beneficial for a range of tasks, there is not a clear understanding yet of the reasons for this effect. In this work, we examine the relationship between pretrained vision transformers and the corresponding finetuned versions on several benchmark datasets and tasks. We present new metrics that specifically investigate the degree to which invariances learned by a pretrained model are retained or forgotten during finetuning. Using these metrics, we present a suite of empirical findings, including that pretraining induces transferable invariances in shallow layers and that invariances from deeper pretrained layers are compressed towards shallower layers during finetuning. Together, these findings contribute to understanding some of the reasons for the successes of pretrained models and the changes that a pretrained model undergoes when finetuned on a downstream task.

{{</citation>}}


### (44/81) YOGA: Deep Object Detection in the Wild with Lightweight Feature Learning and Multiscale Attention (Raja Sunkara et al., 2023)

{{<citation>}}

Raja Sunkara, Tie Luo. (2023)  
**YOGA: Deep Object Detection in the Wild with Lightweight Feature Learning and Multiscale Attention**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Attention, Object Detection  
[Paper Link](http://arxiv.org/abs/2307.05945v1)  

---


**ABSTRACT**  
We introduce YOGA, a deep learning based yet lightweight object detection model that can operate on low-end edge devices while still achieving competitive accuracy. The YOGA architecture consists of a two-phase feature learning pipeline with a cheap linear transformation, which learns feature maps using only half of the convolution filters required by conventional convolutional neural networks. In addition, it performs multi-scale feature fusion in its neck using an attention mechanism instead of the naive concatenation used by conventional detectors. YOGA is a flexible model that can be easily scaled up or down by several orders of magnitude to fit a broad range of hardware constraints. We evaluate YOGA on COCO-val and COCO-testdev datasets with other over 10 state-of-the-art object detectors. The results show that YOGA strikes the best trade-off between model size and accuracy (up to 22% increase of AP and 23-34% reduction of parameters and FLOPs), making it an ideal choice for deployment in the wild on low-end edge devices. This is further affirmed by our hardware implementation and evaluation on NVIDIA Jetson Nano.

{{</citation>}}


### (45/81) Sem-CS: Semantic CLIPStyler for Text-Based Image Style Transfer (Chanda Grover Kamra et al., 2023)

{{<citation>}}

Chanda Grover Kamra, Indra Deep Mastan, Debayan Gupta. (2023)  
**Sem-CS: Semantic CLIPStyler for Text-Based Image Style Transfer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Style Transfer  
[Paper Link](http://arxiv.org/abs/2307.05934v1)  

---


**ABSTRACT**  
CLIPStyler demonstrated image style transfer with realistic textures using only a style text description (instead of requiring a reference style image). However, the ground semantics of objects in the style transfer output is lost due to style spill-over on salient and background objects (content mismatch) or over-stylization. To solve this, we propose Semantic CLIPStyler (Sem-CS), that performs semantic style transfer. Sem-CS first segments the content image into salient and non-salient objects and then transfers artistic style based on a given style text description. The semantic style transfer is achieved using global foreground loss (for salient objects) and global background loss (for non-salient objects). Our empirical results, including DISTS, NIMA and user study scores, show that our proposed framework yields superior qualitative and quantitative performance. Our code is available at github.com/chandagrover/sem-cs.

{{</citation>}}


### (46/81) SwiFT: Swin 4D fMRI Transformer (Peter Yongho Kim et al., 2023)

{{<citation>}}

Peter Yongho Kim, Junbeom Kwon, Sunghwan Joo, Sangyoon Bae, Donggyu Lee, Yoonho Jung, Shinjae Yoo, Jiook Cha, Taesup Moon. (2023)  
**SwiFT: Swin 4D fMRI Transformer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.05916v1)  

---


**ABSTRACT**  
The modeling of spatiotemporal brain dynamics from high-dimensional data, such as 4D functional MRI, is a formidable task in neuroscience. To address this challenge, we present SwiFT (Swin 4D fMRI Transformer), a Swin Transformer architecture that can learn brain dynamics directly from 4D functional brain MRI data in a memory and computation-efficient manner. SwiFT achieves this by implementing a 4D window multi-head self-attention mechanism and absolute positional embeddings. We evaluate SwiFT using multiple largest-scale human functional brain imaging datasets in tasks such as predicting sex, age, and cognitive intelligence. Our experimental outcomes reveal that SwiFT consistently outperforms recent state-of-the-art models. To the best of our knowledge, SwiFT is the first Swin Transformer architecture that can process dimensional spatiotemporal brain functional data in an end-to-end fashion. Furthermore, due to the end-to-end learning capability, we also show that contrastive loss-based self-supervised pre-training of SwiFT is also feasible for achieving improved performance on a downstream task. We believe that our work holds substantial potential in facilitating scalable learning of functional brain imaging in neuroscience research by reducing the hurdles associated with applying Transformer models to high-dimensional fMRI.

{{</citation>}}


### (47/81) Close-up View synthesis by Interpolating Optical Flow (Xinyi Bai et al., 2023)

{{<citation>}}

Xinyi Bai, Ze Wang, Lu Yang, Hong Cheng. (2023)  
**Close-up View synthesis by Interpolating Optical Flow**  

---
Primary Category: cs.CV  
Categories: I-4-5, I-4-0, I-4-1, cs-AI, cs-CV, cs-RO, cs.CV  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2307.05913v1)  

---


**ABSTRACT**  
The virtual viewpoint is perceived as a new technique in virtual navigation, as yet not supported due to the lack of depth information and obscure camera parameters. In this paper, a method for achieving close-up virtual view is proposed and it only uses optical flow to build parallax effects to realize pseudo 3D projection without using depth sensor. We develop a bidirectional optical flow method to obtain any virtual viewpoint by proportional interpolation of optical flow. Moreover, with the ingenious application of the optical-flow-value, we achieve clear and visual-fidelity magnified results through lens stretching in any corner, which overcomes the visual distortion and image blur through viewpoint magnification and transition in Google Street View system.

{{</citation>}}


### (48/81) Multi-Object Tracking as Attention Mechanism (Hiroshi Fukui et al., 2023)

{{<citation>}}

Hiroshi Fukui, Taiki Miyagawa, Yusuke Morishita. (2023)  
**Multi-Object Tracking as Attention Mechanism**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2307.05874v1)  

---


**ABSTRACT**  
We propose a conceptually simple and thus fast multi-object tracking (MOT) model that does not require any attached modules, such as the Kalman filter, Hungarian algorithm, transformer blocks, or graph networks. Conventional MOT models are built upon the multi-step modules listed above, and thus the computational cost is high. Our proposed end-to-end MOT model, \textit{TicrossNet}, is composed of a base detector and a cross-attention module only. As a result, the overhead of tracking does not increase significantly even when the number of instances ($N_t$) increases. We show that TicrossNet runs \textit{in real-time}; specifically, it achieves 32.6 FPS on MOT17 and 31.0 FPS on MOT20 (Tesla V100), which includes as many as $>$100 instances per frame. We also demonstrate that TicrossNet is robust to $N_t$; thus, it does not have to change the size of the base detector, depending on $N_t$, as is often done by other models for real-time processing.

{{</citation>}}


### (49/81) GLA-GCN: Global-local Adaptive Graph Convolutional Network for 3D Human (Bruce X. B. Yu et al., 2023)

{{<citation>}}

Bruce X. B. Yu, Zhi Zhang, Yongxu Liu, Sheng-hua Zhong, Yan Liu, Chang Wen Chen. (2023)  
**GLA-GCN: Global-local Adaptive Graph Convolutional Network for 3D Human**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Graph Convolutional Network  
[Paper Link](http://arxiv.org/abs/2307.05853v1)  

---


**ABSTRACT**  
3D human pose estimation has been researched for decades with promising fruits. 3D human pose lifting is one of the promising research directions toward the task where both estimated pose and ground truth pose data are used for training. Existing pose lifting works mainly focus on improving the performance of estimated pose, but they usually underperform when testing on the ground truth pose data. We observe that the performance of the estimated pose can be easily improved by preparing good quality 2D pose, such as fine-tuning the 2D pose or using advanced 2D pose detectors. As such, we concentrate on improving the 3D human pose lifting via ground truth data for the future improvement of more quality estimated pose data. Towards this goal, a simple yet effective model called Global-local Adaptive Graph Convolutional Network (GLA-GCN) is proposed in this work. Our GLA-GCN globally models the spatiotemporal structure via a graph representation and backtraces local joint features for 3D human pose estimation via individually connected layers. To validate our model design, we conduct extensive experiments on three benchmark datasets: Human3.6M, HumanEva-I, and MPI-INF-3DHP. Experimental results show that our GLA-GCN implemented with ground truth 2D poses significantly outperforms state-of-the-art methods (e.g., up to around 3%, 17%, and 13% error reductions on Human3.6M, HumanEva-I, and MPI-INF-3DHP, respectively).

{{</citation>}}


## cs.CL (10)



### (50/81) Distilling Large Language Models for Biomedical Knowledge Extraction: A Case Study on Adverse Drug Events (Yu Gu et al., 2023)

{{<citation>}}

Yu Gu, Sheng Zhang, Naoto Usuyama, Yonas Woldesenbet, Cliff Wong, Praneeth Sanapathi, Mu Wei, Naveen Valluri, Erika Strandberg, Tristan Naumann, Hoifung Poon. (2023)  
**Distilling Large Language Models for Biomedical Knowledge Extraction: A Case Study on Adverse Drug Events**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, GPT, GPT-3.5, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2307.06439v1)  

---


**ABSTRACT**  
Large language models (LLMs), such as GPT-4, have demonstrated remarkable capabilities across a wide range of tasks, including health applications. In this paper, we study how LLMs can be used to scale biomedical knowledge curation. We find that while LLMs already possess decent competency in structuring biomedical text, by distillation into a task-specific student model through self-supervised learning, substantial gains can be attained over out-of-box LLMs, with additional advantages such as cost, efficiency, and white-box model access.   We conduct a case study on adverse drug event (ADE) extraction, which is an important area for improving care. On standard ADE extraction evaluation, a GPT-3.5 distilled PubMedBERT model attained comparable accuracy as supervised state-of-the-art models without using any labeled data. Despite being over 1,000 times smaller, the distilled model outperformed its teacher GPT-3.5 by over 6 absolute points in F1 and GPT-4 by over 5 absolute points.   Ablation studies on distillation model choice (e.g., PubMedBERT vs BioGPT) and ADE extraction architecture shed light on best practice for biomedical knowledge extraction. Similar gains were attained by distillation for other standard biomedical knowledge extraction tasks such as gene-disease associations and protected health information, further illustrating the promise of this approach.

{{</citation>}}


### (51/81) A Comprehensive Overview of Large Language Models (Humza Naveed et al., 2023)

{{<citation>}}

Humza Naveed, Asad Ullah Khan, Shi Qiu, Muhammad Saqib, Saeed Anwar, Muhammad Usman, Nick Barnes, Ajmal Mian. (2023)  
**A Comprehensive Overview of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.06435v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have shown excellent generalization capabilities that have led to the development of numerous models. These models propose various new architectures, tweaking existing architectures with refined training strategies, increasing context length, using high-quality training data, and increasing training time to outperform baselines. Analyzing new developments is crucial for identifying changes that enhance training stability and improve generalization in LLMs. This survey paper comprehensively analyses the LLMs architectures and their categorization, training strategies, training datasets, and performance evaluations and discusses future research directions. Moreover, the paper also discusses the basic building blocks and concepts behind LLMs, followed by a complete overview of LLMs, including their important features and functions. Finally, the paper summarizes significant findings from LLM research and consolidates essential architectural and training strategies for developing advanced LLMs. Given the continuous advancements in LLMs, we intend to regularly update this paper by incorporating new sections and featuring the latest LLM models.

{{</citation>}}


### (52/81) Instruction Mining: High-Quality Instruction Data Selection for Large Language Models (Yihan Cao et al., 2023)

{{<citation>}}

Yihan Cao, Yanbin Kang, Lichao Sun. (2023)  
**Instruction Mining: High-Quality Instruction Data Selection for Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.06290v1)  

---


**ABSTRACT**  
Large language models typically undergo two training stages, pretraining and finetuning. Despite that large-scale pretraining endows the model with strong capabilities to generate natural language responses, these pretrained models can still fail to understand human instructions at times. To enhance language models' ability of interpreting and responding to instructions, instruction finetuning has emerged as a critical method in this area. Recent studies found that large language models can be finetuned to perform well even with a small amount of high-quality instruction-following data. However, the selection of high-quality datasets for finetuning language models still lacks clear guidelines to follow. In this paper, we propose InstructMining, a linear rule for evaluating instruction-following data quality. We formulate InstructMining using specific natural language indicators. To investigate the relationship between data quality and these indicators, we further conduct extensive finetuning experiments. The experiment results are then applied to estimating parameters in InstructMining. To further investigate its performance, we use InstructMining to select high-quality data from unseen datasets. Results demonstrate that InstructMining can help select relatively high-quality samples from various instruction-following datasets. Compared to models finetuned on unfiltered datasets, models finetuned on InstructMining selected datasets perform better on 42.5% cases.

{{</citation>}}


### (53/81) Ashaar: Automatic Analysis and Generation of Arabic Poetry Using Deep Learning Approaches (Zaid Alyafeai et al., 2023)

{{<citation>}}

Zaid Alyafeai, Maged S. Al-Shaibani, Moataz Ahmed. (2023)  
**Ashaar: Automatic Analysis and Generation of Arabic Poetry Using Deep Learning Approaches**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2307.06218v1)  

---


**ABSTRACT**  
Poetry holds immense significance within the cultural and traditional fabric of any nation. It serves as a vehicle for poets to articulate their emotions, preserve customs, and convey the essence of their culture. Arabic poetry is no exception, having played a cherished role in the heritage of the Arabic community throughout history and maintaining its relevance in the present era. Typically, comprehending Arabic poetry necessitates the expertise of a linguist who can analyze its content and assess its quality. This paper presents the introduction of a framework called \textit{Ashaar} https://github.com/ARBML/Ashaar, which encompasses a collection of datasets and pre-trained models designed specifically for the analysis and generation of Arabic poetry. The pipeline established within our proposed approach encompasses various aspects of poetry, such as meter, theme, and era classification. It also incorporates automatic poetry diacritization, enabling more intricate analyses like automated extraction of the \textit{Arudi} style. Additionally, we explore the feasibility of generating conditional poetry through the pre-training of a character-based GPT model. Furthermore, as part of this endeavor, we provide four datasets: one for poetry generation, another for diacritization, and two for Arudi-style prediction. These datasets aim to facilitate research and development in the field of Arabic poetry by enabling researchers and enthusiasts to delve into the nuances of this rich literary tradition.

{{</citation>}}


### (54/81) Sumformer: A Linear-Complexity Alternative to Self-Attention for Speech Recognition (Titouan Parcollet et al., 2023)

{{<citation>}}

Titouan Parcollet, Rogier van Dalen, Shucong Zhang, Sourav Bhattacharya. (2023)  
**Sumformer: A Linear-Complexity Alternative to Self-Attention for Speech Recognition**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: Attention, Self-Attention, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2307.07421v1)  

---


**ABSTRACT**  
Modern speech recognition systems rely on self-attention. Unfortunately, token mixing with self-attention takes quadratic time in the length of the speech utterance, slowing down inference as well as training and increasing memory consumption. Cheaper alternatives to self-attention for ASR have been developed, but fail to consistently reach the same level of accuracy. In practice, however, the self-attention weights of trained speech recognizers take the form of a global average over time. This paper, therefore, proposes a linear-time alternative to self-attention for speech recognition. It summarises a whole utterance with the mean over vectors for all time steps. This single summary is then combined with time-specific information. We call this method ``Summary Mixing''. Introducing Summary Mixing in state-of-the-art ASR models makes it feasible to preserve or exceed previous speech recognition performance while lowering the training and inference times by up to 27% and reducing the memory budget by a factor of two.

{{</citation>}}


### (55/81) Pluggable Neural Machine Translation Models via Memory-augmented Adapters (Yuzhuang Xu et al., 2023)

{{<citation>}}

Yuzhuang Xu, Shuo Wang, Peng Li, Xuebo Liu, Xiaolong Wang, Weidong Liu, Yang Liu. (2023)  
**Pluggable Neural Machine Translation Models via Memory-augmented Adapters**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Machine Translation  
[Paper Link](http://arxiv.org/abs/2307.06029v1)  

---


**ABSTRACT**  
Although neural machine translation (NMT) models perform well in the general domain, it remains rather challenging to control their generation behavior to satisfy the requirement of different users. Given the expensive training cost and the data scarcity challenge of learning a new model from scratch for each user requirement, we propose a memory-augmented adapter to steer pretrained NMT models in a pluggable manner. Specifically, we construct a multi-granular memory based on the user-provided text samples and propose a new adapter architecture to combine the model representations and the retrieved results. We also propose a training strategy using memory dropout to reduce spurious dependencies between the NMT model and the memory. We validate our approach on both style- and domain-specific experiments and the results indicate that our method can outperform several representative pluggable baselines.

{{</citation>}}


### (56/81) PolyLM: An Open Source Polyglot Large Language Model (Xiangpeng Wei et al., 2023)

{{<citation>}}

Xiangpeng Wei, Haoran Wei, Huan Lin, Tianhao Li, Pei Zhang, Xingzhang Ren, Mei Li, Yu Wan, Zhiwei Cao, Binbin Xie, Tianxiang Hu, Shangjie Li, Binyuan Hui, Bowen Yu, Dayiheng Liu, Baosong Yang, Fei Huang, Jun Xie. (2023)  
**PolyLM: An Open Source Polyglot Large Language Model**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BLOOM, LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2307.06018v1)  

---


**ABSTRACT**  
Large language models (LLMs) demonstrate remarkable ability to comprehend, reason, and generate following nature language instructions. However, the development of LLMs has been primarily focused on high-resource languages, such as English, thereby limiting their applicability and research in other languages. Consequently, we present PolyLM, a multilingual LLM trained on 640 billion (B) tokens, avaliable in two model sizes: 1.7B and 13B. To enhance its multilingual capabilities, we 1) integrate bilingual data into training data; and 2) adopt a curriculum learning strategy that increases the proportion of non-English data from 30% in the first stage to 60% in the final stage during pre-training. Further, we propose a multilingual self-instruct method which automatically generates 132.7K diverse multilingual instructions for model fine-tuning. To assess the model's performance, we collect several existing multilingual tasks, including multilingual understanding, question answering, generation, and translation. Extensive experiments show that PolyLM surpasses other open-source models such as LLaMA and BLOOM on multilingual tasks while maintaining comparable performance in English. Our models, alone with the instruction data and multilingual benchmark, are available at: \url{https://modelscope.cn/models/damo/nlp_polylm_13b_text_generation}.

{{</citation>}}


### (57/81) DDNAS: Discretized Differentiable Neural Architecture Search for Text Classification (Kuan-Chun Chen et al., 2023)

{{<citation>}}

Kuan-Chun Chen, Cheng-Te Li, Kuo-Jung Lee. (2023)  
**DDNAS: Discretized Differentiable Neural Architecture Search for Text Classification**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-IR, cs-LG, cs.CL  
Keywords: Text Classification  
[Paper Link](http://arxiv.org/abs/2307.06005v1)  

---


**ABSTRACT**  
Neural Architecture Search (NAS) has shown promising capability in learning text representation. However, existing text-based NAS neither performs a learnable fusion of neural operations to optimize the architecture, nor encodes the latent hierarchical categorization behind text input. This paper presents a novel NAS method, Discretized Differentiable Neural Architecture Search (DDNAS), for text representation learning and classification. With the continuous relaxation of architecture representation, DDNAS can use gradient descent to optimize the search. We also propose a novel discretization layer via mutual information maximization, which is imposed on every search node to model the latent hierarchical categorization in text representation. Extensive experiments conducted on eight diverse real datasets exhibit that DDNAS can consistently outperform the state-of-the-art NAS methods. While DDNAS relies on only three basic operations, i.e., convolution, pooling, and none, to be the candidates of NAS building blocks, its promising performance is noticeable and extensible to obtain further improvement by adding more different operations.

{{</citation>}}


### (58/81) Self-Distilled Quantization: Achieving High Compression Rates in Transformer-Based Language Models (James O' Neill et al., 2023)

{{<citation>}}

James O' Neill, Sourav Dutta. (2023)  
**Self-Distilled Quantization: Achieving High Compression Rates in Transformer-Based Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: GLUE, Language Model, Quantization, Transformer  
[Paper Link](http://arxiv.org/abs/2307.05972v1)  

---


**ABSTRACT**  
We investigate the effects of post-training quantization and quantization-aware training on the generalization of Transformer language models. We present a new method called self-distilled quantization (SDQ) that minimizes accumulative quantization errors and outperforms baselines. We apply SDQ to multilingual models XLM-R-Base and InfoXLM-Base and demonstrate that both models can be reduced from 32-bit floating point weights to 8-bit integer weights while maintaining a high level of performance on the XGLUE benchmark. Our results also highlight the challenges of quantizing multilingual models, which must generalize to languages they were not fine-tuned on.

{{</citation>}}


### (59/81) Predictive Pipelined Decoding: A Compute-Latency Trade-off for Exact LLM Decoding (Seongjun Yang et al., 2023)

{{<citation>}}

Seongjun Yang, Gibbeum Lee, Jaewoong Cho, Dimitris Papailiopoulos, Kangwook Lee. (2023)  
**Predictive Pipelined Decoding: A Compute-Latency Trade-off for Exact LLM Decoding**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.05908v1)  

---


**ABSTRACT**  
This paper presents "Predictive Pipelined Decoding (PPD)," an approach that speeds up greedy decoding in Large Language Models (LLMs) while maintaining the exact same output as the original decoding. Unlike conventional strategies, PPD employs additional compute resources to parallelize the initiation of subsequent token decoding during the current token decoding. This innovative method reduces decoding latency and reshapes the understanding of trade-offs in LLM decoding strategies. We have developed a theoretical framework that allows us to analyze the trade-off between computation and latency. Using this framework, we can analytically estimate the potential reduction in latency associated with our proposed method, achieved through the assessment of the match rate, represented as p_correct. The results demonstrate that the use of extra computational resources has the potential to accelerate LLM greedy decoding.

{{</citation>}}


## cs.RO (5)



### (60/81) Bi-Touch: Bimanual Tactile Manipulation with Sim-to-Real Deep Reinforcement Learning (Yijiong Lin et al., 2023)

{{<citation>}}

Yijiong Lin, Alex Church, Max Yang, Haoran Li, John Lloyd, Dandan Zhang, Nathan F. Lepora. (2023)  
**Bi-Touch: Bimanual Tactile Manipulation with Sim-to-Real Deep Reinforcement Learning**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.06423v1)  

---


**ABSTRACT**  
Bimanual manipulation with tactile feedback will be key to human-level robot dexterity. However, this topic is less explored than single-arm settings, partly due to the availability of suitable hardware along with the complexity of designing effective controllers for tasks with relatively large state-action spaces. Here we introduce a dual-arm tactile robotic system (Bi-Touch) based on the Tactile Gym 2.0 setup that integrates two affordable industrial-level robot arms with low-cost high-resolution tactile sensors (TacTips). We present a suite of bimanual manipulation tasks tailored towards tactile feedback: bi-pushing, bi-reorienting and bi-gathering. To learn effective policies, we introduce appropriate reward functions for these tasks and propose a novel goal-update mechanism with deep reinforcement learning. We also apply these policies to real-world settings with a tactile sim-to-real approach. Our analysis highlights and addresses some challenges met during the sim-to-real application, e.g. the learned policy tended to squeeze an object in the bi-reorienting task due to the sim-to-real gap. Finally, we demonstrate the generalizability and robustness of this system by experimenting with different unseen objects with applied perturbations in the real world. Code and videos are available at https://sites.google.com/view/bi-touch/.

{{</citation>}}


### (61/81) SayPlan: Grounding Large Language Models using 3D Scene Graphs for Scalable Task Planning (Krishan Rana et al., 2023)

{{<citation>}}

Krishan Rana, Jesse Haviland, Sourav Garg, Jad Abou-Chakra, Ian Reid, Niko Suenderhauf. (2023)  
**SayPlan: Grounding Large Language Models using 3D Scene Graphs for Scalable Task Planning**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.06135v1)  

---


**ABSTRACT**  
Large language models (LLMs) have demonstrated impressive results in developing generalist planning agents for diverse tasks. However, grounding these plans in expansive, multi-floor, and multi-room environments presents a significant challenge for robotics. We introduce SayPlan, a scalable approach to LLM-based, large-scale task planning for robotics using 3D scene graph (3DSG) representations. To ensure the scalability of our approach, we: (1) exploit the hierarchical nature of 3DSGs to allow LLMs to conduct a semantic search for task-relevant subgraphs from a smaller, collapsed representation of the full graph; (2) reduce the planning horizon for the LLM by integrating a classical path planner and (3) introduce an iterative replanning pipeline that refines the initial plan using feedback from a scene graph simulator, correcting infeasible actions and avoiding planning failures. We evaluate our approach on two large-scale environments spanning up to 3 floors, 36 rooms and 140 objects, and show that our approach is capable of grounding large-scale, long-horizon task plans from abstract, and natural language instruction for a mobile manipulator robot to execute.

{{</citation>}}


### (62/81) Learning Hierarchical Interactive Multi-Object Search for Mobile Manipulation (Fabian Schmalstieg et al., 2023)

{{<citation>}}

Fabian Schmalstieg, Daniel Honerkamp, Tim Welschehold, Abhinav Valada. (2023)  
**Learning Hierarchical Interactive Multi-Object Search for Mobile Manipulation**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-LG, cs-RO, cs.RO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.06125v1)  

---


**ABSTRACT**  
Existing object-search approaches enable robots to search through free pathways, however, robots operating in unstructured human-centered environments frequently also have to manipulate the environment to their needs. In this work, we introduce a novel interactive multi-object search task in which a robot has to open doors to navigate rooms and search inside cabinets and drawers to find target objects. These new challenges require combining manipulation and navigation skills in unexplored environments. We present HIMOS, a hierarchical reinforcement learning approach that learns to compose exploration, navigation, and manipulation skills. To achieve this, we design an abstract high-level action space around a semantic map memory and leverage the explored environment as instance navigation points. We perform extensive experiments in simulation and the real-world that demonstrate that HIMOS effectively transfers to new environments in a zero-shot manner. It shows robustness to unseen subpolicies, failures in their execution, and different robot kinematics. These capabilities open the door to a wide range of downstream tasks across embodied AI and real-world use cases.

{{</citation>}}


### (63/81) VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models (Wenlong Huang et al., 2023)

{{<citation>}}

Wenlong Huang, Chen Wang, Ruohan Zhang, Yunzhu Li, Jiajun Wu, Li Fei-Fei. (2023)  
**VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs-RO, cs.RO  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.05973v1)  

---


**ABSTRACT**  
Large language models (LLMs) are shown to possess a wealth of actionable knowledge that can be extracted for robot manipulation in the form of reasoning and planning. Despite the progress, most still rely on pre-defined motion primitives to carry out the physical interactions with the environment, which remains a major bottleneck. In this work, we aim to synthesize robot trajectories, i.e., a dense sequence of 6-DoF end-effector waypoints, for a large variety of manipulation tasks given an open-set of instructions and an open-set of objects. We achieve this by first observing that LLMs excel at inferring affordances and constraints given a free-form language instruction. More importantly, by leveraging their code-writing capabilities, they can interact with a visual-language model (VLM) to compose 3D value maps to ground the knowledge into the observation space of the agent. The composed value maps are then used in a model-based planning framework to zero-shot synthesize closed-loop robot trajectories with robustness to dynamic perturbations. We further demonstrate how the proposed framework can benefit from online experiences by efficiently learning a dynamics model for scenes that involve contact-rich interactions. We present a large-scale study of the proposed method in both simulated and real-robot environments, showcasing the ability to perform a large variety of everyday manipulation tasks specified in free-form natural language. Project website: https://voxposer.github.io

{{</citation>}}


### (64/81) GRAINS: Proximity Sensing of Objects in Granular Materials (Zeqing Zhang et al., 2023)

{{<citation>}}

Zeqing Zhang, Ruixing Jia, Youcan Yan, Ruihua Han, Shijie Lin, Qian Jiang, Liangjun Zhang, Jia Pan. (2023)  
**GRAINS: Proximity Sensing of Objects in Granular Materials**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.05935v2)  

---


**ABSTRACT**  
Proximity sensing detects an object's presence without contact. However, research has rarely explored proximity sensing in granular materials (GM) due to GM's lack of visual and complex properties. In this paper, we propose a granular-material-embedded autonomous proximity sensing system (GRAINS) based on three granular phenomena (fluidization, jamming, and failure wedge zone). GRAINS can automatically sense buried objects beneath GM in real-time manner (at least ~20 hertz) and perceive them 0.5 ~ 7 centimeters ahead in different granules without the use of vision or touch. We introduce a new spiral trajectory for the probe raking in GM, combining linear and circular motions, inspired by a common granular fluidization technique. Based on the observation of force-raising when granular jamming occurs in the failure wedge zone in front of the probe during its raking, we employ Gaussian process regression to constantly learn and predict the force patterns and detect the force anomaly resulting from granular jamming to identify the proximity sensing of buried objects. Finally, we apply GRAINS to a Bayesian-optimization-algorithm-guided exploration strategy to successfully localize underground objects and outline their distribution using proximity sensing without contact or digging. This work offers a simple yet reliable method with potential for safe operation in building habitation infrastructure on an alien planet without human intervention.

{{</citation>}}


## stat.ML (1)



### (65/81) Spectral-Bias and Kernel-Task Alignment in Physically Informed Neural Networks (Inbar Seroussi et al., 2023)

{{<citation>}}

Inbar Seroussi, Asaf Miron, Zohar Ringel. (2023)  
**Spectral-Bias and Kernel-Task Alignment in Physically Informed Neural Networks**  

---
Primary Category: stat.ML  
Categories: cond-mat-dis-nn, cs-LG, stat-ML, stat.ML  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2307.06362v1)  

---


**ABSTRACT**  
Physically informed neural networks (PINNs) are a promising emerging method for solving differential equations. As in many other deep learning approaches, the choice of PINN design and training protocol requires careful craftsmanship. Here, we suggest a comprehensive theoretical framework that sheds light on this important problem. Leveraging an equivalence between infinitely over-parameterized neural networks and Gaussian process regression (GPR), we derive an integro-differential equation that governs PINN prediction in the large data-set limit -- the Neurally-Informed Equation (NIE). This equation augments the original one by a kernel term reflecting architecture choices and allows quantifying implicit bias induced by the network via a spectral decomposition of the source term in the original differential equation.

{{</citation>}}


## eess.AS (1)



### (66/81) Feature Embeddings from Large-Scale Acoustic Bird Classifiers Enable Few-Shot Transfer Learning (Burooj Ghani et al., 2023)

{{<citation>}}

Burooj Ghani, Tom Denton, Stefan Kahl, Holger Klinck. (2023)  
**Feature Embeddings from Large-Scale Acoustic Bird Classifiers Enable Few-Shot Transfer Learning**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Embedding, Few-Shot  
[Paper Link](http://arxiv.org/abs/2307.06292v1)  

---


**ABSTRACT**  
Automated bioacoustic analysis aids understanding and protection of both marine and terrestrial animals and their habitats across extensive spatiotemporal scales, and typically involves analyzing vast collections of acoustic data. With the advent of deep learning models, classification of important signals from these datasets has markedly improved. These models power critical data analyses for research and decision-making in biodiversity monitoring, animal behaviour studies, and natural resource management. However, deep learning models are often data-hungry and require a significant amount of labeled training data to perform well. While sufficient training data is available for certain taxonomic groups (e.g., common bird species), many classes (such as rare and endangered species, many non-bird taxa, and call-type), lack enough data to train a robust model from scratch. This study investigates the utility of feature embeddings extracted from large-scale audio classification models to identify bioacoustic classes other than the ones these models were originally trained on. We evaluate models on diverse datasets, including different bird calls and dialect types, bat calls, marine mammals calls, and amphibians calls. The embeddings extracted from the models trained on bird vocalization data consistently allowed higher quality classification than the embeddings trained on general audio datasets. The results of this study indicate that high-quality feature embeddings from large-scale acoustic bird classifiers can be harnessed for few-shot transfer learning, enabling the learning of new classes from a limited quantity of training data. Our findings reveal the potential for efficient analyses of novel bioacoustic tasks, even in scenarios where available training data is limited to a few samples.

{{</citation>}}


## cs.MA (1)



### (67/81) Self-Adaptive Large Language Model (LLM)-Based Multiagent Systems (Nathalia Nascimento et al., 2023)

{{<citation>}}

Nathalia Nascimento, Paulo Alencar, Donald Cowan. (2023)  
**Self-Adaptive Large Language Model (LLM)-Based Multiagent Systems**  

---
Primary Category: cs.MA  
Categories: cs-AI, cs-CL, cs-MA, cs.MA  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2307.06187v1)  

---


**ABSTRACT**  
In autonomic computing, self-adaptation has been proposed as a fundamental paradigm to manage the complexity of multiagent systems (MASs). This achieved by extending a system with support to monitor and adapt itself to achieve specific concerns of interest. Communication in these systems is key given that in scenarios involving agent interaction, it enhances cooperation and reduces coordination challenges by enabling direct, clear information exchange. However, improving the expressiveness of the interaction communication with MASs is not without challenges. In this sense, the interplay between self-adaptive systems and effective communication is crucial for future MAS advancements. In this paper, we propose the integration of large language models (LLMs) such as GPT-based technologies into multiagent systems. We anchor our methodology on the MAPE-K model, which is renowned for its robust support in monitoring, analyzing, planning, and executing system adaptations in response to dynamic environments. We also present a practical illustration of the proposed approach, in which we implement and assess a basic MAS-based application. The approach significantly advances the state-of-the-art of self-adaptive systems by proposing a new paradigm for MAS self-adaptation of autonomous systems based on LLM capabilities.

{{</citation>}}


## cs.AI (5)



### (68/81) Reflective Hybrid Intelligence for Meaningful Human Control in Decision-Support Systems (Catholijn M. Jonker et al., 2023)

{{<citation>}}

Catholijn M. Jonker, Luciano Cavalcante Siebert, Pradeep K. Murukannaiah. (2023)  
**Reflective Hybrid Intelligence for Meaningful Human Control in Decision-Support Systems**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CY, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.06159v1)  

---


**ABSTRACT**  
With the growing capabilities and pervasiveness of AI systems, societies must collectively choose between reduced human autonomy, endangered democracies and limited human rights, and AI that is aligned to human and social values, nurturing collaboration, resilience, knowledge and ethical behaviour. In this chapter, we introduce the notion of self-reflective AI systems for meaningful human control over AI systems. Focusing on decision support systems, we propose a framework that integrates knowledge from psychology and philosophy with formal reasoning methods and machine learning approaches to create AI systems responsive to human values and social norms. We also propose a possible research approach to design and develop self-reflective capability in AI systems. Finally, we argue that self-reflective AI systems can lead to self-reflective hybrid systems (human + AI), thus increasing meaningful human control and empowering human moral reasoning by providing comprehensible information and insights on possible human moral blind spots.

{{</citation>}}


### (69/81) Maneuver Decision-Making Through Automatic Curriculum Reinforcement Learning Without Handcrafted Reward functions (Zhang Hong-Peng, 2023)

{{<citation>}}

Zhang Hong-Peng. (2023)  
**Maneuver Decision-Making Through Automatic Curriculum Reinforcement Learning Without Handcrafted Reward functions**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs-RO, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.06152v1)  

---


**ABSTRACT**  
Maneuver decision-making is the core of unmanned combat aerial vehicle for autonomous air combat. To solve this problem, we propose an automatic curriculum reinforcement learning method, which enables agents to learn effective decisions in air combat from scratch. The range of initial states are used for distinguishing curricula of different difficulty levels, thereby maneuver decision is divided into a series of sub-tasks from easy to difficult, and test results are used to change sub-tasks. As sub-tasks change, agents gradually learn to complete a series of sub-tasks from easy to difficult, enabling them to make effective maneuvering decisions to cope with various states without the need to spend effort designing reward functions. The ablation studied show that the automatic curriculum learning proposed in this article is an essential component for training through reinforcement learning, namely, agents cannot complete effective decisions without curriculum learning. Simulation experiments show that, after training, agents are able to make effective decisions given different states, including tracking, attacking and escaping, which are both rational and interpretable.

{{</citation>}}


### (70/81) CLAIMED -- the open source framework for building coarse-grained operators for accelerated discovery in science (Romeo Kienzler et al., 2023)

{{<citation>}}

Romeo Kienzler, Rafflesia Khan, Jerome Nilmeier, Ivan Nesic, Ibrahim Haddad. (2023)  
**CLAIMED -- the open source framework for building coarse-grained operators for accelerated discovery in science**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-DB, cs-LG, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.06824v1)  

---


**ABSTRACT**  
In modern data-driven science, reproducibility and reusability are key challenges. Scientists are well skilled in the process from data to publication. Although some publication channels require source code and data to be made accessible, rerunning and verifying experiments is usually hard due to a lack of standards. Therefore, reusing existing scientific data processing code from state-of-the-art research is hard as well. This is why we introduce CLAIMED, which has a proven track record in scientific research for addressing the repeatability and reusability issues in modern data-driven science. CLAIMED is a framework to build reusable operators and scalable scientific workflows by supporting the scientist to draw from previous work by re-composing workflows from existing libraries of coarse-grained scientific operators. Although various implementations exist, CLAIMED is programming language, scientific library, and execution environment agnostic.

{{</citation>}}


### (71/81) AI-Generated Imagery: A New Era for the `Readymade' (Amy Smith et al., 2023)

{{<citation>}}

Amy Smith, Michael Cook. (2023)  
**AI-Generated Imagery: A New Era for the `Readymade'**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CY, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.06033v1)  

---


**ABSTRACT**  
While the term `art' defies any concrete definition, this paper aims to examine how digital images produced by generative AI systems, such as Midjourney, have come to be so regularly referred to as such. The discourse around the classification of AI-generated imagery as art is currently somewhat homogeneous, lacking the more nuanced aspects that would apply to more traditional modes of artistic media production. This paper aims to bring important philosophical considerations to the surface of the discussion around AI-generated imagery in the context of art. We employ existing philosophical frameworks and theories of language to suggest that some AI-generated imagery, by virtue of its visual properties within these frameworks, can be presented as `readymades' for consideration as art.

{{</citation>}}


### (72/81) An Effective and Efficient Time-aware Entity Alignment Framework via Two-aspect Three-view Label Propagation (Li Cai et al., 2023)

{{<citation>}}

Li Cai, Xin Mao, Youshao Xiao, Changxu Wu, Man Lan. (2023)  
**An Effective and Efficient Time-aware Entity Alignment Framework via Two-aspect Three-view Label Propagation**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: Entity Alignment, GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2307.06013v1)  

---


**ABSTRACT**  
Entity alignment (EA) aims to find the equivalent entity pairs between different knowledge graphs (KGs), which is crucial to promote knowledge fusion. With the wide use of temporal knowledge graphs (TKGs), time-aware EA (TEA) methods appear to enhance EA. Existing TEA models are based on Graph Neural Networks (GNN) and achieve state-of-the-art (SOTA) performance, but it is difficult to transfer them to large-scale TKGs due to the scalability issue of GNN. In this paper, we propose an effective and efficient non-neural EA framework between TKGs, namely LightTEA, which consists of four essential components: (1) Two-aspect Three-view Label Propagation, (2) Sparse Similarity with Temporal Constraints, (3) Sinkhorn Operator, and (4) Temporal Iterative Learning. All of these modules work together to improve the performance of EA while reducing the time consumption of the model. Extensive experiments on public datasets indicate that our proposed model significantly outperforms the SOTA methods for EA between TKGs, and the time consumed by LightTEA is only dozens of seconds at most, no more than 10% of the most efficient TEA method.

{{</citation>}}


## cs.SD (2)



### (73/81) Can Large Language Models Aid in Annotating Speech Emotional Data? Uncovering New Frontiers (Siddique Latif et al., 2023)

{{<citation>}}

Siddique Latif, Muhammad Usama, Mohammad Ibrahim Malik, Björn W. Schuller. (2023)  
**Can Large Language Models Aid in Annotating Speech Emotional Data? Uncovering New Frontiers**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2307.06090v1)  

---


**ABSTRACT**  
Despite recent advancements in speech emotion recognition (SER) models, state-of-the-art deep learning (DL) approaches face the challenge of the limited availability of annotated data. Large language models (LLMs) have revolutionised our understanding of natural language, introducing emergent properties that broaden comprehension in language, speech, and vision. This paper examines the potential of LLMs to annotate abundant speech data, aiming to enhance the state-of-the-art in SER. We evaluate this capability across various settings using publicly available speech emotion classification datasets. Leveraging ChatGPT, we experimentally demonstrate the promising role of LLMs in speech emotion data annotation. Our evaluation encompasses single-shot and few-shots scenarios, revealing performance variability in SER. Notably, we achieve improved results through data augmentation, incorporating ChatGPT-annotated samples into existing datasets. Our work uncovers new frontiers in speech emotion classification, highlighting the increasing significance of LLMs in this field moving forward.

{{</citation>}}


### (74/81) Language-Routing Mixture of Experts for Multilingual and Code-Switching Speech Recognition (Wenxuan Wang et al., 2023)

{{<citation>}}

Wenxuan Wang, Guodong Ma, Yuke Li, Binbin Du. (2023)  
**Language-Routing Mixture of Experts for Multilingual and Code-Switching Speech Recognition**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Multilingual, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2307.05956v2)  

---


**ABSTRACT**  
Multilingual speech recognition for both monolingual and code-switching speech is a challenging task. Recently, based on the Mixture of Experts (MoE), many works have made good progress in multilingual and code-switching ASR, but present huge computational complexity with the increase of supported languages. In this work, we propose a computation-efficient network named Language-Routing Mixture of Experts (LR-MoE) for multilingual and code-switching ASR. LR-MoE extracts language-specific representations through the Mixture of Language Experts (MLE), which is guided to learn by a frame-wise language routing mechanism. The weight-shared frame-level language identification (LID) network is jointly trained as the shared pre-router of each MoE layer. Experiments show that the proposed method significantly improves multilingual and code-switching speech recognition performances over baseline with comparable computational efficiency.

{{</citation>}}


## cs.MM (1)



### (75/81) Semantic Communications System with Model Division Multiple Access and Controllable Coding Rate for Point Cloud (Xiaoyi Liu et al., 2023)

{{<citation>}}

Xiaoyi Liu, Haotai Liang, Zhicheng Bao, Chen Dong, Xiaodong Xu. (2023)  
**Semantic Communications System with Model Division Multiple Access and Controllable Coding Rate for Point Cloud**  

---
Primary Category: cs.MM  
Categories: cs-MM, cs.MM  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.06027v1)  

---


**ABSTRACT**  
Point cloud, as a 3D representation, is widely used in autonomous driving, virtual reality (VR), and augmented reality (AR). However, traditional communication systems think that the point cloud's semantic information is irrelevant to communication, which hinders the efficient transmission of point clouds in the era of artificial intelligence (AI). This paper proposes a point cloud based semantic communication system (PCSC), which uses AI-based encoding techniques to extract the semantic information of the point cloud and joint source-channel coding (JSCC) technology to overcome the distortion caused by noise channels and solve the "cliff effect" in traditional communication. In addition, the system realizes the controllable coding rate without fine-tuning the network. The method analyzes the coded semantic vector's importance and discards semantically-unimportant information, thereby improving the transmission efficiency. Besides, PCSC and the recently proposed non-orthogonal model division multiple access (MDMA) technology are combined to design a point cloud MDMA transmission system (M-PCSC) for multi-user transmission. Relevant experimental results show that the proposed method outperforms the traditional method 10dB in the same channel bandwidth ratio under the PSNR D1 and PSNR D2 metrics. In terms of transmission, the proposed method can effectively solve the "cliff effect" in the traditional methods.

{{</citation>}}


## cs.IR (1)



### (76/81) Contrastive Learning for Conversion Rate Prediction (Wentao Ouyang et al., 2023)

{{<citation>}}

Wentao Ouyang, Rui Dong, Xiuwu Zhang, Chaofeng Guo, Jinmei Luo, Xiangzheng Liu, Yanlong Du. (2023)  
**Contrastive Learning for Conversion Rate Prediction**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs-LG, cs.IR  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2307.05974v1)  

---


**ABSTRACT**  
Conversion rate (CVR) prediction plays an important role in advertising systems. Recently, supervised deep neural network-based models have shown promising performance in CVR prediction. However, they are data hungry and require an enormous amount of training data. In online advertising systems, although there are millions to billions of ads, users tend to click only a small set of them and to convert on an even smaller set. This data sparsity issue restricts the power of these deep models. In this paper, we propose the Contrastive Learning for CVR prediction (CL4CVR) framework. It associates the supervised CVR prediction task with a contrastive learning task, which can learn better data representations exploiting abundant unlabeled data and improve the CVR prediction performance. To tailor the contrastive learning task to the CVR prediction problem, we propose embedding masking (EM), rather than feature masking, to create two views of augmented samples. We also propose a false negative elimination (FNE) component to eliminate samples with the same feature as the anchor sample, to account for the natural property in user behavior data. We further propose a supervised positive inclusion (SPI) component to include additional positive samples for each anchor sample, in order to make full use of sparse but precious user conversion events. Experimental results on two real-world conversion datasets demonstrate the superior performance of CL4CVR. The source code is available at https://github.com/DongRuiHust/CL4CVR.

{{</citation>}}


## cs.AR (1)



### (77/81) A 137.5 TOPS/W SRAM Compute-in-Memory Macro with 9-b Memory Cell-Embedded ADCs and Signal Margin Enhancement Techniques for AI Edge Applications (Xiaomeng Wang et al., 2023)

{{<citation>}}

Xiaomeng Wang, Fengshi Tian, Xizi Chen, Jiakun Zheng, Xuejiao Liu, Fengbin Tu, Jie Yang, Mohamad Sawan, Kwang-Ting Cheng, Chi-Ying Tsui. (2023)  
**A 137.5 TOPS/W SRAM Compute-in-Memory Macro with 9-b Memory Cell-Embedded ADCs and Signal Margin Enhancement Techniques for AI Edge Applications**  

---
Primary Category: cs.AR  
Categories: cs-AR, cs-NE, cs.AR, eess-SP  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.05944v3)  

---


**ABSTRACT**  
In this paper, we propose a high-precision SRAM-based CIM macro that can perform 4x4-bit MAC operations and yield 9-bit signed output. The inherent discharge branches of SRAM cells are utilized to apply time-modulated MAC and 9-bit ADC readout operations on two bit-line capacitors. The same principle is used for both MAC and A-to-D conversion ensuring high linearity and thus supporting large number of analog MAC accumulations. The memory cell-embedded ADC eliminates the use of separate ADCs and enhances energy and area efficiency. Additionally, two signal margin enhancement techniques, namely the MAC-folding and boosted-clipping schemes, are proposed to further improve the CIM computation accuracy.

{{</citation>}}


## cs.ET (1)



### (78/81) Real-time Trading System based on Selections of Potentially Profitable, Uncorrelated, and Balanced Stocks by NP-hard Combinatorial Optimization (Kosuke Tatsumura et al., 2023)

{{<citation>}}

Kosuke Tatsumura, Ryo Hidaka, Jun Nakayama, Tomoya Kashimata, Masaya Yamasaki. (2023)  
**Real-time Trading System based on Selections of Potentially Profitable, Uncorrelated, and Balanced Stocks by NP-hard Combinatorial Optimization**  

---
Primary Category: cs.ET  
Categories: cs-ET, cs.ET, q-fin-ST  
Keywords: Financial  
[Paper Link](http://arxiv.org/abs/2307.06339v1)  

---


**ABSTRACT**  
Financial portfolio construction problems are often formulated as quadratic and discrete (combinatorial) optimization that belong to the nondeterministic polynomial time (NP)-hard class in computational complexity theory. Ising machines are hardware devices that work in quantum-mechanical/quantum-inspired principles for quickly solving NP-hard optimization problems, which potentially enable making trading decisions based on NP-hard optimization in the time constraints for high-speed trading strategies. Here we report a real-time stock trading system that determines long(buying)/short(selling) positions through NP-hard portfolio optimization for improving the Sharpe ratio using an embedded Ising machine based on a quantum-inspired algorithm called simulated bifurcation. The Ising machine selects a balanced (delta-neutral) group of stocks from an $N$-stock universe according to an objective function involving maximizing instantaneous expected returns defined as deviations from volume-weighted average prices and minimizing the summation of statistical correlation factors (for diversification). It has been demonstrated in the Tokyo Stock Exchange that the trading strategy based on NP-hard portfolio optimization for $N$=128 is executable with the FPGA (field-programmable gate array)-based trading system with a response latency of 164 $\mu$s.

{{</citation>}}


## cs.NI (1)



### (79/81) FIS-ONE: Floor Identification System with One Label for Crowdsourced RF Signals (Weipeng Zhuo et al., 2023)

{{<citation>}}

Weipeng Zhuo, Ka Ho Chiu, Jierun Chen, Ziqi Zhao, S. -H. Gary Chan, Sangtae Ha, Chul-Ho Lee. (2023)  
**FIS-ONE: Floor Identification System with One Label for Crowdsourced RF Signals**  

---
Primary Category: cs.NI  
Categories: cs-LG, cs-NI, cs.NI, eess-SP  
Keywords: Microsoft  
[Paper Link](http://arxiv.org/abs/2307.05914v1)  

---


**ABSTRACT**  
Floor labels of crowdsourced RF signals are crucial for many smart-city applications, such as multi-floor indoor localization, geofencing, and robot surveillance. To build a prediction model to identify the floor number of a new RF signal upon its measurement, conventional approaches using the crowdsourced RF signals assume that at least few labeled signal samples are available on each floor. In this work, we push the envelope further and demonstrate that it is technically feasible to enable such floor identification with only one floor-labeled signal sample on the bottom floor while having the rest of signal samples unlabeled.   We propose FIS-ONE, a novel floor identification system with only one labeled sample. FIS-ONE consists of two steps, namely signal clustering and cluster indexing. We first build a bipartite graph to model the RF signal samples and obtain a latent representation of each node (each signal sample) using our attention-based graph neural network model so that the RF signal samples can be clustered more accurately. Then, we tackle the problem of indexing the clusters with proper floor labels, by leveraging the observation that signals from an access point can be detected on different floors, i.e., signal spillover. Specifically, we formulate a cluster indexing problem as a combinatorial optimization problem and show that it is equivalent to solving a traveling salesman problem, whose (near-)optimal solution can be found efficiently. We have implemented FIS-ONE and validated its effectiveness on the Microsoft dataset and in three large shopping malls. Our results show that FIS-ONE outperforms other baseline algorithms significantly, with up to 23% improvement in adjusted rand index and 25% improvement in normalized mutual information using only one floor-labeled signal sample.

{{</citation>}}


## cs.HC (1)



### (80/81) Exploring the Sector-Specific Influence and Response of AI Tools: A Critical Review (Hitesh Mohapatra et al., 2023)

{{<citation>}}

Hitesh Mohapatra, Soumya Ranjan Mishra. (2023)  
**Exploring the Sector-Specific Influence and Response of AI Tools: A Critical Review**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.05909v1)  

---


**ABSTRACT**  
AI Tool is designed to generate human-like responses in natural language conversations. Using deep learning techniques, AI Tool has been trained on a diverse range of internet text to understand and generate coherent responses to a wide array of prompts and questions. It can provide information, engage in conversations, assist with tasks, and even offer creative suggestions. The underlying technology behind AI Tool is a transformer neural network. Transformers excel at capturing long-range dependencies in text, making them well-suited for language-related tasks. AI Tool, has 175 billion parameters, making it one of the largest and most powerful language models to date. AI Tool has been trained on a massive corpus of text from the internet, which allows it to leverage a broad understanding of language, general knowledge, and various domains. While AI Tool aims to provide accurate and helpful responses, it may occasionally produce incorrect or nonsensical answers. It's essential to critically evaluate the information it provides and verify it from reliable sources when necessary. This work presents an overview on AI Tool. It will helps to research community and others users to understand the uses of AI Tool and its interaction pattern.

{{</citation>}}


## eess.SY (1)



### (81/81) Knowledge-Driven Resource Allocation for D2D Networks: A WMMSE Unrolled Graph Neural Network Approach (Hao Yang et al., 2023)

{{<citation>}}

Hao Yang, Nan Cheng, Ruijin Sun, Wei Quan, Rong Chai, Khalid Aldubaikhy, Abdullah Alqasir, Xuemin Shen. (2023)  
**Knowledge-Driven Resource Allocation for D2D Networks: A WMMSE Unrolled Graph Neural Network Approach**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2307.05882v1)  

---


**ABSTRACT**  
This paper proposes an novel knowledge-driven approach for resource allocation in device-to-device (D2D) networks using a graph neural network (GNN) architecture. To meet the millisecond-level timeliness and scalability required for the dynamic network environment, our proposed approach incorporates the deep unrolling of the weighted minimum mean square error (WMMSE) algorithm, referred to as domain knowledge, into GNN, thereby reducing computational delay and sample complexity while adapting to various data distributions. Specifically, the aggregation and update functions in the GNN architecture are designed by utilizing the summation and power calculation components of the WMMSE algorithm, which leads to improved model generalization and interpretabiliy. Theoretical analysis of the proposed approach reveals its capability to simplify intricate end-to-end mappings and diminish the model exploration space, resulting in increased network expressiveness and enhanced optimization performance. Simulation results demonstrate the robustness, scalability, and strong performance of the proposed knowledge-driven resource allocation approach across diverse communication topologies without retraining. Our findings contribute to the development of efficient and scalable wireless resource management solutions for distributed and dynamic networks with strict latency requirements.

{{</citation>}}
