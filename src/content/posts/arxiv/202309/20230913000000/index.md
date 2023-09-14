---
draft: false
title: "arXiv @ 2023.09.13"
date: 2023-09-13
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.09.13"
    identifier: arxiv_20230913
    parent: 202309_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CR (3)](#cscr-3)
- [cs.LG (16)](#cslg-16)
- [cs.PL (1)](#cspl-1)
- [cs.CV (18)](#cscv-18)
- [cs.CL (30)](#cscl-30)
- [eess.SP (1)](#eesssp-1)
- [stat.ML (1)](#statml-1)
- [cs.SD (1)](#cssd-1)
- [cs.RO (3)](#csro-3)
- [q-fin.ST (1)](#q-finst-1)
- [cs.NI (2)](#csni-2)
- [cs.SE (5)](#csse-5)
- [cs.SI (2)](#cssi-2)
- [cs.CE (1)](#csce-1)
- [cs.AI (4)](#csai-4)
- [cs.HC (1)](#cshc-1)
- [cs.DS (2)](#csds-2)
- [eess.SY (1)](#eesssy-1)
- [eess.IV (1)](#eessiv-1)
- [cs.IT (1)](#csit-1)
- [eess.AS (1)](#eessas-1)
- [cs.IR (1)](#csir-1)

## cs.CR (3)



### (1/97) SkillScanner: Detecting Policy-Violating Voice Applications Through Static Analysis at the Development Phase (Song Liao et al., 2023)

{{<citation>}}

Song Liao, Long Cheng, Haipeng Cai, Linke Guo, Hongxin Hu. (2023)  
**SkillScanner: Detecting Policy-Violating Voice Applications Through Static Analysis at the Development Phase**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Amazon  
[Paper Link](http://arxiv.org/abs/2309.05867v1)  

---


**ABSTRACT**  
The Amazon Alexa marketplace is the largest Voice Personal Assistant (VPA) platform with over 100,000 voice applications (i.e., skills) published to the skills store. In an effort to maintain the quality and trustworthiness of voice-apps, Amazon Alexa has implemented a set of policy requirements to be adhered to by third-party skill developers. However, recent works reveal the prevalence of policy-violating skills in the current skills store. To understand the causes of policy violations in skills, we first conduct a user study with 34 third-party skill developers focusing on whether they are aware of the various policy requirements defined by the Amazon Alexa platform. Our user study results show that there is a notable gap between VPA's policy requirements and skill developers' practices. As a result, it is inevitable that policy-violating skills will be published.   To prevent the inflow of new policy-breaking skills to the skills store from the source, it is critical to identify potential policy violations at the development phase. In this work, we design and develop SkillScanner, an efficient static code analysis tool to facilitate third-party developers to detect policy violations early in the skill development lifecycle. To evaluate the performance of SkillScanner, we conducted an empirical study on 2,451 open source skills collected from GitHub. SkillScanner effectively identified 1,328 different policy violations from 786 skills. Our results suggest that 32% of these policy violations are introduced through code duplication (i.e., code copy and paste). In particular, we found that 42 skill code examples from potential Alexa's official accounts (e.g., "alexa" and "alexa-samples" on GitHub) contain policy violations, which lead to 81 policy violations in other skills due to the copy-pasted code snippets from these Alexa's code examples.

{{</citation>}}


### (2/97) Unveiling the Sentinels: Assessing AI Performance in Cybersecurity Peer Review (Liang Niu et al., 2023)

{{<citation>}}

Liang Niu, Nian Xue, Christina Pöpper. (2023)  
**Unveiling the Sentinels: Assessing AI Performance in Cybersecurity Peer Review**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs.CR  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2309.05457v1)  

---


**ABSTRACT**  
Peer review is the method employed by the scientific community for evaluating research advancements. In the field of cybersecurity, the practice of double-blind peer review is the de-facto standard. This paper touches on the holy grail of peer reviewing and aims to shed light on the performance of AI in reviewing for academic security conferences. Specifically, we investigate the predictability of reviewing outcomes by comparing the results obtained from human reviewers and machine-learning models. To facilitate our study, we construct a comprehensive dataset by collecting thousands of papers from renowned computer science conferences and the arXiv preprint website. Based on the collected data, we evaluate the prediction capabilities of ChatGPT and a two-stage classification approach based on the Doc2Vec model with various classifiers. Our experimental evaluation of review outcome prediction using the Doc2Vec-based approach performs significantly better than the ChatGPT and achieves an accuracy of over 90%. While analyzing the experimental results, we identify the potential advantages and limitations of the tested ML models. We explore areas within the paper-reviewing process that can benefit from automated support approaches, while also recognizing the irreplaceable role of human intellect in certain aspects that cannot be matched by state-of-the-art AI techniques.

{{</citation>}}


### (3/97) FuzzLLM: A Novel and Universal Fuzzing Framework for Proactively Discovering Jailbreak Vulnerabilities in Large Language Models (Dongyu Yao et al., 2023)

{{<citation>}}

Dongyu Yao, Jianshu Zhang, Ian G. Harris, Marcel Carlsson. (2023)  
**FuzzLLM: A Novel and Universal Fuzzing Framework for Proactively Discovering Jailbreak Vulnerabilities in Large Language Models**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.05274v1)  

---


**ABSTRACT**  
Jailbreak vulnerabilities in Large Language Models (LLMs), which exploit meticulously crafted prompts to elicit content that violates service guidelines, have captured the attention of research communities. While model owners can defend against individual jailbreak prompts through safety training strategies, this relatively passive approach struggles to handle the broader category of similar jailbreaks. To tackle this issue, we introduce FuzzLLM, an automated fuzzing framework designed to proactively test and discover jailbreak vulnerabilities in LLMs. We utilize templates to capture the structural integrity of a prompt and isolate key features of a jailbreak class as constraints. By integrating different base classes into powerful combo attacks and varying the elements of constraints and prohibited questions, FuzzLLM enables efficient testing with reduced manual effort. Extensive experiments demonstrate FuzzLLM's effectiveness and comprehensiveness in vulnerability discovery across various LLMs.

{{</citation>}}


## cs.LG (16)



### (4/97) Uncovering mesa-optimization algorithms in Transformers (Johannes von Oswald et al., 2023)

{{<citation>}}

Johannes von Oswald, Eyvind Niklasson, Maximilian Schlegel, Seijin Kobayashi, Nicolas Zucchet, Nino Scherrer, Nolan Miller, Mark Sandler, Blaise Agüera y Arcas, Max Vladymyrov, Razvan Pascanu, João Sacramento. (2023)  
**Uncovering mesa-optimization algorithms in Transformers**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.05858v1)  

---


**ABSTRACT**  
Transformers have become the dominant model in deep learning, but the reason for their superior performance is poorly understood. Here, we hypothesize that the strong performance of Transformers stems from an architectural bias towards mesa-optimization, a learned process running within the forward pass of a model consisting of the following two steps: (i) the construction of an internal learning objective, and (ii) its corresponding solution found through optimization. To test this hypothesis, we reverse-engineer a series of autoregressive Transformers trained on simple sequence modeling tasks, uncovering underlying gradient-based mesa-optimization algorithms driving the generation of predictions. Moreover, we show that the learned forward-pass optimization algorithm can be immediately repurposed to solve supervised few-shot tasks, suggesting that mesa-optimization might underlie the in-context learning capabilities of large language models. Finally, we propose a novel self-attention layer, the mesa-layer, that explicitly and efficiently solves optimization problems specified in context. We find that this layer can lead to improved performance in synthetic and preliminary language modeling experiments, adding weight to our hypothesis that mesa-optimization is an important operation hidden within the weights of trained Transformers.

{{</citation>}}


### (5/97) ChemSpaceAL: An Efficient Active Learning Methodology Applied to Protein-Specific Molecular Generation (Gregory W. Kyro et al., 2023)

{{<citation>}}

Gregory W. Kyro, Anton Morgunov, Rafael I. Brent, Victor S. Batista. (2023)  
**ChemSpaceAL: An Efficient Active Learning Methodology Applied to Protein-Specific Molecular Generation**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, q-bio-BM  
Keywords: Active Learning, GPT  
[Paper Link](http://arxiv.org/abs/2309.05853v1)  

---


**ABSTRACT**  
The incredible capabilities of generative artificial intelligence models have inevitably led to their application in the domain of drug discovery. It is therefore of tremendous interest to develop methodologies that enhance the abilities and applicability of these powerful tools. In this work, we present a novel and efficient semi-supervised active learning methodology that allows for the fine-tuning of a generative model with respect to an objective function by strategically operating within a constructed representation of the sample space. In the context of targeted molecular generation, we demonstrate the ability to fine-tune a GPT-based molecular generator with respect to an attractive interaction-based scoring function by strategically operating within a chemical space proxy, thereby maximizing attractive interactions between the generated molecules and a protein target. Importantly, our approach does not require the individual evaluation of all data points that are used for fine-tuning, enabling the incorporation of computationally expensive metrics. We are hopeful that the inherent generality of this methodology ensures that it will remain applicable as this exciting field evolves. To facilitate implementation and reproducibility, we have made all of our software available through the open-source ChemSpaceAL Python package.

{{</citation>}}


### (6/97) Effective Abnormal Activity Detection on Multivariate Time Series Healthcare Data (Mengjia Niu et al., 2023)

{{<citation>}}

Mengjia Niu, Yuchen Zhao, Hamed Haddadi. (2023)  
**Effective Abnormal Activity Detection on Multivariate Time Series Healthcare Data**  

---
Primary Category: cs.LG  
Categories: J-3; I-2-6, cs-AI, cs-LG, cs.LG  
Keywords: Anomaly Detection, Time Series  
[Paper Link](http://arxiv.org/abs/2309.05845v1)  

---


**ABSTRACT**  
Multivariate time series (MTS) data collected from multiple sensors provide the potential for accurate abnormal activity detection in smart healthcare scenarios. However, anomalies exhibit diverse patterns and become unnoticeable in MTS data. Consequently, achieving accurate anomaly detection is challenging since we have to capture both temporal dependencies of time series and inter-relationships among variables. To address this problem, we propose a Residual-based Anomaly Detection approach, Rs-AD, for effective representation learning and abnormal activity detection. We evaluate our scheme on a real-world gait dataset and the experimental results demonstrate an F1 score of 0.839.

{{</citation>}}


### (7/97) Optimizing Audio Augmentations for Contrastive Learning of Health-Related Acoustic Signals (Louis Blankemeier et al., 2023)

{{<citation>}}

Louis Blankemeier, Sebastien Baur, Wei-Hung Weng, Jake Garrison, Yossi Matias, Shruthi Prabhakara, Diego Ardila, Zaid Nabulsi. (2023)  
**Optimizing Audio Augmentations for Contrastive Learning of Health-Related Acoustic Signals**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SD, cs.LG, eess-AS  
Keywords: Augmentation, Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2309.05843v1)  

---


**ABSTRACT**  
Health-related acoustic signals, such as cough and breathing sounds, are relevant for medical diagnosis and continuous health monitoring. Most existing machine learning approaches for health acoustics are trained and evaluated on specific tasks, limiting their generalizability across various healthcare applications. In this paper, we leverage a self-supervised learning framework, SimCLR with a Slowfast NFNet backbone, for contrastive learning of health acoustics. A crucial aspect of optimizing Slowfast NFNet for this application lies in identifying effective audio augmentations. We conduct an in-depth analysis of various audio augmentation strategies and demonstrate that an appropriate augmentation strategy enhances the performance of the Slowfast NFNet audio encoder across a diverse set of health acoustic tasks. Our findings reveal that when augmentations are combined, they can produce synergistic effects that exceed the benefits seen when each is applied individually.

{{</citation>}}


### (8/97) Exploring Geometric Deep Learning For Precipitation Nowcasting (Shan Zhao et al., 2023)

{{<citation>}}

Shan Zhao, Sudipan Saha, Zhitong Xiong, Niklas Boers, Xiao Xiang Zhu. (2023)  
**Exploring Geometric Deep Learning For Precipitation Nowcasting**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, physics-ao-ph  
Keywords: Graph Convolutional Network  
[Paper Link](http://arxiv.org/abs/2309.05828v1)  

---


**ABSTRACT**  
Precipitation nowcasting (up to a few hours) remains a challenge due to the highly complex local interactions that need to be captured accurately. Convolutional Neural Networks rely on convolutional kernels convolving with grid data and the extracted features are trapped by limited receptive field, typically expressed in excessively smooth output compared to ground truth. Thus they lack the capacity to model complex spatial relationships among the grids. Geometric deep learning aims to generalize neural network models to non-Euclidean domains. Such models are more flexible in defining nodes and edges and can effectively capture dynamic spatial relationship among geographical grids. Motivated by this, we explore a geometric deep learning-based temporal Graph Convolutional Network (GCN) for precipitation nowcasting. The adjacency matrix that simulates the interactions among grid cells is learned automatically by minimizing the L1 loss between prediction and ground truth pixel value during the training procedure. Then, the spatial relationship is refined by GCN layers while the temporal information is extracted by 1D convolution with various kernel lengths. The neighboring information is fed as auxiliary input layers to improve the final result. We test the model on sequences of radar reflectivity maps over the Trento/Italy area. The results show that GCNs improves the effectiveness of modeling the local details of the cloud profile as well as the prediction accuracy by achieving decreased error measures.

{{</citation>}}


### (9/97) KD-FixMatch: Knowledge Distillation Siamese Neural Networks (Chien-Chih Wang et al., 2023)

{{<citation>}}

Chien-Chih Wang, Shaoyuan Xu, Jinmiao Fu, Yang Liu, Bryan Wang. (2023)  
**KD-FixMatch: Knowledge Distillation Siamese Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2309.05826v1)  

---


**ABSTRACT**  
Semi-supervised learning (SSL) has become a crucial approach in deep learning as a way to address the challenge of limited labeled data. The success of deep neural networks heavily relies on the availability of large-scale high-quality labeled data. However, the process of data labeling is time-consuming and unscalable, leading to shortages in labeled data. SSL aims to tackle this problem by leveraging additional unlabeled data in the training process. One of the popular SSL algorithms, FixMatch, trains identical weight-sharing teacher and student networks simultaneously using a siamese neural network (SNN). However, it is prone to performance degradation when the pseudo labels are heavily noisy in the early training stage. We present KD-FixMatch, a novel SSL algorithm that addresses the limitations of FixMatch by incorporating knowledge distillation. The algorithm utilizes a combination of sequential and simultaneous training of SNNs to enhance performance and reduce performance degradation. Firstly, an outer SNN is trained using labeled and unlabeled data. After that, the network of the well-trained outer SNN generates pseudo labels for the unlabeled data, from which a subset of unlabeled data with trusted pseudo labels is then carefully created through high-confidence sampling and deep embedding clustering. Finally, an inner SNN is trained with the labeled data, the unlabeled data, and the subset of unlabeled data with trusted pseudo labels. Experiments on four public data sets demonstrate that KD-FixMatch outperforms FixMatch in all cases. Our results indicate that KD-FixMatch has a better training starting point that leads to improved model performance compared to FixMatch.

{{</citation>}}


### (10/97) Enhancing Hyperedge Prediction with Context-Aware Self-Supervised Learning (Yunyong Ko et al., 2023)

{{<citation>}}

Yunyong Ko, Hanghang Tong, Sang-Wook Kim. (2023)  
**Enhancing Hyperedge Prediction with Context-Aware Self-Supervised Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SI, cs.LG  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2309.05798v1)  

---


**ABSTRACT**  
Hypergraphs can naturally model group-wise relations (e.g., a group of users who co-purchase an item) as hyperedges. Hyperedge prediction is to predict future or unobserved hyperedges, which is a fundamental task in many real-world applications (e.g., group recommendation). Despite the recent breakthrough of hyperedge prediction methods, the following challenges have been rarely studied: (C1) How to aggregate the nodes in each hyperedge candidate for accurate hyperedge prediction? and (C2) How to mitigate the inherent data sparsity problem in hyperedge prediction? To tackle both challenges together, in this paper, we propose a novel hyperedge prediction framework (CASH) that employs (1) context-aware node aggregation to precisely capture complex relations among nodes in each hyperedge for (C1) and (2) self-supervised contrastive learning in the context of hyperedge prediction to enhance hypergraph representations for (C2). Furthermore, as for (C2), we propose a hyperedge-aware augmentation method to fully exploit the latent semantics behind the original hypergraph and consider both node-level and group-level contrasts (i.e., dual contrasts) for better node and hyperedge representations. Extensive experiments on six real-world hypergraphs reveal that CASH consistently outperforms all competing methods in terms of the accuracy in hyperedge prediction and each of the proposed strategies is effective in improving the model accuracy of CASH. For the detailed information of CASH, we provide the code and datasets at: https://github.com/yy-ko/cash.

{{</citation>}}


### (11/97) Hypothesis Search: Inductive Reasoning with Language Models (Ruocheng Wang et al., 2023)

{{<citation>}}

Ruocheng Wang, Eric Zelikman, Gabriel Poesia, Yewen Pu, Nick Haber, Noah D. Goodman. (2023)  
**Hypothesis Search: Inductive Reasoning with Language Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2309.05660v1)  

---


**ABSTRACT**  
Inductive reasoning is a core problem-solving capacity: humans can identify underlying principles from a few examples, which can then be robustly generalized to novel scenarios. Recent work has evaluated large language models (LLMs) on inductive reasoning tasks by directly prompting them yielding "in context learning." This can work well for straightforward inductive tasks, but performs very poorly on more complex tasks such as the Abstraction and Reasoning Corpus (ARC). In this work, we propose to improve the inductive reasoning ability of LLMs by generating explicit hypotheses at multiple levels of abstraction: we prompt the LLM to propose multiple abstract hypotheses about the problem, in natural language, then implement the natural language hypotheses as concrete Python programs. These programs can be directly verified by running on the observed examples and generalized to novel inputs. Because of the prohibitive cost of generation with state-of-the-art LLMs, we consider a middle step to filter the set of hypotheses that will be implemented into programs: we either ask the LLM to summarize into a smaller set of hypotheses, or ask human annotators to select a subset of the hypotheses. We verify our pipeline's effectiveness on the ARC visual inductive reasoning benchmark, its variant 1D-ARC, and string transformation dataset SyGuS. On a random 40-problem subset of ARC, our automated pipeline using LLM summaries achieves 27.5% accuracy, significantly outperforming the direct prompting baseline (accuracy of 12.5%). With the minimal human input of selecting from LLM-generated candidates, the performance is boosted to 37.5%. (And we argue this is a lower bound on the performance of our approach without filtering.) Our ablation studies show that abstract hypothesis generation and concrete program representations are both beneficial for LLMs to perform inductive reasoning tasks.

{{</citation>}}


### (12/97) Mind the Uncertainty: Risk-Aware and Actively Exploring Model-Based Reinforcement Learning (Marin Vlastelica et al., 2023)

{{<citation>}}

Marin Vlastelica, Sebastian Blaes, Cristina Pineri, Georg Martius. (2023)  
**Mind the Uncertainty: Risk-Aware and Actively Exploring Model-Based Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-RO, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.05582v1)  

---


**ABSTRACT**  
We introduce a simple but effective method for managing risk in model-based reinforcement learning with trajectory sampling that involves probabilistic safety constraints and balancing of optimism in the face of epistemic uncertainty and pessimism in the face of aleatoric uncertainty of an ensemble of stochastic neural networks.Various experiments indicate that the separation of uncertainties is essential to performing well with data-driven MPC approaches in uncertain and safety-critical control environments.

{{</citation>}}


### (13/97) Learning Objective-Specific Active Learning Strategies with Attentive Neural Processes (Tim Bakker et al., 2023)

{{<citation>}}

Tim Bakker, Herke van Hoof, Max Welling. (2023)  
**Learning Objective-Specific Active Learning Strategies with Attentive Neural Processes**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2309.05477v1)  

---


**ABSTRACT**  
Pool-based active learning (AL) is a promising technology for increasing data-efficiency of machine learning models. However, surveys show that performance of recent AL methods is very sensitive to the choice of dataset and training setting, making them unsuitable for general application. In order to tackle this problem, the field Learning Active Learning (LAL) suggests to learn the active learning strategy itself, allowing it to adapt to the given setting. In this work, we propose a novel LAL method for classification that exploits symmetry and independence properties of the active learning problem with an Attentive Conditional Neural Process model. Our approach is based on learning from a myopic oracle, which gives our model the ability to adapt to non-standard objectives, such as those that do not equally weight the error on all data points. We experimentally verify that our Neural Process model outperforms a variety of baselines in these settings. Finally, our experiments show that our model exhibits a tendency towards improved stability to changing datasets. However, performance is sensitive to choice of classifier and more work is necessary to reduce the performance the gap with the myopic oracle and to improve scalability. We present our work as a proof-of-concept for LAL on nonstandard objectives and hope our analysis and modelling considerations inspire future LAL work.

{{</citation>}}


### (14/97) A parameterised model for link prediction using node centrality and similarity measure based on graph embedding (Haohui Lu et al., 2023)

{{<citation>}}

Haohui Lu, Shahadat Uddin. (2023)  
**A parameterised model for link prediction using node centrality and similarity measure based on graph embedding**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SI, cs.LG  
Keywords: GNN, Graph Convolutional Network, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2309.05434v1)  

---


**ABSTRACT**  
Link prediction is a key aspect of graph machine learning, with applications as diverse as disease prediction, social network recommendations, and drug discovery. It involves predicting new links that may form between network nodes. Despite the clear importance of link prediction, existing models have significant shortcomings. Graph Convolutional Networks, for instance, have been proven to be highly efficient for link prediction on a variety of datasets. However, they encounter severe limitations when applied to short-path networks and ego networks, resulting in poor performance. This presents a critical problem space that this work aims to address. In this paper, we present the Node Centrality and Similarity Based Parameterised Model (NCSM), a novel method for link prediction tasks. NCSM uniquely integrates node centrality and similarity measures as edge features in a customised Graph Neural Network (GNN) layer, effectively leveraging the topological information of large networks. This model represents the first parameterised GNN-based link prediction model that considers topological information. The proposed model was evaluated on five benchmark graph datasets, each comprising thousands of nodes and edges. Experimental results highlight NCSM's superiority over existing state-of-the-art models like Graph Convolutional Networks and Variational Graph Autoencoder, as it outperforms them across various metrics and datasets. This exceptional performance can be attributed to NCSM's innovative integration of node centrality, similarity measures, and its efficient use of topological information.

{{</citation>}}


### (15/97) Career Path Recommendations for Long-term Income Maximization: A Reinforcement Learning Approach (Spyros Avlonitis et al., 2023)

{{<citation>}}

Spyros Avlonitis, Dor Lavi, Masoud Mansoury, David Graus. (2023)  
**Career Path Recommendations for Long-term Income Maximization: A Reinforcement Learning Approach**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.05391v1)  

---


**ABSTRACT**  
This study explores the potential of reinforcement learning algorithms to enhance career planning processes. Leveraging data from Randstad The Netherlands, the study simulates the Dutch job market and develops strategies to optimize employees' long-term income. By formulating career planning as a Markov Decision Process (MDP) and utilizing machine learning algorithms such as Sarsa, Q-Learning, and A2C, we learn optimal policies that recommend career paths with high-income occupations and industries. The results demonstrate significant improvements in employees' income trajectories, with RL models, particularly Q-Learning and Sarsa, achieving an average increase of 5% compared to observed career paths. The study acknowledges limitations, including narrow job filtering, simplifications in the environment formulation, and assumptions regarding employment continuity and zero application costs. Future research can explore additional objectives beyond income optimization and address these limitations to further enhance career planning processes.

{{</citation>}}


### (16/97) Fully-Connected Spatial-Temporal Graph for Multivariate Time Series Data (Yucheng Wang et al., 2023)

{{<citation>}}

Yucheng Wang, Yuecong Xu, Jianfei Yang, Min Wu, Xiaoli Li, Lihua Xie, Zhenghua Chen. (2023)  
**Fully-Connected Spatial-Temporal Graph for Multivariate Time Series Data**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Time Series  
[Paper Link](http://arxiv.org/abs/2309.05305v1)  

---


**ABSTRACT**  
Multivariate Time-Series (MTS) data is crucial in various application fields. With its sequential and multi-source (multiple sensors) properties, MTS data inherently exhibits Spatial-Temporal (ST) dependencies, involving temporal correlations between timestamps and spatial correlations between sensors in each timestamp. To effectively leverage this information, Graph Neural Network-based methods (GNNs) have been widely adopted. However, existing approaches separately capture spatial dependency and temporal dependency and fail to capture the correlations between Different sEnsors at Different Timestamps (DEDT). Overlooking such correlations hinders the comprehensive modelling of ST dependencies within MTS data, thus restricting existing GNNs from learning effective representations. To address this limitation, we propose a novel method called Fully-Connected Spatial-Temporal Graph Neural Network (FC-STGNN), including two key components namely FC graph construction and FC graph convolution. For graph construction, we design a decay graph to connect sensors across all timestamps based on their temporal distances, enabling us to fully model the ST dependencies by considering the correlations between DEDT. Further, we devise FC graph convolution with a moving-pooling GNN layer to effectively capture the ST dependencies for learning effective representations. Extensive experiments show the effectiveness of FC-STGNN on multiple MTS datasets compared to SOTA methods.

{{</citation>}}


### (17/97) EANet: Expert Attention Network for Online Trajectory Prediction (Pengfei Yao et al., 2023)

{{<citation>}}

Pengfei Yao, Tianlu Mao, Min Shi, Jingkai Sun, Zhaoqi Wang. (2023)  
**EANet: Expert Attention Network for Online Trajectory Prediction**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-RO, cs.LG  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2309.05683v1)  

---


**ABSTRACT**  
Trajectory prediction plays a crucial role in autonomous driving. Existing mainstream research and continuoual learning-based methods all require training on complete datasets, leading to poor prediction accuracy when sudden changes in scenarios occur and failing to promptly respond and update the model. Whether these methods can make a prediction in real-time and use data instances to update the model immediately(i.e., online learning settings) remains a question. The problem of gradient explosion or vanishing caused by data instance streams also needs to be addressed. Inspired by Hedge Propagation algorithm, we propose Expert Attention Network, a complete online learning framework for trajectory prediction. We introduce expert attention, which adjusts the weights of different depths of network layers, avoiding the model updated slowly due to gradient problem and enabling fast learning of new scenario's knowledge to restore prediction accuracy. Furthermore, we propose a short-term motion trend kernel function which is sensitive to scenario change, allowing the model to respond quickly. To the best of our knowledge, this work is the first attempt to address the online learning problem in trajectory prediction. The experimental results indicate that traditional methods suffer from gradient problems and that our method can quickly reduce prediction errors and reach the state-of-the-art prediction accuracy.

{{</citation>}}


### (18/97) Examining the Effect of Pre-training on Time Series Classification (Jiashu Pu et al., 2023)

{{<citation>}}

Jiashu Pu, Shiwei Zhao, Ling Cheng, Yongzhu Chang, Runze Wu, Tangjie Lv, Rongsheng Zhang. (2023)  
**Examining the Effect of Pre-training on Time Series Classification**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2309.05256v1)  

---


**ABSTRACT**  
Although the pre-training followed by fine-tuning paradigm is used extensively in many fields, there is still some controversy surrounding the impact of pre-training on the fine-tuning process. Currently, experimental findings based on text and image data lack consensus. To delve deeper into the unsupervised pre-training followed by fine-tuning paradigm, we have extended previous research to a new modality: time series. In this study, we conducted a thorough examination of 150 classification datasets derived from the Univariate Time Series (UTS) and Multivariate Time Series (MTS) benchmarks. Our analysis reveals several key conclusions. (i) Pre-training can only help improve the optimization process for models that fit the data poorly, rather than those that fit the data well. (ii) Pre-training does not exhibit the effect of regularization when given sufficient training time. (iii) Pre-training can only speed up convergence if the model has sufficient ability to fit the data. (iv) Adding more pre-training data does not improve generalization, but it can strengthen the advantage of pre-training on the original data volume, such as faster convergence. (v) While both the pre-training task and the model structure determine the effectiveness of the paradigm on a given dataset, the model structure plays a more significant role.

{{</citation>}}


### (19/97) Graph Contextual Contrasting for Multivariate Time Series Classification (Yucheng Wang et al., 2023)

{{<citation>}}

Yucheng Wang, Yuecong Xu, Jianfei Yang, Min Wu, Xiaoli Li, Lihua Xie, Zhenghua Chen. (2023)  
**Graph Contextual Contrasting for Multivariate Time Series Classification**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2309.05202v1)  

---


**ABSTRACT**  
Contrastive learning, as a self-supervised learning paradigm, becomes popular for Multivariate Time-Series (MTS) classification. It ensures the consistency across different views of unlabeled samples and then learns effective representations for these samples. Existing contrastive learning methods mainly focus on achieving temporal consistency with temporal augmentation and contrasting techniques, aiming to preserve temporal patterns against perturbations for MTS data. However, they overlook spatial consistency that requires the stability of individual sensors and their correlations. As MTS data typically originate from multiple sensors, ensuring spatial consistency becomes essential for the overall performance of contrastive learning on MTS data. Thus, we propose Graph Contextual Contrasting (GCC) for spatial consistency across MTS data. Specifically, we propose graph augmentations including node and edge augmentations to preserve the stability of sensors and their correlations, followed by graph contrasting with both node- and graph-level contrasting to extract robust sensor- and global-level features. We further introduce multi-window temporal contrasting to ensure temporal consistency in the data for each sensor. Extensive experiments demonstrate that our proposed GCC achieves state-of-the-art performance on various MTS classification tasks.

{{</citation>}}


## cs.PL (1)



### (20/97) Large Language Models for Compiler Optimization (Chris Cummins et al., 2023)

{{<citation>}}

Chris Cummins, Volker Seeker, Dejan Grubisic, Mostafa Elhoushi, Youwei Liang, Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Kim Hazelwood, Gabriel Synnaeve, Hugh Leather. (2023)  
**Large Language Models for Compiler Optimization**  

---
Primary Category: cs.PL  
Categories: cs-AI, cs-CL, cs-LG, cs-PL, cs.PL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.07062v1)  

---


**ABSTRACT**  
We explore the novel application of Large Language Models to code optimization. We present a 7B-parameter transformer model trained from scratch to optimize LLVM assembly for code size. The model takes as input unoptimized assembly and outputs a list of compiler options to best optimize the program. Crucially, during training, we ask the model to predict the instruction counts before and after optimization, and the optimized code itself. These auxiliary learning tasks significantly improve the optimization performance of the model and improve the model's depth of understanding.   We evaluate on a large suite of test programs. Our approach achieves a 3.0% improvement in reducing instruction counts over the compiler, outperforming two state-of-the-art baselines that require thousands of compilations. Furthermore, the model shows surprisingly strong code reasoning abilities, generating compilable code 91% of the time and perfectly emulating the output of the compiler 70% of the time.

{{</citation>}}


## cs.CV (18)



### (21/97) Self-Correlation and Cross-Correlation Learning for Few-Shot Remote Sensing Image Semantic Segmentation (Linhan Wang et al., 2023)

{{<citation>}}

Linhan Wang, Shuo Lei, Jianfeng He, Shengkun Wang, Min Zhang, Chang-Tien Lu. (2023)  
**Self-Correlation and Cross-Correlation Learning for Few-Shot Remote Sensing Image Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Few-Shot, Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2309.05840v1)  

---


**ABSTRACT**  
Remote sensing image semantic segmentation is an important problem for remote sensing image interpretation. Although remarkable progress has been achieved, existing deep neural network methods suffer from the reliance on massive training data. Few-shot remote sensing semantic segmentation aims at learning to segment target objects from a query image using only a few annotated support images of the target class. Most existing few-shot learning methods stem primarily from their sole focus on extracting information from support images, thereby failing to effectively address the large variance in appearance and scales of geographic objects. To tackle these challenges, we propose a Self-Correlation and Cross-Correlation Learning Network for the few-shot remote sensing image semantic segmentation. Our model enhances the generalization by considering both self-correlation and cross-correlation between support and query images to make segmentation predictions. To further explore the self-correlation with the query image, we propose to adopt a classical spectral method to produce a class-agnostic segmentation mask based on the basic visual information of the image. Extensive experiments on two remote sensing image datasets demonstrate the effectiveness and superiority of our model in few-shot remote sensing image semantic segmentation. Code and models will be accessed at https://github.com/linhanwang/SCCNe.

{{</citation>}}


### (22/97) Mobile Vision Transformer-based Visual Object Tracking (Goutam Yelluru Gopal et al., 2023)

{{<citation>}}

Goutam Yelluru Gopal, Maria A. Amer. (2023)  
**Mobile Vision Transformer-based Visual Object Tracking**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.05829v1)  

---


**ABSTRACT**  
The introduction of robust backbones, such as Vision Transformers, has improved the performance of object tracking algorithms in recent years. However, these state-of-the-art trackers are computationally expensive since they have a large number of model parameters and rely on specialized hardware (e.g., GPU) for faster inference. On the other hand, recent lightweight trackers are fast but are less accurate, especially on large-scale datasets. We propose a lightweight, accurate, and fast tracking algorithm using Mobile Vision Transformers (MobileViT) as the backbone for the first time. We also present a novel approach of fusing the template and search region representations in the MobileViT backbone, thereby generating superior feature encoding for target localization. The experimental results show that our MobileViT-based Tracker, MVT, surpasses the performance of recent lightweight trackers on the large-scale datasets GOT10k and TrackingNet, and with a high inference speed. In addition, our method outperforms the popular DiMP-50 tracker despite having 4.7 times fewer model parameters and running at 2.8 times its speed on a GPU. The tracker code and models are available at https://github.com/goutamyg/MVT

{{</citation>}}


### (23/97) TransferDoc: A Self-Supervised Transferable Document Representation Learning Model Unifying Vision and Language (Souhail Bakkali et al., 2023)

{{<citation>}}

Souhail Bakkali, Sanket Biswas, Zuheng Ming, Mickael Coustaty, Marçal Rusiñol, Oriol Ramos Terrades, Josep Lladós. (2023)  
**TransferDoc: A Self-Supervised Transferable Document Representation Learning Model Unifying Vision and Language**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: OCR, Representation Learning, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2309.05756v1)  

---


**ABSTRACT**  
The field of visual document understanding has witnessed a rapid growth in emerging challenges and powerful multi-modal strategies. However, they rely on an extensive amount of document data to learn their pretext objectives in a ``pre-train-then-fine-tune'' paradigm and thus, suffer a significant performance drop in real-world online industrial settings. One major reason is the over-reliance on OCR engines to extract local positional information within a document page. Therefore, this hinders the model's generalizability, flexibility and robustness due to the lack of capturing global information within a document image. We introduce TransferDoc, a cross-modal transformer-based architecture pre-trained in a self-supervised fashion using three novel pretext objectives. TransferDoc learns richer semantic concepts by unifying language and visual representations, which enables the production of more transferable models. Besides, two novel downstream tasks have been introduced for a ``closer-to-real'' industrial evaluation scenario where TransferDoc outperforms other state-of-the-art approaches.

{{</citation>}}


### (24/97) Learning the Geodesic Embedding with Graph Neural Networks (Bo Pang et al., 2023)

{{<citation>}}

Bo Pang, Zhongtian Zheng, Guoping Wang, Peng-Shuai Wang. (2023)  
**Learning the Geodesic Embedding with Graph Neural Networks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-GR, cs.CV  
Keywords: Embedding, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2309.05613v1)  

---


**ABSTRACT**  
We present GeGnn, a learning-based method for computing the approximate geodesic distance between two arbitrary points on discrete polyhedra surfaces with constant time complexity after fast precomputation. Previous relevant methods either focus on computing the geodesic distance between a single source and all destinations, which has linear complexity at least or require a long precomputation time. Our key idea is to train a graph neural network to embed an input mesh into a high-dimensional embedding space and compute the geodesic distance between a pair of points using the corresponding embedding vectors and a lightweight decoding function. To facilitate the learning of the embedding, we propose novel graph convolution and graph pooling modules that incorporate local geodesic information and are verified to be much more effective than previous designs. After training, our method requires only one forward pass of the network per mesh as precomputation. Then, we can compute the geodesic distance between a pair of points using our decoding function, which requires only several matrix multiplications and can be massively parallelized on GPUs. We verify the efficiency and effectiveness of our method on ShapeNet and demonstrate that our method is faster than existing methods by orders of magnitude while achieving comparable or better accuracy. Additionally, our method exhibits robustness on noisy and incomplete meshes and strong generalization ability on out-of-distribution meshes. The code and pretrained model can be found on https://github.com/IntelligentGeometry/GeGnn.

{{</citation>}}


### (25/97) OpenFashionCLIP: Vision-and-Language Contrastive Learning with Open-Source Fashion Data (Giuseppe Cartella et al., 2023)

{{<citation>}}

Giuseppe Cartella, Alberto Baldrati, Davide Morelli, Marcella Cornia, Marco Bertini, Rita Cucchiara. (2023)  
**OpenFashionCLIP: Vision-and-Language Contrastive Learning with Open-Source Fashion Data**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2309.05551v1)  

---


**ABSTRACT**  
The inexorable growth of online shopping and e-commerce demands scalable and robust machine learning-based solutions to accommodate customer requirements. In the context of automatic tagging classification and multimodal retrieval, prior works either defined a low generalizable supervised learning approach or more reusable CLIP-based techniques while, however, training on closed source data. In this work, we propose OpenFashionCLIP, a vision-and-language contrastive learning method that only adopts open-source fashion data stemming from diverse domains, and characterized by varying degrees of specificity. Our approach is extensively validated across several tasks and benchmarks, and experimental results highlight a significant out-of-domain generalization capability and consistent improvements over state-of-the-art methods both in terms of accuracy and recall. Source code and trained models are publicly available at: https://github.com/aimagelab/open-fashion-clip.

{{</citation>}}


### (26/97) ReSimAD: Zero-Shot 3D Domain Transfer for Autonomous Driving with Source Reconstruction and Target Simulation (Bo Zhang et al., 2023)

{{<citation>}}

Bo Zhang, Xinyu Cai, Jiakang Yuan, Donglin Yang, Jianfei Guo, Renqiu Xia, Botian Shi, Min Dou, Tao Chen, Si Liu, Junchi Yan, Yu Qiao. (2023)  
**ReSimAD: Zero-Shot 3D Domain Transfer for Autonomous Driving with Source Reconstruction and Target Simulation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2309.05527v1)  

---


**ABSTRACT**  
Domain shifts such as sensor type changes and geographical situation variations are prevalent in Autonomous Driving (AD), which poses a challenge since AD model relying on the previous-domain knowledge can be hardly directly deployed to a new domain without additional costs. In this paper, we provide a new perspective and approach of alleviating the domain shifts, by proposing a Reconstruction-Simulation-Perception (ReSimAD) scheme. Specifically, the implicit reconstruction process is based on the knowledge from the previous old domain, aiming to convert the domain-related knowledge into domain-invariant representations, \textit{e.g.}, 3D scene-level meshes. Besides, the point clouds simulation process of multiple new domains is conditioned on the above reconstructed 3D meshes, where the target-domain-like simulation samples can be obtained, thus reducing the cost of collecting and annotating new-domain data for the subsequent perception process. For experiments, we consider different cross-domain situations such as Waymo-to-KITTI, Waymo-to-nuScenes, Waymo-to-ONCE, \textit{etc}, to verify the \textbf{zero-shot} target-domain perception using ReSimAD. Results demonstrate that our method is beneficial to boost the domain generalization ability, even promising for 3D pre-training.

{{</citation>}}


### (27/97) Stream-based Active Learning by Exploiting Temporal Properties in Perception with Temporal Predicted Loss (Sebastian Schmidt et al., 2023)

{{<citation>}}

Sebastian Schmidt, Stephan Günnemann. (2023)  
**Stream-based Active Learning by Exploiting Temporal Properties in Perception with Temporal Predicted Loss**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2309.05517v1)  

---


**ABSTRACT**  
Active learning (AL) reduces the amount of labeled data needed to train a machine learning model by intelligently choosing which instances to label. Classic pool-based AL requires all data to be present in a datacenter, which can be challenging with the increasing amounts of data needed in deep learning. However, AL on mobile devices and robots, like autonomous cars, can filter the data from perception sensor streams before reaching the datacenter. We exploited the temporal properties for such image streams in our work and proposed the novel temporal predicted loss (TPL) method. To evaluate the stream-based setting properly, we introduced the GTA V streets and the A2D2 streets dataset and made both publicly available. Our experiments showed that our approach significantly improves the diversity of the selection while being an uncertainty-based method. As pool-based approaches are more common in perception applications, we derived a concept for comparing pool-based and stream-based AL, where TPL out-performed state-of-the-art pool- or stream-based approaches for different models. TPL demonstrated a gain of 2.5 precept points (pp) less required data while being significantly faster than pool-based methods.

{{</citation>}}


### (28/97) Zero-Shot Co-salient Object Detection Framework (Haoke Xiao et al., 2023)

{{<citation>}}

Haoke Xiao, Lv Tang, Bo Li, Zhiming Luo, Shaozi Li. (2023)  
**Zero-Shot Co-salient Object Detection Framework**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2309.05499v1)  

---


**ABSTRACT**  
Co-salient Object Detection (CoSOD) endeavors to replicate the human visual system's capacity to recognize common and salient objects within a collection of images. Despite recent advancements in deep learning models, these models still rely on training with well-annotated CoSOD datasets. The exploration of training-free zero-shot CoSOD frameworks has been limited. In this paper, taking inspiration from the zero-shot transfer capabilities of foundational computer vision models, we introduce the first zero-shot CoSOD framework that harnesses these models without any training process. To achieve this, we introduce two novel components in our proposed framework: the group prompt generation (GPG) module and the co-saliency map generation (CMP) module. We evaluate the framework's performance on widely-used datasets and observe impressive results. Our approach surpasses existing unsupervised methods and even outperforms fully supervised methods developed before 2020, while remaining competitive with some fully supervised methods developed before 2022.

{{</citation>}}


### (29/97) Learning Semantic Segmentation with Query Points Supervision on Aerial Images (Santiago Rivier et al., 2023)

{{<citation>}}

Santiago Rivier, Carlos Hinojosa, Silvio Giancola, Bernard Ghanem. (2023)  
**Learning Semantic Segmentation with Query Points Supervision on Aerial Images**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2309.05490v1)  

---


**ABSTRACT**  
Semantic segmentation is crucial in remote sensing, where high-resolution satellite images are segmented into meaningful regions. Recent advancements in deep learning have significantly improved satellite image segmentation. However, most of these methods are typically trained in fully supervised settings that require high-quality pixel-level annotations, which are expensive and time-consuming to obtain. In this work, we present a weakly supervised learning algorithm to train semantic segmentation algorithms that only rely on query point annotations instead of full mask labels. Our proposed approach performs accurate semantic segmentation and improves efficiency by significantly reducing the cost and time required for manual annotation. Specifically, we generate superpixels and extend the query point labels into those superpixels that group similar meaningful semantics. Then, we train semantic segmentation models, supervised with images partially labeled with the superpixels pseudo-labels. We benchmark our weakly supervised training approach on an aerial image dataset and different semantic segmentation architectures, showing that we can reach competitive performance compared to fully supervised training while reducing the annotation effort.

{{</citation>}}


### (30/97) Dual-view Curricular Optimal Transport for Cross-lingual Cross-modal Retrieval (Yabing Wang et al., 2023)

{{<citation>}}

Yabing Wang, Shuhui Wang, Hao Luo, Jianfeng Dong, Fan Wang, Meng Han, Xun Wang, Meng Wang. (2023)  
**Dual-view Curricular Optimal Transport for Cross-lingual Cross-modal Retrieval**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV  
Keywords: Machine Translation  
[Paper Link](http://arxiv.org/abs/2309.05451v1)  

---


**ABSTRACT**  
Current research on cross-modal retrieval is mostly English-oriented, as the availability of a large number of English-oriented human-labeled vision-language corpora. In order to break the limit of non-English labeled data, cross-lingual cross-modal retrieval (CCR) has attracted increasing attention. Most CCR methods construct pseudo-parallel vision-language corpora via Machine Translation (MT) to achieve cross-lingual transfer. However, the translated sentences from MT are generally imperfect in describing the corresponding visual contents. Improperly assuming the pseudo-parallel data are correctly correlated will make the networks overfit to the noisy correspondence. Therefore, we propose Dual-view Curricular Optimal Transport (DCOT) to learn with noisy correspondence in CCR. In particular, we quantify the confidence of the sample pair correlation with optimal transport theory from both the cross-lingual and cross-modal views, and design dual-view curriculum learning to dynamically model the transportation costs according to the learning stage of the two views. Extensive experiments are conducted on two multilingual image-text datasets and one video-text dataset, and the results demonstrate the effectiveness and robustness of the proposed method. Besides, our proposed method also shows a good expansibility to cross-lingual image-text baselines and a decent generalization on out-of-domain data.

{{</citation>}}


### (31/97) CNN or ViT? Revisiting Vision Transformers Through the Lens of Convolution (Chenghao Li et al., 2023)

{{<citation>}}

Chenghao Li, Chaoning Zhang. (2023)  
**CNN or ViT? Revisiting Vision Transformers Through the Lens of Convolution**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.05375v1)  

---


**ABSTRACT**  
The success of Vision Transformer (ViT) has been widely reported on a wide range of image recognition tasks. The merit of ViT over CNN has been largely attributed to large training datasets or auxiliary pre-training. Without pre-training, the performance of ViT on small datasets is limited because the global self-attention has limited capacity in local modeling. Towards boosting ViT on small datasets without pre-training, this work improves its local modeling by applying a weight mask on the original self-attention matrix. A straightforward way to locally adapt the self-attention matrix can be realized by an element-wise learnable weight mask (ELM), for which our preliminary results show promising results. However, the element-wise simple learnable weight mask not only induces a non-trivial additional parameter overhead but also increases the optimization complexity. To this end, this work proposes a novel Gaussian mixture mask (GMM) in which one mask only has two learnable parameters and it can be conveniently used in any ViT variants whose attention mechanism allows the use of masks. Experimental results on multiple small datasets demonstrate that the effectiveness of our proposed Gaussian mask for boosting ViTs for free (almost zero additional parameter or computation cost). Our code will be publicly available at \href{https://github.com/CatworldLee/Gaussian-Mixture-Mask-Attention}{https://github.com/CatworldLee/Gaussian-Mixture-Mask-Attention}.

{{</citation>}}


### (32/97) Diff-Privacy: Diffusion-based Face Privacy Protection (Xiao He et al., 2023)

{{<citation>}}

Xiao He, Mingrui Zhu, Dongxin Chen, Nannan Wang, Xinbo Gao. (2023)  
**Diff-Privacy: Diffusion-based Face Privacy Protection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.05330v1)  

---


**ABSTRACT**  
Privacy protection has become a top priority as the proliferation of AI techniques has led to widespread collection and misuse of personal data. Anonymization and visual identity information hiding are two important facial privacy protection tasks that aim to remove identification characteristics from facial images at the human perception level. However, they have a significant difference in that the former aims to prevent the machine from recognizing correctly, while the latter needs to ensure the accuracy of machine recognition. Therefore, it is difficult to train a model to complete these two tasks simultaneously. In this paper, we unify the task of anonymization and visual identity information hiding and propose a novel face privacy protection method based on diffusion models, dubbed Diff-Privacy. Specifically, we train our proposed multi-scale image inversion module (MSI) to obtain a set of SDM format conditional embeddings of the original image. Based on the conditional embeddings, we design corresponding embedding scheduling strategies and construct different energy functions during the denoising process to achieve anonymization and visual identity information hiding. Extensive experiments have been conducted to validate the effectiveness of our proposed framework in protecting facial privacy.

{{</citation>}}


### (33/97) FusionFormer: A Multi-sensory Fusion in Bird's-Eye-View and Temporal Consistent Transformer for 3D Objection (Chunyong Hu et al., 2023)

{{<citation>}}

Chunyong Hu, Hang Zheng, Kun Li, Jianyun Xu, Weibo Mao, Maochun Luo, Lingxuan Wang, Mingxia Chen, Kaixuan Liu, Yiru Zhao, Peihan Hao, Minzhe Liu, Kaicheng Yu. (2023)  
**FusionFormer: A Multi-sensory Fusion in Bird's-Eye-View and Temporal Consistent Transformer for 3D Objection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.05257v1)  

---


**ABSTRACT**  
Multi-sensor modal fusion has demonstrated strong advantages in 3D object detection tasks. However, existing methods that fuse multi-modal features through a simple channel concatenation require transformation features into bird's eye view space and may lose the information on Z-axis thus leads to inferior performance. To this end, we propose FusionFormer, an end-to-end multi-modal fusion framework that leverages transformers to fuse multi-modal features and obtain fused BEV features. And based on the flexible adaptability of FusionFormer to the input modality representation, we propose a depth prediction branch that can be added to the framework to improve detection performance in camera-based detection tasks. In addition, we propose a plug-and-play temporal fusion module based on transformers that can fuse historical frame BEV features for more stable and reliable detection results. We evaluate our method on the nuScenes dataset and achieve 72.6% mAP and 75.1% NDS for 3D object detection tasks, outperforming state-of-the-art methods.

{{</citation>}}


### (34/97) Towards Better Data Exploitation In Self-Supervised Monocular Depth Estimation (Jinfeng Liu et al., 2023)

{{<citation>}}

Jinfeng Liu, Lingtong Kong, Jie Yang, Wei Liu. (2023)  
**Towards Better Data Exploitation In Self-Supervised Monocular Depth Estimation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2309.05254v1)  

---


**ABSTRACT**  
Depth estimation plays an important role in the robotic perception system. Self-supervised monocular paradigm has gained significant attention since it can free training from the reliance on depth annotations. Despite recent advancements, existing self-supervised methods still underutilize the available training data, limiting their generalization ability. In this paper, we take two data augmentation techniques, namely Resizing-Cropping and Splitting-Permuting, to fully exploit the potential of training datasets. Specifically, the original image and the generated two augmented images are fed into the training pipeline simultaneously and we leverage them to conduct self-distillation. Additionally, we introduce the detail-enhanced DepthNet with an extra full-scale branch in the encoder and a grid decoder to enhance the restoration of fine details in depth maps. Experimental results demonstrate our method can achieve state-of-the-art performance on the KITTI benchmark, with both raw ground truth and improved ground truth. Moreover, our models also show superior generalization performance when transferring to Make3D and NYUv2 datasets. Our codes are available at https://github.com/Sauf4896/BDEdepth.

{{</citation>}}


### (35/97) HAT: Hybrid Attention Transformer for Image Restoration (Xiangyu Chen et al., 2023)

{{<citation>}}

Xiangyu Chen, Xintao Wang, Wenlong Zhang, Xiangtao Kong, Yu Qiao, Jiantao Zhou, Chao Dong. (2023)  
**HAT: Hybrid Attention Transformer for Image Restoration**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2309.05239v1)  

---


**ABSTRACT**  
Transformer-based methods have shown impressive performance in image restoration tasks, such as image super-resolution and denoising. However, we find that these networks can only utilize a limited spatial range of input information through attribution analysis. This implies that the potential of Transformer is still not fully exploited in existing networks. In order to activate more input pixels for better restoration, we propose a new Hybrid Attention Transformer (HAT). It combines both channel attention and window-based self-attention schemes, thus making use of their complementary advantages. Moreover, to better aggregate the cross-window information, we introduce an overlapping cross-attention module to enhance the interaction between neighboring window features. In the training stage, we additionally adopt a same-task pre-training strategy to further exploit the potential of the model for further improvement. Extensive experiments have demonstrated the effectiveness of the proposed modules. We further scale up the model to show that the performance of the SR task can be greatly improved. Besides, we extend HAT to more image restoration applications, including real-world image super-resolution, Gaussian image denoising and image compression artifacts reduction. Experiments on benchmark and real-world datasets demonstrate that our HAT achieves state-of-the-art performance both quantitatively and qualitatively. Codes and models are publicly available at https://github.com/XPixelGroup/HAT.

{{</citation>}}


### (36/97) SparseSwin: Swin Transformer with Sparse Transformer Block (Krisna Pinasthika et al., 2023)

{{<citation>}}

Krisna Pinasthika, Blessius Sheldo Putra Laksono, Riyandi Banovbi Putera Irsal, Syifa Hukma Shabiyya, Novanto Yudistira. (2023)  
**SparseSwin: Swin Transformer with Sparse Transformer Block**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: ImageNet, Transformer  
[Paper Link](http://arxiv.org/abs/2309.05224v1)  

---


**ABSTRACT**  
Advancements in computer vision research have put transformer architecture as the state of the art in computer vision tasks. One of the known drawbacks of the transformer architecture is the high number of parameters, this can lead to a more complex and inefficient algorithm. This paper aims to reduce the number of parameters and in turn, made the transformer more efficient. We present Sparse Transformer (SparTa) Block, a modified transformer block with an addition of a sparse token converter that reduces the number of tokens used. We use the SparTa Block inside the Swin T architecture (SparseSwin) to leverage Swin capability to downsample its input and reduce the number of initial tokens to be calculated. The proposed SparseSwin model outperforms other state of the art models in image classification with an accuracy of 86.96%, 97.43%, and 85.35% on the ImageNet100, CIFAR10, and CIFAR100 datasets respectively. Despite its fewer parameters, the result highlights the potential of a transformer architecture using a sparse token converter with a limited number of tokens to optimize the use of the transformer and improve its performance.

{{</citation>}}


### (37/97) Phase-Specific Augmented Reality Guidance for Microscopic Cataract Surgery Using Long-Short Spatiotemporal Aggregation Transformer (Puxun Tu et al., 2023)

{{<citation>}}

Puxun Tu, Hongfei Ye, Haochen Shi, Jeff Young, Meng Xie, Peiquan Zhao, Ce Zheng, Xiaoyi Jiang, Xiaojun Chen. (2023)  
**Phase-Specific Augmented Reality Guidance for Microscopic Cataract Surgery Using Long-Short Spatiotemporal Aggregation Transformer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.05209v1)  

---


**ABSTRACT**  
Phacoemulsification cataract surgery (PCS) is a routine procedure conducted using a surgical microscope, heavily reliant on the skill of the ophthalmologist. While existing PCS guidance systems extract valuable information from surgical microscopic videos to enhance intraoperative proficiency, they suffer from non-phasespecific guidance, leading to redundant visual information. In this study, our major contribution is the development of a novel phase-specific augmented reality (AR) guidance system, which offers tailored AR information corresponding to the recognized surgical phase. Leveraging the inherent quasi-standardized nature of PCS procedures, we propose a two-stage surgical microscopic video recognition network. In the first stage, we implement a multi-task learning structure to segment the surgical limbus region and extract limbus region-focused spatial feature for each frame. In the second stage, we propose the long-short spatiotemporal aggregation transformer (LS-SAT) network to model local fine-grained and global temporal relationships, and combine the extracted spatial features to recognize the current surgical phase. Additionally, we collaborate closely with ophthalmologists to design AR visual cues by utilizing techniques such as limbus ellipse fitting and regional restricted normal cross-correlation rotation computation. We evaluated the network on publicly available and in-house datasets, with comparison results demonstrating its superior performance compared to related works. Ablation results further validated the effectiveness of the limbus region-focused spatial feature extractor and the combination of temporal features. Furthermore, the developed system was evaluated in a clinical setup, with results indicating remarkable accuracy and real-time performance. underscoring its potential for clinical applications.

{{</citation>}}


### (38/97) HiLM-D: Towards High-Resolution Understanding in Multimodal Large Language Models for Autonomous Driving (Xinpeng Ding et al., 2023)

{{<citation>}}

Xinpeng Ding, Jianhua Han, Hang Xu, Wei Zhang, Xiaomeng Li. (2023)  
**HiLM-D: Towards High-Resolution Understanding in Multimodal Large Language Models for Autonomous Driving**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: BLEU, Language Model  
[Paper Link](http://arxiv.org/abs/2309.05186v1)  

---


**ABSTRACT**  
Autonomous driving systems generally employ separate models for different tasks resulting in intricate designs. For the first time, we leverage singular multimodal large language models (MLLMs) to consolidate multiple autonomous driving tasks from videos, i.e., the Risk Object Localization and Intention and Suggestion Prediction (ROLISP) task. ROLISP uses natural language to simultaneously identify and interpret risk objects, understand ego-vehicle intentions, and provide motion suggestions, eliminating the necessity for task-specific architectures. However, lacking high-resolution (HR) information, existing MLLMs often miss small objects (e.g., traffic cones) and overly focus on salient ones (e.g., large trucks) when applied to ROLISP. We propose HiLM-D (Towards High-Resolution Understanding in MLLMs for Autonomous Driving), an efficient method to incorporate HR information into MLLMs for the ROLISP task. Especially, HiLM-D integrates two branches: (i) the low-resolution reasoning branch, can be any MLLMs, processes low-resolution videos to caption risk objects and discern ego-vehicle intentions/suggestions; (ii) the high-resolution perception branch (HR-PB), prominent to HiLM-D,, ingests HR images to enhance detection by capturing vision-specific HR feature maps and prioritizing all potential risks over merely salient objects. Our HR-PB serves as a plug-and-play module, seamlessly fitting into current MLLMs. Experiments on the ROLISP benchmark reveal HiLM-D's notable advantage over leading MLLMs, with improvements of 4.8% in BLEU-4 for captioning and 17.2% in mIoU for detection.

{{</citation>}}


## cs.CL (30)



### (39/97) PACE: Prompting and Augmentation for Calibrated Confidence Estimation with GPT-4 in Cloud Incident Root Cause Analysis (Dylan Zhang et al., 2023)

{{<citation>}}

Dylan Zhang, Xuchao Zhang, Chetan Bansal, Pedro Las-Casas, Rodrigo Fonseca, Saravan Rajmohan. (2023)  
**PACE: Prompting and Augmentation for Calibrated Confidence Estimation with GPT-4 in Cloud Incident Root Cause Analysis**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs-SE, cs.CL  
Keywords: AI, Augmentation, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2309.05833v1)  

---


**ABSTRACT**  
In recent years, the transition to cloud-based platforms in the IT sector has emphasized the significance of cloud incident root cause analysis to ensure service reliability and maintain customer trust. Central to this process is the efficient determination of root causes, a task made challenging due to the complex nature of contemporary cloud infrastructures. Despite the proliferation of AI-driven tools for root cause identification, their applicability remains limited by the inconsistent quality of their outputs. This paper introduces a method for enhancing confidence estimation in root cause analysis tools by prompting retrieval-augmented large language models (LLMs). This approach operates in two phases. Initially, the model evaluates its confidence based on historical incident data, considering its assessment of the evidence strength. Subsequently, the model reviews the root cause generated by the predictor. An optimization step then combines these evaluations to determine the final confidence assignment. Experimental results illustrate that our method enables the model to articulate its confidence effectively, providing a more calibrated score. We address research questions evaluating the ability of our method to produce calibrated confidence scores using LLMs, the impact of domain-specific retrieved examples on confidence estimates, and its potential generalizability across various root cause analysis models. Through this, we aim to bridge the confidence estimation gap, aiding on-call engineers in decision-making and bolstering the efficiency of cloud incident management.

{{</citation>}}


### (40/97) Hi Model, generating 'nice' instead of 'good' is not as bad as generating 'rice'! Towards Context and Semantic Infused Dialogue Generation Loss Function and Evaluation Metric (Abhisek Tiwari et al., 2023)

{{<citation>}}

Abhisek Tiwari, Muhammed Sinan, Kaushik Roy, Amit Sheth, Sriparna Saha, Pushpak Bhattacharyya. (2023)  
**Hi Model, generating 'nice' instead of 'good' is not as bad as generating 'rice'! Towards Context and Semantic Infused Dialogue Generation Loss Function and Evaluation Metric**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BLEU, Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2309.05804v1)  

---


**ABSTRACT**  
Over the past two decades, dialogue modeling has made significant strides, moving from simple rule-based responses to personalized and persuasive response generation. However, despite these advancements, the objective functions and evaluation metrics for dialogue generation have remained stagnant, i.e., cross-entropy and BLEU, respectively. These lexical-based metrics have the following key limitations: (a) word-to-word matching without semantic consideration: It assigns the same credit for failure to generate 'nice' and 'rice' for 'good'. (b) missing context attribute for evaluating the generated response: Even if a generated response is relevant to the ongoing dialogue context, it may still be penalized for not matching the gold utterance provided in the corpus. In this paper, we first investigate these limitations comprehensively and propose a new loss function called Semantic Infused Contextualized diaLogue (SemTextualLogue) loss function. Furthermore, we formulate a new evaluation metric called Dialuation, which incorporates both context relevance and semantic appropriateness while evaluating a generated response. We conducted experiments on two benchmark dialogue corpora, encompassing both task-oriented and open-domain scenarios. We found that the dialogue generation model trained with SemTextualLogue loss attained superior performance (in both quantitative and qualitative evaluation) compared to the traditional cross-entropy loss function across the datasets and evaluation metrics.

{{</citation>}}


### (41/97) Large Language Model for Science: A Study on P vs. NP (Qingxiu Dong et al., 2023)

{{<citation>}}

Qingxiu Dong, Li Dong, Ke Xu, Guangyan Zhou, Yaru Hao, Zhifang Sui, Furu Wei. (2023)  
**Large Language Model for Science: A Study on P vs. NP**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2309.05689v1)  

---


**ABSTRACT**  
In this work, we use large language models (LLMs) to augment and accelerate research on the P versus NP problem, one of the most important open problems in theoretical computer science and mathematics. Specifically, we propose Socratic reasoning, a general framework that promotes in-depth thinking with LLMs for complex problem-solving. Socratic reasoning encourages LLMs to recursively discover, solve, and integrate problems while facilitating self-evaluation and refinement. Our pilot study on the P vs. NP problem shows that GPT-4 successfully produces a proof schema and engages in rigorous reasoning throughout 97 dialogue turns, concluding "P $\neq$ NP", which is in alignment with (Xu and Zhou, 2023). The investigation uncovers novel insights within the extensive solution space of LLMs, shedding light on LLM for Science.

{{</citation>}}


### (42/97) MAmmoTH: Building Math Generalist Models through Hybrid Instruction Tuning (Xiang Yue et al., 2023)

{{<citation>}}

Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang, Huan Sun, Yu Su, Wenhu Chen. (2023)  
**MAmmoTH: Building Math Generalist Models through Hybrid Instruction Tuning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2309.05653v1)  

---


**ABSTRACT**  
We introduce MAmmoTH, a series of open-source large language models (LLMs) specifically tailored for general math problem-solving. The MAmmoTH models are trained on MathInstruct, our meticulously curated instruction tuning dataset. MathInstruct is compiled from 13 math datasets with intermediate rationales, six of which have rationales newly curated by us. It presents a unique hybrid of chain-of-thought (CoT) and program-of-thought (PoT) rationales, and also ensures extensive coverage of diverse fields in math. The hybrid of CoT and PoT not only unleashes the potential of tool use but also allows different thought processes for different math problems. As a result, the MAmmoTH series substantially outperform existing open-source models on nine mathematical reasoning datasets across all scales with an average accuracy gain between 13% and 29%. Remarkably, our MAmmoTH-7B model reaches 35% on MATH (a competition-level dataset), which exceeds the best open-source 7B model (WizardMath) by 25%, and the MAmmoTH-34B model achieves 46% accuracy on MATH, even surpassing GPT-4's CoT result. Our work underscores the importance of diverse problem coverage and the use of hybrid rationales in developing superior math generalist models.

{{</citation>}}


### (43/97) Effective Proxy for Human Labeling: Ensemble Disagreement Scores in Large Language Models for Industrial NLP (Wei Du et al., 2023)

{{<citation>}}

Wei Du, Laksh Advani, Yashmeet Gambhir, Daniel J Perry, Prashant Shiralkar, Zhengzheng Xing, Aaron Colak. (2023)  
**Effective Proxy for Human Labeling: Ensemble Disagreement Scores in Large Language Models for Industrial NLP**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2309.05619v1)  

---


**ABSTRACT**  
Large language models (LLMs) have demonstrated significant capability to generalize across a large number of NLP tasks. For industry applications, it is imperative to assess the performance of the LLM on unlabeled production data from time to time to validate for a real-world setting. Human labeling to assess model error requires considerable expense and time delay. Here we demonstrate that ensemble disagreement scores work well as a proxy for human labeling for language models in zero-shot, few-shot, and fine-tuned settings, per our evaluation on keyphrase extraction (KPE) task. We measure fidelity of the results by comparing to true error measured from human labeled ground truth. We contrast with the alternative of using another LLM as a source of machine labels, or silver labels. Results across various languages and domains show disagreement scores provide a better estimation of model performance with mean average error (MAE) as low as 0.4% and on average 13.8% better than using silver labels.

{{</citation>}}


### (44/97) Memory Injections: Correcting Multi-Hop Reasoning Failures during Inference in Transformer-Based Language Models (Mansi Sakarvadia et al., 2023)

{{<citation>}}

Mansi Sakarvadia, Aswathy Ajith, Arham Khan, Daniel Grzenda, Nathaniel Hudson, André Bauer, Kyle Chard, Ian Foster. (2023)  
**Memory Injections: Correcting Multi-Hop Reasoning Failures during Inference in Transformer-Based Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: GPT, Language Model, Reasoning, Transformer  
[Paper Link](http://arxiv.org/abs/2309.05605v2)  

---


**ABSTRACT**  
Answering multi-hop reasoning questions requires retrieving and synthesizing information from diverse sources. Large Language Models (LLMs) struggle to perform such reasoning consistently. Here we propose an approach to pinpoint and rectify multi-hop reasoning failures through targeted memory injections on LLM attention heads. First, we analyze the per-layer activations of GPT-2 models in response to single and multi-hop prompts. We then propose a mechanism that allows users to inject pertinent prompt-specific information, which we refer to as "memories," at critical LLM locations during inference. By thus enabling the LLM to incorporate additional relevant information during inference, we enhance the quality of multi-hop prompt completions. We show empirically that a simple, efficient, and targeted memory injection into a key attention layer can often increase the probability of the desired next token in multi-hop tasks, by up to 424%.

{{</citation>}}


### (45/97) An Empirical Study of NetOps Capability of Pre-Trained Large Language Models (Yukai Miao et al., 2023)

{{<citation>}}

Yukai Miao, Yu Bai, Li Chen, Dan Li, Haifeng Sun, Xizheng Wang, Ziqiu Luo, Dapeng Sun, Xiuting Xu, Qi Zhang, Chao Xiang, Xinchi Li. (2023)  
**An Empirical Study of NetOps Capability of Pre-Trained Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-NI, cs.CL  
Keywords: ChatGPT, Falcon, GPT, GPT-4, LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2309.05557v2)  

---


**ABSTRACT**  
Large language models (LLMs) can respond to human language queries and have shown powerful potential applications in network operations (NetOps). Thanks to the large amount of commonsense knowledge inherent, LLMs achieve much better inference accuracy than traditional models and emerge with strong abilities in generalization, reasoning, and code generation. These abilities may have a crucial boost to automated and intelligent NetOps. However, it remains under-explored how well LLMs perform in various NetOps tasks. In this work, we make a systematic assessment of the capabilities, strengths, and limitations of selected LLMs in the field of NetOps. The evaluation is conducted on a collection of 5,732 questions about NetOps, encompassing 26 publicly available general-domain LLMs, including ChatGPT, LLaMA, Falcon, etc. We also finetune some of these LLMs with our collected NetOps corpus and evaluate the resulting models. The evaluation method follows the widely adopted benchmarks for general-domain LLMs, combined with Chain-of-Thought Prompts and Retrieval-Augmented Generation. The results show that only GPT-4 achieves high accuracy equivalent to passing the NetOps certification exam for humans, while all the other LLMs have much lower accuracy. However, some open models like LLaMA 2 still demonstrate significant potential. Furthermore, we evaluate the impact of factors such as model parameters, prompt engineering, instruction fine-tuning etc. This work shall be treated as the initial effort to systematic evaluation of LLMs in NetOps, and a more rigorous study is required for production use. The evaluation code and dataset will be released to benefit future research.

{{</citation>}}


### (46/97) PAI-Diffusion: Constructing and Serving a Family of Open Chinese Diffusion Models for Text-to-image Synthesis on the Cloud (Chengyu Wang et al., 2023)

{{<citation>}}

Chengyu Wang, Zhongjie Duan, Bingyan Liu, Xinyi Zou, Cen Chen, Kui Jia, Jun Huang. (2023)  
**PAI-Diffusion: Constructing and Serving a Family of Open Chinese Diffusion Models for Text-to-image Synthesis on the Cloud**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CV, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.05534v1)  

---


**ABSTRACT**  
Text-to-image synthesis for the Chinese language poses unique challenges due to its large vocabulary size, and intricate character relationships. While existing diffusion models have shown promise in generating images from textual descriptions, they often neglect domain-specific contexts and lack robustness in handling the Chinese language. This paper introduces PAI-Diffusion, a comprehensive framework that addresses these limitations. PAI-Diffusion incorporates both general and domain-specific Chinese diffusion models, enabling the generation of contextually relevant images. It explores the potential of using LoRA and ControlNet for fine-grained image style transfer and image editing, empowering users with enhanced control over image generation. Moreover, PAI-Diffusion seamlessly integrates with Alibaba Cloud's Machine Learning Platform for AI, providing accessible and scalable solutions. All the Chinese diffusion model checkpoints, LoRAs, and ControlNets, including domain-specific ones, are publicly available. A user-friendly Chinese WebUI and the diffusers-api elastic inference toolkit, also open-sourced, further facilitate the easy deployment of PAI-Diffusion models in various environments, making it a valuable resource for Chinese text-to-image synthesis.

{{</citation>}}


### (47/97) Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs (Wenhua Cheng et al., 2023)

{{<citation>}}

Wenhua Cheng, Weiwei Zhang, Haihao Shen, Yiyang Cai, Xin He, Kaokao Lv. (2023)  
**Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model, Quantization  
[Paper Link](http://arxiv.org/abs/2309.05516v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have proven their exceptional capabilities in performing language-related tasks. However, their deployment poses significant challenges due to their considerable memory and storage requirements. In response to this issue, weight-only quantization, particularly 3 and 4-bit weight-only quantization, has emerged as one of the most viable solutions. As the number of bits decreases, the quantization grid broadens, thus emphasizing the importance of up and down rounding. While previous studies have demonstrated that fine-tuning up and down rounding with the addition of perturbations can enhance accuracy in some scenarios, our study is driven by the precise and limited boundary of these perturbations, where only the threshold for altering the rounding value is of significance. Consequently, we propose a concise and highly effective approach for optimizing the weight rounding task. Our method, named SignRound, involves lightweight block-wise tuning using signed gradient descent, enabling us to achieve outstanding results within 400 steps. SignRound outperforms the established baseline of rounding-to-nearest (RTN) and competes impressively against recent methods, without introducing additional inference overhead. The source code will be publicly available at https://github.com/intel/neural-compressor soon.

{{</citation>}}


### (48/97) Long-Range Transformer Architectures for Document Understanding (Thibault Douzon et al., 2023)

{{<citation>}}

Thibault Douzon, Stefan Duffner, Christophe Garcia, Jérémy Espinas. (2023)  
**Long-Range Transformer Architectures for Document Understanding**  

---
Primary Category: cs.CL  
Categories: 68T01, I-2-7, cs-CL, cs.CL  
Keywords: Computer Vision, Information Retrieval, Natural Language Understanding, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.05503v1)  

---


**ABSTRACT**  
Since their release, Transformers have revolutionized many fields from Natural Language Understanding to Computer Vision. Document Understanding (DU) was not left behind with first Transformer based models for DU dating from late 2019. However, the computational complexity of the self-attention operation limits their capabilities to small sequences. In this paper we explore multiple strategies to apply Transformer based models to long multi-page documents. We introduce 2 new multi-modal (text + layout) long-range models for DU. They are based on efficient implementations of Transformers for long sequences. Long-range models can process whole documents at once effectively and are less impaired by the document's length. We compare them to LayoutLM, a classical Transformer adapted for DU and pre-trained on millions of documents. We further propose 2D relative attention bias to guide self-attention towards relevant tokens without harming model efficiency. We observe improvements on multi-page business documents on Information Retrieval for a small performance cost on smaller sequences. Relative 2D attention revealed to be effective on dense text for both normal and long-range models.

{{</citation>}}


### (49/97) Black-Box Analysis: GPTs Across Time in Legal Textual Entailment Task (Ha-Thanh Nguyen et al., 2023)

{{<citation>}}

Ha-Thanh Nguyen, Randy Goebel, Francesca Toni, Kostas Stathis, Ken Satoh. (2023)  
**Black-Box Analysis: GPTs Across Time in Legal Textual Entailment Task**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT, GPT-3.5, GPT-4, Legal, Textual Entailment, Transformer  
[Paper Link](http://arxiv.org/abs/2309.05501v1)  

---


**ABSTRACT**  
The evolution of Generative Pre-trained Transformer (GPT) models has led to significant advancements in various natural language processing applications, particularly in legal textual entailment. We present an analysis of GPT-3.5 (ChatGPT) and GPT-4 performances on COLIEE Task 4 dataset, a prominent benchmark in this domain. The study encompasses data from Heisei 18 (2006) to Reiwa 3 (2021), exploring the models' abilities to discern entailment relationships within Japanese statute law across different periods. Our preliminary experimental results unveil intriguing insights into the models' strengths and weaknesses in handling legal textual entailment tasks, as well as the patterns observed in model performance. In the context of proprietary models with undisclosed architectures and weights, black-box analysis becomes crucial for evaluating their capabilities. We discuss the influence of training data distribution and the implications on the models' generalizability. This analysis serves as a foundation for future research, aiming to optimize GPT-based models and enable their successful adoption in legal information extraction and entailment applications.

{{</citation>}}


### (50/97) NeCo@ALQAC 2023: Legal Domain Knowledge Acquisition for Low-Resource Languages through Data Enrichment (Hai-Long Nguyen et al., 2023)

{{<citation>}}

Hai-Long Nguyen, Dieu-Quynh Nguyen, Hoang-Trung Nguyen, Thu-Trang Pham, Huu-Dong Nguyen, Thach-Anh Nguyen, Thi-Hai-Yen Vuong, Ha-Thanh Nguyen. (2023)  
**NeCo@ALQAC 2023: Legal Domain Knowledge Acquisition for Low-Resource Languages through Data Enrichment**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Legal, Low-Resource, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2309.05500v1)  

---


**ABSTRACT**  
In recent years, natural language processing has gained significant popularity in various sectors, including the legal domain. This paper presents NeCo Team's solutions to the Vietnamese text processing tasks provided in the Automated Legal Question Answering Competition 2023 (ALQAC 2023), focusing on legal domain knowledge acquisition for low-resource languages through data enrichment. Our methods for the legal document retrieval task employ a combination of similarity ranking and deep learning models, while for the second task, which requires extracting an answer from a relevant legal article in response to a question, we propose a range of adaptive techniques to handle different question types. Our approaches achieve outstanding results on both tasks of the competition, demonstrating the potential benefits and effectiveness of question answering systems in the legal field, particularly for low-resource languages.

{{</citation>}}


### (51/97) Personality Detection and Analysis using Twitter Data (Abhilash Datta et al., 2023)

{{<citation>}}

Abhilash Datta, Souvic Chakraborty, Animesh Mukherjee. (2023)  
**Personality Detection and Analysis using Twitter Data**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs.CL  
Keywords: Personality Detection, Twitter  
[Paper Link](http://arxiv.org/abs/2309.05497v1)  

---


**ABSTRACT**  
Personality types are important in various fields as they hold relevant information about the characteristics of a human being in an explainable format. They are often good predictors of a person's behaviors in a particular environment and have applications ranging from candidate selection to marketing and mental health. Recently automatic detection of personality traits from texts has gained significant attention in computational linguistics. Most personality detection and analysis methods have focused on small datasets making their experimental observations often limited. To bridge this gap, we focus on collecting and releasing the largest automatically curated dataset for the research community which has 152 million tweets and 56 thousand data points for the Myers-Briggs personality type (MBTI) prediction task. We perform a series of extensive qualitative and quantitative studies on our dataset to analyze the data patterns in a better way and infer conclusions. We show how our intriguing analysis results often follow natural intuition. We also perform a series of ablation studies to show how the baselines perform for our dataset.

{{</citation>}}


### (52/97) CrisisTransformers: Pre-trained language models and sentence encoders for crisis-related social media texts (Rabindra Lamsal et al., 2023)

{{<citation>}}

Rabindra Lamsal, Maria Rodriguez Read, Shanika Karunasekera. (2023)  
**CrisisTransformers: Pre-trained language models and sentence encoders for crisis-related social media texts**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, NLP, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.05494v1)  

---


**ABSTRACT**  
Social media platforms play an essential role in crisis communication, but analyzing crisis-related social media texts is challenging due to their informal nature. Transformer-based pre-trained models like BERT and RoBERTa have shown success in various NLP tasks, but they are not tailored for crisis-related texts. Furthermore, general-purpose sentence encoders are used to generate sentence embeddings, regardless of the textual complexities in crisis-related texts. Advances in applications like text classification, semantic search, and clustering contribute to effective processing of crisis-related texts, which is essential for emergency responders to gain a comprehensive view of a crisis event, whether historical or real-time. To address these gaps in crisis informatics literature, this study introduces CrisisTransformers, an ensemble of pre-trained language models and sentence encoders trained on an extensive corpus of over 15 billion word tokens from tweets associated with more than 30 crisis events, including disease outbreaks, natural disasters, conflicts, and other critical incidents. We evaluate existing models and CrisisTransformers on 18 crisis-specific public datasets. Our pre-trained models outperform strong baselines across all datasets in classification tasks, and our best-performing sentence encoder improves the state-of-the-art by 17.43% in sentence encoding tasks. Additionally, we investigate the impact of model initialization on convergence and evaluate the significance of domain-specific models in generating semantically meaningful sentence embeddings. All models are publicly released (https://huggingface.co/crisistransformers), with the anticipation that they will serve as a robust baseline for tasks involving the analysis of crisis-related social media texts.

{{</citation>}}


### (53/97) Zero-shot Learning with Minimum Instruction to Extract Social Determinants and Family History from Clinical Notes using GPT Model (Neel Jitesh Bhate et al., 2023)

{{<citation>}}

Neel Jitesh Bhate, Ansh Mittal, Zhe He, Xiao Luo. (2023)  
**Zero-shot Learning with Minimum Instruction to Extract Social Determinants and Family History from Clinical Notes using GPT Model**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Clinical, GPT, GPT-3.5, NER  
[Paper Link](http://arxiv.org/abs/2309.05475v1)  

---


**ABSTRACT**  
Demographics, Social determinants of health, and family history documented in the unstructured text within the electronic health records are increasingly being studied to understand how this information can be utilized with the structured data to improve healthcare outcomes. After the GPT models were released, many studies have applied GPT models to extract this information from the narrative clinical notes. Different from the existing work, our research focuses on investigating the zero-shot learning on extracting this information together by providing minimum information to the GPT model. We utilize de-identified real-world clinical notes annotated for demographics, various social determinants, and family history information. Given that the GPT model might provide text different from the text in the original data, we explore two sets of evaluation metrics, including the traditional NER evaluation metrics and semantic similarity evaluation metrics, to completely understand the performance. Our results show that the GPT-3.5 method achieved an average of 0.975 F1 on demographics extraction, 0.615 F1 on social determinants extraction, and 0.722 F1 on family history extraction. We believe these results can be further improved through model fine-tuning or few-shots learning. Through the case studies, we also identified the limitations of the GPT models, which need to be addressed in future research.

{{</citation>}}


### (54/97) Textbooks Are All You Need II: phi-1.5 technical report (Yuanzhi Li et al., 2023)

{{<citation>}}

Yuanzhi Li, Sébastien Bubeck, Ronen Eldan, Allie Del Giorno, Suriya Gunasekar, Yin Tat Lee. (2023)  
**Textbooks Are All You Need II: phi-1.5 technical report**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2309.05463v1)  

---


**ABSTRACT**  
We continue the investigation into the power of smaller Transformer-based language models as initiated by \textbf{TinyStories} -- a 10 million parameter model that can produce coherent English -- and the follow-up work on \textbf{phi-1}, a 1.3 billion parameter model with Python coding performance close to the state-of-the-art. The latter work proposed to use existing Large Language Models (LLMs) to generate ``textbook quality" data as a way to enhance the learning process compared to traditional web data. We follow the ``Textbooks Are All You Need" approach, focusing this time on common sense reasoning in natural language, and create a new 1.3 billion parameter model named \textbf{phi-1.5}, with performance on natural language tasks comparable to models 5x larger, and surpassing most non-frontier LLMs on more complex reasoning tasks such as grade-school mathematics and basic coding. More generally, \textbf{phi-1.5} exhibits many of the traits of much larger LLMs, both good -- such as the ability to ``think step by step" or perform some rudimentary in-context learning -- and bad, including hallucinations and the potential for toxic and biased generations -- encouragingly though, we are seeing improvement on that front thanks to the absence of web data. We open-source \textbf{phi-1.5} to promote further research on these urgent topics.

{{</citation>}}


### (55/97) Flesch or Fumble? Evaluating Readability Standard Alignment of Instruction-Tuned Language Models (Joseph Marvin Imperial et al., 2023)

{{<citation>}}

Joseph Marvin Imperial, Harish Tayyar Madabushi. (2023)  
**Flesch or Fumble? Evaluating Readability Standard Alignment of Instruction-Tuned Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BLOOM, ChatGPT, GPT, Language Model, T5  
[Paper Link](http://arxiv.org/abs/2309.05454v1)  

---


**ABSTRACT**  
Readability metrics and standards such as Flesch Kincaid Grade Level (FKGL) and the Common European Framework of Reference for Languages (CEFR) exist to guide teachers and educators to properly assess the complexity of educational materials before administering them for classroom use. In this study, we select a diverse set of open and closed-source instruction-tuned language models and investigate their performances in writing story completions and simplifying narratives$-$tasks that teachers perform$-$using standard-guided prompts controlling text readability. Our extensive findings provide empirical proof of how globally recognized models like ChatGPT may be considered less effective and may require more refined prompts for these generative tasks compared to other open-sourced models such as BLOOMZ and FlanT5$-$which have shown promising results.

{{</citation>}}


### (56/97) Evaluating the Deductive Competence of Large Language Models (S. M. Seals et al., 2023)

{{<citation>}}

S. M. Seals, Valerie L. Shalin. (2023)  
**Evaluating the Deductive Competence of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.05452v1)  

---


**ABSTRACT**  
The development of highly fluent large language models (LLMs) has prompted increased interest in assessing their reasoning and problem-solving capabilities. We investigate whether several LLMs can solve a classic type of deductive reasoning problem from the cognitive science literature. The tested LLMs have limited abilities to solve these problems in their conventional form. We performed follow up experiments to investigate if changes to the presentation format and content improve model performance. We do find performance differences between conditions; however, they do not improve overall performance. Moreover, we find that performance interacts with presentation format and content in unexpected ways that differ from human performance. Overall, our results suggest that LLMs have unique reasoning biases that are only partially predicted from human reasoning performance.

{{</citation>}}


### (57/97) Improving Information Extraction on Business Documents with Specific Pre-Training Tasks (Thibault Douzon et al., 2023)

{{<citation>}}

Thibault Douzon, Stefan Duffner, Christophe Garcia, Jérémy Espinas. (2023)  
**Improving Information Extraction on Business Documents with Specific Pre-Training Tasks**  

---
Primary Category: cs.CL  
Categories: 68T01, I-2-7, cs-AI, cs-CL, cs.CL  
Keywords: Information Extraction, Language Model, Natural Language Processing, Transformer  
[Paper Link](http://arxiv.org/abs/2309.05429v1)  

---


**ABSTRACT**  
Transformer-based Language Models are widely used in Natural Language Processing related tasks. Thanks to their pre-training, they have been successfully adapted to Information Extraction in business documents. However, most pre-training tasks proposed in the literature for business documents are too generic and not sufficient to learn more complex structures. In this paper, we use LayoutLM, a language model pre-trained on a collection of business documents, and introduce two new pre-training tasks that further improve its capacity to extract relevant information. The first is aimed at better understanding the complex layout of documents, and the second focuses on numeric values and their order of magnitude. These tasks force the model to learn better-contextualized representations of the scanned documents. We further introduce a new post-processing algorithm to decode BIESO tags in Information Extraction that performs better with complex entities. Our method significantly improves extraction performance on both public (from 93.88 to 95.50 F1 score) and private (from 84.35 to 84.84 F1 score) datasets composed of expense receipts, invoices, and purchase orders.

{{</citation>}}


### (58/97) Experimenting with UD Adaptation of an Unsupervised Rule-based Approach for Sentiment Analysis of Mexican Tourist Texts (Olga Kellert et al., 2023)

{{<citation>}}

Olga Kellert, Mahmud Uz Zaman, Nicholas Hill Matlis, Carlos Gómez-Rodríguez. (2023)  
**Experimenting with UD Adaptation of an Unsupervised Rule-based Approach for Sentiment Analysis of Mexican Tourist Texts**  

---
Primary Category: cs.CL  
Categories: 68T50, I-2-7, cs-CL, cs.CL  
Keywords: NLP, Sentiment Analysis  
[Paper Link](http://arxiv.org/abs/2309.05312v1)  

---


**ABSTRACT**  
This paper summarizes the results of experimenting with Universal Dependencies (UD) adaptation of an Unsupervised, Compositional and Recursive (UCR) rule-based approach for Sentiment Analysis (SA) submitted to the Shared Task at Rest-Mex 2023 (Team Olga/LyS-SALSA) (within the IberLEF 2023 conference). By using basic syntactic rules such as rules of modification and negation applied on words from sentiment dictionaries, our approach exploits some advantages of an unsupervised method for SA: (1) interpretability and explainability of SA, (2) robustness across datasets, languages and domains and (3) usability by non-experts in NLP. We compare our approach with other unsupervised approaches of SA that in contrast to our UCR rule-based approach use simple heuristic rules to deal with negation and modification. Our results show a considerable improvement over these approaches. We discuss future improvements of our results by using modality features as another shifting rule of polarity and word disambiguation techniques to identify the right sentiment words.

{{</citation>}}


### (59/97) Analysing Cross-Lingual Transfer in Low-Resourced African Named Entity Recognition (Michael Beukman et al., 2023)

{{<citation>}}

Michael Beukman, Manuel Fokam. (2023)  
**Analysing Cross-Lingual Transfer in Low-Resourced African Named Entity Recognition**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Low-Resource, NLP, Named Entity Recognition  
[Paper Link](http://arxiv.org/abs/2309.05311v1)  

---


**ABSTRACT**  
Transfer learning has led to large gains in performance for nearly all NLP tasks while making downstream models easier and faster to train. This has also been extended to low-resourced languages, with some success. We investigate the properties of cross-lingual transfer learning between ten low-resourced languages, from the perspective of a named entity recognition task. We specifically investigate how much adaptive fine-tuning and the choice of transfer language affect zero-shot transfer performance. We find that models that perform well on a single language often do so at the expense of generalising to others, while models with the best generalisation to other languages suffer in individual language performance. Furthermore, the amount of data overlap between the source and target datasets is a better predictor of transfer performance than either the geographical or genetic distance between the languages.

{{</citation>}}


### (60/97) Minuteman: Machine and Human Joining Forces in Meeting Summarization (František Kmječ et al., 2023)

{{<citation>}}

František Kmječ, Ondřej Bojar. (2023)  
**Minuteman: Machine and Human Joining Forces in Meeting Summarization**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Summarization  
[Paper Link](http://arxiv.org/abs/2309.05272v1)  

---


**ABSTRACT**  
Many meetings require creating a meeting summary to keep everyone up to date. Creating minutes of sufficient quality is however very cognitively demanding. Although we currently possess capable models for both audio speech recognition (ASR) and summarization, their fully automatic use is still problematic. ASR models frequently commit errors when transcribing named entities while the summarization models tend to hallucinate and misinterpret the transcript. We propose a novel tool -- Minuteman -- to enable efficient semi-automatic meeting minuting. The tool provides a live transcript and a live meeting summary to the users, who can edit them in a collaborative manner, enabling correction of ASR errors and imperfect summary points in real time. The resulting application eases the cognitive load of the notetakers and allows them to easily catch up if they missed a part of the meeting due to absence or a lack of focus. We conduct several tests of the application in varied settings, exploring the worthiness of the concept and the possible user strategies.

{{</citation>}}


### (61/97) CONFLATOR: Incorporating Switching Point based Rotatory Positional Encodings for Code-Mixed Language Modeling (Mohsin Ali et al., 2023)

{{<citation>}}

Mohsin Ali, Kandukuri Sai Teja, Neeharika Gupta, Parth Patwa, Anubhab Chatterjee, Vinija Jain, Aman Chadha, Amitava Das. (2023)  
**CONFLATOR: Incorporating Switching Point based Rotatory Positional Encodings for Code-Mixed Language Modeling**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2309.05270v1)  

---


**ABSTRACT**  
The mixing of two or more languages is called Code-Mixing (CM). CM is a social norm in multilingual societies. Neural Language Models (NLMs) like transformers have been very effective on many NLP tasks. However, NLM for CM is an under-explored area. Though transformers are capable and powerful, they cannot always encode positional/sequential information since they are non-recurrent. Therefore, to enrich word information and incorporate positional information, positional encoding is defined. We hypothesize that Switching Points (SPs), i.e., junctions in the text where the language switches (L1 -> L2 or L2-> L1), pose a challenge for CM Language Models (LMs), and hence give special emphasis to switching points in the modeling process. We experiment with several positional encoding mechanisms and show that rotatory positional encodings along with switching point information yield the best results.   We introduce CONFLATOR: a neural language modeling approach for code-mixed languages. CONFLATOR tries to learn to emphasize switching points using smarter positional encoding, both at unigram and bigram levels. CONFLATOR outperforms the state-of-the-art on two tasks based on code-mixed Hindi and English (Hinglish): (i) sentiment analysis and (ii) machine translation.

{{</citation>}}


### (62/97) Unsupervised Bias Detection in College Student Newspapers (Adam M. Lehavi et al., 2023)

{{<citation>}}

Adam M. Lehavi, William McCormack, Noah Kornfeld, Solomon Glazer. (2023)  
**Unsupervised Bias Detection in College Student Newspapers**  

---
Primary Category: cs.CL  
Categories: I-2-7; H-2-8; J-4, cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2309.06557v1)  

---


**ABSTRACT**  
This paper presents a pipeline with minimal human influence for scraping and detecting bias on college newspaper archives. This paper introduces a framework for scraping complex archive sites that automated tools fail to grab data from, and subsequently generates a dataset of 14 student papers with 23,154 entries. This data can also then be queried by keyword to calculate bias by comparing the sentiment of a large language model summary to the original article. The advantages of this approach are that it is less comparative than reconstruction bias and requires less labelled data than generating keyword sentiment. Results are calculated on politically charged words as well as control words to show how conclusions can be drawn. The complete method facilitates the extraction of nuanced insights with minimal assumptions and categorizations, paving the way for a more objective understanding of bias within student newspaper sources.

{{</citation>}}


### (63/97) Detecting Natural Language Biases with Prompt-based Learning (Md Abdul Aowal et al., 2023)

{{<citation>}}

Md Abdul Aowal, Maliha T Islam, Priyanka Mary Mammen, Sandesh Shetty. (2023)  
**Detecting Natural Language Biases with Prompt-based Learning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, Bias, T5  
[Paper Link](http://arxiv.org/abs/2309.05227v1)  

---


**ABSTRACT**  
In this project, we want to explore the newly emerging field of prompt engineering and apply it to the downstream task of detecting LM biases. More concretely, we explore how to design prompts that can indicate 4 different types of biases: (1) gender, (2) race, (3) sexual orientation, and (4) religion-based. Within our project, we experiment with different manually crafted prompts that can draw out the subtle biases that may be present in the language model. We apply these prompts to multiple variations of popular and well-recognized models: BERT, RoBERTa, and T5 to evaluate their biases. We provide a comparative analysis of these models and assess them using a two-fold method: use human judgment to decide whether model predictions are biased and utilize model-level judgment (through further prompts) to understand if a model can self-diagnose the biases of its own prediction.

{{</citation>}}


### (64/97) Understanding the Impact of Post-Training Quantization on Large Language Models (Somnath Roy, 2023)

{{<citation>}}

Somnath Roy. (2023)  
**Understanding the Impact of Post-Training Quantization on Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, Falcon, GPT, Language Model, Quantization  
[Paper Link](http://arxiv.org/abs/2309.05210v2)  

---


**ABSTRACT**  
Large language models (LLMs) are rapidly increasing in size, with the number of parameters becoming a key factor in the success of many commercial models, such as ChatGPT, Claude, and Bard. Even the recently released publicly accessible models for commercial usage, such as Falcon and Llama2, come equipped with billions of parameters. This significant increase in the number of parameters makes deployment and operation very costly. The remarkable progress in the field of quantization for large neural networks in general and LLMs in particular, has made these models more accessible by enabling them to be deployed on consumer-grade GPUs. Quantized models generally demonstrate comparable performance levels to their unquantized base counterparts. Nonetheless, there exists a notable gap in our comprehensive understanding of how these quantized models respond to hyperparameters, such as temperature, max new tokens, and topk, particularly for next word prediction. The present analysis reveals that nf4 and fp4 are equally proficient 4-bit quantization techniques, characterized by similar attributes such as inference speed, memory consumption, and the quality of generated content. Nevertheless, these quantization methods exhibit distinct behaviors at varying temperature settings, both in the context of smaller and larger models. It is noteworthy that, in general, 4-bit quantized models of varying sizes exhibit heightened sensitivity to lower temperature settings, unlike their unquantized counterparts. Additionally, int8 quantization is associated with significantly slower inference speeds, whereas unquantized fp16 models consistently yield the fastest inference speeds across models of all sizes.

{{</citation>}}


### (65/97) From Artificially Real to Real: Leveraging Pseudo Data from Large Language Models for Low-Resource Molecule Discovery (Yuhan Chen et al., 2023)

{{<citation>}}

Yuhan Chen, Nuwa Xi, Yanrui Du, Haochun Wang, Chen Jianyu, Sendong Zhao, Bing Qin. (2023)  
**From Artificially Real to Real: Leveraging Pseudo Data from Large Language Models for Low-Resource Molecule Discovery**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Low-Resource  
[Paper Link](http://arxiv.org/abs/2309.05203v1)  

---


**ABSTRACT**  
Molecule discovery serves as a cornerstone in numerous scientific domains, fueling the development of new materials and innovative drug designs. Recent developments of in-silico molecule discovery have highlighted the promising results of cross-modal techniques, which bridge molecular structures with their descriptive annotations. However, these cross-modal methods frequently encounter the issue of data scarcity, hampering their performance and application. In this paper, we address the low-resource challenge by utilizing artificially-real data generated by Large Language Models (LLMs). We first introduce a retrieval-based prompting strategy to construct high-quality pseudo data, then explore the optimal method to effectively leverage this pseudo data. Experiments show that using pseudo data for domain adaptation outperforms all existing methods, while also requiring a smaller model scale, reduced data size and lower training cost, highlighting its efficiency. Furthermore, our method shows a sustained improvement as the volume of pseudo data increases, revealing the great potential of pseudo data in advancing low-resource cross-modal molecule discovery.

{{</citation>}}


### (66/97) Two is Better Than One: Answering Complex Questions by Multiple Knowledge Sources with Generalized Links (Minhao Zhang et al., 2023)

{{<citation>}}

Minhao Zhang, Yongliang Ma, Yanzeng Li, Ruoyu Zhang, Lei Zou, Ming Zhou. (2023)  
**Two is Better Than One: Answering Complex Questions by Multiple Knowledge Sources with Generalized Links**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2309.05201v1)  

---


**ABSTRACT**  
Incorporating multiple knowledge sources is proven to be beneficial for answering complex factoid questions. To utilize multiple knowledge bases (KB), previous works merge all KBs into a single graph via entity alignment and reduce the problem to question-answering (QA) over the fused KB. In reality, various link relations between KBs might be adopted in QA over multi-KBs. In addition to the identity between the alignable entities (i.e. full link), unalignable entities expressing the different aspects or types of an abstract concept may also be treated identical in a question (i.e. partial link). Hence, the KB fusion in prior works fails to represent all types of links, restricting their ability to comprehend multi-KBs for QA. In this work, we formulate the novel Multi-KB-QA task that leverages the full and partial links among multiple KBs to derive correct answers, a benchmark with diversified link and query types is also constructed to efficiently evaluate Multi-KB-QA performance. Finally, we propose a method for Multi-KB-QA that encodes all link relations in the KB embedding to score and rank candidate answers. Experiments show that our method markedly surpasses conventional KB-QA systems in Multi-KB-QA, justifying the necessity of devising this task.

{{</citation>}}


### (67/97) Does Writing with Language Models Reduce Content Diversity? (Vishakh Padmakumar et al., 2023)

{{<citation>}}

Vishakh Padmakumar, He He. (2023)  
**Does Writing with Language Models Reduce Content Diversity?**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs-HC, cs-LG, cs.CL  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2309.05196v1)  

---


**ABSTRACT**  
Large language models (LLMs) have led to a surge in collaborative writing with model assistance. As different users incorporate suggestions from the same model, there is a risk of decreased diversity in the produced content, potentially limiting diverse perspectives in public discourse. In this work, we measure the impact of co-writing on diversity via a controlled experiment, where users write argumentative essays in three setups -- using a base LLM (GPT3), a feedback-tuned LLM (InstructGPT), and writing without model help. We develop a set of diversity metrics and find that writing with InstructGPT (but not the GPT3) results in a statistically significant reduction in diversity. Specifically, it increases the similarity between the writings of different authors and reduces the overall lexical and content diversity. We additionally find that this effect is mainly attributable to InstructGPT contributing less diverse text to co-written essays. In contrast, the user-contributed text remains unaffected by model collaboration. This suggests that the recent improvement in generation quality from adapting models to human feedback might come at the cost of more homogeneous and less diverse content.

{{</citation>}}


### (68/97) DePT: Decomposed Prompt Tuning for Parameter-Efficient Fine-tuning (Zhengxiang Shi et al., 2023)

{{<citation>}}

Zhengxiang Shi, Aldo Lipani. (2023)  
**DePT: Decomposed Prompt Tuning for Parameter-Efficient Fine-tuning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs.CL  
Keywords: Language Model, NLP, Transformer  
[Paper Link](http://arxiv.org/abs/2309.05173v1)  

---


**ABSTRACT**  
Prompt tuning (PT), where a small amount of trainable soft (continuous) prompt vectors is affixed to the input of language models (LM), has shown promising results across various tasks and models for parameter-efficient fine-tuning (PEFT). PT stands out from other PEFT approaches because it maintains competitive performance with fewer trainable parameters and does not drastically scale up its parameters as the model size expands. However, PT introduces additional soft prompt tokens, leading to longer input sequences, which significantly impacts training and inference time and memory usage due to the Transformer's quadratic complexity. Particularly concerning for Large Language Models (LLMs) that face heavy daily querying. To address this issue, we propose Decomposed Prompt Tuning (DePT), which decomposes the soft prompt into a shorter soft prompt and a pair of low-rank matrices that are then optimised with two different learning rates. This allows DePT to achieve better performance while saving over 20% memory and time costs compared to vanilla PT and its variants, without changing trainable parameter sizes. Through extensive experiments on 23 natural language processing (NLP) and vision-language (VL) tasks, we demonstrate that DePT outperforms state-of-the-art PEFT approaches, including the full fine-tuning baseline in some scenarios. Additionally, we empirically show that DEPT grows more efficient as the model size increases. Our further study reveals that DePT integrates seamlessly with parameter-efficient transfer learning in the few-shot learning setting and highlights its adaptability to various model architectures and sizes.

{{</citation>}}


## eess.SP (1)



### (69/97) Reinforcement Learning for Supply Chain Attacks Against Frequency and Voltage Control (Amr S. Mohamed et al., 2023)

{{<citation>}}

Amr S. Mohamed, Sumin Lee, Deepa Kundur. (2023)  
**Reinforcement Learning for Supply Chain Attacks Against Frequency and Voltage Control**  

---
Primary Category: eess.SP  
Categories: cs-SY, eess-SP, eess-SY, eess.SP  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.05814v1)  

---


**ABSTRACT**  
The ongoing modernization of the power system, involving new equipment installations and upgrades, exposes the power system to the introduction of malware into its operation through supply chain attacks. Supply chain attacks present a significant threat to power systems, allowing cybercriminals to bypass network defenses and execute deliberate attacks at the physical layer. Given the exponential advancements in machine intelligence, cybercriminals will leverage this technology to create sophisticated and adaptable attacks that can be incorporated into supply chain attacks. We demonstrate the use of reinforcement learning for developing intelligent attacks incorporated into supply chain attacks against generation control devices. We simulate potential disturbances impacting frequency and voltage regulation. The presented method can provide valuable guidance for defending against supply chain attacks.

{{</citation>}}


## stat.ML (1)



### (70/97) On the Fine-Grained Hardness of Inverting Generative Models (Feyza Duman Keles et al., 2023)

{{<citation>}}

Feyza Duman Keles, Chinmay Hegde. (2023)  
**On the Fine-Grained Hardness of Inverting Generative Models**  

---
Primary Category: stat.ML  
Categories: cs-CC, cs-LG, stat-ML, stat.ML  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2309.05795v1)  

---


**ABSTRACT**  
The objective of generative model inversion is to identify a size-$n$ latent vector that produces a generative model output that closely matches a given target. This operation is a core computational primitive in numerous modern applications involving computer vision and NLP. However, the problem is known to be computationally challenging and NP-hard in the worst case. This paper aims to provide a fine-grained view of the landscape of computational hardness for this problem. We establish several new hardness lower bounds for both exact and approximate model inversion. In exact inversion, the goal is to determine whether a target is contained within the range of a given generative model. Under the strong exponential time hypothesis (SETH), we demonstrate that the computational complexity of exact inversion is lower bounded by $\Omega(2^n)$ via a reduction from $k$-SAT; this is a strengthening of known results. For the more practically relevant problem of approximate inversion, the goal is to determine whether a point in the model range is close to a given target with respect to the $\ell_p$-norm. When $p$ is a positive odd integer, under SETH, we provide an $\Omega(2^n)$ complexity lower bound via a reduction from the closest vectors problem (CVP). Finally, when $p$ is even, under the exponential time hypothesis (ETH), we provide a lower bound of $2^{\Omega (n)}$ via a reduction from Half-Clique and Vertex-Cover.

{{</citation>}}


## cs.SD (1)



### (71/97) Natural Language Supervision for General-Purpose Audio Representations (Benjamin Elizalde et al., 2023)

{{<citation>}}

Benjamin Elizalde, Soham Deshmukh, Huaming Wang. (2023)  
**Natural Language Supervision for General-Purpose Audio Representations**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Contrastive Learning, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2309.05767v1)  

---


**ABSTRACT**  
Audio-Language models jointly learn multimodal text and audio representations that enable Zero-Shot inference. Models rely on the encoders to create powerful representations of the input and generalize to multiple tasks ranging from sounds, music, and speech. Although models have achieved remarkable performance, there is still a performance gap with task-specific models. In this paper, we propose a Contrastive Language-Audio Pretraining model that is pretrained with a diverse collection of 4.6M audio-text pairs employing two innovative encoders for Zero-Shot inference. To learn audio representations, we trained an audio encoder on 22 audio tasks, instead of the standard training of sound event classification. To learn language representations, we trained an autoregressive decoder-only model instead of the standard encoder-only models. Then, the audio and language representations are brought into a joint multimodal space using Contrastive Learning. We used our encoders to improve the downstream performance by a margin. We extensively evaluated the generalization of our representations on 26 downstream tasks, the largest in the literature. Our model achieves state of the art results in several tasks leading the way towards general-purpose audio representations.

{{</citation>}}


## cs.RO (3)



### (72/97) Dynamic Handover: Throw and Catch with Bimanual Hands (Binghao Huang et al., 2023)

{{<citation>}}

Binghao Huang, Yuanpei Chen, Tianyu Wang, Yuzhe Qin, Yaodong Yang, Nikolay Atanasov, Xiaolong Wang. (2023)  
**Dynamic Handover: Throw and Catch with Bimanual Hands**  

---
Primary Category: cs.RO  
Categories: cs-LG, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.05655v1)  

---


**ABSTRACT**  
Humans throw and catch objects all the time. However, such a seemingly common skill introduces a lot of challenges for robots to achieve: The robots need to operate such dynamic actions at high-speed, collaborate precisely, and interact with diverse objects. In this paper, we design a system with two multi-finger hands attached to robot arms to solve this problem. We train our system using Multi-Agent Reinforcement Learning in simulation and perform Sim2Real transfer to deploy on the real robots. To overcome the Sim2Real gap, we provide multiple novel algorithm designs including learning a trajectory prediction model for the object. Such a model can help the robot catcher has a real-time estimation of where the object will be heading, and then react accordingly. We conduct our experiments with multiple objects in the real-world system, and show significant improvements over multiple baselines. Our project page is available at \url{https://binghao-huang.github.io/dynamic_handover/}.

{{</citation>}}


### (73/97) Design and Validation of a Wireless Drone Docking Station (Dario Stuhne et al., 2023)

{{<citation>}}

Dario Stuhne, Goran Vasiljevic, Stjepan Bogdan, Zdenko Kovacic. (2023)  
**Design and Validation of a Wireless Drone Docking Station**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2309.05433v1)  

---


**ABSTRACT**  
Drones are increasingly operating autonomously, and the need for extending drone power autonomy is rapidly increasing. One of the most promising solutions to extend drone power autonomy is the use of docking stations to support both landing and recharging of the drone. To this end, we introduce a novel wireless drone docking station with three commercial wireless charging modules. We have developed two independent units, both in mechanical and electrical aspects: the energy transmitting unit and the energy receiving unit. We have also studied the efficiency of wireless power transfer and demonstrated the advantages of connecting three receiver modules connected in series and parallel. We have achieved maximum output power of 96.5 W with a power transfer efficiency of 56.6% for the series connection of coils. Finally, we implemented the system in practice on a drone and tested both energy transfer and landing.

{{</citation>}}


### (74/97) Effect of Adapting to Human Preferences on Trust in Human-Robot Teaming (Shreyas Bhat et al., 2023)

{{<citation>}}

Shreyas Bhat, Joseph B. Lyons, Cong Shi, X. Jessie Yang. (2023)  
**Effect of Adapting to Human Preferences on Trust in Human-Robot Teaming**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.05179v1)  

---


**ABSTRACT**  
We present the effect of adapting to human preferences on trust in a human-robot teaming task. The team performs a task in which the robot acts as an action recommender to the human. It is assumed that the behavior of the human and the robot is based on some reward function they try to optimize. We use a new human trust-behavior model that enables the robot to learn and adapt to the human's preferences in real-time during their interaction using Bayesian Inverse Reinforcement Learning. We present three strategies for the robot to interact with a human: a non-learner strategy, in which the robot assumes that the human's reward function is the same as the robot's, a non-adaptive learner strategy that learns the human's reward function for performance estimation, but still optimizes its own reward function, and an adaptive-learner strategy that learns the human's reward function for performance estimation and also optimizes this learned reward function. Results show that adapting to the human's reward function results in the highest trust in the robot.

{{</citation>}}


## q-fin.ST (1)



### (75/97) Desenvolvimento de modelo para predição de cotações de ação baseada em análise de sentimentos de tweets (Mario Mitsuo Akita et al., 2023)

{{<citation>}}

Mario Mitsuo Akita, Everton Josue da Silva. (2023)  
**Desenvolvimento de modelo para predição de cotações de ação baseada em análise de sentimentos de tweets**  

---
Primary Category: q-fin.ST  
Categories: cs-LG, q-fin-ST, q-fin.ST  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2309.06538v1)  

---


**ABSTRACT**  
Training machine learning models for predicting stock market share prices is an active area of research since the automatization of trading such papers was available in real time. While most of the work in this field of research is done by training Neural networks based on past prices of stock shares, in this work, we use iFeel 2.0 platform to extract 19 sentiment features from posts obtained from microblog platform Twitter that mention the company Petrobras. Then, we used those features to train XBoot models to predict future stock prices for the referred company. Later, we simulated the trading of Petrobras' shares based on the model's outputs and determined the gain of R$88,82 (net) in a 250-day period when compared to a 100 random models' average performance.

{{</citation>}}


## cs.NI (2)



### (76/97) A Comparative Analysis of Deep Reinforcement Learning-based xApps in O-RAN (Maria Tsampazi et al., 2023)

{{<citation>}}

Maria Tsampazi, Salvatore D'Oro, Michele Polese, Leonardo Bonati, Gwenael Poitau, Michael Healy, Tommaso Melodia. (2023)  
**A Comparative Analysis of Deep Reinforcement Learning-based xApps in O-RAN**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI, eess-SP  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.05621v1)  

---


**ABSTRACT**  
The highly heterogeneous ecosystem of Next Generation (NextG) wireless communication systems calls for novel networking paradigms where functionalities and operations can be dynamically and optimally reconfigured in real time to adapt to changing traffic conditions and satisfy stringent and diverse Quality of Service (QoS) demands. Open Radio Access Network (RAN) technologies, and specifically those being standardized by the O-RAN Alliance, make it possible to integrate network intelligence into the once monolithic RAN via intelligent applications, namely, xApps and rApps. These applications enable flexible control of the network resources and functionalities, network management, and orchestration through data-driven control loops. Despite recent work demonstrating the effectiveness of Deep Reinforcement Learning (DRL) in controlling O-RAN systems, how to design these solutions in a way that does not create conflicts and unfair resource allocation policies is still an open challenge. In this paper, we perform a comparative analysis where we dissect the impact of different DRL-based xApp designs on network performance. Specifically, we benchmark 12 different xApps that embed DRL agents trained using different reward functions, with different action spaces and with the ability to hierarchically control different network parameters. We prototype and evaluate these xApps on Colosseum, the world's largest O-RAN-compliant wireless network emulator with hardware-in-the-loop. We share the lessons learned and discuss our experimental results, which demonstrate how certain design choices deliver the highest performance while others might result in a competitive behavior between different classes of traffic with similar objectives.

{{</citation>}}


### (77/97) Advancing Federated Learning in 6G: A Trusted Architecture with Graph-based Analysis (Wenxuan Ye et al., 2023)

{{<citation>}}

Wenxuan Ye, Chendi Qian, Xueli An, Xueqiang Yan, Georg Carle. (2023)  
**Advancing Federated Learning in 6G: A Trusted Architecture with Graph-based Analysis**  

---
Primary Category: cs.NI  
Categories: cs-LG, cs-NI, cs.NI  
Keywords: AI, GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2309.05525v1)  

---


**ABSTRACT**  
Integrating native AI support into the network architecture is an essential objective of 6G. Federated Learning (FL) emerges as a potential paradigm, facilitating decentralized AI model training across a diverse range of devices under the coordination of a central server. However, several challenges hinder its wide application in the 6G context, such as malicious attacks and privacy snooping on local model updates, and centralization pitfalls. This work proposes a trusted architecture for supporting FL, which utilizes Distributed Ledger Technology (DLT) and Graph Neural Network (GNN), including three key features. First, a pre-processing layer employing homomorphic encryption is incorporated to securely aggregate local models, preserving the privacy of individual models. Second, given the distributed nature and graph structure between clients and nodes in the pre-processing layer, GNN is leveraged to identify abnormal local models, enhancing system security. Third, DLT is utilized to decentralize the system by selecting one of the candidates to perform the central server's functions. Additionally, DLT ensures reliable data management by recording data exchanges in an immutable and transparent ledger. The feasibility of the novel architecture is validated through simulations, demonstrating improved performance in anomalous model detection and global model accuracy compared to relevant baselines.

{{</citation>}}


## cs.SE (5)



### (78/97) Demystifying Practices, Challenges and Expected Features of Using GitHub Copilot (Beiqi Zhang et al., 2023)

{{<citation>}}

Beiqi Zhang, Peng Liang, Xiyu Zhou, Aakash Ahmad, Muhammad Waseem. (2023)  
**Demystifying Practices, Challenges and Expected Features of Using GitHub Copilot**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.05687v1)  

---


**ABSTRACT**  
With the advances in machine learning, there is a growing interest in AI-enabled tools for autocompleting source code. GitHub Copilot has been trained on billions of lines of open source GitHub code, and is one of such tools that has been increasingly used since its launch in June 2021. However, little effort has been devoted to understanding the practices, challenges, and expected features of using Copilot in programming for auto-completed source code from the point of view of practitioners. To this end, we conducted an empirical study by collecting and analyzing the data from Stack Overflow (SO) and GitHub Discussions. We searched and manually collected 303 SO posts and 927 GitHub discussions related to the usage of Copilot. We identified the programming languages, Integrated Development Environments (IDEs), technologies used with Copilot, functions implemented, benefits, limitations, and challenges when using Copilot. The results show that when practitioners use Copilot: (1) The major programming languages used with Copilot are JavaScript and Python, (2) the main IDE used with Copilot is Visual Studio Code, (3) the most common used technology with Copilot is Node.js, (4) the leading function implemented by Copilot is data processing, (5) the main purpose of users using Copilot is to help generate code, (6) the significant benefit of using Copilot is useful code generation, (7) the main limitation encountered by practitioners when using Copilot is difficulty of integration, and (8) the most common expected feature is that Copilot can be integrated with more IDEs. Our results suggest that using Copilot is like a double-edged sword, which requires developers to carefully consider various aspects when deciding whether or not to use it. Our study provides empirically grounded foundations that could inform developers and practitioners, as well as provide a basis for future investigations.

{{</citation>}}


### (79/97) Kani: A Lightweight and Highly Hackable Framework for Building Language Model Applications (Andrew Zhu et al., 2023)

{{<citation>}}

Andrew Zhu, Liam Dugan, Alyssa Hwang, Chris Callison-Burch. (2023)  
**Kani: A Lightweight and Highly Hackable Framework for Building Language Model Applications**  

---
Primary Category: cs.SE  
Categories: I-2-7, cs-AI, cs-CL, cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.05542v1)  

---


**ABSTRACT**  
Language model applications are becoming increasingly popular and complex, often including features like tool usage and retrieval augmentation. However, existing frameworks for such applications are often opinionated, deciding for developers how their prompts ought to be formatted and imposing limitations on customizability and reproducibility. To solve this we present Kani: a lightweight, flexible, and model-agnostic open-source framework for building language model applications. Kani helps developers implement a variety of complex features by supporting the core building blocks of chat interaction: model interfacing, chat management, and robust function calling. All Kani core functions are easily overridable and well documented to empower developers to customize functionality for their own needs. Kani thus serves as a useful tool for researchers, hobbyists, and industry professionals alike to accelerate their development while retaining interoperability and fine-grained control.

{{</citation>}}


### (80/97) When ChatGPT Meets Smart Contract Vulnerability Detection: How Far Are We? (Chong Chen et al., 2023)

{{<citation>}}

Chong Chen, Jianzhong Su, Jiachi Chen, Yanlin Wang, Tingting Bi, Yanli Wang, Xingwei Lin, Ting Chen, Zibin Zheng. (2023)  
**When ChatGPT Meets Smart Contract Vulnerability Detection: How Far Are We?**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: ChatGPT, GPT, Vulnerability Detection  
[Paper Link](http://arxiv.org/abs/2309.05520v2)  

---


**ABSTRACT**  
With the development of blockchain technology, smart contracts have become an important component of blockchain applications. Despite their crucial role, the development of smart contracts may introduce vulnerabilities and potentially lead to severe consequences, such as financial losses. Meanwhile, large language models, represented by ChatGPT, have gained great attentions, showcasing great capabilities in code analysis tasks. In this paper, we presented an empirical study to investigate the performance of ChatGPT in identifying smart contract vulnerabilities. Initially, we evaluated ChatGPT's effectiveness using a publicly available smart contract dataset. Our findings discover that while ChatGPT achieves a high recall rate, its precision in pinpointing smart contract vulnerabilities is limited. Furthermore, ChatGPT's performance varies when detecting different vulnerability types. We delved into the root causes for the false positives generated by ChatGPT, and categorized them into four groups. Second, by comparing ChatGPT with other state-of-the-art smart contract vulnerability detection tools, we found that ChatGPT's F-score is lower than others for 3 out of the 7 vulnerabilities. In the case of the remaining 4 vulnerabilities, ChatGPT exhibits a slight advantage over these tools. Finally, we analyzed the limitation of ChatGPT in smart contract vulnerability detection, revealing that the robustness of ChatGPT in this field needs to be improved from two aspects: its uncertainty in answering questions; and the limited length of the detected code. In general, our research provides insights into the strengths and weaknesses of employing large language models, specifically ChatGPT, for the detection of smart contract vulnerabilities.

{{</citation>}}


### (81/97) Incentive-Based Software Security: Fair Micro-Payments for Writing Secure Code (Stefan Rass et al., 2023)

{{<citation>}}

Stefan Rass, Martin Pinzger. (2023)  
**Incentive-Based Software Security: Fair Micro-Payments for Writing Secure Code**  

---
Primary Category: cs.SE  
Categories: cs-GT, cs-SE, cs.SE  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2309.05338v1)  

---


**ABSTRACT**  
We describe a mechanism to create fair and explainable incentives for software developers to reward contributions to security of a product. We use cooperative game theory to model the actions of the developer team inside a risk management workflow, considering the team to actively work against known threats, and thereby receive micro-payments based on their performance. The use of the Shapley-value provides natural explanations here directly through (new) interpretations of the axiomatic grounding of the imputation. The resulting mechanism is straightforward to implement, and relies on standard tools from collaborative software development, such as are available for git repositories and mining thereof. The micropayment model itself is deterministic and does not rely on uncertain information outside the scope of the developer team or the enterprise, hence is void of assumptions about adversarial incentives, or user behavior, up to their role in the risk management process that the mechanism is part of. We corroborate our model with a worked example based on real-life data.

{{</citation>}}


### (82/97) Enabling Runtime Verification of Causal Discovery Algorithms with Automated Conditional Independence Reasoning (Extended Version) (Pingchuan Ma et al., 2023)

{{<citation>}}

Pingchuan Ma, Zhenlan Ji, Peisen Yao, Shuai Wang, Kui Ren. (2023)  
**Enabling Runtime Verification of Causal Discovery Algorithms with Automated Conditional Independence Reasoning (Extended Version)**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2309.05264v1)  

---


**ABSTRACT**  
Causal discovery is a powerful technique for identifying causal relationships among variables in data. It has been widely used in various applications in software engineering. Causal discovery extensively involves conditional independence (CI) tests. Hence, its output quality highly depends on the performance of CI tests, which can often be unreliable in practice. Moreover, privacy concerns arise when excessive CI tests are performed.   Despite the distinct nature between unreliable and excessive CI tests, this paper identifies a unified and principled approach to addressing both of them. Generally, CI statements, the outputs of CI tests, adhere to Pearl's axioms, which are a set of well-established integrity constraints on conditional independence. Hence, we can either detect erroneous CI statements if they violate Pearl's axioms or prune excessive CI statements if they are logically entailed by Pearl's axioms. Holistically, both problems boil down to reasoning about the consistency of CI statements under Pearl's axioms (referred to as CIR problem).   We propose a runtime verification tool called CICheck, designed to harden causal discovery algorithms from reliability and privacy perspectives. CICheck employs a sound and decidable encoding scheme that translates CIR into SMT problems. To solve the CIR problem efficiently, CICheck introduces a four-stage decision procedure with three lightweight optimizations that actively prove or refute consistency, and only resort to costly SMT-based reasoning when necessary. Based on the decision procedure to CIR, CICheck includes two variants: ED-CICheck and ED-CICheck, which detect erroneous CI tests (to enhance reliability) and prune excessive CI tests (to enhance privacy), respectively. [abridged due to length limit]

{{</citation>}}


## cs.SI (2)



### (83/97) Quantitative Analysis of Forecasting Models:In the Aspect of Online Political Bias (Srinath Sai Tripuraneni et al., 2023)

{{<citation>}}

Srinath Sai Tripuraneni, Sadia Kamal, Arunkumar Bagavathi. (2023)  
**Quantitative Analysis of Forecasting Models:In the Aspect of Online Political Bias**  

---
Primary Category: cs.SI  
Categories: cs-CY, cs-LG, cs-SI, cs.SI  
Keywords: Bias, Twitter  
[Paper Link](http://arxiv.org/abs/2309.05589v1)  

---


**ABSTRACT**  
Understanding and mitigating political bias in online social media platforms are crucial tasks to combat misinformation and echo chamber effects. However, characterizing political bias temporally using computational methods presents challenges due to the high frequency of noise in social media datasets. While existing research has explored various approaches to political bias characterization, the ability to forecast political bias and anticipate how political conversations might evolve in the near future has not been extensively studied. In this paper, we propose a heuristic approach to classify social media posts into five distinct political leaning categories. Since there is a lack of prior work on forecasting political bias, we conduct an in-depth analysis of existing baseline models to identify which model best fits to forecast political leaning time series. Our approach involves utilizing existing time series forecasting models on two social media datasets with different political ideologies, specifically Twitter and Gab. Through our experiments and analyses, we seek to shed light on the challenges and opportunities in forecasting political bias in social media platforms. Ultimately, our work aims to pave the way for developing more effective strategies to mitigate the negative impact of political bias in the digital realm.

{{</citation>}}


### (84/97) Circle Feature Graphormer: Can Circle Features Stimulate Graph Transformer? (Jingsong Lv et al., 2023)

{{<citation>}}

Jingsong Lv, Hongyang Chen, Yao Qi, Lei Yu. (2023)  
**Circle Feature Graphormer: Can Circle Features Stimulate Graph Transformer?**  

---
Primary Category: cs.SI  
Categories: cs-AI, cs-LG, cs-SI, cs.SI  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.06574v1)  

---


**ABSTRACT**  
In this paper, we introduce two local graph features for missing link prediction tasks on ogbl-citation2. We define the features as Circle Features, which are borrowed from the concept of circle of friends. We propose the detailed computing formulas for the above features. Firstly, we define the first circle feature as modified swing for common graph, which comes from bipartite graph. Secondly, we define the second circle feature as bridge, which indicates the importance of two nodes for different circle of friends. In addition, we firstly propose the above features as bias to enhance graph transformer neural network, such that graph self-attention mechanism can be improved. We implement a Circled Feature aware Graph transformer (CFG) model based on SIEG network, which utilizes a double tower structure to capture both global and local structure features. Experimental results show that CFG achieves the state-of-the-art performance on dataset ogbl-citation2.

{{</citation>}}


## cs.CE (1)



### (85/97) Unraveling Managerial Tangents in Firm Disclosure: Concealing Issues or Being Exposed? (Xuan Zhou et al., 2023)

{{<citation>}}

Xuan Zhou, Yushen Huang. (2023)  
**Unraveling Managerial Tangents in Firm Disclosure: Concealing Issues or Being Exposed?**  

---
Primary Category: cs.CE  
Categories: cs-CE, cs.CE  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2309.05555v1)  

---


**ABSTRACT**  
Earnings calls influence stock prices and are traditionally analyzed using sentiment and linguistic traces. Our research introduces a "Topic-Switching Index," a novel metric quantified through the transformer model FinBERT, to measure managerial evasion during Q$\&$A sessions in earnings calls. We find a negative correlation between this index and subsequent stock prices, indicating that investors penalize managerial evasiveness. This study is the first to quantify such evasive tactics, adding a new dimension to how earnings calls are understood and suggesting that topic shifting is an overlooked but significant factor. We also show the predictability of the index under three different classifier models and it stands out in all circumstances.

{{</citation>}}


## cs.AI (4)



### (86/97) On the meaning of uncertainty for ethical AI: philosophy and practice (Cassandra Bird et al., 2023)

{{<citation>}}

Cassandra Bird, Daniel Williamson, Sabina Leonelli. (2023)  
**On the meaning of uncertainty for ethical AI: philosophy and practice**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI, math-ST, stat-TH  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.05529v1)  

---


**ABSTRACT**  
Whether and how data scientists, statisticians and modellers should be accountable for the AI systems they develop remains a controversial and highly debated topic, especially given the complexity of AI systems and the difficulties in comparing and synthesising competing claims arising from their deployment for data analysis. This paper proposes to address this issue by decreasing the opacity and heightening the accountability of decision making using AI systems, through the explicit acknowledgement of the statistical foundations that underpin their development and the ways in which these dictate how their results should be interpreted and acted upon by users. In turn, this enhances (1) the responsiveness of the models to feedback, (2) the quality and meaning of uncertainty on their outputs and (3) their transparency to evaluation. To exemplify this approach, we extend Posterior Belief Assessment to offer a route to belief ownership from complex and competing AI structures. We argue that this is a significant way to bring ethical considerations into mathematical reasoning, and to implement ethical AI in statistical practice. We demonstrate these ideas within the context of competing models used to advise the UK government on the spread of the Omicron variant of COVID-19 during December 2021.

{{</citation>}}


### (87/97) NExT-GPT: Any-to-Any Multimodal LLM (Shengqiong Wu et al., 2023)

{{<citation>}}

Shengqiong Wu, Hao Fei, Leigang Qu, Wei Ji, Tat-Seng Chua. (2023)  
**NExT-GPT: Any-to-Any Multimodal LLM**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-LG, cs.AI  
Keywords: AI, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2309.05519v2)  

---


**ABSTRACT**  
While recently Multimodal Large Language Models (MM-LLMs) have made exciting strides, they mostly fall prey to the limitation of only input-side multimodal understanding, without the ability to produce content in multiple modalities. As we humans always perceive the world and communicate with people through various modalities, developing any-to-any MM-LLMs capable of accepting and delivering content in any modality becomes essential to human-level AI. To fill the gap, we present an end-to-end general-purpose any-to-any MM-LLM system, NExT-GPT. We connect an LLM with multimodal adaptors and different diffusion decoders, enabling NExT-GPT to perceive inputs and generate outputs in arbitrary combinations of text, images, videos, and audio. By leveraging the existing well-trained highly-performing encoders and decoders, NExT-GPT is tuned with only a small amount of parameter (1%) of certain projection layers, which not only benefits low-cost training and also facilitates convenient expansion to more potential modalities. Moreover, we introduce a modality-switching instruction tuning (MosIT) and manually curate a high-quality dataset for MosIT, based on which NExT-GPT is empowered with complex cross-modal semantic understanding and content generation. Overall, our research showcases the promising possibility of building an AI agent capable of modeling universal modalities, paving the way for more human-like AI research in the community. Project page: https://next-gpt.github.io/

{{</citation>}}


### (88/97) UniKG: A Benchmark and Universal Embedding for Large-Scale Knowledge Graphs (Yide Qiu et al., 2023)

{{<citation>}}

Yide Qiu, Shaoxiang Ling, Tong Zhang, Bo Huang, Zhen Cui. (2023)  
**UniKG: A Benchmark and Universal Embedding for Large-Scale Knowledge Graphs**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: Embedding, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2309.05269v1)  

---


**ABSTRACT**  
Irregular data in real-world are usually organized as heterogeneous graphs (HGs) consisting of multiple types of nodes and edges. To explore useful knowledge from real-world data, both the large-scale encyclopedic HG datasets and corresponding effective learning methods are crucial, but haven't been well investigated. In this paper, we construct a large-scale HG benchmark dataset named UniKG from Wikidata to facilitate knowledge mining and heterogeneous graph representation learning. Overall, UniKG contains more than 77 million multi-attribute entities and 2000 diverse association types, which significantly surpasses the scale of existing HG datasets. To perform effective learning on the large-scale UniKG, two key measures are taken, including (i) the semantic alignment strategy for multi-attribute entities, which projects the feature description of multi-attribute nodes into a common embedding space to facilitate node aggregation in a large receptive field; (ii) proposing a novel plug-and-play anisotropy propagation module (APM) to learn effective multi-hop anisotropy propagation kernels, which extends methods of large-scale homogeneous graphs to heterogeneous graphs. These two strategies enable efficient information propagation among a tremendous number of multi-attribute entities and meantimes adaptively mine multi-attribute association through the multi-hop aggregation in large-scale HGs. We set up a node classification task on our UniKG dataset, and evaluate multiple baseline methods which are constructed by embedding our APM into large-scale homogenous graph learning methods. Our UniKG dataset and the baseline codes have been released at https://github.com/Yide-Qiu/UniKG.

{{</citation>}}


### (89/97) Quantifying and Attributing the Hallucination of Large Language Models via Association Analysis (Li Du et al., 2023)

{{<citation>}}

Li Du, Yequan Wang, Xingrun Xing, Yiqun Ya, Xiang Li, Xin Jiang, Xuezhi Fang. (2023)  
**Quantifying and Attributing the Hallucination of Large Language Models via Association Analysis**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2309.05217v1)  

---


**ABSTRACT**  
Although demonstrating superb performance on various NLP tasks, large language models (LLMs) still suffer from the hallucination problem, which threatens the reliability of LLMs. To measure the level of hallucination of LLMs, previous works first categorize the hallucination according to the phenomenon similarity, then quantify the proportion that model outputs contain hallucinatory contents. However, such hallucination rates could easily be distorted by confounders. Moreover, such hallucination rates could not reflect the reasons for the hallucination, as similar hallucinatory phenomena may originate from different sources. To address these issues, we propose to combine the hallucination level quantification and hallucination reason investigation through an association analysis, which builds the relationship between the hallucination rate of LLMs with a set of risk factors. In this way, we are able to observe the hallucination level under each value of each risk factor, examining the contribution and statistical significance of each risk factor, meanwhile excluding the confounding effect of other factors. Additionally, by recognizing the risk factors according to a taxonomy of model capability, we reveal a set of potential deficiencies in commonsense memorization, relational reasoning, and instruction following, which may further provide guidance for the pretraining and supervised fine-tuning process of LLMs to mitigate the hallucination.

{{</citation>}}


## cs.HC (1)



### (90/97) A Co-design Study for Multi-Stakeholder Job Recommender System Explanations (Roan Schellingerhout et al., 2023)

{{<citation>}}

Roan Schellingerhout, Francesco Barile, Nava Tintarev. (2023)  
**A Co-design Study for Multi-Stakeholder Job Recommender System Explanations**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.05507v1)  

---


**ABSTRACT**  
Recent legislation proposals have significantly increased the demand for eXplainable Artificial Intelligence (XAI) in many businesses, especially in so-called `high-risk' domains, such as recruitment. Within recruitment, AI has become commonplace, mainly in the form of job recommender systems (JRSs), which try to match candidates to vacancies, and vice versa. However, common XAI techniques often fall short in this domain due to the different levels and types of expertise of the individuals involved, making explanations difficult to generalize. To determine the explanation preferences of the different stakeholder types - candidates, recruiters, and companies - we created and validated a semi-structured interview guide. Using grounded theory, we structurally analyzed the results of these interviews and found that different stakeholder types indeed have strongly differing explanation preferences. Candidates indicated a preference for brief, textual explanations that allow them to quickly judge potential matches. On the other hand, hiring managers preferred visual graph-based explanations that provide a more technical and comprehensive overview at a glance. Recruiters found more exhaustive textual explanations preferable, as those provided them with more talking points to convince both parties of the match. Based on these findings, we describe guidelines on how to design an explanation interface that fulfills the requirements of all three stakeholder types. Furthermore, we provide the validated interview guide, which can assist future research in determining the explanation preferences of different stakeholder types.

{{</citation>}}


## cs.DS (2)



### (91/97) CLAM-Accelerated K-Nearest Neighbors Entropy-Scaling Search of Large High-Dimensional Datasets via an Actualization of the Manifold Hypothesis (Morgan E. Prior et al., 2023)

{{<citation>}}

Morgan E. Prior, Thomas J. Howard III, Oliver McLaughlin, Najib Ishaq, Noah M. Daniels. (2023)  
**CLAM-Accelerated K-Nearest Neighbors Entropy-Scaling Search of Large High-Dimensional Datasets via an Actualization of the Manifold Hypothesis**  

---
Primary Category: cs.DS  
Categories: E-1; F-2-1; H-3-3, cs-DS, cs.DS  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.05491v1)  

---


**ABSTRACT**  
Many fields are experiencing a Big Data explosion, with data collection rates outpacing the rate of computing performance improvements predicted by Moore's Law.   Researchers are often interested in similarity search on such data.   We present CAKES (CLAM-Accelerated $K$-NN Entropy Scaling Search), a novel algorithm for $k$-nearest-neighbor ($k$-NN) search which leverages geometric and topological properties inherent in large datasets.   CAKES assumes the manifold hypothesis and performs best when data occupy a low dimensional manifold, even if the data occupy a very high dimensional embedding space.   We demonstrate performance improvements ranging from hundreds to tens of thousands of times faster when compared to state-of-the-art approaches such as FAISS and HNSW, when benchmarked on 5 standard datasets.   Unlike locality-sensitive hashing approaches, CAKES can work with any user-defined distance function.   When data occupy a metric space, CAKES exhibits perfect recall.

{{</citation>}}


### (92/97) Data Summarization beyond Monotonicity: Non-monotone Two-Stage Submodular Maximization (Shaojie Tang, 2023)

{{<citation>}}

Shaojie Tang. (2023)  
**Data Summarization beyond Monotonicity: Non-monotone Two-Stage Submodular Maximization**  

---
Primary Category: cs.DS  
Categories: cs-AI, cs-DS, cs-LG, cs.DS  
Keywords: Summarization  
[Paper Link](http://arxiv.org/abs/2309.05183v1)  

---


**ABSTRACT**  
The objective of a two-stage submodular maximization problem is to reduce the ground set using provided training functions that are submodular, with the aim of ensuring that optimizing new objective functions over the reduced ground set yields results comparable to those obtained over the original ground set. This problem has applications in various domains including data summarization. Existing studies often assume the monotonicity of the objective function, whereas our work pioneers the extension of this research to accommodate non-monotone submodular functions. We have introduced the first constant-factor approximation algorithms for this more general case.

{{</citation>}}


## eess.SY (1)



### (93/97) Assessing Wind Impact on Semi-Autonomous Drone Landings for In-Contact Power Line Inspection (Etienne Gendron et al., 2023)

{{<citation>}}

Etienne Gendron, Marc-Antoine Leclerc, Samuel Hovington, Etienne Perron, David Rancourt, Alexis Lussier-Desbiens, Philippe Hamelin, Alexandre Girard. (2023)  
**Assessing Wind Impact on Semi-Autonomous Drone Landings for In-Contact Power Line Inspection**  

---
Primary Category: eess.SY  
Categories: cs-RO, cs-SY, eess-SY, eess.SY  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2309.05467v1)  

---


**ABSTRACT**  
In recent years, the use of inspection drones has become increasingly popular for high-voltage electric cable inspections due to their efficiency, cost-effectiveness, and ability to access hard-to-reach areas. However, safely landing drones on power lines, especially under windy conditions, remains a significant challenge. This study introduces a semi-autonomous control scheme for landing on an electrical line with the NADILE drone (an experimental drone based on original LineDrone key features for inspection of power lines) and assesses the operating envelope under various wind conditions. A Monte Carlo method is employed to analyze the success probability of landing given initial drone states. The performance of the system is evaluated for two landing strategies, variously controllers parameters and four level of wind intensities. The results show that a two-stage landing strategies offers higher probabilities of landing success and give insight regarding the best controller parameters and the maximum wind level for which the system is robust. Lastly, an experimental demonstration of the system landing autonomously on a power line is presented.

{{</citation>}}


## eess.IV (1)



### (94/97) A Localization-to-Segmentation Framework for Automatic Tumor Segmentation in Whole-Body PET/CT Images (Linghan Cai et al., 2023)

{{<citation>}}

Linghan Cai, Jianhao Huang, Zihang Zhu, Jinpeng Lu, Yongbing Zhang. (2023)  
**A Localization-to-Segmentation Framework for Automatic Tumor Segmentation in Whole-Body PET/CT Images**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.05446v1)  

---


**ABSTRACT**  
Fluorodeoxyglucose (FDG) positron emission tomography(PET) combined with computed tomography (CT) is considered the primary solution for detecting some cancers, such as lung cancer and melanoma. Automatic segmentation of tumors in PET/CT images can help reduce doctors' workload, thereby improving diagnostic quality. However, precise tumor segmentation is challenging due to the small size of many tumors and the similarity of high-uptake normal areas to the tumor regions. To address these issues, this paper proposes a localization-to-segmentation framework (L2SNet) for precise tumor segmentation. L2SNet first localizes the possible lesions in the lesion localization phase and then uses the location cues to shape the segmentation results in the lesion segmentation phase. To further improve the segmentation performance of L2SNet, we design an adaptive threshold scheme that takes the segmentation results of the two phases into consideration. The experiments with the MICCAI 2023 Automated Lesion Segmentation in Whole-Body FDG-PET/CT challenge dataset show that our method achieved a competitive result and was ranked in the top 7 methods on the preliminary test set. Our work is available at: https://github.com/MedCAI/L2SNet.

{{</citation>}}


## cs.IT (1)



### (95/97) Low Peak-to-Average Power Ratio FBMC-OQAM System based on Data Mapping and DFT Precoding (Liming Li et al., 2023)

{{<citation>}}

Liming Li, Liqin Ding, Yang Wang, Jiliang Zhang. (2023)  
**Low Peak-to-Average Power Ratio FBMC-OQAM System based on Data Mapping and DFT Precoding**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, eess-SP, math-IT  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2309.05278v1)  

---


**ABSTRACT**  
Filter bank multicarrier with offset quadrature amplitude modulation (FBMC-OQAM) is an alternative to OFDM for enhanced spectrum flexible usage. To reduce the peak-to-average power ratio (PAPR), DFT spreading is usually adopted in OFDM systems. However, in FBMC-OQAM systems, because the OQAM pre-processing splits the spread data into the real and imaginary parts, the DFT spreading can result in only marginal PAPR reduction. This letter proposes a novel map-DFT-spread FBMC-OQAM scheme. In this scheme, the transmitting data symbols are first mapped with a conjugate symmetry rule and then coded by the DFT. According to this method, the OQAM pre-processing can be avoided. Compared with the simple DFT-spread scheme, the proposed scheme achieves a better PAPR reduction. In addition, the effect of the prototype filter on the PAPR is studied via numerical simulation and a trade-off exists between the PAPR and out-of-band performances.

{{</citation>}}


## eess.AS (1)



### (96/97) Enhancing Speaker Diarization with Large Language Models: A Contextual Beam Search Approach (Tae Jin Park et al., 2023)

{{<citation>}}

Tae Jin Park, Kunal Dhawan, Nithin Koluguri, Jagadeesh Balam. (2023)  
**Enhancing Speaker Diarization with Large Language Models: A Contextual Beam Search Approach**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.05248v2)  

---


**ABSTRACT**  
Large language models (LLMs) have shown great promise for capturing contextual information in natural language processing tasks. We propose a novel approach to speaker diarization that incorporates the prowess of LLMs to exploit contextual cues in human dialogues. Our method builds upon an acoustic-based speaker diarization system by adding lexical information from an LLM in the inference stage. We model the multi-modal decoding process probabilistically and perform joint acoustic and lexical beam search to incorporate cues from both modalities: audio and text. Our experiments demonstrate that infusing lexical knowledge from the LLM into an acoustics-only diarization system improves overall speaker-attributed word error rate (SA-WER). The experimental results show that LLMs can provide complementary information to acoustic models for the speaker diarization task via proposed beam search decoding approach showing up to 39.8% relative delta-SA-WER improvement from the baseline system. Thus, we substantiate that the proposed technique is able to exploit contextual information that is inaccessible to acoustics-only systems which is represented by speaker embeddings. In addition, these findings point to the potential of using LLMs to improve speaker diarization and other speech processing tasks by capturing semantic and contextual cues.

{{</citation>}}


## cs.IR (1)



### (97/97) Generating Natural Language Queries for More Effective Systematic Review Screening Prioritisation (Shuai Wang et al., 2023)

{{<citation>}}

Shuai Wang, Harrisen Scells, Martin Potthast, Bevan Koopman, Guido Zuccon. (2023)  
**Generating Natural Language Queries for More Effective Systematic Review Screening Prioritisation**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: BERT, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2309.05238v1)  

---


**ABSTRACT**  
Screening prioritisation in medical systematic reviews aims to rank the set of documents retrieved by complex Boolean queries. The goal is to prioritise the most important documents so that subsequent review steps can be carried out more efficiently and effectively. The current state of the art uses the final title of the review to rank documents using BERT-based neural neural rankers. However, the final title is only formulated at the end of the review process, which makes this approach impractical as it relies on ex post facto information. At the time of screening, only a rough working title is available, with which the BERT-based ranker achieves is significantly worse than the final title. In this paper, we explore alternative sources of queries for screening prioritisation, such as the Boolean query used to retrieve the set of documents to be screened, and queries generated by instruction-based generative large language models such as ChatGPT and Alpaca. Our best approach is not only practical based on the information available at screening time, but is similar in effectiveness with the final title.

{{</citation>}}
