---
draft: false
title: "arXiv @ 2023.07.12"
date: 2023-07-12
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.07.12"
    identifier: arxiv_20230712
    parent: 202307_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.LG (19)](#cslg-19)
- [cs.NI (2)](#csni-2)
- [cs.CV (18)](#cscv-18)
- [cs.DC (1)](#csdc-1)
- [cs.CY (3)](#cscy-3)
- [cs.CL (13)](#cscl-13)
- [cs.SE (5)](#csse-5)
- [eess.SP (1)](#eesssp-1)
- [cs.AI (11)](#csai-11)
- [cs.CR (2)](#cscr-2)
- [cs.HC (1)](#cshc-1)
- [stat.ML (1)](#statml-1)
- [cs.DB (1)](#csdb-1)
- [cs.RO (3)](#csro-3)
- [cs.IT (1)](#csit-1)
- [cs.SI (1)](#cssi-1)
- [cs.SD (3)](#cssd-3)
- [cs.IR (3)](#csir-3)
- [eess.IV (1)](#eessiv-1)
- [physics.comp-ph (1)](#physicscomp-ph-1)
- [cs.LO (1)](#cslo-1)
- [eess.AS (1)](#eessas-1)

## cs.LG (19)



### (1/93) Impact of Feature Encoding on Malware Classification Explainability (Elyes Manai et al., 2023)

{{<citation>}}

Elyes Manai, Mohamed Mejri, Jaouhar Fattahi. (2023)  
**Impact of Feature Encoding on Malware Classification Explainability**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.05614v1)  

---


**ABSTRACT**  
This paper investigates the impact of feature encoding techniques on the explainability of XAI (Explainable Artificial Intelligence) algorithms. Using a malware classification dataset, we trained an XGBoost model and compared the performance of two feature encoding methods: Label Encoding (LE) and One Hot Encoding (OHE). Our findings reveal a marginal performance loss when using OHE instead of LE. However, the more detailed explanations provided by OHE compensated for this loss. We observed that OHE enables deeper exploration of details in both global and local contexts, facilitating more comprehensive answers. Additionally, we observed that using OHE resulted in smaller explanation files and reduced analysis time for human analysts. These findings emphasize the significance of considering feature encoding techniques in XAI research and suggest potential for further exploration by incorporating additional encoding methods and innovative visualization approaches.

{{</citation>}}


### (2/93) Improving Fairness of Graph Neural Networks: A Graph Counterfactual Perspective (Zhimeng Guo et al., 2023)

{{<citation>}}

Zhimeng Guo, Jialiang Li, Teng Xiao, Yao Ma, Suhang Wang. (2023)  
**Improving Fairness of Graph Neural Networks: A Graph Counterfactual Perspective**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2307.04937v1)  

---


**ABSTRACT**  
Graph neural networks have shown great ability in representation (GNNs) learning on graphs, facilitating various tasks. Despite their great performance in modeling graphs, recent works show that GNNs tend to inherit and amplify the bias from training data, causing concerns of the adoption of GNNs in high-stake scenarios. Hence, many efforts have been taken for fairness-aware GNNs. However, most existing fair GNNs learn fair node representations by adopting statistical fairness notions, which may fail to alleviate bias in the presence of statistical anomalies. Motivated by causal theory, there are several attempts utilizing graph counterfactual fairness to mitigate root causes of unfairness. However, these methods suffer from non-realistic counterfactuals obtained by perturbation or generation. In this paper, we take a causal view on fair graph learning problem. Guided by the casual analysis, we propose a novel framework CAF, which can select counterfactuals from training data to avoid non-realistic counterfactuals and adopt selected counterfactuals to learn fair node representations for node classification task. Extensive experiments on synthetic and real-world datasets show the effectiveness of CAF.

{{</citation>}}


### (3/93) Substance or Style: What Does Your Image Embedding Know? (Cyrus Rashtchian et al., 2023)

{{<citation>}}

Cyrus Rashtchian, Charles Herrmann, Chun-Sung Ferng, Ayan Chakrabarti, Dilip Krishnan, Deqing Sun, Da-Cheng Juan, Andrew Tomkins. (2023)  
**Substance or Style: What Does Your Image Embedding Know?**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: Embedding, NLP  
[Paper Link](http://arxiv.org/abs/2307.05610v1)  

---


**ABSTRACT**  
Probes are small networks that predict properties of underlying data from embeddings, and they provide a targeted, effective way to illuminate the information contained in embeddings. While analysis through the use of probes has become standard in NLP, there has been much less exploration in vision. Image foundation models have primarily been evaluated for semantic content. Better understanding the non-semantic information in popular embeddings (e.g., MAE, SimCLR, or CLIP) will shed new light both on the training algorithms and on the uses for these foundation models. We design a systematic transformation prediction task and measure the visual content of embeddings along many axes, including image style, quality, and a range of natural and artificial transformations. Surprisingly, six embeddings (including SimCLR) encode enough non-semantic information to identify dozens of transformations. We also consider a generalization task, where we group similar transformations and hold out several for testing. We find that image-text models (CLIP and ALIGN) are better at recognizing new examples of style transfer than masking-based models (CAN and MAE). Overall, our results suggest that the choice of pre-training algorithm impacts the types of information in the embedding, and certain models are better than others for non-semantic downstream tasks.

{{</citation>}}


### (4/93) Probabilistic Counterexample Guidance for Safer Reinforcement Learning (Extended Version) (Xiaotong Ji et al., 2023)

{{<citation>}}

Xiaotong Ji, Antonio Filieri. (2023)  
**Probabilistic Counterexample Guidance for Safer Reinforcement Learning (Extended Version)**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-LO, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.04927v2)  

---


**ABSTRACT**  
Safe exploration aims at addressing the limitations of Reinforcement Learning (RL) in safety-critical scenarios, where failures during trial-and-error learning may incur high costs. Several methods exist to incorporate external knowledge or to use proximal sensor data to limit the exploration of unsafe states. However, reducing exploration risks in unknown environments, where an agent must discover safety threats during exploration, remains challenging. In this paper, we target the problem of safe exploration by guiding the training with counterexamples of the safety requirement. Our method abstracts both continuous and discrete state-space systems into compact abstract models representing the safety-relevant knowledge acquired by the agent during exploration. We then exploit probabilistic counterexample generation to construct minimal simulation submodels eliciting safety requirement violations, where the agent can efficiently train offline to refine its policy towards minimising the risk of safety violations during the subsequent online exploration. We demonstrate our method's effectiveness in reducing safety violations during online exploration in preliminary experiments by an average of 40.3% compared with QL and DQN standard algorithms and 29.1% compared with previous related work, while achieving comparable cumulative rewards with respect to unrestricted exploration and alternative approaches.

{{</citation>}}


### (5/93) FedYolo: Augmenting Federated Learning with Pretrained Transformers (Xuechen Zhang et al., 2023)

{{<citation>}}

Xuechen Zhang, Mingchen Li, Xiangyu Chang, Jiasi Chen, Amit K. Roy-Chowdhury, Ananda Theertha Suresh, Samet Oymak. (2023)  
**FedYolo: Augmenting Federated Learning with Pretrained Transformers**  

---
Primary Category: cs.LG  
Categories: cs-DC, cs-LG, cs.LG  
Keywords: Transformer, Transformers, Yolo  
[Paper Link](http://arxiv.org/abs/2307.04905v1)  

---


**ABSTRACT**  
The growth and diversity of machine learning applications motivate a rethinking of learning with mobile and edge devices. How can we address diverse client goals and learn with scarce heterogeneous data? While federated learning aims to address these issues, it has challenges hindering a unified solution. Large transformer models have been shown to work across a variety of tasks achieving remarkable few-shot adaptation. This raises the question: Can clients use a single general-purpose model, rather than custom models for each task, while obeying device and network constraints? In this work, we investigate pretrained transformers (PTF) to achieve these on-device learning goals and thoroughly explore the roles of model size and modularity, where the latter refers to adaptation through modules such as prompts or adapters. Focusing on federated learning, we demonstrate that: (1) Larger scale shrinks the accuracy gaps between alternative approaches and improves heterogeneity robustness. Scale allows clients to run more local SGD epochs which can significantly reduce the number of communication rounds. At the extreme, clients can achieve respectable accuracy locally highlighting the potential of fully-local learning. (2) Modularity, by design, enables $>$100$\times$ less communication in bits. Surprisingly, it also boosts the generalization capability of local adaptation methods and the robustness of smaller PTFs. Finally, it enables clients to solve multiple unrelated tasks simultaneously using a single PTF, whereas full updates are prone to catastrophic forgetting. These insights on scale and modularity motivate a new federated learning approach we call "You Only Load Once" (FedYolo): The clients load a full PTF model once and all future updates are accomplished through communication-efficient modules with limited catastrophic-forgetting, where each task is assigned to its own module.

{{</citation>}}


### (6/93) Measuring and Mitigating Interference in Reinforcement Learning (Vincent Liu et al., 2023)

{{<citation>}}

Vincent Liu, Han Wang, Ruo Yu Tao, Khurram Javed, Adam White, Martha White. (2023)  
**Measuring and Mitigating Interference in Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.04887v1)  

---


**ABSTRACT**  
Catastrophic interference is common in many network-based learning systems, and many proposals exist for mitigating it. Before overcoming interference we must understand it better. In this work, we provide a definition and novel measure of interference for value-based reinforcement learning methods such as Fitted Q-Iteration and DQN. We systematically evaluate our measure of interference, showing that it correlates with instability in control performance, across a variety of network architectures. Our new interference measure allows us to ask novel scientific questions about commonly used deep learning architectures and study learning algorithms which mitigate interference. Lastly, we outline a class of algorithms which we call online-aware that are designed to mitigate interference, and show they do reduce interference according to our measure and that they improve stability and performance in several classic control environments.

{{</citation>}}


### (7/93) Diffusion Policies for Out-of-Distribution Generalization in Offline Reinforcement Learning (Suzan Ece Ada et al., 2023)

{{<citation>}}

Suzan Ece Ada, Erhan Oztop, Emre Ugur. (2023)  
**Diffusion Policies for Out-of-Distribution Generalization in Offline Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-NE, cs-RO, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.04726v1)  

---


**ABSTRACT**  
Offline Reinforcement Learning (RL) methods leverage previous experiences to learn better policies than the behavior policy used for experience collection. In contrast to behavior cloning, which assumes the data is collected from expert demonstrations, offline RL can work with non-expert data and multimodal behavior policies. However, offline RL algorithms face challenges in handling distribution shifts and effectively representing policies due to the lack of online interaction during training. Prior work on offline RL uses conditional diffusion models to obtain expressive policies to represent multimodal behavior in the dataset. Nevertheless, they are not tailored toward alleviating the out-of-distribution state generalization. We introduce a novel method incorporating state reconstruction feature learning in the recent class of diffusion policies to address the out-of-distribution generalization problem. State reconstruction loss promotes more descriptive representation learning of states to alleviate the distribution shift incurred by the out-of-distribution states. We design a 2D Multimodal Contextual Bandit environment to demonstrate and evaluate our proposed model. We assess the performance of our model not only in this new environment but also on several D4RL benchmark tasks, achieving state-of-the-art results.

{{</citation>}}


### (8/93) On the power of graph neural networks and the role of the activation function (Sammy Khalife et al., 2023)

{{<citation>}}

Sammy Khalife, Amitabh Basu. (2023)  
**On the power of graph neural networks and the role of the activation function**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2307.04661v1)  

---


**ABSTRACT**  
In this article we present new results about the expressivity of Graph Neural Networks (GNNs). We prove that for any GNN with piecewise polynomial activations, whose architecture size does not grow with the graph input sizes, there exists a pair of non-isomorphic rooted trees of depth two such that the GNN cannot distinguish their root vertex up to an arbitrary number of iterations. The proof relies on tools from the algebra of symmetric polynomials. In contrast, it was already known that unbounded GNNs (those whose size is allowed to change with the graph sizes) with piecewise polynomial activations can distinguish these vertices in only two iterations. Our results imply a strict separation between bounded and unbounded size GNNs, answering an open question formulated by [Grohe, 2021]. We next prove that if one allows activations that are not piecewise polynomial, then in two iterations a single neuron perceptron can distinguish the root vertices of any pair of nonisomorphic trees of depth two (our results hold for activations like the sigmoid, hyperbolic tan and others). This shows how the power of graph neural networks can change drastically if one changes the activation function of the neural networks. The proof of this result utilizes the Lindemann-Weierstrauss theorem from transcendental number theory.

{{</citation>}}


### (9/93) Multimodal brain age estimation using interpretable adaptive population-graph learning (Kyriaki-Margarita Bintsi et al., 2023)

{{<citation>}}

Kyriaki-Margarita Bintsi, Vasileios Baltatzis, Rolandos Alexandros Potamias, Alexander Hammers, Daniel Rueckert. (2023)  
**Multimodal brain age estimation using interpretable adaptive population-graph learning**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Graph Convolutional Network  
[Paper Link](http://arxiv.org/abs/2307.04639v2)  

---


**ABSTRACT**  
Brain age estimation is clinically important as it can provide valuable information in the context of neurodegenerative diseases such as Alzheimer's. Population graphs, which include multimodal imaging information of the subjects along with the relationships among the population, have been used in literature along with Graph Convolutional Networks (GCNs) and have proved beneficial for a variety of medical imaging tasks. A population graph is usually static and constructed manually using non-imaging information. However, graph construction is not a trivial task and might significantly affect the performance of the GCN, which is inherently very sensitive to the graph structure. In this work, we propose a framework that learns a population graph structure optimized for the downstream task. An attention mechanism assigns weights to a set of imaging and non-imaging features (phenotypes), which are then used for edge extraction. The resulting graph is used to train the GCN. The entire pipeline can be trained end-to-end. Additionally, by visualizing the attention weights that were the most important for the graph construction, we increase the interpretability of the graph. We use the UK Biobank, which provides a large variety of neuroimaging and non-imaging phenotypes, to evaluate our method on brain age regression and classification. The proposed method outperforms competing static graph approaches and other state-of-the-art adaptive methods. We further show that the assigned attention scores indicate that there are both imaging and non-imaging phenotypes that are informative for brain age estimation and are in agreement with the relevant literature.

{{</citation>}}


### (10/93) QBitOpt: Fast and Accurate Bitwidth Reallocation during Training (Jorn Peters et al., 2023)

{{<citation>}}

Jorn Peters, Marios Fournarakis, Markus Nagel, Mart van Baalen, Tijmen Blankevoort. (2023)  
**QBitOpt: Fast and Accurate Bitwidth Reallocation during Training**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: ImageNet, QA  
[Paper Link](http://arxiv.org/abs/2307.04535v1)  

---


**ABSTRACT**  
Quantizing neural networks is one of the most effective methods for achieving efficient inference on mobile and embedded devices. In particular, mixed precision quantized (MPQ) networks, whose layers can be quantized to different bitwidths, achieve better task performance for the same resource constraint compared to networks with homogeneous bitwidths. However, finding the optimal bitwidth allocation is a challenging problem as the search space grows exponentially with the number of layers in the network. In this paper, we propose QBitOpt, a novel algorithm for updating bitwidths during quantization-aware training (QAT). We formulate the bitwidth allocation problem as a constraint optimization problem. By combining fast-to-compute sensitivities with efficient solvers during QAT, QBitOpt can produce mixed-precision networks with high task performance guaranteed to satisfy strict resource constraints. This contrasts with existing mixed-precision methods that learn bitwidths using gradients and cannot provide such guarantees. We evaluate QBitOpt on ImageNet and confirm that we outperform existing fixed and mixed-precision methods under average bitwidth constraints commonly found in the literature.

{{</citation>}}


### (11/93) Badgers: generating data quality deficits with Python (Julien Siebert et al., 2023)

{{<citation>}}

Julien Siebert, Daniel Seifert, Patricia Kelbert, Michael Kläs, Adam Trendowicz. (2023)  
**Badgers: generating data quality deficits with Python**  

---
Primary Category: cs.LG  
Categories: 68, D-m, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.04468v1)  

---


**ABSTRACT**  
Generating context specific data quality deficits is necessary to experimentally assess data quality of data-driven (artificial intelligence (AI) or machine learning (ML)) applications. In this paper we present badgers, an extensible open-source Python library to generate data quality deficits (outliers, imbalanced data, drift, etc.) for different modalities (tabular data, time-series, text, etc.). The documentation is accessible at https://fraunhofer-iese.github.io/badgers/ and the source code at https://github.com/Fraunhofer-IESE/badgers

{{</citation>}}


### (12/93) Multi-modal Graph Learning over UMLS Knowledge Graphs (Manuel Burger et al., 2023)

{{<citation>}}

Manuel Burger, Gunnar Rätsch, Rita Kuznetsova. (2023)  
**Multi-modal Graph Learning over UMLS Knowledge Graphs**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2307.04461v1)  

---


**ABSTRACT**  
Clinicians are increasingly looking towards machine learning to gain insights about patient evolutions. We propose a novel approach named Multi-Modal UMLS Graph Learning (MMUGL) for learning meaningful representations of medical concepts using graph neural networks over knowledge graphs based on the unified medical language system. These representations are aggregated to represent entire patient visits and then fed into a sequence model to perform predictions at the granularity of multiple hospital visits of a patient. We improve performance by incorporating prior medical knowledge and considering multiple modalities. We compare our method to existing architectures proposed to learn representations at different granularities on the MIMIC-III dataset and show that our approach outperforms these methods. The results demonstrate the significance of multi-modal medical concept representations based on prior medical knowledge.

{{</citation>}}


### (13/93) Formulating A Strategic Plan Based On Statistical Analyses And Applications For Financial Companies Through A Real-World Use Case (Saman Sarraf, 2023)

{{<citation>}}

Saman Sarraf. (2023)  
**Formulating A Strategic Plan Based On Statistical Analyses And Applications For Financial Companies Through A Real-World Use Case**  

---
Primary Category: cs.LG  
Categories: cs-CE, cs-LG, cs.LG  
Keywords: Financial  
[Paper Link](http://arxiv.org/abs/2307.04778v2)  

---


**ABSTRACT**  
Business statistics play a crucial role in implementing a data-driven strategic plan at the enterprise level to employ various analytics where the outcomes of such a plan enable an enterprise to enhance the decision-making process or to mitigate risks to the organization. In this work, a strategic plan informed by the statistical analysis is introduced for a financial company called LendingClub, where the plan is comprised of exploring the possibility of onboarding a big data platform along with advanced feature selection capacities. The main objectives of such a plan are to increase the company's revenue while reducing the risks of granting loans to borrowers who cannot return their loans. In this study, different hypotheses formulated to address the company's concerns are studied, where the results reveal that the amount of loans profoundly impacts the number of borrowers charging off their loans. Also, the proposed strategic plan includes onboarding advanced analytics such as machine learning technologies that allow the company to build better generalized data-driven predictive models.

{{</citation>}}


### (14/93) Policy Finetuning in Reinforcement Learning via Design of Experiments using Offline Data (Ruiqi Zhang et al., 2023)

{{<citation>}}

Ruiqi Zhang, Andrea Zanette. (2023)  
**Policy Finetuning in Reinforcement Learning via Design of Experiments using Offline Data**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.04354v1)  

---


**ABSTRACT**  
In some applications of reinforcement learning, a dataset of pre-collected experience is already available but it is also possible to acquire some additional online data to help improve the quality of the policy. However, it may be preferable to gather additional data with a single, non-reactive exploration policy and avoid the engineering costs associated with switching policies.   In this paper we propose an algorithm with provable guarantees that can leverage an offline dataset to design a single non-reactive policy for exploration. We theoretically analyze the algorithm and measure the quality of the final policy as a function of the local coverage of the original dataset and the amount of additional data collected.

{{</citation>}}


### (15/93) Continual Learning as Computationally Constrained Reinforcement Learning (Saurabh Kumar et al., 2023)

{{<citation>}}

Saurabh Kumar, Henrik Marklund, Ashish Rao, Yifan Zhu, Hong Jun Jeon, Yueyang Liu, Benjamin Van Roy. (2023)  
**Continual Learning as Computationally Constrained Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.04345v1)  

---


**ABSTRACT**  
An agent that efficiently accumulates knowledge to develop increasingly sophisticated skills over a long lifetime could advance the frontier of artificial intelligence capabilities. The design of such agents, which remains a long-standing challenge of artificial intelligence, is addressed by the subject of continual learning. This monograph clarifies and formalizes concepts of continual learning, introducing a framework and set of tools to stimulate further research.

{{</citation>}}


### (16/93) Privacy-Preserving Graph Machine Learning from Data to Computation: A Survey (Dongqi Fu et al., 2023)

{{<citation>}}

Dongqi Fu, Wenxuan Bao, Ross Maciejewski, Hanghang Tong, Jingrui He. (2023)  
**Privacy-Preserving Graph Machine Learning from Data to Computation: A Survey**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.04338v1)  

---


**ABSTRACT**  
In graph machine learning, data collection, sharing, and analysis often involve multiple parties, each of which may require varying levels of data security and privacy. To this end, preserving privacy is of great importance in protecting sensitive information. In the era of big data, the relationships among data entities have become unprecedentedly complex, and more applications utilize advanced data structures (i.e., graphs) that can support network structures and relevant attribute information. To date, many graph-based AI models have been proposed (e.g., graph neural networks) for various domain tasks, like computer vision and natural language processing. In this paper, we focus on reviewing privacy-preserving techniques of graph machine learning. We systematically review related works from the data to the computational aspects. We first review methods for generating privacy-preserving graph data. Then we describe methods for transmitting privacy-preserved information (e.g., graph model parameters) to realize the optimization-based computation when data sharing among multiple parties is risky or impossible. In addition to discussing relevant theoretical methodology and software tools, we also discuss current challenges and highlight several possible future research opportunities for privacy-preserving graph machine learning. Finally, we envision a unified and comprehensive secure graph machine learning system.

{{</citation>}}


### (17/93) Enhancing Adversarial Robustness via Score-Based Optimization (Boya Zhang et al., 2023)

{{<citation>}}

Boya Zhang, Weijian Luo, Zhihua Zhang. (2023)  
**Enhancing Adversarial Robustness via Score-Based Optimization**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CR, cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2307.04333v1)  

---


**ABSTRACT**  
Adversarial attacks have the potential to mislead deep neural network classifiers by introducing slight perturbations. Developing algorithms that can mitigate the effects of these attacks is crucial for ensuring the safe use of artificial intelligence. Recent studies have suggested that score-based diffusion models are effective in adversarial defenses. However, existing diffusion-based defenses rely on the sequential simulation of the reversed stochastic differential equations of diffusion models, which are computationally inefficient and yield suboptimal results. In this paper, we introduce a novel adversarial defense scheme named ScoreOpt, which optimizes adversarial samples at test-time, towards original clean data in the direction guided by score-based priors. We conduct comprehensive experiments on multiple datasets, including CIFAR10, CIFAR100 and ImageNet. Our experimental results demonstrate that our approach outperforms existing adversarial defenses in terms of both robustness performance and inference speed.

{{</citation>}}


### (18/93) CT-BERT: Learning Better Tabular Representations Through Cross-Table Pre-training (Chao Ye et al., 2023)

{{<citation>}}

Chao Ye, Guoshan Lu, Haobo Wang, Liyao Li, Sai Wu, Gang Chen, Junbo Zhao. (2023)  
**CT-BERT: Learning Better Tabular Representations Through Cross-Table Pre-training**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: BERT, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2307.04308v1)  

---


**ABSTRACT**  
Tabular data -- also known as structured data -- is one of the most common data forms in existence, thanks to the stable development and scaled deployment of database systems in the last few decades. At present however, despite the blast brought by large pre-trained models in other domains such as ChatGPT or SAM, how can we extract common knowledge across tables at a scale that may eventually lead to generalizable representation for tabular data remains a full blank. Indeed, there have been a few works around this topic. Most (if not all) of them are limited in the scope of a single table or fixed form of a schema. In this work, we first identify the crucial research challenges behind tabular data pre-training, particularly towards the cross-table scenario. We position the contribution of this work in two folds: (i)-we collect and curate nearly 2k high-quality tabular datasets, each of which is guaranteed to possess clear semantics, clean labels, and other necessary meta information. (ii)-we propose a novel framework that allows cross-table pre-training dubbed as CT-BERT. Noticeably, in light of pioneering the scaled cross-table training, CT-BERT is fully compatible with both supervised and self-supervised schemes, where the specific instantiation of CT-BERT is very much dependent on the downstream tasks. We further propose and implement a contrastive-learning-based and masked table modeling (MTM) objective into CT-BERT, that is inspired from computer vision and natural language processing communities but sophistically tailored to tables. The extensive empirical results on 15 datasets demonstrate CT-BERT's state-of-the-art performance, where both its supervised and self-supervised setups significantly outperform the prior approaches.

{{</citation>}}


### (19/93) Generalizing Graph ODE for Learning Complex System Dynamics across Environments (Zijie Huang et al., 2023)

{{<citation>}}

Zijie Huang, Yizhou Sun, Wei Wang. (2023)  
**Generalizing Graph ODE for Learning Complex System Dynamics across Environments**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CE, cs-LG, cs-MA, cs-NE, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2307.04287v1)  

---


**ABSTRACT**  
Learning multi-agent system dynamics has been extensively studied for various real-world applications, such as molecular dynamics in biology. Most of the existing models are built to learn single system dynamics from observed historical data and predict the future trajectory. In practice, however, we might observe multiple systems that are generated across different environments, which differ in latent exogenous factors such as temperature and gravity. One simple solution is to learn multiple environment-specific models, but it fails to exploit the potential commonalities among the dynamics across environments and offers poor prediction results where per-environment data is sparse or limited. Here, we present GG-ODE (Generalized Graph Ordinary Differential Equations), a machine learning framework for learning continuous multi-agent system dynamics across environments. Our model learns system dynamics using neural ordinary differential equations (ODE) parameterized by Graph Neural Networks (GNNs) to capture the continuous interaction among agents. We achieve the model generalization by assuming the dynamics across different environments are governed by common physics laws that can be captured via learning a shared ODE function. The distinct latent exogenous factors learned for each environment are incorporated into the ODE function to account for their differences. To improve model performance, we additionally design two regularization losses to (1) enforce the orthogonality between the learned initial states and exogenous factors via mutual information minimization; and (2) reduce the temporal variance of learned exogenous factors within the same system via contrastive learning. Experiments over various physical simulations show that our model can accurately predict system dynamics, especially in the long range, and can generalize well to new systems with few observations.

{{</citation>}}


## cs.NI (2)



### (20/93) Virtual Network Embedding without Explicit Virtual Network Specification (Jiangnan Cheng et al., 2023)

{{<citation>}}

Jiangnan Cheng, Yingjie Bi, Ao Tang. (2023)  
**Virtual Network Embedding without Explicit Virtual Network Specification**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2307.05609v1)  

---


**ABSTRACT**  
Network virtualization enables Internet service providers to run multiple heterogeneous and dedicated network architectures for different customers on a shared substrate. In existing works on virtual network embedding (VNE), each customer formulates a virtual network request (VNR) where a virtual network (VN) is required. Motivated by a concrete example where VN is not a proper VNR formulation to reflect the traffic demand of a customer, we propose a new VNR formulation described by the traffic demand between several access node pairs to complement the existing VNR formulation. Moreover, three different groups of VNE variants are systematically examined. Simulations demonstrate that shared channel embedding, as a new embedding variant under the proposed VNR formulation, improves the acceptance rate and reduces cost and link utility compared to traditional independent channel embedding.

{{</citation>}}


### (21/93) Practical Trustworthiness Model for DNN in Dedicated 6G Application (Anouar Nechi et al., 2023)

{{<citation>}}

Anouar Nechi, Ahmed Mahmoudi, Christoph Herold, Daniel Widmer, Thomas Kürner, Mladen Berekovic, Saleh Mulhem. (2023)  
**Practical Trustworthiness Model for DNN in Dedicated 6G Application**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI, eess-SP  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.04677v1)  

---


**ABSTRACT**  
Artificial intelligence (AI) is considered an efficient response to several challenges facing 6G technology. However, AI still suffers from a huge trust issue due to its ambiguous way of making predictions. Therefore, there is a need for a method to evaluate the AI's trustworthiness in practice for future 6G applications. This paper presents a practical model to analyze the trustworthiness of AI in a dedicated 6G application. In particular, we present two customized Deep Neural Networks (DNNs) to solve the Automatic Modulation Recognition (AMR) problem in Terahertz communications-based 6G technology. Then, a specific trustworthiness model and its attributes, namely data robustness, parameter sensitivity, and security covering adversarial examples, are introduced. The evaluation results indicate that the proposed trustworthiness attributes are crucial to evaluate the trustworthiness of DNN for this 6G application.

{{</citation>}}


## cs.CV (18)



### (22/93) Rapid Deforestation and Burned Area Detection using Deep Multimodal Learning on Satellite Imagery (Gabor Fodor et al., 2023)

{{<citation>}}

Gabor Fodor, Marcos V. Conde. (2023)  
**Rapid Deforestation and Burned Area Detection using Deep Multimodal Learning on Satellite Imagery**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: Amazon  
[Paper Link](http://arxiv.org/abs/2307.04916v1)  

---


**ABSTRACT**  
Deforestation estimation and fire detection in the Amazon forest poses a significant challenge due to the vast size of the area and the limited accessibility. However, these are crucial problems that lead to severe environmental consequences, including climate change, global warming, and biodiversity loss. To effectively address this problem, multimodal satellite imagery and remote sensing offer a promising solution for estimating deforestation and detecting wildfire in the Amazonia region. This research paper introduces a new curated dataset and a deep learning-based approach to solve these problems using convolutional neural networks (CNNs) and comprehensive data processing techniques. Our dataset includes curated images and diverse channel bands from Sentinel, Landsat, VIIRS, and MODIS satellites. We design the dataset considering different spatial and temporal resolution requirements. Our method successfully achieves high-precision deforestation estimation and burned area detection on unseen images from the region. Our code, models and dataset are open source: https://github.com/h2oai/cvpr-multiearth-deforestation-segmentation

{{</citation>}}


### (23/93) CREPE: Learnable Prompting With CLIP Improves Visual Relationship Prediction (Rakshith Subramanyam et al., 2023)

{{<citation>}}

Rakshith Subramanyam, T. S. Jayram, Rushil Anirudh, Jayaraman J. Thiagarajan. (2023)  
**CREPE: Learnable Prompting With CLIP Improves Visual Relationship Prediction**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.04838v2)  

---


**ABSTRACT**  
In this paper, we explore the potential of Vision-Language Models (VLMs), specifically CLIP, in predicting visual object relationships, which involves interpreting visual features from images into language-based relations. Current state-of-the-art methods use complex graphical models that utilize language cues and visual features to address this challenge. We hypothesize that the strong language priors in CLIP embeddings can simplify these graphical models paving for a simpler approach. We adopt the UVTransE relation prediction framework, which learns the relation as a translational embedding with subject, object, and union box embeddings from a scene. We systematically explore the design of CLIP-based subject, object, and union-box representations within the UVTransE framework and propose CREPE (CLIP Representation Enhanced Predicate Estimation). CREPE utilizes text-based representations for all three bounding boxes and introduces a novel contrastive training strategy to automatically infer the text prompt for union-box. Our approach achieves state-of-the-art performance in predicate estimation, mR@5 27.79, and mR@20 31.95 on the Visual Genome benchmark, achieving a 15.3\% gain in performance over recent state-of-the-art at mR@20. This work demonstrates CLIP's effectiveness in object relation prediction and encourages further research on VLMs in this challenging domain.

{{</citation>}}


### (24/93) SITTA: A Semantic Image-Text Alignment for Image Captioning (Fabian Paischer et al., 2023)

{{<citation>}}

Fabian Paischer, Thomas Adler, Markus Hofmarcher, Sepp Hochreiter. (2023)  
**SITTA: A Semantic Image-Text Alignment for Image Captioning**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs-LG, cs.CV  
Keywords: Image Captioning  
[Paper Link](http://arxiv.org/abs/2307.05591v1)  

---


**ABSTRACT**  
Textual and semantic comprehension of images is essential for generating proper captions. The comprehension requires detection of objects, modeling of relations between them, an assessment of the semantics of the scene and, finally, representing the extracted knowledge in a language space. To achieve rich language capabilities while ensuring good image-language mappings, pretrained language models (LMs) were conditioned on pretrained multi-modal (image-text) models that allow for image inputs. This requires an alignment of the image representation of the multi-modal model with the language representations of a generative LM. However, it is not clear how to best transfer semantics detected by the vision encoder of the multi-modal model to the LM. We introduce two novel ways of constructing a linear mapping that successfully transfers semantics between the embedding spaces of the two pretrained models. The first aligns the embedding space of the multi-modal language encoder with the embedding space of the pretrained LM via token correspondences. The latter leverages additional data that consists of image-text pairs to construct the mapping directly from vision to language space. Using our semantic mappings, we unlock image captioning for LMs without access to gradient information. By using different sources of data we achieve strong captioning performance on MS-COCO and Flickr30k datasets. Even in the face of limited data, our method partly exceeds the performance of other zero-shot and even finetuned competitors. Our ablation studies show that even LMs at a scale of merely 250M parameters can generate decent captions employing our semantic mappings. Our approach makes image captioning more accessible for institutions with restricted computational resources.

{{</citation>}}


### (25/93) Divide, Evaluate, and Refine: Evaluating and Improving Text-to-Image Alignment with Iterative VQA Feedback (Jaskirat Singh et al., 2023)

{{<citation>}}

Jaskirat Singh, Liang Zheng. (2023)  
**Divide, Evaluate, and Refine: Evaluating and Improving Text-to-Image Alignment with Iterative VQA Feedback**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs-MM, cs.CV, stat-ML  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2307.04749v1)  

---


**ABSTRACT**  
The field of text-conditioned image generation has made unparalleled progress with the recent advent of latent diffusion models. While remarkable, as the complexity of given text input increases, the state-of-the-art diffusion models may still fail in generating images which accurately convey the semantics of the given prompt. Furthermore, it has been observed that such misalignments are often left undetected by pretrained multi-modal models such as CLIP. To address these problems, in this paper we explore a simple yet effective decompositional approach towards both evaluation and improvement of text-to-image alignment. In particular, we first introduce a Decompositional-Alignment-Score which given a complex prompt decomposes it into a set of disjoint assertions. The alignment of each assertion with generated images is then measured using a VQA model. Finally, alignment scores for different assertions are combined aposteriori to give the final text-to-image alignment score. Experimental analysis reveals that the proposed alignment metric shows significantly higher correlation with human ratings as opposed to traditional CLIP, BLIP scores. Furthermore, we also find that the assertion level alignment scores provide a useful feedback which can then be used in a simple iterative procedure to gradually increase the expression of different assertions in the final image outputs. Human user studies indicate that the proposed approach surpasses previous state-of-the-art by 8.7% in overall text-to-image alignment accuracy. Project page for our paper is available at https://1jsingh.github.io/divide-evaluate-and-refine

{{</citation>}}


### (26/93) CVPR MultiEarth 2023 Deforestation Estimation Challenge:SpaceVision4Amazon (Sunita Arya et al., 2023)

{{<citation>}}

Sunita Arya, S Manthira Moorthi, Debajyoti Dhar. (2023)  
**CVPR MultiEarth 2023 Deforestation Estimation Challenge:SpaceVision4Amazon**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: Amazon  
[Paper Link](http://arxiv.org/abs/2307.04715v1)  

---


**ABSTRACT**  
In this paper, we present a deforestation estimation method based on attention guided UNet architecture using Electro-Optical (EO) and Synthetic Aperture Radar (SAR) satellite imagery. For optical images, Landsat-8 and for SAR imagery, Sentinel-1 data have been used to train and validate the proposed model. Due to the unavailability of temporally and spatially collocated data, individual model has been trained for each sensor. During training time Landsat-8 model achieved training and validation pixel accuracy of 93.45% and Sentinel-2 model achieved 83.87% pixel accuracy. During the test set evaluation, the model achieved pixel accuracy of 84.70% with F1-Score of 0.79 and IoU of 0.69.

{{</citation>}}


### (27/93) Joint Salient Object Detection and Camouflaged Object Detection via Uncertainty-aware Learning (Aixuan Li et al., 2023)

{{<citation>}}

Aixuan Li, Jing Zhang, Yunqiu Lv, Tong Zhang, Yiran Zhong, Mingyi He, Yuchao Dai. (2023)  
**Joint Salient Object Detection and Camouflaged Object Detection via Uncertainty-aware Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2307.04651v1)  

---


**ABSTRACT**  
Salient objects attract human attention and usually stand out clearly from their surroundings. In contrast, camouflaged objects share similar colors or textures with the environment. In this case, salient objects are typically non-camouflaged, and camouflaged objects are usually not salient. Due to this inherent contradictory attribute, we introduce an uncertainty-aware learning pipeline to extensively explore the contradictory information of salient object detection (SOD) and camouflaged object detection (COD) via data-level and task-wise contradiction modeling. We first exploit the dataset correlation of these two tasks and claim that the easy samples in the COD dataset can serve as hard samples for SOD to improve the robustness of the SOD model. Based on the assumption that these two models should lead to activation maps highlighting different regions of the same input image, we further introduce a contrastive module with a joint-task contrastive learning framework to explicitly model the contradictory attributes of these two tasks. Different from conventional intra-task contrastive learning for unsupervised representation learning, our contrastive module is designed to model the task-wise correlation, leading to cross-task representation learning. To better understand the two tasks from the perspective of uncertainty, we extensively investigate the uncertainty estimation techniques for modeling the main uncertainties of the two tasks, namely task uncertainty (for SOD) and data uncertainty (for COD), and aiming to effectively estimate the challenging regions for each task to achieve difficulty-aware learning. Experimental results on benchmark datasets demonstrate that our solution leads to both state-of-the-art performance and informative uncertainty estimation.

{{</citation>}}


### (28/93) Active Learning for Video Classification with Frame Level Queries (Debanjan Goswami et al., 2023)

{{<citation>}}

Debanjan Goswami, Shayok Chakraborty. (2023)  
**Active Learning for Video Classification with Frame Level Queries**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2307.05587v1)  

---


**ABSTRACT**  
Deep learning algorithms have pushed the boundaries of computer vision research and have depicted commendable performance in a variety of applications. However, training a robust deep neural network necessitates a large amount of labeled training data, acquiring which involves significant time and human effort. This problem is even more serious for an application like video classification, where a human annotator has to watch an entire video end-to-end to furnish a label. Active learning algorithms automatically identify the most informative samples from large amounts of unlabeled data; this tremendously reduces the human annotation effort in inducing a machine learning model, as only the few samples that are identified by the algorithm, need to be labeled manually. In this paper, we propose a novel active learning framework for video classification, with the goal of further reducing the labeling onus on the human annotators. Our framework identifies a batch of exemplar videos, together with a set of informative frames for each video; the human annotator needs to merely review the frames and provide a label for each video. This involves much less manual work than watching the complete video to come up with a label. We formulate a criterion based on uncertainty and diversity to identify the informative videos and exploit representative sampling techniques to extract a set of exemplar frames from each video. To the best of our knowledge, this is the first research effort to develop an active learning framework for video classification, where the annotators need to inspect only a few frames to produce a label, rather than watching the end-to-end video.

{{</citation>}}


### (29/93) Weakly-supervised positional contrastive learning: application to cirrhosis classification (Emma Sarfati et al., 2023)

{{<citation>}}

Emma Sarfati, Alexandre Bône, Marc-Michel Rohé, Pietro Gori, Isabelle Bloch. (2023)  
**Weakly-supervised positional contrastive learning: application to cirrhosis classification**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.04617v2)  

---


**ABSTRACT**  
Large medical imaging datasets can be cheaply and quickly annotated with low-confidence, weak labels (e.g., radiological scores). Access to high-confidence labels, such as histology-based diagnoses, is rare and costly. Pretraining strategies, like contrastive learning (CL) methods, can leverage unlabeled or weakly-annotated datasets. These methods typically require large batch sizes, which poses a difficulty in the case of large 3D images at full resolution, due to limited GPU memory. Nevertheless, volumetric positional information about the spatial context of each 2D slice can be very important for some medical applications. In this work, we propose an efficient weakly-supervised positional (WSP) contrastive learning strategy where we integrate both the spatial context of each 2D slice and a weak label via a generic kernel-based loss function. We illustrate our method on cirrhosis prediction using a large volume of weakly-labeled images, namely radiological low-confidence annotations, and small strongly-labeled (i.e., high-confidence) datasets. The proposed model improves the classification AUC by 5% with respect to a baseline model on our internal dataset, and by 26% on the public LIHC dataset from the Cancer Genome Atlas. The code is available at: https://github.com/Guerbet-AI/wsp-contrastive.

{{</citation>}}


### (30/93) MiVOLO: Multi-input Transformer for Age and Gender Estimation (Maksim Kuprashevich et al., 2023)

{{<citation>}}

Maksim Kuprashevich, Irina Tolstykh. (2023)  
**MiVOLO: Multi-input Transformer for Age and Gender Estimation**  

---
Primary Category: cs.CV  
Categories: I-2-0; I-4-0; I-4-9, cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.04616v1)  

---


**ABSTRACT**  
Age and gender recognition in the wild is a highly challenging task: apart from the variability of conditions, pose complexities, and varying image quality, there are cases where the face is partially or completely occluded. We present MiVOLO (Multi Input VOLO), a straightforward approach for age and gender estimation using the latest vision transformer. Our method integrates both tasks into a unified dual input/output model, leveraging not only facial information but also person image data. This improves the generalization ability of our model and enables it to deliver satisfactory results even when the face is not visible in the image. To evaluate our proposed model, we conduct experiments on four popular benchmarks and achieve state-of-the-art performance, while demonstrating real-time processing capabilities. Additionally, we introduce a novel benchmark based on images from the Open Images Dataset. The ground truth annotations for this benchmark have been meticulously generated by human annotators, resulting in high accuracy answers due to the smart aggregation of votes. Furthermore, we compare our model's age recognition performance with human-level accuracy and demonstrate that it significantly outperforms humans across a majority of age ranges. Finally, we grant public access to our models, along with the code for validation and inference. In addition, we provide extra annotations for used datasets and introduce our new benchmark.

{{</citation>}}


### (31/93) Source-Free Open-Set Domain Adaptation for Histopathological Images via Distilling Self-Supervised Vision Transformer (Guillaume Vray et al., 2023)

{{<citation>}}

Guillaume Vray, Devavrat Tomar, Behzad Bozorgtabar, Jean-Philippe Thiran. (2023)  
**Source-Free Open-Set Domain Adaptation for Histopathological Images via Distilling Self-Supervised Vision Transformer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised, Transformer  
[Paper Link](http://arxiv.org/abs/2307.04596v1)  

---


**ABSTRACT**  
There is a strong incentive to develop computational pathology models to i) ease the burden of tissue typology annotation from whole slide histological images; ii) transfer knowledge, e.g., tissue class separability from the withheld source domain to the distributionally shifted unlabeled target domain, and simultaneously iii) detect Open Set samples, i.e., unseen novel categories not present in the training source domain. This paper proposes a highly practical setting by addressing the abovementioned challenges in one fell swoop, i.e., source-free Open Set domain adaptation (SF-OSDA), which addresses the situation where a model pre-trained on the inaccessible source dataset can be adapted on the unlabeled target dataset containing Open Set samples. The central tenet of our proposed method is distilling knowledge from a self-supervised vision transformer trained in the target domain. We propose a novel style-based data augmentation used as hard positives for self-training a vision transformer in the target domain, yielding strongly contextualized embedding. Subsequently, semantically similar target images are clustered while the source model provides their corresponding weak pseudo-labels with unreliable confidence. Furthermore, we propose cluster relative maximum logit score (CRMLS) to rectify the confidence of the weak pseudo-labels and compute weighted class prototypes in the contextualized embedding space that are utilized for adapting the source model on the target domain. Our method significantly outperforms the previous methods, including open set detection, test-time adaptation, and SF-OSDA methods, setting the new state-of-the-art on three public histopathological datasets of colorectal cancer (CRC) assessment- Kather-16, Kather-19, and CRCTP. Our code is available at https://github.com/LTS5/Proto-SF-OSDA.

{{</citation>}}


### (32/93) SparseVSR: Lightweight and Noise Robust Visual Speech Recognition (Adriana Fernandez-Lopez et al., 2023)

{{<citation>}}

Adriana Fernandez-Lopez, Honglie Chen, Pingchuan Ma, Alexandros Haliassos, Stavros Petridis, Maja Pantic. (2023)  
**SparseVSR: Lightweight and Noise Robust Visual Speech Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2307.04552v1)  

---


**ABSTRACT**  
Recent advances in deep neural networks have achieved unprecedented success in visual speech recognition. However, there remains substantial disparity between current methods and their deployment in resource-constrained devices. In this work, we explore different magnitude-based pruning techniques to generate a lightweight model that achieves higher performance than its dense model equivalent, especially under the presence of visual noise. Our sparse models achieve state-of-the-art results at 10% sparsity on the LRS3 dataset and outperform the dense equivalent up to 70% sparsity. We evaluate our 50% sparse model on 7 different visual noise types and achieve an overall absolute improvement of more than 2% WER compared to the dense equivalent. Our results confirm that sparse networks are more resistant to noise than dense networks.

{{</citation>}}


### (33/93) Learning Large Margin Sparse Embeddings for Open Set Medical Diagnosis (Mingyuan Liu et al., 2023)

{{<citation>}}

Mingyuan Liu, Lu Xu, Jicong Zhang. (2023)  
**Learning Large Margin Sparse Embeddings for Open Set Medical Diagnosis**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2307.04541v1)  

---


**ABSTRACT**  
Fueled by deep learning, computer-aided diagnosis achieves huge advances. However, out of controlled lab environments, algorithms could face multiple challenges. Open set recognition (OSR), as an important one, states that categories unseen in training could appear in testing. In medical fields, it could derive from incompletely collected training datasets and the constantly emerging new or rare diseases. OSR requires an algorithm to not only correctly classify known classes, but also recognize unknown classes and forward them to experts for further diagnosis. To tackle OSR, we assume that known classes could densely occupy small parts of the embedding space and the remaining sparse regions could be recognized as unknowns. Following it, we propose Open Margin Cosine Loss (OMCL) unifying two mechanisms. The former, called Margin Loss with Adaptive Scale (MLAS), introduces angular margin for reinforcing intra-class compactness and inter-class separability, together with an adaptive scaling factor to strengthen the generalization capacity. The latter, called Open-Space Suppression (OSS), opens the classifier by recognizing sparse embedding space as unknowns using proposed feature space descriptors. Besides, since medical OSR is still a nascent field, two publicly available benchmark datasets are proposed for comparison. Extensive ablation studies and feature visualization demonstrate the effectiveness of each design. Compared with state-of-the-art methods, MLAS achieves superior performances, measured by ACC, AUROC, and OSCR.

{{</citation>}}


### (34/93) Q-YOLOP: Quantization-aware You Only Look Once for Panoptic Driving Perception (Chi-Chih Chang et al., 2023)

{{<citation>}}

Chi-Chih Chang, Wei-Cheng Lin, Pei-Shuo Wang, Sheng-Feng Yu, Yu-Chen Lu, Kuan-Cheng Lin, Kai-Chiang Wu. (2023)  
**Q-YOLOP: Quantization-aware You Only Look Once for Panoptic Driving Perception**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: QA, Quantization  
[Paper Link](http://arxiv.org/abs/2307.04537v1)  

---


**ABSTRACT**  
In this work, we present an efficient and quantization-aware panoptic driving perception model (Q- YOLOP) for object detection, drivable area segmentation, and lane line segmentation, in the context of autonomous driving. Our model employs the Efficient Layer Aggregation Network (ELAN) as its backbone and task-specific heads for each task. We employ a four-stage training process that includes pretraining on the BDD100K dataset, finetuning on both the BDD100K and iVS datasets, and quantization-aware training (QAT) on BDD100K. During the training process, we use powerful data augmentation techniques, such as random perspective and mosaic, and train the model on a combination of the BDD100K and iVS datasets. Both strategies enhance the model's generalization capabilities. The proposed model achieves state-of-the-art performance with an mAP@0.5 of 0.622 for object detection and an mIoU of 0.612 for segmentation, while maintaining low computational and memory requirements.

{{</citation>}}


### (35/93) Test-Time Adaptation for Nighttime Color-Thermal Semantic Segmentation (Yexin Liu et al., 2023)

{{<citation>}}

Yexin Liu, Weiming Zhang, Guoyang Zhao, Jinjing Zhu, Athanasios Vasilakos, Lin Wang. (2023)  
**Test-Time Adaptation for Nighttime Color-Thermal Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2307.04470v1)  

---


**ABSTRACT**  
The ability to scene understanding in adverse visual conditions, e.g., nighttime, has sparked active research for RGB-Thermal (RGB-T) semantic segmentation. However, it is essentially hampered by two critical problems: 1) the day-night gap of RGB images is larger than that of thermal images, and 2) the class-wise performance of RGB images at night is not consistently higher or lower than that of thermal images. we propose the first test-time adaptation (TTA) framework, dubbed Night-TTA, to address the problems for nighttime RGBT semantic segmentation without access to the source (daytime) data during adaptation. Our method enjoys three key technical parts. Firstly, as one modality (e.g., RGB) suffers from a larger domain gap than that of the other (e.g., thermal), Imaging Heterogeneity Refinement (IHR) employs an interaction branch on the basis of RGB and thermal branches to prevent cross-modal discrepancy and performance degradation. Then, Class Aware Refinement (CAR) is introduced to obtain reliable ensemble logits based on pixel-level distribution aggregation of the three branches. In addition, we also design a specific learning scheme for our TTA framework, which enables the ensemble logits and three student logits to collaboratively learn to improve the quality of predictions during the testing phase of our Night TTA. Extensive experiments show that our method achieves state-of-the-art (SoTA) performance with a 13.07% boost in mIoU.

{{</citation>}}


### (36/93) SAM-IQA: Can Segment Anything Boost Image Quality Assessment? (Xinpeng Li et al., 2023)

{{<citation>}}

Xinpeng Li, Ting Jiang, Haoqiang Fan, Shuaicheng Liu. (2023)  
**SAM-IQA: Can Segment Anything Boost Image Quality Assessment?**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: ImageNet, QA  
[Paper Link](http://arxiv.org/abs/2307.04455v1)  

---


**ABSTRACT**  
Image Quality Assessment (IQA) is a challenging task that requires training on massive datasets to achieve accurate predictions. However, due to the lack of IQA data, deep learning-based IQA methods typically rely on pre-trained networks trained on massive datasets as feature extractors to enhance their generalization ability, such as the ResNet network trained on ImageNet. In this paper, we utilize the encoder of Segment Anything, a recently proposed segmentation model trained on a massive dataset, for high-level semantic feature extraction. Most IQA methods are limited to extracting spatial-domain features, while frequency-domain features have been shown to better represent noise and blur. Therefore, we leverage both spatial-domain and frequency-domain features by applying Fourier and standard convolutions on the extracted features, respectively. Extensive experiments are conducted to demonstrate the effectiveness of all the proposed components, and results show that our approach outperforms the state-of-the-art (SOTA) in four representative datasets, both qualitatively and quantitatively. Our experiments confirm the powerful feature extraction capabilities of Segment Anything and highlight the value of combining spatial-domain and frequency-domain features in IQA tasks. Code: https://github.com/Hedlen/SAM-IQA

{{</citation>}}


### (37/93) Automatic diagnosis of knee osteoarthritis severity using Swin transformer (Aymen Sekhri et al., 2023)

{{<citation>}}

Aymen Sekhri, Marouane Tliba, Mohamed Amine Kerkouri, Yassine Nasser, Aladine Chetouani, Alessandro Bruno, Rachid Jennane. (2023)  
**Automatic diagnosis of knee osteoarthritis severity using Swin transformer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.04442v1)  

---


**ABSTRACT**  
Knee osteoarthritis (KOA) is a widespread condition that can cause chronic pain and stiffness in the knee joint. Early detection and diagnosis are crucial for successful clinical intervention and management to prevent severe complications, such as loss of mobility. In this paper, we propose an automated approach that employs the Swin Transformer to predict the severity of KOA. Our model uses publicly available radiographic datasets with Kellgren and Lawrence scores to enable early detection and severity assessment. To improve the accuracy of our model, we employ a multi-prediction head architecture that utilizes multi-layer perceptron classifiers. Additionally, we introduce a novel training approach that reduces the data drift between multiple datasets to ensure the generalization ability of the model. The results of our experiments demonstrate the effectiveness and feasibility of our approach in predicting KOA severity accurately.

{{</citation>}}


### (38/93) One-Shot Pruning for Fast-adapting Pre-trained Models on Devices (Haiyan Zhao et al., 2023)

{{<citation>}}

Haiyan Zhao, Guodong Long. (2023)  
**One-Shot Pruning for Fast-adapting Pre-trained Models on Devices**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2307.04365v1)  

---


**ABSTRACT**  
Large-scale pre-trained models have been remarkably successful in resolving downstream tasks. Nonetheless, deploying these models on low-capability devices still requires an effective approach, such as model pruning. However, pruning the model from scratch can pose a practical challenge given the limited resources of each downstream task or device. To tackle this issue, we present a scalable one-shot pruning method that leverages pruned knowledge of similar tasks to extract a sub-network from the pre-trained model for a new task. Specifically, we create a score mask using the pruned models of similar tasks to identify task-specific filters/nodes in the pre-trained model for the new task. Based on this mask, we conduct a single round of pruning to extract a suitably-sized sub-network that can quickly adapt to the new task with only a few training iterations. Our experimental analysis demonstrates the effectiveness of the proposed method on the convolutional neural networks (CNNs) and vision transformers (ViT) with various datasets. The proposed method consistently outperforms popular pruning baseline methods in terms of accuracy and efficiency when dealing with diverse downstream tasks with different memory constraints.

{{</citation>}}


### (39/93) Hierarchical Semantic Tree Concept Whitening for Interpretable Image Classification (Haixing Dai et al., 2023)

{{<citation>}}

Haixing Dai, Lu Zhang, Lin Zhao, Zihao Wu, Zhengliang Liu, David Liu, Xiaowei Yu, Yanjun Lyu, Changying Li, Ninghao Liu, Tianming Liu, Dajiang Zhu. (2023)  
**Hierarchical Semantic Tree Concept Whitening for Interpretable Image Classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2307.04343v1)  

---


**ABSTRACT**  
With the popularity of deep neural networks (DNNs), model interpretability is becoming a critical concern. Many approaches have been developed to tackle the problem through post-hoc analysis, such as explaining how predictions are made or understanding the meaning of neurons in middle layers. Nevertheless, these methods can only discover the patterns or rules that naturally exist in models. In this work, rather than relying on post-hoc schemes, we proactively instill knowledge to alter the representation of human-understandable concepts in hidden layers. Specifically, we use a hierarchical tree of semantic concepts to store the knowledge, which is leveraged to regularize the representations of image data instances while training deep models. The axes of the latent space are aligned with the semantic concepts, where the hierarchical relations between concepts are also preserved. Experiments on real-world image datasets show that our method improves model interpretability, showing better disentanglement of semantic concepts, without negatively affecting model classification performance.

{{</citation>}}


## cs.DC (1)



### (40/93) Coded Distributed Image Classification (Jiepeng Tang et al., 2023)

{{<citation>}}

Jiepeng Tang, Navneet Agrawal, Slawomir Stanczak, Jingge Zhu. (2023)  
**Coded Distributed Image Classification**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2307.04915v1)  

---


**ABSTRACT**  
In this paper, we present a coded computation (CC) scheme for distributed computation of the inference phase of machine learning (ML) tasks, specifically, the task of image classification. Building upon Agrawal et al.~2022, the proposed scheme combines the strengths of deep learning and Lagrange interpolation technique to mitigate the effect of straggling workers, and recovers approximate results with reasonable accuracy using outputs from any $R$ out of $N$ workers, where $R\leq N$. Our proposed scheme guarantees a minimum recovery threshold $R$ for non-polynomial problems, which can be adjusted as a tunable parameter in the system. Moreover, unlike existing schemes, our scheme maintains flexibility with respect to worker availability and system design. We propose two system designs for our CC scheme that allows flexibility in distributing the computational load between the master and the workers based on the accessibility of input data. Our experimental results demonstrate the superiority of our scheme compared to the state-of-the-art CC schemes for image classification tasks, and pave the path for designing new schemes for distributed computation of any general ML classification tasks.

{{</citation>}}


## cs.CY (3)



### (41/93) Self-Diagnosis and Large Language Models: A New Front for Medical Misinformation (Francois Barnard et al., 2023)

{{<citation>}}

Francois Barnard, Marlize Van Sittert, Sirisha Rambhatla. (2023)  
**Self-Diagnosis and Large Language Models: A New Front for Medical Misinformation**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.04910v1)  

---


**ABSTRACT**  
Improving healthcare quality and access remains a critical concern for countries worldwide. Consequently, the rise of large language models (LLMs) has erupted a wealth of discussion around healthcare applications among researchers and consumers alike. While the ability of these models to pass medical exams has been used to argue in favour of their use in medical training and diagnosis, the impact of their inevitable use as a self-diagnostic tool and their role in spreading healthcare misinformation has not been evaluated. In this work, we critically evaluate LLMs' capabilities from the lens of a general user self-diagnosing, as well as the means through which LLMs may aid in the spread of medical misinformation. To accomplish this, we develop a testing methodology which can be used to evaluate responses to open-ended questions mimicking real-world use cases. In doing so, we reveal that a) these models perform worse than previously known, and b) they exhibit peculiar behaviours, including overconfidence when stating incorrect recommendations, which increases the risk of spreading medical misinformation.

{{</citation>}}


### (42/93) International Institutions for Advanced AI (Lewis Ho et al., 2023)

{{<citation>}}

Lewis Ho, Joslyn Barnhart, Robert Trager, Yoshua Bengio, Miles Brundage, Allison Carnegie, Rumman Chowdhury, Allan Dafoe, Gillian Hadfield, Margaret Levi, Duncan Snidal. (2023)  
**International Institutions for Advanced AI**  

---
Primary Category: cs.CY  
Categories: K-4-1, cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.04699v2)  

---


**ABSTRACT**  
International institutions may have an important role to play in ensuring advanced AI systems benefit humanity. International collaborations can unlock AI's ability to further sustainable development, and coordination of regulatory efforts can reduce obstacles to innovation and the spread of benefits. Conversely, the potential dangerous capabilities of powerful and general-purpose AI systems create global externalities in their development and deployment, and international efforts to further responsible AI practices could help manage the risks they pose. This paper identifies a set of governance functions that could be performed at an international level to address these challenges, ranging from supporting access to frontier AI systems to setting international safety standards. It groups these functions into four institutional models that exhibit internal synergies and have precedents in existing organizations: 1) a Commission on Frontier AI that facilitates expert consensus on opportunities and risks from advanced AI, 2) an Advanced AI Governance Organization that sets international standards to manage global threats from advanced models, supports their implementation, and possibly monitors compliance with a future governance regime, 3) a Frontier AI Collaborative that promotes access to cutting-edge AI, and 4) an AI Safety Project that brings together leading researchers and engineers to further AI safety research. We explore the utility of these models and identify open questions about their viability.

{{</citation>}}


### (43/93) Demonstrations of the Potential of AI-based Political Issue Polling (Nathan E. Sanders et al., 2023)

{{<citation>}}

Nathan E. Sanders, Alex Ulinich, Bruce Schneier. (2023)  
**Demonstrations of the Potential of AI-based Political Issue Polling**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2307.04781v1)  

---


**ABSTRACT**  
Political polling is a multi-billion dollar industry with outsized influence on the societal trajectory of the United States and nations around the world. However, it has been challenged by factors that stress its cost, availability, and accuracy. At the same time, artificial intelligence (AI) chatbots have become compelling stand-ins for human behavior, powered by increasingly sophisticated large language models (LLMs). Could AI chatbots be an effective tool for anticipating public opinion on controversial issues to the extent that they could be used by campaigns, interest groups, and polling firms? We have developed a prompt engineering methodology for eliciting human-like survey responses from ChatGPT, which simulate the response to a policy question of a person described by a set of demographic factors, and produce both an ordinal numeric response score and a textual justification. We execute large scale experiments, querying for thousands of simulated responses at a cost far lower than human surveys. We compare simulated data to human issue polling data from the Cooperative Election Study (CES). We find that ChatGPT is effective at anticipating both the mean level and distribution of public opinion on a variety of policy issues such as abortion bans and approval of the US Supreme Court, particularly in their ideological breakdown (correlation typically >85%). However, it is less successful at anticipating demographic-level differences. Moreover, ChatGPT tends to overgeneralize to new policy issues that arose after its training data was collected, such as US support for involvement in the war in Ukraine. Our work has implications for our understanding of the strengths and limitations of the current generation of AI chatbots as virtual publics or online listening platforms, future directions for LLM development, and applications of AI tools to the political domain. (Abridged)

{{</citation>}}


## cs.CL (13)



### (44/93) KU-DMIS-MSRA at RadSum23: Pre-trained Vision-Language Model for Radiology Report Summarization (Gangwoo Kim et al., 2023)

{{<citation>}}

Gangwoo Kim, Hajung Kim, Lei Ji, Seongsu Bae, Chanhwi Kim, Mujeen Sung, Hyunjae Kim, Kun Yan, Eric Chang, Jaewoo Kang. (2023)  
**KU-DMIS-MSRA at RadSum23: Pre-trained Vision-Language Model for Radiology Report Summarization**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL, eess-IV  
Keywords: Language Model, NLP, Summarization  
[Paper Link](http://arxiv.org/abs/2307.07409v1)  

---


**ABSTRACT**  
In this paper, we introduce CheXOFA, a new pre-trained vision-language model (VLM) for the chest X-ray domain. Our model is initially pre-trained on various multimodal datasets within the general domain before being transferred to the chest X-ray domain. Following a prominent VLM, we unify various domain-specific tasks into a simple sequence-to-sequence schema. It enables the model to effectively learn the required knowledge and skills from limited resources in the domain. Demonstrating superior performance on the benchmark datasets provided by the BioNLP shared task, our model benefits from its training across multiple tasks and domains. With subtle techniques including ensemble and factual calibration, our system achieves first place on the RadSum23 leaderboard for the hidden test set.

{{</citation>}}


### (45/93) SimpleMTOD: A Simple Language Model for Multimodal Task-Oriented Dialogue with Symbolic Scene Representation (Bhathiya Hemanthage et al., 2023)

{{<citation>}}

Bhathiya Hemanthage, Christian Dondrup, Phil Bartie, Oliver Lemon. (2023)  
**SimpleMTOD: A Simple Language Model for Multimodal Task-Oriented Dialogue with Symbolic Scene Representation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: BLEU, Dialog, Dialogue, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2307.04907v1)  

---


**ABSTRACT**  
SimpleMTOD is a simple language model which recasts several sub-tasks in multimodal task-oriented dialogues as sequence prediction tasks. SimpleMTOD is built on a large-scale transformer-based auto-regressive architecture, which has already proven to be successful in uni-modal task-oriented dialogues, and effectively leverages transfer learning from pre-trained GPT-2. In-order to capture the semantics of visual scenes, we introduce both local and de-localized tokens for objects within a scene. De-localized tokens represent the type of an object rather than the specific object itself and so possess a consistent meaning across the dataset. SimpleMTOD achieves a state-of-the-art BLEU score (0.327) in the Response Generation sub-task of the SIMMC 2.0 test-std dataset while performing on par in other multimodal sub-tasks: Disambiguation, Coreference Resolution, and Dialog State Tracking. This is despite taking a minimalist approach for extracting visual (and non-visual) information. In addition the model does not rely on task-specific architectural changes such as classification heads.

{{</citation>}}


### (46/93) Entity Identifier: A Natural Text Parsing-based Framework For Entity Relation Extraction (El Mehdi Chouham et al., 2023)

{{<citation>}}

El Mehdi Chouham, Jessica López Espejel, Mahaman Sanoussi Yahaya Alassan, Walid Dahhane, El Hassane Ettifouri. (2023)  
**Entity Identifier: A Natural Text Parsing-based Framework For Entity Relation Extraction**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Relation Extraction  
[Paper Link](http://arxiv.org/abs/2307.04892v1)  

---


**ABSTRACT**  
The field of programming has a diversity of paradigms that are used according to the working framework. While current neural code generation methods are able to learn and generate code directly from text, we believe that this approach is not optimal for certain code tasks, particularly the generation of classes in an object-oriented project. Specifically, we use natural language processing techniques to extract structured information from requirements descriptions, in order to automate the generation of CRUD (Create, Read, Update, Delete) class code. To facilitate this process, we introduce a pipeline for extracting entity and relation information, as well as a representation called an "Entity Tree" to model this information. We also create a dataset to evaluate the effectiveness of our approach.

{{</citation>}}


### (47/93) BeaverTails: Towards Improved Safety Alignment of LLM via a Human-Preference Dataset (Jiaming Ji et al., 2023)

{{<citation>}}

Jiaming Ji, Mickel Liu, Juntao Dai, Xuehai Pan, Chi Zhang, Ce Bian, Chi Zhang, Ruiyang Sun, Yizhou Wang, Yaodong Yang. (2023)  
**BeaverTails: Towards Improved Safety Alignment of LLM via a Human-Preference Dataset**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2307.04657v1)  

---


**ABSTRACT**  
In this paper, we introduce the BeaverTails dataset, aimed at fostering research on safety alignment in large language models (LLMs). This dataset uniquely separates annotations of helpfulness and harmlessness for question-answering pairs, thus offering distinct perspectives on these crucial attributes. In total, we have compiled safety meta-labels for 30,207 question-answer (QA) pairs and gathered 30,144 pairs of expert comparison data for both the helpfulness and harmlessness metrics. We further showcase applications of BeaverTails in content moderation and reinforcement learning with human feedback (RLHF), emphasizing its potential for practical safety measures in LLMs. We believe this dataset provides vital resources for the community, contributing towards the safe development and deployment of LLMs. Our project page is available at the following URL: https://sites.google.com/view/pku-beavertails.

{{</citation>}}


### (48/93) Hate Speech Detection via Dual Contrastive Learning (Junyu Lu et al., 2023)

{{<citation>}}

Junyu Lu, Hongfei Lin, Xiaokun Zhang, Zhaoqing Li, Tongyue Zhang, Linlin Zong, Fenglong Ma, Bo Xu. (2023)  
**Hate Speech Detection via Dual Contrastive Learning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Contrastive Learning, Hate Speech Detection  
[Paper Link](http://arxiv.org/abs/2307.05578v1)  

---


**ABSTRACT**  
The fast spread of hate speech on social media impacts the Internet environment and our society by increasing prejudice and hurting people. Detecting hate speech has aroused broad attention in the field of natural language processing. Although hate speech detection has been addressed in recent work, this task still faces two inherent unsolved challenges. The first challenge lies in the complex semantic information conveyed in hate speech, particularly the interference of insulting words in hate speech detection. The second challenge is the imbalanced distribution of hate speech and non-hate speech, which may significantly deteriorate the performance of models. To tackle these challenges, we propose a novel dual contrastive learning (DCL) framework for hate speech detection. Our framework jointly optimizes the self-supervised and the supervised contrastive learning loss for capturing span-level information beyond the token-level emotional semantics used in existing models, particularly detecting speech containing abusive and insulting words. Moreover, we integrate the focal loss into the dual contrastive learning framework to alleviate the problem of data imbalance. We conduct experiments on two publicly available English datasets, and experimental results show that the proposed model outperforms the state-of-the-art models and precisely detects hate speeches.

{{</citation>}}


### (49/93) Detecting LLM-Generated Text in Computing Education: A Comparative Study for ChatGPT Cases (Michael Sheinman Orenstrakh et al., 2023)

{{<citation>}}

Michael Sheinman Orenstrakh, Oscar Karnalim, Carlos Anibal Suarez, Michael Liut. (2023)  
**Detecting LLM-Generated Text in Computing Education: A Comparative Study for ChatGPT Cases**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs.CL  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2307.07411v1)  

---


**ABSTRACT**  
Due to the recent improvements and wide availability of Large Language Models (LLMs), they have posed a serious threat to academic integrity in education. Modern LLM-generated text detectors attempt to combat the problem by offering educators with services to assess whether some text is LLM-generated. In this work, we have collected 124 submissions from computer science students before the creation of ChatGPT. We then generated 40 ChatGPT submissions. We used this data to evaluate eight publicly-available LLM-generated text detectors through the measures of accuracy, false positives, and resilience. The purpose of this work is to inform the community of what LLM-generated text detectors work and which do not, but also to provide insights for educators to better maintain academic integrity in their courses. Our results find that CopyLeaks is the most accurate LLM-generated text detector, GPTKit is the best LLM-generated text detector to reduce false positives, and GLTR is the most resilient LLM-generated text detector. We also express concerns over 52 false positives (of 114 human written submissions) generated by GPTZero. Finally, we note that all LLM-generated text detectors are less accurate with code, other languages (aside from English), and after the use of paraphrasing tools (like QuillBot). Modern detectors are still in need of improvements so that they can offer a full-proof solution to help maintain academic integrity. Further, their usability can be improved by facilitating a smooth API integration, providing clear documentation of their features and the understandability of their model(s), and supporting more commonly used languages.

{{</citation>}}


### (50/93) Improving Factuality of Abstractive Summarization via Contrastive Reward Learning (I-Chun Chern et al., 2023)

{{<citation>}}

I-Chun Chern, Zhiruo Wang, Sanjan Das, Bhavuk Sharma, Pengfei Liu, Graham Neubig. (2023)  
**Improving Factuality of Abstractive Summarization via Contrastive Reward Learning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Summarization  
[Paper Link](http://arxiv.org/abs/2307.04507v1)  

---


**ABSTRACT**  
Modern abstractive summarization models often generate summaries that contain hallucinated or contradictory information. In this paper, we propose a simple but effective contrastive learning framework that incorporates recent developments in reward learning and factuality metrics. Empirical studies demonstrate that the proposed framework enables summarization models to learn from feedback of factuality metrics using contrastive reward learning, leading to more factual summaries by human evaluations. This suggests that further advances in learning and evaluation algorithms can feed directly into providing more factual summaries.

{{</citation>}}


### (51/93) Enhancing Biomedical Text Summarization and Question-Answering: On the Utility of Domain-Specific Pre-Training (Dima Galat et al., 2023)

{{<citation>}}

Dima Galat, Marian-Andrei Rizoiu. (2023)  
**Enhancing Biomedical Text Summarization and Question-Answering: On the Utility of Domain-Specific Pre-Training**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Summarization, Text Summarization  
[Paper Link](http://arxiv.org/abs/2307.04412v1)  

---


**ABSTRACT**  
Biomedical summarization requires large datasets to train for text generation. We show that while transfer learning offers a viable option for addressing this challenge, an in-domain pre-training does not always offer advantages in a BioASQ summarization task. We identify a suitable model architecture and use it to show a benefit of a general-domain pre-training followed by a task-specific fine-tuning in the context of a BioASQ summarization task, leading to a novel three-step fine-tuning approach that works with only a thousand in-domain examples. Our results indicate that a Large Language Model without domain-specific pre-training can have a significant edge in some domain-specific biomedical text generation tasks.

{{</citation>}}


### (52/93) TIM: Teaching Large Language Models to Translate with Comparison (Jiali Zeng et al., 2023)

{{<citation>}}

Jiali Zeng, Fandong Meng, Yongjing Yin, Jie Zhou. (2023)  
**TIM: Teaching Large Language Models to Translate with Comparison**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.04408v1)  

---


**ABSTRACT**  
Open-sourced large language models (LLMs) have demonstrated remarkable efficacy in various tasks with instruction tuning. However, these models can sometimes struggle with tasks that require more specialized knowledge such as translation. One possible reason for such deficiency is that instruction tuning aims to generate fluent and coherent text that continues from a given instruction without being constrained by any task-specific requirements. Moreover, it can be more challenging for tuning smaller LLMs with lower-quality training data. To address this issue, we propose a novel framework using examples in comparison to teach LLMs to learn translation. Our approach involves presenting the model with examples of correct and incorrect translations and using a preference loss to guide the model's learning. We evaluate our method on WMT2022 test sets and show that it outperforms existing methods. Our findings offer a new perspective on fine-tuning LLMs for translation tasks and provide a promising solution for generating high-quality translations. Please refer to Github for more details: https://github.com/lemon0830/TIM.

{{</citation>}}


### (53/93) Enhancing Cross-lingual Transfer via Phonemic Transcription Integration (Hoang H. Nguyen et al., 2023)

{{<citation>}}

Hoang H. Nguyen, Chenwei Zhang, Tao Zhang, Eugene Rohrbaugh, Philip S. Yu. (2023)  
**Enhancing Cross-lingual Transfer via Phonemic Transcription Integration**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Named Entity Recognition  
[Paper Link](http://arxiv.org/abs/2307.04361v1)  

---


**ABSTRACT**  
Previous cross-lingual transfer methods are restricted to orthographic representation learning via textual scripts. This limitation hampers cross-lingual transfer and is biased towards languages sharing similar well-known scripts. To alleviate the gap between languages from different writing scripts, we propose PhoneXL, a framework incorporating phonemic transcriptions as an additional linguistic modality beyond the traditional orthographic transcriptions for cross-lingual transfer. Particularly, we propose unsupervised alignment objectives to capture (1) local one-to-one alignment between the two different modalities, (2) alignment via multi-modality contexts to leverage information from additional modalities, and (3) alignment via multilingual contexts where additional bilingual dictionaries are incorporated. We also release the first phonemic-orthographic alignment dataset on two token-level tasks (Named Entity Recognition and Part-of-Speech Tagging) among the understudied but interconnected Chinese-Japanese-Korean-Vietnamese (CJKV) languages. Our pilot study reveals phonemic transcription provides essential information beyond the orthography to enhance cross-lingual transfer and bridge the gap among CJKV languages, leading to consistent improvements on cross-lingual token-level tasks over orthographic-based multilingual PLMs.

{{</citation>}}


### (54/93) Event Extraction as Question Generation and Answering (Di Lu et al., 2023)

{{<citation>}}

Di Lu, Shihao Ran, Joel Tetreault, Alejandro Jaimes. (2023)  
**Event Extraction as Question Generation and Answering**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Event Extraction, QA, Question Answering, Question Generation  
[Paper Link](http://arxiv.org/abs/2307.05567v1)  

---


**ABSTRACT**  
Recent work on Event Extraction has reframed the task as Question Answering (QA), with promising results. The advantage of this approach is that it addresses the error propagation issue found in traditional token-based classification approaches by directly predicting event arguments without extracting candidates first. However, the questions are typically based on fixed templates and they rarely leverage contextual information such as relevant arguments. In addition, prior QA-based approaches have difficulty handling cases where there are multiple arguments for the same role. In this paper, we propose QGA-EE, which enables a Question Generation (QG) model to generate questions that incorporate rich contextual information instead of using fixed templates. We also propose dynamic templates to assist the training of QG model. Experiments show that QGA-EE outperforms all prior single-task-based models on the ACE05 English dataset.

{{</citation>}}


### (55/93) Learning to Generate Equitable Text in Dialogue from Biased Training Data (Anthony Sicilia et al., 2023)

{{<citation>}}

Anthony Sicilia, Malihe Alikhani. (2023)  
**Learning to Generate Equitable Text in Dialogue from Biased Training Data**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Bias, Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2307.04303v1)  

---


**ABSTRACT**  
The ingrained principles of fairness in a dialogue system's decision-making process and generated responses are crucial for user engagement, satisfaction, and task achievement. Absence of equitable and inclusive principles can hinder the formation of common ground, which in turn negatively impacts the overall performance of the system. For example, misusing pronouns in a user interaction may cause ambiguity about the intended subject. Yet, there is no comprehensive study of equitable text generation in dialogue. Aptly, in this work, we use theories of computational learning to study this problem. We provide formal definitions of equity in text generation, and further, prove formal connections between learning human-likeness and learning equity: algorithms for improving equity ultimately reduce to algorithms for improving human-likeness (on augmented data). With this insight, we also formulate reasonable conditions under which text generation algorithms can learn to generate equitable text without any modifications to the biased training data on which they learn. To exemplify our theory in practice, we look at a group of algorithms for the GuessWhat?! visual dialogue game and, using this example, test our theory empirically. Our theory accurately predicts relative-performance of multiple algorithms in generating equitable text as measured by both human and automated evaluation.

{{</citation>}}


### (56/93) HistRED: A Historical Document-Level Relation Extraction Dataset (Soyoung Yang et al., 2023)

{{<citation>}}

Soyoung Yang, Minseok Choi, Youngwoo Cho, Jaegul Choo. (2023)  
**HistRED: A Historical Document-Level Relation Extraction Dataset**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Relation Extraction  
[Paper Link](http://arxiv.org/abs/2307.04285v1)  

---


**ABSTRACT**  
Despite the extensive applications of relation extraction (RE) tasks in various domains, little has been explored in the historical context, which contains promising data across hundreds and thousands of years. To promote the historical RE research, we present HistRED constructed from Yeonhaengnok. Yeonhaengnok is a collection of records originally written in Hanja, the classical Chinese writing, which has later been translated into Korean. HistRED provides bilingual annotations such that RE can be performed on Korean and Hanja texts. In addition, HistRED supports various self-contained subtexts with different lengths, from a sentence level to a document level, supporting diverse context settings for researchers to evaluate the robustness of their RE models. To demonstrate the usefulness of our dataset, we propose a bilingual RE model that leverages both Korean and Hanja contexts to predict relations between entities. Our model outperforms monolingual baselines on HistRED, showing that employing multiple language contexts supplements the RE predictions. The dataset is publicly available at: https://huggingface.co/datasets/Soyoung/HistRED under CC BY-NC-ND 4.0 license.

{{</citation>}}


## cs.SE (5)



### (57/93) A Novel Approach to Identify Security Controls in Source Code (Ahmet Okutan et al., 2023)

{{<citation>}}

Ahmet Okutan, Ali Shokri, Viktoria Koscinski, Mohamad Fazelinia, Mehdi Mirakhorli. (2023)  
**A Novel Approach to Identify Security Controls in Source Code**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: BERT, NLP, Security, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.05605v1)  

---


**ABSTRACT**  
Secure by Design has become the mainstream development approach ensuring that software systems are not vulnerable to cyberattacks. Architectural security controls need to be carefully monitored over the software development life cycle to avoid critical design flaws. Unfortunately, functional requirements usually get in the way of the security features, and the development team may not correctly address critical security requirements. Identifying tactic-related code pieces in a software project enables an efficient review of the security controls' implementation as well as a resilient software architecture. This paper enumerates a comprehensive list of commonly used security controls and creates a dataset for each one of them by pulling related and unrelated code snippets from the open API of the StackOverflow question and answer platform. It uses the state-of-the-art NLP technique Bidirectional Encoder Representations from Transformers (BERT) and the Tactic Detector from our prior work to show that code pieces that implement security controls could be identified with high confidence. The results show that our model trained on tactic-related and unrelated code snippets derived from StackOverflow is able to identify tactic-related code pieces with F-Measure values above 0.9.

{{</citation>}}


### (58/93) Model-Driven Engineering for Artificial Intelligence -- A Systematic Literature Review (Simon Raedler et al., 2023)

{{<citation>}}

Simon Raedler, Luca Berardinelli, Karolin Winter, Abbas Rahimi, Stefanie Rinderle-Ma. (2023)  
**Model-Driven Engineering for Artificial Intelligence -- A Systematic Literature Review**  

---
Primary Category: cs.SE  
Categories: A-1; H-1-0; I-2-4, cs-AI, cs-SE, cs.SE  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.04599v1)  

---


**ABSTRACT**  
Objective: This study aims to investigate the existing body of knowledge in the field of Model-Driven Engineering MDE in support of AI (MDE4AI) to sharpen future research further and define the current state of the art.   Method: We conducted a Systemic Literature Review (SLR), collecting papers from five major databases resulting in 703 candidate studies, eventually retaining 15 primary studies. Each primary study will be evaluated and discussed with respect to the adoption of (1) MDE principles and practices and (2) the phases of AI development support aligned with the stages of the CRISP-DM methodology.   Results: The study's findings show that the pillar concepts of MDE (metamodel, concrete syntax and model transformation), are leveraged to define domain-specific languages (DSL) explicitly addressing AI concerns. Different MDE technologies are used, leveraging different language workbenches. The most prominent AI-related concerns are training and modeling of the AI algorithm, while minor emphasis is given to the time-consuming preparation of the data sets. Early project phases that support interdisciplinary communication of requirements, such as the CRISP-DM \textit{Business Understanding} phase, are rarely reflected.   Conclusion: The study found that the use of MDE for AI is still in its early stages, and there is no single tool or method that is widely used. Additionally, current approaches tend to focus on specific stages of development rather than providing support for the entire development process. As a result, the study suggests several research directions to further improve the use of MDE for AI and to guide future research in this area.

{{</citation>}}


### (59/93) Calculating Originality of LLM Assisted Source Code (Shipra Sharma et al., 2023)

{{<citation>}}

Shipra Sharma, Balwinder Sodhi. (2023)  
**Calculating Originality of LLM Assisted Source Code**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.04492v1)  

---


**ABSTRACT**  
The ease of using a Large Language Model (LLM) to answer a wide variety of queries and their high availability has resulted in LLMs getting integrated into various applications. LLM-based recommenders are now routinely used by students as well as professional software programmers for code generation and testing. Though LLM-based technology has proven useful, its unethical and unattributed use by students and professionals is a growing cause of concern. As such, there is a need for tools and technologies which may assist teachers and other evaluators in identifying whether any portion of a source code is LLM generated.   In this paper, we propose a neural network-based tool that instructors can use to determine the original effort (and LLM's contribution) put by students in writing source codes. Our tool is motivated by minimum description length measures like Kolmogorov complexity. Our initial experiments with moderate sized (up to 500 lines of code) have shown promising results that we report in this paper.

{{</citation>}}


### (60/93) Unmasking the giant: A comprehensive evaluation of ChatGPT's proficiency in coding algorithms and data structures (Sayed Erfan Arefin et al., 2023)

{{<citation>}}

Sayed Erfan Arefin, Tasnia Ashrafi Heya, Hasan Al-Qudah, Ynes Ineza, Abdul Serwadda. (2023)  
**Unmasking the giant: A comprehensive evaluation of ChatGPT's proficiency in coding algorithms and data structures**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-CL, cs-SE, cs.SE  
Keywords: AI, ChatGPT, GPT, GPT-3.5, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2307.05360v2)  

---


**ABSTRACT**  
The transformative influence of Large Language Models (LLMs) is profoundly reshaping the Artificial Intelligence (AI) technology domain. Notably, ChatGPT distinguishes itself within these models, demonstrating remarkable performance in multi-turn conversations and exhibiting code proficiency across an array of languages. In this paper, we carry out a comprehensive evaluation of ChatGPT's coding capabilities based on what is to date the largest catalog of coding challenges. Our focus is on the python programming language and problems centered on data structures and algorithms, two topics at the very foundations of Computer Science. We evaluate ChatGPT for its ability to generate correct solutions to the problems fed to it, its code quality, and nature of run-time errors thrown by its code. Where ChatGPT code successfully executes, but fails to solve the problem at hand, we look into patterns in the test cases passed in order to gain some insights into how wrong ChatGPT code is in these kinds of situations. To infer whether ChatGPT might have directly memorized some of the data that was used to train it, we methodically design an experiment to investigate this phenomena. Making comparisons with human performance whenever feasible, we investigate all the above questions from the context of both its underlying learning models (GPT-3.5 and GPT-4), on a vast array sub-topics within the main topics, and on problems having varying degrees of difficulty.

{{</citation>}}


### (61/93) Can Large Language Models Write Good Property-Based Tests? (Vasudev Vikram et al., 2023)

{{<citation>}}

Vasudev Vikram, Caroline Lemieux, Rohan Padhye. (2023)  
**Can Large Language Models Write Good Property-Based Tests?**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2307.04346v1)  

---


**ABSTRACT**  
Property-based testing (PBT), while an established technique in the software testing research community, is still relatively underused in real-world software. Pain points in writing property-based tests include implementing diverse random input generators and thinking of meaningful properties to test. Developers, however, are more amenable to writing documentation; plenty of library API documentation is available and can be used as natural language specifications for property-based tests. As large language models (LLMs) have recently shown promise in a variety of coding tasks, we explore the potential of using LLMs to synthesize property-based tests. We call our approach PBT-GPT, and propose three different strategies of prompting the LLM for PBT. We characterize various failure modes of PBT-GPT and detail an evaluation methodology for automatically synthesized property-based tests. PBT-GPT achieves promising results in our preliminary studies on sample Python library APIs in $\texttt{numpy}$, $\texttt{networkx}$, and $\texttt{datetime}$.

{{</citation>}}


## eess.SP (1)



### (62/93) Fast dynamic time warping and clustering in C++ (Volkan Kumtepeli et al., 2023)

{{<citation>}}

Volkan Kumtepeli, Rebecca Perriment, David A. Howey. (2023)  
**Fast dynamic time warping and clustering in C++**  

---
Primary Category: eess.SP  
Categories: cs-LG, cs-SY, eess-SP, eess-SY, eess.SP  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2307.04904v1)  

---


**ABSTRACT**  
We present an approach for computationally efficient dynamic time warping (DTW) and clustering of time-series data. The method frames the dynamic warping of time series datasets as an optimisation problem solved using dynamic programming, and then clusters time series data by solving a second optimisation problem using mixed-integer programming (MIP). There is also an option to use k-medoids clustering for increased speed, when a certificate for global optimality is not essential. The improved efficiency of our approach is due to task-level parallelisation of the clustering alongside DTW. Our approach was tested using the UCR Time Series Archive, and was found to be, on average, 33% faster than the next fastest option when using the same clustering method. This increases to 64% faster when considering only larger datasets (with more than 1000 time series). The MIP clustering is most effective on small numbers of longer time series, because the DTW computation is faster than other approaches, but the clustering problem becomes increasingly computationally expensive as the number of time series to be clustered increases.

{{</citation>}}


## cs.AI (11)



### (63/93) Learning to Solve Constraint Satisfaction Problems with Recurrent Transformer (Zhun Yang et al., 2023)

{{<citation>}}

Zhun Yang, Adam Ishay, Joohyung Lee. (2023)  
**Learning to Solve Constraint Satisfaction Problems with Recurrent Transformer**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: Graph Neural Network, Graph Neural Networks, Transformer  
[Paper Link](http://arxiv.org/abs/2307.04895v1)  

---


**ABSTRACT**  
Constraint satisfaction problems (CSPs) are about finding values of variables that satisfy the given constraints. We show that Transformer extended with recurrence is a viable approach to learning to solve CSPs in an end-to-end manner, having clear advantages over state-of-the-art methods such as Graph Neural Networks, SATNet, and some neuro-symbolic models. With the ability of Transformer to handle visual input, the proposed Recurrent Transformer can straightforwardly be applied to visual constraint reasoning problems while successfully addressing the symbol grounding problem. We also show how to leverage deductive knowledge of discrete constraints in the Transformer's inductive learning to achieve sample-efficient learning and semi-supervised learning for CSPs.

{{</citation>}}


### (64/93) AI For Global Climate Cooperation 2023 Competition Proceedings (Yoshua Bengio et al., 2023)

{{<citation>}}

Yoshua Bengio, Prateek Gupta, Lu Li, Soham Phade, Sunil Srinivasa, Andrew Williams, Tianyu Zhang, Yang Zhang, Stephan Zheng. (2023)  
**AI For Global Climate Cooperation 2023 Competition Proceedings**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.06951v1)  

---


**ABSTRACT**  
The international community must collaborate to mitigate climate change and sustain economic growth. However, collaboration is hard to achieve, partly because no global authority can ensure compliance with international climate agreements. Combining AI with climate-economic simulations offers a promising solution to design international frameworks, including negotiation protocols and climate agreements, that promote and incentivize collaboration. In addition, these frameworks should also have policy goals fulfillment, and sustained commitment, taking into account climate-economic dynamics and strategic behaviors. These challenges require an interdisciplinary approach across machine learning, economics, climate science, law, policy, ethics, and other fields.   Towards this objective, we organized AI for Global Climate Cooperation, a Mila competition in which teams submitted proposals and analyses of international frameworks, based on (modifications of) RICE-N, an AI-driven integrated assessment model (IAM). In particular, RICE-N supports modeling regional decision-making using AI agents. Furthermore, the IAM then models the climate-economic impact of those decisions into the future.   Whereas the first track focused only on performance metrics, the proposals submitted to the second track were evaluated both quantitatively and qualitatively. The quantitative evaluation focused on a combination of (i) the degree of mitigation of global temperature rise and (ii) the increase in economic productivity. On the other hand, an interdisciplinary panel of human experts in law, policy, sociology, economics and environmental science, evaluated the solutions qualitatively. In particular, the panel considered the effectiveness, simplicity, feasibility, ethics, and notions of climate justice of the protocols. In the third track, the participants were asked to critique and improve RICE-N.

{{</citation>}}


### (65/93) Large Language Models as General Pattern Machines (Suvir Mirchandani et al., 2023)

{{<citation>}}

Suvir Mirchandani, Fei Xia, Pete Florence, Brian Ichter, Danny Driess, Montserrat Gonzalez Arenas, Kanishka Rao, Dorsa Sadigh, Andy Zeng. (2023)  
**Large Language Models as General Pattern Machines**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-RO, cs.AI  
Keywords: AI, Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2307.04721v1)  

---


**ABSTRACT**  
We observe that pre-trained large language models (LLMs) are capable of autoregressively completing complex token sequences -- from arbitrary ones procedurally generated by probabilistic context-free grammars (PCFG), to more rich spatial patterns found in the Abstract Reasoning Corpus (ARC), a general AI benchmark, prompted in the style of ASCII art. Surprisingly, pattern completion proficiency can be partially retained even when the sequences are expressed using tokens randomly sampled from the vocabulary. These results suggest that without any additional training, LLMs can serve as general sequence modelers, driven by in-context learning. In this work, we investigate how these zero-shot capabilities may be applied to problems in robotics -- from extrapolating sequences of numbers that represent states over time to complete simple motions, to least-to-most prompting of reward-conditioned trajectories that can discover and represent closed-loop policies (e.g., a stabilizing controller for CartPole). While difficult to deploy today for real systems due to latency, context size limitations, and compute costs, the approach of using LLMs to drive low-level control may provide an exciting glimpse into how the patterns among words could be transferred to actions.

{{</citation>}}


### (66/93) Understanding Real-World AI Planning Domains: A Conceptual Framework (Ebaa Alnazer et al., 2023)

{{<citation>}}

Ebaa Alnazer, Ilche Georgievski. (2023)  
**Understanding Real-World AI Planning Domains: A Conceptual Framework**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.04701v1)  

---


**ABSTRACT**  
Planning is a pivotal ability of any intelligent system being developed for real-world applications. AI planning is concerned with researching and developing planning systems that automatically compute plans that satisfy some user objective. Identifying and understanding the relevant and realistic aspects that characterise real-world application domains are crucial to the development of AI planning systems. This provides guidance to knowledge engineers and software engineers in the process of designing, identifying, and categorising resources required for the development process. To the best of our knowledge, such support does not exist. We address this research gap by developing a conceptual framework that identifies and categorises the aspects of real-world planning domains in varying levels of granularity. Our framework provides not only a common terminology but also a comprehensive overview of a broad range of planning aspects exemplified using the domain of sustainable buildings as a prominent application domain of AI planning. The framework has the potential to impact the design, development, and applicability of AI planning systems in real-world application domains.

{{</citation>}}


### (67/93) A Semi-Automated Solution Approach Selection Tool for Any Use Case via Scopus and OpenAI: a Case Study for AI/ML in Oncology (Deniz Kenan Kılıç et al., 2023)

{{<citation>}}

Deniz Kenan Kılıç, Alex Elkjær Vasegaard, Aurélien Desoeuvres, Peter Nielsen. (2023)  
**A Semi-Automated Solution Approach Selection Tool for Any Use Case via Scopus and OpenAI: a Case Study for AI/ML in Oncology**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-IR, cs-LG, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.04573v1)  

---


**ABSTRACT**  
In today's vast literature landscape, a manual review is very time-consuming. To address this challenge, this paper proposes a semi-automated tool for solution method review and selection. It caters to researchers, practitioners, and decision-makers while serving as a benchmark for future work. The tool comprises three modules: (1) paper selection and scoring, using a keyword selection scheme to query Scopus API and compute relevancy; (2) solution method extraction in papers utilizing OpenAI API; (3) sensitivity analysis and post-analyzes. It reveals trends, relevant papers, and methods. AI in the oncology case study and several use cases are presented with promising results, comparing the tool to manual ground truth.

{{</citation>}}


### (68/93) Pathway toward prior knowledge-integrated machine learning in engineering (Xia Chen et al., 2023)

{{<citation>}}

Xia Chen, Philipp Geyer. (2023)  
**Pathway toward prior knowledge-integrated machine learning in engineering**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.06950v1)  

---


**ABSTRACT**  
Despite the digitalization trend and data volume surge, first-principles models (also known as logic-driven, physics-based, rule-based, or knowledge-based models) and data-driven approaches have existed in parallel, mirroring the ongoing AI debate on symbolism versus connectionism. Research for process development to integrate both sides to transfer and utilize domain knowledge in the data-driven process is rare. This study emphasizes efforts and prevailing trends to integrate multidisciplinary domain professions into machine acknowledgeable, data-driven processes in a two-fold organization: examining information uncertainty sources in knowledge representation and exploring knowledge decomposition with a three-tier knowledge-integrated machine learning paradigm. This approach balances holist and reductionist perspectives in the engineering domain.

{{</citation>}}


### (69/93) Exploring Large Language Model for Graph Data Understanding in Online Job Recommendations (Likang Wu et al., 2023)

{{<citation>}}

Likang Wu, Zhaopeng Qiu, Zhi Zheng, Hengshu Zhu, Enhong Chen. (2023)  
**Exploring Large Language Model for Graph Data Understanding in Online Job Recommendations**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-IR, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.05722v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have revolutionized natural language processing tasks, demonstrating their exceptional capabilities in various domains. However, their potential for behavior graph understanding in job recommendations remains largely unexplored. This paper focuses on unveiling the capability of large language models in understanding behavior graphs and leveraging this understanding to enhance recommendations in online recruitment, including the promotion of out-of-distribution (OOD) application. We present a novel framework that harnesses the rich contextual information and semantic representations provided by large language models to analyze behavior graphs and uncover underlying patterns and relationships. Specifically, we propose a meta-path prompt constructor that leverages LLM recommender to understand behavior graphs for the first time and design a corresponding path augmentation module to alleviate the prompt bias introduced by path-based sequence input. By leveraging this capability, our framework enables personalized and accurate job recommendations for individual users. We evaluate the effectiveness of our approach on a comprehensive dataset and demonstrate its ability to improve the relevance and quality of recommended quality. This research not only sheds light on the untapped potential of large language models but also provides valuable insights for developing advanced recommendation systems in the recruitment market. The findings contribute to the growing field of natural language processing and offer practical implications for enhancing job search experiences.

{{</citation>}}


### (70/93) PapagAI:Automated Feedback for Reflective Essays (Veronika Solopova et al., 2023)

{{<citation>}}

Veronika Solopova, Adrian Gruszczynski, Eiad Rostom, Fritz Cremer, Sascha Witte, Chengming Zhang, Fernando Ramos López Lea Plößl, Florian Hofmann, Ralf Romeike, Michaela Gläser-Zikuda, Christoph Benzmüller, Tim Landgraf. (2023)  
**PapagAI:Automated Feedback for Reflective Essays**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.07523v1)  

---


**ABSTRACT**  
Written reflective practice is a regular exercise pre-service teachers perform during their higher education. Usually, their lecturers are expected to provide individual feedback, which can be a challenging task to perform on a regular basis. In this paper, we present the first open-source automated feedback tool based on didactic theory and implemented as a hybrid AI system. We describe the components and discuss the advantages and disadvantages of our system compared to the state-of-art generative large language models. The main objective of our work is to enable better learning outcomes for students and to complement the teaching activities of lecturers.

{{</citation>}}


### (71/93) RLTF: Reinforcement Learning from Unit Test Feedback (Jiate Liu et al., 2023)

{{<citation>}}

Jiate Liu, Yiqin Zhu, Kaiwen Xiao, Qiang Fu, Xiao Han, Wei Yang, Deheng Ye. (2023)  
**RLTF: Reinforcement Learning from Unit Test Feedback**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-LG, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.04349v1)  

---


**ABSTRACT**  
The goal of program synthesis, or code generation, is to generate executable code based on given descriptions. Recently, there has been an increasing number of studies employing reinforcement learning (RL) to improve the performance of large language models (LLMs) for code. However, these RL methods have only used offline frameworks, limiting their exploration of new sample spaces. Additionally, current approaches that utilize unit test signals are rather simple, not accounting for specific error locations within the code. To address these issues, we proposed RLTF, i.e., Reinforcement Learning from Unit Test Feedback, a novel online RL framework with unit test feedback of multi-granularity for refining code LLMs. Our approach generates data in real-time during training and simultaneously utilizes fine-grained feedback signals to guide the model towards producing higher-quality code. Extensive experiments show that RLTF achieves state-of-the-art performance on the APPS and the MBPP benchmarks. Our code can be found at: https://github.com/Zyq-scut/RLTF.

{{</citation>}}


### (72/93) Injecting Logical Constraints into Neural Networks via Straight-Through Estimators (Zhun Yang et al., 2023)

{{<citation>}}

Zhun Yang, Joohyung Lee, Chiyoun Park. (2023)  
**Injecting Logical Constraints into Neural Networks via Straight-Through Estimators**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs-NE, cs.AI  
Keywords: AI, GNN  
[Paper Link](http://arxiv.org/abs/2307.04347v1)  

---


**ABSTRACT**  
Injecting discrete logical constraints into neural network learning is one of the main challenges in neuro-symbolic AI. We find that a straight-through-estimator, a method introduced to train binary neural networks, could effectively be applied to incorporate logical constraints into neural network learning. More specifically, we design a systematic way to represent discrete logical constraints as a loss function; minimizing this loss using gradient descent via a straight-through-estimator updates the neural network's weights in the direction that the binarized outputs satisfy the logical constraints. The experimental results show that by leveraging GPUs and batch training, this method scales significantly better than existing neuro-symbolic methods that require heavy symbolic computation for computing gradients. Also, we demonstrate that our method applies to different types of neural networks, such as MLP, CNN, and GNN, making them learn with no or fewer labeled data by learning directly from known constraints.

{{</citation>}}


### (73/93) Source-Aware Embedding Training on Heterogeneous Information Networks (Tsai Hor Chan et al., 2023)

{{<citation>}}

Tsai Hor Chan, Chi Ho Wong, Jiajun Shen, Guosheng Yin. (2023)  
**Source-Aware Embedding Training on Heterogeneous Information Networks**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs-SI, cs.AI  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2307.04336v1)  

---


**ABSTRACT**  
Heterogeneous information networks (HINs) have been extensively applied to real-world tasks, such as recommendation systems, social networks, and citation networks. While existing HIN representation learning methods can effectively learn the semantic and structural features in the network, little awareness was given to the distribution discrepancy of subgraphs within a single HIN. However, we find that ignoring such distribution discrepancy among subgraphs from multiple sources would hinder the effectiveness of graph embedding learning algorithms. This motivates us to propose SUMSHINE (Scalable Unsupervised Multi-Source Heterogeneous Information Network Embedding) -- a scalable unsupervised framework to align the embedding distributions among multiple sources of an HIN. Experimental results on real-world datasets in a variety of downstream tasks validate the performance of our method over the state-of-the-art heterogeneous information network embedding algorithms.

{{</citation>}}


## cs.CR (2)



### (74/93) ChatGPT for Digital Forensic Investigation: The Good, The Bad, and The Unknown (Mark Scanlon et al., 2023)

{{<citation>}}

Mark Scanlon, Frank Breitinger, Christopher Hargreaves, Jan-Niclas Hilgert, John Sheppard. (2023)  
**ChatGPT for Digital Forensic Investigation: The Good, The Bad, and The Unknown**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CL, cs-CR, cs.CR  
Keywords: BERT, ChatGPT, GPT, GPT-3.5, GPT-4, LLaMA, Language Model, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.10195v1)  

---


**ABSTRACT**  
The disruptive application of ChatGPT (GPT-3.5, GPT-4) to a variety of domains has become a topic of much discussion in the scientific community and society at large. Large Language Models (LLMs), e.g., BERT, Bard, Generative Pre-trained Transformers (GPTs), LLaMA, etc., have the ability to take instructions, or prompts, from users and generate answers and solutions based on very large volumes of text-based training data. This paper assesses the impact and potential impact of ChatGPT on the field of digital forensics, specifically looking at its latest pre-trained LLM, GPT-4. A series of experiments are conducted to assess its capability across several digital forensic use cases including artefact understanding, evidence searching, code generation, anomaly detection, incident response, and education. Across these topics, its strengths and risks are outlined and a number of general conclusions are drawn. Overall this paper concludes that while there are some potential low-risk applications of ChatGPT within digital forensics, many are either unsuitable at present, since the evidence would need to be uploaded to the service, or they require sufficient knowledge of the topic being asked of the tool to identify incorrect assumptions, inaccuracies, and mistakes. However, to an appropriately knowledgeable user, it could act as a useful supporting tool in some circumstances.

{{</citation>}}


### (75/93) False Sense of Security: Leveraging XAI to Analyze the Reasoning and True Performance of Context-less DGA Classifiers (Arthur Drichel et al., 2023)

{{<citation>}}

Arthur Drichel, Ulrike Meyer. (2023)  
**False Sense of Security: Leveraging XAI to Analyze the Reasoning and True Performance of Context-less DGA Classifiers**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs.CR  
Keywords: AI, Reasoning, Security  
[Paper Link](http://arxiv.org/abs/2307.04358v1)  

---


**ABSTRACT**  
The problem of revealing botnet activity through Domain Generation Algorithm (DGA) detection seems to be solved, considering that available deep learning classifiers achieve accuracies of over 99.9%. However, these classifiers provide a false sense of security as they are heavily biased and allow for trivial detection bypass. In this work, we leverage explainable artificial intelligence (XAI) methods to analyze the reasoning of deep learning classifiers and to systematically reveal such biases. We show that eliminating these biases from DGA classifiers considerably deteriorates their performance. Nevertheless we are able to design a context-aware detection system that is free of the identified biases and maintains the detection rate of state-of-the art deep learning classifiers. In this context, we propose a visual analysis system that helps to better understand a classifier's reasoning, thereby increasing trust in and transparency of detection methods and facilitating decision-making.

{{</citation>}}


## cs.HC (1)



### (76/93) AmadeusGPT: a natural language interface for interactive animal behavioral analysis (Shaokai Ye et al., 2023)

{{<citation>}}

Shaokai Ye, Jessy Lauer, Mu Zhou, Alexander Mathis, Mackenzie W. Mathis. (2023)  
**AmadeusGPT: a natural language interface for interactive animal behavioral analysis**  

---
Primary Category: cs.HC  
Categories: cs-CV, cs-HC, cs.HC, q-bio-NC  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2307.04858v1)  

---


**ABSTRACT**  
The process of quantifying and analyzing animal behavior involves translating the naturally occurring descriptive language of their actions into machine-readable code. Yet, codifying behavior analysis is often challenging without deep understanding of animal behavior and technical machine learning knowledge. To limit this gap, we introduce AmadeusGPT: a natural language interface that turns natural language descriptions of behaviors into machine-executable code. Large-language models (LLMs) such as GPT3.5 and GPT4 allow for interactive language-based queries that are potentially well suited for making interactive behavior analysis. However, the comprehension capability of these LLMs is limited by the context window size, which prevents it from remembering distant conversations. To overcome the context window limitation, we implement a novel dual-memory mechanism to allow communication between short-term and long-term memory using symbols as context pointers for retrieval and saving. Concretely, users directly use language-based definitions of behavior and our augmented GPT develops code based on the core AmadeusGPT API, which contains machine learning, computer vision, spatio-temporal reasoning, and visualization modules. Users then can interactively refine results, and seamlessly add new behavioral modules as needed. We benchmark AmadeusGPT and show we can produce state-of-the-art performance on the MABE 2022 behavior challenge tasks. Note, an end-user would not need to write any code to achieve this. Thus, collectively AmadeusGPT presents a novel way to merge deep biological knowledge, large-language models, and core computer vision modules into a more naturally intelligent system. Code and demos can be found at: https://github.com/AdaptiveMotorControlLab/AmadeusGPT.

{{</citation>}}


## stat.ML (1)



### (77/93) Dynamics of Temporal Difference Reinforcement Learning (Blake Bordelon et al., 2023)

{{<citation>}}

Blake Bordelon, Paul Masset, Henry Kuo, Cengiz Pehlevan. (2023)  
**Dynamics of Temporal Difference Reinforcement Learning**  

---
Primary Category: stat.ML  
Categories: cond-mat-dis-nn, cs-AI, cs-LG, stat-ML, stat.ML  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.04841v1)  

---


**ABSTRACT**  
Reinforcement learning has been successful across several applications in which agents have to learn to act in environments with sparse feedback. However, despite this empirical success there is still a lack of theoretical understanding of how the parameters of reinforcement learning models and the features used to represent states interact to control the dynamics of learning. In this work, we use concepts from statistical physics, to study the typical case learning curves for temporal difference learning of a value function with linear function approximators. Our theory is derived under a Gaussian equivalence hypothesis where averages over the random trajectories are replaced with temporally correlated Gaussian feature averages and we validate our assumptions on small scale Markov Decision Processes. We find that the stochastic semi-gradient noise due to subsampling the space of possible episodes leads to significant plateaus in the value error, unlike in traditional gradient descent dynamics. We study how learning dynamics and plateaus depend on feature structure, learning rate, discount factor, and reward function. We then analyze how strategies like learning rate annealing and reward shaping can favorably alter learning dynamics and plateaus. To conclude, our work introduces new tools to open a new direction towards developing a theory of learning dynamics in reinforcement learning.

{{</citation>}}


## cs.DB (1)



### (78/93) The LDBC Social Network Benchmark Interactive workload v2: A transactional graph query benchmark with deep delete operations (David Püroja et al., 2023)

{{<citation>}}

David Püroja, Jack Waudby, Peter Boncz, Gábor Szárnyas. (2023)  
**The LDBC Social Network Benchmark Interactive workload v2: A transactional graph query benchmark with deep delete operations**  

---
Primary Category: cs.DB  
Categories: H-2-4, cs-DB, cs.DB  
Keywords: Social Network  
[Paper Link](http://arxiv.org/abs/2307.04820v1)  

---


**ABSTRACT**  
The LDBC Social Network Benchmark's Interactive workload captures an OLTP scenario operating on a correlated social network graph. It consists of complex graph queries executed concurrently with a stream of updates operation. Since its initial release in 2015, the Interactive workload has become the de facto industry standard for benchmarking transactional graph data management systems. As graph systems have matured and the community's understanding of graph processing features has evolved, we initiated the renewal of this benchmark. This paper describes the Interactive v2 workload with several new features: delete operations, a cheapest path-finding query, support for larger data sets, and a novel temporal parameter curation algorithm that ensures stable runtimes for path queries.

{{</citation>}}


## cs.RO (3)



### (79/93) RoCo: Dialectic Multi-Robot Collaboration with Large Language Models (Zhao Mandi et al., 2023)

{{<citation>}}

Zhao Mandi, Shreeya Jain, Shuran Song. (2023)  
**RoCo: Dialectic Multi-Robot Collaboration with Large Language Models**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-LG, cs-RO, cs.RO  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.04738v1)  

---


**ABSTRACT**  
We propose a novel approach to multi-robot collaboration that harnesses the power of pre-trained large language models (LLMs) for both high-level communication and low-level path planning. Robots are equipped with LLMs to discuss and collectively reason task strategies. They then generate sub-task plans and task space waypoint paths, which are used by a multi-arm motion planner to accelerate trajectory planning. We also provide feedback from the environment, such as collision checking, and prompt the LLM agents to improve their plan and waypoints in-context. For evaluation, we introduce RoCoBench, a 6-task benchmark covering a wide range of multi-robot collaboration scenarios, accompanied by a text-only dataset for agent representation and reasoning. We experimentally demonstrate the effectiveness of our approach -- it achieves high success rates across all tasks in RoCoBench and adapts to variations in task semantics. Our dialog setup offers high interpretability and flexibility -- in real world experiments, we show RoCo easily incorporates human-in-the-loop, where a user can communicate and collaborate with a robot agent to complete tasks together. See project website https://project-roco.github.io for videos and code.

{{</citation>}}


### (80/93) A Versatile Door Opening System with Mobile Manipulator through Adaptive Position-Force Control and Reinforcement Learning (Gyuree Kang et al., 2023)

{{<citation>}}

Gyuree Kang, Hyunki Seong, Daegyu Lee, D. Hyunchul Shim. (2023)  
**A Versatile Door Opening System with Mobile Manipulator through Adaptive Position-Force Control and Reinforcement Learning**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs-SY, cs.RO, eess-SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.04422v1)  

---


**ABSTRACT**  
The ability of robots to navigate through doors is crucial for their effective operation in indoor environments. Consequently, extensive research has been conducted to develop robots capable of opening specific doors. However, the diverse combinations of door handles and opening directions necessitate a more versatile door opening system for robots to successfully operate in real-world environments. In this paper, we propose a mobile manipulator system that can autonomously open various doors without prior knowledge. By using convolutional neural networks, point cloud extraction techniques, and external force measurements during exploratory motion, we obtained information regarding handle types, poses, and door characteristics. Through two different approaches, adaptive position-force control and deep reinforcement learning, we successfully opened doors without precise trajectory or excessive external force. The adaptive position-force control method involves moving the end-effector in the direction of the door opening while responding compliantly to external forces, ensuring safety and manipulator workspace. Meanwhile, the deep reinforcement learning policy minimizes applied forces and eliminates unnecessary movements, enabling stable operation across doors with different poses and widths. The RL-based approach outperforms the adaptive position-force control method in terms of compensating for external forces, ensuring smooth motion, and achieving efficient speed. It reduces the maximum force required by 3.27 times and improves motion smoothness by 1.82 times. However, the non-learning-based adaptive position-force control method demonstrates more versatility in opening a wider range of doors, encompassing revolute doors with four distinct opening directions and varying widths.

{{</citation>}}


### (81/93) Legal Decision-making for Highway Automated Driving (Xiaohan Ma et al., 2023)

{{<citation>}}

Xiaohan Ma, Wenhao Yu, Chengxiang Zhao, Changjun Wang, Wenhui Zhou, Guangming Zhao, Mingyue Ma, Weida Wang, Lin Yang, Rui Mu, Hong Wang, Jun Li. (2023)  
**Legal Decision-making for Highway Automated Driving**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs-SY, cs.RO, eess-SY  
Keywords: Legal  
[Paper Link](http://arxiv.org/abs/2307.04327v1)  

---


**ABSTRACT**  
Compliance with traffic laws is a fundamental requirement for human drivers on the road, and autonomous vehicles must adhere to traffic laws as well. However, current autonomous vehicles prioritize safety and collision avoidance primarily in their decision-making and planning, which will lead to misunderstandings and distrust from human drivers and may even result in accidents in mixed traffic flow. Therefore, ensuring the compliance of the autonomous driving decision-making system is essential for ensuring the safety of autonomous driving and promoting the widespread adoption of autonomous driving technology. To this end, the paper proposes a trigger-based layered compliance decision-making framework. This framework utilizes the decision intent at the highest level as a signal to activate an online violation monitor that identifies the type of violation committed by the vehicle. Then, a four-layer architecture for compliance decision-making is employed to generate compliantly trajectories. Using this system, autonomous vehicles can detect and correct potential violations in real-time, thereby enhancing safety and building public confidence in autonomous driving technology. Finally, the proposed method is evaluated on the DJI AD4CHE highway dataset under four typical highway scenarios: speed limit, following distance, overtaking, and lane-changing. The results indicate that the proposed method increases the vehicle's overall compliance rate from 13.85% to 84.46%, while reducing the proportion of active violations to 0%, demonstrating its effectiveness.

{{</citation>}}


## cs.IT (1)



### (82/93) Deceptive Information Retrieval (Sajani Vithana et al., 2023)

{{<citation>}}

Sajani Vithana, Sennur Ulukus. (2023)  
**Deceptive Information Retrieval**  

---
Primary Category: cs.IT  
Categories: cs-CR, cs-IT, cs-NI, cs.IT, eess-SP, math-IT  
Keywords: Information Retrieval  
[Paper Link](http://arxiv.org/abs/2307.04727v1)  

---


**ABSTRACT**  
We introduce the problem of deceptive information retrieval (DIR), in which a user wishes to download a required file out of multiple independent files stored in a system of databases while \emph{deceiving} the databases by making the databases' predictions on the user-required file index incorrect with high probability. Conceptually, DIR is an extension of private information retrieval (PIR). In PIR, a user downloads a required file without revealing its index to any of the databases. The metric of deception is defined as the probability of error of databases' prediction on the user-required file, minus the corresponding probability of error in PIR. The problem is defined on time-sensitive data that keeps updating from time to time. In the proposed scheme, the user deceives the databases by sending \emph{real} queries to download the required file at the time of the requirement and \emph{dummy} queries at multiple distinct future time instances to manipulate the probabilities of sending each query for each file requirement, using which the databases' make the predictions on the user-required file index. The proposed DIR scheme is based on a capacity achieving probabilistic PIR scheme, and achieves rates lower than the PIR capacity due to the additional downloads made to deceive the databases. When the required level of deception is zero, the proposed scheme achieves the PIR capacity.

{{</citation>}}


## cs.SI (1)



### (83/93) Quantifying the Echo Chamber Effect: An Embedding Distance-based Approach (Faisal Alatawi et al., 2023)

{{<citation>}}

Faisal Alatawi, Paras Sheth, Huan Liu. (2023)  
**Quantifying the Echo Chamber Effect: An Embedding Distance-based Approach**  

---
Primary Category: cs.SI  
Categories: cs-AI, cs-LG, cs-SI, cs.SI  
Keywords: Embedding, Twitter  
[Paper Link](http://arxiv.org/abs/2307.04668v2)  

---


**ABSTRACT**  
The rise of social media platforms has facilitated the formation of echo chambers, which are online spaces where users predominantly encounter viewpoints that reinforce their existing beliefs while excluding dissenting perspectives. This phenomenon significantly hinders information dissemination across communities and fuels societal polarization. Therefore, it is crucial to develop methods for quantifying echo chambers. In this paper, we present the Echo Chamber Score (ECS), a novel metric that assesses the cohesion and separation of user communities by measuring distances between users in the embedding space. In contrast to existing approaches, ECS is able to function without labels for user ideologies and makes no assumptions about the structure of the interaction graph. To facilitate measuring distances between users, we propose EchoGAE, a self-supervised graph autoencoder-based user embedding model that leverages users' posts and the interaction graph to embed them in a manner that reflects their ideological similarity. To assess the effectiveness of ECS, we use a Twitter dataset consisting of four topics - two polarizing and two non-polarizing. Our results showcase ECS's effectiveness as a tool for quantifying echo chambers and shedding light on the dynamics of online discourse.

{{</citation>}}


## cs.SD (3)



### (84/93) EchoVest: Real-Time Sound Classification and Depth Perception Expressed through Transcutaneous Electrical Nerve Stimulation (Jesse Choe et al., 2023)

{{<citation>}}

Jesse Choe, Siddhant Sood, Ryan Park. (2023)  
**EchoVest: Real-Time Sound Classification and Depth Perception Expressed through Transcutaneous Electrical Nerve Stimulation**  

---
Primary Category: cs.SD  
Categories: cs-LG, cs-SD, cs.SD, eess-AS, eess-SP  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.04604v1)  

---


**ABSTRACT**  
Over 1.5 billion people worldwide live with hearing impairment. Despite various technologies that have been created for individuals with such disabilities, most of these technologies are either extremely expensive or inaccessible for everyday use in low-medium income countries. In order to combat this issue, we have developed a new assistive device, EchoVest, for blind/deaf people to intuitively become more aware of their environment. EchoVest transmits vibrations to the user's body by utilizing transcutaneous electric nerve stimulation (TENS) based on the source of the sounds. EchoVest also provides various features, including sound localization, sound classification, noise reduction, and depth perception. We aimed to outperform CNN-based machine-learning models, the most commonly used machine learning model for classification tasks, in accuracy and computational costs. To do so, we developed and employed a novel audio pipeline that adapts the Audio Spectrogram Transformer (AST) model, an attention-based model, for our sound classification purposes, and Fast Fourier Transforms for noise reduction. The application of Otsu's Method helped us find the optimal thresholds for background noise sound filtering and gave us much greater accuracy. In order to calculate direction and depth accurately, we applied Complex Time Difference of Arrival algorithms and SOTA localization. Our last improvement was to use blind source separation to make our algorithms applicable to multiple microphone inputs. The final algorithm achieved state-of-the-art results on numerous checkpoints, including a 95.7\% accuracy on the ESC-50 dataset for environmental sound classification.

{{</citation>}}


### (85/93) Automatic Piano Transcription with Hierarchical Frequency-Time Transformer (Keisuke Toyama et al., 2023)

{{<citation>}}

Keisuke Toyama, Taketo Akama, Yukara Ikemiya, Yuhta Takida, Wei-Hsiang Liao, Yuki Mitsufuji. (2023)  
**Automatic Piano Transcription with Hierarchical Frequency-Time Transformer**  

---
Primary Category: cs.SD  
Categories: cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.04305v1)  

---


**ABSTRACT**  
Taking long-term spectral and temporal dependencies into account is essential for automatic piano transcription. This is especially helpful when determining the precise onset and offset for each note in the polyphonic piano content. In this case, we may rely on the capability of self-attention mechanism in Transformers to capture these long-term dependencies in the frequency and time axes. In this work, we propose hFT-Transformer, which is an automatic music transcription method that uses a two-level hierarchical frequency-time Transformer architecture. The first hierarchy includes a convolutional block in the time axis, a Transformer encoder in the frequency axis, and a Transformer decoder that converts the dimension in the frequency axis. The output is then fed into the second hierarchy which consists of another Transformer encoder in the time axis. We evaluated our method with the widely used MAPS and MAESTRO v3.0.0 datasets, and it demonstrated state-of-the-art performance on all the F1-scores of the metrics among Frame, Note, Note with Offset, and Note with Offset and Velocity estimations.

{{</citation>}}


### (86/93) Edge Storage Management Recipe with Zero-Shot Data Compression for Road Anomaly Detection (YeongHyeon Park et al., 2023)

{{<citation>}}

YeongHyeon Park, Uju Gim, Myung Jin Kim. (2023)  
**Edge Storage Management Recipe with Zero-Shot Data Compression for Road Anomaly Detection**  

---
Primary Category: cs.SD  
Categories: cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Anomaly Detection, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2307.04298v1)  

---


**ABSTRACT**  
Recent studies show edge computing-based road anomaly detection systems which may also conduct data collection simultaneously. However, the edge computers will have small data storage but we need to store the collected audio samples for a long time in order to update existing models or develop a novel method. Therefore, we should consider an approach for efficient storage management methods while preserving high-fidelity audio. A hardware-perspective approach, such as using a low-resolution microphone, is an intuitive way to reduce file size but is not recommended because it fundamentally cuts off high-frequency components. On the other hand, a computational file compression approach that encodes collected high-resolution audio into a compact code should be recommended because it also provides a corresponding decoding method. Motivated by this, we propose a way of simple yet effective pre-trained autoencoder-based data compression method. The pre-trained autoencoder is trained for the purpose of audio super-resolution so it can be utilized to encode or decode any arbitrary sampling rate. Moreover, it will reduce the communication cost for data transmission from the edge to the central server. Via the comparative experiments, we confirm that the zero-shot audio compression and decompression highly preserve anomaly detection performance while enhancing storage and transmission efficiency.

{{</citation>}}


## cs.IR (3)



### (87/93) InPars Toolkit: A Unified and Reproducible Synthetic Data Generation Pipeline for Neural Information Retrieval (Hugo Abonizio et al., 2023)

{{<citation>}}

Hugo Abonizio, Luiz Bonifacio, Vitor Jeronymo, Roberto Lotufo, Jakub Zavrel, Rodrigo Nogueira. (2023)  
**InPars Toolkit: A Unified and Reproducible Synthetic Data Generation Pipeline for Neural Information Retrieval**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Information Retrieval, Language Model  
[Paper Link](http://arxiv.org/abs/2307.04601v1)  

---


**ABSTRACT**  
Recent work has explored Large Language Models (LLMs) to overcome the lack of training data for Information Retrieval (IR) tasks. The generalization abilities of these models have enabled the creation of synthetic in-domain data by providing instructions and a few examples on a prompt. InPars and Promptagator have pioneered this approach and both methods have demonstrated the potential of using LLMs as synthetic data generators for IR tasks. This makes them an attractive solution for IR tasks that suffer from a lack of annotated data. However, the reproducibility of these methods was limited, because InPars' training scripts are based on TPUs -- which are not widely accessible -- and because the code for Promptagator was not released and its proprietary LLM is not publicly accessible. To fully realize the potential of these methods and make their impact more widespread in the research community, the resources need to be accessible and easy to reproduce by researchers and practitioners. Our main contribution is a unified toolkit for end-to-end reproducible synthetic data generation research, which includes generation, filtering, training and evaluation. Additionally, we provide an interface to IR libraries widely used by the community and support for GPU. Our toolkit not only reproduces the InPars method and partially reproduces Promptagator, but also provides a plug-and-play functionality allowing the use of different LLMs, exploring filtering methods and finetuning various reranker models on the generated data. We also made available all the synthetic data generated in this work for the 18 different datasets in the BEIR benchmark which took more than 2,000 GPU hours to be generated as well as the reranker models finetuned on the synthetic data. Code and data are available at https://github.com/zetaalphavector/InPars

{{</citation>}}


### (88/93) Alleviating Matthew Effect of Offline Reinforcement Learning in Interactive Recommendation (Chongming Gao et al., 2023)

{{<citation>}}

Chongming Gao, Kexin Huang, Jiawei Chen, Yuan Zhang, Biao Li, Peng Jiang, Shiqi Wang, Zhong Zhang, Xiangnan He. (2023)  
**Alleviating Matthew Effect of Offline Reinforcement Learning in Interactive Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.04571v1)  

---


**ABSTRACT**  
Offline reinforcement learning (RL), a technology that offline learns a policy from logged data without the need to interact with online environments, has become a favorable choice in decision-making processes like interactive recommendation. Offline RL faces the value overestimation problem. To address it, existing methods employ conservatism, e.g., by constraining the learned policy to be close to behavior policies or punishing the rarely visited state-action pairs. However, when applying such offline RL to recommendation, it will cause a severe Matthew effect, i.e., the rich get richer and the poor get poorer, by promoting popular items or categories while suppressing the less popular ones. It is a notorious issue that needs to be addressed in practical recommender systems.   In this paper, we aim to alleviate the Matthew effect in offline RL-based recommendation. Through theoretical analyses, we find that the conservatism of existing methods fails in pursuing users' long-term satisfaction. It inspires us to add a penalty term to relax the pessimism on states with high entropy of the logging policy and indirectly penalizes actions leading to less diverse states. This leads to the main technical contribution of the work: Debiased model-based Offline RL (DORL) method. Experiments show that DORL not only captures user interests well but also alleviates the Matthew effect. The implementation is available via https://github.com/chongminggao/DORL-codes.

{{</citation>}}


### (89/93) Graph Contrastive Learning with Multi-Objective for Personalized Product Retrieval in Taobao Search (Longbin Li et al., 2023)

{{<citation>}}

Longbin Li, Chao Zhang, Sen Li, Yun Zhong, Qingwen Liu, Xiaoyi Zeng. (2023)  
**Graph Contrastive Learning with Multi-Objective for Personalized Product Retrieval in Taobao Search**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2307.04322v1)  

---


**ABSTRACT**  
In e-commerce search, personalized retrieval is a crucial technique for improving user shopping experience. Recent works in this domain have achieved significant improvements by the representation learning paradigm, e.g., embedding-based retrieval (EBR) and collaborative filtering (CF). EBR methods do not sufficiently exploit the useful collaborative signal and are difficult to learn the representations of long-tail item well. Graph-based CF methods improve personalization by modeling collaborative signal within the user click graph. However, existing Graph-based methods ignore user's multiple behaviours, such as click/purchase and the relevance constraint between user behaviours and items.In this paper, we propose a Graph Contrastive Learning with Multi-Objective (GCL-MO) collaborative filtering model, which solves the problems of weak relevance and incomplete personalization in e-commerce search. Specifically, GCL-MO builds a homogeneous graph of items and then optimizes a multi-objective function of personalization and relevance. Moreover, we propose a modified contrastive loss for multi-objectives graph learning, which avoids the mutual suppression among positive samples and thus improves the generalization and robustness of long-tail item representations. These learned item embeddings are then used for personalized retrieval by constructing an efficient offline-to-online inverted table. GCL-MO outperforms the online collaborative filtering baseline in both offline/online experimental metrics and shows a significant improvement in the online A/B testing of Taobao search.

{{</citation>}}


## eess.IV (1)



### (90/93) Cluster-Induced Mask Transformers for Effective Opportunistic Gastric Cancer Screening on Non-contrast CT Scans (Mingze Yuan et al., 2023)

{{<citation>}}

Mingze Yuan, Yingda Xia, Xin Chen, Jiawen Yao, Junli Wang, Mingyan Qiu, Hexin Dong, Jingren Zhou, Bin Dong, Le Lu, Li Zhang, Zaiyi Liu, Ling Zhang. (2023)  
**Cluster-Induced Mask Transformers for Effective Opportunistic Gastric Cancer Screening on Non-contrast CT Scans**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.04525v2)  

---


**ABSTRACT**  
Gastric cancer is the third leading cause of cancer-related mortality worldwide, but no guideline-recommended screening test exists. Existing methods can be invasive, expensive, and lack sensitivity to identify early-stage gastric cancer. In this study, we explore the feasibility of using a deep learning approach on non-contrast CT scans for gastric cancer detection. We propose a novel cluster-induced Mask Transformer that jointly segments the tumor and classifies abnormality in a multi-task manner. Our model incorporates learnable clusters that encode the texture and shape prototypes of gastric cancer, utilizing self- and cross-attention to interact with convolutional features. In our experiments, the proposed method achieves a sensitivity of 85.0% and specificity of 92.6% for detecting gastric tumors on a hold-out test set consisting of 100 patients with cancer and 148 normal. In comparison, two radiologists have an average sensitivity of 73.5% and specificity of 84.3%. We also obtain a specificity of 97.7% on an external test set with 903 normal cases. Our approach performs comparably to established state-of-the-art gastric cancer screening tools like blood testing and endoscopy, while also being more sensitive in detecting early-stage cancer. This demonstrates the potential of our approach as a novel, non-invasive, low-cost, and accurate method for opportunistic gastric cancer screening.

{{</citation>}}


## physics.comp-ph (1)



### (91/93) Graph Convolutional Networks for Simulating Multi-phase Flow and Transport in Porous Media (Jiamin Jiang et al., 2023)

{{<citation>}}

Jiamin Jiang, Bo Guo. (2023)  
**Graph Convolutional Networks for Simulating Multi-phase Flow and Transport in Porous Media**  

---
Primary Category: physics.comp-ph  
Categories: cs-LG, physics-comp-ph, physics.comp-ph  
Keywords: Graph Convolutional Network  
[Paper Link](http://arxiv.org/abs/2307.04449v1)  

---


**ABSTRACT**  
Numerical simulation of multi-phase fluid dynamics in porous media is critical for many subsurface applications. Data-driven surrogate modeling provides computationally inexpensive alternatives to high-fidelity numerical simulators. While the commonly used convolutional neural networks (CNNs) are powerful in approximating partial differential equation solutions, it remains challenging for CNNs to handle irregular and unstructured simulation meshes. However, subsurface simulation models often involve unstructured meshes with complex mesh geometries, which limits the application of CNNs. To address this challenge, here we construct surrogate models based on Graph Convolutional Networks (GCNs) to approximate the spatial-temporal solutions of multi-phase flow and transport processes. We propose a new GCN architecture suited to the hyperbolic character of the coupled PDE system, to better capture the saturation dynamics. Results of 2D heterogeneous test cases show that our surrogates predict the evolutions of the pressure and saturation states with high accuracy, and the predicted rollouts remain stable for multiple timesteps. Moreover, the GCN-based models generalize well to irregular domain geometries and unstructured meshes that are unseen in the training dataset.

{{</citation>}}


## cs.LO (1)



### (92/93) Some Preliminary Steps Towards Metaverse Logic (Antonio L. Furtado et al., 2023)

{{<citation>}}

Antonio L. Furtado, Marco A. Casanova, Edirlei Soares de Lima. (2023)  
**Some Preliminary Steps Towards Metaverse Logic**  

---
Primary Category: cs.LO  
Categories: F-4-1, cs-AI, cs-LO, cs.LO  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2307.05574v1)  

---


**ABSTRACT**  
Assuming that the term 'metaverse' could be understood as a computer-based implementation of multiverse applications, we started to look in the present work for a logic that would be powerful enough to handle the situations arising both in the real and in the fictional underlying application domains. Realizing that first-order logic fails to account for the unstable behavior of even the most simpleminded information system domains, we resorted to non-conventional extensions, in an attempt to sketch a minimal composite logic strategy. The discussion was kept at a rather informal level, always trying to convey the intuition behind the theoretical notions in natural language terms, and appealing to an AI agent, namely ChatGPT, in the hope that algorithmic and common-sense approaches can be usefully combined.

{{</citation>}}


## eess.AS (1)



### (93/93) A Demand-Driven Perspective on Generative Audio AI (Sangshin Oh et al., 2023)

{{<citation>}}

Sangshin Oh, Minsung Kang, Hyeongi Moon, Keunwoo Choi, Ben Sangbae Chon. (2023)  
**A Demand-Driven Perspective on Generative Audio AI**  

---
Primary Category: eess.AS  
Categories: cs-AI, eess-AS, eess.AS  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.04292v1)  

---


**ABSTRACT**  
To achieve successful deployment of AI research, it is crucial to understand the demands of the industry. In this paper, we present the results of a survey conducted with professional audio engineers, in order to determine research priorities and define various research tasks. We also summarize the current challenges in audio quality and controllability based on the survey. Our analysis emphasizes that the availability of datasets is currently the main bottleneck for achieving high-quality audio generation. Finally, we suggest potential solutions for some revealed issues with empirical evidence.

{{</citation>}}
