---
draft: false
title: "arXiv @ 2023.12.21"
date: 2023-12-21
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.12.21"
    identifier: arxiv_20231221
    parent: 202312_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.LG (32)](#cslg-32)
- [cs.SI (6)](#cssi-6)
- [cs.CV (34)](#cscv-34)
- [cs.CL (27)](#cscl-27)
- [stat.ML (3)](#statml-3)
- [cs.SE (5)](#csse-5)
- [eess.SP (1)](#eesssp-1)
- [cs.CE (1)](#csce-1)
- [cs.CR (2)](#cscr-2)
- [cs.CY (1)](#cscy-1)
- [math.NA (1)](#mathna-1)
- [cs.IR (3)](#csir-3)
- [cs.AI (11)](#csai-11)
- [astro-ph.EP (1)](#astro-phep-1)
- [q-bio.QM (1)](#q-bioqm-1)
- [cs.GR (1)](#csgr-1)
- [cs.MM (1)](#csmm-1)
- [cs.SD (3)](#cssd-3)
- [eess.IV (2)](#eessiv-2)
- [cs.HC (3)](#cshc-3)
- [cs.DC (2)](#csdc-2)
- [cs.MA (1)](#csma-1)
- [cs.NI (3)](#csni-3)
- [econ.GN (1)](#econgn-1)
- [math.OC (1)](#mathoc-1)
- [q-fin.PM (1)](#q-finpm-1)

## cs.LG (32)



### (1/148) Can Transformers Learn Sequential Function Classes In Context? (Ryan Campbell et al., 2023)

{{<citation>}}

Ryan Campbell, Emma Guo, Evan Hu, Reya Vir, Ethan Hsiao. (2023)  
**Can Transformers Learn Sequential Function Classes In Context?**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: GPT, NLP, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.12655v2)  

---


**ABSTRACT**  
In-context learning (ICL) has revolutionized the capabilities of transformer models in NLP. In our project, we extend the understanding of the mechanisms underpinning ICL by exploring whether transformers can learn from sequential, non-textual function class data distributions. We introduce a novel sliding window sequential function class and employ toy-sized transformers with a GPT-2 architecture to conduct our experiments. Our analysis indicates that these models can indeed leverage ICL when trained on non-textual sequential function classes. Additionally, our experiments with randomized y-label sequences highlights that transformers retain some ICL capabilities even when the label associations are obfuscated. We provide evidence that transformers can reason with and understand sequentiality encoded within function classes, as reflected by the effective learning of our proposed tasks. Our results also show that the performance deteriorated with increasing randomness in the labels, though not to the extent one might expect, implying a potential robustness of learned sequentiality against label noise. Future research may want to look into how previous explanations of transformers, such as induction heads and task vectors, relate to sequentiality in ICL in these toy examples. Our investigation lays the groundwork for further research into how transformers process and perceive sequential data.

{{</citation>}}


### (2/148) SimQ-NAS: Simultaneous Quantization Policy and Neural Architecture Search (Sharath Nittur Sridhar et al., 2023)

{{<citation>}}

Sharath Nittur Sridhar, Maciej Szankin, Fang Chen, Sairam Sundaresan, Anthony Sarah. (2023)  
**SimQ-NAS: Simultaneous Quantization Policy and Neural Architecture Search**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: BERT, Quantization  
[Paper Link](http://arxiv.org/abs/2312.13301v1)  

---


**ABSTRACT**  
Recent one-shot Neural Architecture Search algorithms rely on training a hardware-agnostic super-network tailored to a specific task and then extracting efficient sub-networks for different hardware platforms. Popular approaches separate the training of super-networks from the search for sub-networks, often employing predictors to alleviate the computational overhead associated with search. Additionally, certain methods also incorporate the quantization policy within the search space. However, while the quantization policy search for convolutional neural networks is well studied, the extension of these methods to transformers and especially foundation models remains under-explored. In this paper, we demonstrate that by using multi-objective search algorithms paired with lightly trained predictors, we can efficiently search for both the sub-network architecture and the corresponding quantization policy and outperform their respective baselines across different performance objectives such as accuracy, model size, and latency. Specifically, we demonstrate that our approach performs well across both uni-modal (ViT and BERT) and multi-modal (BEiT-3) transformer-based architectures as well as convolutional architectures (ResNet). For certain networks, we demonstrate an improvement of up to $4.80x$ and $3.44x$ for latency and model size respectively, without degradation in accuracy compared to the fully quantized INT8 baselines.

{{</citation>}}


### (3/148) BadRL: Sparse Targeted Backdoor Attack Against Reinforcement Learning (Jing Cui et al., 2023)

{{<citation>}}

Jing Cui, Yufei Han, Yuzhe Ma, Jianbin Jiao, Junge Zhang. (2023)  
**BadRL: Sparse Targeted Backdoor Attack Against Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CR, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.12585v1)  

---


**ABSTRACT**  
Backdoor attacks in reinforcement learning (RL) have previously employed intense attack strategies to ensure attack success. However, these methods suffer from high attack costs and increased detectability. In this work, we propose a novel approach, BadRL, which focuses on conducting highly sparse backdoor poisoning efforts during training and testing while maintaining successful attacks. Our algorithm, BadRL, strategically chooses state observations with high attack values to inject triggers during training and testing, thereby reducing the chances of detection. In contrast to the previous methods that utilize sample-agnostic trigger patterns, BadRL dynamically generates distinct trigger patterns based on targeted state observations, thereby enhancing its effectiveness. Theoretical analysis shows that the targeted backdoor attack is always viable and remains stealthy under specific assumptions. Empirical results on various classic RL tasks illustrate that BadRL can substantially degrade the performance of a victim agent with minimal poisoning efforts 0.003% of total training steps) during training and infrequent attacks during testing.

{{</citation>}}


### (4/148) Comprehensive Validation on Reweighting Samples for Bias Mitigation via AIF360 (Christina Hastings Blow et al., 2023)

{{<citation>}}

Christina Hastings Blow, Lijun Qian, Camille Gibson, Pamela Obiomon, Xishuang Dong. (2023)  
**Comprehensive Validation on Reweighting Samples for Bias Mitigation via AIF360**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI, Bias  
[Paper Link](http://arxiv.org/abs/2312.12560v1)  

---


**ABSTRACT**  
Fairness AI aims to detect and alleviate bias across the entire AI development life cycle, encompassing data curation, modeling, evaluation, and deployment-a pivotal aspect of ethical AI implementation. Addressing data bias, particularly concerning sensitive attributes like gender and race, reweighting samples proves efficient for fairness AI. This paper contributes a systematic examination of reweighting samples for traditional machine learning (ML) models, employing five models for binary classification on the Adult Income and COMPUS datasets with various protected attributes. The study evaluates prediction results using five fairness metrics, uncovering the nuanced and model-specific nature of reweighting sample effectiveness in achieving fairness in traditional ML models, as well as revealing the complexity of bias dynamics.

{{</citation>}}


### (5/148) Sample Efficient Reinforcement Learning with Partial Dynamics Knowledge (Meshal Alharbi et al., 2023)

{{<citation>}}

Meshal Alharbi, Mardavij Roozbehani, Munther Dahleh. (2023)  
**Sample Efficient Reinforcement Learning with Partial Dynamics Knowledge**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, math-OC, stat-ML  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.12558v1)  

---


**ABSTRACT**  
The problem of sample complexity of online reinforcement learning is often studied in the literature without taking into account any partial knowledge about the system dynamics that could potentially accelerate the learning process. In this paper, we study the sample complexity of online Q-learning methods when some prior knowledge about the dynamics is available or can be learned efficiently. We focus on systems that evolve according to an additive disturbance model of the form $S_{h+1} = f(S_h, A_h) + W_h$, where $f$ represents the underlying system dynamics, and $W_h$ are unknown disturbances independent of states and actions. In the setting of finite episodic Markov decision processes with $S$ states, $A$ actions, and episode length $H$, we present an optimistic Q-learning algorithm that achieves $\tilde{\mathcal{O}}(\text{Poly}(H)\sqrt{T})$ regret under perfect knowledge of $f$, where $T$ is the total number of interactions with the system. This is in contrast to the typical $\tilde{\mathcal{O}}(\text{Poly}(H)\sqrt{SAT})$ regret for existing Q-learning methods. Further, if only a noisy estimate $\hat{f}$ of $f$ is available, our method can learn an approximately optimal policy in a number of samples that is independent of the cardinalities of state and action spaces. The sub-optimality gap depends on the approximation error $\hat{f}-f$, as well as the Lipschitz constant of the corresponding optimal value function. Our approach does not require modeling of the transition probabilities and enjoys the same memory complexity as model-free methods.

{{</citation>}}


### (6/148) Blood Glucose Level Prediction: A Graph-based Explainable Method with Federated Learning (Chengzhe Piao et al., 2023)

{{<citation>}}

Chengzhe Piao, Ken Li. (2023)  
**Blood Glucose Level Prediction: A Graph-based Explainable Method with Federated Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.12541v1)  

---


**ABSTRACT**  
In the UK, approximately 400,000 people with type 1 diabetes (T1D) rely on insulin delivery due to insufficient pancreatic insulin production. Managing blood glucose (BG) levels is crucial, with continuous glucose monitoring (CGM) playing a key role. CGM, tracking BG every 5 minutes, enables effective blood glucose level prediction (BGLP) by considering factors like carbohydrate intake and insulin delivery.   Recent research has focused on developing sequential models for BGLP using historical BG data, incorporating additional attributes such as carbohydrate intake, insulin delivery, and time. These methods have shown notable success in BGLP, with some providing temporal explanations. However, they often lack clear correlations between attributes and their impact on BGLP. Additionally, some methods raise privacy concerns by aggregating participant data to learn population patterns.   Addressing these limitations, we introduced a graph attentive memory (GAM) model, combining a graph attention network (GAT) with a gated recurrent unit (GRU). GAT applies graph attention to model attribute correlations, offering transparent, dynamic attribute relationships. Attention weights dynamically gauge attribute significance over time. To ensure privacy, we employed federated learning (FL), facilitating secure population pattern analysis.   Our method was validated using the OhioT1DM'18 and OhioT1DM'20 datasets from 12 participants, focusing on 6 key attributes. We demonstrated our model's stability and effectiveness through hyperparameter impact analysis.

{{</citation>}}


### (7/148) Chasing Fairness in Graphs: A GNN Architecture Perspective (Zhimeng Jiang et al., 2023)

{{<citation>}}

Zhimeng Jiang, Xiaotian Han, Chao Fan, Zirui Liu, Na Zou, Ali Mostafavi, Xia Hu. (2023)  
**Chasing Fairness in Graphs: A GNN Architecture Perspective**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CY, cs-LG, cs.LG  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2312.12369v1)  

---


**ABSTRACT**  
There has been significant progress in improving the performance of graph neural networks (GNNs) through enhancements in graph data, model architecture design, and training strategies. For fairness in graphs, recent studies achieve fair representations and predictions through either graph data pre-processing (e.g., node feature masking, and topology rewiring) or fair training strategies (e.g., regularization, adversarial debiasing, and fair contrastive learning). How to achieve fairness in graphs from the model architecture perspective is less explored. More importantly, GNNs exhibit worse fairness performance compared to multilayer perception since their model architecture (i.e., neighbor aggregation) amplifies biases. To this end, we aim to achieve fairness via a new GNN architecture. We propose \textsf{F}air \textsf{M}essage \textsf{P}assing (FMP) designed within a unified optimization framework for GNNs. Notably, FMP \textit{explicitly} renders sensitive attribute usage in \textit{forward propagation} for node classification task using cross-entropy loss without data pre-processing. In FMP, the aggregation is first adopted to utilize neighbors' information and then the bias mitigation step explicitly pushes demographic group node presentation centers together. In this way, FMP scheme can aggregate useful information from neighbors and mitigate bias to achieve better fairness and prediction tradeoff performance. Experiments on node classification tasks demonstrate that the proposed FMP outperforms several baselines in terms of fairness and accuracy on three real-world datasets. The code is available in {\url{https://github.com/zhimengj0326/FMP}}.

{{</citation>}}


### (8/148) H-ensemble: An Information Theoretic Approach to Reliable Few-Shot Multi-Source-Free Transfer (Yanru Wu et al., 2023)

{{<citation>}}

Yanru Wu, Jianning Wang, Weida Wang, Yang Li. (2023)  
**H-ensemble: An Information Theoretic Approach to Reliable Few-Shot Multi-Source-Free Transfer**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2312.12489v1)  

---


**ABSTRACT**  
Multi-source transfer learning is an effective solution to data scarcity by utilizing multiple source tasks for the learning of the target task. However, access to source data and model details is limited in the era of commercial models, giving rise to the setting of multi-source-free (MSF) transfer learning that aims to leverage source domain knowledge without such access. As a newly defined problem paradigm, MSF transfer learning remains largely underexplored and not clearly formulated. In this work, we adopt an information theoretic perspective on it and propose a framework named H-ensemble, which dynamically learns the optimal linear combination, or ensemble, of source models for the target task, using a generalization of maximal correlation regression. The ensemble weights are optimized by maximizing an information theoretic metric for transferability. Compared to previous works, H-ensemble is characterized by: 1) its adaptability to a novel and realistic MSF setting for few-shot target tasks, 2) theoretical reliability, 3) a lightweight structure easy to interpret and adapt. Our method is empirically validated by ablation studies, along with extensive comparative analysis with other task ensemble and transfer learning methods. We show that the H-ensemble can successfully learn the optimal task ensemble, as well as outperform prior arts.

{{</citation>}}


### (9/148) Prompt-based Domain Discrimination for Multi-source Time Series Domain Adaptation (Junxiang Wang et al., 2023)

{{<citation>}}

Junxiang Wang, Guangji Bai, Wei Cheng, Zhengzhang Chen, Liang Zhao, Haifeng Chen. (2023)  
**Prompt-based Domain Discrimination for Multi-source Time Series Domain Adaptation**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2312.12276v1)  

---


**ABSTRACT**  
Time series domain adaptation stands as a pivotal and intricate challenge with diverse applications, including but not limited to human activity recognition, sleep stage classification, and machine fault diagnosis. Despite the numerous domain adaptation techniques proposed to tackle this complex problem, their primary focus has been on the common representations of time series data. This concentration might inadvertently lead to the oversight of valuable domain-specific information originating from different source domains. To bridge this gap, we introduce POND, a novel prompt-based deep learning model designed explicitly for multi-source time series domain adaptation. POND is tailored to address significant challenges, notably: 1) The unavailability of a quantitative relationship between meta-data information and time series distributions, and 2) The dearth of exploration into extracting domain-specific meta-data information. In this paper, we present an instance-level prompt generator and a fidelity loss mechanism to facilitate the faithful learning of meta-data information. Additionally, we propose a domain discrimination technique to discern domain-specific meta-data information from multiple source domains. Our approach involves a simple yet effective meta-learning algorithm to optimize the objective efficiently. Furthermore, we augment the model's performance by incorporating the Mixture of Expert (MoE) technique. The efficacy and robustness of our proposed POND model are extensively validated through experiments across 50 scenarios encompassing five datasets, which demonstrates that our proposed POND model outperforms the state-of-the-art methods by up to $66\%$ on the F1-score.

{{</citation>}}


### (10/148) Emergence of In-Context Reinforcement Learning from Noise Distillation (Ilya Zisman et al., 2023)

{{<citation>}}

Ilya Zisman, Vladislav Kurenkov, Alexander Nikulin, Viacheslav Sinii, Sergey Kolesnikov. (2023)  
**Emergence of In-Context Reinforcement Learning from Noise Distillation**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.12275v1)  

---


**ABSTRACT**  
In-Context Reinforcement Learning is an emerging field with great potential for advancing Artificial Intelligence. Its core capability lies in generalizing to unseen tasks through interaction with the environment. To master these capabilities, an agent must be trained on specifically curated data that includes a policy improvement that an algorithm seeks to extract and then apply in context in the environment. However, for numerous tasks, training RL agents may be unfeasible, while obtaining human demonstrations can be relatively easy. Additionally, it is rare to be given the optimal policy, typically, only suboptimal demonstrations are available. We propose $AD^{\epsilon}$, a method that leverages demonstrations without policy improvement and enables multi-task in-context learning in the presence of a suboptimal demonstrator. This is achieved by artificially creating a history of incremental improvement, wherein noise is systematically introduced into the demonstrator's policy. Consequently, each successive transition illustrates a marginally better trajectory than the previous one. Our approach was tested on the Dark Room and Dark Key-to-Door environments, resulting in over a $\textbf{2}$x improvement compared to the best available policy in the data.

{{</citation>}}


### (11/148) Sharing is CAIRing: Characterizing Principles and Assessing Properties of Universal Privacy Evaluation for Synthetic Tabular Data (Tobias Hyrup et al., 2023)

{{<citation>}}

Tobias Hyrup, Anton Danholt Lautrup, Arthur Zimek, Peter Schneider-Kamp. (2023)  
**Sharing is CAIRing: Characterizing Principles and Assessing Properties of Universal Privacy Evaluation for Synthetic Tabular Data**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-CY, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.12216v1)  

---


**ABSTRACT**  
Data sharing is a necessity for innovative progress in many domains, especially in healthcare. However, the ability to share data is hindered by regulations protecting the privacy of natural persons. Synthetic tabular data provide a promising solution to address data sharing difficulties but does not inherently guarantee privacy. Still, there is a lack of agreement on appropriate methods for assessing the privacy-preserving capabilities of synthetic data, making it difficult to compare results across studies. To the best of our knowledge, this is the first work to identify properties that constitute good universal privacy evaluation metrics for synthetic tabular data. The goal of such metrics is to enable comparability across studies and to allow non-technical stakeholders to understand how privacy is protected. We identify four principles for the assessment of metrics: Comparability, Applicability, Interpretability, and Representativeness (CAIR). To quantify and rank the degree to which evaluation metrics conform to the CAIR principles, we design a rubric using a scale of 1-4. Each of the four properties is scored on four parameters, yielding 16 total dimensions. We study the applicability and usefulness of the CAIR principles and rubric by assessing a selection of metrics popular in other studies. The results provide granular insights into the strengths and weaknesses of existing metrics that not only rank the metrics but highlight areas of potential improvements. We expect that the CAIR principles will foster agreement among researchers and organizations on which universal privacy evaluation metrics are appropriate for synthetic tabular data.

{{</citation>}}


### (12/148) CUDC: A Curiosity-Driven Unsupervised Data Collection Method with Adaptive Temporal Distances for Offline Reinforcement Learning (Chenyu Sun et al., 2023)

{{<citation>}}

Chenyu Sun, Hangwei Qian, Chunyan Miao. (2023)  
**CUDC: A Curiosity-Driven Unsupervised Data Collection Method with Adaptive Temporal Distances for Offline Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, stat-ML  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.12191v1)  

---


**ABSTRACT**  
Offline reinforcement learning (RL) aims to learn an effective policy from a pre-collected dataset. Most existing works are to develop sophisticated learning algorithms, with less emphasis on improving the data collection process. Moreover, it is even challenging to extend the single-task setting and collect a task-agnostic dataset that allows an agent to perform multiple downstream tasks. In this paper, we propose a Curiosity-driven Unsupervised Data Collection (CUDC) method to expand feature space using adaptive temporal distances for task-agnostic data collection and ultimately improve learning efficiency and capabilities for multi-task offline RL. To achieve this, CUDC estimates the probability of the k-step future states being reachable from the current states, and adapts how many steps into the future that the dynamics model should predict. With this adaptive reachability mechanism in place, the feature representation can be diversified, and the agent can navigate itself to collect higher-quality data with curiosity. Empirically, CUDC surpasses existing unsupervised methods in efficiency and learning performance in various downstream offline RL tasks of the DeepMind control suite.

{{</citation>}}


### (13/148) Poincaré Differential Privacy for Hierarchy-Aware Graph Embedding (Yuecen Wei et al., 2023)

{{<citation>}}

Yuecen Wei, Haonan Yuan, Xingcheng Fu, Qingyun Sun, Hao Peng, Xianxian Li, Chunming Hu. (2023)  
**Poincaré Differential Privacy for Hierarchy-Aware Graph Embedding**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG  
Keywords: Embedding, GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2312.12183v2)  

---


**ABSTRACT**  
Hierarchy is an important and commonly observed topological property in real-world graphs that indicate the relationships between supervisors and subordinates or the organizational behavior of human groups. As hierarchy is introduced as a new inductive bias into the Graph Neural Networks (GNNs) in various tasks, it implies latent topological relations for attackers to improve their inference attack performance, leading to serious privacy leakage issues. In addition, existing privacy-preserving frameworks suffer from reduced protection ability in hierarchical propagation due to the deficiency of adaptive upper-bound estimation of the hierarchical perturbation boundary. It is of great urgency to effectively leverage the hierarchical property of data while satisfying privacy guarantees. To solve the problem, we propose the Poincar\'e Differential Privacy framework, named PoinDP, to protect the hierarchy-aware graph embedding based on hyperbolic geometry. Specifically, PoinDP first learns the hierarchy weights for each entity based on the Poincar\'e model in hyperbolic space. Then, the Personalized Hierarchy-aware Sensitivity is designed to measure the sensitivity of the hierarchical structure and adaptively allocate the privacy protection strength. Besides, the Hyperbolic Gaussian Mechanism (HGM) is proposed to extend the Gaussian mechanism in Euclidean space to hyperbolic space to realize random perturbations that satisfy differential privacy under the hyperbolic space metric. Extensive experiment results on five real-world datasets demonstrate the proposed PoinDP's advantages of effective privacy protection while maintaining good performance on the node classification task.

{{</citation>}}


### (14/148) Survey on Trustworthy Graph Neural Networks: From A Causal Perspective (Wenzhao Jiang et al., 2023)

{{<citation>}}

Wenzhao Jiang, Hao Liu, Hui Xiong. (2023)  
**Survey on Trustworthy Graph Neural Networks: From A Causal Perspective**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, stat-ME  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2312.12477v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) have emerged as powerful representation learning tools for capturing complex dependencies within diverse graph-structured data. Despite their success in a wide range of graph mining tasks, GNNs have raised serious concerns regarding their trustworthiness, including susceptibility to distribution shift, biases towards certain populations, and lack of explainability. Recently, integrating causal learning techniques into GNNs has sparked numerous ground-breaking studies since most of the trustworthiness issues can be alleviated by capturing the underlying data causality rather than superficial correlations. In this survey, we provide a comprehensive review of recent research efforts on causality-inspired GNNs. Specifically, we first present the key trustworthy risks of existing GNN models through the lens of causality. Moreover, we introduce a taxonomy of Causality-Inspired GNNs (CIGNNs) based on the type of causal learning capability they are equipped with, i.e., causal reasoning and causal representation learning. Besides, we systematically discuss typical methods within each category and demonstrate how they mitigate trustworthiness risks. Finally, we summarize useful resources and discuss several future directions, hoping to shed light on new research opportunities in this emerging field. The representative papers, along with open-source data and codes, are available in https://github.com/usail-hkust/Causality-Inspired-GNNs.

{{</citation>}}


### (15/148) Probabilistic Prediction of Longitudinal Trajectory Considering Driving Heterogeneity with Interpretability (Shuli Wang et al., 2023)

{{<citation>}}

Shuli Wang, Kun Gao, Lanfang Zhang, Yang Liu, Lei Chen. (2023)  
**Probabilistic Prediction of Longitudinal Trajectory Considering Driving Heterogeneity with Interpretability**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2312.12123v1)  

---


**ABSTRACT**  
Automated vehicles are envisioned to navigate safely in complex mixed-traffic scenarios alongside human-driven vehicles. To promise a high degree of safety, accurately predicting the maneuvers of surrounding vehicles and their future positions is a critical task and attracts much attention. However, most existing studies focused on reasoning about positional information based on objective historical trajectories without fully considering the heterogeneity of driving behaviors. Therefore, this study proposes a trajectory prediction framework that combines Mixture Density Networks (MDN) and considers the driving heterogeneity to provide probabilistic and personalized predictions. Specifically, based on a certain length of historical trajectory data, the situation-specific driving preferences of each driver are identified, where key driving behavior feature vectors are extracted to characterize heterogeneity in driving behavior among different drivers. With the inputs of the short-term historical trajectory data and key driving behavior feature vectors, a probabilistic LSTMMD-DBV model combined with LSTM-based encoder-decoder networks and MDN layers is utilized to carry out personalized predictions. Finally, the SHapley Additive exPlanations (SHAP) method is employed to interpret the trained model for predictions. The proposed framework is tested based on a wide-range vehicle trajectory dataset. The results indicate that the proposed model can generate probabilistic future trajectories with remarkably improved predictions compared to existing benchmark models. Moreover, the results confirm that the additional input of driving behavior feature vectors representing the heterogeneity of driving behavior could provide more information and thus contribute to improving the prediction accuracy.

{{</citation>}}


### (16/148) Curated LLM: Synergy of LLMs and Data Curation for tabular augmentation in ultra low-data regimes (Nabeel Seedat et al., 2023)

{{<citation>}}

Nabeel Seedat, Nicolas Huynh, Boris van Breugel, Mihaela van der Schaar. (2023)  
**Curated LLM: Synergy of LLMs and Data Curation for tabular augmentation in ultra low-data regimes**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.12112v1)  

---


**ABSTRACT**  
Machine Learning (ML) in low-data settings remains an underappreciated yet crucial problem. This challenge is pronounced in low-to-middle income countries where access to large datasets is often limited or even absent. Hence, data augmentation methods to increase the sample size of datasets needed for ML are key to unlocking the transformative potential of ML in data-deprived regions and domains. Unfortunately, the limited training set constrains traditional tabular synthetic data generators in their ability to generate a large and diverse augmented dataset needed for ML tasks. To address this technical challenge, we introduce CLLM, which leverages the prior knowledge of Large Language Models (LLMs) for data augmentation in the low-data regime. While diverse, not all the data generated by LLMs will help increase utility for a downstream task, as for any generative model. Consequently, we introduce a principled curation process, leveraging learning dynamics, coupled with confidence and uncertainty metrics, to obtain a high-quality dataset. Empirically, on multiple real-world datasets, we demonstrate the superior performance of LLMs in the low-data regime compared to conventional generators. We further show our curation mechanism improves the downstream performance for all generators, including LLMs. Additionally, we provide insights and understanding into the LLM generation and curation mechanism, shedding light on the features that enable them to output high-quality augmented datasets. CLLM paves the way for wider usage of ML in data scarce domains and regions, by allying the strengths of LLMs with a robust data-centric approach.

{{</citation>}}


### (17/148) Learning to Reweight for Graph Neural Network (Zhengyu Chen et al., 2023)

{{<citation>}}

Zhengyu Chen, Teng Xiao, Kun Kuang, Zheqi Lv, Min Zhang, Jinluan Yang, Chengqiang Lu, Hongxia Yang, Fei Wu. (2023)  
**Learning to Reweight for Graph Neural Network**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2312.12475v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) show promising results for graph tasks. However, existing GNNs' generalization ability will degrade when there exist distribution shifts between testing and training graph data. The cardinal impetus underlying the severe degeneration is that the GNNs are architected predicated upon the I.I.D assumptions. In such a setting, GNNs are inclined to leverage imperceptible statistical correlations subsisting in the training set to predict, albeit it is a spurious correlation. In this paper, we study the problem of the generalization ability of GNNs in Out-Of-Distribution (OOD) settings. To solve this problem, we propose the Learning to Reweight for Generalizable Graph Neural Network (L2R-GNN) to enhance the generalization ability for achieving satisfactory performance on unseen testing graphs that have different distributions with training graphs. We propose a novel nonlinear graph decorrelation method, which can substantially improve the out-of-distribution generalization ability and compares favorably to previous methods in restraining the over-reduced sample size. The variables of the graph representation are clustered based on the stability of the correlation, and the graph decorrelation method learns weights to remove correlations between the variables of different clusters rather than any two variables. Besides, we interpose an efficacious stochastic algorithm upon bi-level optimization for the L2R-GNN framework, which facilitates simultaneously learning the optimal weights and GNN parameters, and avoids the overfitting problem. Experimental results show that L2R-GNN greatly outperforms baselines on various graph prediction benchmarks under distribution shifts.

{{</citation>}}


### (18/148) XLand-MiniGrid: Scalable Meta-Reinforcement Learning Environments in JAX (Alexander Nikulin et al., 2023)

{{<citation>}}

Alexander Nikulin, Vladislav Kurenkov, Ilya Zisman, Artem Agarkov, Viacheslav Sinii, Sergey Kolesnikov. (2023)  
**XLand-MiniGrid: Scalable Meta-Reinforcement Learning Environments in JAX**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.12044v1)  

---


**ABSTRACT**  
We present XLand-MiniGrid, a suite of tools and grid-world environments for meta-reinforcement learning research inspired by the diversity and depth of XLand and the simplicity and minimalism of MiniGrid. XLand-Minigrid is written in JAX, designed to be highly scalable, and can potentially run on GPU or TPU accelerators, democratizing large-scale experimentation with limited resources. To demonstrate the generality of our library, we have implemented some well-known single-task environments as well as new meta-learning environments capable of generating $10^8$ distinct tasks. We have empirically shown that the proposed environments can scale up to $2^{13}$ parallel instances on the GPU, reaching tens of millions of steps per second.

{{</citation>}}


### (19/148) A Performance Evaluation of a Quantized Large Language Model on Various Smartphones (Tolga Çöplü et al., 2023)

{{<citation>}}

Tolga Çöplü, Marc Loedi, Arto Bendiken, Mykhailo Makohin, Joshua J. Bouw, Stephen Cobb. (2023)  
**A Performance Evaluation of a Quantized Large Language Model on Various Smartphones**  

---
Primary Category: cs.LG  
Categories: I-2-7, cs-AI, cs-LG, cs.LG  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2312.12472v1)  

---


**ABSTRACT**  
This paper explores the feasibility and performance of on-device large language model (LLM) inference on various Apple iPhone models. Amidst the rapid evolution of generative AI, on-device LLMs offer solutions to privacy, security, and connectivity challenges inherent in cloud-based models. Leveraging existing literature on running multi-billion parameter LLMs on resource-limited devices, our study examines the thermal effects and interaction speeds of a high-performing LLM across different smartphone generations. We present real-world performance results, providing insights into on-device inference capabilities.

{{</citation>}}


### (20/148) When Model Meets New Normals: Test-time Adaptation for Unsupervised Time-series Anomaly Detection (Dongmin Kim et al., 2023)

{{<citation>}}

Dongmin Kim, Sunghyun Park, Jaegul Choo. (2023)  
**When Model Meets New Normals: Test-time Adaptation for Unsupervised Time-series Anomaly Detection**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2312.11976v1)  

---


**ABSTRACT**  
Time-series anomaly detection deals with the problem of detecting anomalous timesteps by learning normality from the sequence of observations. However, the concept of normality evolves over time, leading to a "new normal problem", where the distribution of normality can be changed due to the distribution shifts between training and test data. This paper highlights the prevalence of the new normal problem in unsupervised time-series anomaly detection studies. To tackle this issue, we propose a simple yet effective test-time adaptation strategy based on trend estimation and a self-supervised approach to learning new normalities during inference. Extensive experiments on real-world benchmarks demonstrate that incorporating the proposed strategy into the anomaly detector consistently improves the model's performance compared to the baselines, leading to robustness to the distribution shifts.

{{</citation>}}


### (21/148) Time-Series Contrastive Learning against False Negatives and Class Imbalance (Xiyuan Jin et al., 2023)

{{<citation>}}

Xiyuan Jin, Jing Wang, Lei Liu, Youfang Lin. (2023)  
**Time-Series Contrastive Learning against False Negatives and Class Imbalance**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2312.11939v1)  

---


**ABSTRACT**  
As an exemplary self-supervised approach for representation learning, time-series contrastive learning has exhibited remarkable advancements in contemporary research. While recent contrastive learning strategies have focused on how to construct appropriate positives and negatives, in this study, we conduct theoretical analysis and find they have overlooked the fundamental issues: false negatives and class imbalance inherent in the InfoNCE loss-based framework. Therefore, we introduce a straightforward modification grounded in the SimCLR framework, universally adaptable to models engaged in the instance discrimination task. By constructing instance graphs to facilitate interactive learning among instances, we emulate supervised contrastive learning via the multiple-instances discrimination task, mitigating the harmful impact of false negatives. Moreover, leveraging the graph structure and few-labeled data, we perform semi-supervised consistency classification and enhance the representative ability of minority classes. We compared our method with the most popular time-series contrastive learning methods on four real-world time-series datasets and demonstrated our significant advantages in overall performance.

{{</citation>}}


### (22/148) Dynamic Frequency Domain Graph Convolutional Network for Traffic Forecasting (Yujie Li et al., 2023)

{{<citation>}}

Yujie Li, Zezhi Shao, Yongjun Xu, Qiang Qiu, Zhaogang Cao, Fei Wang. (2023)  
**Dynamic Frequency Domain Graph Convolutional Network for Traffic Forecasting**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Graph Convolutional Network  
[Paper Link](http://arxiv.org/abs/2312.11933v1)  

---


**ABSTRACT**  
Complex spatial dependencies in transportation networks make traffic prediction extremely challenging. Much existing work is devoted to learning dynamic graph structures among sensors, and the strategy of mining spatial dependencies from traffic data, known as data-driven, tends to be an intuitive and effective approach. However, Time-Shift of traffic patterns and noise induced by random factors hinder data-driven spatial dependence modeling. In this paper, we propose a novel dynamic frequency domain graph convolution network (DFDGCN) to capture spatial dependencies. Specifically, we mitigate the effects of time-shift by Fourier transform, and introduce the identity embedding of sensors and time embedding when capturing data for graph learning since traffic data with noise is not entirely reliable. The graph is combined with static predefined and self-adaptive graphs during graph convolution to predict future traffic data through classical causal convolutions. Extensive experiments on four real-world datasets demonstrate that our model is effective and outperforms the baselines.

{{</citation>}}


### (23/148) Empowering Dual-Level Graph Self-Supervised Pretraining with Motif Discovery (Pengwei Yan et al., 2023)

{{<citation>}}

Pengwei Yan, Kaisong Song, Zhuoren Jiang, Yangyang Kang, Tianqianjin Lin, Changlong Sun, Xiaozhong Liu. (2023)  
**Empowering Dual-Level Graph Self-Supervised Pretraining with Motif Discovery**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SI, cs.LG, stat-ME  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2312.11927v1)  

---


**ABSTRACT**  
While self-supervised graph pretraining techniques have shown promising results in various domains, their application still experiences challenges of limited topology learning, human knowledge dependency, and incompetent multi-level interactions. To address these issues, we propose a novel solution, Dual-level Graph self-supervised Pretraining with Motif discovery (DGPM), which introduces a unique dual-level pretraining structure that orchestrates node-level and subgraph-level pretext tasks. Unlike prior approaches, DGPM autonomously uncovers significant graph motifs through an edge pooling module, aligning learned motif similarities with graph kernel-based similarities. A cross-matching task enables sophisticated node-motif interactions and novel representation learning. Extensive experiments on 15 datasets validate DGPM's effectiveness and generalizability, outperforming state-of-the-art methods in unsupervised representation learning and transfer learning settings. The autonomously discovered motifs demonstrate the potential of DGPM to enhance robustness and interpretability.

{{</citation>}}


### (24/148) A Case Study in CUDA Kernel Fusion: Implementing FlashAttention-2 on NVIDIA Hopper Architecture using the CUTLASS Library (Ganesh Bikshandi et al., 2023)

{{<citation>}}

Ganesh Bikshandi, Jay Shah. (2023)  
**A Case Study in CUDA Kernel Fusion: Implementing FlashAttention-2 on NVIDIA Hopper Architecture using the CUTLASS Library**  

---
Primary Category: cs.LG  
Categories: cs-DC, cs-LG, cs.LG  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.11918v1)  

---


**ABSTRACT**  
We provide an optimized implementation of the forward pass of FlashAttention-2, a popular memory-aware scaled dot-product attention algorithm, as a custom fused CUDA kernel targeting NVIDIA Hopper architecture and written using the open-source CUTLASS library. In doing so, we explain the challenges and techniques involved in fusing online-softmax with back-to-back GEMM kernels, utilizing the Hopper-specific Tensor Memory Accelerator (TMA) and Warpgroup Matrix-Multiply-Accumulate (WGMMA) instructions, defining and transforming CUTLASS Layouts and Tensors, overlapping copy and GEMM operations, and choosing optimal tile sizes for the Q, K and V attention matrices while balancing the register pressure and shared memory utilization. In head-to-head benchmarks on a single H100 PCIe GPU for some common choices of hyperparameters, we observe 20-50% higher FLOPs/s over a version of FlashAttention-2 optimized for last-generation NVIDIA Ampere architecture.

{{</citation>}}


### (25/148) Distilling Autoregressive Models to Obtain High-Performance Non-Autoregressive Solvers for Vehicle Routing Problems with Faster Inference Speed (Yubin Xiao et al., 2023)

{{<citation>}}

Yubin Xiao, Di Wang, Boyang Li, Mingzhao Wang, Xuan Wu, Changliang Zhou, You Zhou. (2023)  
**Distilling Autoregressive Models to Obtain High-Performance Non-Autoregressive Solvers for Vehicle Routing Problems with Faster Inference Speed**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2312.12469v1)  

---


**ABSTRACT**  
Neural construction models have shown promising performance for Vehicle Routing Problems (VRPs) by adopting either the Autoregressive (AR) or Non-Autoregressive (NAR) learning approach. While AR models produce high-quality solutions, they generally have a high inference latency due to their sequential generation nature. Conversely, NAR models generate solutions in parallel with a low inference latency but generally exhibit inferior performance. In this paper, we propose a generic Guided Non-Autoregressive Knowledge Distillation (GNARKD) method to obtain high-performance NAR models having a low inference latency. GNARKD removes the constraint of sequential generation in AR models while preserving the learned pivotal components in the network architecture to obtain the corresponding NAR models through knowledge distillation. We evaluate GNARKD by applying it to three widely adopted AR models to obtain NAR VRP solvers for both synthesized and real-world instances. The experimental results demonstrate that GNARKD significantly reduces the inference time (4-5 times faster) with acceptable performance drop (2-3\%). To the best of our knowledge, this study is first-of-its-kind to obtain NAR VRP solvers from AR ones through knowledge distillation.

{{</citation>}}


### (26/148) Short-Term Multi-Horizon Line Loss Rate Forecasting of a Distribution Network Using Attention-GCN-LSTM (Jie Liu et al., 2023)

{{<citation>}}

Jie Liu, Yijia Cao, Yong Li, Yixiu Guo, Wei Deng. (2023)  
**Short-Term Multi-Horizon Line Loss Rate Forecasting of a Distribution Network Using Attention-GCN-LSTM**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, eess-SP  
Keywords: Attention, Graph Convolutional Network, LSTM  
[Paper Link](http://arxiv.org/abs/2312.11898v1)  

---


**ABSTRACT**  
Accurately predicting line loss rates is vital for effective line loss management in distribution networks, especially over short-term multi-horizons ranging from one hour to one week. In this study, we propose Attention-GCN-LSTM, a novel method that combines Graph Convolutional Networks (GCN), Long Short-Term Memory (LSTM), and a three-level attention mechanism to address this challenge. By capturing spatial and temporal dependencies, our model enables accurate forecasting of line loss rates across multiple horizons. Through comprehensive evaluation using real-world data from 10KV feeders, our Attention-GCN-LSTM model consistently outperforms existing algorithms, exhibiting superior performance in terms of prediction accuracy and multi-horizon forecasting. This model holds significant promise for enhancing line loss management in distribution networks.

{{</citation>}}


### (27/148) Sparse is Enough in Fine-tuning Pre-trained Large Language Model (Weixi Song et al., 2023)

{{<citation>}}

Weixi Song, Zuchao Li, Lefei Zhang, Hai Zhao, Bo Du. (2023)  
**Sparse is Enough in Fine-tuning Pre-trained Large Language Model**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: GLUE, Language Model  
[Paper Link](http://arxiv.org/abs/2312.11875v1)  

---


**ABSTRACT**  
With the prevalence of pre-training-fine-tuning paradigm, how to efficiently adapt the pre-trained model to the downstream tasks has been an intriguing issue. Parameter-Efficient Fine-Tuning (PEFT) methods have been proposed for low-cost adaptation, including Adapters, Bia-only, and the recently widely used Low-Rank Adaptation. Although these methods have demonstrated their effectiveness to some extent and have been widely applied, the underlying principles are still unclear. In this paper, we reveal the transition of loss landscape in the downstream domain from random initialization to pre-trained initialization, that is, from low-amplitude oscillation to high-amplitude oscillation. The parameter gradients exhibit a property akin to sparsity, where a small fraction of components dominate the total gradient norm, for instance, 1% of the components account for 99% of the gradient. This property ensures that the pre-trained model can easily find a flat minimizer which guarantees the model's ability to generalize even with a low number of trainable parameters. Based on this, we propose a gradient-based sparse fine-tuning algorithm, named Sparse Increment Fine-Tuning (SIFT), and validate its effectiveness on a range of tasks including the GLUE Benchmark and Instruction-tuning. The code is accessible at https://github.com/song-wx/SIFT/.

{{</citation>}}


### (28/148) Learning Flexible Body Collision Dynamics with Hierarchical Contact Mesh Transformer (Youn-Yeol Yu et al., 2023)

{{<citation>}}

Youn-Yeol Yu, Jeongwhan Choi, Woojin Cho, Kookjin Lee, Nayong Kim, Kiseok Chang, ChangSeung Woo, Ilho Kim, SeokWoo Lee, Joon Young Yang, Sooyoung Yoon, Noseong Park. (2023)  
**Learning Flexible Body Collision Dynamics with Hierarchical Contact Mesh Transformer**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CE, cs-LG, cs.LG  
Keywords: GNN, Transformer  
[Paper Link](http://arxiv.org/abs/2312.12467v1)  

---


**ABSTRACT**  
Recently, many mesh-based graph neural network (GNN) models have been proposed for modeling complex high-dimensional physical systems. Remarkable achievements have been made in significantly reducing the solving time compared to traditional numerical solvers. These methods are typically designed to i) reduce the computational cost in solving physical dynamics and/or ii) propose techniques to enhance the solution accuracy in fluid and rigid body dynamics. However, it remains under-explored whether they are effective in addressing the challenges of flexible body dynamics, where instantaneous collisions occur within a very short timeframe. In this paper, we present Hierarchical Contact Mesh Transformer (HCMT), which uses hierarchical mesh structures and can learn long-range dependencies (occurred by collisions) among spatially distant positions of a body -- two close positions in a higher-level mesh corresponds to two distant positions in a lower-level mesh. HCMT enables long-range interactions, and the hierarchical mesh structure quickly propagates collision effects to faraway positions. To this end, it consists of a contact mesh Transformer and a hierarchical mesh Transformer (CMT and HMT, respectively). Lastly, we propose a flexible body dynamics dataset, consisting of trajectories that reflect experimental settings frequently used in the display industry for product designs. We also compare the performance of several baselines using well-known benchmark datasets. Our results show that HCMT provides significant performance improvements over existing methods.

{{</citation>}}


### (29/148) Neural Network Approximation for Pessimistic Offline Reinforcement Learning (Di Wu et al., 2023)

{{<citation>}}

Di Wu, Yuling Jiao, Li Shen, Haizhao Yang, Xiliang Lu. (2023)  
**Neural Network Approximation for Pessimistic Offline Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, stat-ML  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.11863v1)  

---


**ABSTRACT**  
Deep reinforcement learning (RL) has shown remarkable success in specific offline decision-making scenarios, yet its theoretical guarantees are still under development. Existing works on offline RL theory primarily emphasize a few trivial settings, such as linear MDP or general function approximation with strong assumptions and independent data, which lack guidance for practical use. The coupling of deep learning and Bellman residuals makes this problem challenging, in addition to the difficulty of data dependence. In this paper, we establish a non-asymptotic estimation error of pessimistic offline RL using general neural network approximation with $\mathcal{C}$-mixing data regarding the structure of networks, the dimension of datasets, and the concentrability of data coverage, under mild assumptions. Our result shows that the estimation error consists of two parts: the first converges to zero at a desired rate on the sample size with partially controllable concentrability, and the second becomes negligible if the residual constraint is tight. This result demonstrates the explicit efficiency of deep adversarial offline RL frameworks. We utilize the empirical process tool for $\mathcal{C}$-mixing sequences and the neural network approximation theory for the H\"{o}lder class to achieve this. We also develop methods to bound the Bellman estimation error caused by function approximation with empirical Bellman constraint perturbations. Additionally, we present a result that lessens the curse of dimensionality using data with low intrinsic dimensionality and function classes with low complexity. Our estimation provides valuable insights into the development of deep offline RL and guidance for algorithm model design.

{{</citation>}}


### (30/148) SimCalib: Graph Neural Network Calibration based on Similarity between Nodes (Boshi Tang et al., 2023)

{{<citation>}}

Boshi Tang, Zhiyong Wu, Xixin Wu, Qiaochu Huang, Jun Chen, Shun Lei, Helen Meng. (2023)  
**SimCalib: Graph Neural Network Calibration based on Similarity between Nodes**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SI, cs.LG  
Keywords: GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2312.11858v1)  

---


**ABSTRACT**  
Graph neural networks (GNNs) have exhibited impressive performance in modeling graph data as exemplified in various applications. Recently, the GNN calibration problem has attracted increasing attention, especially in cost-sensitive scenarios. Previous work has gained empirical insights on the issue, and devised effective approaches for it, but theoretical supports still fall short. In this work, we shed light on the relationship between GNN calibration and nodewise similarity via theoretical analysis. A novel calibration framework, named SimCalib, is accordingly proposed to consider similarity between nodes at global and local levels. At the global level, the Mahalanobis distance between the current node and class prototypes is integrated to implicitly consider similarity between the current node and all nodes in the same class. At the local level, the similarity of node representation movement dynamics, quantified by nodewise homophily and relative degree, is considered. Informed about the application of nodewise movement patterns in analyzing nodewise behavior on the over-smoothing problem, we empirically present a possible relationship between over-smoothing and GNN calibration problem. Experimentally, we discover a correlation between nodewise similarity and model calibration improvement, in alignment with our theoretical results. Additionally, we conduct extensive experiments investigating different design factors and demonstrate the effectiveness of our proposed SimCalib framework for GNN calibration by achieving state-of-the-art performance on 14 out of 16 benchmarks.

{{</citation>}}


### (31/148) The Validity of a Machine Learning-Based Video Game in the Objective Screening of Attention Deficit Hyperactivity Disorder in Children Aged 5 to 12 Years (Zeinab Zakani et al., 2023)

{{<citation>}}

Zeinab Zakani, Hadi Moradi, Sogand Ghasemzadeh, Maryam Riazi, Fatemeh Mortazavi. (2023)  
**The Validity of a Machine Learning-Based Video Game in the Objective Screening of Attention Deficit Hyperactivity Disorder in Children Aged 5 to 12 Years**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.11832v1)  

---


**ABSTRACT**  
Objective: Early identification of ADHD is necessary to provide the opportunity for timely treatment. However, screening the symptoms of ADHD on a large scale is not easy. This study aimed to validate a video game (FishFinder) for the screening of ADHD using objective measurement of the core symptoms of this disorder. Method: The FishFinder measures attention and impulsivity through in-game performance and evaluates the child's hyperactivity using smartphone motion sensors. This game was tested on 26 children with ADHD and 26 healthy children aged 5 to 12 years. A Support Vector Machine was employed to detect children with ADHD. results: This system showed 92.3% accuracy, 90% sensitivity, and 93.7% specificity using a combination of in-game and movement features. Conclusions: The FishFinder demonstrated a strong ability to identify ADHD in children. So, this game can be used as an affordable, accessible, and enjoyable method for the objective screening of ADHD.

{{</citation>}}


### (32/148) An Adaptive Placement and Parallelism Framework for Accelerating RLHF Training (Youshao Xiao et al., 2023)

{{<citation>}}

Youshao Xiao, Weichang Wu, Zhenglei Zhou, Fagui Mao, Shangchun Zhao, Lin Ju, Lei Liang, Xiaolu Zhang, Jun Zhou. (2023)  
**An Adaptive Placement and Parallelism Framework for Accelerating RLHF Training**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: AI, ChatGPT, GPT, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.11819v1)  

---


**ABSTRACT**  
Recently, ChatGPT or InstructGPT like large language models (LLM) has made a significant impact in the AI world. These models are incredibly versatile, capable of performing language tasks on par or even exceeding the capabilities of human experts. Many works have attempted to reproduce the complex InstructGPT's RLHF (Reinforcement Learning with Human Feedback) training pipeline. However, the mainstream distributed RLHF training methods typically adopt a fixed model placement strategy, referred to as the Flattening strategy. This strategy treats all four models involved in RLHF as a single entity and places them on all devices, regardless of their differences. Unfortunately, this strategy exacerbates the generation bottlenecks in the RLHF training and degrades the overall training efficiency. To address these issues, we propose an adaptive model placement framework that offers two flexible model placement strategies. These strategies allow for the agile allocation of models across devices in a fine-grained manner. The Interleaving strategy helps reduce memory redundancy and communication costs during RLHF training. On the other hand, the Separation strategy improves the throughput of model training by separating the training and generation stages of the RLHF pipeline. Notably, this framework seamlessly integrates with other mainstream techniques for acceleration and enables automatic hyperparameter search. Extensive experiments have demonstrated that our Interleaving and Separation strategies can achieve notable improvements up to 11x, compared to the current state-of-the-art (SOTA) approaches. These experiments encompassed a wide range of training scenarios, involving models of varying sizes and devices of different scales. The results highlight the effectiveness and superiority of our approaches in accelerating the training of distributed RLHF.

{{</citation>}}


## cs.SI (6)



### (33/148) Toxic Bias: Perspective API Misreads German as More Toxic (Gianluca Nogara et al., 2023)

{{<citation>}}

Gianluca Nogara, Francesco Pierri, Stefano Cresci, Luca Luceri, Petter Törnberg, Silvia Giordano. (2023)  
**Toxic Bias: Perspective API Misreads German as More Toxic**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Bias, Google  
[Paper Link](http://arxiv.org/abs/2312.12651v1)  

---


**ABSTRACT**  
Proprietary public APIs play a crucial and growing role as research tools among social scientists. Among such APIs, Google's machine learning-based Perspective API is extensively utilized for assessing the toxicity of social media messages, providing both an important resource for researchers and automatic content moderation. However, this paper exposes an important bias in Perspective API concerning German language text. Through an in-depth examination of several datasets, we uncover intrinsic language biases within the multilingual model of Perspective API. We find that the toxicity assessment of German content produces significantly higher toxicity levels than other languages. This finding is robust across various translations, topics, and data sources, and has significant consequences for both research and moderation strategies that rely on Perspective API. For instance, we show that, on average, four times more tweets and users would be moderated when using the German language compared to their English translation. Our findings point to broader risks associated with the widespread use of proprietary APIs within the computational social sciences.

{{</citation>}}


### (34/148) Fairness and Consensus in a Gossip Model of Social Networks (Joan S. Betancourt et al., 2023)

{{<citation>}}

Joan S. Betancourt, Jesús Aranda, Juan Fco. Díaz, Frank Valencia. (2023)  
**Fairness and Consensus in a Gossip Model of Social Networks**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Social Network  
[Paper Link](http://arxiv.org/abs/2312.12251v1)  

---


**ABSTRACT**  
We present a new formal model for opinion learning in social networks. Our model is based on DeGroot, but it allows for asynchronous interaction between agents. We show that our model might not reach consensus under standard weak fairness assumption for distributed systems. However, we show that it reaches consensus under strong-connectedness, absence of puppet agents and a new condition called bounded wait. We study several notions of fairness and their implications for consensus, introducing bounded fairness and $m$-consecutive bounded fairness. As an important corollary, we obtain consensus for random executions of the model. We also show that our model can be generalized to allow for dynamic influence between agents, where consensus with connectivity and bounded wait holds as long as the influence is bounded.

{{</citation>}}


### (35/148) Potentials of ChatGPT for Annotating Vaccine Related Tweets (Md. Rafiul Biswas et al., 2023)

{{<citation>}}

Md. Rafiul Biswas, Farida Mohsen, Zubair Shah, Wajdi Zaghouani. (2023)  
**Potentials of ChatGPT for Annotating Vaccine Related Tweets**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2312.12016v1)  

---


**ABSTRACT**  
This study evaluates ChatGPT's performance in annotating vaccine-related Arabic tweets by comparing its annotations with human annotations. A dataset of 2,100 tweets representing various factors contributing to vaccine hesitancy was examined. Two domain experts annotated the data, with a third resolving conflicts. ChatGPT was then employed to annotate the same dataset using specific prompts for each factor. The ChatGPT annotations were evaluated through zero-shot, one-shot, and few-shot learning tests, with an average accuracy of 82.14%, 83.85%, and 85.57%, respectively. Precision averaged around 86%, minimizing false positives. The average recall and F1-score ranged from 0.74 to 0.80 and 0.65 to 0.93, respectively. AUC for zero-shot, one-shot, and few-shot learning was 0.79, 0.80, and 0.83. In cases of ambiguity, both human annotators and ChatGPT faced challenges. These findings suggest that ChatGPT holds promise as a tool for annotating vaccine-related tweets.

{{</citation>}}


### (36/148) Analyzing Public Reactions, Perceptions, and Attitudes during the MPox Outbreak: Findings from Topic Modeling of Tweets (Nirmalya Thakur et al., 2023)

{{<citation>}}

Nirmalya Thakur, Yuvraj Nihal Duggal, Zihui Liu. (2023)  
**Analyzing Public Reactions, Perceptions, and Attitudes during the MPox Outbreak: Findings from Topic Modeling of Tweets**  

---
Primary Category: cs.SI  
Categories: cs-AI, cs-CL, cs-CY, cs-SI, cs.SI  
Keywords: Topic Model, Topic Modeling, Twitter  
[Paper Link](http://arxiv.org/abs/2312.11895v1)  

---


**ABSTRACT**  
The recent outbreak of the MPox virus has resulted in a tremendous increase in the usage of Twitter. Prior works in this area of research have primarily focused on the sentiment analysis and content analysis of these Tweets, and the few works that have focused on topic modeling have multiple limitations. This paper aims to address this research gap and makes two scientific contributions to this field. First, it presents the results of performing Topic Modeling on 601,432 Tweets about the 2022 Mpox outbreak that were posted on Twitter between 7 May 2022 and 3 March 2023. The results indicate that the conversations on Twitter related to Mpox during this time range may be broadly categorized into four distinct themes - Views and Perspectives about Mpox, Updates on Cases and Investigations about Mpox, Mpox and the LGBTQIA+ Community, and Mpox and COVID-19. Second, the paper presents the findings from the analysis of these Tweets. The results show that the theme that was most popular on Twitter (in terms of the number of Tweets posted) during this time range was Views and Perspectives about Mpox. This was followed by the theme of Mpox and the LGBTQIA+ Community, which was followed by the themes of Mpox and COVID-19 and Updates on Cases and Investigations about Mpox, respectively. Finally, a comparison with related studies in this area of research is also presented to highlight the novelty and significance of this research work.

{{</citation>}}


### (37/148) Hierarchical and Incremental Structural Entropy Minimization for Unsupervised Social Event Detection (Yuwei Cao et al., 2023)

{{<citation>}}

Yuwei Cao, Hao Peng, Zhengtao Yu, Philip S. Yu. (2023)  
**Hierarchical and Incremental Structural Entropy Minimization for Unsupervised Social Event Detection**  

---
Primary Category: cs.SI  
Categories: cs-LG, cs-SI, cs.SI  
Keywords: Event Detection, GNN  
[Paper Link](http://arxiv.org/abs/2312.11891v1)  

---


**ABSTRACT**  
As a trending approach for social event detection, graph neural network (GNN)-based methods enable a fusion of natural language semantics and the complex social network structural information, thus showing SOTA performance. However, GNN-based methods can miss useful message correlations. Moreover, they require manual labeling for training and predetermining the number of events for prediction. In this work, we address social event detection via graph structural entropy (SE) minimization. While keeping the merits of the GNN-based methods, the proposed framework, HISEvent, constructs more informative message graphs, is unsupervised, and does not require the number of events given a priori. Specifically, we incrementally explore the graph neighborhoods using 1-dimensional (1D) SE minimization to supplement the existing message graph with edges between semantically related messages. We then detect events from the message graph by hierarchically minimizing 2-dimensional (2D) SE. Our proposed 1D and 2D SE minimization algorithms are customized for social event detection and effectively tackle the efficiency problem of the existing SE minimization algorithms. Extensive experiments show that HISEvent consistently outperforms GNN-based methods and achieves the new SOTA for social event detection under both closed- and open-set settings while being efficient and robust.

{{</citation>}}


### (38/148) A Large-Scale Dataset of Search Interests Related to Disease X Originating from Different Geographic Regions (Nirmalya Thakur et al., 2023)

{{<citation>}}

Nirmalya Thakur, Shuqi Cui, Kesha A. Patel, Isabella Hall, Yuvraj Nihal Duggal. (2023)  
**A Large-Scale Dataset of Search Interests Related to Disease X Originating from Different Geographic Regions**  

---
Primary Category: cs.SI  
Categories: cs-CY, cs-SI, cs.SI, physics-soc-ph  
Keywords: AI, Google  
[Paper Link](http://arxiv.org/abs/2312.11885v1)  

---


**ABSTRACT**  
The World Health Organization added Disease X to their shortlist of blueprint priority diseases to represent a hypothetical, unknown pathogen that could cause a future epidemic. During different virus outbreaks of the past, such as COVID-19, Influenza, Lyme Disease, and Zika virus, researchers from various disciplines utilized Google Trends to mine multimodal components of web behavior to study, investigate, and analyze the global awareness, preparedness, and response associated with these respective virus outbreaks. As the world prepares for Disease X, a dataset on web behavior related to Disease X would be crucial to contribute towards the timely advancement of research in this field. Furthermore, none of the prior works in this field have focused on the development of a dataset to compile relevant web behavior data, which would help to prepare for Disease X. To address these research challenges, this work presents a dataset of web behavior related to Disease X, which emerged from different geographic regions of the world, between February 2018 and August 2023. Specifically, this dataset presents the search interests related to Disease X from 94 geographic regions. The dataset was developed by collecting data using Google Trends. The relevant search interests for all these regions for each month in this time range are available in this dataset. This paper also discusses the compliance of this dataset with the FAIR principles of scientific data management. Finally, an analysis of this dataset is presented to uphold the applicability, relevance, and usefulness of this dataset for the investigation of different research questions in the interrelated fields of Big Data, Data Mining, Healthcare, Epidemiology, and Data Analysis with a specific focus on Disease X.

{{</citation>}}


## cs.CV (34)



### (39/148) RealCraft: Attention Control as A Solution for Zero-shot Long Video Editing (Shutong Jin et al., 2023)

{{<citation>}}

Shutong Jin, Ruiyu Wang, Florian T. Pokorny. (2023)  
**RealCraft: Attention Control as A Solution for Zero-shot Long Video Editing**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.12635v2)  

---


**ABSTRACT**  
Although large-scale text-to-image generative models have shown promising performance in synthesizing high-quality images, directly applying these models to image editing remains a significant challenge. This challenge is further amplified in video editing due to the additional dimension of time. Especially for editing real videos as it necessitates maintaining a stable semantic layout across the frames while executing localized edits precisely without disrupting the existing backgrounds. In this paper, we propose RealCraft, an attention-control-based method for zero-shot editing in real videos. By employing the object-centric manipulation of cross-attention between prompts and frames and spatial-temporal attention within the frames, we achieve precise shape-wise editing along with enhanced consistency. Our model can be used directly with Stable Diffusion and operates without the need for additional localized information. We showcase our zero-shot attention-control-based method across a range of videos, demonstrating localized, high-fidelity, shape-precise and time-consistent editing in videos of various lengths, up to 64 frames.

{{</citation>}}


### (40/148) Hierarchical Vision Transformers for Context-Aware Prostate Cancer Grading in Whole Slide Images (Clément Grisi et al., 2023)

{{<citation>}}

Clément Grisi, Geert Litjens, Jeroen van der Laak. (2023)  
**Hierarchical Vision Transformers for Context-Aware Prostate Cancer Grading in Whole Slide Images**  

---
Primary Category: cs.CV  
Categories: 68T07, I-2-10, cs-AI, cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.12619v1)  

---


**ABSTRACT**  
Vision Transformers (ViTs) have ushered in a new era in computer vision, showcasing unparalleled performance in many challenging tasks. However, their practical deployment in computational pathology has largely been constrained by the sheer size of whole slide images (WSIs), which result in lengthy input sequences. Transformers faced a similar limitation when applied to long documents, and Hierarchical Transformers were introduced to circumvent it. Given the analogous challenge with WSIs and their inherent hierarchical structure, Hierarchical Vision Transformers (H-ViTs) emerge as a promising solution in computational pathology. This work delves into the capabilities of H-ViTs, evaluating their efficiency for prostate cancer grading in WSIs. Our results show that they achieve competitive performance against existing state-of-the-art solutions.

{{</citation>}}


### (41/148) Weakly Supervised Open-Vocabulary Object Detection (Jianghang Lin et al., 2023)

{{<citation>}}

Jianghang Lin, Yunhang Shen, Bingquan Wang, Shaohui Lin, Ke Li, Liujuan Cao. (2023)  
**Weakly Supervised Open-Vocabulary Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2312.12437v1)  

---


**ABSTRACT**  
Despite weakly supervised object detection (WSOD) being a promising step toward evading strong instance-level annotations, its capability is confined to closed-set categories within a single training dataset. In this paper, we propose a novel weakly supervised open-vocabulary object detection framework, namely WSOVOD, to extend traditional WSOD to detect novel concepts and utilize diverse datasets with only image-level annotations. To achieve this, we explore three vital strategies, including dataset-level feature adaptation, image-level salient object localization, and region-level vision-language alignment. First, we perform data-aware feature extraction to produce an input-conditional coefficient, which is leveraged into dataset attribute prototypes to identify dataset bias and help achieve cross-dataset generalization. Second, a customized location-oriented weakly supervised region proposal network is proposed to utilize high-level semantic layouts from the category-agnostic segment anything model to distinguish object boundaries. Lastly, we introduce a proposal-concept synchronized multiple-instance network, i.e., object mining and refinement with visual-semantic alignment, to discover objects matched to the text embeddings of concepts. Extensive experiments on Pascal VOC and MS COCO demonstrate that the proposed WSOVOD achieves new state-of-the-art compared with previous WSOD methods in both close-set object localization and detection tasks. Meanwhile, WSOVOD enables cross-dataset and open-vocabulary learning to achieve on-par or even better performance than well-established fully-supervised open-vocabulary object detection (FSOVOD).

{{</citation>}}


### (42/148) A Challenger to GPT-4V? Early Explorations of Gemini in Visual Expertise (Chaoyou Fu et al., 2023)

{{<citation>}}

Chaoyou Fu, Renrui Zhang, Zihan Wang, Yubo Huang, Zhengye Zhang, Longtian Qiu, Gaoxiang Ye, Yunhang Shen, Mengdan Zhang, Peixian Chen, Sirui Zhao, Shaohui Lin, Deqiang Jiang, Di Yin, Peng Gao, Ke Li, Hongsheng Li, Xing Sun. (2023)  
**A Challenger to GPT-4V? Early Explorations of Gemini in Visual Expertise**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs-MM, cs.CV  
Keywords: AI, GPT, GPT-4, Google, Language Model  
[Paper Link](http://arxiv.org/abs/2312.12436v2)  

---


**ABSTRACT**  
The surge of interest towards Multi-modal Large Language Models (MLLMs), e.g., GPT-4V(ision) from OpenAI, has marked a significant trend in both academia and industry. They endow Large Language Models (LLMs) with powerful capabilities in visual understanding, enabling them to tackle diverse multi-modal tasks. Very recently, Google released Gemini, its newest and most capable MLLM built from the ground up for multi-modality. In light of the superior reasoning capabilities, can Gemini challenge GPT-4V's leading position in multi-modal learning? In this paper, we present a preliminary exploration of Gemini Pro's visual understanding proficiency, which comprehensively covers four domains: fundamental perception, advanced cognition, challenging vision tasks, and various expert capacities. We compare Gemini Pro with the state-of-the-art GPT-4V to evaluate its upper limits, along with the latest open-sourced MLLM, Sphinx, which reveals the gap between manual efforts and black-box systems. The qualitative samples indicate that, while GPT-4V and Gemini showcase different answering styles and preferences, they can exhibit comparable visual reasoning capabilities, and Sphinx still trails behind them concerning domain generalizability. Specifically, GPT-4V tends to elaborate detailed explanations and intermediate steps, and Gemini prefers to output a direct and concise answer. The quantitative evaluation on the popular MME benchmark also demonstrates the potential of Gemini to be a strong challenger to GPT-4V. Our early investigation of Gemini also observes some common issues of MLLMs, indicating that there still remains a considerable distance towards artificial general intelligence. Our project for tracking the progress of MLLM is released at https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models.

{{</citation>}}


### (43/148) The Endoscapes Dataset for Surgical Scene Segmentation, Object Detection, and Critical View of Safety Assessment: Official Splits and Benchmark (Aditya Murali et al., 2023)

{{<citation>}}

Aditya Murali, Deepak Alapatt, Pietro Mascagni, Armine Vardazaryan, Alain Garcia, Nariaki Okamoto, Guido Costamagna, Didier Mutter, Jacques Marescaux, Bernard Dallemagne, Nicolas Padoy. (2023)  
**The Endoscapes Dataset for Surgical Scene Segmentation, Object Detection, and Critical View of Safety Assessment: Official Splits and Benchmark**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2312.12429v1)  

---


**ABSTRACT**  
This technical report provides a detailed overview of Endoscapes, a dataset of laparoscopic cholecystectomy (LC) videos with highly intricate annotations targeted at automated assessment of the Critical View of Safety (CVS). Endoscapes comprises 201 LC videos with frames annotated sparsely but regularly with segmentation masks, bounding boxes, and CVS assessment by three different clinical experts. Altogether, there are 11090 frames annotated with CVS and 1933 frames annotated with tool and anatomy bounding boxes from the 201 videos, as well as an additional 422 frames from 50 of the 201 videos annotated with tool and anatomy segmentation masks. In this report, we provide detailed dataset statistics (size, class distribution, dataset splits, etc.) and a comprehensive performance benchmark for instance segmentation, object detection, and CVS prediction. The dataset and model checkpoints are publically available at https://github.com/CAMMA-public/Endoscapes.

{{</citation>}}


### (44/148) DDOS: The Drone Depth and Obstacle Segmentation Dataset (Benedikt Kolbeinsson et al., 2023)

{{<citation>}}

Benedikt Kolbeinsson, Krystian Mikolajczyk. (2023)  
**DDOS: The Drone Depth and Obstacle Segmentation Dataset**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2312.12494v1)  

---


**ABSTRACT**  
Accurate depth and semantic segmentation are crucial for various computer vision tasks. However, the scarcity of annotated real-world aerial datasets poses a significant challenge for training and evaluating robust models. Additionally, the detection and segmentation of thin objects, such as wires, cables, and fences, present a critical concern for ensuring the safe operation of drones. To address these limitations, we present a novel synthetic dataset specifically designed for depth and semantic segmentation tasks in aerial views. Leveraging photo-realistic rendering techniques, our dataset provides a valuable resource for training models using a synthetic-supervision training scheme while introducing new drone-specific metrics for depth accuracy.

{{</citation>}}


### (45/148) Jack of All Tasks, Master of Many: Designing General-purpose Coarse-to-Fine Vision-Language Model (Shraman Pramanick et al., 2023)

{{<citation>}}

Shraman Pramanick, Guangxing Han, Rui Hou, Sayan Nag, Ser-Nam Lim, Nicolas Ballas, Qifan Wang, Rama Chellappa, Amjad Almahairi. (2023)  
**Jack of All Tasks, Master of Many: Designing General-purpose Coarse-to-Fine Vision-Language Model**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.12423v1)  

---


**ABSTRACT**  
The ability of large language models (LLMs) to process visual inputs has given rise to general-purpose vision systems, unifying various vision-language (VL) tasks by instruction tuning. However, due to the enormous diversity in input-output formats in the vision domain, existing general-purpose models fail to successfully integrate segmentation and multi-image inputs with coarse-level tasks into a single framework. In this work, we introduce VistaLLM, a powerful visual system that addresses coarse- and fine-grained VL tasks over single and multiple input images using a unified framework. VistaLLM utilizes an instruction-guided image tokenizer that filters global embeddings using task descriptions to extract compressed and refined features from numerous images. Moreover, VistaLLM employs a gradient-aware adaptive sampling technique to represent binary segmentation masks as sequences, significantly improving over previously used uniform sampling. To bolster the desired capability of VistaLLM, we curate CoinIt, a comprehensive coarse-to-fine instruction tuning dataset with 6.8M samples. We also address the lack of multi-image grounding datasets by introducing a novel task, AttCoSeg (Attribute-level Co-Segmentation), which boosts the model's reasoning and grounding capability over multiple input images. Extensive experiments on a wide range of V- and VL tasks demonstrate the effectiveness of VistaLLM by achieving consistent state-of-the-art performance over strong baselines across all downstream tasks. Our project page can be found at https://shramanpramanick.github.io/VistaLLM/.

{{</citation>}}


### (46/148) LASA: Instance Reconstruction from Real Scans using A Large-scale Aligned Shape Annotation Dataset (Haolin Liu et al., 2023)

{{<citation>}}

Haolin Liu, Chongjie Ye, Yinyu Nie, Yingfan He, Xiaoguang Han. (2023)  
**LASA: Instance Reconstruction from Real Scans using A Large-scale Aligned Shape Annotation Dataset**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2312.12418v1)  

---


**ABSTRACT**  
Instance shape reconstruction from a 3D scene involves recovering the full geometries of multiple objects at the semantic instance level. Many methods leverage data-driven learning due to the intricacies of scene complexity and significant indoor occlusions. Training these methods often requires a large-scale, high-quality dataset with aligned and paired shape annotations with real-world scans. Existing datasets are either synthetic or misaligned, restricting the performance of data-driven methods on real data. To this end, we introduce LASA, a Large-scale Aligned Shape Annotation Dataset comprising 10,412 high-quality CAD annotations aligned with 920 real-world scene scans from ArkitScenes, created manually by professional artists. On this top, we propose a novel Diffusion-based Cross-Modal Shape Reconstruction (DisCo) method. It is empowered by a hybrid feature aggregation design to fuse multi-modal inputs and recover high-fidelity object geometries. Besides, we present an Occupancy-Guided 3D Object Detection (OccGOD) method and demonstrate that our shape annotations provide scene occupancy clues that can further improve 3D object detection. Supported by LASA, extensive experiments show that our methods achieve state-of-the-art performance in both instance-level scene reconstruction and 3D object detection tasks.

{{</citation>}}


### (47/148) Vision-Based Automatic Groceries Tracking System -- Smart Homes (Divya Mereddy, 2023)

{{<citation>}}

Divya Mereddy. (2023)  
**Vision-Based Automatic Groceries Tracking System -- Smart Homes**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.12486v1)  

---


**ABSTRACT**  
With advanced AI, while every industry is growing at rocket speed, the smart home industry has not reached the next generation. There is still a huge leap of innovation that needs to happen before we call a home a Smart home. A Smart home should predict residents' needs and fulfill them in a timely manner. One of the important tasks of maintaining a home is timely grocery tracking and supply maintenance. Grocery tracking models are very famous in the retail industry but they are nonexistent in the common household. Groceries detection in household refrigerators or storage closets is very complicated compared to retail shelving data. In this paper, home grocery tracking problem is resolved by combining retail shelving data and fruits dataset with real-time 360 view data points collected from home groceries storage. By integrating this vision-based object detection system along with supply chain and user food interest prediction systems, complete automation of groceries ordering can be achieved.

{{</citation>}}


### (48/148) VQA4CIR: Boosting Composed Image Retrieval with Visual Question Answering (Chun-Mei Feng et al., 2023)

{{<citation>}}

Chun-Mei Feng, Yang Bai, Tao Luo, Zhen Li, Salman Khan, Wangmeng Zuo, Xinxing Xu, Rick Siow Mong Goh, Yong Liu. (2023)  
**VQA4CIR: Boosting Composed Image Retrieval with Visual Question Answering**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: LLaMA, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2312.12273v1)  

---


**ABSTRACT**  
Albeit progress has been made in Composed Image Retrieval (CIR), we empirically find that a certain percentage of failure retrieval results are not consistent with their relative captions. To address this issue, this work provides a Visual Question Answering (VQA) perspective to boost the performance of CIR. The resulting VQA4CIR is a post-processing approach and can be directly plugged into existing CIR methods. Given the top-C retrieved images by a CIR method, VQA4CIR aims to decrease the adverse effect of the failure retrieval results being inconsistent with the relative caption. To find the retrieved images inconsistent with the relative caption, we resort to the "QA generation to VQA" self-verification pipeline. For QA generation, we suggest fine-tuning LLM (e.g., LLaMA) to generate several pairs of questions and answers from each relative caption. We then fine-tune LVLM (e.g., LLaVA) to obtain the VQA model. By feeding the retrieved image and question to the VQA model, one can find the images inconsistent with relative caption when the answer by VQA is inconsistent with the answer in the QA pair. Consequently, the CIR performance can be boosted by modifying the ranks of inconsistently retrieved images. Experimental results show that our proposed method outperforms state-of-the-art CIR methods on the CIRR and Fashion-IQ datasets.

{{</citation>}}


### (49/148) ST(OR)2: Spatio-Temporal Object Level Reasoning for Activity Recognition in the Operating Room (Idris Hamoud et al., 2023)

{{<citation>}}

Idris Hamoud, Muhammad Abdullah Jamal, Vinkle Srivastav, Didier Mutter, Nicolas Padoy, Omid Mohareri. (2023)  
**ST(OR)2: Spatio-Temporal Object Level Reasoning for Activity Recognition in the Operating Room**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Reasoning  
[Paper Link](http://arxiv.org/abs/2312.12250v1)  

---


**ABSTRACT**  
Surgical robotics holds much promise for improving patient safety and clinician experience in the Operating Room (OR). However, it also comes with new challenges, requiring strong team coordination and effective OR management. Automatic detection of surgical activities is a key requirement for developing AI-based intelligent tools to tackle these challenges. The current state-of-the-art surgical activity recognition methods however operate on image-based representations and depend on large-scale labeled datasets whose collection is time-consuming and resource-expensive. This work proposes a new sample-efficient and object-based approach for surgical activity recognition in the OR. Our method focuses on the geometric arrangements between clinicians and surgical devices, thus utilizing the significant object interaction dynamics in the OR. We conduct experiments in a low-data regime study for long video activity recognition. We also benchmark our method againstother object-centric approaches on clip-level action classification and show superior performance.

{{</citation>}}


### (50/148) GeomVerse: A Systematic Evaluation of Large Models for Geometric Reasoning (Mehran Kazemi et al., 2023)

{{<citation>}}

Mehran Kazemi, Hamidreza Alvari, Ankit Anand, Jialin Wu, Xi Chen, Radu Soricut. (2023)  
**GeomVerse: A Systematic Evaluation of Large Models for Geometric Reasoning**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2312.12241v1)  

---


**ABSTRACT**  
Large language models have shown impressive results for multi-hop mathematical reasoning when the input question is only textual. Many mathematical reasoning problems, however, contain both text and image. With the ever-increasing adoption of vision language models (VLMs), understanding their reasoning abilities for such problems is crucial. In this paper, we evaluate the reasoning capabilities of VLMs along various axes through the lens of geometry problems. We procedurally create a synthetic dataset of geometry questions with controllable difficulty levels along multiple axes, thus enabling a systematic evaluation. The empirical results obtained using our benchmark for state-of-the-art VLMs indicate that these models are not as capable in subjects like geometry (and, by generalization, other topics requiring similar reasoning) as suggested by previous benchmarks. This is made especially clear by the construction of our benchmark at various depth levels, since solving higher-depth problems requires long chains of reasoning rather than additional memorized knowledge. We release the dataset for further research in this area.

{{</citation>}}


### (51/148) Self-Supervised Detection of Perfect and Partial Input-Dependent Symmetries (Alonso Urbano et al., 2023)

{{<citation>}}

Alonso Urbano, David W. Romero. (2023)  
**Self-Supervised Detection of Perfect and Partial Input-Dependent Symmetries**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2312.12223v1)  

---


**ABSTRACT**  
Group equivariance ensures consistent responses to group transformations of the input, leading to more robust models and enhanced generalization capabilities. However, this property can lead to overly constrained models if the symmetries considered in the group differ from those observed in data. While common methods address this by determining the appropriate level of symmetry at the dataset level, they are limited to supervised settings and ignore scenarios in which multiple levels of symmetry co-exist in the same dataset. For instance, pictures of cars and planes exhibit different levels of rotation, yet both are included in the CIFAR-10 dataset. In this paper, we propose a method able to detect the level of symmetry of each input without the need for labels. To this end, we derive a sufficient and necessary condition to learn the distribution of symmetries in the data. Using the learned distribution, we generate pseudo-labels that allow us to learn the levels of symmetry of each input in a self-supervised manner. We validate the effectiveness of our approach on synthetic datasets with different per-class levels of symmetries e.g. MNISTMultiple, in which digits are uniformly rotated within a class-dependent interval. We demonstrate that our method can be used for practical applications such as the generation of standardized datasets in which the symmetries are not present, as well as the detection of out-of-distribution symmetries during inference. By doing so, both the generalization and robustness of non-equivariant models can be improved. Our code is publicly available at https://github.com/aurban0/ssl-sym.

{{</citation>}}


### (52/148) EarthVQA: Towards Queryable Earth via Relational Reasoning-Based Remote Sensing Visual Question Answering (Junjue Wang et al., 2023)

{{<citation>}}

Junjue Wang, Zhuo Zheng, Zihang Chen, Ailong Ma, Yanfei Zhong. (2023)  
**EarthVQA: Towards Queryable Earth via Relational Reasoning-Based Remote Sensing Visual Question Answering**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: QA, Question Answering, Reasoning  
[Paper Link](http://arxiv.org/abs/2312.12222v1)  

---


**ABSTRACT**  
Earth vision research typically focuses on extracting geospatial object locations and categories but neglects the exploration of relations between objects and comprehensive reasoning. Based on city planning needs, we develop a multi-modal multi-task VQA dataset (EarthVQA) to advance relational reasoning-based judging, counting, and comprehensive analysis. The EarthVQA dataset contains 6000 images, corresponding semantic masks, and 208,593 QA pairs with urban and rural governance requirements embedded. As objects are the basis for complex relational reasoning, we propose a Semantic OBject Awareness framework (SOBA) to advance VQA in an object-centric way. To preserve refined spatial locations and semantics, SOBA leverages a segmentation network for object semantics generation. The object-guided attention aggregates object interior features via pseudo masks, and bidirectional cross-attention further models object external relations hierarchically. To optimize object counting, we propose a numerical difference loss that dynamically adds difference penalties, unifying the classification and regression tasks. Experimental results show that SOBA outperforms both advanced general and remote sensing methods. We believe this dataset and framework provide a strong benchmark for Earth vision's complex analysis. The project page is at https://Junjue-Wang.github.io/homepage/EarthVQA.

{{</citation>}}


### (53/148) Zero-shot Building Attribute Extraction from Large-Scale Vision and Language Models (Fei Pan et al., 2023)

{{<citation>}}

Fei Pan, Sangryul Jeon, Brian Wang, Frank Mckenna, Stella X. Yu. (2023)  
**Zero-shot Building Attribute Extraction from Large-Scale Vision and Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2312.12479v1)  

---


**ABSTRACT**  
Existing building recognition methods, exemplified by BRAILS, utilize supervised learning to extract information from satellite and street-view images for classification and segmentation. However, each task module requires human-annotated data, hindering the scalability and robustness to regional variations and annotation imbalances. In response, we propose a new zero-shot workflow for building attribute extraction that utilizes large-scale vision and language models to mitigate reliance on external annotations. The proposed workflow contains two key components: image-level captioning and segment-level captioning for the building images based on the vocabularies pertinent to structural and civil engineering. These two components generate descriptive captions by computing feature representations of the image and the vocabularies, and facilitating a semantic match between the visual and textual representations. Consequently, our framework offers a promising avenue to enhance AI-driven captioning for building attribute extraction in the structural and civil engineering domains, ultimately reducing reliance on human annotations while bolstering performance and adaptability.

{{</citation>}}


### (54/148) Integrating Human Vision Perception in Vision Transformers for Classifying Waste Items (Akshat Kishore Shrivastava et al., 2023)

{{<citation>}}

Akshat Kishore Shrivastava, Tapan Kumar Gandhi. (2023)  
**Integrating Human Vision Perception in Vision Transformers for Classifying Waste Items**  

---
Primary Category: cs.CV  
Categories: 68T45, I-2; I-4, cs-CV, cs.CV, eess-IV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.12143v2)  

---


**ABSTRACT**  
In this paper, we propose an novel methodology aimed at simulating the learning phenomenon of nystagmus through the application of differential blurring on datasets. Nystagmus is a biological phenomenon that influences human vision throughout life, notably by diminishing head shake from infancy to adulthood. Leveraging this concept, we address the issue of waste classification, a pressing global concern. The proposed framework comprises two modules, with the second module closely resembling the original Vision Transformer, a state-of-the-art model model in classification tasks. The primary motivation behind our approach is to enhance the model's precision and adaptability, mirroring the real-world conditions that the human visual system undergoes. This novel methodology surpasses the standard Vision Transformer model in waste classification tasks, exhibiting an improvement with a margin of 2%. This improvement underscores the potential of our methodology in improving model precision by drawing inspiration from human vision perception. Further research in the proposed methodology could yield greater performance results, and can be extrapolated to other global issues.

{{</citation>}}


### (55/148) FontDiffuser: One-Shot Font Generation via Denoising Diffusion with Multi-Scale Content Aggregation and Style Contrastive Learning (Zhenhua Yang et al., 2023)

{{<citation>}}

Zhenhua Yang, Dezhi Peng, Yuxin Kong, Yuyi Zhang, Cong Yao, Lianwen Jin. (2023)  
**FontDiffuser: One-Shot Font Generation via Denoising Diffusion with Multi-Scale Content Aggregation and Style Contrastive Learning**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2312.12142v1)  

---


**ABSTRACT**  
Automatic font generation is an imitation task, which aims to create a font library that mimics the style of reference images while preserving the content from source images. Although existing font generation methods have achieved satisfactory performance, they still struggle with complex characters and large style variations. To address these issues, we propose FontDiffuser, a diffusion-based image-to-image one-shot font generation method, which innovatively models the font imitation task as a noise-to-denoise paradigm. In our method, we introduce a Multi-scale Content Aggregation (MCA) block, which effectively combines global and local content cues across different scales, leading to enhanced preservation of intricate strokes of complex characters. Moreover, to better manage the large variations in style transfer, we propose a Style Contrastive Refinement (SCR) module, which is a novel structure for style representation learning. It utilizes a style extractor to disentangle styles from images, subsequently supervising the diffusion model via a meticulously designed style contrastive loss. Extensive experiments demonstrate FontDiffuser's state-of-the-art performance in generating diverse characters and styles. It consistently excels on complex characters and large style changes compared to previous methods. The code is available at https://github.com/yeungchenwa/FontDiffuser.

{{</citation>}}


### (56/148) Object-Aware Domain Generalization for Object Detection (Wooju Lee et al., 2023)

{{<citation>}}

Wooju Lee, Dasol Hong, Hyungtae Lim, Hyun Myung. (2023)  
**Object-Aware Domain Generalization for Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2312.12133v1)  

---


**ABSTRACT**  
Single-domain generalization (S-DG) aims to generalize a model to unseen environments with a single-source domain. However, most S-DG approaches have been conducted in the field of classification. When these approaches are applied to object detection, the semantic features of some objects can be damaged, which can lead to imprecise object localization and misclassification. To address these problems, we propose an object-aware domain generalization (OA-DG) method for single-domain generalization in object detection. Our method consists of data augmentation and training strategy, which are called OA-Mix and OA-Loss, respectively. OA-Mix generates multi-domain data with multi-level transformation and object-aware mixing strategy. OA-Loss enables models to learn domain-invariant representations for objects and backgrounds from the original and OA-Mixed images. Our proposed method outperforms state-of-the-art works on standard benchmarks. Our code is available at https://github.com/WoojuLee24/OA-DG.

{{</citation>}}


### (57/148) ZS-SRT: An Efficient Zero-Shot Super-Resolution Training Method for Neural Radiance Fields (Xiang Feng et al., 2023)

{{<citation>}}

Xiang Feng, Yongbo He, Yubo Wang, Chengkai Wang, Zhenzhong Kuang, Jiajun Ding, Feiwei Qin, Jun Yu, Jianping Fan. (2023)  
**ZS-SRT: An Efficient Zero-Shot Super-Resolution Training Method for Neural Radiance Fields**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-GR, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2312.12122v1)  

---


**ABSTRACT**  
Neural Radiance Fields (NeRF) have achieved great success in the task of synthesizing novel views that preserve the same resolution as the training views. However, it is challenging for NeRF to synthesize high-quality high-resolution novel views with low-resolution training data. To solve this problem, we propose a zero-shot super-resolution training framework for NeRF. This framework aims to guide the NeRF model to synthesize high-resolution novel views via single-scene internal learning rather than requiring any external high-resolution training data. Our approach consists of two stages. First, we learn a scene-specific degradation mapping by performing internal learning on a pretrained low-resolution coarse NeRF. Second, we optimize a super-resolution fine NeRF by conducting inverse rendering with our mapping function so as to backpropagate the gradients from low-resolution 2D space into the super-resolution 3D sampling space. Then, we further introduce a temporal ensemble strategy in the inference phase to compensate for the scene estimation errors. Our method is featured on two points: (1) it does not consume high-resolution views or additional scene data to train super-resolution NeRF; (2) it can speed up the training process by adopting a coarse-to-fine strategy. By conducting extensive experiments on public datasets, we have qualitatively and quantitatively demonstrated the effectiveness of our method.

{{</citation>}}


### (58/148) Domain Generalization in LiDAR Semantic Segmentation Leveraged by Density Discriminative Feature Embedding (Jaeyeul Kim et al., 2023)

{{<citation>}}

Jaeyeul Kim, Jungwan Woo, Jeonghoon Kim, Sunghoon Im. (2023)  
**Domain Generalization in LiDAR Semantic Segmentation Leveraged by Density Discriminative Feature Embedding**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Embedding, Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2312.12098v1)  

---


**ABSTRACT**  
While significant progress has been achieved in LiDAR-based perception, domain generalization continues to present challenges, often resulting in reduced performance when encountering unfamiliar datasets due to domain discrepancies. One of the primary hurdles stems from the variability of LiDAR sensors, leading to inconsistencies in point cloud density distribution. Such inconsistencies can undermine the effectiveness of perception models. We address this challenge by introducing a new approach that acknowledges a fundamental characteristic of LiDAR: the variation in point density due to the distance from the LiDAR to the scene, and the number of beams relative to the field of view. Understanding this, we view each LiDAR's point cloud at various distances as having distinct density distributions, which can be consistent across different LiDAR models. With this insight, we propose the Density Discriminative Feature Embedding (DDFE) module, crafted to specifically extract features related to density while ensuring domain invariance across different LiDAR sensors. In addition, we introduce a straightforward but effective density augmentation technique, designed to broaden the density spectrum and enhance the capabilities of the DDFE. The proposed DDFE stands out as a versatile and lightweight domain generalization module. It can be seamlessly integrated into various 3D backbone networks, consistently outperforming existing state-of-the-art domain generalization approaches. We commit to releasing the source code publicly to foster community collaboration and advancement.

{{</citation>}}


### (59/148) Diffusing More Objects for Semi-Supervised Domain Adaptation with Less Labeling (Leander van den Heuvel et al., 2023)

{{<citation>}}

Leander van den Heuvel, Gertjan Burghouts, David W. Zhang, Gwenn Englebienne, Sabina B. van Rooij. (2023)  
**Diffusing More Objects for Semi-Supervised Domain Adaptation with Less Labeling**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2312.12000v1)  

---


**ABSTRACT**  
For object detection, it is possible to view the prediction of bounding boxes as a reverse diffusion process. Using a diffusion model, the random bounding boxes are iteratively refined in a denoising step, conditioned on the image. We propose a stochastic accumulator function that starts each run with random bounding boxes and combines the slightly different predictions. We empirically verify that this improves detection performance. The improved detections are leveraged on unlabelled images as weighted pseudo-labels for semi-supervised learning. We evaluate the method on a challenging out-of-domain test set. Our method brings significant improvements and is on par with human-selected pseudo-labels, while not requiring any human involvement.

{{</citation>}}


### (60/148) Continual Learning: Forget-free Winning Subnetworks for Video Representations (Haeyong Kang et al., 2023)

{{<citation>}}

Haeyong Kang, Jaehong Yoon, Sung Ju Hwang, Chang D. Yoo. (2023)  
**Continual Learning: Forget-free Winning Subnetworks for Video Representations**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2312.11973v1)  

---


**ABSTRACT**  
Inspired by the Regularized Lottery Ticket Hypothesis (RLTH), which highlights the presence of competitive subnetworks within dense networks for continual learning tasks, we introduce Winning Subnetworks (WSN). This approach utilizes reused weights in dense networks to enhance learning in Task Incremental Learning (TIL) scenarios. To mitigate overfitting in Few-Shot Class Incremental Learning (FSCIL), we have developed WSN variants referred to as the Soft subnetwork (SoftNet). Furthermore, addressing WSN's limitation of sparse reused weights in Video Incremental Learning (VIL), we propose the Fourier Subneural Operator (FSO). The FSO, operating in Fourier space, adaptively and compactly encodes videos, discovering reusable subnetworks with diverse bandwidths. We have applied FSO's Fourier representations to various continual learning contexts, including VIL, TIL, and FSCIL. Our extensive experiments across these scenarios demonstrate FSO's remarkable efficacy in continual learning, significantly enhancing task performance at various convolutional representational levels: it boosts performance in the higher layers for TIL and FSCIL and the lower layers for VIL.

{{</citation>}}


### (61/148) Expressive Forecasting of 3D Whole-body Human Motions (Pengxiang Ding et al., 2023)

{{<citation>}}

Pengxiang Ding, Qiongjie Cui, Min Zhang, Mengyuan Liu, Haofan Wang, Donglin Wang. (2023)  
**Expressive Forecasting of 3D Whole-body Human Motions**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.11972v1)  

---


**ABSTRACT**  
Human motion forecasting, with the goal of estimating future human behavior over a period of time, is a fundamental task in many real-world applications. However, existing works typically concentrate on predicting the major joints of the human body without considering the delicate movements of the human hands. In practical applications, hand gesture plays an important role in human communication with the real world, and expresses the primary intention of human beings. In this work, we are the first to formulate a whole-body human pose forecasting task, which jointly predicts the future body and hand activities. Correspondingly, we propose a novel Encoding-Alignment-Interaction (EAI) framework that aims to predict both coarse (body joints) and fine-grained (gestures) activities collaboratively, enabling expressive and cross-facilitated forecasting of 3D whole-body human motions. Specifically, our model involves two key constituents: cross-context alignment (XCA) and cross-context interaction (XCI). Considering the heterogeneous information within the whole-body, XCA aims to align the latent features of various human components, while XCI focuses on effectively capturing the context interaction among the human components. We conduct extensive experiments on a newly-introduced large-scale benchmark and achieve state-of-the-art performance. The code is public for research purposes at https://github.com/Dingpx/EAI.

{{</citation>}}


### (62/148) Context Disentangling and Prototype Inheriting for Robust Visual Grounding (Wei Tang et al., 2023)

{{<citation>}}

Wei Tang, Liang Li, Xuejing Liu, Lu Jin, Jinhui Tang, Zechao Li. (2023)  
**Context Disentangling and Prototype Inheriting for Robust Visual Grounding**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.11967v1)  

---


**ABSTRACT**  
Visual grounding (VG) aims to locate a specific target in an image based on a given language query. The discriminative information from context is important for distinguishing the target from other objects, particularly for the targets that have the same category as others. However, most previous methods underestimate such information. Moreover, they are usually designed for the standard scene (without any novel object), which limits their generalization to the open-vocabulary scene. In this paper, we propose a novel framework with context disentangling and prototype inheriting for robust visual grounding to handle both scenes. Specifically, the context disentangling disentangles the referent and context features, which achieves better discrimination between them. The prototype inheriting inherits the prototypes discovered from the disentangled visual features by a prototype bank to fully utilize the seen data, especially for the open-vocabulary scene. The fused features, obtained by leveraging Hadamard product on disentangled linguistic and visual features of prototypes to avoid sharp adjusting the importance between the two types of features, are then attached with a special token and feed to a vision Transformer encoder for bounding box regression. Extensive experiments are conducted on both standard and open-vocabulary scenes. The performance comparisons indicate that our method outperforms the state-of-the-art methods in both scenarios. {The code is available at https://github.com/WayneTomas/TransCP.

{{</citation>}}


### (63/148) Transformer Network for Multi-Person Tracking and Re-Identification in Unconstrained Environment (Hamza Mukhtar et al., 2023)

{{<citation>}}

Hamza Mukhtar, Muhammad Usman Ghani Khan. (2023)  
**Transformer Network for Multi-Person Tracking and Re-Identification in Unconstrained Environment**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.11929v1)  

---


**ABSTRACT**  
Multi-object tracking (MOT) has profound applications in a variety of fields, including surveillance, sports analytics, self-driving, and cooperative robotics. Despite considerable advancements, existing MOT methodologies tend to falter when faced with non-uniform movements, occlusions, and appearance-reappearance scenarios of the objects. Recognizing this inadequacy, we put forward an integrated MOT method that not only marries object detection and identity linkage within a singular, end-to-end trainable framework but also equips the model with the ability to maintain object identity links over long periods of time. Our proposed model, named STMMOT, is built around four key modules: 1) candidate proposal generation, which generates object proposals via a vision-transformer encoder-decoder architecture that detects the object from each frame in the video; 2) scale variant pyramid, a progressive pyramid structure to learn the self-scale and cross-scale similarities in multi-scale feature maps; 3) spatio-temporal memory encoder, extracting the essential information from the memory associated with each object under tracking; and 4) spatio-temporal memory decoder, simultaneously resolving the tasks of object detection and identity association for MOT. Our system leverages a robust spatio-temporal memory module that retains extensive historical observations and effectively encodes them using an attention-based aggregator. The uniqueness of STMMOT lies in representing objects as dynamic query embeddings that are updated continuously, which enables the prediction of object states with attention mechanisms and eradicates the need for post-processing.

{{</citation>}}


### (64/148) MaskINT: Video Editing via Interpolative Non-autoregressive Masked Transformers (Haoyu Ma et al., 2023)

{{<citation>}}

Haoyu Ma, Shahin Mahdizadehaghdam, Bichen Wu, Zhipeng Fan, Yuchao Gu, Wenliang Zhao, Lior Shapira, Xiaohui Xie. (2023)  
**MaskINT: Video Editing via Interpolative Non-autoregressive Masked Transformers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.12468v1)  

---


**ABSTRACT**  
Recent advances in generative AI have significantly enhanced image and video editing, particularly in the context of text prompt control. State-of-the-art approaches predominantly rely on diffusion models to accomplish these tasks. However, the computational demands of diffusion-based methods are substantial, often necessitating large-scale paired datasets for training, and therefore challenging the deployment in practical applications. This study addresses this challenge by breaking down the text-based video editing process into two separate stages. In the first stage, we leverage an existing text-to-image diffusion model to simultaneously edit a few keyframes without additional fine-tuning. In the second stage, we introduce an efficient model called MaskINT, which is built on non-autoregressive masked generative transformers and specializes in frame interpolation between the keyframes, benefiting from structural guidance provided by intermediate frames. Our comprehensive set of experiments illustrates the efficacy and efficiency of MaskINT when compared to other diffusion-based methodologies. This research offers a practical solution for text-based video editing and showcases the potential of non-autoregressive masked generative transformers in this domain.

{{</citation>}}


### (65/148) Text-Conditioned Resampler For Long Form Video Understanding (Bruno Korbar et al., 2023)

{{<citation>}}

Bruno Korbar, Yongqin Xian, Alessio Tonioni, Andrew Zisserman, Federico Tombari. (2023)  
**Text-Conditioned Resampler For Long Form Video Understanding**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2312.11897v1)  

---


**ABSTRACT**  
Videos are highly redundant data source and it is often enough to identify a few key moments to solve any given task. In this paper, we present a text-conditioned video resampler (TCR) module that uses a pre-trained and frozen visual encoder and large language model (LLM) to process long video sequences for a task. TCR localises relevant visual features from the video given a text condition and provides them to a LLM to generate a text response. Due to its lightweight design and use of cross-attention, TCR can process more than 100 frames at a time allowing the model to use much longer chunks of video than earlier works. We make the following contributions: (i) we design a transformer-based sampling architecture that can process long videos conditioned on a task, together with a training method that enables it to bridge pre-trained visual and language models; (ii) we empirically validate its efficacy on a wide variety of evaluation tasks, and set a new state-of-the-art on NextQA, EgoSchema, and the EGO4D-LTA challenge; and (iii) we determine tasks which require longer video contexts and that can thus be used effectively for further evaluation of long-range video models.

{{</citation>}}


### (66/148) Beyond Prototypes: Semantic Anchor Regularization for Better Representation Learning (Yanqi Ge et al., 2023)

{{<citation>}}

Yanqi Ge, Qiang Nie, Ye Huang, Yong Liu, Chengjie Wang, Feng Zheng, Wen Li, Lixin Duan. (2023)  
**Beyond Prototypes: Semantic Anchor Regularization for Better Representation Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2312.11872v1)  

---


**ABSTRACT**  
One of the ultimate goals of representation learning is to achieve compactness within a class and well-separability between classes. Many outstanding metric-based and prototype-based methods following the Expectation-Maximization paradigm, have been proposed for this objective. However, they inevitably introduce biases into the learning process, particularly with long-tail distributed training data. In this paper, we reveal that the class prototype is not necessarily to be derived from training features and propose a novel perspective to use pre-defined class anchors serving as feature centroid to unidirectionally guide feature learning. However, the pre-defined anchors may have a large semantic distance from the pixel features, which prevents them from being directly applied. To address this issue and generate feature centroid independent from feature learning, a simple yet effective Semantic Anchor Regularization (SAR) is proposed. SAR ensures the interclass separability of semantic anchors in the semantic space by employing a classifier-aware auxiliary cross-entropy loss during training via disentanglement learning. By pulling the learned features to these semantic anchors, several advantages can be attained: 1) the intra-class compactness and naturally inter-class separability, 2) induced bias or errors from feature learning can be avoided, and 3) robustness to the long-tailed problem. The proposed SAR can be used in a plug-and-play manner in the existing models. Extensive experiments demonstrate that the SAR performs better than previous sophisticated prototype-based methods. The implementation is available at https://github.com/geyanqi/SAR.

{{</citation>}}


### (67/148) Self-supervised Learning for Enhancing Geometrical Modeling in 3D-Aware Generative Adversarial Network (Jiarong Guo et al., 2023)

{{<citation>}}

Jiarong Guo, Xiaogang Xu, Hengshuang Zhao. (2023)  
**Self-supervised Learning for Enhancing Geometrical Modeling in 3D-Aware Generative Adversarial Network**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2312.11856v1)  

---


**ABSTRACT**  
3D-aware Generative Adversarial Networks (3D-GANs) currently exhibit artifacts in their 3D geometrical modeling, such as mesh imperfections and holes. These shortcomings are primarily attributed to the limited availability of annotated 3D data, leading to a constrained "valid latent area" for satisfactory modeling. To address this, we present a Self-Supervised Learning (SSL) technique tailored as an auxiliary loss for any 3D-GAN, designed to improve its 3D geometrical modeling capabilities. Our approach pioneers an inversion technique for 3D-GANs, integrating an encoder that performs adaptive spatially-varying range operations. Utilizing this inversion, we introduce the Cyclic Generative Constraint (CGC), aiming to densify the valid latent space. The CGC operates via augmented local latent vectors that maintain the same geometric form, and it imposes constraints on the cycle path outputs, specifically the generator-encoder-generator sequence. This SSL methodology seamlessly integrates with the inherent GAN loss, ensuring the integrity of pre-existing 3D-GAN architectures without necessitating alterations. We validate our approach with comprehensive experiments across various datasets and architectures, underscoring its efficacy. Our project website: https://3dgan-ssl.github.io

{{</citation>}}


### (68/148) GCNext: Towards the Unity of Graph Convolutions for Human Motion Prediction (Xinshun Wang et al., 2023)

{{<citation>}}

Xinshun Wang, Qiongjie Cui, Chen Chen, Mengyuan Liu. (2023)  
**GCNext: Towards the Unity of Graph Convolutions for Human Motion Prediction**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Graph Convolutional Network  
[Paper Link](http://arxiv.org/abs/2312.11850v1)  

---


**ABSTRACT**  
The past few years has witnessed the dominance of Graph Convolutional Networks (GCNs) over human motion prediction.Various styles of graph convolutions have been proposed, with each one meticulously designed and incorporated into a carefully-crafted network architecture. This paper breaks the limits of existing knowledge by proposing Universal Graph Convolution (UniGC), a novel graph convolution concept that re-conceptualizes different graph convolutions as its special cases. Leveraging UniGC on network-level, we propose GCNext, a novel GCN-building paradigm that dynamically determines the best-fitting graph convolutions both sample-wise and layer-wise. GCNext offers multiple use cases, including training a new GCN from scratch or refining a preexisting GCN. Experiments on Human3.6M, AMASS, and 3DPW datasets show that, by incorporating unique module-to-network designs, GCNext yields up to 9x lower computational cost than existing GCN methods, on top of achieving state-of-the-art performance.

{{</citation>}}


### (69/148) Decoupled Textual Embeddings for Customized Image Generation (Yufei Cai et al., 2023)

{{<citation>}}

Yufei Cai, Yuxiang Wei, Zhilong Ji, Jinfeng Bai, Hu Han, Wangmeng Zuo. (2023)  
**Decoupled Textual Embeddings for Customized Image Generation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2312.11826v1)  

---


**ABSTRACT**  
Customized text-to-image generation, which aims to learn user-specified concepts with a few images, has drawn significant attention recently. However, existing methods usually suffer from overfitting issues and entangle the subject-unrelated information (e.g., background and pose) with the learned concept, limiting the potential to compose concept into new scenes. To address these issues, we propose the DETEX, a novel approach that learns the disentangled concept embedding for flexible customized text-to-image generation. Unlike conventional methods that learn a single concept embedding from the given images, our DETEX represents each image using multiple word embeddings during training, i.e., a learnable image-shared subject embedding and several image-specific subject-unrelated embeddings. To decouple irrelevant attributes (i.e., background and pose) from the subject embedding, we further present several attribute mappers that encode each image as several image-specific subject-unrelated embeddings. To encourage these unrelated embeddings to capture the irrelevant information, we incorporate them with corresponding attribute words and propose a joint training strategy to facilitate the disentanglement. During inference, we only use the subject embedding for image generation, while selectively using image-specific embeddings to retain image-specified attributes. Extensive experiments demonstrate that the subject embedding obtained by our method can faithfully represent the target concept, while showing superior editability compared to the state-of-the-art methods. Our code will be made published available.

{{</citation>}}


### (70/148) Advancements and Challenges in Arabic Optical Character Recognition: A Comprehensive Survey (Mahmoud SalahEldin Kasem et al., 2023)

{{<citation>}}

Mahmoud SalahEldin Kasem, Mohamed Mahmoud, Hyun-Soo Kang. (2023)  
**Advancements and Challenges in Arabic Optical Character Recognition: A Comprehensive Survey**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: OCR  
[Paper Link](http://arxiv.org/abs/2312.11812v1)  

---


**ABSTRACT**  
Optical character recognition (OCR) is a vital process that involves the extraction of handwritten or printed text from scanned or printed images, converting it into a format that can be understood and processed by machines. This enables further data processing activities such as searching and editing. The automatic extraction of text through OCR plays a crucial role in digitizing documents, enhancing productivity, improving accessibility, and preserving historical records. This paper seeks to offer an exhaustive review of contemporary applications, methodologies, and challenges associated with Arabic Optical Character Recognition (OCR). A thorough analysis is conducted on prevailing techniques utilized throughout the OCR process, with a dedicated effort to discern the most efficacious approaches that demonstrate enhanced outcomes. To ensure a thorough evaluation, a meticulous keyword-search methodology is adopted, encompassing a comprehensive analysis of articles relevant to Arabic OCR, including both backward and forward citation reviews. In addition to presenting cutting-edge techniques and methods, this paper critically identifies research gaps within the realm of Arabic OCR. By highlighting these gaps, we shed light on potential areas for future exploration and development, thereby guiding researchers toward promising avenues in the field of Arabic OCR. The outcomes of this study provide valuable insights for researchers, practitioners, and stakeholders involved in Arabic OCR, ultimately fostering advancements in the field and facilitating the creation of more accurate and efficient OCR systems for the Arabic language.

{{</citation>}}


### (71/148) CAManim: Animating end-to-end network activation maps (Emily Kaczmarek et al., 2023)

{{<citation>}}

Emily Kaczmarek, Olivier X. Miguel, Alexa C. Bowie, Robin Ducharme, Alysha L. J. Dingwall-Harvey, Steven Hawken, Christine M. Armour, Mark C. Walker, Kevin Dick. (2023)  
**CAManim: Animating end-to-end network activation maps**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.11772v1)  

---


**ABSTRACT**  
Deep neural networks have been widely adopted in numerous domains due to their high performance and accessibility to developers and application-specific end-users. Fundamental to image-based applications is the development of Convolutional Neural Networks (CNNs), which possess the ability to automatically extract features from data. However, comprehending these complex models and their learned representations, which typically comprise millions of parameters and numerous layers, remains a challenge for both developers and end-users. This challenge arises due to the absence of interpretable and transparent tools to make sense of black-box models. There exists a growing body of Explainable Artificial Intelligence (XAI) literature, including a collection of methods denoted Class Activation Maps (CAMs), that seek to demystify what representations the model learns from the data, how it informs a given prediction, and why it, at times, performs poorly in certain tasks. We propose a novel XAI visualization method denoted CAManim that seeks to simultaneously broaden and focus end-user understanding of CNN predictions by animating the CAM-based network activation maps through all layers, effectively depicting from end-to-end how a model progressively arrives at the final layer activation. Herein, we demonstrate that CAManim works with any CAM-based method and various CNN architectures. Beyond qualitative model assessments, we additionally propose a novel quantitative assessment that expands upon the Remove and Debias (ROAD) metric, pairing the qualitative end-to-end network visual explanations assessment with our novel quantitative "yellow brick ROAD" assessment (ybROAD). This builds upon prior research to address the increasing demand for interpretable, robust, and transparent model assessment methodology, ultimately improving an end-user's trust in a given model's predictions.

{{</citation>}}


### (72/148) Bridging the Gap: Generalising State-of-the-Art U-Net Models to Sub-Saharan African Populations (Alyssa R. Amod et al., 2023)

{{<citation>}}

Alyssa R. Amod, Alexandra Smith, Pearly Joubert, Confidence Raymond, Dong Zhang, Udunna C. Anazodo, Dodzi Motchon, Tinashe E. M. Mutsvangwa, Sébastien Quetin. (2023)  
**Bridging the Gap: Generalising State-of-the-Art U-Net Models to Sub-Saharan African Populations**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.11770v1)  

---


**ABSTRACT**  
A critical challenge for tumour segmentation models is the ability to adapt to diverse clinical settings, particularly when applied to poor-quality neuroimaging data. The uncertainty surrounding this adaptation stems from the lack of representative datasets, leaving top-performing models without exposure to common artifacts found in MRI data throughout Sub-Saharan Africa (SSA). We replicated a framework that secured the 2nd position in the 2022 BraTS competition to investigate the impact of dataset composition on model performance and pursued four distinct approaches through training a model with: 1) BraTS-Africa data only (train_SSA, N=60), 2) BraTS-Adult Glioma data only (train_GLI, N=1251), 3) both datasets together (train_ALL, N=1311), and 4) through further training the train_GLI model with BraTS-Africa data (train_ftSSA). Notably, training on a smaller low-quality dataset alone (train_SSA) yielded subpar results, and training on a larger high-quality dataset alone (train_GLI) struggled to delineate oedematous tissue in the low-quality validation set. The most promising approach (train_ftSSA) involved pre-training a model on high-quality neuroimages and then fine-tuning it on the smaller, low-quality dataset. This approach outperformed the others, ranking second in the MICCAI BraTS Africa global challenge external testing phase. These findings underscore the significance of larger sample sizes and broad exposure to data in improving segmentation performance. Furthermore, we demonstrated that there is potential for improving such models by fine-tuning them with a wider range of data locally.

{{</citation>}}


## cs.CL (27)



### (73/148) Building a Llama2-finetuned LLM for Odia Language Utilizing Domain Knowledge Instruction Set (Guneet Singh Kohli et al., 2023)

{{<citation>}}

Guneet Singh Kohli, Shantipriya Parida, Sambit Sekhar, Samirit Saha, Nipun B Nair, Parul Agarwal, Sonal Khosla, Kusumlata Patiyal, Debasish Dhal. (2023)  
**Building a Llama2-finetuned LLM for Odia Language Utilizing Domain Knowledge Instruction Set**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.12624v1)  

---


**ABSTRACT**  
Building LLMs for languages other than English is in great demand due to the unavailability and performance of multilingual LLMs, such as understanding the local context. The problem is critical for low-resource languages due to the need for instruction sets. In a multilingual country like India, there is a need for LLMs supporting Indic languages to provide generative AI and LLM-based technologies and services to its citizens.   This paper presents our approach of i) generating a large Odia instruction set, including domain knowledge data suitable for LLM fine-tuning, and ii) building a Llama2-finetuned model tailored for enhanced performance in the Odia domain. The proposed work will help researchers build an instruction set and LLM, particularly for Indic languages. We will release the model and instruction set for the public for research and noncommercial purposes.

{{</citation>}}


### (74/148) An Empirical study of Unsupervised Neural Machine Translation: analyzing NMT output, model's behavior and sentences' contribution (Isidora Chara Tourni et al., 2023)

{{<citation>}}

Isidora Chara Tourni, Derry Wijaya. (2023)  
**An Empirical study of Unsupervised Neural Machine Translation: analyzing NMT output, model's behavior and sentences' contribution**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Machine Translation  
[Paper Link](http://arxiv.org/abs/2312.12588v1)  

---


**ABSTRACT**  
Unsupervised Neural Machine Translation (UNMT) focuses on improving NMT results under the assumption there is no human translated parallel data, yet little work has been done so far in highlighting its advantages compared to supervised methods and analyzing its output in aspects other than translation accuracy. We focus on three very diverse languages, French, Gujarati, and Kazakh, and train bilingual NMT models, to and from English, with various levels of supervision, in high- and low- resource setups, measure quality of the NMT output and compare the generated sequences' word order and semantic similarity to source and reference sentences. We also use Layer-wise Relevance Propagation to evaluate the source and target sentences' contribution to the result, expanding the findings of previous works to the UNMT paradigm.

{{</citation>}}


### (75/148) Avoiding Data Contamination in Language Model Evaluation: Dynamic Test Construction with Latest Materials (Yucheng Li et al., 2023)

{{<citation>}}

Yucheng Li, Frank Geurin, Chenghua Lin. (2023)  
**Avoiding Data Contamination in Language Model Evaluation: Dynamic Test Construction with Latest Materials**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.12343v1)  

---


**ABSTRACT**  
Data contamination in evaluation is getting increasingly prevalent with the emerge of language models pre-trained on super large, automatically-crawled corpora. This problem leads to significant challenges in accurate assessment of model capabilities and generalisations. In this paper, we propose LatestEval, an automatic method leverages the most recent texts to create uncontaminated reading comprehension evaluations. LatestEval avoids data contamination by only using texts published within a recent time window, ensuring no overlap with the training corpora of pre-trained language models. We develop LatestEval automated pipeline to 1) gather latest texts; 2) identify key information, and 3) construct questions targeting the information while removing the existing answers from the context. This encourages models to infer the answers themselves based on the remaining context, rather than just copy-paste. Our experiments demonstrate that language models exhibit negligible memorisation behaviours on LatestEval as opposed to previous benchmarks, suggesting a significantly reduced risk of data contamination and leading to a more robust evaluation. Data and code are publicly available at: https://github.com/liyucheng09/LatestEval.

{{</citation>}}


### (76/148) PowMix: A Versatile Regularizer for Multimodal Sentiment Analysis (Efthymios Georgiou et al., 2023)

{{<citation>}}

Efthymios Georgiou, Yannis Avrithis, Alexandros Potamianos. (2023)  
**PowMix: A Versatile Regularizer for Multimodal Sentiment Analysis**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Sentiment Analysis  
[Paper Link](http://arxiv.org/abs/2312.12334v1)  

---


**ABSTRACT**  
Multimodal sentiment analysis (MSA) leverages heterogeneous data sources to interpret the complex nature of human sentiments. Despite significant progress in multimodal architecture design, the field lacks comprehensive regularization methods. This paper introduces PowMix, a versatile embedding space regularizer that builds upon the strengths of unimodal mixing-based regularization approaches and introduces novel algorithmic components that are specifically tailored to multimodal tasks. PowMix is integrated before the fusion stage of multimodal architectures and facilitates intra-modal mixing, such as mixing text with text, to act as a regularizer. PowMix consists of five components: 1) a varying number of generated mixed examples, 2) mixing factor reweighting, 3) anisotropic mixing, 4) dynamic mixing, and 5) cross-modal label mixing. Extensive experimentation across benchmark MSA datasets and a broad spectrum of diverse architectural designs demonstrate the efficacy of PowMix, as evidenced by consistent performance improvements over baselines and existing mixing methods. An in-depth ablation study highlights the critical contribution of each PowMix component and how they synergistically enhance performance. Furthermore, algorithmic analysis demonstrates how PowMix behaves in different scenarios, particularly comparing early versus late fusion architectures. Notably, PowMix enhances overall performance without sacrificing model robustness or magnifying text dominance. It also retains its strong performance in situations of limited data. Our findings position PowMix as a promising versatile regularization strategy for MSA. Code will be made available.

{{</citation>}}


### (77/148) Instruct-SCTG: Guiding Sequential Controlled Text Generation through Instructions (Yinhong Liu et al., 2023)

{{<citation>}}

Yinhong Liu, Yixuan Su, Ehsan Shareghi, Nigel Collier. (2023)  
**Instruct-SCTG: Guiding Sequential Controlled Text Generation through Instructions**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Text Generation  
[Paper Link](http://arxiv.org/abs/2312.12299v1)  

---


**ABSTRACT**  
Instruction-tuned large language models have shown remarkable performance in aligning generated text with user intentions across various tasks. However, maintaining human-like discourse structure in the generated text remains a challenging research question. In this paper, we propose Instruct-SCTG, a flexible and effective sequential framework that harnesses instruction-tuned language models to generate structurally coherent text in both fine-tuned and zero-shot setups. Our framework generates articles in a section-by-section manner, aligned with the desired human structure using natural language instructions. Furthermore, we introduce a new automatic metric that measures discourse divergence in a fuzzy manner. Extensive experiments on three datasets from representative domains of news and recipes demonstrate the state-of-the-art performance of our framework in imposing discourse structure during text generation, as verified by both automatic and human evaluation. Our code will be available on Github.

{{</citation>}}


### (78/148) Geo-located Aspect Based Sentiment Analysis (ABSA) for Crowdsourced Evaluation of Urban Environments (Demircan Tas et al., 2023)

{{<citation>}}

Demircan Tas, Rohit Priyadarshi Sanatani. (2023)  
**Geo-located Aspect Based Sentiment Analysis (ABSA) for Crowdsourced Evaluation of Urban Environments**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, Sentiment Analysis, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.12253v1)  

---


**ABSTRACT**  
Sentiment analysis methods are rapidly being adopted by the field of Urban Design and Planning, for the crowdsourced evaluation of urban environments. However, most models used within this domain are able to identify positive or negative sentiment associated with a textual appraisal as a whole, without inferring information about specific urban aspects contained within it, or the sentiment associated with them. While Aspect Based Sentiment Analysis (ABSA) is becoming increasingly popular, most existing ABSA models are trained on non-urban themes such as restaurants, electronics, consumer goods and the like. This body of research develops an ABSA model capable of extracting urban aspects contained within geo-located textual urban appraisals, along with corresponding aspect sentiment classification. We annotate a dataset of 2500 crowdsourced reviews of public parks, and train a Bidirectional Encoder Representations from Transformers (BERT) model with Local Context Focus (LCF) on this data. Our model achieves significant improvement in prediction accuracy on urban reviews, for both Aspect Term Extraction (ATE) and Aspect Sentiment Classification (ASC) tasks. For demonstrative analysis, positive and negative urban aspects across Boston are spatially visualized. We hope that this model is useful for designers and planners for fine-grained urban sentiment evaluation.

{{</citation>}}


### (79/148) Parameter-Efficient Fine-Tuning Methods for Pretrained Language Models: A Critical Review and Assessment (Lingling Xu et al., 2023)

{{<citation>}}

Lingling Xu, Haoran Xie, Si-Zhao Joe Qin, Xiaohui Tao, Fu Lee Wang. (2023)  
**Parameter-Efficient Fine-Tuning Methods for Pretrained Language Models: A Critical Review and Assessment**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, NLP, Pretrained Language Models  
[Paper Link](http://arxiv.org/abs/2312.12148v1)  

---


**ABSTRACT**  
With the continuous growth in the number of parameters of transformer-based pretrained language models (PLMs), particularly the emergence of large language models (LLMs) with billions of parameters, many natural language processing (NLP) tasks have demonstrated remarkable success. However, the enormous size and computational demands of these models pose significant challenges for adapting them to specific downstream tasks, especially in environments with limited computational resources. Parameter Efficient Fine-Tuning (PEFT) offers an effective solution by reducing the number of fine-tuning parameters and memory usage while achieving comparable performance to full fine-tuning. The demands for fine-tuning PLMs, especially LLMs, have led to a surge in the development of PEFT methods, as depicted in Fig. 1. In this paper, we present a comprehensive and systematic review of PEFT methods for PLMs. We summarize these PEFT methods, discuss their applications, and outline future directions. Furthermore, we conduct experiments using several representative PEFT methods to better understand their effectiveness in parameter efficiency and memory efficiency. By offering insights into the latest advancements and practical applications, this survey serves as an invaluable resource for researchers and practitioners seeking to navigate the challenges and opportunities presented by PEFT in the context of PLMs.

{{</citation>}}


### (80/148) Exploring the Residual Stream of Transformers (Zeping Yu et al., 2023)

{{<citation>}}

Zeping Yu, Kailai Yang, Zhiwei Liu, Sophia Ananiadou. (2023)  
**Exploring the Residual Stream of Transformers**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.12141v1)  

---


**ABSTRACT**  
Transformer-based models have achieved great breakthroughs in recent years. However, there are many significant questions that have not been answered in the field of explaining the reason why the models have powerful outputs. We do not know how to locate the models' important parameters storing the knowledge for predicting the next word, and whether these parameters are stored on the same layer/module or different ones. Moreover, we do not understand the mechanism to merge the knowledge into the final embedding for next word prediction. In this paper, we explore the residual stream of transformers to increase the interpretability. We find the mechanism behind residual connection is a direct addition function on before-softmax values, so the probabilities of tokens with larger before-softmax values will increase. Moreover, we prove that using log probability increase as contribution scores is reasonable, and based on this we can locate important parameters. Besides, we propose a method to analyze how previous layers affect upper layers by comparing the inner products. The experimental results and case study show that our research can increase the interpretability of transformer-based models. We will release our code on https://github.com/zepingyu0512/residualstream.

{{</citation>}}


### (81/148) Knowledge Graph Error Detection with Contrastive Confidence Adaption (Xiangyu Liu et al., 2023)

{{<citation>}}

Xiangyu Liu, Yang Liu, Wei Hu. (2023)  
**Knowledge Graph Error Detection with Contrastive Confidence Adaption**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2312.12108v1)  

---


**ABSTRACT**  
Knowledge graphs (KGs) often contain various errors. Previous works on detecting errors in KGs mainly rely on triplet embedding from graph structure. We conduct an empirical study and find that these works struggle to discriminate noise from semantically-similar correct triplets. In this paper, we propose a KG error detection model CCA to integrate both textual and graph structural information from triplet reconstruction for better distinguishing semantics. We design interactive contrastive learning to capture the differences between textual and structural patterns. Furthermore, we construct realistic datasets with semantically-similar noise and adversarial noise. Experimental results demonstrate that CCA outperforms state-of-the-art baselines, especially in detecting semantically-similar noise and adversarial noise.

{{</citation>}}


### (82/148) Founder-GPT: Self-play to evaluate the Founder-Idea fit (Sichao Xiong et al., 2023)

{{<citation>}}

Sichao Xiong, Yigit Ihlamur. (2023)  
**Founder-GPT: Self-play to evaluate the Founder-Idea fit**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CE, cs-CL, cs.CL  
Keywords: Embedding, GPT  
[Paper Link](http://arxiv.org/abs/2312.12037v2)  

---


**ABSTRACT**  
This research introduces an innovative evaluation method for the "founder-idea" fit in early-stage startups, utilizing advanced large language model techniques to assess founders' profiles against their startup ideas to enhance decision-making. Embeddings, self-play, tree-of-thought, and critique-based refinement techniques show early promising results that each idea's success patterns are unique and they should be evaluated based on the context of the founder's background.

{{</citation>}}


### (83/148) Synergistic Anchored Contrastive Pre-training for Few-Shot Relation Extraction (DaLuo et al., 2023)

{{<citation>}}

DaLuo, Yanglei Gan, Rui Hou, Run Lin, Qiao Liu, Yuxiang Cai, Wannian Gao. (2023)  
**Synergistic Anchored Contrastive Pre-training for Few-Shot Relation Extraction**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Few-Shot, Language Model, NLP, Relation Extraction  
[Paper Link](http://arxiv.org/abs/2312.12021v1)  

---


**ABSTRACT**  
Few-shot Relation Extraction (FSRE) aims to extract relational facts from a sparse set of labeled corpora. Recent studies have shown promising results in FSRE by employing Pre-trained Language Models (PLMs) within the framework of supervised contrastive learning, which considers both instances and label facts. However, how to effectively harness massive instance-label pairs to encompass the learned representation with semantic richness in this learning paradigm is not fully explored. To address this gap, we introduce a novel synergistic anchored contrastive pre-training framework. This framework is motivated by the insight that the diverse viewpoints conveyed through instance-label pairs capture incomplete yet complementary intrinsic textual semantics. Specifically, our framework involves a symmetrical contrastive objective that encompasses both sentence-anchored and label-anchored contrastive losses. By combining these two losses, the model establishes a robust and uniform representation space. This space effectively captures the reciprocal alignment of feature distributions among instances and relational facts, simultaneously enhancing the maximization of mutual information across diverse perspectives within the same relation. Experimental results demonstrate that our framework achieves significant performance enhancements compared to baseline models in downstream FSRE tasks. Furthermore, our approach exhibits superior adaptability to handle the challenges of domain shift and zero-shot relation extraction. Our code is available online at https://github.com/AONE-NLP/FSRE-SaCon.

{{</citation>}}


### (84/148) Active Preference Inference using Language Models and Probabilistic Reasoning (Top Piriyakulkij et al., 2023)

{{<citation>}}

Top Piriyakulkij, Volodymyr Kuleshov, Kevin Ellis. (2023)  
**Active Preference Inference using Language Models and Probabilistic Reasoning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2312.12009v1)  

---


**ABSTRACT**  
Actively inferring user preferences, for example by asking good questions, is important for any human-facing decision-making system. Active inference allows such systems to adapt and personalize themselves to nuanced individual preferences. To enable this ability for instruction-tuned large language models (LLMs), one may prompt them to ask users questions to infer their preferences, transforming the language models into more robust, interactive systems. However, out of the box, these models are not efficient at extracting preferences: the questions they generate are not informative, requiring a high number of user interactions and impeding the usability of the downstream system. In this work, we introduce an inference-time algorithm that helps LLMs quickly infer preferences by using more informative questions. Our algorithm uses a probabilistic model whose conditional distributions are defined by prompting an LLM, and returns questions that optimize expected entropy and expected model change. Results in a simplified interactive web shopping setting with real product items show that an LLM equipped with our entropy reduction algorithm outperforms baselines with the same underlying LLM on task performance while using fewer user interactions.

{{</citation>}}


### (85/148) Can ChatGPT be Your Personal Medical Assistant? (Md. Rafiul Biswas et al., 2023)

{{<citation>}}

Md. Rafiul Biswas, Ashhadul Islam, Zubair Shah, Wajdi Zaghouani, Samir Brahim Belhaouari. (2023)  
**Can ChatGPT be Your Personal Medical Assistant?**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SI, cs.CL  
Keywords: ChatGPT, GPT, GPT-3.5  
[Paper Link](http://arxiv.org/abs/2312.12006v1)  

---


**ABSTRACT**  
The advanced large language model (LLM) ChatGPT has shown its potential in different domains and remains unbeaten due to its characteristics compared to other LLMs. This study aims to evaluate the potential of using a fine-tuned ChatGPT model as a personal medical assistant in the Arabic language. To do so, this study uses publicly available online questions and answering datasets in Arabic language. There are almost 430K questions and answers for 20 disease-specific categories. GPT-3.5-turbo model was fine-tuned with a portion of this dataset. The performance of this fine-tuned model was evaluated through automated and human evaluation. The automated evaluations include perplexity, coherence, similarity, and token count. Native Arabic speakers with medical knowledge evaluated the generated text by calculating relevance, accuracy, precision, logic, and originality. The overall result shows that ChatGPT has a bright future in medical assistance.

{{</citation>}}


### (86/148) Climate Change from Large Language Models (Hongyin Zhu et al., 2023)

{{<citation>}}

Hongyin Zhu, Prayag Tiwari. (2023)  
**Climate Change from Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.11985v2)  

---


**ABSTRACT**  
Climate change presents significant challenges to the global community, and it is imperative to raise widespread awareness of the climate crisis and educate users about low-carbon living. Artificial intelligence, particularly large language models (LLMs), have emerged as powerful tools in mitigating the climate crisis, leveraging their extensive knowledge, broad user base, and natural language interaction capabilities. However, despite the growing body of research on climate change, there is a lack of comprehensive assessments of climate crisis knowledge within LLMs. This paper aims to resolve this gap by proposing an automatic evaluation framework. We employ a hybrid approach to data acquisition that combines data synthesis and manual collection to compile a diverse set of questions related to the climate crisis. These questions cover various aspects of climate change, including its causes, impacts, mitigation strategies, and adaptation measures. We then evaluate the model knowledge through prompt engineering based on the collected questions and generated answers. We propose a set of comprehensive metrics to evaluate the climate crisis knowledge, incorporating indicators from 10 different perspectives. Experimental results show that our method is effective in evaluating the knowledge of LLMs regarding the climate crisis. We evaluate several state-of-the-art LLMs and find that their knowledge falls short in terms of timeliness.

{{</citation>}}


### (87/148) Fluctuation-based Adaptive Structured Pruning for Large Language Models (Yongqi An et al., 2023)

{{<citation>}}

Yongqi An, Xu Zhao, Tao Yu, Ming Tang, Jinqiao Wang. (2023)  
**Fluctuation-based Adaptive Structured Pruning for Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, Pruning  
[Paper Link](http://arxiv.org/abs/2312.11983v1)  

---


**ABSTRACT**  
Network Pruning is a promising way to address the huge computing resource demands of the deployment and inference of Large Language Models (LLMs). Retraining-free is important for LLMs' pruning methods. However, almost all of the existing retraining-free pruning approaches for LLMs focus on unstructured pruning, which requires specific hardware support for acceleration. In this paper, we propose a novel retraining-free structured pruning framework for LLMs, named FLAP (FLuctuation-based Adaptive Structured Pruning). It is hardware-friendly by effectively reducing storage and enhancing inference speed. For effective structured pruning of LLMs, we highlight three critical elements that demand the utmost attention: formulating structured importance metrics, adaptively searching the global compressed model, and implementing compensation mechanisms to mitigate performance loss. First, FLAP determines whether the output feature map is easily recoverable when a column of weight is removed, based on the fluctuation pruning metric. Then it standardizes the importance scores to adaptively determine the global compressed model structure. At last, FLAP adds additional bias terms to recover the output feature maps using the baseline values. We thoroughly evaluate our approach on a variety of language benchmarks. Without any retraining, our method significantly outperforms the state-of-the-art methods, including LLM-Pruner and the extension of Wanda in structured pruning. The code is released at https://github.com/CASIA-IVA-Lab/FLAP.

{{</citation>}}


### (88/148) Relation-Aware Question Answering for Heterogeneous Knowledge Graphs (Haowei Du et al., 2023)

{{<citation>}}

Haowei Du, Quzhe Huang, Chen Li, Chen Zhang, Yang Li, Dongyan Zhao. (2023)  
**Relation-Aware Question Answering for Heterogeneous Knowledge Graphs**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Knowledge Graph, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2312.11922v1)  

---


**ABSTRACT**  
Multi-hop Knowledge Base Question Answering(KBQA) aims to find the answer entity in a knowledge graph (KG), which requires multiple steps of reasoning. Existing retrieval-based approaches solve this task by concentrating on the specific relation at different hops and predicting the intermediate entity within the reasoning path. During the reasoning process of these methods, the representation of relations are fixed but the initial relation representation may not be optimal. We claim they fail to utilize information from head-tail entities and the semantic connection between relations to enhance the current relation representation, which undermines the ability to capture information of relations in KGs. To address this issue, we construct a \textbf{dual relation graph} where each node denotes a relation in the original KG (\textbf{primal entity graph}) and edges are constructed between relations sharing same head or tail entities. Then we iteratively do primal entity graph reasoning, dual relation graph information propagation, and interaction between these two graphs. In this way, the interaction between entity and relation is enhanced, and we derive better entity and relation representations. Experiments on two public datasets, WebQSP and CWQ, show that our approach achieves a significant performance gain over the prior state-of-the-art. Our code is available on \url{https://github.com/yanmenxue/RAH-KBQA}.

{{</citation>}}


### (89/148) External Knowledge Augmented Polyphone Disambiguation Using Large Language Model (Chen Li, 2023)

{{<citation>}}

Chen Li. (2023)  
**External Knowledge Augmented Polyphone Disambiguation Using Large Language Model**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2312.11920v1)  

---


**ABSTRACT**  
One of the key issues in Mandarin Chinese text-to-speech (TTS) systems is polyphone disambiguation when doing grapheme-to-phoneme (G2P) conversion. In this paper, we introduce a novel method to solve the problem as a generation task. Following the trending research of large language models (LLM) and prompt learning, the proposed method consists of three modules. Retrieval module incorporates external knowledge which is a multi-level semantic dictionary of Chinese polyphonic characters to format the sentence into a prompt. Generation module adopts the decoder-only Transformer architecture to induce the target text. Postprocess module corrects the generated text into a valid result if needed. Experimental results show that our method outperforms the existing methods on a public dataset called CPP. We also empirically study the impacts of different templates of the prompt, different sizes of training data, and whether to incorporate external knowledge.

{{</citation>}}


### (90/148) Difficulty-Focused Contrastive Learning for Knowledge Tracing with a Large Language Model-Based Difficulty Prediction (Unggi Lee et al., 2023)

{{<citation>}}

Unggi Lee, Sungjun Yoon, Joon Seo Yun, Kyoungsoo Park, YoungHoon Jung, Damji Stratton, Hyeoncheol Kim. (2023)  
**Difficulty-Focused Contrastive Learning for Knowledge Tracing with a Large Language Model-Based Difficulty Prediction**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SI, cs.CL  
Keywords: Contrastive Learning, Language Model  
[Paper Link](http://arxiv.org/abs/2312.11890v1)  

---


**ABSTRACT**  
This paper presents novel techniques for enhancing the performance of knowledge tracing (KT) models by focusing on the crucial factor of question and concept difficulty level. Despite the acknowledged significance of difficulty, previous KT research has yet to exploit its potential for model optimization and has struggled to predict difficulty from unseen data. To address these problems, we propose a difficulty-centered contrastive learning method for KT models and a Large Language Model (LLM)-based framework for difficulty prediction. These innovative methods seek to improve the performance of KT models and provide accurate difficulty estimates for unseen data. Our ablation study demonstrates the efficacy of these techniques by demonstrating enhanced KT model performance. Nonetheless, the complex relationship between language and difficulty merits further investigation.

{{</citation>}}


### (91/148) ConsistentEE: A Consistent and Hardness-Guided Early Exiting Method for Accelerating Language Models Inference (Ziqian Zeng et al., 2023)

{{<citation>}}

Ziqian Zeng, Yihuai Hong, Hongliang Dai, Huiping Zhuang, Cen Chen. (2023)  
**ConsistentEE: A Consistent and Hardness-Guided Early Exiting Method for Accelerating Language Models Inference**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.11882v1)  

---


**ABSTRACT**  
Early Exiting is one of the most popular methods to achieve efficient inference. Current early exiting methods adopt the (weighted) sum of the cross entropy loss of all internal classifiers during training, imposing all these classifiers to predict all instances correctly. However, during inference, as long as one internal classifier predicts an instance correctly, it can accelerate without losing accuracy. Thus, there is a notable gap between training and inference. We propose ConsistentEE, an early exiting method that is consistent in training and inference. ConsistentEE formulates the early exiting process as a reinforcement learning problem. A policy network is added to decide whether an instance should exit or continue. The training objective of ConsistentEE only require each instance to be predicted correctly by one internal classifier. Additionally, we introduce the concept Memorize Layer to measure the hardness of an instance. We incorporate memorized layer into reward function design, which allows ``easy'' instances to focus more on acceleration while ``hard'' instances to focus more on accuracy. Experimental results show that our method outperforms other baselines on various natural language understanding and generation tasks.

{{</citation>}}


### (92/148) A Revisit of Fake News Dataset with Augmented Fact-checking by ChatGPT (Zizhong Li et al., 2023)

{{<citation>}}

Zizhong Li, Haopeng Zhang, Jiawei Zhang. (2023)  
**A Revisit of Fake News Dataset with Augmented Fact-checking by ChatGPT**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, Fake News, GPT  
[Paper Link](http://arxiv.org/abs/2312.11870v1)  

---


**ABSTRACT**  
The proliferation of fake news has emerged as a critical issue in recent years, requiring significant efforts to detect it. However, the existing fake news detection datasets are sourced from human journalists, which are likely to have inherent bias limitations due to the highly subjective nature of this task. In this paper, we revisit the existing fake news dataset verified by human journalists with augmented fact-checking by large language models (ChatGPT), and we name the augmented fake news dataset ChatGPT-FC. We quantitatively analyze the distinctions and resemblances between human journalists and LLM in assessing news subject credibility, news creator credibility, time-sensitive, and political framing. Our findings highlight LLM's potential to serve as a preliminary screening method, offering a promising avenue to mitigate the inherent biases of human journalists and enhance fake news detection.

{{</citation>}}


### (93/148) Predicting Human Translation Difficulty with Neural Machine Translation (Zheng Wei Lim et al., 2023)

{{<citation>}}

Zheng Wei Lim, Ekaterina Vylomova, Charles Kemp, Trevor Cohn. (2023)  
**Predicting Human Translation Difficulty with Neural Machine Translation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Machine Translation  
[Paper Link](http://arxiv.org/abs/2312.11852v1)  

---


**ABSTRACT**  
Human translators linger on some words and phrases more than others, and predicting this variation is a step towards explaining the underlying cognitive processes. Using data from the CRITT Translation Process Research Database, we evaluate the extent to which surprisal and attentional features derived from a Neural Machine Translation (NMT) model account for reading and production times of human translators. We find that surprisal and attention are complementary predictors of translation difficulty, and that surprisal derived from a NMT model is the single most successful predictor of production duration. Our analyses draw on data from hundreds of translators operating across 13 language pairs, and represent the most comprehensive investigation of human translation difficulty to date.

{{</citation>}}


### (94/148) TESS: A Multi-intent Parser for Conversational Multi-Agent Systems with Decentralized Natural Language Understanding Models (Burak Aksar et al., 2023)

{{<citation>}}

Burak Aksar, Yara Rizk, Tathagata Chakraborti. (2023)  
**TESS: A Multi-intent Parser for Conversational Multi-Agent Systems with Decentralized Natural Language Understanding Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-MA, cs.CL  
Keywords: NLU, Natural Language Understanding  
[Paper Link](http://arxiv.org/abs/2312.11828v1)  

---


**ABSTRACT**  
Chatbots have become one of the main pathways for the delivery of business automation tools. Multi-agent systems offer a framework for designing chatbots at scale, making it easier to support complex conversations that span across multiple domains as well as enabling developers to maintain and expand their capabilities incrementally over time. However, multi-agent systems complicate the natural language understanding (NLU) of user intents, especially when they rely on decentralized NLU models: some utterances (termed single intent) may invoke a single agent while others (termed multi-intent) may explicitly invoke multiple agents. Without correctly parsing multi-intent inputs, decentralized NLU approaches will not achieve high prediction accuracy. In this paper, we propose an efficient parsing and orchestration pipeline algorithm to service multi-intent utterances from the user in the context of a multi-agent system. Our proposed approach achieved comparable performance to competitive deep learning models on three different datasets while being up to 48 times faster.

{{</citation>}}


### (95/148) Designing Guiding Principles for NLP for Healthcare: A Case Study of Maternal Health (Maria Antoniak et al., 2023)

{{<citation>}}

Maria Antoniak, Aakanksha Naik, Carla S. Alvarado, Lucy Lu Wang, Irene Y. Chen. (2023)  
**Designing Guiding Principles for NLP for Healthcare: A Case Study of Maternal Health**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, NLP  
[Paper Link](http://arxiv.org/abs/2312.11803v1)  

---


**ABSTRACT**  
Objective: An ethical framework for the use of large language models (LLMs) is urgently needed to shape how natural language processing (NLP) tools are used for healthcare applications. Drawing directly from the voices of those most affected, we propose a set of guiding principles for the use of NLP in healthcare, with examples based on applications in maternal health.   Materials and Methods: We led an interactive session centered on an LLM-based chatbot demonstration during a full-day workshop with 39 participants, and additionally surveyed 30 healthcare workers and 30 birthing people about their values, needs, and perceptions of AI and LLMs. We conducted quantitative and qualitative analyses of the interactive discussions to consolidate our findings into a set of guiding principles.   Results: Using the case study of maternal health, we propose nine principles for ethical use of LLMs, grouped into three categories: (i) contextual significance, (ii) measurements, and (iii) who/what is valued. We describe rationales underlying these principles and provide practical advice.   Discussion: Healthcare faces existing challenges including the balance of power in clinician-patient relationships, systemic health disparities, historical injustices, and economic constraints. Our principles serve as a framework for surfacing key considerations when deploying LLMs in medicine, as well as providing a methodological pattern for other researchers to follow.   Conclusion: This set of principles can serve as a resource to practitioners working on maternal health and other healthcare fields to emphasize the importance of technical nuance, historical context, and inclusive design when developing LLMs for use in clinical settings.

{{</citation>}}


### (96/148) MELO: Enhancing Model Editing with Neuron-Indexed Dynamic LoRA (Lang Yu et al., 2023)

{{<citation>}}

Lang Yu, Qin Chen, Jie Zhou, Liang He. (2023)  
**MELO: Enhancing Model Editing with Neuron-Indexed Dynamic LoRA**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2312.11795v1)  

---


**ABSTRACT**  
Large language models (LLMs) have shown great success in various Natural Language Processing (NLP) tasks, whist they still need updates after deployment to fix errors or keep pace with the changing knowledge in the world. Researchers formulate such problem as Model Editing and have developed various editors focusing on different axes of editing properties. However, current editors can hardly support all properties and rely on heavy computational resources. In this paper, we propose a plug-in Model Editing method based on neuron-indexed dynamic LoRA (MELO), which alters the behavior of language models by dynamically activating certain LoRA blocks according to the index built in an inner vector database. Our method satisfies various editing properties with high efficiency and can be easily integrated into multiple LLM backbones. Experimental results show that our proposed MELO achieves state-of-the-art editing performance on three sequential editing tasks (document classification, question answering and hallucination correction), while requires the least trainable parameters and computational cost.

{{</citation>}}


### (97/148) COOPER: Coordinating Specialized Agents towards a Complex Dialogue Goal (Yi Cheng et al., 2023)

{{<citation>}}

Yi Cheng, Wenge Liu, Jian Wang, Chak Tou Leong, Yi Ouyang, Wenjie Li, Xian Wu, Yefeng Zheng. (2023)  
**COOPER: Coordinating Specialized Agents towards a Complex Dialogue Goal**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2312.11792v1)  

---


**ABSTRACT**  
In recent years, there has been a growing interest in exploring dialogues with more complex goals, such as negotiation, persuasion, and emotional support, which go beyond traditional service-focused dialogue systems. Apart from the requirement for much more sophisticated strategic reasoning and communication skills, a significant challenge of these tasks lies in the difficulty of objectively measuring the achievement of their goals in a quantifiable way, making it difficult for existing research to directly optimize the dialogue procedure towards them. In our work, we emphasize the multifaceted nature of complex dialogue goals and argue that it is more feasible to accomplish them by comprehensively considering and jointly promoting their different aspects. To this end, we propose a novel dialogue framework, Cooper, which coordinates multiple specialized agents, each dedicated to a specific dialogue goal aspect separately, to approach the complex objective. Through this divide-and-conquer manner, we make complex dialogue goals more approachable and elicit greater intelligence via the collaboration of individual agents. Experiments on persuasion and emotional support dialogues demonstrate the superiority of our method over a set of competitive baselines.

{{</citation>}}


### (98/148) Zero-Shot Fact-Checking with Semantic Triples and Knowledge Graphs (Zhangdie Yuan et al., 2023)

{{<citation>}}

Zhangdie Yuan, Andreas Vlachos. (2023)  
**Zero-Shot Fact-Checking with Semantic Triples and Knowledge Graphs**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Fact-Checking, Knowledge Graph, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2312.11785v1)  

---


**ABSTRACT**  
Despite progress in automated fact-checking, most systems require a significant amount of labeled training data, which is expensive. In this paper, we propose a novel zero-shot method, which instead of operating directly on the claim and evidence sentences, decomposes them into semantic triples augmented using external knowledge graphs, and uses large language models trained for natural language inference. This allows it to generalize to adversarial datasets and domains that supervised models require specific training data for. Our empirical results show that our approach outperforms previous zero-shot approaches on FEVER, FEVER-Symmetric, FEVER 2.0, and Climate-FEVER, while being comparable or better than supervised models on the adversarial and the out-of-domain datasets.

{{</citation>}}


### (99/148) Are you talking to ['xem'] or ['x', 'em']? On Tokenization and Addressing Misgendering in LLMs with Pronoun Tokenization Parity (Anaelia Ovalle et al., 2023)

{{<citation>}}

Anaelia Ovalle, Ninareh Mehrabi, Palash Goyal, Jwala Dhamala, Kai-Wei Chang, Richard Zemel, Aram Galstyan, Rahul Gupta. (2023)  
**Are you talking to ['xem'] or ['x', 'em']? On Tokenization and Addressing Misgendering in LLMs with Pronoun Tokenization Parity**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2312.11779v2)  

---


**ABSTRACT**  
A large body of NLP research has documented the ways gender biases manifest and amplify within large language models (LLMs), though this research has predominantly operated within a gender binary-centric context. A growing body of work has identified the harmful limitations of this gender-exclusive framing; many LLMs cannot correctly and consistently refer to persons outside the gender binary, especially if they use neopronouns. While data scarcity has been identified as a possible culprit, the precise mechanisms through which it influences LLM misgendering remain underexplored. Our work addresses this gap by studying data scarcity's role in subword tokenization and, consequently, the formation of LLM word representations. We uncover how the Byte-Pair Encoding (BPE) tokenizer, a backbone for many popular LLMs, contributes to neopronoun misgendering through out-of-vocabulary behavior. We introduce pronoun tokenization parity (PTP), a novel approach to reduce LLM neopronoun misgendering by preserving a token's functional structure. We evaluate PTP's efficacy using pronoun consistency-based metrics and a novel syntax-based metric. Through several controlled experiments, finetuning LLMs with PTP improves neopronoun consistency from 14.5% to 58.4%, highlighting the significant role tokenization plays in LLM pronoun consistency.

{{</citation>}}


## stat.ML (3)



### (100/148) Online Variational Sequential Monte Carlo (Alessandro Mastrototaro et al., 2023)

{{<citation>}}

Alessandro Mastrototaro, Jimmy Olsson. (2023)  
**Online Variational Sequential Monte Carlo**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-ML, stat.ML  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.12616v1)  

---


**ABSTRACT**  
Being the most classical generative model for serial data, state-space models (SSM) are fundamental in AI and statistical machine learning. In SSM, any form of parameter learning or latent state inference typically involves the computation of complex latent-state posteriors. In this work, we build upon the variational sequential Monte Carlo (VSMC) method, which provides computationally efficient and accurate model parameter estimation and Bayesian latent-state inference by combining particle methods and variational inference. While standard VSMC operates in the offline mode, by re-processing repeatedly a given batch of data, we distribute the approximation of the gradient of the VSMC surrogate ELBO in time using stochastic approximation, allowing for online learning in the presence of streams of data. This results in an algorithm, online VSMC, that is capable of performing efficiently, entirely on-the-fly, both parameter estimation and particle proposal adaptation. In addition, we provide rigorous theoretical results describing the algorithm's convergence properties as the number of data tends to infinity as well as numerical illustrations of its excellent convergence properties and usefulness also in batch-processing settings.

{{</citation>}}


### (101/148) LightGCNet: A Lightweight Geometric Constructive Neural Network for Data-Driven Soft sensors (Jing Nan et al., 2023)

{{<citation>}}

Jing Nan, Yan Qin, Wei Dai, Chau Yuen. (2023)  
**LightGCNet: A Lightweight Geometric Constructive Neural Network for Data-Driven Soft sensors**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-ML, stat.ML  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.12022v1)  

---


**ABSTRACT**  
Data-driven soft sensors provide a potentially cost-effective and more accurate modeling approach to measure difficult-to-measure indices in industrial processes compared to mechanistic approaches. Artificial intelligence (AI) techniques, such as deep learning, have become a popular soft sensors modeling approach in the area of machine learning and big data. However, soft sensors models based deep learning potentially lead to complex model structures and excessive training time. In addition, industrial processes often rely on distributed control systems (DCS) characterized by resource constraints. Herein, guided by spatial geometric, a lightweight geometric constructive neural network, namely LightGCNet, is proposed, which utilizes compact angle constraint to assign the hidden parameters from dynamic intervals. At the same time, a node pool strategy and spatial geometric relationships are used to visualize and optimize the process of assigning hidden parameters, enhancing interpretability. In addition, the universal approximation property of LightGCNet is proved by spatial geometric analysis. Two versions algorithmic implementations of LightGCNet are presented in this article. Simulation results concerning both benchmark datasets and the ore grinding process indicate remarkable merits of LightGCNet in terms of small network size, fast learning speed, and sound generalization.

{{</citation>}}


### (102/148) Modelling and characterization of fine Particulate Matter dynamics in Bujumbura using low cost sensors (Egide Ndamuzi et al., 2023)

{{<citation>}}

Egide Ndamuzi, Rachel Akimana, Paterne Gahungu, Elie Bimenyimana. (2023)  
**Modelling and characterization of fine Particulate Matter dynamics in Bujumbura using low cost sensors**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-ML, stat.ML  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2312.12003v1)  

---


**ABSTRACT**  
Air pollution is a result of multiple sources including both natural and anthropogenic activities. The rapid urbanization of the cities such as Bujumbura economic capital of Burundi, is one of these factors. The very first characterization of the spatio-temporal variability of PM2.5 in Bujumbura and the forecasting of PM2.5 concentration have been conducted in this paper using data collected during a year, from august 2022 to august 2023, by low cost sensors installed in Bujumbura city. For each commune, an hourly, daily and seasonal analysis were carried out and the results showed that the mass concentrations of PM2.5 in the three municipalities differ from one commune to another. The average hourly and annual PM2.5 concentrations exceed the World Health Organization standards. The range is between 28.3 and 35.0 microgram/m3 . In order to make prediction of PM2.5 concentration, an investigation of RNN with Long Short Term Memory (LSTM) has been undertaken.

{{</citation>}}


## cs.SE (5)



### (103/148) Studying the Practices of Testing Machine Learning Software in the Wild (Moses Openja et al., 2023)

{{<citation>}}

Moses Openja, Foutse Khomh, Armstrong Foundjem, Zhen Ming, Jiang, Mouna Abidi, Ahmed E. Hassan. (2023)  
**Studying the Practices of Testing Machine Learning Software in the Wild**  

---
Primary Category: cs.SE  
Categories: cs-LG, cs-SE, cs.SE  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2312.12604v1)  

---


**ABSTRACT**  
Background: We are witnessing an increasing adoption of machine learning (ML), especially deep learning (DL) algorithms in many software systems, including safety-critical systems such as health care systems or autonomous driving vehicles. Ensuring the software quality of these systems is yet an open challenge for the research community, mainly due to the inductive nature of ML software systems. Traditionally, software systems were constructed deductively, by writing down the rules that govern the behavior of the system as program code. However, for ML software, these rules are inferred from training data. Few recent research advances in the quality assurance of ML systems have adapted different concepts from traditional software testing, such as mutation testing, to help improve the reliability of ML software systems. However, it is unclear if any of these proposed testing techniques from research are adopted in practice. There is little empirical evidence about the testing strategies of ML engineers. Aims: To fill this gap, we perform the first fine-grained empirical study on ML testing practices in the wild, to identify the ML properties being tested, the followed testing strategies, and their implementation throughout the ML workflow. Method: First, we systematically summarized the different testing strategies (e.g., Oracle Approximation), the tested ML properties (e.g., Correctness, Bias, and Fairness), and the testing methods (e.g., Unit test) from the literature. Then, we conducted a study to understand the practices of testing ML software. Results: In our findings: 1) we identified four (4) major categories of testing strategy including Grey-box, White-box, Black-box, and Heuristic-based techniques that are used by the ML engineers to find software bugs. 2) We identified 16 ML properties that are tested in the ML workflow.

{{</citation>}}


### (104/148) A Case Study on Test Case Construction with Large Language Models: Unveiling Practical Insights and Challenges (Roberto Francisco de Lima Junior et al., 2023)

{{<citation>}}

Roberto Francisco de Lima Junior, Luiz Fernando Paes de Barros Presta, Lucca Santos Borborema, Vanderson Nogueira da Silva, Marcio Leal de Melo Dahia, Anderson Carlos Sousa e Santos. (2023)  
**A Case Study on Test Case Construction with Large Language Models: Unveiling Practical Insights and Challenges**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.12598v1)  

---


**ABSTRACT**  
This paper presents a detailed case study examining the application of Large Language Models (LLMs) in the construction of test cases within the context of software engineering. LLMs, characterized by their advanced natural language processing capabilities, are increasingly garnering attention as tools to automate and enhance various aspects of the software development lifecycle. Leveraging a case study methodology, we systematically explore the integration of LLMs in the test case construction process, aiming to shed light on their practical efficacy, challenges encountered, and implications for software quality assurance.   The study encompasses the selection of a representative software application, the formulation of test case construction methodologies employing LLMs, and the subsequent evaluation of outcomes. Through a blend of qualitative and quantitative analyses, we assess the impact of LLMs on test case comprehensiveness, accuracy, and efficiency. Additionally, we delve into challenges such as model interpretability, ethical considerations, and adaptation to diverse software contexts.   The findings from this case study contribute nuanced insights into the practical utility of LLMs in the domain of test case construction, elucidating their potential benefits and limitations. By addressing real-world scenarios and complexities, this research aims to inform software practitioners and researchers alike about the tangible implications of incorporating LLMs into the software testing landscape, fostering a more comprehensive understanding of their role in optimizing the software development process.

{{</citation>}}


### (105/148) Word Closure-Based Metamorphic Testing for Machine Translation (Xiaoyuan Xie et al., 2023)

{{<citation>}}

Xiaoyuan Xie, Shuo Jin, Songqiang Chen, Shing-Chi Cheung. (2023)  
**Word Closure-Based Metamorphic Testing for Machine Translation**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Machine Translation  
[Paper Link](http://arxiv.org/abs/2312.12056v1)  

---


**ABSTRACT**  
With the wide application of machine translation, the testing of Machine Translation Systems (MTSs) has attracted much attention. Recent works apply Metamorphic Testing (MT) to address the oracle problem in MTS testing. Existing MT methods for MTS generally follow the workflow of input transformation and output relation comparison, which generates a follow-up input sentence by mutating the source input and compares the source and follow-up output translations to detect translation errors, respectively. These methods use various input transformations to generate test case pairs and have successfully triggered numerous translation errors. However, they have limitations in performing fine-grained and rigorous output relation comparison and thus may report false alarms and miss true errors. In this paper, we propose a word closure-based output comparison method to address the limitations of the existing MTS MT methods. Specifically, we first build a new comparison unit called word closure, where each closure includes a group of correlated input and output words in the test case pair. Word closures suggest the linkages between the appropriate fragment in the source output translation and its counterpart in the follow-up output for comparison. Next, we compare the semantics on the level of word closure to identify the translation errors. In this way, we perform a fine-grained and rigorous semantic comparison for the outputs and thus realize more effective violation identification. We evaluate our method with the test cases generated by five existing input transformations and translation outputs from three popular MTSs. Results show that our method significantly outperforms the existing works in violation identification by improving the precision and recall and achieving an average increase of 29.8% in F1 score. It also helps to increase the F1 score of translation error localization by 35.9%.

{{</citation>}}


### (106/148) Xpert: Empowering Incident Management with Query Recommendations via Large Language Models (Yuxuan Jiang et al., 2023)

{{<citation>}}

Yuxuan Jiang, Chaoyun Zhang, Shilin He, Zhihao Yang, Minghua Ma, Si Qin, Yu Kang, Yingnong Dang, Saravan Rajmohan, Qingwei Lin, Dongmei Zhang. (2023)  
**Xpert: Empowering Incident Management with Query Recommendations via Large Language Models**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-PL, cs-SE, cs.SE  
Keywords: Language Model, Microsoft  
[Paper Link](http://arxiv.org/abs/2312.11988v1)  

---


**ABSTRACT**  
Large-scale cloud systems play a pivotal role in modern IT infrastructure. However, incidents occurring within these systems can lead to service disruptions and adversely affect user experience. To swiftly resolve such incidents, on-call engineers depend on crafting domain-specific language (DSL) queries to analyze telemetry data. However, writing these queries can be challenging and time-consuming. This paper presents a thorough empirical study on the utilization of queries of KQL, a DSL employed for incident management in a large-scale cloud management system at Microsoft. The findings obtained underscore the importance and viability of KQL queries recommendation to enhance incident management.   Building upon these valuable insights, we introduce Xpert, an end-to-end machine learning framework that automates KQL recommendation process. By leveraging historical incident data and large language models, Xpert generates customized KQL queries tailored to new incidents. Furthermore, Xpert incorporates a novel performance metric called Xcore, enabling a thorough evaluation of query quality from three comprehensive perspectives. We conduct extensive evaluations of Xpert, demonstrating its effectiveness in offline settings. Notably, we deploy Xpert in the real production environment of a large-scale incident management system in Microsoft, validating its efficiency in supporting incident management. To the best of our knowledge, this paper represents the first empirical study of its kind, and Xpert stands as a pioneering DSL query recommendation framework designed for incident management.

{{</citation>}}


### (107/148) Predicting Line-Level Defects by Capturing Code Contexts with Hierarchical Transformers (Parvez Mahbub et al., 2023)

{{<citation>}}

Parvez Mahbub, Mohammad Masudur Rahman. (2023)  
**Predicting Line-Level Defects by Capturing Code Contexts with Hierarchical Transformers**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: QA, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.11889v1)  

---


**ABSTRACT**  
Software defects consume 40% of the total budget in software development and cost the global economy billions of dollars every year. Unfortunately, despite the use of many software quality assurance (SQA) practices in software development (e.g., code review, continuous integration), defects may still exist in the official release of a software product. Therefore, prioritizing SQA efforts for the vulnerable areas of the codebase is essential to ensure the high quality of a software release. Predicting software defects at the line level could help prioritize the SQA effort but is a highly challenging task given that only ~3% of lines of a codebase could be defective. Existing works on line-level defect prediction often fall short and cannot fully leverage the line-level defect information. In this paper, we propose Bugsplorer, a novel deep-learning technique for line-level defect prediction. It leverages a hierarchical structure of transformer models to represent two types of code elements: code tokens and code lines. Unlike the existing techniques that are optimized for file-level defect prediction, Bugsplorer is optimized for a line-level defect prediction objective. Our evaluation with five performance metrics shows that Bugsplorer has a promising capability of predicting defective lines with 26-72% better accuracy than that of the state-of-the-art technique. It can rank the first 20% defective lines within the top 1-3% suspicious lines. Thus, Bugsplorer has the potential to significantly reduce SQA costs by ranking defective lines higher.

{{</citation>}}


## eess.SP (1)



### (108/148) Real-Time Diagnostic Integrity Meets Efficiency: A Novel Platform-Agnostic Architecture for Physiological Signal Compression (Neel R Vora et al., 2023)

{{<citation>}}

Neel R Vora, Amir Hajighasemi, Cody T. Reynolds, Amirmohammad Radmehr, Mohamed Mohamed, Jillur Rahman Saurav, Abdul Aziz, Jai Prakash Veerla, Mohammad S Nasr, Hayden Lotspeich, Partha Sai Guttikonda, Thuong Pham, Aarti Darji, Parisa Boodaghi Malidarreh, Helen H Shang, Jay Harvey, Kan Ding, Phuc Nguyen, Jacob M Luber. (2023)  
**Real-Time Diagnostic Integrity Meets Efficiency: A Novel Platform-Agnostic Architecture for Physiological Signal Compression**  

---
Primary Category: eess.SP  
Categories: cs-DC, eess-SP, eess.SP, q-bio-TO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.12587v1)  

---


**ABSTRACT**  
Head-based signals such as EEG, EMG, EOG, and ECG collected by wearable systems will play a pivotal role in clinical diagnosis, monitoring, and treatment of important brain disorder diseases.   However, the real-time transmission of the significant corpus physiological signals over extended periods consumes substantial power and time, limiting the viability of battery-dependent physiological monitoring wearables.   This paper presents a novel deep-learning framework employing a variational autoencoder (VAE) for physiological signal compression to reduce wearables' computational complexity and energy consumption.   Our approach achieves an impressive compression ratio of 1:293 specifically for spectrogram data, surpassing state-of-the-art compression techniques such as JPEG2000, H.264, Direct Cosine Transform (DCT), and Huffman Encoding, which do not excel in handling physiological signals.   We validate the efficacy of the compressed algorithms using collected physiological signals from real patients in the Hospital and deploy the solution on commonly used embedded AI chips (i.e., ARM Cortex V8 and Jetson Nano). The proposed framework achieves a 91{\%} seizure detection accuracy using XGBoost, confirming the approach's reliability, practicality, and scalability.

{{</citation>}}


## cs.CE (1)



### (109/148) An integrated EOS, pore-crush, strength and damage model framework for near-field ground-shock (Kane C. Bennett et al., 2023)

{{<citation>}}

Kane C. Bennett, Alyson M. Stahl, Thomas R. Canfield, Garrett G. Euler. (2023)  
**An integrated EOS, pore-crush, strength and damage model framework for near-field ground-shock**  

---
Primary Category: cs.CE  
Categories: cs-CE, cs.CE  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.12577v1)  

---


**ABSTRACT**  
An integrated Equation of State (EOS) and strength/pore-crush/damage model framework is provided for modeling near to source (near-field) ground-shock response, where large deformations and pressures necessitate coupling EOS with pressure-dependent plastic yield and damage. Nonlinear pressure-dependence of strength up to high-pressures is combined with a Modified Cam-Clay-like cap-plasticity model in a way to allow degradation of strength from pore-crush damage, what we call the ``Yp-Cap'' model. Nonlinear hardening under compaction allows modeling the crush-out of pores in combination with a fully saturated EOS, i.e., for modeling partially saturated ground-shock response, where air-filled voids crush. Attention is given to algorithmic clarity and efficiency of the provided model, and the model is employed in example numerical simulations, including finite element simulations of underground explosions to exemplify its robustness and utility.

{{</citation>}}


## cs.CR (2)



### (110/148) Can Large Language Models Identify And Reason About Security Vulnerabilities? Not Yet (Saad Ullah et al., 2023)

{{<citation>}}

Saad Ullah, Mingji Han, Saurabh Pujar, Hammond Pearce, Ayse Coskun, Gianluca Stringhini. (2023)  
**Can Large Language Models Identify And Reason About Security Vulnerabilities? Not Yet**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: GPT, GPT-4, Language Model, PaLM, Security  
[Paper Link](http://arxiv.org/abs/2312.12575v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have been suggested for use in automated vulnerability repair, but benchmarks showing they can consistently identify security-related bugs are lacking. We thus perform the most detailed investigation to date on whether LLMs can reliably identify security-related bugs. We construct a series of 228 code scenarios and analyze eight of the most capable LLMs across eight different investigative dimensions in an automated framework. Our evaluation shows LLMs provide non-deterministic responses, incorrect and unfaithful reasoning, and perform poorly in real-world scenarios outside their knowledge cut-off date. Most importantly, our findings reveal significant non-robustness in even the most advanced models like `PaLM2' and `GPT-4': by merely changing function or variable names, or by the addition of library functions in the source code, these models can yield incorrect answers in 26% and 17% of cases, respectively. These findings demonstrate that further LLM advances are needed before LLMs can be used as general purpose security assistants.

{{</citation>}}


### (111/148) SoK: Security of Cross-chain Bridges: Attack Surfaces, Defenses, and Open Problems (Mengya Zhang et al., 2023)

{{<citation>}}

Mengya Zhang, Xiaokuan Zhang, Josh Barbee, Yinqian Zhang, Zhiqiang Lin. (2023)  
**SoK: Security of Cross-chain Bridges: Attack Surfaces, Defenses, and Open Problems**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2312.12573v1)  

---


**ABSTRACT**  
Cross-chain bridges are used to facilitate token and data exchanges across blockchains. Although bridges are becoming increasingly popular, they are still in their infancy and have been attacked multiple times recently, causing significant financial loss. Although there are numerous reports online explaining each of the incidents on cross-chain bridges, they are scattered over the Internet, and there is no work that analyzes the security landscape of cross-chain bridges in a holistic manner. To fill the gap, in this paper, we performed a systematic study of cross-chain bridge security issues. First, we summarize the characteristics of existing cross-chain bridges, including their usages, verification mechanisms, communication models, and three categorizations. Based on these characteristics, we identify 12 potential attack vectors that attackers may exploit. Next, we introduce a taxonomy that categorizes cross-chain attacks in the past two years into 10 distinct types, and then provide explanations for each vulnerability type, accompanied by Solidity code examples. We also discuss existing and potential defenses, as well as open questions and future research directions on cross-chain bridges. We believe that this systematization can shed light on designing and implementing cross-chain bridges with higher security and, more importantly, facilitating future research on building a better cross-chain bridge ecosystem.

{{</citation>}}


## cs.CY (1)



### (112/148) Beyond Fairness: Alternative Moral Dimensions for Assessing Algorithms and Designing Systems (Kimi Wenzel et al., 2023)

{{<citation>}}

Kimi Wenzel, Geoff Kaufman, Laura Dabbish. (2023)  
**Beyond Fairness: Alternative Moral Dimensions for Assessing Algorithms and Designing Systems**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs-HC, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.12559v1)  

---


**ABSTRACT**  
The ethics of artificial intelligence (AI) systems has risen as an imminent concern across scholarly communities. This concern has propagated a great interest in algorithmic fairness. Large research agendas are now devoted to increasing algorithmic fairness, assessing algorithmic fairness, and understanding human perceptions of fairness. We argue that there is an overreliance on fairness as a single dimension of morality, which comes at the expense of other important human values. Drawing from moral psychology, we present five moral dimensions that go beyond fairness, and suggest three ways these alternative dimensions may contribute to ethical AI development.

{{</citation>}}


## math.NA (1)



### (113/148) Tensor Train Decomposition for Adversarial Attacks on Computer Vision Models (Andrei Chertkov et al., 2023)

{{<citation>}}

Andrei Chertkov, Ivan Oseledets. (2023)  
**Tensor Train Decomposition for Adversarial Attacks on Computer Vision Models**  

---
Primary Category: math.NA  
Categories: cs-NA, math-NA, math.NA  
Keywords: Adversarial Attack, Computer Vision, ImageNet  
[Paper Link](http://arxiv.org/abs/2312.12556v1)  

---


**ABSTRACT**  
Deep neural networks (DNNs) are widely used today, but they are vulnerable to adversarial attacks. To develop effective methods of defense, it is important to understand the potential weak spots of DNNs. Often attacks are organized taking into account the architecture of models (white-box approach) and based on gradient methods, but for real-world DNNs this approach in most cases is impossible. At the same time, several gradient-free optimization algorithms are used to attack black-box models. However, classical methods are often ineffective in the multidimensional case. To organize black-box attacks for computer vision models, in this work, we propose the use of an optimizer based on the low-rank tensor train (TT) format, which has gained popularity in various practical multidimensional applications in recent years. Combined with the attribution of the target image, which is built by the auxiliary (white-box) model, the TT-based optimization method makes it possible to organize an effective black-box attack by small perturbation of pixels in the target image. The superiority of the proposed approach over three popular baselines is demonstrated for five modern DNNs on the ImageNet dataset.

{{</citation>}}


## cs.IR (3)



### (114/148) Efficient Title Reranker for Fast and Improved Knowledge-Intense NLP (Ziyi Chen et al., 2023)

{{<citation>}}

Ziyi Chen, Heyi Tao, Daqian Zuo, Jize Jiang, Jun Yang, Yuxiang Wei. (2023)  
**Efficient Title Reranker for Fast and Improved Knowledge-Intense NLP**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-CL, cs-IR, cs-LG, cs.IR  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2312.12430v2)  

---


**ABSTRACT**  
We introduce Efficient Title Reranker via Broadcasting Query Encoder, a novel title reranking technique to achieve efficient title reranking 20x-40x faster than vanilla passage reranker. However, one of the challenges with the training of Efficient Title Reranker is the instability. Analyzing the issue, we found some very difficult ground truths might act as noisy labels causing accuracy to drop as well as some extreme values in model probability output causing nan. To address these issues, we introduce the Sigmoid Trick, a novel technique that reduces the gradient update of both cases resulting in better retrieval efficacy. Experiments showed the effectiveness of ETR and sigmoid trick as we achieved four state-of-the-art positions on the kilt knowledge benchmark.

{{</citation>}}


### (115/148) PEPT: Expert Finding Meets Personalized Pre-training (Qiyao Peng et al., 2023)

{{<citation>}}

Qiyao Peng, Hongtao Liu, Hongyan Xu, Yinghui Wang, Wenjun Wang. (2023)  
**PEPT: Expert Finding Meets Personalized Pre-training**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2312.12162v1)  

---


**ABSTRACT**  
Finding appropriate experts is essential in Community Question Answering (CQA) platforms as it enables the effective routing of questions to potential users who can provide relevant answers. The key is to personalized learning expert representations based on their historical answered questions, and accurately matching them with target questions. There have been some preliminary works exploring the usability of PLMs in expert finding, such as pre-training expert or question representations. However, these models usually learn pure text representations of experts from histories, disregarding personalized and fine-grained expert modeling. For alleviating this, we present a personalized pre-training and fine-tuning paradigm, which could effectively learn expert interest and expertise simultaneously. Specifically, in our pre-training framework, we integrate historical answered questions of one expert with one target question, and regard it as a candidate aware expert-level input unit. Then, we fuse expert IDs into the pre-training for guiding the model to model personalized expert representations, which can help capture the unique characteristics and expertise of each individual expert. Additionally, in our pre-training task, we design: 1) a question-level masked language model task to learn the relatedness between histories, enabling the modeling of question-level expert interest; 2) a vote-oriented task to capture question-level expert expertise by predicting the vote score the expert would receive. Through our pre-training framework and tasks, our approach could holistically learn expert representations including interests and expertise. Our method has been extensively evaluated on six real-world CQA datasets, and the experimental results consistently demonstrate the superiority of our approach over competitive baseline methods.

{{</citation>}}


### (116/148) VITA: 'Carefully Chosen and Weighted Less' Is Better in Medication Recommendation (Taeri Kim et al., 2023)

{{<citation>}}

Taeri Kim, Jiho Heo, Hongil Kim, Kijung Shin, Sang-Wook Kim. (2023)  
**VITA: 'Carefully Chosen and Weighted Less' Is Better in Medication Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.12100v1)  

---


**ABSTRACT**  
We address the medication recommendation problem, which aims to recommend effective medications for a patient's current visit by utilizing information (e.g., diagnoses and procedures) given at the patient's current and past visits. While there exist a number of recommender systems designed for this problem, we point out that they are challenged in accurately capturing the relation (spec., the degree of relevance) between the current and each of the past visits for the patient when obtaining her current health status, which is the basis for recommending medications. To address this limitation, we propose a novel medication recommendation framework, named VITA, based on the following two novel ideas: (1) relevant-Visit selectIon; (2) Target-aware Attention. Through extensive experiments using real-world datasets, we demonstrate the superiority of VITA (spec., up to 5.56% higher accuracy, in terms of Jaccard, than the best competitor) and the effectiveness of its two core ideas. The code is available at https://github.com/jhheo0123/VITA.

{{</citation>}}


## cs.AI (11)



### (117/148) On Alternating-time Temporal Logic, Hyperproperties, and Strategy Sharing (Raven Beutner et al., 2023)

{{<citation>}}

Raven Beutner, Bernd Finkbeiner. (2023)  
**On Alternating-time Temporal Logic, Hyperproperties, and Strategy Sharing**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LO, cs-MA, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.12403v1)  

---


**ABSTRACT**  
Alternating-time temporal logic (ATL$^*$) is a well-established framework for formal reasoning about multi-agent systems. However, while ATL$^*$ can reason about the strategic ability of agents (e.g., some coalition $A$ can ensure that a goal is reached eventually), we cannot compare multiple strategic interactions, nor can we require multiple agents to follow the same strategy. For example, we cannot state that coalition $A$ can reach a goal sooner (or more often) than some other coalition $A'$. In this paper, we propose HyperATLS$^*_S$, an extension of ATL$^*$ in which we can (1) compare the outcome of multiple strategic interactions w.r.t. a hyperproperty, i.e., a property that refers to multiple paths at the same time, and (2) enforce that some agents share the same strategy. We show that HyperATL$^*_S$ is a rich specification language that captures important AI-related properties that were out of reach of existing logics. We prove that model checking of HyperATL$^*_S$ on concurrent game structures is decidable. We implement our model-checking algorithm in a tool we call HyMASMC and evaluate it on a range of benchmarks.

{{</citation>}}


### (118/148) Toward enriched Cognitive Learning with XAI (Muhammad Suffian et al., 2023)

{{<citation>}}

Muhammad Suffian, Ulrike Kuhl, Jose M. Alonso-Moral, Alessandro Bogliolo. (2023)  
**Toward enriched Cognitive Learning with XAI**  

---
Primary Category: cs.AI  
Categories: I-2-8, cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.12290v1)  

---


**ABSTRACT**  
As computational systems supported by artificial intelligence (AI) techniques continue to play an increasingly pivotal role in making high-stakes recommendations and decisions across various domains, the demand for explainable AI (XAI) has grown significantly, extending its impact into cognitive learning research. Providing explanations for novel concepts is recognised as a fundamental aid in the learning process, particularly when addressing challenges stemming from knowledge deficiencies and skill application. Addressing these difficulties involves timely explanations and guidance throughout the learning process, prompting the interest of AI experts in developing explainer models. In this paper, we introduce an intelligent system (CL-XAI) for Cognitive Learning which is supported by XAI, focusing on two key research objectives: exploring how human learners comprehend the internal mechanisms of AI models using XAI tools and evaluating the effectiveness of such tools through human feedback. The use of CL-XAI is illustrated with a game-inspired virtual use case where learners tackle combinatorial problems to enhance problem-solving skills and deepen their understanding of complex concepts, highlighting the potential for transformative advances in cognitive learning and co-learning.

{{</citation>}}


### (119/148) Mindful Explanations: Prevalence and Impact of Mind Attribution in XAI Research (Susanne Hindennach et al., 2023)

{{<citation>}}

Susanne Hindennach, Lei Shi, Filip Miletić, Andreas Bulling. (2023)  
**Mindful Explanations: Prevalence and Impact of Mind Attribution in XAI Research**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.12119v1)  

---


**ABSTRACT**  
When users perceive AI systems as mindful, independent agents, they hold them responsible instead of the AI experts who created and designed these systems. So far, it has not been studied whether explanations support this shift in responsibility through the use of mind-attributing verbs like "to think". To better understand the prevalence of mind-attributing explanations we analyse AI explanations in 3,533 explainable AI (XAI) research articles from the Semantic Scholar Open Research Corpus (S2ORC). Using methods from semantic shift detection, we identify three dominant types of mind attribution: (1) metaphorical (e.g. "to learn" or "to predict"), (2) awareness (e.g. "to consider"), and (3) agency (e.g. "to make decisions"). We then analyse the impact of mind-attributing explanations on awareness and responsibility in a vignette-based experiment with 199 participants. We find that participants who were given a mind-attributing explanation were more likely to rate the AI system as aware of the harm it caused. Moreover, the mind-attributing explanation had a responsibility-concealing effect: Considering the AI experts' involvement lead to reduced ratings of AI responsibility for participants who were given a non-mind-attributing or no explanation. In contrast, participants who read the mind-attributing explanation still held the AI system responsible despite considering the AI experts' involvement. Taken together, our work underlines the need to carefully phrase explanations about AI systems in scientific writing to reduce mind attribution and clearly communicate human responsibility.

{{</citation>}}


### (120/148) I-CEE: Tailoring Explanations of Image Classifications Models to User Expertise (Yao Rong et al., 2023)

{{<citation>}}

Yao Rong, Peizhu Qian, Vaibhav Unhelkar, Enkelejda Kasneci. (2023)  
**I-CEE: Tailoring Explanations of Image Classifications Models to User Expertise**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CV, cs-HC, cs-LG, cs.AI  
Keywords: AI, Image Classification  
[Paper Link](http://arxiv.org/abs/2312.12102v1)  

---


**ABSTRACT**  
Effectively explaining decisions of black-box machine learning models is critical to responsible deployment of AI systems that rely on them. Recognizing their importance, the field of explainable AI (XAI) provides several techniques to generate these explanations. Yet, there is relatively little emphasis on the user (the explainee) in this growing body of work and most XAI techniques generate "one-size-fits-all" explanations. To bridge this gap and achieve a step closer towards human-centered XAI, we present I-CEE, a framework that provides Image Classification Explanations tailored to User Expertise. Informed by existing work, I-CEE explains the decisions of image classification models by providing the user with an informative subset of training data (i.e., example images), corresponding local explanations, and model decisions. However, unlike prior work, I-CEE models the informativeness of the example images to depend on user expertise, resulting in different examples for different users. We posit that by tailoring the example set to user expertise, I-CEE can better facilitate users' understanding and simulatability of the model. To evaluate our approach, we conduct detailed experiments in both simulation and with human participants (N = 100) on multiple datasets. Experiments with simulated users show that I-CEE improves users' ability to accurately predict the model's decisions (simulatability) compared to baselines, providing promising preliminary results. Experiments with human participants demonstrate that our method significantly improves user simulatability accuracy, highlighting the importance of human-centered XAI

{{</citation>}}


### (121/148) Large Language Models Empowered Agent-based Modeling and Simulation: A Survey and Perspectives (Chen Gao et al., 2023)

{{<citation>}}

Chen Gao, Xiaochong Lan, Nian Li, Yuan Yuan, Jingtao Ding, Zhilun Zhou, Fengli Xu, Yong Li. (2023)  
**Large Language Models Empowered Agent-based Modeling and Simulation: A Survey and Perspectives**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-CY, cs-MA, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.11970v1)  

---


**ABSTRACT**  
Agent-based modeling and simulation has evolved as a powerful tool for modeling complex systems, offering insights into emergent behaviors and interactions among diverse agents. Integrating large language models into agent-based modeling and simulation presents a promising avenue for enhancing simulation capabilities. This paper surveys the landscape of utilizing large language models in agent-based modeling and simulation, examining their challenges and promising future directions. In this survey, since this is an interdisciplinary field, we first introduce the background of agent-based modeling and simulation and large language model-empowered agents. We then discuss the motivation for applying large language models to agent-based simulation and systematically analyze the challenges in environment perception, human alignment, action generation, and evaluation. Most importantly, we provide a comprehensive overview of the recent works of large language model-empowered agent-based modeling and simulation in multiple scenarios, which can be divided into four domains: cyber, physical, social, and hybrid, covering simulation of both real-world and virtual environments. Finally, since this area is new and quickly evolving, we discuss the open problems and promising future directions.

{{</citation>}}


### (122/148) Vertical Symbolic Regression (Nan Jiang et al., 2023)

{{<citation>}}

Nan Jiang, Md Nasim, Yexiang Xue. (2023)  
**Vertical Symbolic Regression**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.11955v1)  

---


**ABSTRACT**  
Automating scientific discovery has been a grand goal of Artificial Intelligence (AI) and will bring tremendous societal impact. Learning symbolic expressions from experimental data is a vital step in AI-driven scientific discovery. Despite exciting progress, most endeavors have focused on the horizontal discovery paths, i.e., they directly search for the best expression in the full hypothesis space involving all the independent variables. Horizontal paths are challenging due to the exponentially large hypothesis space involving all the independent variables. We propose Vertical Symbolic Regression (VSR) to expedite symbolic regression. The VSR starts by fitting simple expressions involving a few independent variables under controlled experiments where the remaining variables are held constant. It then extends the expressions learned in previous rounds by adding new independent variables and using new control variable experiments allowing these variables to vary. The first few steps in vertical discovery are significantly cheaper than the horizontal path, as their search is in reduced hypothesis spaces involving a small set of variables. As a consequence, vertical discovery has the potential to supercharge state-of-the-art symbolic regression approaches in handling complex equations with many contributing factors. Theoretically, we show that the search space of VSR can be exponentially smaller than that of horizontal approaches when learning a class of expressions. Experimentally, VSR outperforms several baselines in learning symbolic expressions involving many independent variables.

{{</citation>}}


### (123/148) Large Language Models Play StarCraft II: Benchmarks and A Chain of Summarization Approach (Weiyu Ma et al., 2023)

{{<citation>}}

Weiyu Ma, Qirui Mi, Xue Yan, Yuqiao Wu, Runji Lin, Haifeng Zhang, Jun Wang. (2023)  
**Large Language Models Play StarCraft II: Benchmarks and A Chain of Summarization Approach**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, GPT, Language Model, Summarization  
[Paper Link](http://arxiv.org/abs/2312.11865v1)  

---


**ABSTRACT**  
StarCraft II is a challenging benchmark for AI agents due to the necessity of both precise micro level operations and strategic macro awareness. Previous works, such as Alphastar and SCC, achieve impressive performance on tackling StarCraft II , however, still exhibit deficiencies in long term strategic planning and strategy interpretability. Emerging large language model (LLM) agents, such as Voyage and MetaGPT, presents the immense potential in solving intricate tasks. Motivated by this, we aim to validate the capabilities of LLMs on StarCraft II, a highly complex RTS game.To conveniently take full advantage of LLMs` reasoning abilities, we first develop textual StratCraft II environment, called TextStarCraft II, which LLM agent can interact. Secondly, we propose a Chain of Summarization method, including single frame summarization for processing raw observations and multi frame summarization for analyzing game information, providing command recommendations, and generating strategic decisions. Our experiment consists of two parts: first, an evaluation by human experts, which includes assessing the LLMs`s mastery of StarCraft II knowledge and the performance of LLM agents in the game; second, the in game performance of LLM agents, encompassing aspects like win rate and the impact of Chain of Summarization.Experiment results demonstrate that: 1. LLMs possess the relevant knowledge and complex planning abilities needed to address StarCraft II scenarios; 2. Human experts consider the performance of LLM agents to be close to that of an average player who has played StarCraft II for eight years; 3. LLM agents are capable of defeating the built in AI at the Harder(Lv5) difficulty level. We have open sourced the code and released demo videos of LLM agent playing StarCraft II.

{{</citation>}}


### (124/148) A Dual-way Enhanced Framework from Text Matching Point of View for Multimodal Entity Linking (Shezheng Song et al., 2023)

{{<citation>}}

Shezheng Song, Shan Zhao, Chengyu Wang, Tianwei Yan, Shasha Li, Xiaoguang Mao, Meng Wang. (2023)  
**A Dual-way Enhanced Framework from Text Matching Point of View for Multimodal Entity Linking**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CV, cs.AI  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2312.11816v1)  

---


**ABSTRACT**  
Multimodal Entity Linking (MEL) aims at linking ambiguous mentions with multimodal information to entity in Knowledge Graph (KG) such as Wikipedia, which plays a key role in many applications. However, existing methods suffer from shortcomings, including modality impurity such as noise in raw image and ambiguous textual entity representation, which puts obstacles to MEL. We formulate multimodal entity linking as a neural text matching problem where each multimodal information (text and image) is treated as a query, and the model learns the mapping from each query to the relevant entity from candidate entities. This paper introduces a dual-way enhanced (DWE) framework for MEL: (1) our model refines queries with multimodal data and addresses semantic gaps using cross-modal enhancers between text and image information. Besides, DWE innovatively leverages fine-grained image attributes, including facial characteristic and scene feature, to enhance and refine visual features. (2)By using Wikipedia descriptions, DWE enriches entity semantics and obtains more comprehensive textual representation, which reduces between textual representation and the entities in KG. Extensive experiments on three public benchmarks demonstrate that our method achieves state-of-the-art (SOTA) performance, indicating the superiority of our model. The code is released on https://github.com/season1blue/DWE

{{</citation>}}


### (125/148) Urban Generative Intelligence (UGI): A Foundational Platform for Agents in Embodied City Environment (Fengli Xu et al., 2023)

{{<citation>}}

Fengli Xu, Jun Zhang, Chen Gao, Jie Feng, Yong Li. (2023)  
**Urban Generative Intelligence (UGI): A Foundational Platform for Agents in Embodied City Environment**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CY, cs.AI  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2312.11813v1)  

---


**ABSTRACT**  
Urban environments, characterized by their complex, multi-layered networks encompassing physical, social, economic, and environmental dimensions, face significant challenges in the face of rapid urbanization. These challenges, ranging from traffic congestion and pollution to social inequality, call for advanced technological interventions. Recent developments in big data, artificial intelligence, urban computing, and digital twins have laid the groundwork for sophisticated city modeling and simulation. However, a gap persists between these technological capabilities and their practical implementation in addressing urban challenges in an systemic-intelligent way. This paper proposes Urban Generative Intelligence (UGI), a novel foundational platform integrating Large Language Models (LLMs) into urban systems to foster a new paradigm of urban intelligence. UGI leverages CityGPT, a foundation model trained on city-specific multi-source data, to create embodied agents for various urban tasks. These agents, operating within a textual urban environment emulated by city simulator and urban knowledge graph, interact through a natural language interface, offering an open platform for diverse intelligent and embodied agent development. This platform not only addresses specific urban issues but also simulates complex urban systems, providing a multidisciplinary approach to understand and manage urban complexity. This work signifies a transformative step in city science and urban intelligence, harnessing the power of LLMs to unravel and address the intricate dynamics of urban systems. The code repository with demonstrations will soon be released here https://github.com/tsinghua-fib-lab/UGI.

{{</citation>}}


### (126/148) Curriculum Learning for Cooperation in Multi-Agent Reinforcement Learning (Rupali Bhati et al., 2023)

{{<citation>}}

Rupali Bhati, Sai Krishna Gottipati, Clodéric Mars, Matthew E. Taylor. (2023)  
**Curriculum Learning for Cooperation in Multi-Agent Reinforcement Learning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs-MA, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.11768v1)  

---


**ABSTRACT**  
While there has been significant progress in curriculum learning and continuous learning for training agents to generalize across a wide variety of environments in the context of single-agent reinforcement learning, it is unclear if these algorithms would still be valid in a multi-agent setting. In a competitive setting, a learning agent can be trained by making it compete with a curriculum of increasingly skilled opponents. However, a general intelligent agent should also be able to learn to act around other agents and cooperate with them to achieve common goals. When cooperating with other agents, the learning agent must (a) learn how to perform the task (or subtask), and (b) increase the overall team reward. In this paper, we aim to answer the question of what kind of cooperative teammate, and a curriculum of teammates should a learning agent be trained with to achieve these two objectives. Our results on the game Overcooked show that a pre-trained teammate who is less skilled is the best teammate for overall team reward but the worst for the learning of the agent. Moreover, somewhat surprisingly, a curriculum of teammates with decreasing skill levels performs better than other types of curricula.

{{</citation>}}


### (127/148) MineObserver 2.0: A Deep Learning & In-Game Framework for Assessing Natural Language Descriptions of Minecraft Imagery (Jay Mahajan et al., 2023)

{{<citation>}}

Jay Mahajan, Samuel Hum, Jack Henhapl, Diya Yunus, Matthew Gadbury, Emi Brown, Jeff Ginger, H. Chad Lane. (2023)  
**MineObserver 2.0: A Deep Learning & In-Game Framework for Assessing Natural Language Descriptions of Minecraft Imagery**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, Computer Vision, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2312.11761v1)  

---


**ABSTRACT**  
MineObserver 2.0 is an AI framework that uses Computer Vision and Natural Language Processing for assessing the accuracy of learner-generated descriptions of Minecraft images that include some scientifically relevant content. The system automatically assesses the accuracy of participant observations, written in natural language, made during science learning activities that take place in Minecraft. We demonstrate our system working in real-time and describe a teacher support dashboard to showcase observations, both of which advance our previous work. We present the results of a study showing that MineObserver 2.0 improves over its predecessor both in perceived accuracy of the system's generated descriptions as well as in usefulness of the system's feedback. In future work we intend improve system-generated descriptions, give teachers more control and upgrade the system to perform continuous learning to more effectively and rapidly respond to novel observations made by learners.

{{</citation>}}


## astro-ph.EP (1)



### (128/148) Celestial Machine Learning: Discovering the Planarity, Heliocentricity, and Orbital Equation of Mars with AI Feynman (Zi-Yu Khoo et al., 2023)

{{<citation>}}

Zi-Yu Khoo, Gokul Rajiv, Abel Yang, Jonathan Sze Choong Low, Stéphane Bressan. (2023)  
**Celestial Machine Learning: Discovering the Planarity, Heliocentricity, and Orbital Equation of Mars with AI Feynman**  

---
Primary Category: astro-ph.EP  
Categories: astro-ph-EP, astro-ph.EP, cs-LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.12315v1)  

---


**ABSTRACT**  
Can a machine or algorithm discover or learn the elliptical orbit of Mars from astronomical sightings alone? Johannes Kepler required two paradigm shifts to discover his First Law regarding the elliptical orbit of Mars. Firstly, a shift from the geocentric to the heliocentric frame of reference. Secondly, the reduction of the orbit of Mars from a three- to a two-dimensional space. We extend AI Feynman, a physics-inspired tool for symbolic regression, to discover the heliocentricity and planarity of Mars' orbit and emulate his discovery of Kepler's first law.

{{</citation>}}


## q-bio.QM (1)



### (129/148) New Horizons: Pioneering Pharmaceutical R&D with Generative AI from lab to the clinic -- an industry perspective (Guy Doron et al., 2023)

{{<citation>}}

Guy Doron, Sam Genway, Mark Roberts, Sai Jasti. (2023)  
**New Horizons: Pioneering Pharmaceutical R&D with Generative AI from lab to the clinic -- an industry perspective**  

---
Primary Category: q-bio.QM  
Categories: 92C50, I-2-0; J-3, cs-LG, q-bio-QM, q-bio.QM  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2312.12482v1)  

---


**ABSTRACT**  
The rapid advance of generative AI is reshaping the strategic vision for R&D across industries. The unique challenges of pharmaceutical R&D will see applications of generative AI deliver value along the entire value chain from early discovery to regulatory approval. This perspective reviews these challenges and takes a three-horizon approach to explore the generative AI applications already delivering impact, the disruptive opportunities which are just around the corner, and the longer-term transformation which will shape the future of the industry. Selected applications are reviewed for their potential to drive increase productivity, accelerate timelines, improve the quality of research, data and decision making, and support a sustainable future for the industry. Recommendations are given for Pharma R&D leaders developing a generative AI strategy today which will lay the groundwork for getting real value from the technology and safeguarding future growth. Generative AI is today providing new, efficient routes to accessing and combining organisational data to drive productivity. Next, this impact will reach clinical development, enhancing the patient experience, driving operational efficiency, and unlocking digital innovation to better tackle the future burden of disease. Looking to the furthest horizon, rapid acquisition of rich multi-omics data, which capture the 'language of life', in combination with next generation AI technologies will allow organisations to close the loop around phases of the pipeline through rapid, automated generation and testing of hypotheses from bench to bedside. This provides a vision for the future of R&D with sustainability at the core, with reduced timescales and reduced dependency on resources, while offering new hope to patients to treat the untreatable and ultimately cure diseases.

{{</citation>}}


## cs.GR (1)



### (130/148) Sketch Vision: Artificial Intelligence with Sight for Imagination (Demircan Tas, 2023)

{{<citation>}}

Demircan Tas. (2023)  
**Sketch Vision: Artificial Intelligence with Sight for Imagination**  

---
Primary Category: cs.GR  
Categories: cs-GR, cs.GR  
Keywords: Sketch  
[Paper Link](http://arxiv.org/abs/2312.12270v1)  

---


**ABSTRACT**  
Visual design relies on seeing things in different ways, acting on them, and seeing results to act again. Parametric design tools are often not robust to design changes that result from sketching over the visualization of their output. We propose a sketch to 3d workflow as an experiment medium for evaluating neural networks and their latent spaces as a representation that is robust to overlay sketching.

{{</citation>}}


## cs.MM (1)



### (131/148) Low-Consumption Partial Transcoding by HEVC (Mohsen Abdoli et al., 2023)

{{<citation>}}

Mohsen Abdoli, Félix Henry, Gordon Clare. (2023)  
**Low-Consumption Partial Transcoding by HEVC**  

---
Primary Category: cs.MM  
Categories: cs-MM, cs.MM  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.12174v1)  

---


**ABSTRACT**  
A transcoding scheme for the High Efficiency Video Coding (HEVC) is proposed that allows any partial frame modification to be followed by a partial re-compression of only the modified areas, while guaranteeing identical reconstruction of non-modified areas. To this end, first, syntax elements of all Coding Units (CU) in the frame are parsed and decoded according to their scan order. Then CUs that are collocated with a replaced area are re-encoded with new content to generate a partial set of new syntax elements. In order to avoid spatial propagation of the decoding mismatch due to the new content, CUs on the border of the replaced area are losslessly coded such that reconstruction of immediately neighboring CUs in the scan order are protected from the modification. The proposed method has been implemented on top of the HEVC test Model (HM) in All-Intra (AI) coding configuration and experiments show that, depending on the test parameters, it can offer both a bitrate saving (up to 4% in terms of BD-BR) and a transcoding acceleration (up to 83%) compared to a full transcoding scheme.

{{</citation>}}


## cs.SD (3)



### (132/148) Noise robust distillation of self-supervised speech models via correlation metrics (Fabian Ritter-Gutierrez et al., 2023)

{{<citation>}}

Fabian Ritter-Gutierrez, Kuan-Po Huang, Dianwen Ng, Jeremy H. M. Wong, Hung-yi Lee, Eng Siong Chng, Nancy F. Chen. (2023)  
**Noise robust distillation of self-supervised speech models via correlation metrics**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2312.12153v1)  

---


**ABSTRACT**  
Compared to large speech foundation models, small distilled models exhibit degraded noise robustness. The student's robustness can be improved by introducing noise at the inputs during pre-training. Despite this, using the standard distillation loss still yields a student with degraded performance. Thus, this paper proposes improving student robustness via distillation with correlation metrics. Teacher behavior is learned by maximizing the teacher and student cross-correlation matrix between their representations towards identity. Noise robustness is encouraged via the student's self-correlation minimization. The proposed method is agnostic of the teacher model and consistently outperforms the previous approach. This work also proposes an heuristic to weigh the importance of the two correlation terms automatically. Experiments show consistently better clean and noise generalization on Intent Classification, Keyword Spotting, and Automatic Speech Recognition tasks on SUPERB Challenge.

{{</citation>}}


### (133/148) Ms-senet: Enhancing Speech Emotion Recognition Through Multi-scale Feature Fusion With Squeeze-and-excitation Blocks (Mengbo Li et al., 2023)

{{<citation>}}

Mengbo Li, Yulun Wu, Dichucheng Li, Yuanzhong Zheng, Yaoxuan Wang, Haojun Fei. (2023)  
**Ms-senet: Enhancing Speech Emotion Recognition Through Multi-scale Feature Fusion With Squeeze-and-excitation Blocks**  

---
Primary Category: cs.SD  
Categories: cs-HC, cs-SD, cs.SD, eess-AS  
Keywords: Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2312.11974v1)  

---


**ABSTRACT**  
Speech Emotion Recognition (SER) has become a growing focus of research in human-computer interaction. Spatiotemporal features play a crucial role in SER, yet current research lacks comprehensive spatiotemporal feature learning. This paper focuses on addressing this gap by proposing a novel approach. In this paper, we employ Convolutional Neural Network (CNN) with varying kernel sizes for spatial and temporal feature extraction. Additionally, we introduce Squeeze-and-Excitation (SE) modules to capture and fuse multi-scale features, facilitating effective information fusion for improved emotion recognition and a deeper understanding of the temporal evolution of speech emotion. Moreover, we employ skip connections and Spatial Dropout (SD) layers to prevent overfitting and increase the model's depth. Our method outperforms the previous state-of-the-art method, achieving an average UAR and WAR improvement of 1.62% and 1.32%, respectively, across six benchmark SER datasets. Further experiments demonstrated that our method can fully extract spatiotemporal features in low-resource conditions.

{{</citation>}}


### (134/148) MossFormer2: Combining Transformer and RNN-Free Recurrent Network for Enhanced Time-Domain Monaural Speech Separation (Shengkui Zhao et al., 2023)

{{<citation>}}

Shengkui Zhao, Yukun Ma, Chongjia Ni, Chong Zhang, Hao Wang, Trung Hieu Nguyen, Kun Zhou, Jiaqi Yip, Dianwen Ng, Bin Ma. (2023)  
**MossFormer2: Combining Transformer and RNN-Free Recurrent Network for Enhanced Time-Domain Monaural Speech Separation**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.11825v1)  

---


**ABSTRACT**  
Our previously proposed MossFormer has achieved promising performance in monaural speech separation. However, it predominantly adopts a self-attention-based MossFormer module, which tends to emphasize longer-range, coarser-scale dependencies, with a deficiency in effectively modelling finer-scale recurrent patterns. In this paper, we introduce a novel hybrid model that provides the capabilities to model both long-range, coarse-scale dependencies and fine-scale recurrent patterns by integrating a recurrent module into the MossFormer framework. Instead of applying the recurrent neural networks (RNNs) that use traditional recurrent connections, we present a recurrent module based on a feedforward sequential memory network (FSMN), which is considered "RNN-free" recurrent network due to the ability to capture recurrent patterns without using recurrent connections. Our recurrent module mainly comprises an enhanced dilated FSMN block by using gated convolutional units (GCU) and dense connections. In addition, a bottleneck layer and an output layer are also added for controlling information flow. The recurrent module relies on linear projections and convolutions for seamless, parallel processing of the entire sequence. The integrated MossFormer2 hybrid model demonstrates remarkable enhancements over MossFormer and surpasses other state-of-the-art methods in WSJ0-2/3mix, Libri2Mix, and WHAM!/WHAMR! benchmarks.

{{</citation>}}


## eess.IV (2)



### (135/148) Object Detection for Automated Coronary Artery Using Deep Learning (Hadis Keshavarz et al., 2023)

{{<citation>}}

Hadis Keshavarz, Hossein Sadr. (2023)  
**Object Detection for Automated Coronary Artery Using Deep Learning**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2312.12135v1)  

---


**ABSTRACT**  
In the era of digital medicine, medical imaging serves as a widespread technique for early disease detection, with a substantial volume of images being generated and stored daily in electronic patient records. X-ray angiography imaging is a standard and one of the most common methods for rapidly diagnosing coronary artery diseases. The notable achievements of recent deep learning algorithms align with the increased use of electronic health records and diagnostic imaging. Deep neural networks, leveraging abundant data, advanced algorithms, and powerful computational capabilities, prove highly effective in the analysis and interpretation of images. In this context, Object detection methods have become a promising approach, particularly through convolutional neural networks (CNN), streamlining medical image analysis by eliminating manual feature extraction. This allows for direct feature extraction from images, ensuring high accuracy in results. Therefore, in our paper, we utilized the object detection method on X-ray angiography images to precisely identify the location of coronary artery stenosis. As a result, this model enables automatic and real-time detection of stenosis locations, assisting in the crucial and sensitive decision-making process for healthcare professionals.

{{</citation>}}


### (136/148) Progressive Frequency-Aware Network for Laparoscopic Image Desmoking (Jiale Zhang et al., 2023)

{{<citation>}}

Jiale Zhang, Wenfeng Huang, Xiangyun Liao, Qiong Wang. (2023)  
**Progressive Frequency-Aware Network for Laparoscopic Image Desmoking**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Attention, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.12023v1)  

---


**ABSTRACT**  
Laparoscopic surgery offers minimally invasive procedures with better patient outcomes, but smoke presence challenges visibility and safety. Existing learning-based methods demand large datasets and high computational resources. We propose the Progressive Frequency-Aware Network (PFAN), a lightweight GAN framework for laparoscopic image desmoking, combining the strengths of CNN and Transformer for progressive information extraction in the frequency domain. PFAN features CNN-based Multi-scale Bottleneck-Inverting (MBI) Blocks for capturing local high-frequency information and Locally-Enhanced Axial Attention Transformers (LAT) for efficiently handling global low-frequency information. PFAN efficiently desmokes laparoscopic images even with limited training data. Our method outperforms state-of-the-art approaches in PSNR, SSIM, CIEDE2000, and visual quality on the Cholec80 dataset and retains only 629K parameters. Our code and models are made publicly available at: https://github.com/jlzcode/PFAN.

{{</citation>}}


## cs.HC (3)



### (137/148) Designing and Evaluating General-Purpose User Representations Based on Behavioral Logs from a Measurement Process Perspective: A Case Study with Snapchat (Qixiang Fang et al., 2023)

{{<citation>}}

Qixiang Fang, Zhihan Zhou, Francesco Barbieri, Yozen Liu, Leonardo Neves, Dong Nguyen, Daniel L. Oberski, Maarten W. Bos, Ron Dotsch. (2023)  
**Designing and Evaluating General-Purpose User Representations Based on Behavioral Logs from a Measurement Process Perspective: A Case Study with Snapchat**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs-IR, cs.HC  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.12111v1)  

---


**ABSTRACT**  
In human-computer interaction, understanding user behaviors and tailoring systems accordingly is pivotal. To this end, general-purpose user representation learning based on behavior logs is emerging as a powerful tool in user modeling, offering adaptability to various downstream tasks such as item recommendations and ad conversion prediction, without the need to fine-tune the upstream user model. While this methodology has shown promise in contexts like search engines and e-commerce platforms, its fit for instant messaging apps, a cornerstone of modern digital communication, remains largely uncharted. These apps, with their distinct interaction patterns, data structures, and user expectations, necessitate specialized attention. We explore this user modeling approach with Snapchat data as a case study. Furthermore, we introduce a novel design and evaluation framework rooted in the principles of the Measurement Process Framework from social science research methodology. Using this new framework, we design a Transformer-based user model that can produce high-quality general-purpose user representations for instant messaging platforms like Snapchat.

{{</citation>}}


### (138/148) Toward Responsible AI Use: Considerations for Sustainability Impact Assessment (Eva Thelisson et al., 2023)

{{<citation>}}

Eva Thelisson, Grzegorz Mika, Quentin Schneiter, Kirtan Padh, Himanshu Verma. (2023)  
**Toward Responsible AI Use: Considerations for Sustainability Impact Assessment**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2312.11996v1)  

---


**ABSTRACT**  
As AI/ML models, including Large Language Models, continue to scale with massive datasets, so does their consumption of undeniably limited natural resources, and impact on society. In this collaboration between AI, Sustainability, HCI and legal researchers, we aim to enable a transition to sustainable AI development by enabling stakeholders across the AI value chain to assess and quantitfy the environmental and societal impact of AI. We present the ESG Digital and Green Index (DGI), which offers a dashboard for assessing a company's performance in achieving sustainability targets. This includes monitoring the efficiency and sustainable use of limited natural resources related to AI technologies (water, electricity, etc). It also addresses the societal and governance challenges related to AI. The DGI creates incentives for companies to align their pathway with the Sustainable Development Goals (SDGs). The value, challenges and limitations of our methodology and findings are discussed in the paper.

{{</citation>}}


### (139/148) CreativeConnect: Supporting Reference Recombination for Graphic Design Ideation with Generative AI (DaEun Choi et al., 2023)

{{<citation>}}

DaEun Choi, Sumin Hong, Jeongeon Park, John Joon Young Chung, Juho Kim. (2023)  
**CreativeConnect: Supporting Reference Recombination for Graphic Design Ideation with Generative AI**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2312.11949v1)  

---


**ABSTRACT**  
Graphic designers often get inspiration through the recombination of references. Our formative study (N=6) reveals that graphic designers focus on conceptual keywords during this process, and want support for discovering the keywords, expanding them, and exploring diverse recombination options of them, while still having room for their creativity. We propose CreativeConnect, a system with generative AI pipelines that helps users discover useful elements from the reference image using keywords, recommends relevant keywords, generates diverse recombination options with user-selected keywords, and shows recombinations as sketches with text descriptions. Our user study (N=16) showed that CreativeConnect helped users discover keywords from the reference and generate multiple ideas based on them, ultimately helping users produce more design ideas and higher self-reported creativity, compared to the baseline system without generative pipelines. While CreativeConnect was effective in ideation, we discussed how CreativeConnect can be extended to support other types of tasks in creativity support.

{{</citation>}}


## cs.DC (2)



### (140/148) GraphScope Flex: LEGO-like Graph Computing Stack (Tao He et al., 2023)

{{<citation>}}

Tao He, Shuxian Hu, Longbin Lai, Dongze Li, Neng Li, Xue Li, Lexiao Liu, Xiaojian Luo, Binqing Lyu, Ke Meng, Sijie Shen, Li Su, Lei Wang, Jingbo Xu, Wenyuan Yu, Weibin Zeng, Lei Zhang, Siyuan Zhang, Jingren Zhou, Xiaoli Zhou, Diwen Zhu. (2023)  
**GraphScope Flex: LEGO-like Graph Computing Stack**  

---
Primary Category: cs.DC  
Categories: cs-DB, cs-DC, cs.DC  
Keywords: Social Network  
[Paper Link](http://arxiv.org/abs/2312.12107v1)  

---


**ABSTRACT**  
Graph computing has become increasingly crucial in processing large-scale graph data, with numerous systems developed for this purpose. Two years ago, we introduced GraphScope as a system addressing a wide array of graph computing needs, including graph traversal, analytics, and learning in one system. Since its inception, GraphScope has achieved significant technological advancements and gained widespread adoption across various industries. However, one key lesson from this journey has been understanding the limitations of a "one-size-fits-all" approach, especially when dealing with the diversity of programming interfaces, applications, and data storage formats in graph computing. In response to these challenges, we present GraphScope Flex, the next iteration of GraphScope. GraphScope Flex is designed to be both resource-efficient and cost-effective, while also providing flexibility and user-friendliness through its LEGO-like modularity. This paper explores the architectural innovations and fundamental design principles of GraphScope Flex, all of which are direct outcomes of the lessons learned during our ongoing development process. We validate the adaptability and efficiency of GraphScope Flex with extensive evaluations on synthetic and real-world datasets. The results show that GraphScope Flex achieves 2.4X throughput and up to 55.7X speedup over other systems on the LDBC Social Network and Graphalytics benchmarks, respectively. Furthermore, GraphScope Flex accomplishes up to a 2,400X performance gain in real-world applications, demonstrating its proficiency across a wide range of graph computing scenarios with increased effectiveness.

{{</citation>}}


### (141/148) Model-Heterogeneous Federated Learning for Internet of Things: Enabling Technologies and Future Directions (Boyu Fan et al., 2023)

{{<citation>}}

Boyu Fan, Siyang Jiang, Xiang Su, Pan Hui. (2023)  
**Model-Heterogeneous Federated Learning for Internet of Things: Enabling Technologies and Future Directions**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.12091v1)  

---


**ABSTRACT**  
Internet of Things (IoT) interconnects a massive amount of devices, generating heterogeneous data with diverse characteristics. IoT data emerges as a vital asset for data-intensive IoT applications, such as healthcare, smart city and predictive maintenance, harnessing the vast volume of heterogeneous data to its maximum advantage. These applications leverage different Artificial Intelligence (AI) algorithms to discover new insights. While machine learning effectively uncovers implicit patterns through model training, centralizing IoT data for training poses significant privacy and security concerns. Federated Learning (FL) offers an promising solution, allowing IoT devices to conduct local learning without sharing raw data with third parties. Model-heterogeneous FL empowers clients to train models with varying complexities based on their hardware capabilities, aligning with heterogeneity of devices in real-world IoT environments. In this article, we review the state-of-the-art model-heterogeneous FL methods and provide insights into their merits and limitations. Moreover, we showcase their applicability to IoT and identify the open problems and future directions. To the best of our knowledge, this is the first article that focuses on the topic of model-heterogeneous FL for IoT.

{{</citation>}}


## cs.MA (1)



### (142/148) Cautiously-Optimistic Knowledge Sharing for Cooperative Multi-Agent Reinforcement Learning (Yanwen Ba et al., 2023)

{{<citation>}}

Yanwen Ba, Xuan Liu, Xinning Chen, Hao Wang, Yang Xu, Kenli Li, Shigeng Zhang. (2023)  
**Cautiously-Optimistic Knowledge Sharing for Cooperative Multi-Agent Reinforcement Learning**  

---
Primary Category: cs.MA  
Categories: cs-MA, cs.MA  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.12095v1)  

---


**ABSTRACT**  
While decentralized training is attractive in multi-agent reinforcement learning (MARL) for its excellent scalability and robustness, its inherent coordination challenges in collaborative tasks result in numerous interactions for agents to learn good policies. To alleviate this problem, action advising methods make experienced agents share their knowledge about what to do, while less experienced agents strictly follow the received advice. However, this method of sharing and utilizing knowledge may hinder the team's exploration of better states, as agents can be unduly influenced by suboptimal or even adverse advice, especially in the early stages of learning. Inspired by the fact that humans can learn not only from the success but also from the failure of others, this paper proposes a novel knowledge sharing framework called Cautiously-Optimistic kNowledge Sharing (CONS). CONS enables each agent to share both positive and negative knowledge and cautiously assimilate knowledge from others, thereby enhancing the efficiency of early-stage exploration and the agents' robustness to adverse advice. Moreover, considering the continuous improvement of policies, agents value negative knowledge more in the early stages of learning and shift their focus to positive knowledge in the later stages. Our framework can be easily integrated into existing Q-learning based methods without introducing additional training costs. We evaluate CONS in several challenging multi-agent tasks and find it excels in environments where optimal behavioral patterns are difficult to discover, surpassing the baselines in terms of convergence rate and final performance.

{{</citation>}}


## cs.NI (3)



### (143/148) Resource-efficient Generative Mobile Edge Networks in 6G Era: Fundamentals, Framework and Case Study (Bingkun Lai et al., 2023)

{{<citation>}}

Bingkun Lai, Jinbo Wen, Jiawen Kang, Hongyang Du, Jiangtian Nie, Changyan Yi, Dong In Kim, Shengli Xie. (2023)  
**Resource-efficient Generative Mobile Edge Networks in 6G Era: Fundamentals, Framework and Case Study**  

---
Primary Category: cs.NI  
Categories: cs-AI, cs-GT, cs-NI, cs.NI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.12063v1)  

---


**ABSTRACT**  
As the next-generation wireless communication system, Sixth-Generation (6G) technologies are emerging, enabling various mobile edge networks that can revolutionize wireless communication and connectivity. By integrating Generative Artificial Intelligence (GAI) with mobile edge networks, generative mobile edge networks possess immense potential to enhance the intelligence and efficiency of wireless communication networks. In this article, we propose the concept of generative mobile edge networks and overview widely adopted GAI technologies and their applications in mobile edge networks. We then discuss the potential challenges faced by generative mobile edge networks in resource-constrained scenarios. To address these challenges, we develop a universal resource-efficient generative incentive mechanism framework, in which we design resource-efficient methods for network overhead reduction, formulate appropriate incentive mechanisms for the resource allocation problem, and utilize Generative Diffusion Models (GDMs) to find the optimal incentive mechanism solutions. Furthermore, we conduct a case study on resource-constrained mobile edge networks, employing model partition for efficient AI task offloading and proposing a GDM-based Stackelberg model to motivate edge devices to contribute computing resources for mobile edge intelligence. Finally, we propose several open directions that could contribute to the future popularity of generative mobile edge networks.

{{</citation>}}


### (144/148) Traffic Load Prediction and Power Consumption Reduction for Multi-band Networks (Ndolane Diouf et al., 2023)

{{<citation>}}

Ndolane Diouf, Cesar Vargas Anamuro, Cédric Gueguen, Massa Ndong, Kharouna Talla, Xavier Lagrange. (2023)  
**Traffic Load Prediction and Power Consumption Reduction for Multi-band Networks**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI, eess-SP  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2312.11958v1)  

---


**ABSTRACT**  
Energy is a major expense issue for mobile operators. In the case of wireless networks, base stations have been identified as the main source of energy consumption. In this paper, we study the energy consumption reduction problem based on real measurements for a commercial multi-band LTE network. Specifically, we are interested in sleep modes to turn off certain frequency bands during low traffic periods and consequently reduce power consumption. We determine the number of frequency bands really needed at each time period. The frequency bands that are not needed can be disabled to reduce energy consumption. In order to allow the operator to predict how many bands can be switched off without major impact on the quality of service, we propose to use a deep learning algorithm, such as Long-Short Term Memory (LSTM). Based on the captured data traces, we have shown that the proposed LSTM model can save an average of 8% to 21% of the energy consumption during working days.

{{</citation>}}


### (145/148) Improvement of inter-protocol fairness for BBR congestion control using machine learning (Vaishnavi Mhaske et al., 2023)

{{<citation>}}

Vaishnavi Mhaske, Khushi Jain, Sai Karthik Thatikonda, Asif Kunwar. (2023)  
**Improvement of inter-protocol fairness for BBR congestion control using machine learning**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2312.11790v1)  

---


**ABSTRACT**  
Google's BBR (Bottleneck Bandwidth and Round-trip Propagation Time) approach is used to enhance internet network transmission. It is particularly intended to efficiently handle enormous amounts of data. Traditional TCP (Transmission Control Protocol) algorithms confront the most difficulty in calculating the proper quantity of data to send in order to prevent congestion and bottlenecks. This wastes bandwidth and causes network delays. BBR addresses this issue by adaptively assessing the available bandwidth (also known as bottleneck bandwidth) along the network channel and calculating the round-trip time (RTT) between the sender and receiver. Although when several flows compete for bandwidth, BBR may supply more bandwidth to one flow at the expense of another, resulting in unequal resource distribution. This paper proposes to integrate machine learning with BBR to enhance fairness in resource allocation. This novel strategy can improve bandwidth allocation and provide a more equal distribution of resources among competing flows by using historical BBR data to train an ML model. Further we also implemented a classifier model that is graphic neural network in the congestion control method.

{{</citation>}}


## econ.GN (1)



### (146/148) Skills or Degree? The Rise of Skill-Based Hiring for AI and Green Jobs (Eugenia Gonzalez Ehlinger et al., 2023)

{{<citation>}}

Eugenia Gonzalez Ehlinger, Fabian Stephany. (2023)  
**Skills or Degree? The Rise of Skill-Based Hiring for AI and Green Jobs**  

---
Primary Category: econ.GN  
Categories: cs-AI, econ-GN, econ.GN, q-fin-EC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.11942v1)  

---


**ABSTRACT**  
For emerging professions, such as jobs in the field of Artificial Intelligence (AI) or sustainability (green), labour supply does not meet industry demand. In this scenario of labour shortages, our work aims to understand whether employers have started focusing on individual skills rather than on formal qualifications in their recruiting. By analysing a large time series dataset of around one million online job vacancies between 2019 and 2022 from the UK and drawing on diverse literature on technological change and labour market signalling, we provide evidence that employers have started so-called "skill-based hiring" for AI and green roles, as more flexible hiring practices allow them to increase the available talent pool. In our observation period the demand for AI roles grew twice as much as average labour demand. At the same time, the mention of university education for AI roles declined by 23%, while AI roles advertise five times as many skills as job postings on average. Our regression analysis also shows that university degrees no longer show an educational premium for AI roles, while for green positions the educational premium persists. In contrast, AI skills have a wage premium of 16%, similar to having a PhD (17%). Our work recommends making use of alternative skill building formats such as apprenticeships, on-the-job training, MOOCs, vocational education and training, micro-certificates, and online bootcamps to use human capital to its full potential and to tackle talent shortages.

{{</citation>}}


## math.OC (1)



### (147/148) Fast, Scalable, Warm-Start Semidefinite Programming with Spectral Bundling and Sketching (Rico Angell et al., 2023)

{{<citation>}}

Rico Angell, Andrew McCallum. (2023)  
**Fast, Scalable, Warm-Start Semidefinite Programming with Spectral Bundling and Sketching**  

---
Primary Category: math.OC  
Categories: cs-LG, math-OC, math.OC  
Keywords: Sketch  
[Paper Link](http://arxiv.org/abs/2312.11801v1)  

---


**ABSTRACT**  
While semidefinite programming (SDP) has traditionally been limited to moderate-sized problems, recent algorithms augmented with matrix sketching techniques have enabled solving larger SDPs. However, these methods achieve scalability at the cost of an increase in the number of necessary iterations, resulting in slower convergence as the problem size grows. Furthermore, they require iteration-dependent parameter schedules that prohibit effective utilization of warm-start initializations important in practical applications with incrementally-arriving data or mixed-integer programming. We present SpecBM, a provably correct, fast and scalable algorithm for solving massive SDPs that can leverage a warm-start initialization to further accelerate convergence. Our proposed algorithm is a spectral bundle method for solving general SDPs containing both equality and inequality constraints. Moveover, when augmented with an optional matrix sketching technique, our algorithm achieves the dramatically improved scalability of previous work while sustaining convergence speed. We empirically demonstrate the effectiveness of our method, both with and without warm-starting, across multiple applications with large instances. For example, on a problem with 600 million decision variables, SpecBM achieved a solution of standard accuracy in less than 7 minutes, where the previous state-of-the-art scalable SDP solver requires more than 16 hours. Our method solves an SDP with more than 10^13 decision variables on a single machine with 16 cores and no more than 128GB RAM; the previous state-of-the-art method had not achieved an accurate solution after 72 hours on the same instance. We make our implementation in pure JAX publicly available.

{{</citation>}}


## q-fin.PM (1)



### (148/148) Learning Merton's Strategies in an Incomplete Market: Recursive Entropy Regularization and Biased Gaussian Exploration (Min Dai et al., 2023)

{{<citation>}}

Min Dai, Yuchao Dong, Yanwei Jia, Xun Yu Zhou. (2023)  
**Learning Merton's Strategies in an Incomplete Market: Recursive Entropy Regularization and Biased Gaussian Exploration**  

---
Primary Category: q-fin.PM  
Categories: cs-LG, q-fin-CP, q-fin-PM, q-fin.PM  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2312.11797v1)  

---


**ABSTRACT**  
We study Merton's expected utility maximization problem in an incomplete market, characterized by a factor process in addition to the stock price process, where all the model primitives are unknown. We take the reinforcement learning (RL) approach to learn optimal portfolio policies directly by exploring the unknown market, without attempting to estimate the model parameters. Based on the entropy-regularization framework for general continuous-time RL formulated in Wang et al. (2020), we propose a recursive weighting scheme on exploration that endogenously discounts the current exploration reward by the past accumulative amount of exploration. Such a recursive regularization restores the optimality of Gaussian exploration. However, contrary to the existing results, the optimal Gaussian policy turns out to be biased in general, due to the interwinding needs for hedging and for exploration. We present an asymptotic analysis of the resulting errors to show how the level of exploration affects the learned policies. Furthermore, we establish a policy improvement theorem and design several RL algorithms to learn Merton's optimal strategies. At last, we carry out both simulation and empirical studies with a stochastic volatility environment to demonstrate the efficiency and robustness of the RL algorithms in comparison to the conventional plug-in method.

{{</citation>}}
