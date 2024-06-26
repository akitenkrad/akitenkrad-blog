---
draft: false
title: "arXiv @ 2023.08.30"
date: 2023-08-30
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.08.30"
    identifier: arxiv_20230830
    parent: 202308_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.LG (22)](#cslg-22)
- [cs.AI (17)](#csai-17)
- [quant-ph (1)](#quant-ph-1)
- [cs.CV (22)](#cscv-22)
- [cs.CL (20)](#cscl-20)
- [cond-mat.mtrl-sci (1)](#cond-matmtrl-sci-1)
- [cs.SD (3)](#cssd-3)
- [cs.HC (4)](#cshc-4)
- [cs.CR (4)](#cscr-4)
- [eess.AS (4)](#eessas-4)
- [cs.CY (4)](#cscy-4)
- [cs.SE (2)](#csse-2)
- [cs.SI (1)](#cssi-1)
- [cs.IT (1)](#csit-1)
- [cs.IR (2)](#csir-2)
- [stat.ML (1)](#statml-1)
- [astro-ph.IM (1)](#astro-phim-1)
- [eess.SY (1)](#eesssy-1)
- [cs.NI (1)](#csni-1)
- [cs.DL (1)](#csdl-1)
- [cs.LO (1)](#cslo-1)
- [cs.MM (1)](#csmm-1)
- [cs.RO (1)](#csro-1)
- [q-bio.GN (1)](#q-biogn-1)

## cs.LG (22)



### (1/117) Reinforcement Learning for Sampling on Temporal Medical Imaging Sequences (Zhishen Huang, 2023)

{{<citation>}}

Zhishen Huang. (2023)  
**Reinforcement Learning for Sampling on Temporal Medical Imaging Sequences**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, eess-IV  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.14946v1)  

---


**ABSTRACT**  
Accelerated magnetic resonance imaging resorts to either Fourier-domain subsampling or better reconstruction algorithms to deal with fewer measurements while still generating medical images of high quality. Determining the optimal sampling strategy given a fixed reconstruction protocol often has combinatorial complexity. In this work, we apply double deep Q-learning and REINFORCE algorithms to learn the sampling strategy for dynamic image reconstruction. We consider the data in the format of time series, and the reconstruction method is a pre-trained autoencoder-typed neural network. We present a proof of concept that reinforcement learning algorithms are effective to discover the optimal sampling pattern which underlies the pre-trained reconstructor network (i.e., the dynamics in the environment). The code for replicating experiments can be found at https://github.com/zhishenhuang/RLsamp.

{{</citation>}}


### (2/117) Maestro: Uncovering Low-Rank Structures via Trainable Decomposition (Samuel Horvath et al., 2023)

{{<citation>}}

Samuel Horvath, Stefanos Laskaridis, Shashank Rajput, Hongyi Wang. (2023)  
**Maestro: Uncovering Low-Rank Structures via Trainable Decomposition**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.14929v1)  

---


**ABSTRACT**  
Deep Neural Networks (DNNs) have been a large driver and enabler for AI breakthroughs in recent years. These models have been getting larger in their attempt to become more accurate and tackle new upcoming use-cases, including AR/VR and intelligent assistants. However, the training process of such large models is a costly and time-consuming process, which typically yields a single model to fit all targets. To mitigate this, various techniques have been proposed in the literature, including pruning, sparsification or quantization of the model weights and updates. While able to achieve high compression rates, they often incur computational overheads or accuracy penalties. Alternatively, factorization methods have been leveraged to incorporate low-rank compression in the training process. Similarly, such techniques (e.g.,~SVD) frequently rely on the computationally expensive decomposition of layers and are potentially sub-optimal for non-linear models, such as DNNs. In this work, we take a further step in designing efficient low-rank models and propose Maestro, a framework for trainable low-rank layers. Instead of regularly applying a priori decompositions such as SVD, the low-rank structure is built into the training process through a generalized variant of Ordered Dropout. This method imposes an importance ordering via sampling on the decomposed DNN structure. Our theoretical analysis demonstrates that our method recovers the SVD decomposition of linear mapping on uniformly distributed data and PCA for linear autoencoders. We further apply our technique on DNNs and empirically illustrate that Maestro enables the extraction of lower footprint models that preserve model performance while allowing for graceful accuracy-latency tradeoff for the deployment to devices of different capabilities.

{{</citation>}}


### (3/117) Optimal Economic Gas Turbine Dispatch with Deep Reinforcement Learning (Manuel Sage et al., 2023)

{{<citation>}}

Manuel Sage, Martin Staniszewski, Yaoyao Fiona Zhao. (2023)  
**Optimal Economic Gas Turbine Dispatch with Deep Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-SY, cs.LG, eess-SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.14924v1)  

---


**ABSTRACT**  
Dispatching strategies for gas turbines (GTs) are changing in modern electricity grids. A growing incorporation of intermittent renewable energy requires GTs to operate more but shorter cycles and more frequently on partial loads. Deep reinforcement learning (DRL) has recently emerged as a tool that can cope with this development and dispatch GTs economically. The key advantages of DRL are a model-free optimization and the ability to handle uncertainties, such as those introduced by varying loads or renewable energy production. In this study, three popular DRL algorithms are implemented for an economic GT dispatch problem on a case study in Alberta, Canada. We highlight the benefits of DRL by incorporating an existing thermodynamic software provided by Siemens Energy into the environment model and by simulating uncertainty via varying electricity prices, loads, and ambient conditions. Among the tested algorithms and baseline methods, Deep Q-Networks (DQN) obtained the highest rewards while Proximal Policy Optimization (PPO) was the most sample efficient. We further propose and implement a method to assign GT operation and maintenance cost dynamically based on operating hours and cycles. Compared to existing methods, our approach better approximates the true cost of modern GT dispatch and hence leads to more realistic policies.

{{</citation>}}


### (4/117) Statistically Efficient Variance Reduction with Double Policy Estimation for Off-Policy Evaluation in Sequence-Modeled Reinforcement Learning (Hanhan Zhou et al., 2023)

{{<citation>}}

Hanhan Zhou, Tian Lan, Vaneet Aggarwal. (2023)  
**Statistically Efficient Variance Reduction with Double Policy Estimation for Off-Policy Evaluation in Sequence-Modeled Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-DC, cs-LG, cs.LG  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.14897v1)  

---


**ABSTRACT**  
Offline reinforcement learning aims to utilize datasets of previously gathered environment-action interaction records to learn a policy without access to the real environment. Recent work has shown that offline reinforcement learning can be formulated as a sequence modeling problem and solved via supervised learning with approaches such as decision transformer. While these sequence-based methods achieve competitive results over return-to-go methods, especially on tasks that require longer episodes or with scarce rewards, importance sampling is not considered to correct the policy bias when dealing with off-policy data, mainly due to the absence of behavior policy and the use of deterministic evaluation policies. To this end, we propose DPE: an RL algorithm that blends offline sequence modeling and offline reinforcement learning with Double Policy Estimation (DPE) in a unified framework with statistically proven properties on variance reduction. We validate our method in multiple tasks of OpenAI Gym with D4RL benchmarks. Our method brings a performance improvements on selected methods which outperforms SOTA baselines in several tasks, demonstrating the advantages of enabling double policy estimation for sequence-modeled reinforcement learning.

{{</citation>}}


### (5/117) Continual Learning with Dynamic Sparse Training: Exploring Algorithms for Effective Model Updates (Murat Onur Yildirim et al., 2023)

{{<citation>}}

Murat Onur Yildirim, Elif Ceren Gok Yildirim, Ghada Sokar, Decebal Constantin Mocanu, Joaquin Vanschoren. (2023)  
**Continual Learning with Dynamic Sparse Training: Exploring Algorithms for Effective Model Updates**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2308.14831v1)  

---


**ABSTRACT**  
Continual learning (CL) refers to the ability of an intelligent system to sequentially acquire and retain knowledge from a stream of data with as little computational overhead as possible. To this end; regularization, replay, architecture, and parameter isolation approaches were introduced to the literature. Parameter isolation using a sparse network which enables to allocate distinct parts of the neural network to different tasks and also allows to share of parameters between tasks if they are similar. Dynamic Sparse Training (DST) is a prominent way to find these sparse networks and isolate them for each task. This paper is the first empirical study investigating the effect of different DST components under the CL paradigm to fill a critical research gap and shed light on the optimal configuration of DST for CL if it exists. Therefore, we perform a comprehensive study in which we investigate various DST components to find the best topology per task on well-known CIFAR100 and miniImageNet benchmarks in a task-incremental CL setup since our primary focus is to evaluate the performance of various DST criteria, rather than the process of mask selection. We found that, at a low sparsity level, Erdos-Renyi Kernel (ERK) initialization utilizes the backbone more efficiently and allows to effectively learn increments of tasks. At a high sparsity level, however, uniform initialization demonstrates more reliable and robust performance. In terms of growth strategy; performance is dependent on the defined initialization strategy, and the extent of sparsity. Finally, adaptivity within DST components is a promising way for better continual learners.

{{</citation>}}


### (6/117) RESTORE: Graph Embedding Assessment Through Reconstruction (Hong Yung Yip et al., 2023)

{{<citation>}}

Hong Yung Yip, Chidaksh Ravuru, Neelabha Banerjee, Shashwat Jha, Amit Sheth, Aman Chadha, Amitava Das. (2023)  
**RESTORE: Graph Embedding Assessment Through Reconstruction**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Embedding, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2308.14659v1)  

---


**ABSTRACT**  
Following the success of Word2Vec embeddings, graph embeddings (GEs) have gained substantial traction. GEs are commonly generated and evaluated extrinsically on downstream applications, but intrinsic evaluations of the original graph properties in terms of topological structure and semantic information have been lacking. Understanding these will help identify the deficiency of the various families of GE methods when vectorizing graphs in terms of preserving the relevant knowledge or learning incorrect knowledge. To address this, we propose RESTORE, a framework for intrinsic GEs assessment through graph reconstruction. We show that reconstructing the original graph from the underlying GEs yields insights into the relative amount of information preserved in a given vector form. We first introduce the graph reconstruction task. We generate GEs from three GE families based on factorization methods, random walks, and deep learning (with representative algorithms from each family) on the CommonSense Knowledge Graph (CSKG). We analyze their effectiveness in preserving the (a) topological structure of node-level graph reconstruction with an increasing number of hops and (b) semantic information on various word semantic and analogy tests. Our evaluations show deep learning-based GE algorithm (SDNE) is overall better at preserving (a) with a mean average precision (mAP) of 0.54 and 0.35 for 2 and 3-hop reconstruction respectively, while the factorization-based algorithm (HOPE) is better at encapsulating (b) with an average Euclidean distance of 0.14, 0.17, and 0.11 for 1, 2, and 3-hop reconstruction respectively. The modest performance of these GEs leaves room for further research avenues on better graph representation learning.

{{</citation>}}


### (7/117) Edge Generation Scheduling for DAG Tasks using Deep Reinforcement Learning (Binqi Sun et al., 2023)

{{<citation>}}

Binqi Sun, Mirco Theile, Ziyuan Qin, Daniele Bernardini, Debayan Roy, Andrea Bastoni, Marco Caccamo. (2023)  
**Edge Generation Scheduling for DAG Tasks using Deep Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-DC, cs-DM, cs-LG, cs.LG, math-OC  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.14647v1)  

---


**ABSTRACT**  
Directed acyclic graph (DAG) tasks are currently adopted in the real-time domain to model complex applications from the automotive, avionics, and industrial domain that implement their functionalities through chains of intercommunicating tasks. This paper studies the problem of scheduling real-time DAG tasks by presenting a novel schedulability test based on the concept of trivial schedulability. Using this schedulability test, we propose a new DAG scheduling framework (edge generation scheduling -- EGS) that attempts to minimize the DAG width by iteratively generating edges while guaranteeing the deadline constraint. We study how to efficiently solve the problem of generating edges by developing a deep reinforcement learning algorithm combined with a graph representation neural network to learn an efficient edge generation policy for EGS. We evaluate the effectiveness of the proposed algorithm by comparing it with state-of-the-art DAG scheduling heuristics and an optimal mixed-integer linear programming baseline. Experimental results show that the proposed algorithm outperforms the state-of-the-art by requiring fewer processors to schedule the same DAG tasks.

{{</citation>}}


### (8/117) AI in the Gray: Exploring Moderation Policies in Dialogic Large Language Models vs. Human Answers in Controversial Topics (Vahid Ghafouri et al., 2023)

{{<citation>}}

Vahid Ghafouri, Vibhor Agarwal, Yong Zhang, Nishanth Sastry, Jose Such, Guillermo Suarez-Tangil. (2023)  
**AI in the Gray: Exploring Moderation Policies in Dialogic Large Language Models vs. Human Answers in Controversial Topics**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-CY, cs-LG, cs-SI, cs.LG  
Keywords: AI, ChatBot, ChatGPT, Dialog, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2308.14608v1)  

---


**ABSTRACT**  
The introduction of ChatGPT and the subsequent improvement of Large Language Models (LLMs) have prompted more and more individuals to turn to the use of ChatBots, both for information and assistance with decision-making. However, the information the user is after is often not formulated by these ChatBots objectively enough to be provided with a definite, globally accepted answer.   Controversial topics, such as "religion", "gender identity", "freedom of speech", and "equality", among others, can be a source of conflict as partisan or biased answers can reinforce preconceived notions or promote disinformation. By exposing ChatGPT to such debatable questions, we aim to understand its level of awareness and if existing models are subject to socio-political and/or economic biases. We also aim to explore how AI-generated answers compare to human ones. For exploring this, we use a dataset of a social media platform created for the purpose of debating human-generated claims on polemic subjects among users, dubbed Kialo.   Our results show that while previous versions of ChatGPT have had important issues with controversial topics, more recent versions of ChatGPT (gpt-3.5-turbo) are no longer manifesting significant explicit biases in several knowledge areas. In particular, it is well-moderated regarding economic aspects. However, it still maintains degrees of implicit libertarian leaning toward right-winged ideals which suggest the need for increased moderation from the socio-political point of view. In terms of domain knowledge on controversial topics, with the exception of the "Philosophical" category, ChatGPT is performing well in keeping up with the collective human level of knowledge. Finally, we see that sources of Bing AI have slightly more tendency to the center when compared to human answers. All the analyses we make are generalizable to other types of biases and domains.

{{</citation>}}


### (9/117) Prediction of Tourism Flow with Sparse Geolocation Data (Julian Lemmel et al., 2023)

{{<citation>}}

Julian Lemmel, Zahra Babaiee, Marvin Kleinlehner, Ivan Majic, Philipp Neubauer, Johannes Scholz, Radu Grosu, Sophie A. Neubauer. (2023)  
**Prediction of Tourism Flow with Sparse Geolocation Data**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-AP  
Keywords: GNN, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.14516v1)  

---


**ABSTRACT**  
Modern tourism in the 21st century is facing numerous challenges. Among these the rapidly growing number of tourists visiting space-limited regions like historical cities, museums and bottlenecks such as bridges is one of the biggest. In this context, a proper and accurate prediction of tourism volume and tourism flow within a certain area is important and critical for visitor management tasks such as sustainable treatment of the environment and prevention of overcrowding. Static flow control methods like conventional low-level controllers or limiting access to overcrowded venues could not solve the problem yet. In this paper, we empirically evaluate the performance of state-of-the-art deep-learning methods such as RNNs, GNNs, and Transformers as well as the classic statistical ARIMA method. Granular limited data supplied by a tourism region is extended by exogenous data such as geolocation trajectories of individual tourists, weather and holidays. In the field of visitor flow prediction with sparse data, we are thereby capable of increasing the accuracy of our predictions, incorporating modern input feature handling as well as mapping geolocation data on top of discrete POI data.

{{</citation>}}


### (10/117) Group Regression for Query Based Object Detection and Tracking (Felicia Ruppel et al., 2023)

{{<citation>}}

Felicia Ruppel, Florian Faion, Claudius Gläser, Klaus Dietmayer. (2023)  
**Group Regression for Query Based Object Detection and Tracking**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2308.14481v1)  

---


**ABSTRACT**  
Group regression is commonly used in 3D object detection to predict box parameters of similar classes in a joint head, aiming to benefit from similarities while separating highly dissimilar classes. For query-based perception methods, this has, so far, not been feasible. We close this gap and present a method to incorporate multi-class group regression, especially designed for the 3D domain in the context of autonomous driving, into existing attention and query-based perception approaches. We enhance a transformer based joint object detection and tracking model with this approach, and thoroughly evaluate its behavior and performance. For group regression, the classes of the nuScenes dataset are divided into six groups of similar shape and prevalence, each being regressed by a dedicated head. We show that the proposed method is applicable to many existing transformer based perception approaches and can bring potential benefits. The behavior of query group regression is thoroughly analyzed in comparison to a unified regression head, e.g. in terms of class-switching behavior and distribution of the output parameters. The proposed method offers many possibilities for further research, such as in the direction of deep multi-hypotheses tracking.

{{</citation>}}


### (11/117) Self-Supervision for Tackling Unsupervised Anomaly Detection: Pitfalls and Opportunities (Leman Akoglu et al., 2023)

{{<citation>}}

Leman Akoglu, Jaemin Yoo. (2023)  
**Self-Supervision for Tackling Unsupervised Anomaly Detection: Pitfalls and Opportunities**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2308.14380v1)  

---


**ABSTRACT**  
Self-supervised learning (SSL) is a growing torrent that has recently transformed machine learning and its many real world applications, by learning on massive amounts of unlabeled data via self-generated supervisory signals. Unsupervised anomaly detection (AD) has also capitalized on SSL, by self-generating pseudo-anomalies through various data augmentation functions or external data exposure. In this vision paper, we first underline the importance of the choice of SSL strategies on AD performance, by presenting evidences and studies from the AD literature. Equipped with the understanding that SSL incurs various hyperparameters (HPs) to carefully tune, we present recent developments on unsupervised model selection and augmentation tuning for SSL-based AD. We then highlight emerging challenges and future opportunities; on designing new pretext tasks and augmentation functions for different data modalities, creating novel model selection solutions for systematically tuning the SSL HPs, as well as on capitalizing on the potential of pretrained foundation models on AD through effective density estimation.

{{</citation>}}


### (12/117) Meta Attentive Graph Convolutional Recurrent Network for Traffic Forecasting (Adnan Zeb et al., 2023)

{{<citation>}}

Adnan Zeb, Yongchao Ye, Shiyao Zhang, James J. Q. Yu. (2023)  
**Meta Attentive Graph Convolutional Recurrent Network for Traffic Forecasting**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2308.14377v1)  

---


**ABSTRACT**  
Traffic forecasting is a fundamental problem in intelligent transportation systems. Existing traffic predictors are limited by their expressive power to model the complex spatial-temporal dependencies in traffic data, mainly due to the following limitations. Firstly, most approaches are primarily designed to model the local shared patterns, which makes them insufficient to capture the specific patterns associated with each node globally. Hence, they fail to learn each node's unique properties and diversified patterns. Secondly, most existing approaches struggle to accurately model both short- and long-term dependencies simultaneously. In this paper, we propose a novel traffic predictor, named Meta Attentive Graph Convolutional Recurrent Network (MAGCRN). MAGCRN utilizes a Graph Convolutional Recurrent Network (GCRN) as a core module to model local dependencies and improves its operation with two novel modules: 1) a Node-Specific Meta Pattern Learning (NMPL) module to capture node-specific patterns globally and 2) a Node Attention Weight Generation Module (NAWG) module to capture short- and long-term dependencies by connecting the node-specific features with the ones learned initially at each time step during GCRN operation. Experiments on six real-world traffic datasets demonstrate that NMPL and NAWG together enable MAGCRN to outperform state-of-the-art baselines on both short- and long-term predictions.

{{</citation>}}


### (13/117) Are Existing Out-Of-Distribution Techniques Suitable for Network Intrusion Detection? (Andrea Corsini et al., 2023)

{{<citation>}}

Andrea Corsini, Shanchieh Jay Yang. (2023)  
**Are Existing Out-Of-Distribution Techniques Suitable for Network Intrusion Detection?**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CR, cs-LG, cs.LG  
Keywords: Intrusion Detection  
[Paper Link](http://arxiv.org/abs/2308.14376v1)  

---


**ABSTRACT**  
Machine learning (ML) has become increasingly popular in network intrusion detection. However, ML-based solutions always respond regardless of whether the input data reflects known patterns, a common issue across safety-critical applications. While several proposals exist for detecting Out-Of-Distribution (OOD) in other fields, it remains unclear whether these approaches can effectively identify new forms of intrusions for network security. New attacks, not necessarily affecting overall distributions, are not guaranteed to be clearly OOD as instead, images depicting new classes are in computer vision. In this work, we investigate whether existing OOD detectors from other fields allow the identification of unknown malicious traffic. We also explore whether more discriminative and semantically richer embedding spaces within models, such as those created with contrastive learning and multi-class tasks, benefit detection. Our investigation covers a set of six OOD techniques that employ different detection strategies. These techniques are applied to models trained in various ways and subsequently exposed to unknown malicious traffic from the same and different datasets (network environments). Our findings suggest that existing detectors can identify a consistent portion of new malicious traffic, and that improved embedding spaces enhance detection. We also demonstrate that simple combinations of certain detectors can identify almost 100% of malicious traffic in our tested scenarios.

{{</citation>}}


### (14/117) Target-independent XLA optimization using Reinforcement Learning (Milan Ganai et al., 2023)

{{<citation>}}

Milan Ganai, Haichen Li, Theodore Enns, Yida Wang, Randy Huang. (2023)  
**Target-independent XLA optimization using Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: BERT, GPT, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.14364v1)  

---


**ABSTRACT**  
An important challenge in Machine Learning compilers like XLA is multi-pass optimization and analysis. There has been recent interest chiefly in XLA target-dependent optimization on the graph-level, subgraph-level, and kernel-level phases. We specifically focus on target-independent optimization XLA HLO pass ordering: our approach aims at finding the optimal sequence of compiler optimization passes, which is decoupled from target-dependent optimization. However, there is little domain specific study in pass ordering for XLA HLO. To this end, we propose introducing deep Reinforcement Learning (RL) based search for optimal XLA HLO pass ordering. We also propose enhancements to the deep RL algorithms to further improve optimal search performance and open the research direction for domain-specific guidance for RL. We create an XLA Gym experimentation framework as a tool to enable RL algorithms to interact with the compiler for passing optimizations and thereby train agents. Overall, in our experimentation we observe an average of $13.3\%$ improvement in operation count reduction on a benchmark of GPT-2 training graphs and $10.4\%$ improvement on a diverse benchmark including GPT-2, BERT, and ResNet graphs using the proposed approach over the compiler's default phase ordering.

{{</citation>}}


### (15/117) Can Transformer and GNN Help Each Other? (Peiyan Zhang et al., 2023)

{{<citation>}}

Peiyan Zhang, Yuchen Yan, Chaozhuo Li, Senzhang Wang, Xing Xie, Sunghun Kim. (2023)  
**Can Transformer and GNN Help Each Other?**  

---
Primary Category: cs.LG  
Categories: H-3-3, cs-IR, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.14355v1)  

---


**ABSTRACT**  
Although Transformer has achieved great success in natural language process and computer vision, it has difficulty generalizing to medium and large-scale graph data for two important reasons: (i) High complexity. (ii) Failing to capture the complex and entangled structure information. In graph representation learning, Graph Neural Networks(GNNs) can fuse the graph structure and node attributes but have limited receptive fields. Therefore, we question whether can we combine Transformers and GNNs to help each other. In this paper, we propose a new model named TransGNN where the Transformer layer and GNN layer are used alternately to improve each other. Specifically, to expand the receptive field and disentangle the information aggregation from edges, we propose using Transformer to aggregate more relevant nodes' information to improve the message passing of GNNs. Besides, to capture the graph structure information, we utilize positional encoding and make use of the GNN layer to fuse the structure into node attributes, which improves the Transformer in graph data. We also propose to sample the most relevant nodes for Transformer and two efficient samples update strategies to lower the complexity. At last, we theoretically prove that TransGNN is more expressive than GNNs only with extra linear complexity. The experiments on eight datasets corroborate the effectiveness of TransGNN on node and graph classification tasks.

{{</citation>}}


### (16/117) EdgeMoE: Fast On-Device Inference of MoE-based Large Language Models (Rongjie Yi et al., 2023)

{{<citation>}}

Rongjie Yi, Liwei Guo, Shiyun Wei, Ao Zhou, Shangguang Wang, Mengwei Xu. (2023)  
**EdgeMoE: Fast On-Device Inference of MoE-based Large Language Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2308.14352v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) such as GPTs and LLaMa have ushered in a revolution in machine intelligence, owing to their exceptional capabilities in a wide range of machine learning tasks. However, the transition of LLMs from data centers to edge devices presents a set of challenges and opportunities. While this shift can enhance privacy and availability, it is hampered by the enormous parameter sizes of these models, leading to impractical runtime costs. In light of these considerations, we introduce EdgeMoE, the first on-device inference engine tailored for mixture-of-expert (MoE) LLMs, a popular variant of sparse LLMs that exhibit nearly constant computational complexity as their parameter size scales. EdgeMoE achieves both memory and computational efficiency by strategically partitioning the model across the storage hierarchy. Specifically, non-expert weights are stored in the device's memory, while expert weights are kept in external storage and are fetched into memory only when they are activated. This design is underpinned by a crucial insight that expert weights, though voluminous, are infrequently accessed due to sparse activation patterns. To further mitigate the overhead associated with expert I/O swapping, EdgeMoE incorporates two innovative techniques: (1) Expert-wise bitwidth adaptation: This method reduces the size of expert weights with an acceptable level of accuracy loss. (2) Expert management: It predicts the experts that will be activated in advance and preloads them into the compute-I/O pipeline, thus further optimizing the process. In empirical evaluations conducted on well-established MoE LLMs and various edge devices, EdgeMoE demonstrates substantial memory savings and performance improvements when compared to competitive baseline solutions.

{{</citation>}}


### (17/117) HRGCN: Heterogeneous Graph-level Anomaly Detection with Hierarchical Relation-augmented Graph Neural Networks (Jiaxi Li et al., 2023)

{{<citation>}}

Jiaxi Li, Guansong Pang, Ling Chen, Mohammad-Reza Namazi-Rad. (2023)  
**HRGCN: Heterogeneous Graph-level Anomaly Detection with Hierarchical Relation-augmented Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Anomaly Detection, GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2308.14340v1)  

---


**ABSTRACT**  
This work considers the problem of heterogeneous graph-level anomaly detection. Heterogeneous graphs are commonly used to represent behaviours between different types of entities in complex industrial systems for capturing as much information about the system operations as possible. Detecting anomalous heterogeneous graphs from a large set of system behaviour graphs is crucial for many real-world applications like online web/mobile service and cloud access control. To address the problem, we propose HRGCN, an unsupervised deep heterogeneous graph neural network, to model complex heterogeneous relations between different entities in the system for effectively identifying these anomalous behaviour graphs. HRGCN trains a hierarchical relation-augmented Heterogeneous Graph Neural Network (HetGNN), which learns better graph representations by modelling the interactions among all the system entities and considering both source-to-destination entity (node) types and their relation (edge) types. Extensive evaluation on two real-world application datasets shows that HRGCN outperforms state-of-the-art competing anomaly detection approaches. We further present a real-world industrial case study to justify the effectiveness of HRGCN in detecting anomalous (e.g., congested) network devices in a mobile communication service. HRGCN is available at https://github.com/jiaxililearn/HRGCN.

{{</citation>}}


### (18/117) DiffSmooth: Certifiably Robust Learning via Diffusion Models and Local Smoothing (Jiawei Zhang et al., 2023)

{{<citation>}}

Jiawei Zhang, Zhongzhu Chen, Huan Zhang, Chaowei Xiao, Bo Li. (2023)  
**DiffSmooth: Certifiably Robust Learning via Diffusion Models and Local Smoothing**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2308.14333v1)  

---


**ABSTRACT**  
Diffusion models have been leveraged to perform adversarial purification and thus provide both empirical and certified robustness for a standard model. On the other hand, different robustly trained smoothed models have been studied to improve the certified robustness. Thus, it raises a natural question: Can diffusion model be used to achieve improved certified robustness on those robustly trained smoothed models? In this work, we first theoretically show that recovered instances by diffusion models are in the bounded neighborhood of the original instance with high probability; and the "one-shot" denoising diffusion probabilistic models (DDPM) can approximate the mean of the generated distribution of a continuous-time diffusion model, which approximates the original instance under mild conditions. Inspired by our analysis, we propose a certifiably robust pipeline DiffSmooth, which first performs adversarial purification via diffusion models and then maps the purified instances to a common region via a simple yet effective local smoothing strategy. We conduct extensive experiments on different datasets and show that DiffSmooth achieves SOTA-certified robustness compared with eight baselines. For instance, DiffSmooth improves the SOTA-certified accuracy from $36.0\%$ to $53.0\%$ under $\ell_2$ radius $1.5$ on ImageNet. The code is available at [https://github.com/javyduck/DiffSmooth].

{{</citation>}}


### (19/117) Reinforcement Learning for Generative AI: A Survey (Yuanjiang Cao et al., 2023)

{{<citation>}}

Yuanjiang Cao, Quan Z. Sheng, Julian McAuley, Lina Yao. (2023)  
**Reinforcement Learning for Generative AI: A Survey**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI, Generative AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.14328v2)  

---


**ABSTRACT**  
Deep Generative AI has been a long-standing essential topic in the machine learning community, which can impact a number of application areas like text generation and computer vision. The major paradigm to train a generative model is maximum likelihood estimation, which pushes the learner to capture and approximate the target data distribution by decreasing the divergence between the model distribution and the target distribution. This formulation successfully establishes the objective of generative tasks, while it is incapable of satisfying all the requirements that a user might expect from a generative model. Reinforcement learning, serving as a competitive option to inject new training signals by creating new objectives that exploit novel signals, has demonstrated its power and flexibility to incorporate human inductive bias from multiple angles, such as adversarial learning, hand-designed rules and learned reward model to build a performant model. Thereby, reinforcement learning has become a trending research field and has stretched the limits of generative AI in both model design and application. It is reasonable to summarize and conclude advances in recent years with a comprehensive review. Although there are surveys in different application areas recently, this survey aims to shed light on a high-level review that spans a range of application areas. We provide a rigorous taxonomy in this area and make sufficient coverage on various models and applications. Notably, we also surveyed the fast-developing large language model area. We conclude this survey by showing the potential directions that might tackle the limit of current models and expand the frontiers for generative AI.

{{</citation>}}


### (20/117) Solving Attention Kernel Regression Problem via Pre-conditioner (Zhao Song et al., 2023)

{{<citation>}}

Zhao Song, Junze Yin, Lichen Zhang. (2023)  
**Solving Attention Kernel Regression Problem via Pre-conditioner**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2308.14304v1)  

---


**ABSTRACT**  
Large language models have shown impressive performance in many tasks. One of the major features from the computation perspective is computing the attention matrix. Previous works [Zandieh, Han, Daliri, and Karba 2023, Alman and Song 2023] have formally studied the possibility and impossibility of approximating the attention matrix. In this work, we define and study a new problem which is called the attention kernel regression problem. We show how to solve the attention kernel regression in the input sparsity time of the data matrix.

{{</citation>}}


### (21/117) Unleash Model Potential: Bootstrapped Meta Self-supervised Learning (Jingyao Wang et al., 2023)

{{<citation>}}

Jingyao Wang, Zeen Song, Wenwen Qiang, Changwen Zheng. (2023)  
**Unleash Model Potential: Bootstrapped Meta Self-supervised Learning**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2308.14267v1)  

---


**ABSTRACT**  
The long-term goal of machine learning is to learn general visual representations from a small amount of data without supervision, mimicking three advantages of human cognition: i) no need for labels, ii) robustness to data scarcity, and iii) learning from experience. Self-supervised learning and meta-learning are two promising techniques to achieve this goal, but they both only partially capture the advantages and fail to address all the problems. Self-supervised learning struggles to overcome the drawbacks of data scarcity, while ignoring prior knowledge that can facilitate learning and generalization. Meta-learning relies on supervised information and suffers from a bottleneck of insufficient learning. To address these issues, we propose a novel Bootstrapped Meta Self-Supervised Learning (BMSSL) framework that aims to simulate the human learning process. We first analyze the close relationship between meta-learning and self-supervised learning. Based on this insight, we reconstruct tasks to leverage the strengths of both paradigms, achieving advantages i and ii. Moreover, we employ a bi-level optimization framework that alternates between solving specific tasks with a learned ability (first level) and improving this ability (second level), attaining advantage iii. To fully harness its power, we introduce a bootstrapped target based on meta-gradient to make the model its own teacher. We validate the effectiveness of our approach with comprehensive theoretical and empirical study.

{{</citation>}}


### (22/117) A Comparison of Personalized and Generalized Approaches to Emotion Recognition Using Consumer Wearable Devices: Machine Learning Study (Joe Li et al., 2023)

{{<citation>}}

Joe Li, Peter Washington. (2023)  
**A Comparison of Personalized and Generalized Approaches to Emotion Recognition Using Consumer Wearable Devices: Machine Learning Study**  

---
Primary Category: cs.LG  
Categories: cs-CY, cs-HC, cs-LG, cs.LG, eess-SP  
Keywords: Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2308.14245v1)  

---


**ABSTRACT**  
Background: Studies have shown the potential adverse health effects, ranging from headaches to cardiovascular disease, associated with long-term negative emotions and chronic stress. Since many indicators of stress are imperceptible to observers, the early detection and intervention of stress remains a pressing medical need. Physiological signals offer a non-invasive method of monitoring emotions and are easily collected by smartwatches. Existing research primarily focuses on developing generalized machine learning-based models for emotion classification. Objective: We aim to study the differences between personalized and generalized machine learning models for three-class emotion classification (neutral, stress, and amusement) using wearable biosignal data. Methods: We developed a convolutional encoder for the three-class emotion classification problem using data from WESAD, a multimodal dataset with physiological signals for 15 subjects. We compared the results between a subject-exclusive generalized, subject-inclusive generalized, and personalized model. Results: For the three-class classification problem, our personalized model achieved an average accuracy of 95.06% and F1-score of 91.71, our subject-inclusive generalized model achieved an average accuracy of 66.95% and F1-score of 42.50, and our subject-exclusive generalized model achieved an average accuracy of 67.65% and F1-score of 43.05. Conclusions: Our results emphasize the need for increased research in personalized emotion recognition models given that they outperform generalized models in certain contexts. We also demonstrate that personalized machine learning models for emotion classification are viable and can achieve high performance.

{{</citation>}}


## cs.AI (17)



### (23/117) Transfusor: Transformer Diffusor for Controllable Human-like Generation of Vehicle Lane Changing Trajectories (Jiqian Dong et al., 2023)

{{<citation>}}

Jiqian Dong, Sikai Chen, Samuel Labi. (2023)  
**Transfusor: Transformer Diffusor for Controllable Human-like Generation of Vehicle Lane Changing Trajectories**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-RO, cs.AI  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.14943v1)  

---


**ABSTRACT**  
With ongoing development of autonomous driving systems and increasing desire for deployment, researchers continue to seek reliable approaches for ADS systems. The virtual simulation test (VST) has become a prominent approach for testing autonomous driving systems (ADS) and advanced driver assistance systems (ADAS) due to its advantages of fast execution, low cost, and high repeatability. However, the success of these simulation-based experiments heavily relies on the realism of the testing scenarios. It is needed to create more flexible and high-fidelity testing scenarios in VST in order to increase the safety and reliabilityof ADS and ADAS.To address this challenge, this paper introduces the "Transfusor" model, which leverages the transformer and diffusor models (two cutting-edge deep learning generative technologies). The primary objective of the Transfusor model is to generate highly realistic and controllable human-like lane-changing trajectories in highway scenarios. Extensive experiments were carried out, and the results demonstrate that the proposed model effectively learns the spatiotemporal characteristics of humans' lane-changing behaviors and successfully generates trajectories that closely mimic real-world human driving. As such, the proposed model can play a critical role of creating more flexible and high-fidelity testing scenarios in the VST, ultimately leading to safer and more reliable ADS and ADAS.

{{</citation>}}


### (24/117) Identifying and Mitigating the Security Risks of Generative AI (Clark Barrett et al., 2023)

{{<citation>}}

Clark Barrett, Brad Boyd, Ellie Burzstein, Nicholas Carlini, Brad Chen, Jihye Choi, Amrita Roy Chowdhury, Mihai Christodorescu, Anupam Datta, Soheil Feizi, Kathleen Fisher, Tatsunori Hashimoto, Dan Hendrycks, Somesh Jha, Daniel Kang, Florian Kerschbaum, Eric Mitchell, John Mitchell, Zulfikar Ramzan, Khawaja Shams, Dawn Song, Ankur Taly, Diyi Yang. (2023)  
**Identifying and Mitigating the Security Risks of Generative AI**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, Generative AI, Google, Security  
[Paper Link](http://arxiv.org/abs/2308.14840v1)  

---


**ABSTRACT**  
Every major technical invention resurfaces the dual-use dilemma -- the new technology has the potential to be used for good as well as for harm. Generative AI (GenAI) techniques, such as large language models (LLMs) and diffusion models, have shown remarkable capabilities (e.g., in-context learning, code-completion, and text-to-image generation and editing). However, GenAI can be used just as well by attackers to generate new attacks and increase the velocity and efficacy of existing attacks.   This paper reports the findings of a workshop held at Google (co-organized by Stanford University and the University of Wisconsin-Madison) on the dual-use dilemma posed by GenAI. This paper is not meant to be comprehensive, but is rather an attempt to synthesize some of the interesting findings from the workshop. We discuss short-term and long-term goals for the community on this topic. We hope this paper provides both a launching point for a discussion on this important topic as well as interesting problems that the research community can work to address.

{{</citation>}}


### (25/117) Distributionally Robust Statistical Verification with Imprecise Neural Networks (Souradeep Dutta et al., 2023)

{{<citation>}}

Souradeep Dutta, Michele Caprio, Vivian Lin, Matthew Cleaveland, Kuk Jin Jang, Ivan Ruchkin, Oleg Sokolsky, Insup Lee. (2023)  
**Distributionally Robust Statistical Verification with Imprecise Neural Networks**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs-RO, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.14815v2)  

---


**ABSTRACT**  
A particularly challenging problem in AI safety is providing guarantees on the behavior of high-dimensional autonomous systems. Verification approaches centered around reachability analysis fail to scale, and purely statistical approaches are constrained by the distributional assumptions about the sampling process. Instead, we pose a distributionally robust version of the statistical verification problem for black-box systems, where our performance guarantees hold over a large family of distributions. This paper proposes a novel approach based on a combination of active learning, uncertainty quantification, and neural network verification. A central piece of our approach is an ensemble technique called Imprecise Neural Networks, which provides the uncertainty to guide active learning. The active learning uses an exhaustive neural-network verification tool Sherlock to collect samples. An evaluation on multiple physical simulators in the openAI gym Mujoco environments with reinforcement-learned controllers demonstrates that our approach can provide useful and scalable guarantees for high-dimensional systems.

{{</citation>}}


### (26/117) Bayesian artificial brain with ChatGPT (Renato A. Krohling, 2023)

{{<citation>}}

Renato A. Krohling. (2023)  
**Bayesian artificial brain with ChatGPT**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: ChatGPT, GPT, Transformer  
[Paper Link](http://arxiv.org/abs/2308.14732v1)  

---


**ABSTRACT**  
This paper aims to investigate the mathematical problem-solving capabilities of Chat Generative Pre-Trained Transformer (ChatGPT) in case of Bayesian reasoning. The study draws inspiration from Zhu & Gigerenzer's research in 2006, which posed the question: Can children reason the Bayesian way? In the pursuit of answering this question, a set of 10 Bayesian reasoning problems were presented. The results of their work revealed that children's ability to reason effectively using Bayesian principles is contingent upon a well-structured information representation. In this paper, we present the same set of 10 Bayesian reasoning problems to ChatGPT. Remarkably, the results demonstrate that ChatGPT provides the right solutions to all problems.

{{</citation>}}


### (27/117) Hierarchical Time Series Forecasting with Bayesian Modeling (Gal Elgavish, 2023)

{{<citation>}}

Gal Elgavish. (2023)  
**Hierarchical Time Series Forecasting with Bayesian Modeling**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2308.14719v1)  

---


**ABSTRACT**  
We encounter time series data in many domains such as finance, physics, business, and weather. One of the main tasks of time series analysis, one that helps to take informed decisions under uncertainty, is forecasting. Time series are often hierarchically structured, e.g., a company sales might be broken down into different regions, and each region into different stores. In some cases the number of series in the hierarchy is too big to fit in a single model to produce forecasts in relevant time, and a decentralized approach is beneficial.   One way to do this is to train independent forecasting models for each series and for some summary statistics series implied by the hierarchy (e.g. the sum of all series) and to pass those models to a reconciliation algorithm to improve those forecasts by sharing information between the series.   In this work we focus on the reconciliation step, and propose a method to do so from a Bayesian perspective - Bayesian forecast reconciliation. We also define the common case of linear Gaussian reconciliation, where the forecasts are Gaussian and the hierarchy has linear structure, and show that we can compute reconciliation in closed form. We evaluate these methods on synthetic and real data sets, and compare them to other work in this field.

{{</citation>}}


### (28/117) Learning Visual Tracking and Reaching with Deep Reinforcement Learning on a UR10e Robotic Arm (Colin Bellinger et al., 2023)

{{<citation>}}

Colin Bellinger, Laurence Lamarche-Cliche. (2023)  
**Learning Visual Tracking and Reaching with Deep Reinforcement Learning on a UR10e Robotic Arm**  

---
Primary Category: cs.AI  
Categories: 68T40, I-2; I-4; J-2, cs-AI, cs-RO, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.14652v1)  

---


**ABSTRACT**  
As technology progresses, industrial and scientific robots are increasingly being used in diverse settings. In many cases, however, programming the robot to perform such tasks is technically complex and costly. To maximize the utility of robots in industrial and scientific settings, they require the ability to quickly shift from one task to another. Reinforcement learning algorithms provide the potential to enable robots to learn optimal solutions to complete new tasks without directly reprogramming them. The current state-of-the-art in reinforcement learning, however, generally relies on fast simulations and parallelization to achieve optimal performance. These are often not possible in robotics applications. Thus, a significant amount of research is required to facilitate the efficient and safe, training and deployment of industrial and scientific reinforcement learning robots. This technical report outlines our initial research into the application of deep reinforcement learning on an industrial UR10e robot. The report describes the reinforcement learning environments created to facilitate policy learning with the UR10e, a robotic arm from Universal Robots, and presents our initial results in training deep Q-learning and proximal policy optimization agents on the developed reinforcement learning environments. Our results show that proximal policy optimization learns a better, more stable policy with less data than deep Q-learning. The corresponding code for this work is available at \url{https://github.com/cbellinger27/bendRL_reacher_tracker}

{{</citation>}}


### (29/117) Context-Aware Composition of Agent Policies by Markov Decision Process Entity Embeddings and Agent Ensembles (Nicole Merkle et al., 2023)

{{<citation>}}

Nicole Merkle, Ralf Mikut. (2023)  
**Context-Aware Composition of Agent Policies by Markov Decision Process Entity Embeddings and Agent Ensembles**  

---
Primary Category: cs.AI  
Categories: F-2-2; I-2-7, cs-AI, cs-LG, cs-PF, cs.AI  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2308.14521v2)  

---


**ABSTRACT**  
Computational agents support humans in many areas of life and are therefore found in heterogeneous contexts. This means they operate in rapidly changing environments and can be confronted with huge state and action spaces. In order to perform services and carry out activities in a goal-oriented manner, agents require prior knowledge and therefore have to develop and pursue context-dependent policies. However, prescribing policies in advance is limited and inflexible, especially in dynamically changing environments. Moreover, the context of an agent determines its choice of actions. Since the environments can be stochastic and complex in terms of the number of states and feasible actions, activities are usually modelled in a simplified way by Markov decision processes so that, e.g., agents with reinforcement learning are able to learn policies, that help to capture the context and act accordingly to optimally perform activities. However, training policies for all possible contexts using reinforcement learning is time-consuming. A requirement and challenge for agents is to learn strategies quickly and respond immediately in cross-context environments and applications, e.g., the Internet, service robotics, cyber-physical systems. In this work, we propose a novel simulation-based approach that enables a) the representation of heterogeneous contexts through knowledge graphs and entity embeddings and b) the context-aware composition of policies on demand by ensembles of agents running in parallel. The evaluation we conducted with the "Virtual Home" dataset indicates that agents with a need to switch seamlessly between different contexts, can request on-demand composed policies that lead to the successful completion of context-appropriate activities without having to learn these policies in lengthy training steps and episodes, in contrast to agents that use reinforcement learning.

{{</citation>}}


### (30/117) ASCAPE: An open AI ecosystem to support the quality of life of cancer patients (Konstantinos Lampropoulos et al., 2023)

{{<citation>}}

Konstantinos Lampropoulos, Thanos Kosmidis, Serge Autexier, Milos Savic, Manos Athanatos, Miltiadis Kokkonidis, Tzortzia Koutsouri, Anamaria Vizitiu, Antonios Valachis, Miriam Quintero Padron. (2023)  
**ASCAPE: An open AI ecosystem to support the quality of life of cancer patients**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.14390v1)  

---


**ABSTRACT**  
The latest cancer statistics indicate a decrease in cancer-related mortality. However, due to the growing and ageing population, the absolute number of people living with cancer is set to keep increasing. This paper presents ASCAPE, an open AI infrastructure that takes advantage of the recent advances in Artificial Intelligence (AI) and Machine Learning (ML) to support cancer patients quality of life (QoL). With ASCAPE health stakeholders (e.g. hospitals) can locally process their private medical data and then share the produced knowledge (ML models) through the open AI infrastructure.

{{</citation>}}


### (31/117) Rethinking Mobile AI Ecosystem in the LLM Era (Jinliang Yuan et al., 2023)

{{<citation>}}

Jinliang Yuan, Chen Yang, Dongqi Cai, Shihe Wang, Xin Yuan, Zeling Zhang, Xiang Li, Dingge Zhang, Hanzi Mei, Xianqing Jia, Shangguang Wang, Mengwei Xu. (2023)  
**Rethinking Mobile AI Ecosystem in the LLM Era**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, Computer Vision, Language Model, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2308.14363v1)  

---


**ABSTRACT**  
In today's landscape, smartphones have evolved into hubs for hosting a multitude of deep learning models aimed at local execution. A key realization driving this work is the notable fragmentation among these models, characterized by varied architectures, operators, and implementations. This fragmentation imposes a significant burden on the comprehensive optimization of hardware, system settings, and algorithms.   Buoyed by the recent strides in large foundation models, this work introduces a pioneering paradigm for mobile AI: a collaborative management approach between the mobile OS and hardware, overseeing a foundational model capable of serving a broad spectrum of mobile AI tasks, if not all. This foundational model resides within the NPU and remains impervious to app or OS revisions, akin to firmware. Concurrently, each app contributes a concise, offline fine-tuned "adapter" tailored to distinct downstream tasks. From this concept emerges a concrete instantiation known as \sys. It amalgamates a curated selection of publicly available Large Language Models (LLMs) and facilitates dynamic data flow. This concept's viability is substantiated through the creation of an exhaustive benchmark encompassing 38 mobile AI tasks spanning 50 datasets, including domains such as Computer Vision (CV), Natural Language Processing (NLP), audio, sensing, and multimodal inputs. Spanning this benchmark, \sys unveils its impressive performance. It attains accuracy parity in 85\% of tasks, demonstrates improved scalability in terms of storage and memory, and offers satisfactory inference speed on Commercial Off-The-Shelf (COTS) mobile devices fortified with NPU support. This stands in stark contrast to task-specific models tailored for individual applications.

{{</citation>}}


### (32/117) Effect of Attention and Self-Supervised Speech Embeddings on Non-Semantic Speech Tasks (Payal Mohapatra et al., 2023)

{{<citation>}}

Payal Mohapatra, Akash Pandey, Yueyuan Sui, Qi Zhu. (2023)  
**Effect of Attention and Self-Supervised Speech Embeddings on Non-Semantic Speech Tasks**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: Attention, BERT, Embedding, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2308.14359v2)  

---


**ABSTRACT**  
Human emotion understanding is pivotal in making conversational technology mainstream. We view speech emotion understanding as a perception task which is a more realistic setting. With varying contexts (languages, demographics, etc.) different share of people perceive the same speech segment as a non-unanimous emotion. As part of the ACM Multimedia 2023 Computational Paralinguistics ChallengE (ComParE) in the EMotion Share track, we leverage their rich dataset of multilingual speakers and multi-label regression target of 'emotion share' or perception of that emotion. We demonstrate that the training scheme of different foundation models dictates their effectiveness for tasks beyond speech recognition, especially for non-semantic speech tasks like emotion understanding. This is a very complex task due to multilingual speakers, variability in the target labels, and inherent imbalance in the regression dataset. Our results show that HuBERT-Large with a self-attention-based light-weight sequence model provides 4.6% improvement over the reported baseline.

{{</citation>}}


### (33/117) Cognitive Effects in Large Language Models (Jonathan Shaki et al., 2023)

{{<citation>}}

Jonathan Shaki, Sarit Kraus, Michael Wooldridge. (2023)  
**Cognitive Effects in Large Language Models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2308.14337v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) such as ChatGPT have received enormous attention over the past year and are now used by hundreds of millions of people every day. The rapid adoption of this technology naturally raises questions about the possible biases such models might exhibit. In this work, we tested one of these models (GPT-3) on a range of cognitive effects, which are systematic patterns that are usually found in human cognitive tasks. We found that LLMs are indeed prone to several human cognitive effects. Specifically, we show that the priming, distance, SNARC, and size congruity effects were presented with GPT-3, while the anchoring effect is absent. We describe our methodology, and specifically the way we converted real-world experiments to text-based experiments. Finally, we speculate on the possible reasons why GPT-3 exhibits these effects and discuss whether they are imitated or reinvented.

{{</citation>}}


### (34/117) Spread Control Method on Unknown Networks Based on Hierarchical Reinforcement Learning (Wenxiang Dong et al., 2023)

{{<citation>}}

Wenxiang Dong, H. Vicky Zhao. (2023)  
**Spread Control Method on Unknown Networks Based on Hierarchical Reinforcement Learning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-SI, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.14311v1)  

---


**ABSTRACT**  
The spread of infectious diseases, rumors, and harmful speech in networks can result in substantial losses, underscoring the significance of studying how to suppress such hazardous events. However, previous studies often assume full knowledge of the network structure, which is often not the case in real-world scenarios. In this paper, we address the challenge of controlling the propagation of hazardous events by removing nodes when the network structure is unknown. To tackle this problem, we propose a hierarchical reinforcement learning method that drastically reduces the action space, making the problem feasible to solve. Simulation experiments demonstrate the superiority of our method over the baseline methods. Remarkably, even though the baseline methods possess extensive knowledge of the network structure, while our method has no prior information about it, our approach still achieves better results.

{{</citation>}}


### (35/117) Artificial Intelligence in Career Counseling: A Test Case with ResumAI (Muhammad Rahman et al., 2023)

{{<citation>}}

Muhammad Rahman, Sachi Figliolini, Joyce Kim, Eivy Cedeno, Charles Kleier, Chirag Shah, Aman Chadha. (2023)  
**Artificial Intelligence in Career Counseling: A Test Case with ResumAI**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2308.14301v1)  

---


**ABSTRACT**  
The rise of artificial intelligence (AI) has led to various means of integration of AI aimed to provide efficiency in tasks, one of which is career counseling. A key part of getting a job is having a solid resume that passes through the first round of programs and recruiters. It is difficult to find good resources or schedule an appointment with a career counselor to help with editing a resume for a specific role. With the rise of ChatGPT, Bard, and several other AI chat programs it is possible to provide specific, automated feedback on various concerns to suggest places for improvement within the context of career counseling. This paper begins with a quick literature review on the ethical considerations and limitations of AI in career counseling. The authors also have created their own website service, called ResumAI, to test and review the functionality of an AI career counselor. The findings of this study will contribute to the understanding of chat AI ResumAI reviewer programs and sites. The implications of the findings for the field of career counseling, AI development, and ethical practice will be discussed.

{{</citation>}}


### (36/117) Traffic Light Control with Reinforcement Learning (Taoyu Pan, 2023)

{{<citation>}}

Taoyu Pan. (2023)  
**Traffic Light Control with Reinforcement Learning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.14295v1)  

---


**ABSTRACT**  
Traffic light control is important for reducing congestion in urban mobility systems. This paper proposes a real-time traffic light control method using deep Q learning. Our approach incorporates a reward function considering queue lengths, delays, travel time, and throughput. The model dynamically decides phase changes based on current traffic conditions. The training of the deep Q network involves an offline stage from pre-generated data with fixed schedules and an online stage using real-time traffic data. A deep Q network structure with a "phase gate" component is used to simplify the model's learning task under different phases. A "memory palace" mechanism is used to address sample imbalance during the training process. We validate our approach using both synthetic and real-world traffic flow data on a road intersecting in Hangzhou, China. Results demonstrate significant performance improvements of the proposed method in reducing vehicle waiting time (57.1% to 100%), queue lengths (40.9% to 100%), and total travel time (16.8% to 68.0%) compared to traditional fixed signal plans.

{{</citation>}}


### (37/117) LLM Powered Sim-to-real Transfer for Traffic Signal Control (Longchao Da et al., 2023)

{{<citation>}}

Longchao Da, Minchiuan Gao, Hao Mei, Hua Wei. (2023)  
**LLM Powered Sim-to-real Transfer for Traffic Signal Control**  

---
Primary Category: cs.AI  
Categories: H-4-0, cs-AI, cs.AI  
Keywords: Language Model, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.14284v1)  

---


**ABSTRACT**  
Numerous solutions are proposed for the Traffic Signal Control (TSC) tasks aiming to provide efficient transportation and mitigate congestion waste. In recent, promising results have been attained by Reinforcement Learning (RL) methods through trial and error in simulators, bringing confidence in solving cities' congestion headaches. However, there still exist performance gaps when simulator-trained policies are deployed to the real world. This issue is mainly introduced by the system dynamic difference between the training simulator and the real-world environments. The Large Language Models (LLMs) are trained on mass knowledge and proved to be equipped with astonishing inference abilities. In this work, we leverage LLMs to understand and profile the system dynamics by a prompt-based grounded action transformation. Accepting the cloze prompt template, and then filling in the answer based on accessible context, the pre-trained LLM's inference ability is exploited and applied to understand how weather conditions, traffic states, and road types influence traffic dynamics, being aware of this, the policies' action is taken and grounded based on realistic dynamics, thus help the agent learn a more realistic policy. We conduct experiments using DQN to show the effectiveness of the proposed PromptGAT's ability in mitigating the performance gap from simulation to reality (sim-to-real).

{{</citation>}}


### (38/117) The Promise and Peril of Artificial Intelligence -- Violet Teaming Offers a Balanced Path Forward (Alexander J. Titus et al., 2023)

{{<citation>}}

Alexander J. Titus, Adam H. Russell. (2023)  
**The Promise and Peril of Artificial Intelligence -- Violet Teaming Offers a Balanced Path Forward**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CR, cs-LG, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.14253v1)  

---


**ABSTRACT**  
Artificial intelligence (AI) promises immense benefits across sectors, yet also poses risks from dual-use potentials, biases, and unintended behaviors. This paper reviews emerging issues with opaque and uncontrollable AI systems and proposes an integrative framework called violet teaming to develop reliable and responsible AI. Violet teaming combines adversarial vulnerability probing (red teaming) with solutions for safety and security (blue teaming) while prioritizing ethics and social benefit. It emerged from AI safety research to manage risks proactively by design. The paper traces the evolution of red, blue, and purple teaming toward violet teaming, and then discusses applying violet techniques to address biosecurity risks of AI in biotechnology. Additional sections review key perspectives across law, ethics, cybersecurity, macrostrategy, and industry best practices essential for operationalizing responsible AI through holistic technical and social considerations. Violet teaming provides both philosophy and method for steering AI trajectories toward societal good. With conscience and wisdom, the extraordinary capabilities of AI can enrich humanity. But without adequate precaution, the risks could prove catastrophic. Violet teaming aims to empower moral technology for the common welfare.

{{</citation>}}


### (39/117) The Cultural Psychology of Large Language Models: Is ChatGPT a Holistic or Analytic Thinker? (Chuanyang Jin et al., 2023)

{{<citation>}}

Chuanyang Jin, Songyang Zhang, Tianmin Shu, Zhihan Cui. (2023)  
**The Cultural Psychology of Large Language Models: Is ChatGPT a Holistic or Analytic Thinker?**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: AI, ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2308.14242v1)  

---


**ABSTRACT**  
The prevalent use of Large Language Models (LLMs) has necessitated studying their mental models, yielding noteworthy theoretical and practical implications. Current research has demonstrated that state-of-the-art LLMs, such as ChatGPT, exhibit certain theory of mind capabilities and possess relatively stable Big Five and/or MBTI personality traits. In addition, cognitive process features form an essential component of these mental models. Research in cultural psychology indicated significant differences in the cognitive processes of Eastern and Western people when processing information and making judgments. While Westerners predominantly exhibit analytical thinking that isolates things from their environment to analyze their nature independently, Easterners often showcase holistic thinking, emphasizing relationships and adopting a global viewpoint. In our research, we probed the cultural cognitive traits of ChatGPT. We employed two scales that directly measure the cognitive process: the Analysis-Holism Scale (AHS) and the Triadic Categorization Task (TCT). Additionally, we used two scales that investigate the value differences shaped by cultural thinking: the Dialectical Self Scale (DSS) and the Self-construal Scale (SCS). In cognitive process tests (AHS/TCT), ChatGPT consistently tends towards Eastern holistic thinking, but regarding value judgments (DSS/SCS), ChatGPT does not significantly lean towards the East or the West. We suggest that the result could be attributed to both the training paradigm and the training data in LLM development. We discuss the potential value of this finding for AI research and directions for future research.

{{</citation>}}


## quant-ph (1)



### (40/117) Distributionally Robust Variational Quantum Algorithms with Shifted Noise (Zichang He et al., 2023)

{{<citation>}}

Zichang He, Bo Peng, Yuri Alexeev, Zheng Zhang. (2023)  
**Distributionally Robust Variational Quantum Algorithms with Shifted Noise**  

---
Primary Category: quant-ph  
Categories: cs-ET, quant-ph, quant-ph  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2308.14935v1)  

---


**ABSTRACT**  
Given their potential to demonstrate near-term quantum advantage, variational quantum algorithms (VQAs) have been extensively studied. Although numerous techniques have been developed for VQA parameter optimization, it remains a significant challenge. A practical issue is the high sensitivity of quantum noise to environmental changes, and its propensity to shift in real time. This presents a critical problem as an optimized VQA ansatz may not perform effectively under a different noise environment. For the first time, we explore how to optimize VQA parameters to be robust against unknown shifted noise. We model the noise level as a random variable with an unknown probability density function (PDF), and we assume that the PDF may shift within an uncertainty set. This assumption guides us to formulate a distributionally robust optimization problem, with the goal of finding parameters that maintain effectiveness under shifted noise. We utilize a distributionally robust Bayesian optimization solver for our proposed formulation. This provides numerical evidence in both the Quantum Approximate Optimization Algorithm (QAOA) and the Variational Quantum Eigensolver (VQE) with hardware-efficient ansatz, indicating that we can identify parameters that perform more robustly under shifted noise. We regard this work as the first step towards improving the reliability of VQAs influenced by real-time noise.

{{</citation>}}


## cs.CV (22)



### (41/117) Application of Quantum Pre-Processing Filter for Binary Image Classification with Small Samples (Farina Riaz et al., 2023)

{{<citation>}}

Farina Riaz, Shahab Abdulla, Hajime Suzuki, Srinjoy Ganguly, Ravinesh C. Deo, Susan Hopkins. (2023)  
**Application of Quantum Pre-Processing Filter for Binary Image Classification with Small Samples**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2308.14930v1)  

---


**ABSTRACT**  
Over the past few years, there has been significant interest in Quantum Machine Learning (QML) among researchers, as it has the potential to transform the field of machine learning. Several models that exploit the properties of quantum mechanics have been developed for practical applications. In this study, we investigated the application of our previously proposed quantum pre-processing filter (QPF) to binary image classification. We evaluated the QPF on four datasets: MNIST (handwritten digits), EMNIST (handwritten digits and alphabets), CIFAR-10 (photographic images) and GTSRB (real-life traffic sign images). Similar to our previous multi-class classification results, the application of QPF improved the binary image classification accuracy using neural network against MNIST, EMNIST, and CIFAR-10 from 98.9% to 99.2%, 97.8% to 98.3%, and 71.2% to 76.1%, respectively, but degraded it against GTSRB from 93.5% to 92.0%. We then applied QPF in cases using a smaller number of training and testing samples, i.e. 80 and 20 samples per class, respectively. In order to derive statistically stable results, we conducted the experiment with 100 trials choosing randomly different training and testing samples and averaging the results. The result showed that the application of QPF did not improve the image classification accuracy against MNIST and EMNIST but improved it against CIFAR-10 and GTSRB from 65.8% to 67.2% and 90.5% to 91.8%, respectively. Further research will be conducted as part of future work to investigate the potential of QPF to assess the scalability of the proposed approach to larger and complex datasets.

{{</citation>}}


### (42/117) Maturity-Aware Active Learning for Semantic Segmentation with Hierarchically-Adaptive Sample Assessment (Amirsaeed Yazdani et al., 2023)

{{<citation>}}

Amirsaeed Yazdani, Xuelu Li, Vishal Monga. (2023)  
**Maturity-Aware Active Learning for Semantic Segmentation with Hierarchically-Adaptive Sample Assessment**  

---
Primary Category: cs.CV  
Categories: 68-06, I-4-6; I-5-1, cs-CV, cs-LG, cs.CV  
Keywords: Active Learning, Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2308.14904v1)  

---


**ABSTRACT**  
Active Learning (AL) for semantic segmentation is challenging due to heavy class imbalance and different ways of defining "sample" (pixels, areas, etc.), leaving the interpretation of the data distribution ambiguous. We propose "Maturity-Aware Distribution Breakdown-based Active Learning'' (MADBAL), an AL method that benefits from a hierarchical approach to define a multiview data distribution, which takes into account the different "sample" definitions jointly, hence able to select the most impactful segmentation pixels with comprehensive understanding. MADBAL also features a novel uncertainty formulation, where AL supporting modules are included to sense the features' maturity whose weighted influence continuously contributes to the uncertainty detection. In this way, MADBAL makes significant performance leaps even in the early AL stage, hence reducing the training burden significantly. It outperforms state-of-the-art methods on Cityscapes and PASCAL VOC datasets as verified in our extensive experiments.

{{</citation>}}


### (43/117) When hard negative sampling meets supervised contrastive learning (Zijun Long et al., 2023)

{{<citation>}}

Zijun Long, George Killick, Richard McCreadie, Gerardo Aragon Camarasa, Zaiqiao Meng. (2023)  
**When hard negative sampling meets supervised contrastive learning**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2308.14893v1)  

---


**ABSTRACT**  
State-of-the-art image models predominantly follow a two-stage strategy: pre-training on large datasets and fine-tuning with cross-entropy loss. Many studies have shown that using cross-entropy can result in sub-optimal generalisation and stability. While the supervised contrastive loss addresses some limitations of cross-entropy loss by focusing on intra-class similarities and inter-class differences, it neglects the importance of hard negative mining. We propose that models will benefit from performance improvement by weighting negative samples based on their dissimilarity to positive counterparts. In this paper, we introduce a new supervised contrastive learning objective, SCHaNe, which incorporates hard negative sampling during the fine-tuning phase. Without requiring specialized architectures, additional data, or extra computational resources, experimental results indicate that SCHaNe outperforms the strong baseline BEiT-3 in Top-1 accuracy across various benchmarks, with significant gains of up to $3.32\%$ in few-shot learning settings and $3.41\%$ in full dataset fine-tuning. Importantly, our proposed objective sets a new state-of-the-art for base models on ImageNet-1k, achieving an 86.14\% accuracy. Furthermore, we demonstrate that the proposed objective yields better embeddings and explains the improved effectiveness observed in our experiments.

{{</citation>}}


### (44/117) SynthDistill: Face Recognition with Knowledge Distillation from Synthetic Data (Hatef Otroshi Shahreza et al., 2023)

{{<citation>}}

Hatef Otroshi Shahreza, Anjith George, Sébastien Marcel. (2023)  
**SynthDistill: Face Recognition with Knowledge Distillation from Synthetic Data**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2308.14852v1)  

---


**ABSTRACT**  
State-of-the-art face recognition networks are often computationally expensive and cannot be used for mobile applications. Training lightweight face recognition models also requires large identity-labeled datasets. Meanwhile, there are privacy and ethical concerns with collecting and using large face recognition datasets. While generating synthetic datasets for training face recognition models is an alternative option, it is challenging to generate synthetic data with sufficient intra-class variations. In addition, there is still a considerable gap between the performance of models trained on real and synthetic data. In this paper, we propose a new framework (named SynthDistill) to train lightweight face recognition models by distilling the knowledge of a pretrained teacher face recognition model using synthetic data. We use a pretrained face generator network to generate synthetic face images and use the synthesized images to learn a lightweight student network. We use synthetic face images without identity labels, mitigating the problems in the intra-class variation generation of synthetic datasets. Instead, we propose a novel dynamic sampling strategy from the intermediate latent space of the face generator network to include new variations of the challenging images while further exploring new face images in the training batch. The results on five different face recognition datasets demonstrate the superiority of our lightweight model compared to models trained on previous synthetic datasets, achieving a verification accuracy of 99.52% on the LFW dataset with a lightweight network. The results also show that our proposed framework significantly reduces the gap between training with real and synthetic data. The source code for replicating the experiments is publicly released.

{{</citation>}}


### (45/117) PanoSwin: a Pano-style Swin Transformer for Panorama Understanding (Zhixin Ling et al., 2023)

{{<citation>}}

Zhixin Ling, Zhen Xing, Xiangdong Zhou, Manliang Cao, Guichun Zhou. (2023)  
**PanoSwin: a Pano-style Swin Transformer for Panorama Understanding**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.14726v1)  

---


**ABSTRACT**  
In panorama understanding, the widely used equirectangular projection (ERP) entails boundary discontinuity and spatial distortion. It severely deteriorates the conventional CNNs and vision Transformers on panoramas. In this paper, we propose a simple yet effective architecture named PanoSwin to learn panorama representations with ERP. To deal with the challenges brought by equirectangular projection, we explore a pano-style shift windowing scheme and novel pitch attention to address the boundary discontinuity and the spatial distortion, respectively. Besides, based on spherical distance and Cartesian coordinates, we adapt absolute positional embeddings and relative positional biases for panoramas to enhance panoramic geometry information. Realizing that planar image understanding might share some common knowledge with panorama understanding, we devise a novel two-stage learning framework to facilitate knowledge transfer from the planar images to panoramas. We conduct experiments against the state-of-the-art on various panoramic tasks, i.e., panoramic object detection, panoramic classification, and panoramic layout estimation. The experimental results demonstrate the effectiveness of PanoSwin in panorama understanding.

{{</citation>}}


### (46/117) Neural Network-Based Histologic Remission Prediction In Ulcerative Colitis (Yemin li et al., 2023)

{{<citation>}}

Yemin li, Zhongcheng Liu, Xiaoying Lou, Mirigual Kurban, Miao Li, Jie Yang, Kaiwei Che, Jiankun Wang, Max Q. -H Meng, Yan Huang, Qin Guo, Pinjin Hu. (2023)  
**Neural Network-Based Histologic Remission Prediction In Ulcerative Colitis**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.14667v1)  

---


**ABSTRACT**  
BACKGROUND & AIMS: Histological remission (HR) is advocated and considered as a new therapeutic target in ulcerative colitis (UC). Diagnosis of histologic remission currently relies on biopsy; during this process, patients are at risk for bleeding, infection, and post-biopsy fibrosis. In addition, histologic response scoring is complex and time-consuming, and there is heterogeneity among pathologists. Endocytoscopy (EC) is a novel ultra-high magnification endoscopic technique that can provide excellent in vivo assessment of glands. Based on the EC technique, we propose a neural network model that can assess histological disease activity in UC using EC images to address the above issues. The experiment results demonstrate that the proposed method can assist patients in precise treatment and prognostic assessment.   METHODS: We construct a neural network model for UC evaluation. A total of 5105 images of 154 intestinal segments from 87 patients undergoing EC treatment at a center in China between March 2022 and March 2023 are scored according to the Geboes score. Subsequently, 103 intestinal segments are used as the training set, 16 intestinal segments are used as the validation set for neural network training, and the remaining 35 intestinal segments are used as the test set to measure the model performance together with the validation set.   RESULTS: By treating HR as a negative category and histologic activity as a positive category, the proposed neural network model can achieve an accuracy of 0.9, a specificity of 0.95, a sensitivity of 0.75, and an area under the curve (AUC) of 0.81.   CONCLUSION: We develop a specific neural network model that can distinguish histologic remission/activity in EC images of UC, which helps to accelerate clinical histological diagnosis.   keywords: ulcerative colitis; Endocytoscopy; Geboes score; neural network.

{{</citation>}}


### (47/117) A Generalization of Continuous Relaxation in Structured Pruning (Brad Larson et al., 2023)

{{<citation>}}

Brad Larson, Bishal Upadhyaya, Luke McDermott, Siddha Ganju. (2023)  
**A Generalization of Continuous Relaxation in Structured Pruning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Pruning  
[Paper Link](http://arxiv.org/abs/2308.14605v1)  

---


**ABSTRACT**  
Deep learning harnesses massive parallel floating-point processing to train and evaluate large neural networks. Trends indicate that deeper and larger neural networks with an increasing number of parameters achieve higher accuracy than smaller neural networks. This performance improvement, which often requires heavy compute for both training and evaluation, eventually needs to translate well to resource-constrained hardware for practical value. Structured pruning asserts that while large networks enable us to find solutions to complex computer vision problems, a smaller, computationally efficient sub-network can be derived from the large neural network that retains model accuracy but significantly improves computational efficiency.   We generalize structured pruning with algorithms for network augmentation, pruning, sub-network collapse and removal. In addition, we demonstrate efficient and stable convergence up to 93% sparsity and 95% FLOPs reduction without loss of inference accuracy using with continuous relaxation matching or exceeding the state of the art for all structured pruning methods. The resulting CNN executes efficiently on GPU hardware without computationally expensive sparse matrix operations. We achieve this with routine automatable operations on classification and segmentation problems using CIFAR-10, ImageNet, and CityScapes datasets with the ResNet and U-NET network architectures.

{{</citation>}}


### (48/117) Adversarial Attacks on Foundational Vision Models (Nathan Inkawhich et al., 2023)

{{<citation>}}

Nathan Inkawhich, Gwendolyn McDonald, Ryan Luley. (2023)  
**Adversarial Attacks on Foundational Vision Models**  

---
Primary Category: cs.CV  
Categories: cs-CR, cs-CV, cs-LG, cs.CV  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2308.14597v1)  

---


**ABSTRACT**  
Rapid progress is being made in developing large, pretrained, task-agnostic foundational vision models such as CLIP, ALIGN, DINOv2, etc. In fact, we are approaching the point where these models do not have to be finetuned downstream, and can simply be used in zero-shot or with a lightweight probing head. Critically, given the complexity of working at this scale, there is a bottleneck where relatively few organizations in the world are executing the training then sharing the models on centralized platforms such as HuggingFace and torch.hub. The goal of this work is to identify several key adversarial vulnerabilities of these models in an effort to make future designs more robust. Intuitively, our attacks manipulate deep feature representations to fool an out-of-distribution (OOD) detector which will be required when using these open-world-aware models to solve closed-set downstream tasks. Our methods reliably make in-distribution (ID) images (w.r.t. a downstream task) be predicted as OOD and vice versa while existing in extremely low-knowledge-assumption threat models. We show our attacks to be potent in whitebox and blackbox settings, as well as when transferred across foundational model types (e.g., attack DINOv2 with CLIP)! This work is only just the beginning of a long journey towards adversarially robust foundational vision models.

{{</citation>}}


### (49/117) Neural Network Training Strategy to Enhance Anomaly Detection Performance: A Perspective on Reconstruction Loss Amplification (YeongHyeon Park et al., 2023)

{{<citation>}}

YeongHyeon Park, Sungho Kang, Myung Jin Kim, Hyeonho Jeong, Hyunkyu Park, Hyeong Seok Kim, Juneho Yi. (2023)  
**Neural Network Training Strategy to Enhance Anomaly Detection Performance: A Perspective on Reconstruction Loss Amplification**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV, eess-IV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2308.14595v1)  

---


**ABSTRACT**  
Unsupervised anomaly detection (UAD) is a widely adopted approach in industry due to rare anomaly occurrences and data imbalance. A desirable characteristic of an UAD model is contained generalization ability which excels in the reconstruction of seen normal patterns but struggles with unseen anomalies. Recent studies have pursued to contain the generalization capability of their UAD models in reconstruction from different perspectives, such as design of neural network (NN) structure and training strategy. In contrast, we note that containing of generalization ability in reconstruction can also be obtained simply from steep-shaped loss landscape. Motivated by this, we propose a loss landscape sharpening method by amplifying the reconstruction loss, dubbed Loss AMPlification (LAMP). LAMP deforms the loss landscape into a steep shape so the reconstruction error on unseen anomalies becomes greater. Accordingly, the anomaly detection performance is improved without any change of the NN architecture. Our findings suggest that LAMP can be easily applied to any reconstruction error metrics in UAD settings where the reconstruction model is trained with anomaly-free samples only.

{{</citation>}}


### (50/117) Face Presentation Attack Detection by Excavating Causal Clues and Adapting Embedding Statistics (Meiling Fang et al., 2023)

{{<citation>}}

Meiling Fang, Naser Damer. (2023)  
**Face Presentation Attack Detection by Excavating Causal Clues and Adapting Embedding Statistics**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2308.14551v1)  

---


**ABSTRACT**  
Recent face presentation attack detection (PAD) leverages domain adaptation (DA) and domain generalization (DG) techniques to address performance degradation on unknown domains. However, DA-based PAD methods require access to unlabeled target data, while most DG-based PAD solutions rely on a priori, i.e., known domain labels. Moreover, most DA-/DG-based methods are computationally intensive, demanding complex model architectures and/or multi-stage training processes. This paper proposes to model face PAD as a compound DG task from a causal perspective, linking it to model optimization. We excavate the causal factors hidden in the high-level representation via counterfactual intervention. Moreover, we introduce a class-guided MixStyle to enrich feature-level data distribution within classes instead of focusing on domain information. Both class-guided MixStyle and counterfactual intervention components introduce no extra trainable parameters and negligible computational resources. Extensive cross-dataset and analytic experiments demonstrate the effectiveness and efficiency of our method compared to state-of-the-art PADs. The implementation and the trained weights are publicly available.

{{</citation>}}


### (51/117) Semi-Supervised Learning for Visual Bird's Eye View Semantic Segmentation (Junyu Zhu et al., 2023)

{{<citation>}}

Junyu Zhu, Lina Liu, Yu Tang, Feng Wen, Wanlong Li, Yong Liu. (2023)  
**Semi-Supervised Learning for Visual Bird's Eye View Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2308.14525v1)  

---


**ABSTRACT**  
Visual bird's eye view (BEV) semantic segmentation helps autonomous vehicles understand the surrounding environment only from images, including static elements (e.g., roads) and dynamic elements (e.g., vehicles, pedestrians). However, the high cost of annotation procedures of full-supervised methods limits the capability of the visual BEV semantic segmentation, which usually needs HD maps, 3D object bounding boxes, and camera extrinsic matrixes. In this paper, we present a novel semi-supervised framework for visual BEV semantic segmentation to boost performance by exploiting unlabeled images during the training. A consistency loss that makes full use of unlabeled data is then proposed to constrain the model on not only semantic prediction but also the BEV feature. Furthermore, we propose a novel and effective data augmentation method named conjoint rotation which reasonably augments the dataset while maintaining the geometric relationship between the front-view images and the BEV semantic segmentation. Extensive experiments on the nuScenes and Argoverse datasets show that our semi-supervised framework can effectively improve prediction accuracy. To the best of our knowledge, this is the first work that explores improving visual BEV semantic segmentation performance using unlabeled data. The code will be publicly available.

{{</citation>}}


### (52/117) Priority-Centric Human Motion Generation in Discrete Latent Space (Hanyang Kong et al., 2023)

{{<citation>}}

Hanyang Kong, Kehong Gong, Dongze Lian, Michael Bi Mi, Xinchao Wang. (2023)  
**Priority-Centric Human Motion Generation in Discrete Latent Space**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.14480v2)  

---


**ABSTRACT**  
Text-to-motion generation is a formidable task, aiming to produce human motions that align with the input text while also adhering to human capabilities and physical laws. While there have been advancements in diffusion models, their application in discrete spaces remains underexplored. Current methods often overlook the varying significance of different motions, treating them uniformly. It is essential to recognize that not all motions hold the same relevance to a particular textual description. Some motions, being more salient and informative, should be given precedence during generation. In response, we introduce a Priority-Centric Motion Discrete Diffusion Model (M2DM), which utilizes a Transformer-based VQ-VAE to derive a concise, discrete motion representation, incorporating a global self-attention mechanism and a regularization term to counteract code collapse. We also present a motion discrete diffusion model that employs an innovative noise schedule, determined by the significance of each motion token within the entire motion sequence. This approach retains the most salient motions during the reverse diffusion process, leading to more semantically rich and varied motions. Additionally, we formulate two strategies to gauge the importance of motion tokens, drawing from both textual and visual indicators. Comprehensive experiments on the HumanML3D and KIT-ML datasets confirm that our model surpasses existing techniques in fidelity and diversity, particularly for intricate textual descriptions.

{{</citation>}}


### (53/117) Medical needle tip tracking based on Optical Imaging and AI (Zhuoqi Cheng et al., 2023)

{{<citation>}}

Zhuoqi Cheng, Simon Lyck Bjært Sørensen, Mikkel Werge Olsen, René Lynge Eriksen, Thiusius Rajeeth Savarimuthu. (2023)  
**Medical needle tip tracking based on Optical Imaging and AI**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.14477v1)  

---


**ABSTRACT**  
Deep needle insertion to a target often poses a huge challenge, requiring a combination of specialized skills, assistive technology, and extensive training. One of the frequently encountered medical scenarios demanding such expertise includes the needle insertion into a femoral vessel in the groin. After the access to the femoral vessel, various medical procedures, such as cardiac catheterization and extracorporeal membrane oxygenation (ECMO) can be performed. However, even with the aid of Ultrasound imaging, achieving successful insertion can necessitate multiple attempts due to the complexities of anatomy and tissue deformation. To address this challenge, this paper presents an innovative technology for needle tip real-time tracking, aiming for enhanced needle insertion guidance. Specifically, our approach revolves around the creation of scattering imaging using an optical fiber-equipped needle, and uses Convolutional Neural Network (CNN) based algorithms to enable real-time estimation of the needle tip's position and orientation during insertion procedures. The efficacy of the proposed technology was rigorously evaluated through three experiments. The first two experiments involved rubber and bacon phantoms to simulate groin anatomy. The positional errors averaging 2.3+1.5mm and 2.0+1.2mm, and the orientation errors averaging 0.2+0.11rad and 0.16+0.1rad. Furthermore, the system's capabilities were validated through experiments conducted on fresh porcine phantom mimicking more complex anatomical structures, yielding positional accuracy results of 3.2+3.1mm and orientational accuracy of 0.19+0.1rad. Given the average femoral arterial radius of 4 to 5mm, the proposed system is demonstrated with a great potential for precise needle guidance in femoral artery insertion procedures. In addition, the findings highlight the broader potential applications of the system in the medical field.

{{</citation>}}


### (54/117) ExpCLIP: Bridging Text and Facial Expressions via Semantic Alignment (Yicheng Zhong et al., 2023)

{{<citation>}}

Yicheng Zhong, Huawei Wei, Peiji Yang, Zhisheng Wang. (2023)  
**ExpCLIP: Bridging Text and Facial Expressions via Semantic Alignment**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Augmentation, Language Model  
[Paper Link](http://arxiv.org/abs/2308.14448v1)  

---


**ABSTRACT**  
The objective of stylized speech-driven facial animation is to create animations that encapsulate specific emotional expressions. Existing methods often depend on pre-established emotional labels or facial expression templates, which may limit the necessary flexibility for accurately conveying user intent. In this research, we introduce a technique that enables the control of arbitrary styles by leveraging natural language as emotion prompts. This technique presents benefits in terms of both flexibility and user-friendliness. To realize this objective, we initially construct a Text-Expression Alignment Dataset (TEAD), wherein each facial expression is paired with several prompt-like descriptions.We propose an innovative automatic annotation method, supported by Large Language Models (LLMs), to expedite the dataset construction, thereby eliminating the substantial expense of manual annotation. Following this, we utilize TEAD to train a CLIP-based model, termed ExpCLIP, which encodes text and facial expressions into semantically aligned style embeddings. The embeddings are subsequently integrated into the facial animation generator to yield expressive and controllable facial animations. Given the limited diversity of facial emotions in existing speech-driven facial animation training data, we further introduce an effective Expression Prompt Augmentation (EPA) mechanism to enable the animation generator to support unprecedented richness in style control. Comprehensive experiments illustrate that our method accomplishes expressive facial animation generation and offers enhanced flexibility in effectively conveying the desired style.

{{</citation>}}


### (55/117) Multi-Scale and Multi-Layer Contrastive Learning for Domain Generalization (Aristotelis Ballas et al., 2023)

{{<citation>}}

Aristotelis Ballas, Christos Diou. (2023)  
**Multi-Scale and Multi-Layer Contrastive Learning for Domain Generalization**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2308.14418v1)  

---


**ABSTRACT**  
During the past decade, deep neural networks have led to fast-paced progress and significant achievements in computer vision problems, for both academia and industry. Yet despite their success, state-of-the-art image classification approaches fail to generalize well in previously unseen visual contexts, as required by many real-world applications. In this paper, we focus on this domain generalization (DG) problem and argue that the generalization ability of deep convolutional neural networks can be improved by taking advantage of multi-layer and multi-scaled representations of the network. We introduce a framework that aims at improving domain generalization of image classifiers by combining both low-level and high-level features at multiple scales, enabling the network to implicitly disentangle representations in its latent space and learn domain-invariant attributes of the depicted objects. Additionally, to further facilitate robust representation learning, we propose a novel objective function, inspired by contrastive learning, which aims at constraining the extracted representations to remain invariant under distribution shifts. We demonstrate the effectiveness of our method by evaluating on the domain generalization datasets of PACS, VLCS, Office-Home and NICO. Through extensive experimentation, we show that our model is able to surpass the performance of previous DG methods and consistently produce competitive and state-of-the-art results in all datasets.

{{</citation>}}


### (56/117) Semi-Supervised Semantic Depth Estimation using Symbiotic Transformer and NearFarMix Augmentation (Md Awsafur Rahman et al., 2023)

{{<citation>}}

Md Awsafur Rahman, Shaikh Anowarul Fattah. (2023)  
**Semi-Supervised Semantic Depth Estimation using Symbiotic Transformer and NearFarMix Augmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Augmentation, Semi-Supervised, Transformer  
[Paper Link](http://arxiv.org/abs/2308.14400v1)  

---


**ABSTRACT**  
In computer vision, depth estimation is crucial for domains like robotics, autonomous vehicles, augmented reality, and virtual reality. Integrating semantics with depth enhances scene understanding through reciprocal information sharing. However, the scarcity of semantic information in datasets poses challenges. Existing convolutional approaches with limited local receptive fields hinder the full utilization of the symbiotic potential between depth and semantics. This paper introduces a dataset-invariant semi-supervised strategy to address the scarcity of semantic information. It proposes the Depth Semantics Symbiosis module, leveraging the Symbiotic Transformer for achieving comprehensive mutual awareness by information exchange within both local and global contexts. Additionally, a novel augmentation, NearFarMix is introduced to combat overfitting and compensate both depth-semantic tasks by strategically merging regions from two images, generating diverse and structurally consistent samples with enhanced control. Extensive experiments on NYU-Depth-V2 and KITTI datasets demonstrate the superiority of our proposed techniques in indoor and outdoor environments.

{{</citation>}}


### (57/117) FIRE: Food Image to REcipe generation (Prateek Chhikara et al., 2023)

{{<citation>}}

Prateek Chhikara, Dhiraj Chaurasia, Yifan Jiang, Omkar Masur, Filip Ilievski. (2023)  
**FIRE: Food Image to REcipe generation**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: T5, Transformer  
[Paper Link](http://arxiv.org/abs/2308.14391v1)  

---


**ABSTRACT**  
Food computing has emerged as a prominent multidisciplinary field of research in recent years. An ambitious goal of food computing is to develop end-to-end intelligent systems capable of autonomously producing recipe information for a food image. Current image-to-recipe methods are retrieval-based and their success depends heavily on the dataset size and diversity, as well as the quality of learned embeddings. Meanwhile, the emergence of powerful attention-based vision and language models presents a promising avenue for accurate and generalizable recipe generation, which has yet to be extensively explored. This paper proposes FIRE, a novel multimodal methodology tailored to recipe generation in the food computing domain, which generates the food title, ingredients, and cooking instructions based on input food images. FIRE leverages the BLIP model to generate titles, utilizes a Vision Transformer with a decoder for ingredient extraction, and employs the T5 model to generate recipes incorporating titles and ingredients as inputs. We showcase two practical applications that can benefit from integrating FIRE with large language model prompting: recipe customization to fit recipes to user preferences and recipe-to-code transformation to enable automated cooking processes. Our experimental findings validate the efficacy of our proposed approach, underscoring its potential for future advancements and widespread adoption in food computing.

{{</citation>}}


### (58/117) GKGNet: Group K-Nearest Neighbor based Graph Convolutional Network for Multi-Label Image Recognition (Ruijie Yao et al., 2023)

{{<citation>}}

Ruijie Yao, Sheng Jin, Lumin Xu, Wang Zeng, Wentao Liu, Chen Qian, Ping Luo, Ji Wu. (2023)  
**GKGNet: Group K-Nearest Neighbor based Graph Convolutional Network for Multi-Label Image Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Graph Convolutional Network  
[Paper Link](http://arxiv.org/abs/2308.14378v1)  

---


**ABSTRACT**  
Multi-Label Image Recognition (MLIR) is a challenging task that aims to predict multiple object labels in a single image while modeling the complex relationships between labels and image regions. Although convolutional neural networks and vision transformers have succeeded in processing images as regular grids of pixels or patches, these representations are sub-optimal for capturing irregular and discontinuous regions of interest. In this work, we present the first fully graph convolutional model, Group K-nearest neighbor based Graph convolutional Network (GKGNet), which models the connections between semantic label embeddings and image patches in a flexible and unified graph structure. To address the scale variance of different objects and to capture information from multiple perspectives, we propose the Group KGCN module for dynamic graph construction and message passing. Our experiments demonstrate that GKGNet achieves state-of-the-art performance with significantly lower computational costs on the challenging multi-label datasets, \ie MS-COCO and VOC2007 datasets. We will release the code and models to facilitate future research in this area.

{{</citation>}}


### (59/117) MetaWeather: Few-Shot Weather-Degraded Image Restoration via Degradation Pattern Matching (Youngrae Kim et al., 2023)

{{<citation>}}

Youngrae Kim, Younggeol Cho, Thanh-Tung Nguyen, Dongman Lee. (2023)  
**MetaWeather: Few-Shot Weather-Degraded Image Restoration via Degradation Pattern Matching**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2308.14334v1)  

---


**ABSTRACT**  
Real-world vision tasks frequently suffer from the appearance of adverse weather conditions including rain, fog, snow, and raindrops in captured images. Recently, several generic methods for restoring weather-degraded images have been proposed, aiming to remove multiple types of adverse weather effects present in the images. However, these methods have considered weather as discrete and mutually exclusive variables, leading to failure in generalizing to unforeseen weather conditions beyond the scope of the training data, such as the co-occurrence of rain, fog, and raindrops. To this end, weather-degraded image restoration models should have flexible adaptability to the current unknown weather condition to ensure reliable and optimal performance. The adaptation method should also be able to cope with data scarcity for real-world adaptation. This paper proposes MetaWeather, a few-shot weather-degraded image restoration method for arbitrary weather conditions. For this, we devise the core piece of MetaWeather, coined Degradation Pattern Matching Module (DPMM), which leverages representations from a few-shot support set by matching features between input and sample images under new weather conditions. In addition, we build meta-knowledge with episodic meta-learning on top of our MetaWeather architecture to provide flexible adaptability. In the meta-testing phase, we adopt a parameter-efficient fine-tuning method to preserve the prebuilt knowledge and avoid the overfitting problem. Experiments on the BID Task II.A dataset show our method achieves the best performance on PSNR and SSIM compared to state-of-the-art image restoration methods. Code is available at (TBA).

{{</citation>}}


### (60/117) Attention-Guided Lidar Segmentation and Odometry Using Image-to-Point Cloud Saliency Transfer (Guanqun Ding et al., 2023)

{{<citation>}}

Guanqun Ding, Nevrez Imamoglu, Ali Caglayan, Masahiro Murakawa, Ryosuke Nakamura. (2023)  
**Attention-Guided Lidar Segmentation and Odometry Using Image-to-Point Cloud Saliency Transfer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2308.14332v1)  

---


**ABSTRACT**  
LiDAR odometry estimation and 3D semantic segmentation are crucial for autonomous driving, which has achieved remarkable advances recently. However, these tasks are challenging due to the imbalance of points in different semantic categories for 3D semantic segmentation and the influence of dynamic objects for LiDAR odometry estimation, which increases the importance of using representative/salient landmarks as reference points for robust feature learning. To address these challenges, we propose a saliency-guided approach that leverages attention information to improve the performance of LiDAR odometry estimation and semantic segmentation models. Unlike in the image domain, only a few studies have addressed point cloud saliency information due to the lack of annotated training data. To alleviate this, we first present a universal framework to transfer saliency distribution knowledge from color images to point clouds, and use this to construct a pseudo-saliency dataset (i.e. FordSaliency) for point clouds. Then, we adopt point cloud-based backbones to learn saliency distribution from pseudo-saliency labels, which is followed by our proposed SalLiDAR module. SalLiDAR is a saliency-guided 3D semantic segmentation model that integrates saliency information to improve segmentation performance. Finally, we introduce SalLONet, a self-supervised saliency-guided LiDAR odometry network that uses the semantic and saliency predictions of SalLiDAR to achieve better odometry estimation. Our extensive experiments on benchmark datasets demonstrate that the proposed SalLiDAR and SalLONet models achieve state-of-the-art performance against existing methods, highlighting the effectiveness of image-to-LiDAR saliency knowledge transfer. Source code will be available at https://github.com/nevrez/SalLONet.

{{</citation>}}


### (61/117) Bridging Cross-task Protocol Inconsistency for Distillation in Dense Object Detection (Longrong Yang et al., 2023)

{{<citation>}}

Longrong Yang, Xianpan Zhou, Xuewei Li, Liang Qiao, Zheyang Li, Ziwei Yang, Gaoang Wang, Xi Li. (2023)  
**Bridging Cross-task Protocol Inconsistency for Distillation in Dense Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2308.14286v1)  

---


**ABSTRACT**  
Knowledge distillation (KD) has shown potential for learning compact models in dense object detection. However, the commonly used softmax-based distillation ignores the absolute classification scores for individual categories. Thus, the optimum of the distillation loss does not necessarily lead to the optimal student classification scores for dense object detectors. This cross-task protocol inconsistency is critical, especially for dense object detectors, since the foreground categories are extremely imbalanced. To address the issue of protocol differences between distillation and classification, we propose a novel distillation method with cross-task consistent protocols, tailored for the dense object detection. For classification distillation, we address the cross-task protocol inconsistency problem by formulating the classification logit maps in both teacher and student models as multiple binary-classification maps and applying a binary-classification distillation loss to each map. For localization distillation, we design an IoU-based Localization Distillation Loss that is free from specific network structures and can be compared with existing localization distillation losses. Our proposed method is simple but effective, and experimental results demonstrate its superiority over existing methods. Code is available at https://github.com/TinyTigerPan/BCKD.

{{</citation>}}


### (62/117) FaceChain: A Playground for Identity-Preserving Portrait Generation (Yang Liu et al., 2023)

{{<citation>}}

Yang Liu, Cheng Yu, Lei Shang, Ziheng Wu, Xingjun Wang, Yuze Zhao, Lin Zhu, Chen Cheng, Weitao Chen, Chao Xu, Haoyu Xie, Yuan Yao, Wenmeng Zhou, Yingda Chen, Xuansong Xie, Baigui Sun. (2023)  
**FaceChain: A Playground for Identity-Preserving Portrait Generation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.14256v1)  

---


**ABSTRACT**  
Recent advancement in personalized image generation have unveiled the intriguing capability of pre-trained text-to-image models on learning identity information from a collection of portrait images. However, existing solutions can be vulnerable in producing truthful details, and usually suffer from several defects such as (i) The generated face exhibit its own unique characteristics, \ie facial shape and facial feature positioning may not resemble key characteristics of the input, and (ii) The synthesized face may contain warped, blurred or corrupted regions. In this paper, we present FaceChain, a personalized portrait generation framework that combines a series of customized image-generation model and a rich set of face-related perceptual understanding models (\eg, face detection, deep face embedding extraction, and facial attribute recognition), to tackle aforementioned challenges and to generate truthful personalized portraits, with only a handful of portrait images as input. Concretely, we inject several SOTA face models into the generation procedure, achieving a more efficient label-tagging, data-processing, and model post-processing compared to previous solutions, such as DreamBooth ~\cite{ruiz2023dreambooth} , InstantBooth ~\cite{shi2023instantbooth} , or other LoRA-only approaches ~\cite{hu2021lora} . Through the development of FaceChain, we have identified several potential directions to accelerate development of Face/Human-Centric AIGC research and application. We have designed FaceChain as a framework comprised of pluggable components that can be easily adjusted to accommodate different styles and personalized needs. We hope it can grow to serve the burgeoning needs from the communities. FaceChain is open-sourced under Apache-2.0 license at \url{https://github.com/modelscope/facechain}.

{{</citation>}}


## cs.CL (20)



### (63/117) Gender bias and stereotypes in Large Language Models (Hadas Kotek et al., 2023)

{{<citation>}}

Hadas Kotek, Rikker Dockum, David Q. Sun. (2023)  
**Gender bias and stereotypes in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs-LG, cs.CL  
Keywords: Bias, Language Model  
[Paper Link](http://arxiv.org/abs/2308.14921v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have made substantial progress in the past several months, shattering state-of-the-art benchmarks in many domains. This paper investigates LLMs' behavior with respect to gender stereotypes, a known issue for prior models. We use a simple paradigm to test the presence of gender bias, building on but differing from WinoBias, a commonly used gender bias dataset, which is likely to be included in the training data of current LLMs. We test four recently published LLMs and demonstrate that they express biased assumptions about men and women's occupations. Our contributions in this paper are as follows: (a) LLMs are 3-6 times more likely to choose an occupation that stereotypically aligns with a person's gender; (b) these choices align with people's perceptions better than with the ground truth as reflected in official job statistics; (c) LLMs in fact amplify the bias beyond what is reflected in perceptions or the ground truth; (d) LLMs ignore crucial ambiguities in sentence structure 95% of the time in our study items, but when explicitly prompted, they recognize the ambiguity; (e) LLMs provide explanations for their choices that are factually inaccurate and likely obscure the true reason behind their predictions. That is, they provide rationalizations of their biased behavior. This highlights a key property of these models: LLMs are trained on imbalanced datasets; as such, even with the recent successes of reinforcement learning with human feedback, they tend to reflect those imbalances back at us. As with other types of societal biases, we suggest that LLMs must be carefully tested to ensure that they treat minoritized individuals and communities equitably.

{{</citation>}}


### (64/117) Multiscale Contextual Learning for Speech Emotion Recognition in Emergency Call Center Conversations (Théo Deschamps-Berger et al., 2023)

{{<citation>}}

Théo Deschamps-Berger, Lori Lamel, Laurence Devillers. (2023)  
**Multiscale Contextual Learning for Speech Emotion Recognition in Emergency Call Center Conversations**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: Emotion Recognition, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.14894v1)  

---


**ABSTRACT**  
Emotion recognition in conversations is essential for ensuring advanced human-machine interactions. However, creating robust and accurate emotion recognition systems in real life is challenging, mainly due to the scarcity of emotion datasets collected in the wild and the inability to take into account the dialogue context. The CEMO dataset, composed of conversations between agents and patients during emergency calls to a French call center, fills this gap. The nature of these interactions highlights the role of the emotional flow of the conversation in predicting patient emotions, as context can often make a difference in understanding actual feelings. This paper presents a multi-scale conversational context learning approach for speech emotion recognition, which takes advantage of this hypothesis. We investigated this approach on both speech transcriptions and acoustic segments. Experimentally, our method uses the previous or next information of the targeted segment. In the text domain, we tested the context window using a wide range of tokens (from 10 to 100) and at the speech turns level, considering inputs from both the same and opposing speakers. According to our tests, the context derived from previous tokens has a more significant influence on accurate prediction than the following tokens. Furthermore, taking the last speech turn of the same speaker in the conversation seems useful. In the acoustic domain, we conducted an in-depth analysis of the impact of the surrounding emotions on the prediction. While multi-scale conversational context learning using Transformers can enhance performance in the textual modality for emergency call recordings, incorporating acoustic context is more challenging.

{{</citation>}}


### (65/117) Attention Visualizer Package: Revealing Word Importance for Deeper Insight into Encoder-Only Transformer Models (Ala Alam Falaki et al., 2023)

{{<citation>}}

Ala Alam Falaki, Robin Gras. (2023)  
**Attention Visualizer Package: Revealing Word Importance for Deeper Insight into Encoder-Only Transformer Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2308.14850v1)  

---


**ABSTRACT**  
This report introduces the Attention Visualizer package, which is crafted to visually illustrate the significance of individual words in encoder-only transformer-based models. In contrast to other methods that center on tokens and self-attention scores, our approach will examine the words and their impact on the final embedding representation. Libraries like this play a crucial role in enhancing the interpretability and explainability of neural networks. They offer the opportunity to illuminate their internal mechanisms, providing a better understanding of how they operate and can be enhanced. You can access the code and review examples on the following GitHub repository: https://github.com/AlaFalaki/AttentionVisualizer.

{{</citation>}}


### (66/117) Fine-Tuning Llama 2 Large Language Models for Detecting Online Sexual Predatory Chats and Abusive Texts (Thanh Thi Nguyen et al., 2023)

{{<citation>}}

Thanh Thi Nguyen, Campbell Wilson, Janis Dalins. (2023)  
**Fine-Tuning Llama 2 Large Language Models for Detecting Online Sexual Predatory Chats and Abusive Texts**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2308.14683v1)  

---


**ABSTRACT**  
Detecting online sexual predatory behaviours and abusive language on social media platforms has become a critical area of research due to the growing concerns about online safety, especially for vulnerable populations such as children and adolescents. Researchers have been exploring various techniques and approaches to develop effective detection systems that can identify and mitigate these risks. Recent development of large language models (LLMs) has opened a new opportunity to address this problem more effectively. This paper proposes an approach to detection of online sexual predatory chats and abusive language using the open-source pretrained Llama 2 7B-parameter model, recently released by Meta GenAI. We fine-tune the LLM using datasets with different sizes, imbalance degrees, and languages (i.e., English, Roman Urdu and Urdu). Based on the power of LLMs, our approach is generic and automated without a manual search for a synergy between feature extraction and classifier design steps like conventional methods in this domain. Experimental results show a strong performance of the proposed approach, which performs proficiently and consistently across three distinct datasets with five sets of experiments. This study's outcomes indicate that the proposed method can be implemented in real-world applications (even with non-English languages) for flagging sexual predators, offensive or toxic content, hate speech, and discriminatory language in online discussions and comments to maintain respectful internet or digital communities. Furthermore, it can be employed for solving text classification problems with other potential applications such as sentiment analysis, spam and phishing detection, sorting legal documents, fake news detection, language identification, user intent recognition, text-based product categorization, medical record analysis, and resume screening.

{{</citation>}}


### (67/117) ANER: Arabic and Arabizi Named Entity Recognition using Transformer-Based Approach (Abdelrahman 'Boda' Sadallah et al., 2023)

{{<citation>}}

Abdelrahman "Boda" Sadallah, Omar Ahmed, Shimaa Mohamed, Omar Hatem, Doaa Hesham, Ahmed H. Yousef. (2023)  
**ANER: Arabic and Arabizi Named Entity Recognition using Transformer-Based Approach**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, NER, NLP, Named Entity Recognition, Natural Language Processing, Transformer  
[Paper Link](http://arxiv.org/abs/2308.14669v1)  

---


**ABSTRACT**  
One of the main tasks of Natural Language Processing (NLP), is Named Entity Recognition (NER). It is used in many applications and also can be used as an intermediate step for other tasks. We present ANER, a web-based named entity recognizer for the Arabic, and Arabizi languages. The model is built upon BERT, which is a transformer-based encoder. It can recognize 50 different entity classes, covering various fields. We trained our model on the WikiFANE\_Gold dataset which consists of Wikipedia articles. We achieved an F1 score of 88.7\%, which beats CAMeL Tools' F1 score of 83\% on the ANERcorp dataset, which has only 4 classes. We also got an F1 score of 77.7\% on the NewsFANE\_Gold dataset which contains out-of-domain data from News articles. The system is deployed on a user-friendly web interface that accepts users' inputs in Arabic, or Arabizi. It allows users to explore the entities in the text by highlighting them. It can also direct users to get information about entities through Wikipedia directly. We added the ability to do NER using our model, or CAMeL Tools' model through our website. ANER is publicly accessible at \url{http://www.aner.online}. We also deployed our model on HuggingFace at https://huggingface.co/boda/ANER, to allow developers to test and use it.

{{</citation>}}


### (68/117) Joint Multiple Intent Detection and Slot Filling with Supervised Contrastive Learning and Self-Distillation (Nguyen Anh Tu et al., 2023)

{{<citation>}}

Nguyen Anh Tu, Hoang Thi Thu Uyen, Tu Minh Phuong, Ngo Xuan Bach. (2023)  
**Joint Multiple Intent Detection and Slot Filling with Supervised Contrastive Learning and Self-Distillation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Contrastive Learning, Intent Detection  
[Paper Link](http://arxiv.org/abs/2308.14654v1)  

---


**ABSTRACT**  
Multiple intent detection and slot filling are two fundamental and crucial tasks in spoken language understanding. Motivated by the fact that the two tasks are closely related, joint models that can detect intents and extract slots simultaneously are preferred to individual models that perform each task independently. The accuracy of a joint model depends heavily on the ability of the model to transfer information between the two tasks so that the result of one task can correct the result of the other. In addition, since a joint model has multiple outputs, how to train the model effectively is also challenging. In this paper, we present a method for multiple intent detection and slot filling by addressing these challenges. First, we propose a bidirectional joint model that explicitly employs intent information to recognize slots and slot features to detect intents. Second, we introduce a novel method for training the proposed joint model using supervised contrastive learning and self-distillation. Experimental results on two benchmark datasets MixATIS and MixSNIPS show that our method outperforms state-of-the-art models in both tasks. The results also demonstrate the contributions of both bidirectional design and the training method to the accuracy improvement. Our source code is available at https://github.com/anhtunguyen98/BiSLU

{{</citation>}}


### (69/117) Challenges of GPT-3-based Conversational Agents for Healthcare (Fabian Lechner et al., 2023)

{{<citation>}}

Fabian Lechner, Allison Lahnala, Charles Welch, Lucie Flek. (2023)  
**Challenges of GPT-3-based Conversational Agents for Healthcare**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, QA  
[Paper Link](http://arxiv.org/abs/2308.14641v2)  

---


**ABSTRACT**  
The potential to provide patients with faster information access while allowing medical specialists to concentrate on critical tasks makes medical domain dialog agents appealing. However, the integration of large-language models (LLMs) into these agents presents certain limitations that may result in serious consequences. This paper investigates the challenges and risks of using GPT-3-based models for medical question-answering (MedQA). We perform several evaluations contextualized in terms of standard medical principles. We provide a procedure for manually designing patient queries to stress-test high-risk limitations of LLMs in MedQA systems. Our analysis reveals that LLMs fail to respond adequately to these queries, generating erroneous medical information, unsafe recommendations, and content that may be considered offensive.

{{</citation>}}


### (70/117) Breaking the Bank with ChatGPT: Few-Shot Text Classification for Finance (Lefteris Loukas et al., 2023)

{{<citation>}}

Lefteris Loukas, Ilias Stogiannidis, Prodromos Malakasiotis, Stavros Vassos. (2023)  
**Breaking the Bank with ChatGPT: Few-Shot Text Classification for Finance**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL, q-fin-CP  
Keywords: ChatGPT, Few-Shot, GPT, GPT-3.5, GPT-4, Text Classification  
[Paper Link](http://arxiv.org/abs/2308.14634v1)  

---


**ABSTRACT**  
We propose the use of conversational GPT models for easy and quick few-shot text classification in the financial domain using the Banking77 dataset. Our approach involves in-context learning with GPT-3.5 and GPT-4, which minimizes the technical expertise required and eliminates the need for expensive GPU computing while yielding quick and accurate results. Additionally, we fine-tune other pre-trained, masked language models with SetFit, a recent contrastive learning technique, to achieve state-of-the-art results both in full-data and few-shot settings. Our findings show that querying GPT-3.5 and GPT-4 can outperform fine-tuned, non-generative models even with fewer examples. However, subscription fees associated with these solutions may be considered costly for small organizations. Lastly, we find that generative models perform better on the given task when shown representative samples selected by a human expert rather than when shown random ones. We conclude that a) our proposed methods offer a practical solution for few-shot tasks in datasets with limited label availability, and b) our state-of-the-art results can inspire future work in the area.

{{</citation>}}


### (71/117) Spoken Language Intelligence of Large Language Models for Language Learning (Linkai Peng et al., 2023)

{{<citation>}}

Linkai Peng, Baorian Nuchged, Yingming Gao. (2023)  
**Spoken Language Intelligence of Large Language Models for Language Learning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs-SD, cs.CL, eess-AS  
Keywords: AI, GPT, GPT-3.5, Google, LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2308.14536v1)  

---


**ABSTRACT**  
People have long hoped for a conversational system that can assist in real-life situations, and recent progress on large language models (LLMs) is bringing this idea closer to reality. While LLMs are often impressive in performance, their efficacy in real-world scenarios that demand expert knowledge remains unclear. LLMs are believed to hold the most potential and value in education, especially in the development of Artificial intelligence (AI) based virtual teachers capable of facilitating language learning. Our focus is centered on evaluating the efficacy of LLMs in the realm of education, specifically in the areas of spoken language learning which encompass phonetics, phonology, and second language acquisition. We introduce a new multiple-choice question dataset to evaluate the effectiveness of LLMs in the aforementioned scenarios, including understanding and application of spoken language knowledge. In addition, we investigate the influence of various prompting techniques such as zero- and few-shot method (prepending the question with question-answer exemplars), chain-of-thought (CoT, think step-by-step), in-domain exampler and external tools (Google, Wikipedia). We conducted large-scale evaluation on popular LLMs (20 distinct models) using these methods. We achieved significant performance improvements compared to the zero-shot baseline in the practical questions reasoning (GPT-3.5, 49.1% -> 63.1%; LLaMA2-70B-Chat, 42.2% -> 48.6%). We found that models of different sizes have good understanding of concepts in phonetics, phonology, and second language acquisition, but show limitations in reasoning for real-world problems. Additionally, we also explore preliminary findings on conversational communication.

{{</citation>}}


### (72/117) A Multi-Task Semantic Decomposition Framework with Task-specific Pre-training for Few-Shot NER (Guanting Dong et al., 2023)

{{<citation>}}

Guanting Dong, Zechen Wang, Jinxu Zhao, Gang Zhao, Daichi Guo, Dayuan Fu, Tingfeng Hui, Chen Zeng, Keqing He, Xuefeng Li, Liwen Wang, Xinyue Cui, Weiran Xu. (2023)  
**A Multi-Task Semantic Decomposition Framework with Task-specific Pre-training for Few-Shot NER**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Few-Shot, Language Model, NER  
[Paper Link](http://arxiv.org/abs/2308.14533v1)  

---


**ABSTRACT**  
The objective of few-shot named entity recognition is to identify named entities with limited labeled instances. Previous works have primarily focused on optimizing the traditional token-wise classification framework, while neglecting the exploration of information based on NER data characteristics. To address this issue, we propose a Multi-Task Semantic Decomposition Framework via Joint Task-specific Pre-training (MSDP) for few-shot NER. Drawing inspiration from demonstration-based and contrastive learning, we introduce two novel pre-training tasks: Demonstration-based Masked Language Modeling (MLM) and Class Contrastive Discrimination. These tasks effectively incorporate entity boundary information and enhance entity representation in Pre-trained Language Models (PLMs). In the downstream main task, we introduce a multi-task joint optimization framework with the semantic decomposing method, which facilitates the model to integrate two different semantic information for entity classification. Experimental results of two few-shot NER benchmarks demonstrate that MSDP consistently outperforms strong baselines by a large margin. Extensive analyses validate the effectiveness and generalization of MSDP.

{{</citation>}}


### (73/117) LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding (Yushi Bai et al., 2023)

{{<citation>}}

Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang, Juanzi Li. (2023)  
**LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-3.5, QA  
[Paper Link](http://arxiv.org/abs/2308.14508v1)  

---


**ABSTRACT**  
Although large language models (LLMs) demonstrate impressive performance for many language tasks, most of them can only handle texts a few thousand tokens long, limiting their applications on longer sequence inputs, such as books, reports, and codebases. Recent works have proposed methods to improve LLMs' long context capabilities by extending context windows and more sophisticated memory mechanisms. However, comprehensive benchmarks tailored for evaluating long context understanding are lacking. In this paper, we introduce LongBench, the first bilingual, multi-task benchmark for long context understanding, enabling a more rigorous evaluation of long context understanding. LongBench comprises 21 datasets across 6 task categories in both English and Chinese, with an average length of 6,711 words (English) and 13,386 characters (Chinese). These tasks cover key long-text application areas including single-doc QA, multi-doc QA, summarization, few-shot learning, synthetic tasks, and code completion. All datasets in LongBench are standardized into a unified format, allowing for effortless automatic evaluation of LLMs. Upon comprehensive evaluation of 8 LLMs on LongBench, we find that: (1) Commercial model (GPT-3.5-Turbo-16k) outperforms other open-sourced models, but still struggles on longer contexts. (2) Scaled position embedding and fine-tuning on longer sequences lead to substantial improvement on long context understanding. (3) Context compression technique such as retrieval brings improvement for model with weak ability on long contexts, but the performance still lags behind models that have strong long context understanding capability. The code and datasets are available at https://github.com/THUDM/LongBench.

{{</citation>}}


### (74/117) Multimodal Detection of Social Spambots in Twitter using Transformers (Loukas Ilias et al., 2023)

{{<citation>}}

Loukas Ilias, Ioannis Michail Kazelidis, Dimitris Askounis. (2023)  
**Multimodal Detection of Social Spambots in Twitter using Transformers**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, LSTM, Transformer, Transformers, Twitter  
[Paper Link](http://arxiv.org/abs/2308.14484v1)  

---


**ABSTRACT**  
Although not all bots are malicious, the vast majority of them are responsible for spreading misinformation and manipulating the public opinion about several issues, i.e., elections and many more. Therefore, the early detection of social spambots is crucial. Although there have been proposed methods for detecting bots in social media, there are still substantial limitations. For instance, existing research initiatives still extract a large number of features and train traditional machine learning algorithms or use GloVe embeddings and train LSTMs. However, feature extraction is a tedious procedure demanding domain expertise. Also, language models based on transformers have been proved to be better than LSTMs. Other approaches create large graphs and train graph neural networks requiring in this way many hours for training and access to computational resources. To tackle these limitations, this is the first study employing only the user description field and images of three channels denoting the type and content of tweets posted by the users. Firstly, we create digital DNA sequences, transform them to 3d images, and apply pretrained models of the vision domain, including EfficientNet, AlexNet, VGG16, etc. Next, we propose a multimodal approach, where we use TwHIN-BERT for getting the textual representation of the user description field and employ VGG16 for acquiring the visual representation for the image modality. We propose three different fusion methods, namely concatenation, gated multimodal unit, and crossmodal attention, for fusing the different modalities and compare their performances. Extensive experiments conducted on the Cresci '17 dataset demonstrate valuable advantages of our introduced approaches over state-of-the-art ones reaching Accuracy up to 99.98%.

{{</citation>}}


### (75/117) Bridging the KB-Text Gap: Leveraging Structured Knowledge-aware Pre-training for KBQA (Guanting Dong et al., 2023)

{{<citation>}}

Guanting Dong, Rumei Li, Sirui Wang, Yupeng Zhang, Yunsen Xian, Weiran Xu. (2023)  
**Bridging the KB-Text Gap: Leveraging Structured Knowledge-aware Pre-training for KBQA**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-IR, cs.CL  
Keywords: Language Model, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2308.14436v1)  

---


**ABSTRACT**  
Knowledge Base Question Answering (KBQA) aims to answer natural language questions with factual information such as entities and relations in KBs. However, traditional Pre-trained Language Models (PLMs) are directly pre-trained on large-scale natural language corpus, which poses challenges for them in understanding and representing complex subgraphs in structured KBs. To bridge the gap between texts and structured KBs, we propose a Structured Knowledge-aware Pre-training method (SKP). In the pre-training stage, we introduce two novel structured knowledge-aware tasks, guiding the model to effectively learn the implicit relationship and better representations of complex subgraphs. In downstream KBQA task, we further design an efficient linearization strategy and an interval attention mechanism, which assist the model to better encode complex subgraphs and shield the interference of irrelevant subgraphs during reasoning respectively. Detailed experiments and analyses on WebQSP verify the effectiveness of SKP, especially the significant improvement in subgraph retrieval (+4.08% H@10).

{{</citation>}}


### (76/117) GADePo: Graph-Assisted Declarative Pooling Transformers for Document-Level Relation Extraction (Andrei C. Coman et al., 2023)

{{<citation>}}

Andrei C. Coman, Christos Theodoropoulos, Marie-Francine Moens, James Henderson. (2023)  
**GADePo: Graph-Assisted Declarative Pooling Transformers for Document-Level Relation Extraction**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Relation Extraction, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.14423v1)  

---


**ABSTRACT**  
Document-level relation extraction aims to identify relationships between entities within a document. Current methods rely on text-based encoders and employ various hand-coded pooling heuristics to aggregate information from entity mentions and associated contexts. In this paper, we replace these rigid pooling functions with explicit graph relations by leveraging the intrinsic graph processing capabilities of the Transformer model. We propose a joint text-graph Transformer model, and a graph-assisted declarative pooling (GADePo) specification of the input which provides explicit and high-level instructions for information aggregation. This allows the pooling process to be guided by domain-specific knowledge or desired outcomes but still learned by the Transformer, leading to more flexible and customizable pooling strategies. We extensively evaluate our method across diverse datasets and models, and show that our approach yields promising results that are comparable to those achieved by the hand-coded pooling functions.

{{</citation>}}


### (77/117) ZhuJiu: A Multi-dimensional, Multi-faceted Chinese Benchmark for Large Language Models (Baoli Zhang et al., 2023)

{{<citation>}}

Baoli Zhang, Haining Xie, Pengfan Du, Junhao Chen, Pengfei Cao, Yubo Chen, Shengping Liu, Kang Liu, Jun Zhao. (2023)  
**ZhuJiu: A Multi-dimensional, Multi-faceted Chinese Benchmark for Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.14353v1)  

---


**ABSTRACT**  
The unprecedented performance of large language models (LLMs) requires comprehensive and accurate evaluation. We argue that for LLMs evaluation, benchmarks need to be comprehensive and systematic. To this end, we propose the ZhuJiu benchmark, which has the following strengths: (1) Multi-dimensional ability coverage: We comprehensively evaluate LLMs across 7 ability dimensions covering 51 tasks. Especially, we also propose a new benchmark that focuses on knowledge ability of LLMs. (2) Multi-faceted evaluation methods collaboration: We use 3 different yet complementary evaluation methods to comprehensively evaluate LLMs, which can ensure the authority and accuracy of the evaluation results. (3) Comprehensive Chinese benchmark: ZhuJiu is the pioneering benchmark that fully assesses LLMs in Chinese, while also providing equally robust evaluation abilities in English. (4) Avoiding potential data leakage: To avoid data leakage, we construct evaluation data specifically for 37 tasks. We evaluate 10 current mainstream LLMs and conduct an in-depth discussion and analysis of their results. The ZhuJiu benchmark and open-participation leaderboard are publicly released at http://www.zhujiu-benchmark.com/ and we also provide a demo video at https://youtu.be/qypkJ89L1Ic.

{{</citation>}}


### (78/117) DISC-MedLLM: Bridging General Large Language Models and Real-World Medical Consultation (Zhijie Bao et al., 2023)

{{<citation>}}

Zhijie Bao, Wei Chen, Shengze Xiao, Kuang Ren, Jiaao Wu, Cheng Zhong, Jiajie Peng, Xuanjing Huang, Zhongyu Wei. (2023)  
**DISC-MedLLM: Bridging General Large Language Models and Real-World Medical Consultation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.14346v1)  

---


**ABSTRACT**  
We propose DISC-MedLLM, a comprehensive solution that leverages Large Language Models (LLMs) to provide accurate and truthful medical response in end-to-end conversational healthcare services. To construct high-quality Supervised Fine-Tuning (SFT) datasets, we employ three strategies: utilizing medical knowledge-graphs, reconstructing real-world dialogues, and incorporating human-guided preference rephrasing. These datasets are instrumental in training DISC-MedLLM, surpassing existing medical LLMs in both single-turn and multi-turn consultation scenarios. Extensive experimental results demonstrate the effectiveness of the proposed model in bridging the gap between general language models and real-world medical consultation. Additionally, we release the constructed dataset and model weights to further contribute to research and development. Further details and resources can be found at https://github.com/FudanDISC/DISC-MedLLM

{{</citation>}}


### (79/117) Leveraging A Medical Knowledge Graph into Large Language Models for Diagnosis Prediction (Yanjun Gao et al., 2023)

{{<citation>}}

Yanjun Gao, Ruizhe Li, John Caskey, Dmitriy Dligach, Timothy Miller, Matthew M. Churpek, Majid Afshar. (2023)  
**Leveraging A Medical Knowledge Graph into Large Language Models for Diagnosis Prediction**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, Knowledge Graph, Language Model  
[Paper Link](http://arxiv.org/abs/2308.14321v1)  

---


**ABSTRACT**  
Electronic Health Records (EHRs) and routine documentation practices play a vital role in patients' daily care, providing a holistic record of health, diagnoses, and treatment. However, complex and verbose EHR narratives overload healthcare providers, risking diagnostic inaccuracies. While Large Language Models (LLMs) have showcased their potential in diverse language tasks, their application in the healthcare arena needs to ensure the minimization of diagnostic errors and the prevention of patient harm. In this paper, we outline an innovative approach for augmenting the proficiency of LLMs in the realm of automated diagnosis generation, achieved through the incorporation of a medical knowledge graph (KG) and a novel graph model: Dr.Knows, inspired by the clinical diagnostic reasoning process. We derive the KG from the National Library of Medicine's Unified Medical Language System (UMLS), a robust repository of biomedical knowledge. Our method negates the need for pre-training and instead leverages the KG as an auxiliary instrument aiding in the interpretation and summarization of complex medical concepts. Using real-world hospital datasets, our experimental results demonstrate that the proposed approach of combining LLMs with KG has the potential to improve the accuracy of automated diagnosis generation. More importantly, our approach offers an explainable diagnostic pathway, edging us closer to the realization of AI-augmented diagnostic decision support systems.

{{</citation>}}


### (80/117) Evaluating the Robustness to Instructions of Large Language Models (Yuansheng Ni et al., 2023)

{{<citation>}}

Yuansheng Ni, Sichao Jiang, Xinyu wu, Hui Shen, Yuli Zhou. (2023)  
**Evaluating the Robustness to Instructions of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, QA, T5  
[Paper Link](http://arxiv.org/abs/2308.14306v2)  

---


**ABSTRACT**  
Recently, Instruction fine-tuning has risen to prominence as a potential method for enhancing the zero-shot capabilities of Large Language Models (LLMs) on novel tasks. This technique has shown an exceptional ability to boost the performance of moderately sized LLMs, sometimes even reaching performance levels comparable to those of much larger model variants. The focus is on the robustness of instruction-tuned LLMs to seen and unseen tasks. We conducted an exploration of six models including Alpaca, Vicuna, WizardLM, and Traditional Task-oriented Models(Flan-T5-XL/XXL, T0++) using real-world relation extraction datasets as case studies. We carried out a comprehensive evaluation of these instruction-following LLMs which have been tuned based on open-domain instructions and task-oriented instructions. The main discussion is their performance and robustness towards instructions. We have observed that in most cases, the model's performance in dealing with unfamiliar instructions tends to worsen significantly, and the robustness of the model for RE instructions deteriorates compared to QA. Further, we discovered that up until a certain parameter size threshold (3B), the performance of the FLAN-T5 model improves as the parameter count increases. The robustness of different scales of FLAN-T5 models to RE instruction is worse than the robustness to QA instruction.

{{</citation>}}


### (81/117) FonMTL: Towards Multitask Learning for the Fon Language (Bonaventure F. P. Dossou et al., 2023)

{{<citation>}}

Bonaventure F. P. Dossou, Iffanice Houndayi, Pamely Zantou, Gilles Hacheme. (2023)  
**FonMTL: Towards Multitask Learning for the Fon Language**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: NER, Named Entity Recognition, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2308.14280v1)  

---


**ABSTRACT**  
The Fon language, spoken by an average 2 million of people, is a truly low-resourced African language, with a limited online presence, and existing datasets (just to name but a few). Multitask learning is a learning paradigm that aims to improve the generalization capacity of a model by sharing knowledge across different but related tasks: this could be prevalent in very data-scarce scenarios. In this paper, we present the first explorative approach to multitask learning, for model capabilities enhancement in Natural Language Processing for the Fon language. Specifically, we explore the tasks of Named Entity Recognition (NER) and Part of Speech Tagging (POS) for Fon. We leverage two language model heads as encoders to build shared representations for the inputs, and we use linear layers blocks for classification relative to each task. Our results on the NER and POS tasks for Fon, show competitive (or better) performances compared to several multilingual pretrained language models finetuned on single tasks. Additionally, we perform a few ablation studies to leverage the efficiency of two different loss combination strategies and find out that the equal loss weighting approach works best in our case. Our code is open-sourced at https://github.com/bonaventuredossou/multitask_fon.

{{</citation>}}


### (82/117) Goodhart's Law Applies to NLP's Explanation Benchmarks (Jennifer Hsia et al., 2023)

{{<citation>}}

Jennifer Hsia, Danish Pruthi, Aarti Singh, Zachary C. Lipton. (2023)  
**Goodhart's Law Applies to NLP's Explanation Benchmarks**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2308.14272v1)  

---


**ABSTRACT**  
Despite the rising popularity of saliency-based explanations, the research community remains at an impasse, facing doubts concerning their purpose, efficacy, and tendency to contradict each other. Seeking to unite the community's efforts around common goals, several recent works have proposed evaluation metrics. In this paper, we critically examine two sets of metrics: the ERASER metrics (comprehensiveness and sufficiency) and the EVAL-X metrics, focusing our inquiry on natural language processing. First, we show that we can inflate a model's comprehensiveness and sufficiency scores dramatically without altering its predictions or explanations on in-distribution test inputs. Our strategy exploits the tendency for extracted explanations and their complements to be "out-of-support" relative to each other and in-distribution inputs. Next, we demonstrate that the EVAL-X metrics can be inflated arbitrarily by a simple method that encodes the label, even though EVAL-X is precisely motivated to address such exploits. Our results raise doubts about the ability of current metrics to guide explainability research, underscoring the need for a broader reassessment of what precisely these metrics are intended to capture.

{{</citation>}}


## cond-mat.mtrl-sci (1)



### (83/117) Matbench Discovery -- An evaluation framework for machine learning crystal stability prediction (Janosh Riebesell et al., 2023)

{{<citation>}}

Janosh Riebesell, Rhys E. A. Goodall, Anubhav Jain, Philipp Benner, Kristin A. Persson, Alpha A. Lee. (2023)  
**Matbench Discovery -- An evaluation framework for machine learning crystal stability prediction**  

---
Primary Category: cond-mat.mtrl-sci  
Categories: cond-mat-mtrl-sci, cond-mat.mtrl-sci, cs-LG  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2308.14920v1)  

---


**ABSTRACT**  
Matbench Discovery simulates the deployment of machine learning (ML) energy models in a high-throughput search for stable inorganic crystals. We address the disconnect between (i) thermodynamic stability and formation energy and (ii) in-domain vs out-of-distribution performance. Alongside this paper, we publish a Python package to aid with future model submissions and a growing online leaderboard with further insights into trade-offs between various performance metrics. To answer the question which ML methodology performs best at materials discovery, our initial release explores a variety of models including random forests, graph neural networks (GNN), one-shot predictors, iterative Bayesian optimizers and universal interatomic potentials (UIP). Ranked best-to-worst by their test set F1 score on thermodynamic stability prediction, we find CHGNet > M3GNet > MACE > ALIGNN > MEGNet > CGCNN > CGCNN+P > Wrenformer > BOWSR > Voronoi tessellation fingerprints with random forest. The top 3 models are UIPs, the winning methodology for ML-guided materials discovery, achieving F1 scores of ~0.6 for crystal stability classification and discovery acceleration factors (DAF) of up to 5x on the first 10k most stable predictions compared to dummy selection from our test set. We also highlight a sharp disconnect between commonly used global regression metrics and more task-relevant classification metrics. Accurate regressors are susceptible to unexpectedly high false-positive rates if those accurate predictions lie close to the decision boundary at 0 eV/atom above the convex hull where most materials are. Our results highlight the need to focus on classification metrics that actually correlate with improved stability hit rate.

{{</citation>}}


## cs.SD (3)



### (84/117) Pruning Self-Attention for Zero-Shot Multi-Speaker Text-to-Speech (Hyungchan Yoon et al., 2023)

{{<citation>}}

Hyungchan Yoon, Changhwan Kim, Eunwoo Song, Hyun-Wook Yoon, Hong-Goo Kang. (2023)  
**Pruning Self-Attention for Zero-Shot Multi-Speaker Text-to-Speech**  

---
Primary Category: cs.SD  
Categories: cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Attention, Pruning, Self-Attention, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2308.14909v1)  

---


**ABSTRACT**  
For personalized speech generation, a neural text-to-speech (TTS) model must be successfully implemented with limited data from a target speaker. To this end, the baseline TTS model needs to be amply generalized to out-of-domain data (i.e., target speaker's speech). However, approaches to address this out-of-domain generalization problem in TTS have yet to be thoroughly studied. In this work, we propose an effective pruning method for a transformer known as sparse attention, to improve the TTS model's generalization abilities. In particular, we prune off redundant connections from self-attention layers whose attention weights are below the threshold. To flexibly determine the pruning strength for searching optimal degree of generalization, we also propose a new differentiable pruning method that allows the model to automatically learn the thresholds. Evaluations on zero-shot multi-speaker TTS verify the effectiveness of our method in terms of voice quality and speaker similarity.

{{</citation>}}


### (85/117) Time-Frequency Transformer: A Novel Time Frequency Joint Learning Method for Speech Emotion Recognition (Yong Wang et al., 2023)

{{<citation>}}

Yong Wang, Cheng Lu, Yuan Zong, Hailun Lian, Yan Zhao, Sunan Li. (2023)  
**Time-Frequency Transformer: A Novel Time Frequency Joint Learning Method for Speech Emotion Recognition**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Emotion Recognition, Transformer  
[Paper Link](http://arxiv.org/abs/2308.14568v1)  

---


**ABSTRACT**  
In this paper, we propose a novel time-frequency joint learning method for speech emotion recognition, called Time-Frequency Transformer. Its advantage is that the Time-Frequency Transformer can excavate global emotion patterns in the time-frequency domain of speech signal while modeling the local emotional correlations in the time domain and frequency domain respectively. For the purpose, we first design a Time Transformer and Frequency Transformer to capture the local emotion patterns between frames and inside frequency bands respectively, so as to ensure the integrity of the emotion information modeling in both time and frequency domains. Then, a Time-Frequency Transformer is proposed to mine the time-frequency emotional correlations through the local time-domain and frequency-domain emotion features for learning more discriminative global speech emotion representation. The whole process is a time-frequency joint learning process implemented by a series of Transformer models. Experiments on IEMOCAP and CASIA databases indicate that our proposed method outdoes the state-of-the-art methods.

{{</citation>}}


### (86/117) Symbolic & Acoustic: Multi-domain Music Emotion Modeling for Instrumental Music (Kexin Zhu et al., 2023)

{{<citation>}}

Kexin Zhu, Xulong Zhang, Jianzong Wang, Ning Cheng, Jing Xiao. (2023)  
**Symbolic & Acoustic: Multi-domain Music Emotion Modeling for Instrumental Music**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Emotion Recognition, Information Retrieval  
[Paper Link](http://arxiv.org/abs/2308.14317v1)  

---


**ABSTRACT**  
Music Emotion Recognition involves the automatic identification of emotional elements within music tracks, and it has garnered significant attention due to its broad applicability in the field of Music Information Retrieval. It can also be used as the upstream task of many other human-related tasks such as emotional music generation and music recommendation. Due to existing psychology research, music emotion is determined by multiple factors such as the Timbre, Velocity, and Structure of the music. Incorporating multiple factors in MER helps achieve more interpretable and finer-grained methods. However, most prior works were uni-domain and showed weak consistency between arousal modeling performance and valence modeling performance. Based on this background, we designed a multi-domain emotion modeling method for instrumental music that combines symbolic analysis and acoustic analysis. At the same time, because of the rarity of music data and the difficulty of labeling, our multi-domain approach can make full use of limited data. Our approach was implemented and assessed using the publicly available piano dataset EMOPIA, resulting in a notable improvement over our baseline model with a 2.4% increase in overall accuracy, establishing its state-of-the-art performance.

{{</citation>}}


## cs.HC (4)



### (87/117) Trust in Construction AI-Powered Collaborative Robots: A Qualitative Empirical Analysis (Newsha Emaminejad et al., 2023)

{{<citation>}}

Newsha Emaminejad, Reza Akhavian, Ph. D. (2023)  
**Trust in Construction AI-Powered Collaborative Robots: A Qualitative Empirical Analysis**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs-RO, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.14846v1)  

---


**ABSTRACT**  
Construction technology researchers and forward-thinking companies are experimenting with collaborative robots (aka cobots), powered by artificial intelligence (AI), to explore various automation scenarios as part of the digital transformation of the industry. Intelligent cobots are expected to be the dominant type of robots in the future of work in construction. However, the black-box nature of AI-powered cobots and unknown technical and psychological aspects of introducing them to job sites are precursors to trust challenges. By analyzing the results of semi-structured interviews with construction practitioners using grounded theory, this paper investigates the characteristics of trustworthy AI-powered cobots in construction. The study found that while the key trust factors identified in a systematic literature review -- conducted previously by the authors -- resonated with the field experts and end users, other factors such as financial considerations and the uncertainty associated with change were also significant barriers against trusting AI-powered cobots in construction.

{{</citation>}}


### (88/117) Assessing Trust in Construction AI-Powered Collaborative Robots using Structural Equation Modeling (Newsha Emaminejad et al., 2023)

{{<citation>}}

Newsha Emaminejad, Lisa Kath, Reza Akhavian. (2023)  
**Assessing Trust in Construction AI-Powered Collaborative Robots using Structural Equation Modeling**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs-RO, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.14697v1)  

---


**ABSTRACT**  
This study aimed to investigate the key technical and psychological factors that impact the architecture, engineering, and construction (AEC) professionals' trust in collaborative robots (cobots) powered by artificial intelligence (AI). The study employed a nationwide survey of 600 AEC industry practitioners to gather in-depth responses and valuable insights into the future opportunities for promoting the adoption, cultivation, and training of a skilled workforce to leverage this technology effectively. A Structural Equation Modeling (SEM) analysis revealed that safety and reliability are significant factors for the adoption of AI-powered cobots in construction. Fear of being replaced resulting from the use of cobots can have a substantial effect on the mental health of the affected workers. A lower error rate in jobs involving cobots, safety measurements, and security of data collected by cobots from jobsites significantly impact reliability, while the transparency of cobots' inner workings can benefit accuracy, robustness, security, privacy, and communication, and results in higher levels of automation, all of which demonstrated as contributors to trust. The study's findings provide critical insights into the perceptions and experiences of AEC professionals towards adoption of cobots in construction and help project teams determine the adoption approach that aligns with the company's goals workers' welfare.

{{</citation>}}


### (89/117) Skip, Skip, Skip, Accept!!!: A Study on the Usability of Smartphone Manufacturer Provided Default Features and User Privacy (Kopo M. Ramokapane et al., 2023)

{{<citation>}}

Kopo M. Ramokapane, Anthony C. Mazeli, Awais Rashid. (2023)  
**Skip, Skip, Skip, Accept!!!: A Study on the Usability of Smartphone Manufacturer Provided Default Features and User Privacy**  

---
Primary Category: cs.HC  
Categories: J-4, cs-HC, cs.HC  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2308.14593v1)  

---


**ABSTRACT**  
Smartphone manufacturer provided default features (e.g., default location services, iCloud, Google Assistant, ad tracking) enhance the usability and extend the functionality of these devices. Prior studies have highlighted smartphone vulnerabilities and how users' data can be harvested without their knowledge. However, little is known about manufacturer provided default features in this regard -- their usability concerning configuring them during usage, and how users perceive them with regards to privacy. To bridge this gap, we conducted a task-based study with 27 Android and iOS smartphone users in order to learn about their perceptions, concerns and practices, and to understand the usability of these features with regards to privacy. We explored the following: users' awareness of these features, why and when do they change the settings of these features, the challenges they face while configuring these features, and finally the mitigation strategies they adopt. Our findings reveal that users of both platforms have limited awareness of these features and their privacy implications. Awareness of these features does not imply that a user can easily locate and adjust them when needed.   Furthermore, users attribute their failure to configure default features to hidden controls and insufficient knowledge on how to configure them. To cope with difficulties of finding controls, users employ various coping strategies, some of which are platform specific but most often applicable to both platforms. However, some of these coping strategies leave users vulnerable.

{{</citation>}}


### (90/117) Video Multimodal Emotion Recognition System for Real World Applications (Sun-Kyung Lee et al., 2023)

{{<citation>}}

Sun-Kyung Lee, Jong-Hwan Kim. (2023)  
**Video Multimodal Emotion Recognition System for Real World Applications**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI, Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2308.14320v1)  

---


**ABSTRACT**  
This paper proposes a system capable of recognizing a speaker's utterance-level emotion through multimodal cues in a video. The system seamlessly integrates multiple AI models to first extract and pre-process multimodal information from the raw video input. Next, an end-to-end MER model sequentially predicts the speaker's emotions at the utterance level. Additionally, users can interactively demonstrate the system through the implemented interface.

{{</citation>}}


## cs.CR (4)



### (91/117) AI ATAC 1: An Evaluation of Prominent Commercial Malware Detectors (Robert A. Bridges et al., 2023)

{{<citation>}}

Robert A. Bridges, Brian Weber, Justin M. Beaver, Jared M. Smith, Miki E. Verma, Savannah Norem, Kevin Spakes, Cory Watson, Jeff A. Nichols, Brian Jewell, Michael. D. Iannacone, Chelsey Dunivan Stahl, Kelly M. T. Huffer, T. Sean Oesch. (2023)  
**AI ATAC 1: An Evaluation of Prominent Commercial Malware Detectors**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.14835v1)  

---


**ABSTRACT**  
This work presents an evaluation of six prominent commercial endpoint malware detectors, a network malware detector, and a file-conviction algorithm from a cyber technology vendor. The evaluation was administered as the first of the Artificial Intelligence Applications to Autonomous Cybersecurity (AI ATAC) prize challenges, funded by / completed in service of the US Navy. The experiment employed 100K files (50/50% benign/malicious) with a stratified distribution of file types, including ~1K zero-day program executables (increasing experiment size two orders of magnitude over previous work). We present an evaluation process of delivering a file to a fresh virtual machine donning the detection technology, waiting 90s to allow static detection, then executing the file and waiting another period for dynamic detection; this allows greater fidelity in the observational data than previous experiments, in particular, resource and time-to-detection statistics. To execute all 800K trials (100K files $\times$ 8 tools), a software framework is designed to choreographed the experiment into a completely automated, time-synced, and reproducible workflow with substantial parallelization. A cost-benefit model was configured to integrate the tools' recall, precision, time to detection, and resource requirements into a single comparable quantity by simulating costs of use. This provides a ranking methodology for cyber competitions and a lens through which to reason about the varied statistical viewpoints of the results. These statistical and cost-model results provide insights on state of commercial malware detection.

{{</citation>}}


### (92/117) Advancement on Security Applications of Private Intersection Sum Protocol (Yuvaray Athur Raghuvir et al., 2023)

{{<citation>}}

Yuvaray Athur Raghuvir, Senthil Govindarajan, Sanjeevi Vijayakumar, Pradeep Yadlapalli, Fabio Di Troia. (2023)  
**Advancement on Security Applications of Private Intersection Sum Protocol**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-NI, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2308.14741v1)  

---


**ABSTRACT**  
Secure computation protocols combine inputs from involved parties to generate an output while keeping their inputs private. Private Set Intersection (PSI) is a secure computation protocol that allows two parties, who each hold a set of items, to learn the intersection of their sets without revealing anything else about the items. Private Intersection Sum (PIS) extends PSI when the two parties want to learn the cardinality of the intersection, as well as the sum of the associated integer values for each identifier in the intersection, but nothing more. Finally, Private Join and Compute (PJC) is a scalable extension of PIS protocol to help organizations work together with confidential data sets. The extensions proposed in this paper include: (a) extending PJC protocol to additional data columns and applying columnar aggregation based on supported homomorphic operations, (b) exploring Ring Learning with Errors (RLWE) homomorphic encryption schemes to apply arithmetic operations such as sum and sum of squares, (c) ensuring stronger security using mutual authentication of communicating parties using certificates, and (d) developing a Website to operationalize such a service offering. We applied our results to develop a Proof-of-Concept solution called JingBing, a voter list validation service that allows different states to register, acquire secure communication modules, install it, and then conduct authenticated peer-to-peer communication. We conclude our paper with directions for future research to make such a solution scalable for practical real-life scenarios.

{{</citation>}}


### (93/117) Using ChatGPT as a Static Application Security Testing Tool (Atieh Bakhshandeh et al., 2023)

{{<citation>}}

Atieh Bakhshandeh, Abdalsamad Keramatfar, Amir Norouzi, Mohammad Mahdi Chekidehkhoun. (2023)  
**Using ChatGPT as a Static Application Security Testing Tool**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: AI, ChatGPT, GPT, Security  
[Paper Link](http://arxiv.org/abs/2308.14434v1)  

---


**ABSTRACT**  
In recent years, artificial intelligence has had a conspicuous growth in almost every aspect of life. One of the most applicable areas is security code review, in which a lot of AI-based tools and approaches have been proposed. Recently, ChatGPT has caught a huge amount of attention with its remarkable performance in following instructions and providing a detailed response. Regarding the similarities between natural language and code, in this paper, we study the feasibility of using ChatGPT for vulnerability detection in Python source code. Toward this goal, we feed an appropriate prompt along with vulnerable data to ChatGPT and compare its results on two datasets with the results of three widely used Static Application Security Testing tools (Bandit, Semgrep and SonarQube). We implement different kinds of experiments with ChatGPT and the results indicate that ChatGPT reduces the false positive and false negative rates and has the potential to be used for Python source code vulnerability detection.

{{</citation>}}


### (94/117) A Comprehensive Overview of Backdoor Attacks in Large Language Models within Communication Networks (Haomiao Yang et al., 2023)

{{<citation>}}

Haomiao Yang, Kunlan Xiang, Hongwei Li, Rongxing Lu. (2023)  
**A Comprehensive Overview of Backdoor Attacks in Large Language Models within Communication Networks**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.14367v1)  

---


**ABSTRACT**  
The Large Language Models (LLMs) are becoming an integral part of modern communication networks due to their superior proficiency in language comprehension and generation. In the context of these networks, where limited data and computing resources often necessitate the use of third-party data and computing resources, the risk of backdoor attacks becomes highly significant. Such strategies may expose the model within the network to maliciously manipulated training data and processing, providing an opportunity for attackers to embed a hidden backdoor into the model, termed a backdoor attack. Backdoor attack in LLMs refers to embedding a hidden backdoor in LLMs that causes the model to perform normally on benign samples but exhibit degraded performance on poisoned ones. This issue is particularly concerning within communication networks where reliability and security are paramount. Despite the extensive research on backdoor attacks, there remains a lack of in-depth exploration specifically within the context of LLMs employed in communication networks, and a systematic review of such attacks is currently absent. In this survey, we systematically propose a taxonomy of backdoor attacks in LLMs as used in communication networks, dividing them into four major categories: input-triggered, prompt-triggered, instruction-triggered, and demonstration-triggered attacks. Furthermore, we conduct a comprehensive analysis of the benchmark datasets within the network domain. Finally, we identify potential problems and open challenges, offering valuable insights into future research directions for enhancing the security and integrity of LLMs in communication networks.

{{</citation>}}


## eess.AS (4)



### (95/117) Unsupervised Active Learning: Optimizing Labeling Cost-Effectiveness for Automatic Speech Recognition (Zhisheng Zheng et al., 2023)

{{<citation>}}

Zhisheng Zheng, Ziyang Ma, Yu Wang, Xie Chen. (2023)  
**Unsupervised Active Learning: Optimizing Labeling Cost-Effectiveness for Automatic Speech Recognition**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Active Learning, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2308.14814v1)  

---


**ABSTRACT**  
In recent years, speech-based self-supervised learning (SSL) has made significant progress in various tasks, including automatic speech recognition (ASR). An ASR model with decent performance can be realized by fine-tuning an SSL model with a small fraction of labeled data. Reducing the demand for labeled data is always of great practical value. In this paper, we further extend the use of SSL to cut down labeling costs with active learning. Three types of units on different granularities are derived from speech signals in an unsupervised way, and their effects are compared by applying a contrastive data selection method. The experimental results show that our proposed data selection framework can effectively improve the word error rate (WER) by more than 11% with the same amount of labeled data, or halve the labeling cost while maintaining the same WER, compared to random selection.

{{</citation>}}


### (96/117) The USTC-NERCSLIP Systems for the CHiME-7 DASR Challenge (Ruoyu Wang et al., 2023)

{{<citation>}}

Ruoyu Wang, Maokui He, Jun Du, Hengshun Zhou, Shutong Niu, Hang Chen, Yanyan Yue, Gaobin Yang, Shilong Wu, Lei Sun, Yanhui Tu, Haitao Tang, Shuangqing Qian, Tian Gao, Mengzhi Wang, Genshun Wan, Jia Pan, Jianqing Gao, Chin-Hui Lee. (2023)  
**The USTC-NERCSLIP Systems for the CHiME-7 DASR Challenge**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: NER  
[Paper Link](http://arxiv.org/abs/2308.14638v1)  

---


**ABSTRACT**  
This technical report details our submission system to the CHiME-7 DASR Challenge, which focuses on speaker diarization and speech recognition under complex multi-speaker settings. Additionally, it also evaluates the efficiency of systems in handling diverse array devices. To address these issues, we implemented an end-to-end speaker diarization system and introduced a rectification strategy based on multi-channel spatial information. This approach significantly diminished the word error rates (WER). In terms of recognition, we utilized publicly available pre-trained models as the foundational models to train our end-to-end speech recognition models. Our system attained a macro-averaged diarization-attributed WER (DA-WER) of 22.4\% on the CHiME-7 development set, which signifies a relative improvement of 52.5\% over the official baseline system.

{{</citation>}}


### (97/117) Speech Self-Supervised Representations Benchmarking: a Case for Larger Probing Heads (Salah Zaiem et al., 2023)

{{<citation>}}

Salah Zaiem, Youcef Kemiche, Titouan Parcollet, Slim Essid, Mirco Ravanelli. (2023)  
**Speech Self-Supervised Representations Benchmarking: a Case for Larger Probing Heads**  

---
Primary Category: eess.AS  
Categories: cs-LG, cs-SD, eess-AS, eess-SP, eess.AS  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2308.14456v1)  

---


**ABSTRACT**  
Self-supervised learning (SSL) leverages large datasets of unlabeled speech to reach impressive performance with reduced amounts of annotated data. The high number of proposed approaches fostered the emergence of comprehensive benchmarks that evaluate their performance on a set of downstream tasks exploring various aspects of the speech signal. However, while the number of considered tasks has been growing, most proposals rely upon a single downstream architecture that maps the frozen SSL representations to the task labels. This study examines how benchmarking results are affected by changes in the probing head architecture. Interestingly, we found that altering the downstream architecture structure leads to significant fluctuations in the performance ranking of the evaluated models. Against common practices in speech SSL benchmarking, we evaluate larger-capacity probing heads, showing their impact on performance, inference costs, generalization and multi-level feature exploitation.

{{</citation>}}


### (98/117) TextrolSpeech: A Text Style Control Speech Corpus With Codec Language Text-to-Speech Models (Shengpeng Ji et al., 2023)

{{<citation>}}

Shengpeng Ji, Jialong Zuo, Minghui Fang, Ziyue Jiang, Feiyang Chen, Xinyu Duan, Baoxing Huai, Zhou Zhao. (2023)  
**TextrolSpeech: A Text Style Control Speech Corpus With Codec Language Text-to-Speech Models**  

---
Primary Category: eess.AS  
Categories: cs-LG, cs-SD, eess-AS, eess.AS  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2308.14430v1)  

---


**ABSTRACT**  
Recently, there has been a growing interest in the field of controllable Text-to-Speech (TTS). While previous studies have relied on users providing specific style factor values based on acoustic knowledge or selecting reference speeches that meet certain requirements, generating speech solely from natural text prompts has emerged as a new challenge for researchers. This challenge arises due to the scarcity of high-quality speech datasets with natural text style prompt and the absence of advanced text-controllable TTS models. In light of this, 1) we propose TextrolSpeech, which is the first large-scale speech emotion dataset annotated with rich text attributes. The dataset comprises 236,220 pairs of style prompt in natural text descriptions with five style factors and corresponding speech samples. Through iterative experimentation, we introduce a multi-stage prompt programming approach that effectively utilizes the GPT model for generating natural style descriptions in large volumes. 2) Furthermore, to address the need for generating audio with greater style diversity, we propose an efficient architecture called Salle. This architecture treats text controllable TTS as a language model task, utilizing audio codec codes as an intermediate representation to replace the conventional mel-spectrogram. Finally, we successfully demonstrate the ability of the proposed model by showing a comparable performance in the controllable TTS task. Audio samples are available at https://sall-e.github.io/

{{</citation>}}


## cs.CY (4)



### (99/117) Domain-based user embedding for competing events on social media (Wentao Xu et al., 2023)

{{<citation>}}

Wentao Xu, Kazutoshi Sasahara. (2023)  
**Domain-based user embedding for competing events on social media**  

---
Primary Category: cs.CY  
Categories: 94-08, J-4, cs-CY, cs-SI, cs.CY  
Keywords: QA, Twitter  
[Paper Link](http://arxiv.org/abs/2308.14806v2)  

---


**ABSTRACT**  
Online social networks offer vast opportunities for computational social science, but effective user embedding is crucial for downstream tasks. Traditionally, researchers have used pre-defined network-based user features, such as degree, and centrality measures, and/or content-based features, such as posts and reposts. However, these measures may not capture the complex characteristics of social media users. In this study, we propose a user embedding method based on the URL domain co-occurrence network, which is simple but effective for representing social media users in competing events. We assessed the performance of this method in binary classification tasks using benchmark datasets that included Twitter users related to COVID-19 infodemic topics (QAnon, Biden, Ivermectin). Our results revealed that user embeddings generated directly from the retweet network, and those based on language, performed below expectations. In contrast, our domain-based embeddings outperformed these methods while reducing computation time. These findings suggest that the domain-based user embedding can serve as an effective tool to characterize social media users participating in competing events, such as political campaigns and public health crises.

{{</citation>}}


### (100/117) AI Deception: A Survey of Examples, Risks, and Potential Solutions (Peter S. Park et al., 2023)

{{<citation>}}

Peter S. Park, Simon Goldstein, Aidan O'Gara, Michael Chen, Dan Hendrycks. (2023)  
**AI Deception: A Survey of Examples, Risks, and Potential Solutions**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CY, cs-HC, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.14752v1)  

---


**ABSTRACT**  
This paper argues that a range of current AI systems have learned how to deceive humans. We define deception as the systematic inducement of false beliefs in the pursuit of some outcome other than the truth. We first survey empirical examples of AI deception, discussing both special-use AI systems (including Meta's CICERO) built for specific competitive situations, and general-purpose AI systems (such as large language models). Next, we detail several risks from AI deception, such as fraud, election tampering, and losing control of AI systems. Finally, we outline several potential solutions to the problems posed by AI deception: first, regulatory frameworks should subject AI systems that are capable of deception to robust risk-assessment requirements; second, policymakers should implement bot-or-not laws; and finally, policymakers should prioritize the funding of relevant research, including tools to detect AI deception and to make AI systems less deceptive. Policymakers, researchers, and the broader public should work proactively to prevent AI deception from destabilizing the shared foundations of our society.

{{</citation>}}


### (101/117) Helping Fact-Checkers Identify Fake News Stories Shared through Images on WhatsApp (Julio C. S. Reis et al., 2023)

{{<citation>}}

Julio C. S. Reis, Philipe Melo, Fabiano Belém, Fabricio Murai, Jussara M. Almeida, Fabricio Benevenuto. (2023)  
**Helping Fact-Checkers Identify Fake News Stories Shared through Images on WhatsApp**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: Fake News  
[Paper Link](http://arxiv.org/abs/2308.14782v1)  

---


**ABSTRACT**  
WhatsApp has introduced a novel avenue for smartphone users to engage with and disseminate news stories. The convenience of forming interest-based groups and seamlessly sharing content has rendered WhatsApp susceptible to the exploitation of misinformation campaigns. While the process of fact-checking remains a potent tool in identifying fabricated news, its efficacy falters in the face of the unprecedented deluge of information generated on the Internet today. In this work, we explore automatic ranking-based strategies to propose a "fakeness score" model as a means to help fact-checking agencies identify fake news stories shared through images on WhatsApp. Based on the results, we design a tool and integrate it into a real system that has been used extensively for monitoring content during the 2018 Brazilian general election. Our experimental evaluation shows that this tool can reduce by up to 40% the amount of effort required to identify 80% of the fake news in the data when compared to current mechanisms practiced by the fact-checking agencies for the selection of news stories to be checked.

{{</citation>}}


### (102/117) Fairness Through Domain Awareness: Mitigating Popularity Bias For Music Discovery (Rebecca Salganik et al., 2023)

{{<citation>}}

Rebecca Salganik, Fernando Diaz, Golnoosh Farnadi. (2023)  
**Fairness Through Domain Awareness: Mitigating Popularity Bias For Music Discovery**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs-IR, cs-LG, cs.CY  
Keywords: Bias, GNN  
[Paper Link](http://arxiv.org/abs/2308.14601v1)  

---


**ABSTRACT**  
As online music platforms grow, music recommender systems play a vital role in helping users navigate and discover content within their vast musical databases. At odds with this larger goal, is the presence of popularity bias, which causes algorithmic systems to favor mainstream content over, potentially more relevant, but niche items. In this work we explore the intrinsic relationship between music discovery and popularity bias. To mitigate this issue we propose a domain-aware, individual fairness-based approach which addresses popularity bias in graph neural network (GNNs) based recommender systems. Our approach uses individual fairness to reflect a ground truth listening experience, i.e., if two songs sound similar, this similarity should be reflected in their representations. In doing so, we facilitate meaningful music discovery that is robust to popularity bias and grounded in the music domain. We apply our BOOST methodology to two discovery based tasks, performing recommendations at both the playlist level and user level. Then, we ground our evaluation in the cold start setting, showing that our approach outperforms existing fairness benchmarks in both performance and recommendation of lesser-known content. Finally, our analysis explains why our proposed methodology is a novel and promising approach to mitigating popularity bias and improving the discovery of new and niche content in music recommender systems.

{{</citation>}}


## cs.SE (2)



### (103/117) Distilled GPT for Source Code Summarization (Chia-Yi Su et al., 2023)

{{<citation>}}

Chia-Yi Su, Collin McMillan. (2023)  
**Distilled GPT for Source Code Summarization**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: ChatGPT, GPT, GPT-3.5, Language Model, Summarization  
[Paper Link](http://arxiv.org/abs/2308.14731v1)  

---


**ABSTRACT**  
A code summary is a brief natural language description of source code. Summaries are usually only a single sentence long, and yet form the backbone of developer documentation. A short descriptions such as "changes all visible polygons to the color blue" can give a programmer a high-level idea of what code does without the effort of reading the code itself. Recently, products based on Large Language Models such as ChatGPT have demonstrated a strong ability to write these descriptions automatically. However, to use these tools, programmers must send their code to untrusted third parties for processing (e.g., via an API call). This loss of custody is not acceptable to many organizations. In this paper, we present an alternative: we train an open source model using sample output generated by GPT-3.5 in a process related to knowledge distillation. Our model is small enough (350m parameters) to be run on a single 16gb GPU, yet we show in our evaluation that it is large enough to mimic GPT-3.5 on this task.

{{</citation>}}


### (104/117) STEAM: Simulating the InTeractive BEhavior of ProgrAMmers for Automatic Bug Fixing (Yuwei Zhang et al., 2023)

{{<citation>}}

Yuwei Zhang, Zhi Jin, Ying Xing, Ge Li. (2023)  
**STEAM: Simulating the InTeractive BEhavior of ProgrAMmers for Automatic Bug Fixing**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2308.14460v1)  

---


**ABSTRACT**  
Bug fixing holds significant importance in software development and maintenance. Recent research has made notable progress in exploring the potential of large language models (LLMs) for automatic bug fixing. However, existing studies often overlook the collaborative nature of bug resolution, treating it as a single-stage process. To overcome this limitation, we introduce a novel stage-wise framework named STEAM in this paper. The objective of STEAM is to simulate the interactive behavior of multiple programmers involved in various stages across the bug's life cycle. Taking inspiration from bug management practices, we decompose the bug fixing task into four distinct stages: bug reporting, bug diagnosis, patch generation, and patch verification. These stages are performed interactively by LLMs, aiming to imitate the collaborative abilities of programmers during the resolution of software bugs. By harnessing the collective contribution, STEAM effectively enhances the bug-fixing capabilities of LLMs. We implement STEAM by employing the powerful dialogue-based LLM -- ChatGPT. Our evaluation on the widely adopted bug-fixing benchmark demonstrates that STEAM has achieved a new state-of-the-art level of bug-fixing performance.

{{</citation>}}


## cs.SI (1)



### (105/117) Conceptual articles may disrupt the field of marketing but continue to decline in numbers: Evidence from a GPT-assisted study (Jennifer JooYeon Lee et al., 2023)

{{<citation>}}

Jennifer JooYeon Lee, Hyunuk Kim. (2023)  
**Conceptual articles may disrupt the field of marketing but continue to decline in numbers: Evidence from a GPT-assisted study**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2308.14724v1)  

---


**ABSTRACT**  
The present paper addresses if and how an article's academic impact varies by knowledge development approaches. Specifically, it classifies conceptual and empirical articles published in four marketing journals - Journal of Marketing, Journal of Marketing Research, Journal of Consumer Research, and Marketing Science - with the aid of a large language model, GPT. The Kolmogorov-Smirnov (KS) test is implemented for each journal to compare the disruption scores of conceptual and empirical articles. The results show that conceptual research is more likely to disrupt the field of marketing while it tends to decline in its publication quantity. Our paper highlights the importance of conceptual articles and contributes to the understanding of how marketing articles are developed and disseminated to advance knowledge.

{{</citation>}}


## cs.IT (1)



### (106/117) Heterogeneous Drone Small Cells: Optimal 3D Placement for Downlink Power Efficiency and Rate Satisfaction (Nima Namvar et al., 2023)

{{<citation>}}

Nima Namvar, Fatemeh Afghah, Ismail Guvenc. (2023)  
**Heterogeneous Drone Small Cells: Optimal 3D Placement for Downlink Power Efficiency and Rate Satisfaction**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs-SY, cs.IT, eess-SY, math-IT  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2308.14708v1)  

---


**ABSTRACT**  
In this paper, we consider a heterogeneous repository of drone-enabled aerial base stations with varying transmit powers that provide downlink wireless coverage for ground users. One particular challenge is optimal selection and deployment of a subset of available drone base stations (DBSs) to satisfy the downlink data rate requirements while minimizing the overall power consumption. In order to address this challenge, we formulate an optimization problem to select the best subset of available DBSs so as to guarantee wireless coverage with some acceptable transmission rate in the downlink path. In addition to the selection of DBSs, we determine their 3D position so as to minimize their overall power consumption. Moreover, assuming that the DBSs operate in the same frequency band, we develop a novel and computationally efficient beamforming method to alleviate the inter-cell interference impact on the downlink. We propose a Kalai-Smorodinsky bargaining solution to determine the optimal beamforming strategy in the downlink path to compensate for the impairment caused by the interference. Simulation results demonstrate the effectiveness of the proposed solution and provide valuable insights into the performance of the heterogeneous drone-based small cell networks.

{{</citation>}}


## cs.IR (2)



### (107/117) TRIVEA: Transparent Ranking Interpretation using Visual Explanation of Black-Box Algorithmic Rankers (Jun Yuan et al., 2023)

{{<citation>}}

Jun Yuan, Kaustav Bhattacharjee, Akm Zahirul Islam, Aritra Dasgupta. (2023)  
**TRIVEA: Transparent Ranking Interpretation using Visual Explanation of Black-Box Algorithmic Rankers**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-HC, cs-IR, cs.IR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.14622v1)  

---


**ABSTRACT**  
Ranking schemes drive many real-world decisions, like, where to study, whom to hire, what to buy, etc. Many of these decisions often come with high consequences. For example, a university can be deemed less prestigious if not featured in a top-k list, and consumers might not even explore products that do not get recommended to buyers. At the heart of most of these decisions are opaque ranking schemes, which dictate the ordering of data entities, but their internal logic is inaccessible or proprietary. Drawing inferences about the ranking differences is like a guessing game to the stakeholders, like, the rankees (i.e., the entities who are ranked, like product companies) and the decision-makers (i.e., who use the rankings, like buyers). In this paper, we aim to enable transparency in ranking interpretation by using algorithmic rankers that learn from available data and by enabling human reasoning about the learned ranking differences using explainable AI (XAI) methods. To realize this aim, we leverage the exploration-explanation paradigm of human-data interaction to let human stakeholders explore subsets and groupings of complex multi-attribute ranking data using visual explanations of model fit and attribute influence on rankings. We realize this explanation paradigm for transparent ranking interpretation in TRIVEA, a visual analytic system that is fueled by: i) visualizations of model fit derived from algorithmic rankers that learn the associations between attributes and rankings from available data and ii) visual explanations derived from XAI methods that help abstract important patterns, like, the relative influence of attributes in different ranking ranges. Using TRIVEA, end users not trained in data science have the agency to transparently reason about the global and local behavior of the rankings without the need to open black-box ranking models and develop confidence in the resulting attribute-based inferences. We demonstrate the efficacy of TRIVEA using multiple usage scenarios and subjective feedback from researchers with diverse domain expertise. Keywords: Visual Analytics, Learning-to-Rank, Explainable ML, Ranking

{{</citation>}}


### (108/117) RecMind: Large Language Model Powered Agent For Recommendation (Yancheng Wang et al., 2023)

{{<citation>}}

Yancheng Wang, Ziyan Jiang, Zheng Chen, Fan Yang, Yingxue Zhou, Eunah Cho, Xing Fan, Xiaojiang Huang, Yanbin Lu, Yingzhen Yang. (2023)  
**RecMind: Large Language Model Powered Agent For Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.14296v1)  

---


**ABSTRACT**  
Recent advancements in instructing Large Language Models (LLMs) to utilize external tools and execute multi-step plans have significantly enhanced their ability to solve intricate tasks, ranging from mathematical problems to creative writing. Yet, there remains a notable gap in studying the capacity of LLMs in responding to personalized queries such as a recommendation request. To bridge this gap, we have designed an LLM-powered autonomous recommender agent, RecMind, which is capable of providing precise personalized recommendations through careful planning, utilizing tools for obtaining external knowledge, and leveraging individual data. We propose a novel algorithm, Self-Inspiring, to improve the planning ability of the LLM agent. At each intermediate planning step, the LLM 'self-inspires' to consider all previously explored states to plan for next step. This mechanism greatly improves the model's ability to comprehend and utilize historical planning information for recommendation. We evaluate RecMind's performance in various recommendation scenarios, including rating prediction, sequential recommendation, direct recommendation, explanation generation, and review summarization. Our experiment shows that RecMind outperforms existing zero/few-shot LLM-based recommendation methods in different recommendation tasks and achieves competitive performance to a recent model P5, which requires fully pre-train for the recommendation tasks.

{{</citation>}}


## stat.ML (1)



### (109/117) Diversified Ensemble of Independent Sub-Networks for Robust Self-Supervised Representation Learning (Amirhossein Vahidi et al., 2023)

{{<citation>}}

Amirhossein Vahidi, Lisa Wimmer, Hüseyin Anil Gündüz, Bernd Bischl, Eyke Hüllermeier, Mina Rezaei. (2023)  
**Diversified Ensemble of Independent Sub-Networks for Robust Self-Supervised Representation Learning**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-ML, stat.ML  
Keywords: Representation Learning, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2308.14705v1)  

---


**ABSTRACT**  
Ensembling a neural network is a widely recognized approach to enhance model performance, estimate uncertainty, and improve robustness in deep supervised learning. However, deep ensembles often come with high computational costs and memory demands. In addition, the efficiency of a deep ensemble is related to diversity among the ensemble members which is challenging for large, over-parameterized deep neural networks. Moreover, ensemble learning has not yet seen such widespread adoption, and it remains a challenging endeavor for self-supervised or unsupervised representation learning. Motivated by these challenges, we present a novel self-supervised training regime that leverages an ensemble of independent sub-networks, complemented by a new loss function designed to encourage diversity. Our method efficiently builds a sub-model ensemble with high diversity, leading to well-calibrated estimates of model uncertainty, all achieved with minimal computational overhead compared to traditional deep self-supervised ensembles. To evaluate the effectiveness of our approach, we conducted extensive experiments across various tasks, including in-distribution generalization, out-of-distribution detection, dataset corruption, and semi-supervised settings. The results demonstrate that our method significantly improves prediction reliability. Our approach not only achieves excellent accuracy but also enhances calibration, surpassing baseline performance across a wide range of self-supervised architectures in computer vision, natural language processing, and genomics data.

{{</citation>}}


## astro-ph.IM (1)



### (110/117) A Transformer-Conditioned Neural Fields Pipeline with Polar Coordinate Representation for Astronomical Radio Interferometric Data Reconstruction (Ruoqi Wang et al., 2023)

{{<citation>}}

Ruoqi Wang, Qiong Luo, Feng Wang. (2023)  
**A Transformer-Conditioned Neural Fields Pipeline with Polar Coordinate Representation for Astronomical Radio Interferometric Data Reconstruction**  

---
Primary Category: astro-ph.IM  
Categories: astro-ph-IM, astro-ph.IM, cs-AI, cs-CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.14610v1)  

---


**ABSTRACT**  
In radio astronomy, visibility data, which are measurements of wave signals from radio telescopes, are transformed into images for observation of distant celestial objects. However, these resultant images usually contain both real sources and artifacts, due to signal sparsity and other factors. One way to obtain cleaner images is to reconstruct samples into dense forms before imaging. Unfortunately, existing visibility reconstruction methods may miss some components of the frequency data, so blurred object edges and persistent artifacts remain in the images. Furthermore, the computation overhead is high on irregular visibility samples due to the data skew. To address these problems, we propose PolarRec, a reconstruction method for interferometric visibility data, which consists of a transformer-conditioned neural fields pipeline with a polar coordinate representation. This representation matches the way in which telescopes observe a celestial area as the Earth rotates. We further propose Radial Frequency Loss function, using radial coordinates in the polar coordinate system to correlate with the frequency information, to help reconstruct complete visibility. We also group visibility sample points by angular coordinates in the polar coordinate system, and use groups as the granularity for subsequent encoding with a Transformer encoder. Consequently, our method can capture the inherent characteristics of visibility data effectively and efficiently. Our experiments demonstrate that PolarRec markedly improves imaging results by faithfully reconstructing all frequency components in the visibility domain while significantly reducing the computation cost.

{{</citation>}}


## eess.SY (1)



### (111/117) Recent Progress in Energy Management of Connected Hybrid Electric Vehicles Using Reinforcement Learning (Min Hua et al., 2023)

{{<citation>}}

Min Hua, Bin Shuai, Quan Zhou, Jinhai Wang, Yinglong He, Hongming Xu. (2023)  
**Recent Progress in Energy Management of Connected Hybrid Electric Vehicles Using Reinforcement Learning**  

---
Primary Category: eess.SY  
Categories: cs-LG, cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.14602v1)  

---


**ABSTRACT**  
The growing adoption of hybrid electric vehicles (HEVs) presents a transformative opportunity for revolutionizing transportation energy systems. The shift towards electrifying transportation aims to curb environmental concerns related to fossil fuel consumption. This necessitates efficient energy management systems (EMS) to optimize energy efficiency. The evolution of EMS from HEVs to connected hybrid electric vehicles (CHEVs) represent a pivotal shift. For HEVs, EMS now confronts the intricate energy cooperation requirements of CHEVs, necessitating advanced algorithms for route optimization, charging coordination, and load distribution. Challenges persist in both domains, including optimal energy utilization for HEVs, and cooperative eco-driving control (CED) for CHEVs across diverse vehicle types. Reinforcement learning (RL) stands out as a promising tool for addressing these challenges at hand. Specifically, within the realm of CHEVs, the application of multi-agent reinforcement learning (MARL) emerges as a powerful approach for effectively tackling the intricacies of CED control. Despite extensive research, few reviews span from individual vehicles to multi-vehicle scenarios. This review bridges the gap, highlighting challenges, advancements, and potential contributions of RL-based solutions for future sustainable transportation systems.

{{</citation>}}


## cs.NI (1)



### (112/117) Deep Reinforcement Learning for Uplink Scheduling in NOMA-URLLC Networks (Benoît-Marie Robaglia et al., 2023)

{{<citation>}}

Benoît-Marie Robaglia, Marceau Coupechoux, Dimitrios Tsilimantos. (2023)  
**Deep Reinforcement Learning for Uplink Scheduling in NOMA-URLLC Networks**  

---
Primary Category: cs.NI  
Categories: cs-AI, cs-NI, cs.NI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.14523v1)  

---


**ABSTRACT**  
This article addresses the problem of Ultra Reliable Low Latency Communications (URLLC) in wireless networks, a framework with particularly stringent constraints imposed by many Internet of Things (IoT) applications from diverse sectors. We propose a novel Deep Reinforcement Learning (DRL) scheduling algorithm, named NOMA-PPO, to solve the Non-Orthogonal Multiple Access (NOMA) uplink URLLC scheduling problem involving strict deadlines. The challenge of addressing uplink URLLC requirements in NOMA systems is related to the combinatorial complexity of the action space due to the possibility to schedule multiple devices, and to the partial observability constraint that we impose to our algorithm in order to meet the IoT communication constraints and be scalable. Our approach involves 1) formulating the NOMA-URLLC problem as a Partially Observable Markov Decision Process (POMDP) and the introduction of an agent state, serving as a sufficient statistic of past observations and actions, enabling a transformation of the POMDP into a Markov Decision Process (MDP); 2) adapting the Proximal Policy Optimization (PPO) algorithm to handle the combinatorial action space; 3) incorporating prior knowledge into the learning agent with the introduction of a Bayesian policy. Numerical results reveal that not only does our approach outperform traditional multiple access protocols and DRL benchmarks on 3GPP scenarios, but also proves to be robust under various channel and traffic configurations, efficiently exploiting inherent time correlations.

{{</citation>}}


## cs.DL (1)



### (113/117) Do Successful Researchers Reach the Self-Organized Critical Point? (Asim Ghosh et al., 2023)

{{<citation>}}

Asim Ghosh, Bikas K. Chakrabarti. (2023)  
**Do Successful Researchers Reach the Self-Organized Critical Point?**  

---
Primary Category: cs.DL  
Categories: cs-DL, cs.DL, physics-soc-ph  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2308.14435v1)  

---


**ABSTRACT**  
The index of success of the researchers are now mostly measured using the Hirsch index ($h$). Our recent precise demonstration, that statistically $h \sim \sqrt {N_c} \sim \sqrt {N_p}$, where $N_p$ and $N_c$ denote respectively the total number of publications and total citations for the researcher, suggests that average number of citations per paper ($N_c/N_p$), and hence $h$, are statistical numbers (Dunbar numbers) depending on the community or network to which the researcher belongs. We show here, extending our earlier observations, that the indications of success are not reflected by the total citations $N_c$, rather by the inequalities among citations from publications to publications. Specifically, we show that for very successful authors, the yearly variations in the Gini index ($g$, giving the average inequality of citations for the publications) and the Kolkata index ($k$, giving the fraction of total citations received by the top $1 - k$ fraction of publications; $k = 0.80$ corresponds to Pareto's 80/20 law) approach each other to $g = k \simeq 0.82$, signaling a precursor for the arrival of (or departure from) the Self-Organized Critical (SOC) state of his/her publication statistics. Analyzing the citation statistics (from Google Scholar) of thirty successful scientists throughout their recorded publication history, we find that the $g$ and $k$ for very successful among them (mostly Nobel Laureates, highest rank Stanford Cite-Scorers, and a few others) reach and hover just above (and then) below that $g = k \simeq 0.82$ mark, while for others they remain below that mark. We also find that for all the lower (than the SOC mark 0.82) values of $k$ and $g$ fit a linear relationship $k = 1/2 + cg$, with $c = 0.39$.

{{</citation>}}


## cs.LO (1)



### (114/117) Shielded Reinforcement Learning for Hybrid Systems (Asger Horn Brorholt et al., 2023)

{{<citation>}}

Asger Horn Brorholt, Peter Gjøl Jensen, Kim Guldstrand Larsen, Florian Lorber, Christian Schilling. (2023)  
**Shielded Reinforcement Learning for Hybrid Systems**  

---
Primary Category: cs.LO  
Categories: cs-AI, cs-LG, cs-LO, cs-SY, cs.LO, eess-SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.14424v1)  

---


**ABSTRACT**  
Safe and optimal controller synthesis for switched-controlled hybrid systems, which combine differential equations and discrete changes of the system's state, is known to be intricately hard. Reinforcement learning has been leveraged to construct near-optimal controllers, but their behavior is not guaranteed to be safe, even when it is encouraged by reward engineering. One way of imposing safety to a learned controller is to use a shield, which is correct by design. However, obtaining a shield for non-linear and hybrid environments is itself intractable. In this paper, we propose the construction of a shield using the so-called barbaric method, where an approximate finite representation of an underlying partition-based two-player safety game is extracted via systematically picked samples of the true transition function. While hard safety guarantees are out of reach, we experimentally demonstrate strong statistical safety guarantees with a prototype implementation and UPPAAL STRATEGO. Furthermore, we study the impact of the synthesized shield when applied as either a pre-shield (applied before learning a controller) or a post-shield (only applied after learning a controller). We experimentally demonstrate superiority of the pre-shielding approach. We apply our technique on a range of case studies, including two industrial examples, and further study post-optimization of the post-shielding approach.

{{</citation>}}


## cs.MM (1)



### (115/117) UMMAFormer: A Universal Multimodal-adaptive Transformer Framework for Temporal Forgery Localization (Rui Zhang et al., 2023)

{{<citation>}}

Rui Zhang, Hongxia Wang, Mingshan Du, Hanqing Liu, Yang Zhou, Qiang Zeng. (2023)  
**UMMAFormer: A Universal Multimodal-adaptive Transformer Framework for Temporal Forgery Localization**  

---
Primary Category: cs.MM  
Categories: 68T45, I-4, cs-CV, cs-MM, cs.MM  
Keywords: AI, Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2308.14395v1)  

---


**ABSTRACT**  
The emergence of artificial intelligence-generated content (AIGC) has raised concerns about the authenticity of multimedia content in various fields. However, existing research for forgery content detection has focused mainly on binary classification tasks of complete videos, which has limited applicability in industrial settings. To address this gap, we propose UMMAFormer, a novel universal transformer framework for temporal forgery localization (TFL) that predicts forgery segments with multimodal adaptation. Our approach introduces a Temporal Feature Abnormal Attention (TFAA) module based on temporal feature reconstruction to enhance the detection of temporal differences. We also design a Parallel Cross-Attention Feature Pyramid Network (PCA-FPN) to optimize the Feature Pyramid Network (FPN) for subtle feature enhancement. To evaluate the proposed method, we contribute a novel Temporal Video Inpainting Localization (TVIL) dataset specifically tailored for video inpainting scenes. Our experiments show that our approach achieves state-of-the-art performance on benchmark datasets, including Lav-DF, TVIL, and Psynd, significantly outperforming previous methods. The code and data are available at https://github.com/ymhzyj/UMMAFormer/.

{{</citation>}}


## cs.RO (1)



### (116/117) End-to-End Driving via Self-Supervised Imitation Learning Using Camera and LiDAR Data (Jin Bok Park et al., 2023)

{{<citation>}}

Jin Bok Park, Jinkyu Lee, Muhyun Back, Hyunmin Han, David T. Ma, Sang Min Won, Sung Soo Hwang, Il Yong Chun. (2023)  
**End-to-End Driving via Self-Supervised Imitation Learning Using Camera and LiDAR Data**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2308.14329v1)  

---


**ABSTRACT**  
In autonomous driving, the end-to-end (E2E) driving approach that predicts vehicle control signals directly from sensor data is rapidly gaining attention. To learn a safe E2E driving system, one needs an extensive amount of driving data and human intervention. Vehicle control data is constructed by many hours of human driving, and it is challenging to construct large vehicle control datasets. Often, publicly available driving datasets are collected with limited driving scenes, and collecting vehicle control data is only available by vehicle manufacturers. To address these challenges, this paper proposes the first self-supervised learning framework, self-supervised imitation learning (SSIL), that can learn E2E driving networks without using driving command data. To construct pseudo steering angle data, proposed SSIL predicts a pseudo target from the vehicle's poses at the current and previous time points that are estimated with light detection and ranging sensors. Our numerical experiments demonstrate that the proposed SSIL framework achieves comparable E2E driving accuracy with the supervised learning counterpart. In addition, our qualitative analyses using a conventional visual explanation tool show that trained NNs by proposed SSIL and the supervision counterpart attend similar objects in making predictions.

{{</citation>}}


## q-bio.GN (1)



### (117/117) XVir: A Transformer-Based Architecture for Identifying Viral Reads from Cancer Samples (Shorya Consul et al., 2023)

{{<citation>}}

Shorya Consul, John Robertson, Haris Vikalo. (2023)  
**XVir: A Transformer-Based Architecture for Identifying Viral Reads from Cancer Samples**  

---
Primary Category: q-bio.GN  
Categories: I-2-1; J-3, cs-LG, q-bio-GN, q-bio.GN  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.14769v1)  

---


**ABSTRACT**  
It is estimated that approximately 15% of cancers worldwide can be linked to viral infections. The viruses that can cause or increase the risk of cancer include human papillomavirus, hepatitis B and C viruses, Epstein-Barr virus, and human immunodeficiency virus, to name a few. The computational analysis of the massive amounts of tumor DNA data, whose collection is enabled by the recent advancements in sequencing technologies, have allowed studies of the potential association between cancers and viral pathogens. However, the high diversity of oncoviral families makes reliable detection of viral DNA difficult and thus, renders such analysis challenging. In this paper, we introduce XVir, a data pipeline that relies on a transformer-based deep learning architecture to reliably identify viral DNA present in human tumors. In particular, XVir is trained on genomic sequencing reads from viral and human genomes and may be used with tumor sequence information to find evidence of viral DNA in human cancers. Results on semi-experimental data demonstrate that XVir is capable of achieving high detection accuracy, generally outperforming state-of-the-art competing methods while being more compact and less computationally demanding.

{{</citation>}}
