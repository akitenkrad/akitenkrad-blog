---
draft: false
title: "arXiv @ 2024.01.25"
date: 2024-01-25
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2024"]
menu:
  sidebar:
    name: "arXiv @ 2024.01.25"
    identifier: arxiv_20240125
    parent: 202401_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.LG (14)](#cslg-14)
- [eess.AS (4)](#eessas-4)
- [cs.CY (5)](#cscy-5)
- [cs.CL (28)](#cscl-28)
- [cs.AI (12)](#csai-12)
- [cs.CV (21)](#cscv-21)
- [cs.RO (2)](#csro-2)
- [eess.IV (1)](#eessiv-1)
- [cs.NI (1)](#csni-1)
- [cs.SY (1)](#cssy-1)
- [stat.ML (1)](#statml-1)
- [cs.SD (1)](#cssd-1)
- [eess.SY (3)](#eesssy-3)
- [cs.IR (3)](#csir-3)
- [cs.SE (3)](#csse-3)
- [cs.CE (1)](#csce-1)
- [cs.MA (3)](#csma-3)
- [cs.AR (1)](#csar-1)
- [cs.HC (1)](#cshc-1)
- [cs.DC (1)](#csdc-1)
- [cs.CR (1)](#cscr-1)
- [cs.IT (1)](#csit-1)

## cs.LG (14)



### (1/109) NLBAC: A Neural Ordinary Differential Equations-based Framework for Stable and Safe Reinforcement Learning (Liqun Zhao et al., 2024)

{{<citation>}}

Liqun Zhao, Keyan Miao, Konstantinos Gatsis, Antonis Papachristodoulou. (2024)  
**NLBAC: A Neural Ordinary Differential Equations-based Framework for Stable and Safe Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-RO, cs-SY, cs.LG, eess-SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.13148v1)  

---


**ABSTRACT**  
Reinforcement learning (RL) excels in applications such as video games and robotics, but ensuring safety and stability remains challenging when using RL to control real-world systems where using model-free algorithms suffering from low sample efficiency might be prohibitive. This paper first provides safety and stability definitions for the RL system, and then introduces a Neural ordinary differential equations-based Lyapunov-Barrier Actor-Critic (NLBAC) framework that leverages Neural Ordinary Differential Equations (NODEs) to approximate system dynamics and integrates the Control Barrier Function (CBF) and Control Lyapunov Function (CLF) frameworks with the actor-critic method to assist in maintaining the safety and stability for the system. Within this framework, we employ the augmented Lagrangian method to update the RL-based controller parameters. Additionally, we introduce an extra backup controller in situations where CBF constraints for safety and the CLF constraint for stability cannot be satisfied simultaneously. Simulation results demonstrate that the framework leads the system to approach the desired state and allows fewer violations of safety constraints with better sample efficiency compared to other methods.

{{</citation>}}


### (2/109) Multi-Agent Based Transfer Learning for Data-Driven Air Traffic Applications (Chuhao Deng et al., 2024)

{{<citation>}}

Chuhao Deng, Hong-Cheol Choi, Hyunsang Park, Inseok Hwang. (2024)  
**Multi-Agent Based Transfer Learning for Data-Driven Air Traffic Applications**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-MA, cs-SY, cs.LG, eess-SY, stat-ML  
Keywords: BERT, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.14421v1)  

---


**ABSTRACT**  
Research in developing data-driven models for Air Traffic Management (ATM) has gained a tremendous interest in recent years. However, data-driven models are known to have long training time and require large datasets to achieve good performance. To address the two issues, this paper proposes a Multi-Agent Bidirectional Encoder Representations from Transformers (MA-BERT) model that fully considers the multi-agent characteristic of the ATM system and learns air traffic controllers' decisions, and a pre-training and fine-tuning transfer learning framework. By pre-training the MA-BERT on a large dataset from a major airport and then fine-tuning it to other airports and specific air traffic applications, a large amount of the total training time can be saved. In addition, for newly adopted procedures and constructed airports where no historical data is available, this paper shows that the pre-trained MA-BERT can achieve high performance by updating regularly with little data. The proposed transfer learning framework and MA-BERT are tested with the automatic dependent surveillance-broadcast data recorded in 3 airports in South Korea in 2019.

{{</citation>}}


### (3/109) Probabilistic Demand Forecasting with Graph Neural Networks (Nikita Kozodoi et al., 2024)

{{<citation>}}

Nikita Kozodoi, Elizaveta Zinovyeva, Simon Valentin, João Pereira, Rodrigo Agundez. (2024)  
**Probabilistic Demand Forecasting with Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2401.13096v1)  

---


**ABSTRACT**  
Demand forecasting is a prominent business use case that allows retailers to optimize inventory planning, logistics, and core business decisions. One of the key challenges in demand forecasting is accounting for relationships and interactions between articles. Most modern forecasting approaches provide independent article-level predictions that do not consider the impact of related articles. Recent research has attempted addressing this challenge using Graph Neural Networks (GNNs) and showed promising results. This paper builds on previous research on GNNs and makes two contributions. First, we integrate a GNN encoder into a state-of-the-art DeepAR model. The combined model produces probabilistic forecasts, which are crucial for decision-making under uncertainty. Second, we propose to build graphs using article attribute similarity, which avoids reliance on a pre-defined graph structure. Experiments on three real-world datasets show that the proposed approach consistently outperforms non-graph benchmarks. We also show that our approach produces article embeddings that encode article similarity and demand dynamics and are useful for other downstream business tasks beyond forecasting.

{{</citation>}}


### (4/109) Learning safety critics via a non-contractive binary bellman operator (Agustin Castellano et al., 2024)

{{<citation>}}

Agustin Castellano, Hancheng Min, Juan Andrés Bazerque, Enrique Mallada. (2024)  
**Learning safety critics via a non-contractive binary bellman operator**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SY, cs.LG, eess-SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.12849v1)  

---


**ABSTRACT**  
The inability to naturally enforce safety in Reinforcement Learning (RL), with limited failures, is a core challenge impeding its use in real-world applications. One notion of safety of vast practical relevance is the ability to avoid (unsafe) regions of the state space. Though such a safety goal can be captured by an action-value-like function, a.k.a. safety critics, the associated operator lacks the desired contraction and uniqueness properties that the classical Bellman operator enjoys. In this work, we overcome the non-contractiveness of safety critic operators by leveraging that safety is a binary property. To that end, we study the properties of the binary safety critic associated with a deterministic dynamical system that seeks to avoid reaching an unsafe region. We formulate the corresponding binary Bellman equation (B2E) for safety and study its properties. While the resulting operator is still non-contractive, we fully characterize its fixed points representing--except for a spurious solution--maximal persistently safe regions of the state space that can always avoid failure. We provide an algorithm that, by design, leverages axiomatic knowledge of safe data to avoid spurious fixed points.

{{</citation>}}


### (5/109) Iterated Relevance Matrix Analysis (IRMA) for the identification of class-discriminative subspaces (Sofie Lövdal et al., 2024)

{{<citation>}}

Sofie Lövdal, Michael Biehl. (2024)  
**Iterated Relevance Matrix Analysis (IRMA) for the identification of class-discriminative subspaces**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2401.12842v1)  

---


**ABSTRACT**  
We introduce and investigate the iterated application of Generalized Matrix Learning Vector Quantizaton for the analysis of feature relevances in classification problems, as well as for the construction of class-discriminative subspaces. The suggested Iterated Relevance Matrix Analysis (IRMA) identifies a linear subspace representing the classification specific information of the considered data sets using Generalized Matrix Learning Vector Quantization (GMLVQ). By iteratively determining a new discriminative subspace while projecting out all previously identified ones, a combined subspace carrying all class-specific information can be found. This facilitates a detailed analysis of feature relevances, and enables improved low-dimensional representations and visualizations of labeled data sets. Additionally, the IRMA-based class-discriminative subspace can be used for dimensionality reduction and the training of robust classifiers with potentially improved performance.

{{</citation>}}


### (6/109) Enhancing Next Destination Prediction: A Novel LSTM Approach Using Real-World Airline Data (Salih Salihoglu et al., 2024)

{{<citation>}}

Salih Salihoglu, Gulser Koksal, Orhan Abar. (2024)  
**Enhancing Next Destination Prediction: A Novel LSTM Approach Using Real-World Airline Data**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2401.12830v1)  

---


**ABSTRACT**  
In the modern transportation industry, accurate prediction of travelers' next destinations brings multiple benefits to companies, such as customer satisfaction and targeted marketing. This study focuses on developing a precise model that captures the sequential patterns and dependencies in travel data, enabling accurate predictions of individual travelers' future destinations. To achieve this, a novel model architecture with a sliding window approach based on Long Short-Term Memory (LSTM) is proposed for destination prediction in the transportation industry. The experimental results highlight satisfactory performance and high scores achieved by the proposed model across different data sizes and performance metrics. This research contributes to advancing destination prediction methods, empowering companies to deliver personalized recommendations and optimize customer experiences in the dynamic travel landscape.

{{</citation>}}


### (7/109) MAPPING: Debiasing Graph Neural Networks for Fair Node Classification with Limited Sensitive Information Leakage (Ying Song et al., 2024)

{{<citation>}}

Ying Song, Balaji Palanisamy. (2024)  
**MAPPING: Debiasing Graph Neural Networks for Fair Node Classification with Limited Sensitive Information Leakage**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2401.12824v1)  

---


**ABSTRACT**  
Despite remarkable success in diverse web-based applications, Graph Neural Networks(GNNs) inherit and further exacerbate historical discrimination and social stereotypes, which critically hinder their deployments in high-stake domains such as online clinical diagnosis, financial crediting, etc. However, current fairness research that primarily craft on i.i.d data, cannot be trivially replicated to non-i.i.d. graph structures with topological dependence among samples. Existing fair graph learning typically favors pairwise constraints to achieve fairness but fails to cast off dimensional limitations and generalize them into multiple sensitive attributes; besides, most studies focus on in-processing techniques to enforce and calibrate fairness, constructing a model-agnostic debiasing GNN framework at the pre-processing stage to prevent downstream misuses and improve training reliability is still largely under-explored. Furthermore, previous work on GNNs tend to enhance either fairness or privacy individually but few probe into their interplays. In this paper, we propose a novel model-agnostic debiasing framework named MAPPING (\underline{M}asking \underline{A}nd \underline{P}runing and Message-\underline{P}assing train\underline{ING}) for fair node classification, in which we adopt the distance covariance($dCov$)-based fairness constraints to simultaneously reduce feature and topology biases in arbitrary dimensions, and combine them with adversarial debiasing to confine the risks of attribute inference attacks. Experiments on real-world datasets with different GNN variants demonstrate the effectiveness and flexibility of MAPPING. Our results show that MAPPING can achieve better trade-offs between utility and fairness, and mitigate privacy risks of sensitive information leakage.

{{</citation>}}


### (8/109) Dynamic Layer Tying for Parameter-Efficient Transformers (Tamir David Hay et al., 2024)

{{<citation>}}

Tamir David Hay, Lior Wolf. (2024)  
**Dynamic Layer Tying for Parameter-Efficient Transformers**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.12819v1)  

---


**ABSTRACT**  
In the pursuit of reducing the number of trainable parameters in deep transformer networks, we employ Reinforcement Learning to dynamically select layers during training and tie them together. Every few iterations, the RL agent is asked whether to train each layer $i$ independently or to copy the weights of a previous layer $j<i$. This facilitates weight sharing, reduces the number of trainable parameters, and also serves as an effective regularization technique. Experimental evaluations validate that our model modestly outperforms the baseline transformer model with regard to perplexity and drastically reduces the number of trainable parameters. In particular, the memory consumption during training is up to one order of magnitude less than the conventional training method.

{{</citation>}}


### (9/109) DeepRicci: Self-supervised Graph Structure-Feature Co-Refinement for Alleviating Over-squashing (Li Sun et al., 2024)

{{<citation>}}

Li Sun, Zhenhao Huang, Hua Wu, Junda Ye, Hao Peng, Zhengtao Yu, Philip S. Yu. (2024)  
**DeepRicci: Self-supervised Graph Structure-Feature Co-Refinement for Alleviating Over-squashing**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2401.12780v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) have shown great power for learning and mining on graphs, and Graph Structure Learning (GSL) plays an important role in boosting GNNs with a refined graph. In the literature, most GSL solutions either primarily focus on structure refinement with task-specific supervision (i.e., node classification), or overlook the inherent weakness of GNNs themselves (e.g., over-squashing), resulting in suboptimal performance despite sophisticated designs. In light of these limitations, we propose to study self-supervised graph structure-feature co-refinement for effectively alleviating the issue of over-squashing in typical GNNs. In this paper, we take a fundamentally different perspective of the Ricci curvature in Riemannian geometry, in which we encounter the challenges of modeling, utilizing and computing Ricci curvature. To tackle these challenges, we present a self-supervised Riemannian model, DeepRicci. Specifically, we introduce a latent Riemannian space of heterogeneous curvatures to model various Ricci curvatures, and propose a gyrovector feature mapping to utilize Ricci curvature for typical GNNs. Thereafter, we refine node features by geometric contrastive learning among different geometric views, and simultaneously refine graph structure by backward Ricci flow based on a novel formulation of differentiable Ricci curvature. Finally, extensive experiments on public datasets show the superiority of DeepRicci, and the connection between backward Ricci flow and over-squashing. Codes of our work are given in https://github.com/RiemanGraph/.

{{</citation>}}


### (10/109) Falcon: Fair Active Learning using Multi-armed Bandits (Ki Hyun Tae et al., 2024)

{{<citation>}}

Ki Hyun Tae, Hantian Zhang, Jaeyoung Park, Kexin Rong, Steven Euijong Whang. (2024)  
**Falcon: Fair Active Learning using Multi-armed Bandits**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Active Learning, Bias, Falcon  
[Paper Link](http://arxiv.org/abs/2401.12722v2)  

---


**ABSTRACT**  
Biased data can lead to unfair machine learning models, highlighting the importance of embedding fairness at the beginning of data analysis, particularly during dataset curation and labeling. In response, we propose Falcon, a scalable fair active learning framework. Falcon adopts a data-centric approach that improves machine learning model fairness via strategic sample selection. Given a user-specified group fairness measure, Falcon identifies samples from "target groups" (e.g., (attribute=female, label=positive)) that are the most informative for improving fairness. However, a challenge arises since these target groups are defined using ground truth labels that are not available during sample selection. To handle this, we propose a novel trial-and-error method, where we postpone using a sample if the predicted label is different from the expected one and falls outside the target group. We also observe the trade-off that selecting more informative samples results in higher likelihood of postponing due to undesired label prediction, and the optimal balance varies per dataset. We capture the trade-off between informativeness and postpone rate as policies and propose to automatically select the best policy using adversarial multi-armed bandit methods, given their computational efficiency and theoretical guarantees. Experiments show that Falcon significantly outperforms existing fair active learning approaches in terms of fairness and accuracy and is more efficient. In particular, only Falcon supports a proper trade-off between accuracy and fairness where its maximum fairness score is 1.8-4.5x higher than the second-best results.

{{</citation>}}


### (11/109) Consistency Enhancement-Based Deep Multiview Clustering via Contrastive Learning (Hao Yang et al., 2024)

{{<citation>}}

Hao Yang, Hua Mao, Wai Lok Woo, Jie Chen, Xi Peng. (2024)  
**Consistency Enhancement-Based Deep Multiview Clustering via Contrastive Learning**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2401.12648v1)  

---


**ABSTRACT**  
Multiview clustering (MVC) segregates data samples into meaningful clusters by synthesizing information across multiple views. Moreover, deep learning-based methods have demonstrated their strong feature learning capabilities in MVC scenarios. However, effectively generalizing feature representations while maintaining consistency is still an intractable problem. In addition, most existing deep clustering methods based on contrastive learning overlook the consistency of the clustering representations during the clustering process. In this paper, we show how the above problems can be overcome and propose a consistent enhancement-based deep MVC method via contrastive learning (CCEC). Specifically, semantic connection blocks are incorporated into a feature representation to preserve the consistent information among multiple views. Furthermore, the representation process for clustering is enhanced through spectral clustering, and the consistency across multiple views is improved. Experiments conducted on five datasets demonstrate the effectiveness and superiority of our method in comparison with the state-of-the-art (SOTA) methods. The code for this method can be accessed at https://anonymous.4open.science/r/CCEC-E84E/.

{{</citation>}}


### (12/109) Prompt Smells: An Omen for Undesirable Generative AI Outputs (Krishna Ronanki et al., 2024)

{{<citation>}}

Krishna Ronanki, Beatriz Cabrero-Daniel, Christian Berger. (2024)  
**Prompt Smells: An Omen for Undesirable Generative AI Outputs**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SE, cs.LG  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2401.12611v1)  

---


**ABSTRACT**  
Recent Generative Artificial Intelligence (GenAI) trends focus on various applications, including creating stories, illustrations, poems, articles, computer code, music compositions, and videos. Extrinsic hallucinations are a critical limitation of such GenAI, which can lead to significant challenges in achieving and maintaining the trustworthiness of GenAI. In this paper, we propose two new concepts that we believe will aid the research community in addressing limitations associated with the application of GenAI models. First, we propose a definition for the "desirability" of GenAI outputs and three factors which are observed to influence it. Second, drawing inspiration from Martin Fowler's code smells, we propose the concept of "prompt smells" and the adverse effects they are observed to have on the desirability of GenAI outputs. We expect our work will contribute to the ongoing conversation about the desirability of GenAI outputs and help advance the field in a meaningful way.

{{</citation>}}


### (13/109) DAFA: Distance-Aware Fair Adversarial Training (Hyungyu Lee et al., 2024)

{{<citation>}}

Hyungyu Lee, Saehyung Lee, Hyemi Jang, Junsung Park, Ho Bae, Sungroh Yoon. (2024)  
**DAFA: Distance-Aware Fair Adversarial Training**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Adversarial Training  
[Paper Link](http://arxiv.org/abs/2401.12532v1)  

---


**ABSTRACT**  
The disparity in accuracy between classes in standard training is amplified during adversarial training, a phenomenon termed the robust fairness problem. Existing methodologies aimed to enhance robust fairness by sacrificing the model's performance on easier classes in order to improve its performance on harder ones. However, we observe that under adversarial attacks, the majority of the model's predictions for samples from the worst class are biased towards classes similar to the worst class, rather than towards the easy classes. Through theoretical and empirical analysis, we demonstrate that robust fairness deteriorates as the distance between classes decreases. Motivated by these insights, we introduce the Distance-Aware Fair Adversarial training (DAFA) methodology, which addresses robust fairness by taking into account the similarities between classes. Specifically, our method assigns distinct loss weights and adversarial margins to each class and adjusts them to encourage a trade-off in robustness among similar classes. Experimental results across various datasets demonstrate that our method not only maintains average robust accuracy but also significantly improves the worst robust accuracy, indicating a marked improvement in robust fairness compared to existing methods.

{{</citation>}}


### (14/109) Reinforcement Learning for Graph Coloring: Understanding the Power and Limits of Non-Label Invariant Representations (Chase Cummins et al., 2024)

{{<citation>}}

Chase Cummins, Richard Veras. (2024)  
**Reinforcement Learning for Graph Coloring: Understanding the Power and Limits of Non-Label Invariant Representations**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.12470v1)  

---


**ABSTRACT**  
Register allocation is one of the most important problems for modern compilers. With a practically unlimited number of user variables and a small number of CPU registers, assigning variables to registers without conflicts is a complex task. This work demonstrates the use of casting the register allocation problem as a graph coloring problem. Using technologies such as PyTorch and OpenAI Gymnasium Environments we will show that a Proximal Policy Optimization model can learn to solve the graph coloring problem. We will also show that the labeling of a graph is critical to the performance of the model by taking the matrix representation of a graph and permuting it. We then test the model's effectiveness on each of these permutations and show that it is not effective when given a relabeling of the same graph. Our main contribution lies in showing the need for label reordering invariant representations of graphs for machine learning models to achieve consistent performance.

{{</citation>}}


## eess.AS (4)



### (15/109) Locality enhanced dynamic biasing and sampling strategies for contextual ASR (Md Asif Jalal et al., 2024)

{{<citation>}}

Md Asif Jalal, Pablo Peso Parada, George Pavlidis, Vasileios Moschopoulos, Karthikeyan Saravanan, Chrysovalantis-Giorgos Kontoulis, Jisi Zhang, Anastasios Drosou, Gil Ho Lee, Jungin Lee, Seokyeong Jung. (2024)  
**Locality enhanced dynamic biasing and sampling strategies for contextual ASR**  

---
Primary Category: eess.AS  
Categories: cs-CL, cs-SD, eess-AS, eess.AS  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2401.13146v1)  

---


**ABSTRACT**  
Automatic Speech Recognition (ASR) still face challenges when recognizing time-variant rare-phrases. Contextual biasing (CB) modules bias ASR model towards such contextually-relevant phrases. During training, a list of biasing phrases are selected from a large pool of phrases following a sampling strategy. In this work we firstly analyse different sampling strategies to provide insights into the training of CB for ASR with correlation plots between the bias embeddings among various training stages. Secondly, we introduce a neighbourhood attention (NA) that localizes self attention (SA) to the nearest neighbouring frames to further refine the CB output. The results show that this proposed approach provides on average a 25.84% relative WER improvement on LibriSpeech sets and rare-word evaluation compared to the baseline.

{{</citation>}}


### (16/109) Overlap-aware End-to-End Supervised Hierarchical Graph Clustering for Speaker Diarization (Prachi Singh et al., 2024)

{{<citation>}}

Prachi Singh, Sriram Ganapathy. (2024)  
**Overlap-aware End-to-End Supervised Hierarchical Graph Clustering for Speaker Diarization**  

---
Primary Category: eess.AS  
Categories: cs-AI, cs-SD, eess-AS, eess.AS  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2401.12850v1)  

---


**ABSTRACT**  
Speaker diarization, the task of segmenting an audio recording based on speaker identity, constitutes an important speech pre-processing step for several downstream applications. The conventional approach to diarization involves multiple steps of embedding extraction and clustering, which are often optimized in an isolated fashion. While end-to-end diarization systems attempt to learn a single model for the task, they are often cumbersome to train and require large supervised datasets. In this paper, we propose an end-to-end supervised hierarchical clustering algorithm based on graph neural networks (GNN), called End-to-end Supervised HierARchical Clustering (E-SHARC). The E-SHARC approach uses front-end mel-filterbank features as input and jointly learns an embedding extractor and the GNN clustering module, performing representation learning, metric learning, and clustering with end-to-end optimization. Further, with additional inputs from an external overlap detector, the E-SHARC approach is capable of predicting the speakers in the overlapping speech regions. The experimental evaluation on several benchmark datasets like AMI, VoxConverse and DISPLACE, illustrates that the proposed E-SHARC framework improves significantly over the state-of-art diarization systems.

{{</citation>}}


### (17/109) Boosting Unknown-number Speaker Separation with Transformer Decoder-based Attractor (Younglo Lee et al., 2024)

{{<citation>}}

Younglo Lee, Shukjae Choi, Byeong-Yeol Kim, Zhong-Qiu Wang, Shinji Watanabe. (2024)  
**Boosting Unknown-number Speaker Separation with Transformer Decoder-based Attractor**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.12473v1)  

---


**ABSTRACT**  
We propose a novel speech separation model designed to separate mixtures with an unknown number of speakers. The proposed model stacks 1) a dual-path processing block that can model spectro-temporal patterns, 2) a transformer decoder-based attractor (TDA) calculation module that can deal with an unknown number of speakers, and 3) triple-path processing blocks that can model inter-speaker relations. Given a fixed, small set of learned speaker queries and the mixture embedding produced by the dual-path blocks, TDA infers the relations of these queries and generates an attractor vector for each speaker. The estimated attractors are then combined with the mixture embedding by feature-wise linear modulation conditioning, creating a speaker dimension. The mixture embedding, conditioned with speaker information produced by TDA, is fed to the final triple-path blocks, which augment the dual-path blocks with an additional pathway dedicated to inter-speaker processing. The proposed approach outperforms the previous best reported in the literature, achieving 24.0 and 23.7 dB SI-SDR improvement (SI-SDRi) on WSJ0-2 and 3mix respectively, with a single model trained to separate 2- and 3-speaker mixtures. The proposed model also exhibits strong performance and generalizability at counting sources and separating mixtures with up to 5 speakers.

{{</citation>}}


### (18/109) Post-Training Embedding Alignment for Decoupling Enrollment and Runtime Speaker Recognition Models (Chenyang Gao et al., 2024)

{{<citation>}}

Chenyang Gao, Brecht Desplanques, Chelsea J. -T. Ju, Aman Chadha, Andreas Stolcke. (2024)  
**Post-Training Embedding Alignment for Decoupling Enrollment and Runtime Speaker Recognition Models**  

---
Primary Category: eess.AS  
Categories: cs-LG, cs-SD, eess-AS, eess.AS  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2401.12440v1)  

---


**ABSTRACT**  
Automated speaker identification (SID) is a crucial step for the personalization of a wide range of speech-enabled services. Typical SID systems use a symmetric enrollment-verification framework with a single model to derive embeddings both offline for voice profiles extracted from enrollment utterances, and online from runtime utterances. Due to the distinct circumstances of enrollment and runtime, such as different computation and latency constraints, several applications would benefit from an asymmetric enrollment-verification framework that uses different models for enrollment and runtime embedding generation. To support this asymmetric SID where each of the two models can be updated independently, we propose using a lightweight neural network to map the embeddings from the two independent models to a shared speaker embedding space. Our results show that this approach significantly outperforms cosine scoring in a shared speaker logit space for models that were trained with a contrastive loss on large datasets with many speaker identities. This proposed Neural Embedding Speaker Space Alignment (NESSA) combined with an asymmetric update of only one of the models delivers at least 60% of the performance gain achieved by updating both models in the standard symmetric SID approach.

{{</citation>}}


## cs.CY (5)



### (19/109) Unsocial Intelligence: a Pluralistic, Democratic, and Participatory Investigation of AGI Discourse (Borhane Blili-Hamelin et al., 2024)

{{<citation>}}

Borhane Blili-Hamelin, Leif Hancox-Li, Andrew Smart. (2024)  
**Unsocial Intelligence: a Pluralistic, Democratic, and Participatory Investigation of AGI Discourse**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.13142v1)  

---


**ABSTRACT**  
Dreams of machines that rival human intelligence have shaped the field of AI since its inception. Yet there remains no agreed-upon conception of what human-level AI or artificial general intelligence (AGI) means. We investigate key social, political, and ethical assumptions made by influential conceptions of AGI and human-level AI. We then draw on feminist, STS, and social science scholarship on the political and social character of intelligence in both humans and machines to defend a pluralistic, democratic, and participatory conception of the topic. We argue that framing AGI or human-level AI as a technical or value-neutral topic leads to political, ethical, and epistemic harm. AGI should not be developed without explicit attention to the values they encode, the people they include or exclude, and a view toward epistemic justice.

{{</citation>}}


### (20/109) Visibility into AI Agents (Alan Chan et al., 2024)

{{<citation>}}

Alan Chan, Carson Ezell, Max Kaufmann, Kevin Wei, Lewis Hammond, Herbie Bradley, Emma Bluemke, Nitarshan Rajkumar, David Krueger, Noam Kolt, Lennart Heim, Markus Anderljung. (2024)  
**Visibility into AI Agents**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.13138v2)  

---


**ABSTRACT**  
Increased delegation of commercial, scientific, governmental, and personal activities to AI agents -- systems capable of pursuing complex goals with limited supervision -- may exacerbate existing societal risks and introduce new risks. Understanding and mitigating these risks involves critically evaluating existing governance structures, revising and adapting these structures where needed, and ensuring accountability of key stakeholders. Information about where, why, how, and by whom certain AI agents are used, which we refer to as visibility, is critical to these objectives. In this paper, we assess three categories of measures to increase visibility into AI agents: agent identifiers, real-time monitoring, and activity logging. For each, we outline potential implementations that vary in intrusiveness and informativeness. We analyze how the measures apply across a spectrum of centralized through decentralized deployment contexts, accounting for various actors in the supply chain including hardware and software service providers. Finally, we discuss the implications of our measures for privacy and concentration of power. Further work into understanding the measures and mitigating their negative impacts can help to build a foundation for the governance of AI agents.

{{</citation>}}


### (21/109) No AI After Auschwitz? Bridging AI and Memory Ethics in the Context of Information Retrieval of Genocide-Related Information (Mykola Makhortykh, 2024)

{{<citation>}}

Mykola Makhortykh. (2024)  
**No AI After Auschwitz? Bridging AI and Memory Ethics in the Context of Information Retrieval of Genocide-Related Information**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI, Information Retrieval  
[Paper Link](http://arxiv.org/abs/2401.13079v1)  

---


**ABSTRACT**  
The growing application of artificial intelligence (AI) in the field of information retrieval (IR) affects different domains, including cultural heritage. By facilitating organisation and retrieval of large volumes of heritage-related content, AI-driven IR systems inform users about a broad range of historical phenomena, including genocides (e.g. the Holocaust). However, it is currently unclear to what degree IR systems are capable of dealing with multiple ethical challenges associated with the curation of genocide-related information. To address this question, this chapter provides an overview of ethical challenges associated with the human curation of genocide-related information using a three-part framework inspired by Belmont criteria (i.e. curation challenges associated with respect for individuals, beneficence and justice/fairness). Then, the chapter discusses to what degree the above-mentioned challenges are applicable to the ways in which AI-driven IR systems deal with genocide-related information and what can be the potential ways of bridging AI and memory ethics in this context.

{{</citation>}}


### (22/109) Towards Risk Analysis of the Impact of AI on the Deliberate Biological Threat Landscape (Matthew E. Walsh, 2024)

{{<citation>}}

Matthew E. Walsh. (2024)  
**Towards Risk Analysis of the Impact of AI on the Deliberate Biological Threat Landscape**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.12755v1)  

---


**ABSTRACT**  
The perception that the convergence of biological engineering and artificial intelligence (AI) could enable increased biorisk has recently drawn attention to the governance of biotechnology and artificial intelligence. The 2023 Executive Order, Executive Order on the Safe, Secure, and Trustworthy Development and Use of Artificial Intelligence, requires an assessment of how artificial intelligence can increase biorisk. Within this perspective, we present a simplistic framework for evaluating biorisk and demonstrate how this framework falls short in achieving actionable outcomes for a biorisk manager. We then suggest a potential path forward that builds upon existing risk characterization work and justify why characterization efforts of AI-enabled tools for engineering biology is needed.

{{</citation>}}


### (23/109) 'The teachers are confused as well': A Multiple-Stakeholder Ethics Discussion on Large Language Models in Computing Education (Kyrie Zhixuan Zhou et al., 2024)

{{<citation>}}

Kyrie Zhixuan Zhou, Zachary Kilhoffer, Madelyn Rose Sanfilippo, Ted Underwood, Ece Gumusel, Mengyi Wei, Abhinav Choudhry, Jinjun Xiong. (2024)  
**'The teachers are confused as well': A Multiple-Stakeholder Ethics Discussion on Large Language Models in Computing Education**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs-HC, cs.CY  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.12453v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) are advancing quickly and impacting people's lives for better or worse. In higher education, concerns have emerged such as students' misuse of LLMs and degraded education outcomes. To unpack the ethical concerns of LLMs for higher education, we conducted a case study consisting of stakeholder interviews (n=20) in higher education computer science. We found that students use several distinct mental models to interact with LLMs - LLMs serve as a tool for (a) writing, (b) coding, and (c) information retrieval, which differ somewhat in ethical considerations. Students and teachers brought up ethical issues that directly impact them, such as inaccurate LLM responses, hallucinations, biases, privacy leakage, and academic integrity issues. Participants emphasized the necessity of guidance and rules for the use of LLMs in higher education, including teaching digital literacy, rethinking education, and having cautious and contextual policies. We reflect on the ethical challenges and propose solutions.

{{</citation>}}


## cs.CL (28)



### (24/109) The Language Barrier: Dissecting Safety Challenges of LLMs in Multilingual Contexts (Lingfeng Shen et al., 2024)

{{<citation>}}

Lingfeng Shen, Weiting Tan, Sihao Chen, Yunmo Chen, Jingyu Zhang, Haoran Xu, Boyuan Zheng, Philipp Koehn, Daniel Khashabi. (2024)  
**The Language Barrier: Dissecting Safety Challenges of LLMs in Multilingual Contexts**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Multilingual  
[Paper Link](http://arxiv.org/abs/2401.13136v1)  

---


**ABSTRACT**  
As the influence of large language models (LLMs) spans across global communities, their safety challenges in multilingual settings become paramount for alignment research. This paper examines the variations in safety challenges faced by LLMs across different languages and discusses approaches to alleviating such concerns. By comparing how state-of-the-art LLMs respond to the same set of malicious prompts written in higher- vs. lower-resource languages, we observe that (1) LLMs tend to generate unsafe responses much more often when a malicious prompt is written in a lower-resource language, and (2) LLMs tend to generate more irrelevant responses to malicious prompts in lower-resource languages. To understand where the discrepancy can be attributed, we study the effect of instruction tuning with reinforcement learning from human feedback (RLHF) or supervised finetuning (SFT) on the HH-RLHF dataset. Surprisingly, while training with high-resource languages improves model alignment, training in lower-resource languages yields minimal improvement. This suggests that the bottleneck of cross-lingual alignment is rooted in the pretraining stage. Our findings highlight the challenges in cross-lingual LLM safety, and we hope they inform future research in this direction.

{{</citation>}}


### (25/109) Analyzing COVID-19 Vaccination Sentiments in Nigerian Cyberspace: Insights from a Manually Annotated Twitter Dataset (Ibrahim Said Ahmad et al., 2024)

{{<citation>}}

Ibrahim Said Ahmad, Lukman Jibril Aliyu, Abubakar Auwal Khalid, Saminu Muhammad Aliyu, Shamsuddeen Hassan Muhammad, Idris Abdulmumin, Bala Mairiga Abduljalil, Bello Shehu Bello, Amina Imam Abubakar. (2024)  
**Analyzing COVID-19 Vaccination Sentiments in Nigerian Cyberspace: Insights from a Manually Annotated Twitter Dataset**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SI, cs.CL  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2401.13133v1)  

---


**ABSTRACT**  
Numerous successes have been achieved in combating the COVID-19 pandemic, initially using various precautionary measures like lockdowns, social distancing, and the use of face masks. More recently, various vaccinations have been developed to aid in the prevention or reduction of the severity of the COVID-19 infection. Despite the effectiveness of the precautionary measures and the vaccines, there are several controversies that are massively shared on social media platforms like Twitter. In this paper, we explore the use of state-of-the-art transformer-based language models to study people's acceptance of vaccines in Nigeria. We developed a novel dataset by crawling multi-lingual tweets using relevant hashtags and keywords. Our analysis and visualizations revealed that most tweets expressed neutral sentiments about COVID-19 vaccines, with some individuals expressing positive views, and there was no strong preference for specific vaccine types, although Moderna received slightly more positive sentiment. We also found out that fine-tuning a pre-trained LLM with an appropriate dataset can yield competitive results, even if the LLM was not initially pre-trained on the specific language of that dataset.

{{</citation>}}


### (26/109) Towards Trustable Language Models: Investigating Information Quality of Large Language Models (Rick Rejeleene et al., 2024)

{{<citation>}}

Rick Rejeleene, Xiaowei Xu, John Talburt. (2024)  
**Towards Trustable Language Models: Investigating Information Quality of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.13086v1)  

---


**ABSTRACT**  
Large language models (LLM) are generating information at a rapid pace, requiring users to increasingly rely and trust the data. Despite remarkable advances of LLM, Information generated by LLM is not completely trustworthy, due to challenges in information quality. Specifically, integrity of Information quality decreases due to unreliable, biased, tokenization during pre-training of LLM. Moreover, due to decreased information quality issues, has led towards hallucination, fabricated information. Unreliable information can lead towards flawed decisions in businesses, which impacts economic activity. In this work, we introduce novel mathematical information quality evaluation of LLM, we furthermore analyze and highlight information quality challenges, scaling laws to systematically scale language models.

{{</citation>}}


### (27/109) IndiText Boost: Text Augmentation for Low Resource India Languages (Onkar Litake et al., 2024)

{{<citation>}}

Onkar Litake, Niraj Yagnik, Shreyas Labhsetwar. (2024)  
**IndiText Boost: Text Augmentation for Low Resource India Languages**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Augmentation, Text Generation  
[Paper Link](http://arxiv.org/abs/2401.13085v1)  

---


**ABSTRACT**  
Text Augmentation is an important task for low-resource languages. It helps deal with the problem of data scarcity. A data augmentation strategy is used to deal with the problem of data scarcity. Through the years, much work has been done on data augmentation for the English language. In contrast, very less work has been done on Indian languages. This is contrary to the fact that data augmentation is used to deal with data scarcity. In this work, we focus on implementing techniques like Easy Data Augmentation, Back Translation, Paraphrasing, Text Generation using LLMs, and Text Expansion using LLMs for text classification on different languages. We focus on 6 Indian languages namely: Sindhi, Marathi, Hindi, Gujarati, Telugu, and Sanskrit. According to our knowledge, no such work exists for text augmentation on Indian languages. We carry out binary as well as multi-class text classification to make our results more comparable. We get surprising results as basic data augmentation techniques surpass LLMs.

{{</citation>}}


### (28/109) TCE at Qur'an QA 2023 Shared Task: Low Resource Enhanced Transformer-based Ensemble Approach for Qur'anic QA (Mohammed Alaa Elkomy et al., 2024)

{{<citation>}}

Mohammed Alaa Elkomy, Amany Sarhan. (2024)  
**TCE at Qur'an QA 2023 Shared Task: Low Resource Enhanced Transformer-based Ensemble Approach for Qur'anic QA**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: QA, Transformer  
[Paper Link](http://arxiv.org/abs/2401.13060v1)  

---


**ABSTRACT**  
In this paper, we present our approach to tackle Qur'an QA 2023 shared tasks A and B. To address the challenge of low-resourced training data, we rely on transfer learning together with a voting ensemble to improve prediction stability across multiple runs. Additionally, we employ different architectures and learning mechanisms for a range of Arabic pre-trained transformer-based models for both tasks. To identify unanswerable questions, we propose using a thresholding mechanism. Our top-performing systems greatly surpass the baseline performance on the hidden split, achieving a MAP score of 25.05% for task A and a partial Average Precision (pAP) of 57.11% for task B.

{{</citation>}}


### (29/109) In-Context Language Learning: Arhitectures and Algorithms (Ekin Akyürek et al., 2024)

{{<citation>}}

Ekin Akyürek, Bailin Wang, Yoon Kim, Jacob Andreas. (2024)  
**In-Context Language Learning: Arhitectures and Algorithms**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.12973v1)  

---


**ABSTRACT**  
Large-scale neural language models exhibit a remarkable capacity for in-context learning (ICL): they can infer novel functions from datasets provided as input. Most of our current understanding of when and how ICL arises comes from LMs trained on extremely simple learning problems like linear regression and associative recall. There remains a significant gap between these model problems and the "real" ICL exhibited by LMs trained on large text corpora, which involves not just retrieval and function approximation but free-form generation of language and other structured outputs. In this paper, we study ICL through the lens of a new family of model problems we term in context language learning (ICLL). In ICLL, LMs are presented with a set of strings from a formal language, and must generate additional strings from the same language. We focus on in-context learning of regular languages generated by random finite automata. We evaluate a diverse set of neural sequence models (including several RNNs, Transformers, and state-space model variants) on regular ICLL tasks, aiming to answer three questions: (1) Which model classes are empirically capable of ICLL? (2) What algorithmic solutions do successful models implement to perform ICLL? (3) What architectural changes can improve ICLL in less performant models? We first show that Transformers significantly outperform neural sequence models with recurrent or convolutional representations on ICLL tasks. Next, we provide evidence that their ability to do so relies on specialized "n-gram heads" (higher-order variants of induction heads) that compute input-conditional next-token distributions. Finally, we show that hard-wiring these heads into recurrent and convolutional models improves performance not just on ICLL, but natural language modeling -- improving the perplexity of 340M-parameter models by up to 1.14 points (6.7%) on the SlimPajama dataset.

{{</citation>}}


### (30/109) Raidar: geneRative AI Detection viA Rewriting (Chengzhi Mao et al., 2024)

{{<citation>}}

Chengzhi Mao, Carl Vondrick, Hao Wang, Junfeng Yang. (2024)  
**Raidar: geneRative AI Detection viA Rewriting**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.12970v1)  

---


**ABSTRACT**  
We find that large language models (LLMs) are more likely to modify human-written text than AI-generated text when tasked with rewriting. This tendency arises because LLMs often perceive AI-generated text as high-quality, leading to fewer modifications. We introduce a method to detect AI-generated content by prompting LLMs to rewrite text and calculating the editing distance of the output. We dubbed our geneRative AI Detection viA Rewriting method Raidar. Raidar significantly improves the F1 detection scores of existing AI content detection models -- both academic and commercial -- across various domains, including News, creative writing, student essays, code, Yelp reviews, and arXiv papers, with gains of up to 29 points. Operating solely on word symbols without high-dimensional features, our method is compatible with black box LLMs, and is inherently robust on new content. Our results illustrate the unique imprint of machine-generated text through the lens of the machines themselves.

{{</citation>}}


### (31/109) Meta-Prompting: Enhancing Language Models with Task-Agnostic Scaffolding (Mirac Suzgun et al., 2024)

{{<citation>}}

Mirac Suzgun, Adam Tauman Kalai. (2024)  
**Meta-Prompting: Enhancing Language Models with Task-Agnostic Scaffolding**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-HC, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2401.12954v1)  

---


**ABSTRACT**  
We introduce meta-prompting, an effective scaffolding technique designed to enhance the functionality of language models (LMs). This approach transforms a single LM into a multi-faceted conductor, adept at managing and integrating multiple independent LM queries. By employing high-level instructions, meta-prompting guides the LM to break down complex tasks into smaller, more manageable subtasks. These subtasks are then handled by distinct "expert" instances of the same LM, each operating under specific, tailored instructions. Central to this process is the LM itself, in its role as the conductor, which ensures seamless communication and effective integration of the outputs from these expert models. It additionally employs its inherent critical thinking and robust verification processes to refine and authenticate the end result. This collaborative prompting approach empowers a single LM to simultaneously act as a comprehensive orchestrator and a panel of diverse experts, significantly enhancing its performance across a wide array of tasks. The zero-shot, task-agnostic nature of meta-prompting greatly simplifies user interaction by obviating the need for detailed, task-specific instructions. Furthermore, our research demonstrates the seamless integration of external tools, such as a Python interpreter, into the meta-prompting framework, thereby broadening its applicability and utility. Through rigorous experimentation with GPT-4, we establish the superiority of meta-prompting over conventional scaffolding methods: When averaged across all tasks, including the Game of 24, Checkmate-in-One, and Python Programming Puzzles, meta-prompting, augmented with a Python interpreter functionality, surpasses standard prompting by 17.1%, expert (dynamic) prompting by 17.3%, and multipersona prompting by 15.2%.

{{</citation>}}


### (32/109) Transformer-Based Models Are Not Yet Perfect At Learning to Emulate Structural Recursion (Dylan Zhang et al., 2024)

{{<citation>}}

Dylan Zhang, Curt Tigges, Zory Zhang, Stella Biderman, Maxim Raginsky, Talia Ringer. (2024)  
**Transformer-Based Models Are Not Yet Perfect At Learning to Emulate Structural Recursion**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-FL, cs-LO, cs-PL, cs.CL  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.12947v1)  

---


**ABSTRACT**  
This paper investigates the ability of transformer-based models to learn structural recursion from examples. Recursion is a universal concept in both natural and formal languages. Structural recursion is central to the programming language and formal mathematics tasks where symbolic tools currently excel beyond neural models, such as inferring semantic relations between datatypes and emulating program behavior. We introduce a general framework that nicely connects the abstract concepts of structural recursion in the programming language domain to concrete sequence modeling problems and learned models' behavior. The framework includes a representation that captures the general \textit{syntax} of structural recursion, coupled with two different frameworks for understanding their \textit{semantics} -- one that is more natural from a programming languages perspective and one that helps bridge that perspective with a mechanistic understanding of the underlying transformer architecture.   With our framework as a powerful conceptual tool, we identify different issues under various set-ups. The models trained to emulate recursive computations cannot fully capture the recursion yet instead fit short-cut algorithms and thus cannot solve certain edge cases that are under-represented in the training distribution. In addition, it is difficult for state-of-the-art large language models (LLMs) to mine recursive rules from in-context demonstrations. Meanwhile, these LLMs fail in interesting ways when emulating reduction (step-wise computation) of the recursive function.

{{</citation>}}


### (33/109) Multicultural Name Recognition For Previously Unseen Names (Alexandra Loessberg-Zahl, 2024)

{{<citation>}}

Alexandra Loessberg-Zahl. (2024)  
**Multicultural Name Recognition For Previously Unseen Names**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: LSTM, NER, Named Entity Recognition  
[Paper Link](http://arxiv.org/abs/2401.12941v1)  

---


**ABSTRACT**  
State of the art Named Entity Recognition (NER) models have achieved an impressive ability to extract common phrases from text that belong to labels such as location, organization, time, and person. However, typical NER systems that rely on having seen a specific entity in their training data in order to label an entity perform poorly on rare or unseen entities ta in order to label an entity perform poorly on rare or unseen entities (Derczynski et al., 2017). This paper attempts to improve recognition of person names, a diverse category that can grow any time someone is born or changes their name. In order for downstream tasks to not exhibit bias based on cultural background, a model should perform well on names from a variety of backgrounds. In this paper I experiment with the training data and input structure of an English Bi-LSTM name recognition model. I look at names from 103 countries to compare how well the model performs on names from different cultures, specifically in the context of a downstream task where extracted names will be matched to information on file. I find that a model with combined character and word input outperforms word-only models and may improve on accuracy compared to classical NER models that are not geared toward identifying unseen entity values.

{{</citation>}}


### (34/109) From Understanding to Utilization: A Survey on Explainability for Large Language Models (Haoyan Luo et al., 2024)

{{<citation>}}

Haoyan Luo, Lucia Specia. (2024)  
**From Understanding to Utilization: A Survey on Explainability for Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: LLaMA, Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2401.12874v1)  

---


**ABSTRACT**  
This survey paper delves into the burgeoning field of explainability for Large Language Models (LLMs), a critical yet challenging aspect of natural language processing. With LLMs playing a pivotal role in various applications, their "black-box" nature raises concerns about transparency and ethical use. This paper emphasizes the necessity for enhanced explainability in LLMs, addressing both the general public's trust and the technical community's need for a deeper understanding of these models. We concentrate on pre-trained Transformer-based LLMs, such as LLaMA, which present unique interpretability challenges due to their scale and complexity. Our review categorizes existing explainability methods and discusses their application in improving model transparency and reliability. We also discuss representative evaluation methods, highlighting their strengths and limitations. The goal of this survey is to bridge the gap between theoretical understanding and practical application, offering insights for future research and development in the field of LLM explainability.

{{</citation>}}


### (35/109) Improving Machine Translation with Human Feedback: An Exploration of Quality Estimation as a Reward Model (Zhiwei He et al., 2024)

{{<citation>}}

Zhiwei He, Xing Wang, Wenxiang Jiao, Zhuosheng Zhang, Rui Wang, Shuming Shi, Zhaopeng Tu. (2024)  
**Improving Machine Translation with Human Feedback: An Exploration of Quality Estimation as a Reward Model**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Machine Translation  
[Paper Link](http://arxiv.org/abs/2401.12873v1)  

---


**ABSTRACT**  
Insufficient modeling of human preferences within the reward model is a major obstacle for leveraging human feedback to improve translation quality. Fortunately, quality estimation (QE), which predicts the quality of a given translation without reference, has achieved impressive alignment with human evaluations in the last two years. In this work, we investigate the potential of employing the QE model as the reward model (the QE-based reward model) to predict human preferences for feedback training. We first identify the overoptimization problem during QE-based feedback training, manifested as an increase in reward while translation quality declines. We examine the problem and argue that the vulnerability of the QE model might lead to high rewards for incorrect translations, resulting in overoptimization and error propagation. To address the problem, we adopt a simple yet effective method that uses heuristic rules to detect the incorrect translations and assigns a penalty term to the QE-based rewards for the detected incorrect translations. Experimental results show that the proposed QE-based feedback training achieves consistent and significant improvements across various settings, further verified through human preference studies. Our subsequent analysis demonstrates the high data efficiency of the proposed QE-based feedback training: the proposed approach using a small amount of monolingual data can outperform systems using larger parallel corpora.

{{</citation>}}


### (36/109) KAM-CoT: Knowledge Augmented Multimodal Chain-of-Thoughts Reasoning (Debjyoti Mondal et al., 2024)

{{<citation>}}

Debjyoti Mondal, Suraj Modi, Subhadarshi Panda, Rituraj Singh, Godawari Sudhakar Rao. (2024)  
**KAM-CoT: Knowledge Augmented Multimodal Chain-of-Thoughts Reasoning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-3.5, GPT-4, Knowledge Graph, Language Model, QA, Reasoning  
[Paper Link](http://arxiv.org/abs/2401.12863v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have demonstrated impressive performance in natural language processing tasks by leveraging chain of thought (CoT) that enables step-by-step thinking. Extending LLMs with multimodal capabilities is the recent interest, but incurs computational cost and requires substantial hardware resources. To address these challenges, we propose KAM-CoT a framework that integrates CoT reasoning, Knowledge Graphs (KGs), and multiple modalities for a comprehensive understanding of multimodal tasks. KAM-CoT adopts a two-stage training process with KG grounding to generate effective rationales and answers. By incorporating external knowledge from KGs during reasoning, the model gains a deeper contextual understanding reducing hallucinations and enhancing the quality of answers. This knowledge-augmented CoT reasoning empowers the model to handle questions requiring external context, providing more informed answers. Experimental findings show KAM-CoT outperforms the state-of-the-art methods. On the ScienceQA dataset, we achieve an average accuracy of 93.87%, surpassing GPT-3.5 (75.17%) by 18% and GPT-4 (83.99%) by 10%. Remarkably, KAM-CoT achieves these results with only 280M trainable parameters at a time, demonstrating its cost-efficiency and effectiveness.

{{</citation>}}


### (37/109) Benchmarking LLMs via Uncertainty Quantification (Fanghua Ye et al., 2024)

{{<citation>}}

Fanghua Ye, Mingming Yang, Jianhui Pang, Longyue Wang, Derek F. Wong, Emine Yilmaz, Shuming Shi, Zhaopeng Tu. (2024)  
**Benchmarking LLMs via Uncertainty Quantification**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.12794v1)  

---


**ABSTRACT**  
The proliferation of open-source Large Language Models (LLMs) from various institutions has highlighted the urgent need for comprehensive evaluation methods. However, current evaluation platforms, such as the widely recognized HuggingFace open LLM leaderboard, neglect a crucial aspect -- uncertainty, which is vital for thoroughly assessing LLMs. To bridge this gap, we introduce a new benchmarking approach for LLMs that integrates uncertainty quantification. Our examination involves eight LLMs (LLM series) spanning five representative natural language processing tasks. Additionally, we introduce an uncertainty-aware evaluation metric, UAcc, which takes into account both prediction accuracy and prediction uncertainty. Our findings reveal that: I) LLMs with higher accuracy may exhibit lower certainty; II) Larger-scale LLMs may display greater uncertainty compared to their smaller counterparts; and III) Instruction-finetuning tends to increase the uncertainty of LLMs. By taking uncertainty into account, our new UAcc metric can either amplify or diminish the relative improvement of one LLM over another and may even change the relative ranking of two LLMs. These results underscore the significance of incorporating uncertainty in the evaluation of LLMs.

{{</citation>}}


### (38/109) Multilingual and Fully Non-Autoregressive ASR with Large Language Model Fusion: A Comprehensive Study (W. Ronny Huang et al., 2024)

{{<citation>}}

W. Ronny Huang, Cyril Allauzen, Tongzhou Chen, Kilol Gupta, Ke Hu, James Qin, Yu Zhang, Yongqiang Wang, Shuo-Yiin Chang, Tara N. Sainath. (2024)  
**Multilingual and Fully Non-Autoregressive ASR with Large Language Model Fusion: A Comprehensive Study**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: Language Model, Multilingual, PaLM  
[Paper Link](http://arxiv.org/abs/2401.12789v1)  

---


**ABSTRACT**  
In the era of large models, the autoregressive nature of decoding often results in latency serving as a significant bottleneck. We propose a non-autoregressive LM-fused ASR system that effectively leverages the parallelization capabilities of accelerator hardware. Our approach combines the Universal Speech Model (USM) and the PaLM 2 language model in per-segment scoring mode, achieving an average relative WER improvement across all languages of 10.8% on FLEURS and 3.6% on YouTube captioning. Furthermore, our comprehensive ablation study analyzes key parameters such as LLM size, context length, vocabulary size, fusion methodology. For instance, we explore the impact of LLM size ranging from 128M to 340B parameters on ASR performance. This study provides valuable insights into the factors influencing the effectiveness of practical large-scale LM-fused speech recognition systems.

{{</citation>}}


### (39/109) What the Weight?! A Unified Framework for Zero-Shot Knowledge Composition (Carolin Holtermann et al., 2024)

{{<citation>}}

Carolin Holtermann, Markus Frohmann, Navid Rekabsaz, Anne Lauscher. (2024)  
**What the Weight?! A Unified Framework for Zero-Shot Knowledge Composition**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: NLP, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2401.12756v2)  

---


**ABSTRACT**  
The knowledge encapsulated in a model is the core factor determining its final performance on downstream tasks. Much research in NLP has focused on efficient methods for storing and adapting different types of knowledge, e.g., in dedicated modularized structures, and on how to effectively combine these, e.g., by learning additional parameters. However, given the many possible options, a thorough understanding of the mechanisms involved in these compositions is missing, and hence it remains unclear which strategies to utilize. To address this research gap, we propose a novel framework for zero-shot module composition, which encompasses existing and some novel variations for selecting, weighting, and combining parameter modules under a single unified notion. Focusing on the scenario of domain knowledge and adapter layers, our framework provides a systematic unification of concepts, allowing us to conduct the first comprehensive benchmarking study of various zero-shot knowledge composition strategies. In particular, we test two module combination methods and five selection and weighting strategies for their effectiveness and efficiency in an extensive experimental setup. Our results highlight the efficacy of ensembling but also hint at the power of simple though often-ignored weighting methods. Further in-depth analyses allow us to understand the role of weighting vs. top-k selection, and show that, to a certain extent, the performance of adapter composition can even be predicted.

{{</citation>}}


### (40/109) A Comprehensive View of the Biases of Toxicity and Sentiment Analysis Methods Towards Utterances with African American English Expressions (Guilherme H. Resende et al., 2024)

{{<citation>}}

Guilherme H. Resende, Luiz F. Nery, Fabrício Benevenuto, Savvas Zannettou, Flavio Figueiredo. (2024)  
**A Comprehensive View of the Biases of Toxicity and Sentiment Analysis Methods Towards Utterances with African American English Expressions**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SI, cs.CL  
Keywords: AI, Bias, Google, NLP, Natural Language Processing, Sentiment Analysis, Twitter  
[Paper Link](http://arxiv.org/abs/2401.12720v1)  

---


**ABSTRACT**  
Language is a dynamic aspect of our culture that changes when expressed in different technologies/communities. Online social networks have enabled the diffusion and evolution of different dialects, including African American English (AAE). However, this increased usage is not without barriers. One particular barrier is how sentiment (Vader, TextBlob, and Flair) and toxicity (Google's Perspective and the open-source Detoxify) methods present biases towards utterances with AAE expressions. Consider Google's Perspective to understand bias. Here, an utterance such as ``All n*ggers deserve to die respectfully. The police murder us.'' it reaches a higher toxicity than ``African-Americans deserve to die respectfully. The police murder us.''. This score difference likely arises because the tool cannot understand the re-appropriation of the term ``n*gger''. One explanation for this bias is that AI models are trained on limited datasets, and using such a term in training data is more likely to appear in a toxic utterance. While this may be plausible, the tool will make mistakes regardless. Here, we study bias on two Web-based (YouTube and Twitter) datasets and two spoken English datasets. Our analysis shows how most models present biases towards AAE in most settings. We isolate the impact of AAE expression usage via linguistic control features from the Linguistic Inquiry and Word Count (LIWC) software, grammatical control features extracted via Part-of-Speech (PoS) tagging from Natural Language Processing (NLP) models, and the semantic of utterances by comparing sentence embeddings from recent language models. We present consistent results on how a heavy usage of AAE expressions may cause the speaker to be considered substantially more toxic, even when speaking about nearly the same subject. Our study complements similar analyses focusing on small datasets and/or one method only.

{{</citation>}}


### (41/109) Context Matters: Pushing the Boundaries of Open-Ended Answer Generation with Graph-Structured Knowledge Context (Somnath Banerjee et al., 2024)

{{<citation>}}

Somnath Banerjee, Amruit Sahoo, Sayan Layek, Avik Dutta, Rima Hazra, Animesh Mukherjee. (2024)  
**Context Matters: Pushing the Boundaries of Open-Ended Answer Generation with Graph-Structured Knowledge Context**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2401.12671v1)  

---


**ABSTRACT**  
In the continuously advancing AI landscape, crafting context-rich and meaningful responses via Large Language Models (LLMs) is essential. Researchers are becoming more aware of the challenges that LLMs with fewer parameters encounter when trying to provide suitable answers to open-ended questions. To address these hurdles, the integration of cutting-edge strategies, augmentation of rich external domain knowledge to LLMs, offers significant improvements. This paper introduces a novel framework that combines graph-driven context retrieval in conjunction to knowledge graphs based enhancement, honing the proficiency of LLMs, especially in domain specific community question answering platforms like AskUbuntu, Unix, and ServerFault. We conduct experiments on various LLMs with different parameter sizes to evaluate their ability to ground knowledge and determine factual accuracy in answers to open-ended questions. Our methodology GraphContextGen consistently outperforms dominant text-based retrieval systems, demonstrating its robustness and adaptability to a larger number of use cases. This advancement highlights the importance of pairing context rich data retrieval with LLMs, offering a renewed approach to knowledge sourcing and generation in AI systems. We also show that, due to rich contextual data retrieval, the crucial entities, along with the generated answer, remain factually coherent with the gold answer.

{{</citation>}}


### (42/109) SLANG: New Concept Comprehension of Large Language Models (Lingrui Mei et al., 2024)

{{<citation>}}

Lingrui Mei, Shenghua Liu, Yiwei Wang, Baolong Bi, Xueqi Chen. (2024)  
**SLANG: New Concept Comprehension of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.12585v1)  

---


**ABSTRACT**  
The dynamic nature of language, particularly evident in the realm of slang and memes on the Internet, poses serious challenges to the adaptability of large language models (LLMs). Traditionally anchored to static datasets, these models often struggle to keep up with the rapid linguistic evolution characteristic of online communities. This research addresses the critical need to bridge this gap, aiming to enhance LLMs' comprehension of evolving new concepts on the internet, without the high cost and impracticality of continual retraining. To address this issue, we propose a new benchmark $\textbf{SLANG}$ to assess LLMs' proficiency in comprehending emerging linguistic trends and a baseline approach $\textbf{FOCUS}$, which uses causal inference to enhance LLMs to understand new phrases and usage patterns. This approach involves scrutinizing real-world instances of linguistic shifts, serving as contextual beacons, to form more precise and contextually relevant connections between newly emerging expressions and their intended meanings. The empirical analysis shows that our causal inference-based approach outperforms the traditional models in terms of precision and relevance in the interpretation of Internet slang and memes.

{{</citation>}}


### (43/109) LLMCheckup: Conversational Examination of Large Language Models via Interpretability Tools (Qianli Wang et al., 2024)

{{<citation>}}

Qianli Wang, Tatiana Anikina, Nils Feldhus, Josef van Genabith, Leonhard Hennig, Sebastian Möller. (2024)  
**LLMCheckup: Conversational Examination of Large Language Models via Interpretability Tools**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2401.12576v1)  

---


**ABSTRACT**  
Interpretability tools that offer explanations in the form of a dialogue have demonstrated their efficacy in enhancing users' understanding, as one-off explanations may occasionally fall short in providing sufficient information to the user. Current solutions for dialogue-based explanations, however, require many dependencies and are not easily transferable to tasks they were not designed for. With LLMCheckup, we present an easily accessible tool that allows users to chat with any state-of-the-art large language model (LLM) about its behavior. We enable LLMs to generate all explanations by themselves and take care of intent recognition without fine-tuning, by connecting them with a broad spectrum of Explainable AI (XAI) tools, e.g. feature attributions, embedding-based similarity, and prompting strategies for counterfactual and rationale generation. LLM (self-)explanations are presented as an interactive dialogue that supports follow-up questions and generates suggestions. LLMCheckup provides tutorials for operations available in the system, catering to individuals with varying levels of expertise in XAI and supports multiple input modalities. We introduce a new parsing strategy called multi-prompt parsing substantially enhancing the parsing accuracy of LLMs. Finally, we showcase the tasks of fact checking and commonsense question answering.

{{</citation>}}


### (44/109) Automated Fact-Checking of Climate Change Claims with Large Language Models (Markus Leippold et al., 2024)

{{<citation>}}

Markus Leippold, Saeid Ashraf Vaghefi, Dominik Stammbach, Veruska Muccione, Julia Bingler, Jingwei Ni, Chiara Colesanti-Senni, Tobias Wekhof, Tobias Schimanski, Glen Gostlow, Tingyu Yu, Juerg Luterbacher, Christian Huggel. (2024)  
**Automated Fact-Checking of Climate Change Claims with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Fact-Checking, Language Model  
[Paper Link](http://arxiv.org/abs/2401.12566v1)  

---


**ABSTRACT**  
This paper presents Climinator, a novel AI-based tool designed to automate the fact-checking of climate change claims. Utilizing an array of Large Language Models (LLMs) informed by authoritative sources like the IPCC reports and peer-reviewed scientific literature, Climinator employs an innovative Mediator-Advocate framework. This design allows Climinator to effectively synthesize varying scientific perspectives, leading to robust, evidence-based evaluations. Our model demonstrates remarkable accuracy when testing claims collected from Climate Feedback and Skeptical Science. Notably, when integrating an advocate with a climate science denial perspective in our framework, Climinator's iterative debate process reliably converges towards scientific consensus, underscoring its adeptness at reconciling diverse viewpoints into science-based, factual conclusions. While our research is subject to certain limitations and necessitates careful interpretation, our approach holds significant potential. We hope to stimulate further research and encourage exploring its applicability in other contexts, including political fact-checking and legal domains.

{{</citation>}}


### (45/109) BiTA: Bi-Directional Tuning for Lossless Acceleration in Large Language Models (Feng Lin et al., 2024)

{{<citation>}}

Feng Lin, Hanling Yi, Hongbin Li, Yifan Yang, Xiaotian Yu, Guangming Lu, Rong Xiao. (2024)  
**BiTA: Bi-Directional Tuning for Lossless Acceleration in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2401.12522v2)  

---


**ABSTRACT**  
Large language models (LLMs) commonly employ autoregressive generation during inference, leading to high memory bandwidth demand and consequently extended latency. To mitigate this inefficiency, we present Bi-directional Tuning for lossless Acceleration (BiTA), an innovative method expediting LLMs via streamlined semi-autoregressive generation and draft verification. Inspired by the concept of prompt tuning, we enhance LLMs with a parameter-efficient design called bi-directional tuning for the capability in semi-autoregressive generation. Employing efficient tree-based decoding, the models perform draft candidate generation and verification in parallel, ensuring outputs identical to their autoregressive counterparts under greedy sampling. BiTA serves as a lightweight plug-in module, seamlessly boosting the inference efficiency of existing LLMs without requiring additional assistance models or incurring significant extra memory costs. Applying the proposed BiTA, LLaMA-2-70B-Chat achieves a 2.7$\times$ speedup on the MT-Bench benchmark. Extensive experiments confirm our method surpasses state-of-the-art acceleration techniques.

{{</citation>}}


### (46/109) Key Information Retrieval to Classify the Unstructured Data Content of Preferential Trade Agreements (Jiahui Zhao et al., 2024)

{{<citation>}}

Jiahui Zhao, Ziyi Meng, Stepan Gordeev, Zijie Pan, Dongjin Song, Sandro Steinbach, Caiwen Ding. (2024)  
**Key Information Retrieval to Classify the Unstructured Data Content of Preferential Trade Agreements**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-IR, cs-LG, cs.CL  
Keywords: BERT, Information Retrieval, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.12520v1)  

---


**ABSTRACT**  
With the rapid proliferation of textual data, predicting long texts has emerged as a significant challenge in the domain of natural language processing. Traditional text prediction methods encounter substantial difficulties when grappling with long texts, primarily due to the presence of redundant and irrelevant information, which impedes the model's capacity to capture pivotal insights from the text. To address this issue, we introduce a novel approach to long-text classification and prediction. Initially, we employ embedding techniques to condense the long texts, aiming to diminish the redundancy therein. Subsequently,the Bidirectional Encoder Representations from Transformers (BERT) embedding method is utilized for text classification training. Experimental outcomes indicate that our method realizes considerable performance enhancements in classifying long texts of Preferential Trade Agreements. Furthermore, the condensation of text through embedding methods not only augments prediction accuracy but also substantially reduces computational complexity. Overall, this paper presents a strategy for long-text prediction, offering a valuable reference for researchers and engineers in the natural language processing sphere.

{{</citation>}}


### (47/109) Comparing Human-Centered Language Modeling: Is it Better to Model Groups, Individual Traits, or Both? (Nikita Soni et al., 2024)

{{<citation>}}

Nikita Soni, Niranjan Balasubramanian, H. Andrew Schwartz, Dirk Hovy. (2024)  
**Comparing Human-Centered Language Modeling: Is it Better to Model Groups, Individual Traits, or Both?**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.12492v1)  

---


**ABSTRACT**  
Natural language processing has made progress in incorporating human context into its models, but whether it is more effective to use group-wise attributes (e.g., over-45-year-olds) or model individuals remains open. Group attributes are technically easier but coarse: not all 45-year-olds write the same way. In contrast, modeling individuals captures the complexity of each person's identity. It allows for a more personalized representation, but we may have to model an infinite number of users and require data that may be impossible to get. We compare modeling human context via group attributes, individual users, and combined approaches. Combining group and individual features significantly benefits user-level regression tasks like age estimation or personality assessment from a user's documents. Modeling individual users significantly improves the performance of single document-level classification tasks like stance and topic detection. We also find that individual-user modeling does well even without user's historical data.

{{</citation>}}


### (48/109) Assessing and Understanding Creativity in Large Language Models (Yunpu Zhao et al., 2024)

{{<citation>}}

Yunpu Zhao, Rui Zhang, Wenyi Li, Di Huang, Jiaming Guo, Shaohui Peng, Yifan Hao, Yuanbo Wen, Xing Hu, Zidong Du, Qi Guo, Ling Li, Yunji Chen. (2024)  
**Assessing and Understanding Creativity in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.12491v1)  

---


**ABSTRACT**  
In the field of natural language processing, the rapid development of large language model (LLM) has attracted more and more attention. LLMs have shown a high level of creativity in various tasks, but the methods for assessing such creativity are inadequate. The assessment of LLM creativity needs to consider differences from humans, requiring multi-dimensional measurement while balancing accuracy and efficiency. This paper aims to establish an efficient framework for assessing the level of creativity in LLMs. By adapting the modified Torrance Tests of Creative Thinking, the research evaluates the creative performance of various LLMs across 7 tasks, emphasizing 4 criteria including Fluency, Flexibility, Originality, and Elaboration. In this context, we develop a comprehensive dataset of 700 questions for testing and an LLM-based evaluation method. In addition, this study presents a novel analysis of LLMs' responses to diverse prompts and role-play situations. We found that the creativity of LLMs primarily falls short in originality, while excelling in elaboration. Besides, the use of prompts and the role-play settings of the model significantly influence creativity. Additionally, the experimental results also indicate that collaboration among multiple LLMs can enhance originality. Notably, our findings reveal a consensus between human evaluations and LLMs regarding the personality traits that influence creativity. The findings underscore the significant impact of LLM design on creativity and bridges artificial intelligence and human creativity, offering insights into LLMs' creativity and potential applications.

{{</citation>}}


### (49/109) Large Language Models are Superpositions of All Characters: Attaining Arbitrary Role-play via Self-Alignment (Keming Lu et al., 2024)

{{<citation>}}

Keming Lu, Bowen Yu, Chang Zhou, Jingren Zhou. (2024)  
**Large Language Models are Superpositions of All Characters: Attaining Arbitrary Role-play via Self-Alignment**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.12474v1)  

---


**ABSTRACT**  
Considerable efforts have been invested in augmenting the role-playing proficiency of open-source large language models (LLMs) by emulating proprietary counterparts. Nevertheless, we posit that LLMs inherently harbor role-play capabilities, owing to the extensive knowledge of characters and potential dialogues ingrained in their vast training corpora. Thus, in this study, we introduce Ditto, a self-alignment method for role-play. Ditto capitalizes on character knowledge, encouraging an instruction-following LLM to simulate role-play dialogues as a variant of reading comprehension. This method creates a role-play training set comprising 4,000 characters, surpassing the scale of currently available datasets by tenfold regarding the number of roles. Subsequently, we fine-tune the LLM using this self-generated dataset to augment its role-playing capabilities. Upon evaluating our meticulously constructed and reproducible role-play benchmark and the roleplay subset of MT-Bench, Ditto, in various parameter scales, consistently maintains a consistent role identity and provides accurate role-specific knowledge in multi-turn role-play conversations. Notably, it outperforms all open-source role-play baselines, showcasing performance levels comparable to advanced proprietary chatbots. Furthermore, we present the first comprehensive cross-supervision alignment experiment in the role-play domain, revealing that the intrinsic capabilities of LLMs confine the knowledge within role-play. Meanwhile, the role-play styles can be easily acquired with the guidance of smaller models. We open-source related resources at https://github.com/OFA-Sys/Ditto.

{{</citation>}}


### (50/109) Contrastive Learning in Distilled Models (Valerie Lim et al., 2024)

{{<citation>}}

Valerie Lim, Kai Wen Ng, Kenneth Lim. (2024)  
**Contrastive Learning in Distilled Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, Contrastive Learning, NLP, Natural Language Processing, Textual Similarity  
[Paper Link](http://arxiv.org/abs/2401.12472v1)  

---


**ABSTRACT**  
Natural Language Processing models like BERT can provide state-of-the-art word embeddings for downstream NLP tasks. However, these models yet to perform well on Semantic Textual Similarity, and may be too large to be deployed as lightweight edge applications. We seek to apply a suitable contrastive learning method based on the SimCSE paper, to a model architecture adapted from a knowledge distillation based model, DistilBERT, to address these two issues. Our final lightweight model DistilFace achieves an average of 72.1 in Spearman's correlation on STS tasks, a 34.2 percent improvement over BERT base.

{{</citation>}}


### (51/109) Fast Adversarial Training against Textual Adversarial Attacks (Yichen Yang et al., 2024)

{{<citation>}}

Yichen Yang, Xin Liu, Kun He. (2024)  
**Fast Adversarial Training against Textual Adversarial Attacks**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Adversarial Attack, Adversarial Training, BERT  
[Paper Link](http://arxiv.org/abs/2401.12461v1)  

---


**ABSTRACT**  
Many adversarial defense methods have been proposed to enhance the adversarial robustness of natural language processing models. However, most of them introduce additional pre-set linguistic knowledge and assume that the synonym candidates used by attackers are accessible, which is an ideal assumption. We delve into adversarial training in the embedding space and propose a Fast Adversarial Training (FAT) method to improve the model robustness in the synonym-unaware scenario from the perspective of single-step perturbation generation and perturbation initialization. Based on the observation that the adversarial perturbations crafted by single-step and multi-step gradient ascent are similar, FAT uses single-step gradient ascent to craft adversarial examples in the embedding space to expedite the training process. Based on the observation that the perturbations generated on the identical training sample in successive epochs are similar, FAT fully utilizes historical information when initializing the perturbation. Extensive experiments demonstrate that FAT significantly boosts the robustness of BERT models in the synonym-unaware scenario, and outperforms the defense baselines under various attacks with character-level and word-level modifications.

{{</citation>}}


## cs.AI (12)



### (52/109) XAI for All: Can Large Language Models Simplify Explainable AI? (Philip Mavrepis et al., 2024)

{{<citation>}}

Philip Mavrepis, Georgios Makridis, Georgios Fatouros, Vasileios Koukos, Maria Margarita Separdani, Dimosthenis Kyriazis. (2024)  
**XAI for All: Can Large Language Models Simplify Explainable AI?**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-HC, cs.AI  
Keywords: AI, ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2401.13110v1)  

---


**ABSTRACT**  
The field of Explainable Artificial Intelligence (XAI) often focuses on users with a strong technical background, making it challenging for non-experts to understand XAI methods. This paper presents "x-[plAIn]", a new approach to make XAI more accessible to a wider audience through a custom Large Language Model (LLM), developed using ChatGPT Builder. Our goal was to design a model that can generate clear, concise summaries of various XAI methods, tailored for different audiences, including business professionals and academics. The key feature of our model is its ability to adapt explanations to match each audience group's knowledge level and interests. Our approach still offers timely insights, facilitating the decision-making process by the end users. Results from our use-case studies show that our model is effective in providing easy-to-understand, audience-specific explanations, regardless of the XAI method used. This adaptability improves the accessibility of XAI, bridging the gap between complex AI technologies and their practical applications. Our findings indicate a promising direction for LLMs in making advanced AI concepts more accessible to a diverse range of users.

{{</citation>}}


### (53/109) Truck Parking Usage Prediction with Decomposed Graph Neural Networks (Rei Tamaru et al., 2024)

{{<citation>}}

Rei Tamaru, Yang Cheng, Steven Parker, Ernie Perry, Bin Ran, Soyoung Ahn. (2024)  
**Truck Parking Usage Prediction with Decomposed Graph Neural Networks**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2401.12920v1)  

---


**ABSTRACT**  
Truck parking on freight corridors faces various challenges, such as insufficient parking spaces and compliance with Hour-of-Service (HOS) regulations. These constraints often result in unauthorized parking practices, causing safety concerns. To enhance the safety of freight operations, providing accurate parking usage prediction proves to be a cost-effective solution. Despite the existing research demonstrating satisfactory accuracy for predicting individual truck parking site usage, few approaches have been proposed for predicting usage with spatial dependencies of multiple truck parking sites. We present the Regional Temporal Graph Neural Network (RegT-GCN) as a predictive framework for assessing parking usage across the entire state to provide better truck parking information and mitigate unauthorized parking. The framework leverages the topological structures of truck parking site distributions and historical parking data to predict occupancy rates across a state. To achieve this, we introduce a Regional Decomposition approach, which effectively captures the geographical characteristics. We also introduce the spatial module working efficiently with the temporal module. Evaluation results demonstrate that the proposed model surpasses other baseline models, improving the performance by more than $20\%$ compared with the original model. The proposed model allows truck parking sites' percipience of the topological structures and provides higher performance.

{{</citation>}}


### (54/109) Red Teaming Visual Language Models (Mukai Li et al., 2024)

{{<citation>}}

Mukai Li, Lei Li, Yuwei Yin, Masood Ahmed, Zhenguang Liu, Qi Liu. (2024)  
**Red Teaming Visual Language Models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-CV, cs.AI  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2401.12915v1)  

---


**ABSTRACT**  
VLMs (Vision-Language Models) extend the capabilities of LLMs (Large Language Models) to accept multimodal inputs. Since it has been verified that LLMs can be induced to generate harmful or inaccurate content through specific test cases (termed as Red Teaming), how VLMs perform in similar scenarios, especially with their combination of textual and visual inputs, remains a question. To explore this problem, we present a novel red teaming dataset RTVLM, which encompasses 10 subtasks (e.g., image misleading, multi-modal jail-breaking, face fairness, etc) under 4 primary aspects (faithfulness, privacy, safety, fairness). Our RTVLM is the first red-teaming dataset to benchmark current VLMs in terms of these 4 different aspects. Detailed analysis shows that 10 prominent open-sourced VLMs struggle with the red teaming in different degrees and have up to 31% performance gap with GPT-4V. Additionally, we simply apply red teaming alignment to LLaVA-v1.5 with Supervised Fine-tuning (SFT) using RTVLM, and this bolsters the models' performance with 10% in RTVLM test set, 13% in MM-Hal, and without noticeable decline in MM-Bench, overpassing other LLaVA-based models with regular alignment data. This reveals that current open-sourced VLMs still lack red teaming alignment. Our code and datasets will be open-source.

{{</citation>}}


### (55/109) TroVE: Inducing Verifiable and Efficient Toolboxes for Solving Programmatic Tasks (Zhiruo Wang et al., 2024)

{{<citation>}}

Zhiruo Wang, Daniel Fried, Graham Neubig. (2024)  
**TroVE: Inducing Verifiable and Efficient Toolboxes for Solving Programmatic Tasks**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2401.12869v1)  

---


**ABSTRACT**  
Language models (LMs) can solve tasks such as answering questions about tables or images by writing programs. However, using primitive functions often leads to verbose and error-prone programs, and higher-level functions require expert design. To enable better solutions without human labor, we ask code LMs to curate reusable high-level functions, and use them to write solutions. We present TROVE, a training-free method of inducing a verifiable and efficient toolbox of functions, by generating via using, growing, and periodically trimming the toolbox. On 11 datasets from math, table question answering, and image reasoning tasks, TROVE consistently yields simpler solutions with higher accuracy than baselines using CODELLAMA and previous methods using GPT, while using 79-98% smaller toolboxes. TROVE further enables 31% faster and 13% more accurate human verification than baselines. With the same pipeline, it creates diverse functions for varied tasks and datasets, providing insights into their individual characteristics.

{{</citation>}}


### (56/109) How well can large language models explain business processes? (Dirk Fahland et al., 2024)

{{<citation>}}

Dirk Fahland, Fabian Fournier, Lior Limonad, Inna Skarbovsky, Ava J. E. Swevels. (2024)  
**How well can large language models explain business processes?**  

---
Primary Category: cs.AI  
Categories: 68T01, cs-AI, cs.AI  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2401.12846v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) are likely to play a prominent role in future AI-augmented business process management systems (ABPMSs) catering functionalities across all system lifecycle stages. One such system's functionality is Situation-Aware eXplainability (SAX), which relates to generating causally sound and yet human-interpretable explanations that take into account the process context in which the explained condition occurred. In this paper, we present the SAX4BPM framework developed to generate SAX explanations. The SAX4BPM suite consists of a set of services and a central knowledge repository. The functionality of these services is to elicit the various knowledge ingredients that underlie SAX explanations. A key innovative component among these ingredients is the causal process execution view. In this work, we integrate the framework with an LLM to leverage its power to synthesize the various input ingredients for the sake of improved SAX explanations. Since the use of LLMs for SAX is also accompanied by a certain degree of doubt related to its capacity to adequately fulfill SAX along with its tendency for hallucination and lack of inherent capacity to reason, we pursued a methodological evaluation of the quality of the generated explanations. To this aim, we developed a designated scale and conducted a rigorous user study. Our findings show that the input presented to the LLMs aided with the guard-railing of its performance, yielding SAX explanations having better-perceived fidelity. This improvement is moderated by the perception of trust and curiosity. More so, this improvement comes at the cost of the perceived interpretability of the explanation.

{{</citation>}}


### (57/109) A Review of Deep Learning Methods for Photoplethysmography Data (Guangkun Nie et al., 2024)

{{<citation>}}

Guangkun Nie, Jiabao Zhu, Gongzheng Tang, Deyun Zhang, Shijia Geng, Qinghao Zhao, Shenda Hong. (2024)  
**A Review of Deep Learning Methods for Photoplethysmography Data**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI, eess-SP  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2401.12783v1)  

---


**ABSTRACT**  
Photoplethysmography (PPG) is a highly promising device due to its advantages in portability, user-friendly operation, and non-invasive capabilities to measure a wide range of physiological information. Recent advancements in deep learning have demonstrated remarkable outcomes by leveraging PPG signals for tasks related to personal health management and other multifaceted applications. In this review, we systematically reviewed papers that applied deep learning models to process PPG data between January 1st of 2017 and July 31st of 2023 from Google Scholar, PubMed and Dimensions. Each paper is analyzed from three key perspectives: tasks, models, and data. We finally extracted 193 papers where different deep learning frameworks were used to process PPG signals. Based on the tasks addressed in these papers, we categorized them into two major groups: medical-related, and non-medical-related. The medical-related tasks were further divided into seven subgroups, including blood pressure analysis, cardiovascular monitoring and diagnosis, sleep health, mental health, respiratory monitoring and analysis, blood glucose analysis, as well as others. The non-medical-related tasks were divided into four subgroups, which encompass signal processing, biometric identification, electrocardiogram reconstruction, and human activity recognition. In conclusion, significant progress has been made in the field of using deep learning methods to process PPG data recently. This allows for a more thorough exploration and utilization of the information contained in PPG signals. However, challenges remain, such as limited quantity and quality of publicly available databases, a lack of effective validation in real-world scenarios, and concerns about the interpretability, scalability, and complexity of deep learning models. Moreover, there are still emerging research areas that require further investigation.

{{</citation>}}


### (58/109) EL-VIT: Probing Vision Transformer with Interactive Visualization (Hong Zhou et al., 2024)

{{<citation>}}

Hong Zhou, Rui Zhang, Peifeng Lai, Chaoran Guo, Yong Wang, Zhida Sun, Junjie Li. (2024)  
**EL-VIT: Probing Vision Transformer with Interactive Visualization**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.12666v1)  

---


**ABSTRACT**  
Nowadays, Vision Transformer (ViT) is widely utilized in various computer vision tasks, owing to its unique self-attention mechanism. However, the model architecture of ViT is complex and often challenging to comprehend, leading to a steep learning curve. ViT developers and users frequently encounter difficulties in interpreting its inner workings. Therefore, a visualization system is needed to assist ViT users in understanding its functionality. This paper introduces EL-VIT, an interactive visual analytics system designed to probe the Vision Transformer and facilitate a better understanding of its operations. The system consists of four layers of visualization views. The first three layers include model overview, knowledge background graph, and model detail view. These three layers elucidate the operation process of ViT from three perspectives: the overall model architecture, detailed explanation, and mathematical operations, enabling users to understand the underlying principles and the transition process between layers. The fourth interpretation view helps ViT users and experts gain a deeper understanding by calculating the cosine similarity between patches. Our two usage scenarios demonstrate the effectiveness and usability of EL-VIT in helping ViT users understand the working mechanism of ViT.

{{</citation>}}


### (59/109) Knowledge Distillation from Language-Oriented to Emergent Communication for Multi-Agent Remote Control (Yongjun Kim et al., 2024)

{{<citation>}}

Yongjun Kim, Sejin Seo, Jihong Park, Mehdi Bennis, Seong-Lyun Kim, Junil Choi. (2024)  
**Knowledge Distillation from Language-Oriented to Emergent Communication for Multi-Agent Remote Control**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-IT, cs-LG, cs-NI, cs.AI, math-IT  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2401.12624v1)  

---


**ABSTRACT**  
In this work, we compare emergent communication (EC) built upon multi-agent deep reinforcement learning (MADRL) and language-oriented semantic communication (LSC) empowered by a pre-trained large language model (LLM) using human language. In a multi-agent remote navigation task, with multimodal input data comprising location and channel maps, it is shown that EC incurs high training cost and struggles when using multimodal data, whereas LSC yields high inference computing cost due to the LLM's large size. To address their respective bottlenecks, we propose a novel framework of language-guided EC (LEC) by guiding the EC training using LSC via knowledge distillation (KD). Simulations corroborate that LEC achieves faster travel time while avoiding areas with poor channel conditions, as well as speeding up the MADRL training convergence by up to 61.8% compared to EC.

{{</citation>}}


### (60/109) Revolutionizing Retrieval-Augmented Generation with Enhanced PDF Structure Recognition (Demiao Lin, 2024)

{{<citation>}}

Demiao Lin. (2024)  
**Revolutionizing Retrieval-Augmented Generation with Enhanced PDF Structure Recognition**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Embedding, Language Model, QA  
[Paper Link](http://arxiv.org/abs/2401.12599v1)  

---


**ABSTRACT**  
With the rapid development of Large Language Models (LLMs), Retrieval-Augmented Generation (RAG) has become a predominant method in the field of professional knowledge-based question answering. Presently, major foundation model companies have opened up Embedding and Chat API interfaces, and frameworks like LangChain have already integrated the RAG process. It appears that the key models and steps in RAG have been resolved, leading to the question: are professional knowledge QA systems now approaching perfection? This article discovers that current primary methods depend on the premise of accessing high-quality text corpora. However, since professional documents are mainly stored in PDFs, the low accuracy of PDF parsing significantly impacts the effectiveness of professional knowledge-based QA. We conducted an empirical RAG experiment across hundreds of questions from the corresponding real-world professional documents. The results show that, ChatDOC, a RAG system equipped with a panoptic and pinpoint PDF parser, retrieves more accurate and complete segments, and thus better answers. Empirical experiments show that ChatDOC is superior to baseline on nearly 47% of questions, ties for 38% of cases, and falls short on only 15% of cases. It shows that we may revolutionize RAG with enhanced PDF structure recognition.

{{</citation>}}


### (61/109) Balancing the AI Strength of Roles in Self-Play Training with Regret Matching+ (Xiaoxi Wang, 2024)

{{<citation>}}

Xiaoxi Wang. (2024)  
**Balancing the AI Strength of Roles in Self-Play Training with Regret Matching+**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.12557v1)  

---


**ABSTRACT**  
When training artificial intelligence for games encompassing multiple roles, the development of a generalized model capable of controlling any character within the game presents a viable option. This strategy not only conserves computational resources and time during the training phase but also reduces resource requirements during deployment. training such a generalized model often encounters challenges related to uneven capabilities when controlling different roles. A simple method is introduced based on Regret Matching+, which facilitates a more balanced performance of strength by the model when controlling various roles.

{{</citation>}}


### (62/109) Building Minimal and Reusable Causal State Abstractions for Reinforcement Learning (Zizhao Wang et al., 2024)

{{<citation>}}

Zizhao Wang, Caroline Wang, Xuesu Xiao, Yuke Zhu, Peter Stone. (2024)  
**Building Minimal and Reusable Causal State Abstractions for Reinforcement Learning**  

---
Primary Category: cs.AI  
Categories: I-2-9; I-2-8; I-2-6, cs-AI, cs-LG, cs-RO, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.12497v1)  

---


**ABSTRACT**  
Two desiderata of reinforcement learning (RL) algorithms are the ability to learn from relatively little experience and the ability to learn policies that generalize to a range of problem specifications. In factored state spaces, one approach towards achieving both goals is to learn state abstractions, which only keep the necessary variables for learning the tasks at hand. This paper introduces Causal Bisimulation Modeling (CBM), a method that learns the causal relationships in the dynamics and reward functions for each task to derive a minimal, task-specific abstraction. CBM leverages and improves implicit modeling to train a high-fidelity causal dynamics model that can be reused for all tasks in the same environment. Empirical validation on manipulation environments and Deepmind Control Suite reveals that CBM's learned implicit dynamics models identify the underlying causal relationships and state abstractions more accurately than explicit ones. Furthermore, the derived state abstractions allow a task learner to achieve near-oracle levels of sample efficiency and outperform baselines on all tasks.

{{</citation>}}


### (63/109) Towards Socially and Morally Aware RL agent: Reward Design With LLM (Zhaoyue Wang, 2024)

{{<citation>}}

Zhaoyue Wang. (2024)  
**Towards Socially and Morally Aware RL agent: Reward Design With LLM**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Language Model, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.12459v1)  

---


**ABSTRACT**  
When we design and deploy an Reinforcement Learning (RL) agent, reward functions motivates agents to achieve an objective. An incorrect or incomplete specification of the objective can result in behavior that does not align with human values - failing to adhere with social and moral norms that are ambiguous and context dependent, and cause undesired outcomes such as negative side effects and exploration that is unsafe. Previous work have manually defined reward functions to avoid negative side effects, use human oversight for safe exploration, or use foundation models as planning tools. This work studies the ability of leveraging Large Language Models (LLM)' understanding of morality and social norms on safe exploration augmented RL methods. This work evaluates language model's result against human feedbacks and demonstrates language model's capability as direct reward signals.

{{</citation>}}


## cs.CV (21)



### (64/109) Digital Divides in Scene Recognition: Uncovering Socioeconomic Biases in Deep Learning Systems (Michelle R. Greene et al., 2024)

{{<citation>}}

Michelle R. Greene, Mariam Josyula, Wentao Si, Jennifer A. Hart. (2024)  
**Digital Divides in Scene Recognition: Uncovering Socioeconomic Biases in Deep Learning Systems**  

---
Primary Category: cs.CV  
Categories: 68-02, I-2-m, cs-AI, cs-CV, cs.CV  
Keywords: AI, Bias  
[Paper Link](http://arxiv.org/abs/2401.13097v1)  

---


**ABSTRACT**  
Computer-based scene understanding has influenced fields ranging from urban planning to autonomous vehicle performance, yet little is known about how well these technologies work across social differences. We investigate the biases of deep convolutional neural networks (dCNNs) in scene classification, using nearly one million images from global and US sources, including user-submitted home photographs and Airbnb listings. We applied statistical models to quantify the impact of socioeconomic indicators such as family income, Human Development Index (HDI), and demographic factors from public data sources (CIA and US Census) on dCNN performance. Our analyses revealed significant socioeconomic bias, where pretrained dCNNs demonstrated lower classification accuracy, lower classification confidence, and a higher tendency to assign labels that could be offensive when applied to homes (e.g., "ruin", "slum"), especially in images from homes with lower socioeconomic status (SES). This trend is consistent across two datasets of international images and within the diverse economic and racial landscapes of the United States. This research contributes to understanding biases in computer vision, emphasizing the need for more inclusive and representative training datasets. By mitigating the bias in the computer vision pipelines, we can ensure fairer and more equitable outcomes for applied computer vision, including home valuation and smart home security systems. There is urgency in addressing these biases, which can significantly impact critical decisions in urban development and resource allocation. Our findings also motivate the development of AI systems that better understand and serve diverse communities, moving towards technology that equitably benefits all sectors of society.

{{</citation>}}


### (65/109) Open-source data pipeline for street-view images: a case study on community mobility during COVID-19 pandemic (Matthew Martell et al., 2024)

{{<citation>}}

Matthew Martell, Nick Terry, Ribhu Sengupta, Chris Salazar, Nicole A. Errett, Scott B. Miles, Joseph Wartman, Youngjun Choe. (2024)  
**Open-source data pipeline for street-view images: a case study on community mobility during COVID-19 pandemic**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, stat-AP  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2401.13087v1)  

---


**ABSTRACT**  
Street View Images (SVI) are a common source of valuable data for researchers. Researchers have used SVI data for estimating pedestrian volumes, demographic surveillance, and to better understand built and natural environments in cityscapes. However, the most common source of publicly available SVI data is Google Street View. Google Street View images are collected infrequently, making temporal analysis challenging, especially in low population density areas. Our main contribution is the development of an open-source data pipeline for processing 360-degree video recorded from a car-mounted camera. The video data is used to generate SVIs, which then can be used as an input for temporal analysis. We demonstrate the use of the pipeline by collecting a SVI dataset over a 38-month longitudinal survey of Seattle, WA, USA during the COVID-19 pandemic. The output of our pipeline is validated through statistical analyses of pedestrian traffic in the images. We confirm known results in the literature and provide new insights into outdoor pedestrian traffic patterns. This study demonstrates the feasibility and value of collecting and using SVI for research purposes beyond what is possible with currently available SVI data. Limitations and future improvements on the data pipeline and case study are also discussed.

{{</citation>}}


### (66/109) PlaceFormer: Transformer-based Visual Place Recognition using Multi-Scale Patch Selection and Fusion (Shyam Sundar Kannan et al., 2024)

{{<citation>}}

Shyam Sundar Kannan, Byung-Cheol Min. (2024)  
**PlaceFormer: Transformer-based Visual Place Recognition using Multi-Scale Patch Selection and Fusion**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.13082v1)  

---


**ABSTRACT**  
Visual place recognition is a challenging task in the field of computer vision, and autonomous robotics and vehicles, which aims to identify a location or a place from visual inputs. Contemporary methods in visual place recognition employ convolutional neural networks and utilize every region within the image for the place recognition task. However, the presence of dynamic and distracting elements in the image may impact the effectiveness of the place recognition process. Therefore, it is meaningful to focus on task-relevant regions of the image for improved recognition. In this paper, we present PlaceFormer, a novel transformer-based approach for visual place recognition. PlaceFormer employs patch tokens from the transformer to create global image descriptors, which are then used for image retrieval. To re-rank the retrieved images, PlaceFormer merges the patch tokens from the transformer to form multi-scale patches. Utilizing the transformer's self-attention mechanism, it selects patches that correspond to task-relevant areas in an image. These selected patches undergo geometric verification, generating similarity scores across different patch sizes. Subsequently, spatial scores from each patch size are fused to produce a final similarity score. This score is then used to re-rank the images initially retrieved using global image descriptors. Extensive experiments on benchmark datasets demonstrate that PlaceFormer outperforms several state-of-the-art methods in terms of accuracy and computational efficiency, requiring less time and memory.

{{</citation>}}


### (67/109) Free Form Medical Visual Question Answering in Radiology (Abhishek Narayanan et al., 2024)

{{<citation>}}

Abhishek Narayanan, Rushabh Musthyala, Rahul Sankar, Anirudh Prasad Nistala, Pranav Singh, Jacopo Cirrone. (2024)  
**Free Form Medical Visual Question Answering in Radiology**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Computer Vision, Natural Language Processing, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2401.13081v1)  

---


**ABSTRACT**  
Visual Question Answering (VQA) in the medical domain presents a unique, interdisciplinary challenge, combining fields such as Computer Vision, Natural Language Processing, and Knowledge Representation. Despite its importance, research in medical VQA has been scant, only gaining momentum since 2018. Addressing this gap, our research delves into the effective representation of radiology images and the joint learning of multimodal representations, surpassing existing methods. We innovatively augment the SLAKE dataset, enabling our model to respond to a more diverse array of questions, not limited to the immediate content of radiology or pathology images. Our model achieves a top-1 accuracy of 79.55\% with a less complex architecture, demonstrating comparable performance to current state-of-the-art models. This research not only advances medical VQA but also opens avenues for practical applications in diagnostic settings.

{{</citation>}}


### (68/109) Zero-Shot Learning for the Primitives of 3D Affordance in General Objects (Hyeonwoo Kim et al., 2024)

{{<citation>}}

Hyeonwoo Kim, Sookwan Han, Patrick Kwon, Hanbyul Joo. (2024)  
**Zero-Shot Learning for the Primitives of 3D Affordance in General Objects**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2401.12978v2)  

---


**ABSTRACT**  
One of the major challenges in AI is teaching machines to precisely respond and utilize environmental functionalities, thereby achieving the affordance awareness that humans possess. Despite its importance, the field has been lagging in terms of learning, especially in 3D, as annotating affordance accompanies a laborious process due to the numerous variations of human-object interaction. The low availability of affordance data limits the learning in terms of generalization for object categories, and also simplifies the representation of affordance, capturing only a fraction of the affordance. To overcome these challenges, we propose a novel, self-supervised method to generate the 3D affordance examples given only a 3D object, without any manual annotations. The method starts by capturing the 3D object into images and creating 2D affordance images by inserting humans into the image via inpainting diffusion models, where we present the Adaptive Mask algorithm to enable human insertion without altering the original details of the object. The method consequently lifts inserted humans back to 3D to create 3D human-object pairs, where the depth ambiguity is resolved within a depth optimization framework that utilizes pre-generated human postures from multiple viewpoints. We also provide a novel affordance representation defined on relative orientations and proximity between dense human and object points, that can be easily aggregated from any 3D HOI datasets. The proposed representation serves as a primitive that can be manifested to conventional affordance representations via simple transformations, ranging from physically exerted affordances to nonphysical ones. We demonstrate the efficacy of our method and representation by generating the 3D affordance samples and deriving high-quality affordance examples from the representation, including contact, orientation, and spatial occupancies.

{{</citation>}}


### (69/109) On the Efficacy of Text-Based Input Modalities for Action Anticipation (Apoorva Beedu et al., 2024)

{{<citation>}}

Apoorva Beedu, Karan Samel, Irfan Essa. (2024)  
**On the Efficacy of Text-Based Input Modalities for Action Anticipation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV, eess-IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.12972v1)  

---


**ABSTRACT**  
Although the task of anticipating future actions is highly uncertain, information from additional modalities help to narrow down plausible action choices. Each modality provides different environmental context for the model to learn from. While previous multi-modal methods leverage information from modalities such as video and audio, we primarily explore how text inputs for actions and objects can also enable more accurate action anticipation. Therefore, we propose a Multi-modal Anticipative Transformer (MAT), an attention-based video transformer architecture that jointly learns from multi-modal features and text captions. We train our model in two-stages, where the model first learns to predict actions in the video clip by aligning with captions, and during the second stage, we fine-tune the model to predict future actions. Compared to existing methods, MAT has the advantage of learning additional environmental context from two kinds of text inputs: action descriptions during the pre-training stage, and the text inputs for detected objects and actions during modality feature fusion. Through extensive experiments, we evaluate the effectiveness of the pre-training stage, and show that our model outperforms previous methods on all datasets. In addition, we examine the impact of object and action information obtained via text and perform extensive ablations. We evaluate the performance on on three datasets: EpicKitchens-100, EpicKitchens-55 and EGTEA GAZE+; and show that text descriptions do indeed aid in more effective action anticipation.

{{</citation>}}


### (70/109) SGTR+: End-to-end Scene Graph Generation with Transformer (Rongjie Li et al., 2024)

{{<citation>}}

Rongjie Li, Songyang Zhang, Xuming He. (2024)  
**SGTR+: End-to-end Scene Graph Generation with Transformer**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.12835v1)  

---


**ABSTRACT**  
Scene Graph Generation (SGG) remains a challenging visual understanding task due to its compositional property. Most previous works adopt a bottom-up, two-stage or point-based, one-stage approach, which often suffers from high time complexity or suboptimal designs. In this work, we propose a novel SGG method to address the aforementioned issues, formulating the task as a bipartite graph construction problem. To address the issues above, we create a transformer-based end-to-end framework to generate the entity and entity-aware predicate proposal set, and infer directed edges to form relation triplets. Moreover, we design a graph assembling module to infer the connectivity of the bipartite scene graph based on our entity-aware structure, enabling us to generate the scene graph in an end-to-end manner. Based on bipartite graph assembling paradigm, we further propose a new technical design to address the efficacy of entity-aware modeling and optimization stability of graph assembling. Equipped with the enhanced entity-aware design, our method achieves optimal performance and time-complexity. Extensive experimental results show that our design is able to achieve the state-of-the-art or comparable performance on three challenging benchmarks, surpassing most of the existing approaches and enjoying higher efficiency in inference. Code is available: https://github.com/Scarecrow0/SGTR

{{</citation>}}


### (71/109) DatUS^2: Data-driven Unsupervised Semantic Segmentation with Pre-trained Self-supervised Vision Transformer (Sonal Kumar et al., 2024)

{{<citation>}}

Sonal Kumar, Arijit Sur, Rashmi Dutta Baruah. (2024)  
**DatUS^2: Data-driven Unsupervised Semantic Segmentation with Pre-trained Self-supervised Vision Transformer**  

---
Primary Category: cs.CV  
Categories: I-4; I-5, cs-CV, cs-LG, cs.CV  
Keywords: Semantic Segmentation, Transformer  
[Paper Link](http://arxiv.org/abs/2401.12820v1)  

---


**ABSTRACT**  
Successive proposals of several self-supervised training schemes continue to emerge, taking one step closer to developing a universal foundation model. In this process, the unsupervised downstream tasks are recognized as one of the evaluation methods to validate the quality of visual features learned with a self-supervised training scheme. However, unsupervised dense semantic segmentation has not been explored as a downstream task, which can utilize and evaluate the quality of semantic information introduced in patch-level feature representations during self-supervised training of a vision transformer. Therefore, this paper proposes a novel data-driven approach for unsupervised semantic segmentation (DatUS^2) as a downstream task. DatUS^2 generates semantically consistent and dense pseudo annotate segmentation masks for the unlabeled image dataset without using any visual-prior or synchronized data. We compare these pseudo-annotated segmentation masks with ground truth masks for evaluating recent self-supervised training schemes to learn shared semantic properties at the patch level and discriminative semantic properties at the segment level. Finally, we evaluate existing state-of-the-art self-supervised training schemes with our proposed downstream task, i.e., DatUS^2. Also, the best version of DatUS^2 outperforms the existing state-of-the-art method for the unsupervised dense semantic segmentation task with 15.02% MiOU and 21.47% Pixel accuracy on the SUIM dataset. It also achieves a competitive level of accuracy for a large-scale and complex dataset, i.e., the COCO dataset.

{{</citation>}}


### (72/109) Correlation-Embedded Transformer Tracking: A Single-Branch Framework (Fei Xie et al., 2024)

{{<citation>}}

Fei Xie, Wankou Yang, Chunyu Wang, Lei Chu, Yue Cao, Chao Ma, Wenjun Zeng. (2024)  
**Correlation-Embedded Transformer Tracking: A Single-Branch Framework**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.12743v1)  

---


**ABSTRACT**  
Developing robust and discriminative appearance models has been a long-standing research challenge in visual object tracking. In the prevalent Siamese-based paradigm, the features extracted by the Siamese-like networks are often insufficient to model the tracked targets and distractor objects, thereby hindering them from being robust and discriminative simultaneously. While most Siamese trackers focus on designing robust correlation operations, we propose a novel single-branch tracking framework inspired by the transformer. Unlike the Siamese-like feature extraction, our tracker deeply embeds cross-image feature correlation in multiple layers of the feature network. By extensively matching the features of the two images through multiple layers, it can suppress non-target features, resulting in target-aware feature extraction. The output features can be directly used for predicting target locations without additional correlation steps. Thus, we reformulate the two-branch Siamese tracking as a conceptually simple, fully transformer-based Single-Branch Tracking pipeline, dubbed SBT. After conducting an in-depth analysis of the SBT baseline, we summarize many effective design principles and propose an improved tracker dubbed SuperSBT. SuperSBT adopts a hierarchical architecture with a local modeling layer to enhance shallow-level features. A unified relation modeling is proposed to remove complex handcrafted layer pattern designs. SuperSBT is further improved by masked image modeling pre-training, integrating temporal modeling, and equipping with dedicated prediction heads. Thus, SuperSBT outperforms the SBT baseline by 4.7%,3.0%, and 4.5% AUC scores in LaSOT, TrackingNet, and GOT-10K. Notably, SuperSBT greatly raises the speed of SBT from 37 FPS to 81 FPS. Extensive experiments show that our method achieves superior results on eight VOT benchmarks.

{{</citation>}}


### (73/109) Shift-ConvNets: Small Convolutional Kernel with Large Kernel Effects (Dachong Li et al., 2024)

{{<citation>}}

Dachong Li, Li Li, Zhuangzhuang Chen, Jianqiang Li. (2024)  
**Shift-ConvNets: Small Convolutional Kernel with Large Kernel Effects**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2401.12736v1)  

---


**ABSTRACT**  
Recent studies reveal that the remarkable performance of Vision transformers (ViTs) benefits from large receptive fields. For this reason, the large convolutional kernel design becomes an ideal solution to make Convolutional Neural Networks (CNNs) great again. However, the typical large convolutional kernels turn out to be hardware-unfriendly operators, resulting in discount compatibility of various hardware platforms. Thus, it is unwise to simply enlarge the convolutional kernel size. In this paper, we reveal that small convolutional kernels and convolution operations can achieve the closing effects of large kernel sizes. Then, we propose a shift-wise operator that ensures the CNNs capture long-range dependencies with the help of the sparse mechanism, while remaining hardware-friendly. Experimental results show that our shift-wise operator significantly improves the accuracy of a regular CNN while markedly reducing computational requirements. On the ImageNet-1k, our shift-wise enhanced CNN model outperforms the state-of-the-art models. Code & models at https://github.com/lidc54/shift-wiseConv.

{{</citation>}}


### (74/109) Enhancing Object Detection Performance for Small Objects through Synthetic Data Generation and Proportional Class-Balancing Technique: A Comparative Study in Industrial Scenarios (Jibinraj Antony et al., 2024)

{{<citation>}}

Jibinraj Antony, Vinit Hegiste, Ali Nazeri, Hooman Tavakoli, Snehal Walunj, Christiane Plociennik, Martin Ruskowski. (2024)  
**Enhancing Object Detection Performance for Small Objects through Synthetic Data Generation and Proportional Class-Balancing Technique: A Comparative Study in Industrial Scenarios**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2401.12729v2)  

---


**ABSTRACT**  
Object Detection (OD) has proven to be a significant computer vision method in extracting localized class information and has multiple applications in the industry. Although many of the state-of-the-art (SOTA) OD models perform well on medium and large sized objects, they seem to under perform on small objects. In most of the industrial use cases, it is difficult to collect and annotate data for small objects, as it is time-consuming and prone to human errors. Additionally, those datasets are likely to be unbalanced and often result in an inefficient model convergence. To tackle this challenge, this study presents a novel approach that injects additional data points to improve the performance of the OD models. Using synthetic data generation, the difficulties in data collection and annotations for small object data points can be minimized and to create a dataset with balanced distribution. This paper discusses the effects of a simple proportional class-balancing technique, to enable better anchor matching of the OD models. A comparison was carried out on the performances of the SOTA OD models: YOLOv5, YOLOv7 and SSD, for combinations of real and synthetic datasets within an industrial use case.

{{</citation>}}


### (75/109) CCA: Collaborative Competitive Agents for Image Editing (Tiankai Hang et al., 2024)

{{<citation>}}

Tiankai Hang, Shuyang Gu, Dong Chen, Xin Geng, Baining Guo. (2024)  
**CCA: Collaborative Competitive Agents for Image Editing**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.13011v1)  

---


**ABSTRACT**  
This paper presents a novel generative model, Collaborative Competitive Agents (CCA), which leverages the capabilities of multiple Large Language Models (LLMs) based agents to execute complex tasks. Drawing inspiration from Generative Adversarial Networks (GANs), the CCA system employs two equal-status generator agents and a discriminator agent. The generators independently process user instructions and generate results, while the discriminator evaluates the outputs, and provides feedback for the generator agents to further reflect and improve the generation results. Unlike the previous generative model, our system can obtain the intermediate steps of generation. This allows each generator agent to learn from other successful executions due to its transparency, enabling a collaborative competition that enhances the quality and robustness of the system's results. The primary focus of this study is image editing, demonstrating the CCA's ability to handle intricate instructions robustly. The paper's main contributions include the introduction of a multi-agent-based generative model with controllable intermediate steps and iterative optimization, a detailed examination of agent relationships, and comprehensive experiments on image editing. Code is available at \href{https://github.com/TiankaiHang/CCA}{https://github.com/TiankaiHang/CCA}.

{{</citation>}}


### (76/109) ClipSAM: CLIP and SAM Collaboration for Zero-Shot Anomaly Segmentation (Shengze Li et al., 2024)

{{<citation>}}

Shengze Li, Jianjian Cao, Peng Ye, Yuhan Ding, Chongjun Tu, Tao Chen. (2024)  
**ClipSAM: CLIP and SAM Collaboration for Zero-Shot Anomaly Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2401.12665v2)  

---


**ABSTRACT**  
Recently, foundational models such as CLIP and SAM have shown promising performance for the task of Zero-Shot Anomaly Segmentation (ZSAS). However, either CLIP-based or SAM-based ZSAS methods still suffer from non-negligible key drawbacks: 1) CLIP primarily focuses on global feature alignment across different inputs, leading to imprecise segmentation of local anomalous parts; 2) SAM tends to generate numerous redundant masks without proper prompt constraints, resulting in complex post-processing requirements. In this work, we innovatively propose a CLIP and SAM collaboration framework called ClipSAM for ZSAS. The insight behind ClipSAM is to employ CLIP's semantic understanding capability for anomaly localization and rough segmentation, which is further used as the prompt constraints for SAM to refine the anomaly segmentation results. In details, we introduce a crucial Unified Multi-scale Cross-modal Interaction (UMCI) module for interacting language with visual features at multiple scales of CLIP to reason anomaly positions. Then, we design a novel Multi-level Mask Refinement (MMR) module, which utilizes the positional information as multi-level prompts for SAM to acquire hierarchical levels of masks and merges them. Extensive experiments validate the effectiveness of our approach, achieving the optimal segmentation performance on the MVTec-AD and VisA datasets.

{{</citation>}}


### (77/109) NeRF-AD: Neural Radiance Field with Attention-based Disentanglement for Talking Face Synthesis (Chongke Bi et al., 2024)

{{<citation>}}

Chongke Bi, Xiaoxing Liu, Zhilei Liu. (2024)  
**NeRF-AD: Neural Radiance Field with Attention-based Disentanglement for Talking Face Synthesis**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2401.12568v1)  

---


**ABSTRACT**  
Talking face synthesis driven by audio is one of the current research hotspots in the fields of multidimensional signal processing and multimedia. Neural Radiance Field (NeRF) has recently been brought to this research field in order to enhance the realism and 3D effect of the generated faces. However, most existing NeRF-based methods either burden NeRF with complex learning tasks while lacking methods for supervised multimodal feature fusion, or cannot precisely map audio to the facial region related to speech movements. These reasons ultimately result in existing methods generating inaccurate lip shapes. This paper moves a portion of NeRF learning tasks ahead and proposes a talking face synthesis method via NeRF with attention-based disentanglement (NeRF-AD). In particular, an Attention-based Disentanglement module is introduced to disentangle the face into Audio-face and Identity-face using speech-related facial action unit (AU) information. To precisely regulate how audio affects the talking face, we only fuse the Audio-face with audio feature. In addition, AU information is also utilized to supervise the fusion of these two modalities. Extensive qualitative and quantitative experiments demonstrate that our NeRF-AD outperforms state-of-the-art methods in generating realistic talking face videos, including image quality and lip synchronization. To view video results, please refer to https://xiaoxingliu02.github.io/NeRF-AD.

{{</citation>}}


### (78/109) Self-Supervised Vision Transformers Are Efficient Segmentation Learners for Imperfect Labels (Seungho Lee et al., 2024)

{{<citation>}}

Seungho Lee, Seoungyoon Kang, Hyunjung Shim. (2024)  
**Self-Supervised Vision Transformers Are Efficient Segmentation Learners for Imperfect Labels**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.12535v1)  

---


**ABSTRACT**  
This study demonstrates a cost-effective approach to semantic segmentation using self-supervised vision transformers (SSVT). By freezing the SSVT backbone and training a lightweight segmentation head, our approach effectively utilizes imperfect labels, thereby improving robustness to label imperfections. Empirical experiments show significant performance improvements over existing methods for various annotation types, including scribble, point-level, and image-level labels. The research highlights the effectiveness of self-supervised vision transformers in dealing with imperfect labels, providing a practical and efficient solution for semantic segmentation while reducing annotation costs. Through extensive experiments, we confirm that our method outperforms baseline models for all types of imperfect labels. Especially under the zero-shot vision-language-model-based label, our model exhibits 11.5\%p performance gain compared to the baseline.

{{</citation>}}


### (79/109) Convolutional Initialization for Data-Efficient Vision Transformers (Jianqiao Zheng et al., 2024)

{{<citation>}}

Jianqiao Zheng, Xueqian Li, Simon Lucey. (2024)  
**Convolutional Initialization for Data-Efficient Vision Transformers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.12511v1)  

---


**ABSTRACT**  
Training vision transformer networks on small datasets poses challenges. In contrast, convolutional neural networks (CNNs) can achieve state-of-the-art performance by leveraging their architectural inductive bias. In this paper, we investigate whether this inductive bias can be reinterpreted as an initialization bias within a vision transformer network. Our approach is motivated by the finding that random impulse filters can achieve almost comparable performance to learned filters in CNNs. We introduce a novel initialization strategy for transformer networks that can achieve comparable performance to CNNs on small datasets while preserving its architectural flexibility.

{{</citation>}}


### (80/109) Small Language Model Meets with Reinforced Vision Vocabulary (Haoran Wei et al., 2024)

{{<citation>}}

Haoran Wei, Lingyu Kong, Jinyue Chen, Liang Zhao, Zheng Ge, En Yu, Jianjian Sun, Chunrui Han, Xiangyu Zhang. (2024)  
**Small Language Model Meets with Reinforced Vision Vocabulary**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Language Model, QA  
[Paper Link](http://arxiv.org/abs/2401.12503v1)  

---


**ABSTRACT**  
Playing Large Vision Language Models (LVLMs) in 2023 is trendy among the AI community. However, the relatively large number of parameters (more than 7B) of popular LVLMs makes it difficult to train and deploy on consumer GPUs, discouraging many researchers with limited resources. Imagine how cool it would be to experience all the features of current LVLMs on an old GTX1080ti (our only game card). Accordingly, we present Vary-toy in this report, a small-size Vary along with Qwen-1.8B as the base ``large'' language model. In Vary-toy, we introduce an improved vision vocabulary, allowing the model to not only possess all features of Vary but also gather more generality. Specifically, we replace negative samples of natural images with positive sample data driven by object detection in the procedure of generating vision vocabulary, more sufficiently utilizing the capacity of the vocabulary network and enabling it to efficiently encode visual information corresponding to natural objects. For experiments, Vary-toy can achieve 65.6% ANLS on DocVQA, 59.1% accuracy on ChartQA, 88.1% accuracy on RefCOCO, and 29% on MMVet. The code will be publicly available on the homepage.

{{</citation>}}


### (81/109) Exploration and Improvement of Nerf-based 3D Scene Editing Techniques (Shun Fang et al., 2024)

{{<citation>}}

Shun Fang, Ming Cui, Xing Feng, Yanan Zhang. (2024)  
**Exploration and Improvement of Nerf-based 3D Scene Editing Techniques**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-GR, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.12456v1)  

---


**ABSTRACT**  
NeRF's high-quality scene synthesis capability was quickly accepted by scholars in the years after it was proposed, and significant progress has been made in 3D scene representation and synthesis. However, the high computational cost limits intuitive and efficient editing of scenes, making NeRF's development in the scene editing field facing many challenges. This paper reviews the preliminary explorations of scholars on NeRF in the scene or object editing field in recent years, mainly changing the shape and texture of scenes or objects in new synthesized scenes; through the combination of residual models such as GaN and Transformer with NeRF, the generalization ability of NeRF scene editing has been further expanded, including realizing real-time new perspective editing feedback, multimodal editing of text synthesized 3D scenes, 4D synthesis performance, and in-depth exploration in light and shadow editing, initially achieving optimization of indirect touch editing and detail representation in complex scenes. Currently, most NeRF editing methods focus on the touch points and materials of indirect points, but when dealing with more complex or larger 3D scenes, it is difficult to balance accuracy, breadth, efficiency, and quality. Overcoming these challenges may become the direction of future NeRF 3D scene editing technology.

{{</citation>}}


### (82/109) MAST: Video Polyp Segmentation with a Mixture-Attention Siamese Transformer (Geng Chen et al., 2024)

{{<citation>}}

Geng Chen, Junqing Yang, Xiaozhou Pu, Ge-Peng Ji, Huan Xiong, Yongsheng Pan, Hengfei Cui, Yong Xia. (2024)  
**MAST: Video Polyp Segmentation with a Mixture-Attention Siamese Transformer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2401.12439v1)  

---


**ABSTRACT**  
Accurate segmentation of polyps from colonoscopy videos is of great significance to polyp treatment and early prevention of colorectal cancer. However, it is challenging due to the difficulties associated with modelling long-range spatio-temporal relationships within a colonoscopy video. In this paper, we address this challenging task with a novel Mixture-Attention Siamese Transformer (MAST), which explicitly models the long-range spatio-temporal relationships with a mixture-attention mechanism for accurate polyp segmentation. Specifically, we first construct a Siamese transformer architecture to jointly encode paired video frames for their feature representations. We then design a mixture-attention module to exploit the intra-frame and inter-frame correlations, enhancing the features with rich spatio-temporal relationships. Finally, the enhanced features are fed to two parallel decoders for predicting the segmentation maps. To the best of our knowledge, our MAST is the first transformer model dedicated to video polyp segmentation. Extensive experiments on the large-scale SUN-SEG benchmark demonstrate the superior performance of MAST in comparison with the cutting-edge competitors. Our code is publicly available at https://github.com/Junqing-Yang/MAST.

{{</citation>}}


### (83/109) The Neglected Tails of Vision-Language Models (Shubham Parashar et al., 2024)

{{<citation>}}

Shubham Parashar, Zhiqiu Lin, Tian Liu, Xiangjue Dong, Yanan Li, Deva Ramanan, James Caverlee, Shu Kong. (2024)  
**The Neglected Tails of Vision-Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs-LG, cs.CV  
Keywords: AI, ImageNet, Language Model  
[Paper Link](http://arxiv.org/abs/2401.12425v1)  

---


**ABSTRACT**  
Vision-language models (VLMs) excel in zero-shot recognition but exhibit drastically imbalanced performance across visual concepts. For example, CLIP, despite an impressive mean zero-shot accuracy on ImageNet (72.7%), yields $<$10% on ten concepts (e.g., gyromitra and night snake), presumably, because these concepts are under-represented in VLMs' imbalanced pretraining data. Yet, assessing this imbalance is challenging as it is non-trivial to calculate the frequency of specific concepts within VLMs' large-scale pretraining data. Our work makes the first attempt to measure the concept frequency by analyzing pretraining texts. We use off-the-shelf language models to help count relevant texts that contain synonyms of the given concepts and resolve linguistic ambiguity. We confirm that popular VLM datasets like LAION indeed exhibit long-tailed concept distributions, which strongly correlate with per-class accuracies. Further, contemporary multimodal systems, e.g., visual chatbots and text-to-image generators, also struggle with the rare concepts identified by our method. To mitigate VLMs' imbalanced performance in zero-shot recognition, we propose REtrieval-Augmented Learning REAL. First, instead of prompting VLMs using the original class names, REAL uses their most frequent synonyms found in VLMs' pretraining texts. This already outperforms human-engineered and LLM-generated prompts over nine benchmark datasets, likely because VLMs have seen more images associated with the frequently used synonyms. Second, REAL uses all the concept synonyms to retrieve a small, class-balanced set of pretraining data to train a robust classifier. REAL surpasses the recent retrieval-augmented solution REACT, using 400x less storage and 10,000x less training time!

{{</citation>}}


### (84/109) AdaEmbed: Semi-supervised Domain Adaptation in the Embedding Space (Ali Mottaghi et al., 2024)

{{<citation>}}

Ali Mottaghi, Mohammad Abdullah Jamal, Serena Yeung, Omid Mohareri. (2024)  
**AdaEmbed: Semi-supervised Domain Adaptation in the Embedding Space**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2401.12421v1)  

---


**ABSTRACT**  
Semi-supervised domain adaptation (SSDA) presents a critical hurdle in computer vision, especially given the frequent scarcity of labeled data in real-world settings. This scarcity often causes foundation models, trained on extensive datasets, to underperform when applied to new domains. AdaEmbed, our newly proposed methodology for SSDA, offers a promising solution to these challenges. Leveraging the potential of unlabeled data, AdaEmbed facilitates the transfer of knowledge from a labeled source domain to an unlabeled target domain by learning a shared embedding space. By generating accurate and uniform pseudo-labels based on the established embedding space, the model overcomes the limitations of conventional SSDA, thus enhancing performance significantly. Our method's effectiveness is validated through extensive experiments on benchmark datasets such as DomainNet, Office-Home, and VisDA-C, where AdaEmbed consistently outperforms all the baselines, setting a new state of the art for SSDA. With its straightforward implementation and high data efficiency, AdaEmbed stands out as a robust and pragmatic solution for real-world scenarios, where labeled data is scarce. To foster further research and application in this area, we are sharing the codebase of our unified framework for semi-supervised domain adaptation.

{{</citation>}}


## cs.RO (2)



### (85/109) SemanticSLAM: Learning based Semantic Map Construction and Robust Camera Localization (Mingyang Li et al., 2024)

{{<citation>}}

Mingyang Li, Yue Ma, Qinru Qiu. (2024)  
**SemanticSLAM: Learning based Semantic Map Construction and Robust Camera Localization**  

---
Primary Category: cs.RO  
Categories: cs-CV, cs-RO, cs.RO  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2401.13076v1)  

---


**ABSTRACT**  
Current techniques in Visual Simultaneous Localization and Mapping (VSLAM) estimate camera displacement by comparing image features of consecutive scenes. These algorithms depend on scene continuity, hence requires frequent camera inputs. However, processing images frequently can lead to significant memory usage and computation overhead. In this study, we introduce SemanticSLAM, an end-to-end visual-inertial odometry system that utilizes semantic features extracted from an RGB-D sensor. This approach enables the creation of a semantic map of the environment and ensures reliable camera localization. SemanticSLAM is scene-agnostic, which means it doesn't require retraining for different environments. It operates effectively in indoor settings, even with infrequent camera input, without prior knowledge. The strength of SemanticSLAM lies in its ability to gradually refine the semantic map and improve pose estimation. This is achieved by a convolutional long-short-term-memory (ConvLSTM) network, trained to correct errors during map construction. Compared to existing VSLAM algorithms, SemanticSLAM improves pose estimation by 17%. The resulting semantic map provides interpretable information about the environment and can be easily applied to various downstream tasks, such as path planning, obstacle avoidance, and robot navigation. The code will be publicly available at https://github.com/Leomingyangli/SemanticSLAM

{{</citation>}}


### (86/109) Control-Aware Trajectory Predictions for Communication-Efficient Drone Swarm Coordination in Cluttered Environments (Longhao Yan et al., 2024)

{{<citation>}}

Longhao Yan, Jingyuan Zhou, Kaidi Yang. (2024)  
**Control-Aware Trajectory Predictions for Communication-Efficient Drone Swarm Coordination in Cluttered Environments**  

---
Primary Category: cs.RO  
Categories: I-2-9, cs-RO, cs.RO  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2401.12852v1)  

---


**ABSTRACT**  
Swarms of Unmanned Aerial Vehicles (UAV) have demonstrated enormous potential in many industrial and commercial applications. However, before deploying UAVs in the real world, it is essential to ensure they can operate safely in complex environments, especially with limited communication capabilities. To address this challenge, we propose a control-aware learning-based trajectory prediction algorithm that can enable communication-efficient UAV swarm control in a cluttered environment. Specifically, our proposed algorithm can enable each UAV to predict the planned trajectories of its neighbors in scenarios with various levels of communication capabilities. The predicted planned trajectories will serve as input to a distributed model predictive control (DMPC) approach. The proposed algorithm combines (1) a trajectory compression and reconstruction model based on Variational Auto-Encoder, (2) a trajectory prediction model based on EvolveGCN, a graph convolutional network (GCN) that can handle dynamic graphs, and (3) a KKT-informed training approach that applies the Karush-Kuhn-Tucker (KKT) conditions in the training process to encode DMPC information into the trained neural network. We evaluate our proposed algorithm in a funnel-like environment. Results show that the proposed algorithm outperforms state-of-the-art benchmarks, providing close-to-optimal control performance and robustness to limited communication capabilities and measurement noises.

{{</citation>}}


## eess.IV (1)



### (87/109) CIS-UNet: Multi-Class Segmentation of the Aorta in Computed Tomography Angiography via Context-Aware Shifted Window Self-Attention (Muhammad Imran et al., 2024)

{{<citation>}}

Muhammad Imran, Jonathan R Krebs, Veera Rajasekhar Reddy Gopu, Brian Fazzone, Vishal Balaji Sivaraman, Amarjeet Kumar, Chelsea Viscardi, Robert Evans Heithaus, Benjamin Shickel, Yuyin Zhou, Michol A Cooper, Wei Shao. (2024)  
**CIS-UNet: Multi-Class Segmentation of the Aorta in Computed Tomography Angiography via Context-Aware Shifted Window Self-Attention**  

---
Primary Category: eess.IV  
Categories: cs-AI, cs-CV, cs-GT, cs-LG, eess-IV, eess.IV  
Keywords: Attention, Self-Attention  
[Paper Link](http://arxiv.org/abs/2401.13049v1)  

---


**ABSTRACT**  
Advancements in medical imaging and endovascular grafting have facilitated minimally invasive treatments for aortic diseases. Accurate 3D segmentation of the aorta and its branches is crucial for interventions, as inaccurate segmentation can lead to erroneous surgical planning and endograft construction. Previous methods simplified aortic segmentation as a binary image segmentation problem, overlooking the necessity of distinguishing between individual aortic branches. In this paper, we introduce Context Infused Swin-UNet (CIS-UNet), a deep learning model designed for multi-class segmentation of the aorta and thirteen aortic branches. Combining the strengths of Convolutional Neural Networks (CNNs) and Swin transformers, CIS-UNet adopts a hierarchical encoder-decoder structure comprising a CNN encoder, symmetric decoder, skip connections, and a novel Context-aware Shifted Window Self-Attention (CSW-SA) as the bottleneck block. Notably, CSW-SA introduces a unique utilization of the patch merging layer, distinct from conventional Swin transformers. It efficiently condenses the feature map, providing a global spatial context and enhancing performance when applied at the bottleneck layer, offering superior computational efficiency and segmentation accuracy compared to the Swin transformers. We trained our model on computed tomography (CT) scans from 44 patients and tested it on 15 patients. CIS-UNet outperformed the state-of-the-art SwinUNetR segmentation model, which is solely based on Swin transformers, by achieving a superior mean Dice coefficient of 0.713 compared to 0.697, and a mean surface distance of 2.78 mm compared to 3.39 mm. CIS-UNet's superior 3D aortic segmentation offers improved precision and optimization for planning endovascular treatments. Our dataset and code will be publicly available.

{{</citation>}}


## cs.NI (1)



### (88/109) Chatterbox: Robust Transport for LLM Token Streaming under Unstable Network (Hanchen Li et al., 2024)

{{<citation>}}

Hanchen Li, Yuhan Liu, Yihua Cheng, Siddhant Ray, Kuntai Du, Junchen Jiang. (2024)  
**Chatterbox: Robust Transport for LLM Token Streaming under Unstable Network**  

---
Primary Category: cs.NI  
Categories: cs-LG, cs-NI, cs.NI  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2401.12961v1)  

---


**ABSTRACT**  
To render each generated token in real time, the LLM server generates response tokens one by one and streams each generated token (or group of a few tokens) through the network to the user right after it is generated, which we refer to as LLM token streaming. However, under unstable network conditions, the LLM token streaming experience could suffer greatly from stalls since one packet loss could block the rendering of tokens contained in subsequent packets even if they arrive on time. With a real-world measurement study, we show that current applications including ChatGPT, Claude, and Bard all suffer from increased stall under unstable network.   For this emerging token streaming problem in LLM Chatbots, we propose a novel transport layer scheme, called Chatterbox, which puts new generated tokens as well as currently unacknowledged tokens in the next outgoing packet. This ensures that each packet contains some new tokens and can be independently rendered when received, thus avoiding aforementioned stalls caused by missing packets. Through simulation under various network conditions, we show Chatterbox reduces stall ratio (proportion of token rendering wait time) by 71.0% compared to the token streaming method commonly used by real chatbot applications and by 31.6% compared to a custom packet duplication scheme. By tailoring Chatterbox to fit the token-by-token generation of LLM, we enable the Chatbots to respond like an eloquent speaker for users to better enjoy pervasive AI.

{{</citation>}}


## cs.SY (1)



### (89/109) A Safe Reinforcement Learning Algorithm for Supervisory Control of Power Plants (Yixuan Sun et al., 2024)

{{<citation>}}

Yixuan Sun, Sami Khairy, Richard B. Vilim, Rui Hu, Akshay J. Dave. (2024)  
**A Safe Reinforcement Learning Algorithm for Supervisory Control of Power Plants**  

---
Primary Category: cs.SY  
Categories: cs-LG, cs-SY, cs.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.13020v1)  

---


**ABSTRACT**  
Traditional control theory-based methods require tailored engineering for each system and constant fine-tuning. In power plant control, one often needs to obtain a precise representation of the system dynamics and carefully design the control scheme accordingly. Model-free Reinforcement learning (RL) has emerged as a promising solution for control tasks due to its ability to learn from trial-and-error interactions with the environment. It eliminates the need for explicitly modeling the environment's dynamics, which is potentially inaccurate. However, the direct imposition of state constraints in power plant control raises challenges for standard RL methods. To address this, we propose a chance-constrained RL algorithm based on Proximal Policy Optimization for supervisory control. Our method employs Lagrangian relaxation to convert the constrained optimization problem into an unconstrained objective, where trainable Lagrange multipliers enforce the state constraints. Our approach achieves the smallest distance of violation and violation rate in a load-follow maneuver for an advanced Nuclear Power Plant design.

{{</citation>}}


## stat.ML (1)



### (90/109) Reward-Relevance-Filtered Linear Offline Reinforcement Learning (Angela Zhou, 2024)

{{<citation>}}

Angela Zhou. (2024)  
**Reward-Relevance-Filtered Linear Offline Reinforcement Learning**  

---
Primary Category: stat.ML  
Categories: cs-LG, math-OC, stat-ML, stat.ML  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.12934v1)  

---


**ABSTRACT**  
This paper studies offline reinforcement learning with linear function approximation in a setting with decision-theoretic, but not estimation sparsity. The structural restrictions of the data-generating process presume that the transitions factor into a sparse component that affects the reward and could affect additional exogenous dynamics that do not affect the reward. Although the minimally sufficient adjustment set for estimation of full-state transition properties depends on the whole state, the optimal policy and therefore state-action value function depends only on the sparse component: we call this causal/decision-theoretic sparsity. We develop a method for reward-filtering the estimation of the state-action value function to the sparse component by a modification of thresholded lasso in least-squares policy evaluation. We provide theoretical guarantees for our reward-filtered linear fitted-Q-iteration, with sample complexity depending only on the size of the sparse component.

{{</citation>}}


## cs.SD (1)



### (91/109) Emotion-Aware Contrastive Adaptation Network for Source-Free Cross-Corpus Speech Emotion Recognition (Yan Zhao et al., 2024)

{{<citation>}}

Yan Zhao, Jincen Wang, Cheng Lu, Sunan Li, Björn Schuller, Yuan Zong, Wenming Zheng. (2024)  
**Emotion-Aware Contrastive Adaptation Network for Source-Free Cross-Corpus Speech Emotion Recognition**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2401.12925v1)  

---


**ABSTRACT**  
Cross-corpus speech emotion recognition (SER) aims to transfer emotional knowledge from a labeled source corpus to an unlabeled corpus. However, prior methods require access to source data during adaptation, which is unattainable in real-life scenarios due to data privacy protection concerns. This paper tackles a more practical task, namely source-free cross-corpus SER, where a pre-trained source model is adapted to the target domain without access to source data. To address the problem, we propose a novel method called emotion-aware contrastive adaptation network (ECAN). The core idea is to capture local neighborhood information between samples while considering the global class-level adaptation. Specifically, we propose a nearest neighbor contrastive learning to promote local emotion consistency among features of highly similar samples. Furthermore, relying solely on nearest neighborhoods may lead to ambiguous boundaries between clusters. Thus, we incorporate supervised contrastive learning to encourage greater separation between clusters representing different emotions, thereby facilitating improved class-level adaptation. Extensive experiments indicate that our proposed ECAN significantly outperforms state-of-the-art methods under the source-free cross-corpus SER setting on several speech emotion corpora.

{{</citation>}}


## eess.SY (3)



### (92/109) Deep Learning Based Simulators for the Phosphorus Removal Process Control in Wastewater Treatment via Deep Reinforcement Learning Algorithms (Esmaeel Mohammadi et al., 2024)

{{<citation>}}

Esmaeel Mohammadi, Mikkel Stokholm-Bjerregaard, Aviaja Anna Hansen, Per Halkjær Nielsen, Daniel Ortiz-Arroyo, Petar Durdevic. (2024)  
**Deep Learning Based Simulators for the Phosphorus Removal Process Control in Wastewater Treatment via Deep Reinforcement Learning Algorithms**  

---
Primary Category: eess.SY  
Categories: cs-AI, cs-LG, cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.12822v1)  

---


**ABSTRACT**  
Phosphorus removal is vital in wastewater treatment to reduce reliance on limited resources. Deep reinforcement learning (DRL) is a machine learning technique that can optimize complex and nonlinear systems, including the processes in wastewater treatment plants, by learning control policies through trial and error. However, applying DRL to chemical and biological processes is challenging due to the need for accurate simulators. This study trained six models to identify the phosphorus removal process and used them to create a simulator for the DRL environment. Although the models achieved high accuracy (>97%), uncertainty and incorrect prediction behavior limited their performance as simulators over longer horizons. Compounding errors in the models' predictions were identified as one of the causes of this problem. This approach for improving process control involves creating simulation environments for DRL algorithms, using data from supervisory control and data acquisition (SCADA) systems with a sufficient historical horizon without complex system modeling or parameter estimation.

{{</citation>}}


### (93/109) COOCK project Smart Port 2025 D3.1: 'To Twin Or Not To Twin' (Randy Paredis et al., 2024)

{{<citation>}}

Randy Paredis, Hans Vangheluwe, Pamela Adelino Ramos Albertins. (2024)  
**COOCK project Smart Port 2025 D3.1: 'To Twin Or Not To Twin'**  

---
Primary Category: eess.SY  
Categories: cs-SE, cs-SY, eess-SY, eess.SY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.12747v1)  

---


**ABSTRACT**  
This document is a result of the COOCK project "Smart Port 2025: improving and accelerating the operational efficiency of a harbour eco-system through the application of intelligent technologies". It reports on the needs of companies for modelling and simulation and AI-based techniques, with twinning systems in particular. This document categorizes the purposes and Properties of Interest for the use of Digital Twins. It further illustrates some of the twinning usages, and touches on some of the potential architectural compositions for twins. This last topic will be further elaborated in a followup report.

{{</citation>}}


### (94/109) Learning the cost-to-go for mixed-integer nonlinear model predictive control (Christopher A. Orrico et al., 2024)

{{<citation>}}

Christopher A. Orrico, W. P. M. H. Heemels, Dinesh Krishnamoorthy. (2024)  
**Learning the cost-to-go for mixed-integer nonlinear model predictive control**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY, math-OC  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2401.12562v1)  

---


**ABSTRACT**  
Application of nonlinear model predictive control (NMPC) to problems with hybrid dynamical systems, disjoint constraints, or discrete controls often results in mixed-integer formulations with both continuous and discrete decision variables. However, solving mixed-integer nonlinear programming problems (MINLP) in real-time is challenging, which can be a limiting factor in many applications. To address the computational complexity of solving mixed integer nonlinear model predictive control problem in real-time, this paper proposes an approximate mixed integer NMPC formulation based on value function approximation. Leveraging Bellman's principle of optimality, the key idea here is to divide the prediction horizon into two parts, where the optimal value function of the latter part of the prediction horizon is approximated offline using expert demonstrations. Doing so allows us to solve the MINMPC problem with a considerably shorter prediction horizon online, thereby reducing the online computation cost. The paper uses an inverted pendulum example with discrete controls to illustrate this approach.

{{</citation>}}


## cs.IR (3)



### (95/109) Gradient Flow of Energy: A General and Efficient Approach for Entity Alignment Decoding (Yuanyi Wang et al., 2024)

{{<citation>}}

Yuanyi Wang, Haifeng Sun, Jingyu Wang, Qi Qi, Shaoling Sun, Jianxin Liao. (2024)  
**Gradient Flow of Energy: A General and Efficient Approach for Entity Alignment Decoding**  

---
Primary Category: cs.IR  
Categories: cs-CL, cs-IR, cs.IR  
Keywords: Entity Alignment, GNN, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2401.12798v1)  

---


**ABSTRACT**  
Entity alignment (EA), a pivotal process in integrating multi-source Knowledge Graphs (KGs), seeks to identify equivalent entity pairs across these graphs. Most existing approaches regard EA as a graph representation learning task, concentrating on enhancing graph encoders. However, the decoding process in EA - essential for effective operation and alignment accuracy - has received limited attention and remains tailored to specific datasets and model architectures, necessitating both entity and additional explicit relation embeddings. This specificity limits its applicability, particularly in GNN-based models. To address this gap, we introduce a novel, generalized, and efficient decoding approach for EA, relying solely on entity embeddings. Our method optimizes the decoding process by minimizing Dirichlet energy, leading to the gradient flow within the graph, to promote graph homophily. The discretization of the gradient flow produces a fast and scalable approach, termed Triple Feature Propagation (TFP). TFP innovatively channels gradient flow through three views: entity-to-entity, entity-to-relation, and relation-to-entity. This generalized gradient flow enables TFP to harness the multi-view structural information of KGs. Rigorous experimentation on diverse real-world datasets demonstrates that our approach significantly enhances various EA methods. Notably, the approach achieves these advancements with less than 6 seconds of additional computational time, establishing a new benchmark in efficiency and adaptability for future EA methods.

{{</citation>}}


### (96/109) PolyCF: Towards the Optimal Spectral Graph Filters for Collaborative Filtering (Yifang Qin et al., 2024)

{{<citation>}}

Yifang Qin, Wei Ju, Xiao Luo, Yiyang Gu, Zhiping Xiao, Ming Zhang. (2024)  
**PolyCF: Towards the Optimal Spectral Graph Filters for Collaborative Filtering**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2401.12590v2)  

---


**ABSTRACT**  
Collaborative Filtering (CF) is a pivotal research area in recommender systems that capitalizes on collaborative similarities between users and items to provide personalized recommendations. With the remarkable achievements of node embedding-based Graph Neural Networks (GNNs), we explore the upper bounds of expressiveness inherent to embedding-based methodologies and tackle the challenges by reframing the CF task as a graph signal processing problem. To this end, we propose PolyCF, a flexible graph signal filter that leverages polynomial graph filters to process interaction signals. PolyCF exhibits the capability to capture spectral features across multiple eigenspaces through a series of Generalized Gram filters and is able to approximate the optimal polynomial response function for recovering missing interactions. A graph optimization objective and a pair-wise ranking objective are jointly used to optimize the parameters of the convolution kernel. Experiments on three widely adopted datasets demonstrate the superiority of PolyCF over current state-of-the-art CF methods. Moreover, comprehensive studies empirically validate each component's efficacy in the proposed PolyCF.

{{</citation>}}


### (97/109) Persona-centric Metamorphic Relation guided Robustness Evaluation for Multi-turn Dialogue Modelling (Yanbing Chen et al., 2024)

{{<citation>}}

Yanbing Chen, Lin Li, Xiaohui Tao, Dong Zhou. (2024)  
**Persona-centric Metamorphic Relation guided Robustness Evaluation for Multi-turn Dialogue Modelling**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2401.12483v1)  

---


**ABSTRACT**  
Recently there has been significant progress in the field of dialogue system thanks to the introduction of training paradigms such as fine-tune and prompt learning. Persona can function as the prior knowledge for maintaining the personality consistency of dialogue systems, which makes it perform well on accuracy. Nonetheless, the conventional reference-based evaluation method falls short in capturing the genuine text comprehension prowess of the model, significantly relying on the quality of data annotation. In contrast, the application of metamorphic testing offers a more profound insight into the model's distinct capabilities without necessitating supplementary annotation labels. This approach furnishes a more comprehensive portrayal of the model's intricacies and exposes intricacies concealed within reference-based validation techniques. Consequently, we introduce a persona-centric metamorphic relation construction for metamorphic testing, aimed at evaluating both the persona consistency and robustness of personalized dialogue models. For that reason, this work evaluates several widely used training paradigms including learning from scratch, pretrain + fine-tune and prompt learning in personalized dialogue retrieval to know if they are more robust or if they have the same flaws as their predecessor. Under three kinds of designed metamorphic relations with consistent outputs, our experimental results reveal that prompt learning shows stronger robustness compared to training from scratch and fine-tune. Although tested retrieval models gain competitively high retrieval accuracy according to the traditional reference-based validation, they are still fragile and demonstrate various unexpected behaviors, thus there is still room for future improvement in personalized dialogue retrieval.

{{</citation>}}


## cs.SE (3)



### (98/109) What Can Self-Admitted Technical Debt Tell Us About Security? A Mixed-Methods Study (Nicolás E. Díaz Ferreyra et al., 2024)

{{<citation>}}

Nicolás E. Díaz Ferreyra, Mojtaba Shahin, Mansorreh Zahedi, Sodiq Quadri, Ricardo Scandariato. (2024)  
**What Can Self-Admitted Technical Debt Tell Us About Security? A Mixed-Methods Study**  

---
Primary Category: cs.SE  
Categories: cs-HC, cs-SE, cs.SE  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2401.12768v1)  

---


**ABSTRACT**  
Self-Admitted Technical Debt (SATD) encompasses a wide array of sub-optimal design and implementation choices reported in software artefacts (e.g., code comments and commit messages) by developers themselves. Such reports have been central to the study of software maintenance and evolution over the last decades. However, they can also be deemed as dreadful sources of information on potentially exploitable vulnerabilities and security flaws. This work investigates the security implications of SATD from a technical and developer-centred perspective. On the one hand, it analyses whether security pointers disclosed inside SATD sources can be used to characterise vulnerabilities in Open-Source Software (OSS) projects and repositories. On the other hand, it delves into developers' perspectives regarding the motivations behind this practice, its prevalence, and its potential negative consequences. We followed a mixed-methods approach consisting of (i) the analysis of a preexisting dataset containing 94,455 SATD instances and (ii) an online survey with 222 OSS practitioners. We gathered 201 SATD instances through the dataset analysis and mapped them to different Common Weakness Enumeration (CWE) identifiers. Overall, 25 different types of CWEs were spotted across commit messages, pull requests, code comments, and issue sections, from which 8 appear among MITRE's Top-25 most dangerous ones. The survey shows that software practitioners often place security pointers across SATD artefacts to promote a security culture among their peers and help them spot flaky code sections, among other motives. However, they also consider such a practice risky as it may facilitate vulnerability exploits. Our findings suggest that preserving the contextual integrity of security pointers disseminated across SATD artefacts is critical to safeguard both commercial and OSS solutions against zero-day attacks.

{{</citation>}}


### (99/109) Evaluation of large language models for assessing code maintainability (Marc Dillmann et al., 2024)

{{<citation>}}

Marc Dillmann, Julien Siebert, Adam Trendowicz. (2024)  
**Evaluation of large language models for assessing code maintainability**  

---
Primary Category: cs.SE  
Categories: 68, D-2-7, cs-AI, cs-SE, cs.SE  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2401.12714v1)  

---


**ABSTRACT**  
Increased availability of open-source software repositories and recent advances in code analysis using large language models (LLMs) has triggered a wave of new work to automate software engineering tasks that were previously very difficult to automate. In this paper, we investigate a recent line of work that hypothesises that comparing the probability of code generated by LLMs with the probability the current code would have had can indicate potential quality problems. We investigate the association between the cross-entropy of code generated by ten different models (based on GPT2 and Llama2) and the following quality aspects: readability, understandability, complexity, modularisation, and overall maintainability assessed by experts and available in an benchmark dataset. Our results show that, controlling for the number of logical lines of codes (LLOC), cross-entropy computed by LLMs is indeed a predictor of maintainability on a class level (the higher the cross-entropy the lower the maintainability). However, this relation is reversed when one does not control for LLOC (e.g., comparing small classes with longer ones). Furthermore, while the complexity of LLMs affects the range of cross-entropy (smaller models tend to have a wider range of cross-entropy), this plays a significant role in predicting maintainability aspects. Our study limits itself on ten different pretrained models (based on GPT2 and Llama2) and on maintainability aspects collected by Schnappinger et al. When controlling for logical lines of code (LLOC), cross-entropy is a predictor of maintainability. However, while related work has shown the potential usefulness of cross-entropy at the level of tokens or short sequences, at the class level this criterion alone may prove insufficient to predict maintainability and further research is needed to make best use of this information in practice.

{{</citation>}}


### (100/109) Modeling Resilience of Collaborative AI Systems (Diaeddin Rimawi et al., 2024)

{{<citation>}}

Diaeddin Rimawi, Antonio Liotta, Marco Todescato, Barbara Russo. (2024)  
**Modeling Resilience of Collaborative AI Systems**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-RO, cs-SE, cs.SE  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.12632v1)  

---


**ABSTRACT**  
A Collaborative Artificial Intelligence System (CAIS) performs actions in collaboration with the human to achieve a common goal. CAISs can use a trained AI model to control human-system interaction, or they can use human interaction to dynamically learn from humans in an online fashion. In online learning with human feedback, the AI model evolves by monitoring human interaction through the system sensors in the learning state, and actuates the autonomous components of the CAIS based on the learning in the operational state. Therefore, any disruptive event affecting these sensors may affect the AI model's ability to make accurate decisions and degrade the CAIS performance. Consequently, it is of paramount importance for CAIS managers to be able to automatically track the system performance to understand the resilience of the CAIS upon such disruptive events. In this paper, we provide a new framework to model CAIS performance when the system experiences a disruptive event. With our framework, we introduce a model of performance evolution of CAIS. The model is equipped with a set of measures that aim to support CAIS managers in the decision process to achieve the required resilience of the system. We tested our framework on a real-world case study of a robot collaborating online with the human, when the system is experiencing a disruptive event. The case study shows that our framework can be adopted in CAIS and integrated into the online execution of the CAIS activities.

{{</citation>}}


## cs.CE (1)



### (101/109) From Numbers to Words: Multi-Modal Bankruptcy Prediction Using the ECL Dataset (Henri Arno et al., 2024)

{{<citation>}}

Henri Arno, Klaas Mulier, Joke Baeck, Thomas Demeester. (2024)  
**From Numbers to Words: Multi-Modal Bankruptcy Prediction Using the ECL Dataset**  

---
Primary Category: cs.CE  
Categories: cs-CE, cs.CE, q-fin-CP  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2401.12652v1)  

---


**ABSTRACT**  
In this paper, we present ECL, a novel multi-modal dataset containing the textual and numerical data from corporate 10K filings and associated binary bankruptcy labels. Furthermore, we develop and critically evaluate several classical and neural bankruptcy prediction models using this dataset. Our findings suggest that the information contained in each data modality is complementary for bankruptcy prediction. We also see that the binary bankruptcy prediction target does not enable our models to distinguish next year bankruptcy from an unhealthy financial situation resulting in bankruptcy in later years. Finally, we explore the use of LLMs in the context of our task. We show how GPT-based models can be used to extract meaningful summaries from the textual data but zero-shot bankruptcy prediction results are poor. All resources required to access and update the dataset or replicate our experiments are available on github.com/henriarnoUG/ECL.

{{</citation>}}


## cs.MA (3)



### (102/109) Emergent Cooperation under Uncertain Incentive Alignment (Nicole Orzan et al., 2024)

{{<citation>}}

Nicole Orzan, Erman Acar, Davide Grossi, Roxana Rădulescu. (2024)  
**Emergent Cooperation under Uncertain Incentive Alignment**  

---
Primary Category: cs.MA  
Categories: cs-AI, cs-GT, cs-MA, cs.MA  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.12646v1)  

---


**ABSTRACT**  
Understanding the emergence of cooperation in systems of computational agents is crucial for the development of effective cooperative AI. Interaction among individuals in real-world settings are often sparse and occur within a broad spectrum of incentives, which often are only partially known. In this work, we explore how cooperation can arise among reinforcement learning agents in scenarios characterised by infrequent encounters, and where agents face uncertainty about the alignment of their incentives with those of others. To do so, we train the agents under a wide spectrum of environments ranging from fully competitive, to fully cooperative, to mixed-motives. Under this type of uncertainty we study the effects of mechanisms, such as reputation and intrinsic rewards, that have been proposed in the literature to foster cooperation in mixed-motives environments. Our findings show that uncertainty substantially lowers the agents' ability to engage in cooperative behaviour, when that would be the best course of action. In this scenario, the use of effective reputation mechanisms and intrinsic rewards boosts the agents' capability to act nearly-optimally in cooperative environments, while greatly enhancing cooperation in mixed-motive environments as well.

{{</citation>}}


### (103/109) Backpropagation Through Agents (Zhiyuan Li et al., 2024)

{{<citation>}}

Zhiyuan Li, Wenshuai Zhao, Lijun Wu, Joni Pajarinen. (2024)  
**Backpropagation Through Agents**  

---
Primary Category: cs.MA  
Categories: cs-MA, cs.MA  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2401.12574v1)  

---


**ABSTRACT**  
A fundamental challenge in multi-agent reinforcement learning (MARL) is to learn the joint policy in an extremely large search space, which grows exponentially with the number of agents. Moreover, fully decentralized policy factorization significantly restricts the search space, which may lead to sub-optimal policies. In contrast, the auto-regressive joint policy can represent a much richer class of joint policies by factorizing the joint policy into the product of a series of conditional individual policies. While such factorization introduces the action dependency among agents explicitly in sequential execution, it does not take full advantage of the dependency during learning. In particular, the subsequent agents do not give the preceding agents feedback about their decisions. In this paper, we propose a new framework Back-Propagation Through Agents (BPTA) that directly accounts for both agents' own policy updates and the learning of their dependent counterparts. This is achieved by propagating the feedback through action chains. With the proposed framework, our Bidirectional Proximal Policy Optimisation (BPPO) outperforms the state-of-the-art methods. Extensive experiments on matrix games, StarCraftII v2, Multi-agent MuJoCo, and Google Research Football demonstrate the effectiveness of the proposed method.

{{</citation>}}


### (104/109) Multi-agent deep reinforcement learning with centralized training and decentralized execution for transportation infrastructure management (M. Saifullah et al., 2024)

{{<citation>}}

M. Saifullah, K. G. Papakonstantinou, C. P. Andriotis, S. M. Stoffels. (2024)  
**Multi-agent deep reinforcement learning with centralized training and decentralized execution for transportation infrastructure management**  

---
Primary Category: cs.MA  
Categories: cs-AI, cs-LG, cs-MA, cs-SY, cs.MA, eess-SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.12455v1)  

---


**ABSTRACT**  
We present a multi-agent Deep Reinforcement Learning (DRL) framework for managing large transportation infrastructure systems over their life-cycle. Life-cycle management of such engineering systems is a computationally intensive task, requiring appropriate sequential inspection and maintenance decisions able to reduce long-term risks and costs, while dealing with different uncertainties and constraints that lie in high-dimensional spaces. To date, static age- or condition-based maintenance methods and risk-based or periodic inspection plans have mostly addressed this class of optimization problems. However, optimality, scalability, and uncertainty limitations are often manifested under such approaches. The optimization problem in this work is cast in the framework of constrained Partially Observable Markov Decision Processes (POMDPs), which provides a comprehensive mathematical basis for stochastic sequential decision settings with observation uncertainties, risk considerations, and limited resources. To address significantly large state and action spaces, a Deep Decentralized Multi-agent Actor-Critic (DDMAC) DRL method with Centralized Training and Decentralized Execution (CTDE), termed as DDMAC-CTDE is developed. The performance strengths of the DDMAC-CTDE method are demonstrated in a generally representative and realistic example application of an existing transportation network in Virginia, USA. The network includes several bridge and pavement components with nonstationary degradation, agency-imposed constraints, and traffic delay and risk considerations. Compared to traditional management policies for transportation networks, the proposed DDMAC-CTDE method vastly outperforms its counterparts. Overall, the proposed algorithmic framework provides near optimal solutions for transportation infrastructure management under real-world constraints and complexities.

{{</citation>}}


## cs.AR (1)



### (105/109) Full-Stack Optimization for CAM-Only DNN Inference (João Paulo C. de Lima et al., 2024)

{{<citation>}}

João Paulo C. de Lima, Asif Ali Khan, Luigi Carro, Jeronimo Castrillon. (2024)  
**Full-Stack Optimization for CAM-Only DNN Inference**  

---
Primary Category: cs.AR  
Categories: cs-AR, cs-ET, cs-LG, cs.AR  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2401.12630v1)  

---


**ABSTRACT**  
The accuracy of neural networks has greatly improved across various domains over the past years. Their ever-increasing complexity, however, leads to prohibitively high energy demands and latency in von Neumann systems. Several computing-in-memory (CIM) systems have recently been proposed to overcome this, but trade-offs involving accuracy, hardware reliability, and scalability for large models remain a challenge. Additionally, for some CIM designs, the activation movement still requires considerable time and energy. This paper explores the combination of algorithmic optimizations for ternary weight neural networks and associative processors (APs) implemented using racetrack memory (RTM). We propose a novel compilation flow to optimize convolutions on APs by reducing their arithmetic intensity. By leveraging the benefits of RTM-based APs, this approach substantially reduces data transfers within the memory while addressing accuracy, energy efficiency, and reliability concerns. Concretely, our solution improves the energy efficiency of ResNet-18 inference on ImageNet by 7.5x compared to crossbar in-memory accelerators while retaining software accuracy.

{{</citation>}}


## cs.HC (1)



### (106/109) C2Ideas: Supporting Creative Interior Color Design Ideation with Large Language Model (Yihan Hou et al., 2024)

{{<citation>}}

Yihan Hou, Manling Yang, Hao Cui, Lei Wang, Jie Xu, Wei Zeng. (2024)  
**C2Ideas: Supporting Creative Interior Color Design Ideation with Large Language Model**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.12586v2)  

---


**ABSTRACT**  
Interior color design is a creative process that endeavors to allocate colors to furniture and other elements within an interior space. While much research focuses on generating realistic interior designs, these automated approaches often misalign with user intention and disregard design rationales. Informed by a need-finding preliminary study, we develop C2Ideas, an innovative system for designers to creatively ideate color schemes enabled by an intent-aligned and domain-oriented large language model. C2Ideas integrates a three-stage process: Idea Prompting stage distills user intentions into color linguistic prompts; Word-Color Association stage transforms the prompts into semantically and stylistically coherent color schemes; and Interior Coloring stage assigns colors to interior elements complying with design principles. We also develop an interactive interface that enables flexible user refinement and interpretable reasoning. C2Ideas has undergone a series of indoor cases and user studies, demonstrating its effectiveness and high recognition of interactive functionality by designers.

{{</citation>}}


## cs.DC (1)



### (107/109) Can Large Language Models Write Parallel Code? (Daniel Nichols et al., 2024)

{{<citation>}}

Daniel Nichols, Joshua H. Davis, Zhaojun Xie, Arjun Rajaram, Abhinav Bhatele. (2024)  
**Can Large Language Models Write Parallel Code?**  

---
Primary Category: cs.DC  
Categories: cs-AI, cs-DC, cs.DC  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.12554v1)  

---


**ABSTRACT**  
Large Language Models are becoming an increasingly popular tool for software development. Their ability to model and generate source code has been demonstrated in a variety of contexts, including code completion, summarization, translation, and lookup. However, they often struggle to generate code for more complex tasks. In this paper, we explore the ability of state-of-the-art language models to generate parallel code. We propose a benchmark, PCGBench, consisting of a set of 420 tasks for evaluating the ability of language models to generate parallel code, and we evaluate the performance of several state-of-the-art open- and closed-source language models on these tasks. We introduce novel metrics for comparing parallel code generation performance and use them to explore how well each LLM performs on various parallel programming models and computational problem types.

{{</citation>}}


## cs.CR (1)



### (108/109) Multi-Party Private Set Intersection: A Circuit-Based Protocol with Jaccard Similarity for Secure and Efficient Anomaly Detection in Network Traffic (Jiuheng Su et al., 2024)

{{<citation>}}

Jiuheng Su, Zhili Chen, Xiaomin Yang. (2024)  
**Multi-Party Private Set Intersection: A Circuit-Based Protocol with Jaccard Similarity for Secure and Efficient Anomaly Detection in Network Traffic**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2401.12542v1)  

---


**ABSTRACT**  
We present a new circuit-based protocol for multi-party private set intersection (PSI) that allows m parties to compute the intersection of their datasets without revealing any additional information about the items outside the intersection. Building upon the two-party Sort-Compare-Shuffle (SCS) protocol, we seamlessly extend it to a multi-party setting. Demonstrating its practicality through implementation, our protocol exhibits acceptable performance. Specifically, with 7 parties, each possessing a set size of 2^{12}, our protocol completes in just 19 seconds. Moreover, circuit-based protocols like ours have an advantage over using custom protocols to perform more complex computation. We substantiate this advantage by incorporating a module for calculating the Jaccard similarity metric of the private sets which can be used in the application domain of network traffic analysis for anomaly detection. This extension showcases the versatility of our protocol beyond set intersection computations, demonstrating its efficacy in preserving privacy while efficiently identifying abnormal patterns in network flow.

{{</citation>}}


## cs.IT (1)



### (109/109) AIRS-assisted Vehicular Networks with Rate-Splitting SWIPT Receivers: Joint Trajectory and Communication Design (Gyoungyoon Nam et al., 2024)

{{<citation>}}

Gyoungyoon Nam, Seokhyun Lee, Seongah Jeong. (2024)  
**AIRS-assisted Vehicular Networks with Rate-Splitting SWIPT Receivers: Joint Trajectory and Communication Design**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, math-IT  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.12481v1)  

---


**ABSTRACT**  
In this correspondence, we propose to use an intelligent reflective surface (IRS) installed on unmanned aerial vehicle (UAV), referred to as aerial IRS (AIRS), for vehicular networks, where simultaneous wireless information and power transfer (SWIPT) receivers to concurrently allow information decoding (ID) and energy harvesting (EH) are equipped at the battery-limited vehicles. For efficiently supporting the multiple moving vehicles, we adopt rate-splitting multiple access (RSMA) technique. With the aim of maximizing the sum rate of vehicles, we jointly optimize trajectory and phase shift design of AIRS, transmit power and rate allocation for RSMA along with power splitting ratio for SWIPT implementation. Via simulations, the superior performances of the proposed algorithm are validated compared to the conventional partial optimizations.

{{</citation>}}
