---
draft: false
title: "arXiv @ 2023.07.05"
date: 2023-07-05
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.07.05"
    identifier: arxiv_20230705
    parent: 202307_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.AI (4)](#csai-4)
- [eess.SP (3)](#eesssp-3)
- [cs.DC (1)](#csdc-1)
- [cs.LG (22)](#cslg-22)
- [cs.CL (21)](#cscl-21)
- [cs.SD (3)](#cssd-3)
- [cs.IT (1)](#csit-1)
- [cs.RO (2)](#csro-2)
- [cs.CV (18)](#cscv-18)
- [cs.HC (3)](#cshc-3)
- [math.HO (1)](#mathho-1)
- [cs.CR (3)](#cscr-3)
- [eess.IV (3)](#eessiv-3)
- [cs.CY (2)](#cscy-2)
- [physics.bio-ph (1)](#physicsbio-ph-1)
- [cs.NI (3)](#csni-3)
- [physics.soc-ph (1)](#physicssoc-ph-1)
- [q-bio.NC (1)](#q-bionc-1)
- [cs.DB (1)](#csdb-1)

## cs.AI (4)



### (1/94) Learning to Communicate using Contrastive Learning (Yat Long Lo et al., 2023)

{{<citation>}}

Yat Long Lo, Biswa Sengupta, Jakob Foerster, Michael Noukhovitch. (2023)  
**Learning to Communicate using Contrastive Learning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2307.01403v1)  

---


**ABSTRACT**  
Communication is a powerful tool for coordination in multi-agent RL. But inducing an effective, common language is a difficult challenge, particularly in the decentralized setting. In this work, we introduce an alternative perspective where communicative messages sent between agents are considered as different incomplete views of the environment state. By examining the relationship between messages sent and received, we propose to learn to communicate using contrastive learning to maximize the mutual information between messages of a given trajectory. In communication-essential environments, our method outperforms previous work in both performance and learning speed. Using qualitative metrics and representation probing, we show that our method induces more symmetric communication and captures global state information from the environment. Overall, we show the power of contrastive learning and the importance of leveraging messages as encodings for effective communication.

{{</citation>}}


### (2/94) Reliable AI: Does the Next Generation Require Quantum Computing? (Aras Bacho et al., 2023)

{{<citation>}}

Aras Bacho, Holger Boche, Gitta Kutyniok. (2023)  
**Reliable AI: Does the Next Generation Require Quantum Computing?**  

---
Primary Category: cs.AI  
Categories: 15A29, 35J05, 46N10, 68Q04, 68Q12, 68Q17, 68Q25, cs-AI, cs.AI, quant-ph  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.01301v2)  

---


**ABSTRACT**  
In this survey, we aim to explore the fundamental question of whether the next generation of artificial intelligence requires quantum computing. Artificial intelligence is increasingly playing a crucial role in many aspects of our daily lives and is central to the fourth industrial revolution. It is therefore imperative that artificial intelligence is reliable and trustworthy. However, there are still many issues with reliability of artificial intelligence, such as privacy, responsibility, safety, and security, in areas such as autonomous driving, healthcare, robotics, and others. These problems can have various causes, including insufficient data, biases, and robustness problems, as well as fundamental issues such as computability problems on digital hardware. The cause of these computability problems is rooted in the fact that digital hardware is based on the computing model of the Turing machine, which is inherently discrete. Notably, our findings demonstrate that digital hardware is inherently constrained in solving problems about optimization, deep learning, or differential equations. Therefore, these limitations carry substantial implications for the field of artificial intelligence, in particular for machine learning. Furthermore, although it is well known that the quantum computer shows a quantum advantage for certain classes of problems, our findings establish that some of these limitations persist when employing quantum computing models based on the quantum circuit or the quantum Turing machine paradigm. In contrast, analog computing models, such as the Blum-Shub-Smale machine, exhibit the potential to surmount these limitations.

{{</citation>}}


### (3/94) ChatGPT vs. Google: A Comparative Study of Search Performance and User Experience (Ruiyun Xu et al., 2023)

{{<citation>}}

Ruiyun Xu, Yue Feng, Hailiang Chen. (2023)  
**ChatGPT vs. Google: A Comparative Study of Search Performance and User Experience**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-HC, cs-IR, cs.AI  
Keywords: ChatGPT, GPT, Google  
[Paper Link](http://arxiv.org/abs/2307.01135v1)  

---


**ABSTRACT**  
The advent of ChatGPT, a large language model-powered chatbot, has prompted questions about its potential implications for traditional search engines. In this study, we investigate the differences in user behavior when employing search engines and chatbot tools for information-seeking tasks. We carry out a randomized online experiment, dividing participants into two groups: one using a ChatGPT-like tool and the other using a Google Search-like tool. Our findings reveal that the ChatGPT group consistently spends less time on all tasks, with no significant difference in overall task performance between the groups. Notably, ChatGPT levels user search performance across different education levels and excels in answering straightforward questions and providing general solutions but falls short in fact-checking tasks. Users perceive ChatGPT's responses as having higher information quality compared to Google Search, despite displaying a similar level of trust in both tools. Furthermore, participants using ChatGPT report significantly better user experiences in terms of usefulness, enjoyment, and satisfaction, while perceived ease of use remains comparable between the two tools. However, ChatGPT may also lead to overreliance and generate or replicate misinformation, yielding inconsistent results. Our study offers valuable insights for search engine management and highlights opportunities for integrating chatbot technologies into search engine designs.

{{</citation>}}


### (4/94) Towards Explainable AI for Channel Estimation in Wireless Communications (Abdul Karim Gizzini et al., 2023)

{{<citation>}}

Abdul Karim Gizzini, Yahia Medjahdi, Ali J. Ghandour, Laurent Clavier. (2023)  
**Towards Explainable AI for Channel Estimation in Wireless Communications**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-IT, cs.AI, math-IT  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.00952v1)  

---


**ABSTRACT**  
Research into 6G networks has been initiated to support a variety of critical artificial intelligence (AI) assisted applications such as autonomous driving. In such applications, AI-based decisions should be performed in a real-time manner. These decisions include resource allocation, localization, channel estimation, etc. Considering the black-box nature of existing AI-based models, it is highly challenging to understand and trust the decision-making behavior of such models. Therefore, explaining the logic behind those models through explainable AI (XAI) techniques is essential for their employment in critical applications. This manuscript proposes a novel XAI-based channel estimation (XAI-CHEST) scheme that provides detailed reasonable interpretability of the deep learning (DL) models that are employed in doubly-selective channel estimation. The aim of the proposed XAI-CHEST scheme is to identify the relevant model inputs by inducing high noise on the irrelevant ones. As a result, the behavior of the studied DL-based channel estimators can be further analyzed and evaluated based on the generated interpretations. Simulation results show that the proposed XAI-CHEST scheme provides valid interpretations of the DL-based channel estimators for different scenarios.

{{</citation>}}


## eess.SP (3)



### (5/94) Precheck Sequence Based False Base Station Detection During Handover: A Physical Layer Based Security Scheme (Xiangyu Li et al., 2023)

{{<citation>}}

Xiangyu Li, Kaiwen Zheng, Sidong Guo, Xiaoli Ma. (2023)  
**Precheck Sequence Based False Base Station Detection During Handover: A Physical Layer Based Security Scheme**  

---
Primary Category: eess.SP  
Categories: cs-NI, eess-SP, eess.SP  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2307.01396v1)  

---


**ABSTRACT**  
False Base Station (FBS) attack has been a severe security problem for the cellular network since 2G era. During handover, the user equipment (UE) periodically receives state information from surrounding base stations (BSs) and uploads it to the source BS. The source BS compares the uploaded signal power and shifts UE to another BS that can provide the strongest signal. An FBS can transmit signal with the proper power and attract UE to connect to it. In this paper, based on the 3GPP standard, a Precheck Sequence-based Detection (PSD) Scheme is proposed to secure the transition of legal base station (LBS) for UE. This scheme first analyzes the structure of received signals in blocks and symbols. Several additional symbols are added to the current signal sequence for verification. By designing a long table of symbol sequence, every UE which needs handover will be allocated a specific sequence from this table. The simulation results show that the performance of this PSD Scheme is better than that of any existing ones, even when a specific transmit power is designed for FBS.

{{</citation>}}


### (6/94) Unbiased Pain Assessment through Wearables and EHR Data: Multi-attribute Fairness Loss-based CNN Approach (Sharmin Sultana et al., 2023)

{{<citation>}}

Sharmin Sultana, Md Mahmudur Rahman, Atqiya Munawara Mahi, Shao-Hsien Liu, Mohammad Arif Ul Alam. (2023)  
**Unbiased Pain Assessment through Wearables and EHR Data: Multi-attribute Fairness Loss-based CNN Approach**  

---
Primary Category: eess.SP  
Categories: cs-AI, cs-LG, eess-SP, eess.SP  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.05333v1)  

---


**ABSTRACT**  
The combination of diverse health data (IoT, EHR, and clinical surveys) and scalable-adaptable Artificial Intelligence (AI), has enabled the discovery of physical, behavioral, and psycho-social indicators of pain status. Despite the hype and promise to fundamentally alter the healthcare system with technological advancements, much AI adoption in clinical pain evaluation has been hampered by the heterogeneity of the problem itself and other challenges, such as personalization and fairness. Studies have revealed that many AI (i.e., machine learning or deep learning) models display biases and discriminate against specific population segments (such as those based on gender or ethnicity), which breeds skepticism among medical professionals about AI adaptability. In this paper, we propose a Multi-attribute Fairness Loss (MAFL) based CNN model that aims to account for any sensitive attributes included in the data and fairly predict patients' pain status while attempting to minimize the discrepancies between privileged and unprivileged groups. In order to determine whether the trade-off between accuracy and fairness can be satisfied, we compare the proposed model with well-known existing mitigation procedures, and studies reveal that the implemented model performs favorably in contrast to state-of-the-art methods. Utilizing NIH All-Of-US data, where a cohort of 868 distinct individuals with wearables and EHR data gathered over 1500 days has been taken into consideration to analyze our suggested fair pain assessment system.

{{</citation>}}


### (7/94) Classification of sleep stages from EEG, EOG and EMG signals by SSNet (Haifa Almutairi et al., 2023)

{{<citation>}}

Haifa Almutairi, Ghulam Mubashar Hassan, Amitava Datta. (2023)  
**Classification of sleep stages from EEG, EOG and EMG signals by SSNet**  

---
Primary Category: eess.SP  
Categories: cs-AI, cs-LG, eess-SP, eess.SP  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2307.05373v1)  

---


**ABSTRACT**  
Classification of sleep stages plays an essential role in diagnosing sleep-related diseases including Sleep Disorder Breathing (SDB) disease. In this study, we propose an end-to-end deep learning architecture, named SSNet, which comprises of two deep learning networks based on Convolutional Neuron Networks (CNN) and Long Short Term Memory (LSTM). Both deep learning networks extract features from the combination of Electrooculogram (EOG), Electroencephalogram (EEG), and Electromyogram (EMG) signals, as each signal has distinct features that help in the classification of sleep stages. The features produced by the two-deep learning networks are concatenated to pass to the fully connected layer for the classification. The performance of our proposed model is evaluated by using two public datasets Sleep-EDF Expanded dataset and ISRUC-Sleep dataset. The accuracy and Kappa coefficient are 96.36% and 93.40% respectively, for classifying three classes of sleep stages using Sleep-EDF Expanded dataset. Whereas, the accuracy and Kappa coefficient are 96.57% and 83.05% respectively for five classes of sleep stages using Sleep-EDF Expanded dataset. Our model achieves the best performance in classifying sleep stages when compared with the state-of-the-art techniques.

{{</citation>}}


## cs.DC (1)



### (8/94) In-depth Analysis On Parallel Processing Patterns for High-Performance Dataframes (Niranda Perera et al., 2023)

{{<citation>}}

Niranda Perera, Arup Kumar Sarker, Mills Staylor, Gregor von Laszewski, Kaiying Shan, Supun Kamburugamuve, Chathura Widanage, Vibhatha Abeykoon, Thejaka Amila Kanewela, Geoffrey Fox. (2023)  
**In-depth Analysis On Parallel Processing Patterns for High-Performance Dataframes**  

---
Primary Category: cs.DC  
Categories: cs-AI, cs-DC, cs-IR, cs-LG, cs.DC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.01394v1)  

---


**ABSTRACT**  
The Data Science domain has expanded monumentally in both research and industry communities during the past decade, predominantly owing to the Big Data revolution. Artificial Intelligence (AI) and Machine Learning (ML) are bringing more complexities to data engineering applications, which are now integrated into data processing pipelines to process terabytes of data. Typically, a significant amount of time is spent on data preprocessing in these pipelines, and hence improving its e fficiency directly impacts the overall pipeline performance. The community has recently embraced the concept of Dataframes as the de-facto data structure for data representation and manipulation. However, the most widely used serial Dataframes today (R, pandas) experience performance limitations while working on even moderately large data sets. We believe that there is plenty of room for improvement by taking a look at this problem from a high-performance computing point of view. In a prior publication, we presented a set of parallel processing patterns for distributed dataframe operators and the reference runtime implementation, Cylon [1]. In this paper, we are expanding on the initial concept by introducing a cost model for evaluating the said patterns. Furthermore, we evaluate the performance of Cylon on the ORNL Summit supercomputer.

{{</citation>}}


## cs.LG (22)



### (9/94) Adversarial Learning in Real-World Fraud Detection: Challenges and Perspectives (Danele Lunghi et al., 2023)

{{<citation>}}

Danele Lunghi, Alkis Simitsis, Olivier Caelen, Gianluca Bontempi. (2023)  
**Adversarial Learning in Real-World Fraud Detection: Challenges and Perspectives**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG  
Keywords: Fraud Detection  
[Paper Link](http://arxiv.org/abs/2307.01390v1)  

---


**ABSTRACT**  
Data economy relies on data-driven systems and complex machine learning applications are fueled by them. Unfortunately, however, machine learning models are exposed to fraudulent activities and adversarial attacks, which threaten their security and trustworthiness. In the last decade or so, the research interest on adversarial machine learning has grown significantly, revealing how learning applications could be severely impacted by effective attacks. Although early results of adversarial machine learning indicate the huge potential of the approach to specific domains such as image processing, still there is a gap in both the research literature and practice regarding how to generalize adversarial techniques in other domains and applications. Fraud detection is a critical defense mechanism for data economy, as it is for other applications as well, which poses several challenges for machine learning. In this work, we describe how attacks against fraud detection systems differ from other applications of adversarial machine learning, and propose a number of interesting directions to bridge this gap.

{{</citation>}}


### (10/94) Systematic Bias in Sample Inference and its Effect on Machine Learning (Owen O'Neill et al., 2023)

{{<citation>}}

Owen O'Neill, Fintan Costello. (2023)  
**Systematic Bias in Sample Inference and its Effect on Machine Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ME  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2307.01384v1)  

---


**ABSTRACT**  
A commonly observed pattern in machine learning models is an underprediction of the target feature, with the model's predicted target rate for members of a given category typically being lower than the actual target rate for members of that category in the training set. This underprediction is usually larger for members of minority groups; while income level is underpredicted for both men and women in the 'adult' dataset, for example, the degree of underprediction is significantly higher for women (a minority in that dataset). We propose that this pattern of underprediction for minorities arises as a predictable consequence of statistical inference on small samples. When presented with a new individual for classification, an ML model performs inference not on the entire training set, but on a subset that is in some way similar to the new individual, with sizes of these subsets typically following a power law distribution so that most are small (and with these subsets being necessarily smaller for the minority group). We show that such inference on small samples is subject to systematic and directional statistical bias, and that this bias produces the observed patterns of underprediction seen in ML models. Analysing a standard sklearn decision tree model's predictions on a set of over 70 subsets of the 'adult' and COMPAS datasets, we found that a bias prediction measure based on small-sample inference had a significant positive correlations (0.56 and 0.85) with the observed underprediction rate for these subsets.

{{</citation>}}


### (11/94) PlanE: Representation Learning over Planar Graphs (Radoslav Dimitrov et al., 2023)

{{<citation>}}

Radoslav Dimitrov, Zeyang Zhao, Ralph Abboud, İsmail İlkan Ceylan. (2023)  
**PlanE: Representation Learning over Planar Graphs**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2307.01180v1)  

---


**ABSTRACT**  
Graph neural networks are prominent models for representation learning over graphs, where the idea is to iteratively compute representations of nodes of an input graph through a series of transformations in such a way that the learned graph function is isomorphism invariant on graphs, which makes the learned representations graph invariants. On the other hand, it is well-known that graph invariants learned by these class of models are incomplete: there are pairs of non-isomorphic graphs which cannot be distinguished by standard graph neural networks. This is unsurprising given the computational difficulty of graph isomorphism testing on general graphs, but the situation begs to differ for special graph classes, for which efficient graph isomorphism testing algorithms are known, such as planar graphs. The goal of this work is to design architectures for efficiently learning complete invariants of planar graphs. Inspired by the classical planar graph isomorphism algorithm of Hopcroft and Tarjan, we propose PlanE as a framework for planar representation learning. PlanE includes architectures which can learn complete invariants over planar graphs while remaining practically scalable. We empirically validate the strong performance of the resulting model architectures on well-known planar graph benchmarks, achieving multiple state-of-the-art results.

{{</citation>}}


### (12/94) Don't freeze: Finetune encoders for better Self-Supervised HAR (Vitor Fortes Rey et al., 2023)

{{<citation>}}

Vitor Fortes Rey, Dominique Nshimyimana, Paul Lukowicz. (2023)  
**Don't freeze: Finetune encoders for better Self-Supervised HAR**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2307.01168v1)  

---


**ABSTRACT**  
Recently self-supervised learning has been proposed in the field of human activity recognition as a solution to the labelled data availability problem. The idea being that by using pretext tasks such as reconstruction or contrastive predictive coding, useful representations can be learned that then can be used for classification. Those approaches follow the pretrain, freeze and fine-tune procedure. In this paper we will show how a simple change - not freezing the representation - leads to substantial performance gains across pretext tasks. The improvement was found in all four investigated datasets and across all four pretext tasks and is inversely proportional to amount of labelled data. Moreover the effect is present whether the pretext task is carried on the Capture24 dataset or directly in unlabelled data of the target dataset.

{{</citation>}}


### (13/94) Theory of Mind as Intrinsic Motivation for Multi-Agent Reinforcement Learning (Ini Oguntola et al., 2023)

{{<citation>}}

Ini Oguntola, Joseph Campbell, Simon Stepputtis, Katia Sycara. (2023)  
**Theory of Mind as Intrinsic Motivation for Multi-Agent Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-MA, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.01158v2)  

---


**ABSTRACT**  
The ability to model the mental states of others is crucial to human social intelligence, and can offer similar benefits to artificial agents with respect to the social dynamics induced in multi-agent settings. We present a method of grounding semantically meaningful, human-interpretable beliefs within policies modeled by deep networks. We then consider the task of 2nd-order belief prediction. We propose that ability of each agent to predict the beliefs of the other agents can be used as an intrinsic reward signal for multi-agent reinforcement learning. Finally, we present preliminary empirical results in a mixed cooperative-competitive environment.

{{</citation>}}


### (14/94) ENGAGE: Explanation Guided Data Augmentation for Graph Representation Learning (Yucheng Shi et al., 2023)

{{<citation>}}

Yucheng Shi, Kaixiong Zhou, Ninghao Liu. (2023)  
**ENGAGE: Explanation Guided Data Augmentation for Graph Representation Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-IT, cs-LG, cs.LG, math-IT  
Keywords: Augmentation, Representation Learning  
[Paper Link](http://arxiv.org/abs/2307.01053v1)  

---


**ABSTRACT**  
The recent contrastive learning methods, due to their effectiveness in representation learning, have been widely applied to modeling graph data. Random perturbation is widely used to build contrastive views for graph data, which however, could accidentally break graph structures and lead to suboptimal performance. In addition, graph data is usually highly abstract, so it is hard to extract intuitive meanings and design more informed augmentation schemes. Effective representations should preserve key characteristics in data and abandon superfluous information. In this paper, we propose ENGAGE (ExplaNation Guided data AuGmEntation), where explanation guides the contrastive augmentation process to preserve the key parts in graphs and explore removing superfluous information. Specifically, we design an efficient unsupervised explanation method called smoothed activation map as the indicator of node importance in representation learning. Then, we design two data augmentation schemes on graphs for perturbing structural and feature information, respectively. We also provide justification for the proposed method in the framework of information theories. Experiments of both graph-level and node-level tasks, on various model architectures and on different real-world graphs, are conducted to demonstrate the effectiveness and flexibility of ENGAGE. The code of ENGAGE can be found: https://github.com/sycny/ENGAGE.

{{</citation>}}


### (15/94) Neural Chronos ODE: Unveiling Temporal Patterns and Forecasting Future and Past Trends in Time Series Data (C. Coelho et al., 2023)

{{<citation>}}

C. Coelho, M. Fernanda P. Costa, L. L. Ferrás. (2023)  
**Neural Chronos ODE: Unveiling Temporal Patterns and Forecasting Future and Past Trends in Time Series Data**  

---
Primary Category: cs.LG  
Categories: I-5-1; G-1-7, cs-LG, cs.LG  
Keywords: LSTM, Time Series  
[Paper Link](http://arxiv.org/abs/2307.01023v1)  

---


**ABSTRACT**  
This work introduces Neural Chronos Ordinary Differential Equations (Neural CODE), a deep neural network architecture that fits a continuous-time ODE dynamics for predicting the chronology of a system both forward and backward in time. To train the model, we solve the ODE as an initial value problem and a final value problem, similar to Neural ODEs. We also explore two approaches to combining Neural CODE with Recurrent Neural Networks by replacing Neural ODE with Neural CODE (CODE-RNN), and incorporating a bidirectional RNN for full information flow in both time directions (CODE-BiRNN), and variants with other update cells namely GRU and LSTM: CODE-GRU, CODE-BiGRU, CODE-LSTM, CODE-BiLSTM.   Experimental results demonstrate that Neural CODE outperforms Neural ODE in learning the dynamics of a spiral forward and backward in time, even with sparser data. We also compare the performance of CODE-RNN/-GRU/-LSTM and CODE-BiRNN/-BiGRU/-BiLSTM against ODE-RNN/-GRU/-LSTM on three real-life time series data tasks: imputation of missing data for lower and higher dimensional data, and forward and backward extrapolation with shorter and longer time horizons. Our findings show that the proposed architectures converge faster, with CODE-BiRNN/-BiGRU/-BiLSTM consistently outperforming the other architectures on all tasks.

{{</citation>}}


### (16/94) MoVie: Visual Model-Based Policy Adaptation for View Generalization (Sizhe Yang et al., 2023)

{{<citation>}}

Sizhe Yang, Yanjie Ze, Huazhe Xu. (2023)  
**MoVie: Visual Model-Based Policy Adaptation for View Generalization**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs-RO, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.00972v1)  

---


**ABSTRACT**  
Visual Reinforcement Learning (RL) agents trained on limited views face significant challenges in generalizing their learned abilities to unseen views. This inherent difficulty is known as the problem of $\textit{view generalization}$. In this work, we systematically categorize this fundamental problem into four distinct and highly challenging scenarios that closely resemble real-world situations. Subsequently, we propose a straightforward yet effective approach to enable successful adaptation of visual $\textbf{Mo}$del-based policies for $\textbf{Vie}$w generalization ($\textbf{MoVie}$) during test time, without any need for explicit reward signals and any modification during training time. Our method demonstrates substantial advancements across all four scenarios encompassing a total of $\textbf{18}$ tasks sourced from DMControl, xArm, and Adroit, with a relative improvement of $\mathbf{33}$%, $\mathbf{86}$%, and $\mathbf{152}$% respectively. The superior results highlight the immense potential of our approach for real-world robotics applications. Videos are available at https://yangsizhe.github.io/MoVie/ .

{{</citation>}}


### (17/94) REAL: A Representative Error-Driven Approach for Active Learning (Cheng Chen et al., 2023)

{{<citation>}}

Cheng Chen, Yong Wang, Lizi Liao, Yueguo Chen, Xiaoyong Du. (2023)  
**REAL: A Representative Error-Driven Approach for Active Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2307.00968v2)  

---


**ABSTRACT**  
Given a limited labeling budget, active learning (AL) aims to sample the most informative instances from an unlabeled pool to acquire labels for subsequent model training. To achieve this, AL typically measures the informativeness of unlabeled instances based on uncertainty and diversity. However, it does not consider erroneous instances with their neighborhood error density, which have great potential to improve the model performance. To address this limitation, we propose $REAL$, a novel approach to select data instances with $\underline{R}$epresentative $\underline{E}$rrors for $\underline{A}$ctive $\underline{L}$earning. It identifies minority predictions as \emph{pseudo errors} within a cluster and allocates an adaptive sampling budget for the cluster based on estimated error density. Extensive experiments on five text classification datasets demonstrate that $REAL$ consistently outperforms all best-performing baselines regarding accuracy and F1-macro scores across a wide range of hyperparameter settings. Our analysis also shows that $REAL$ selects the most representative pseudo errors that match the distribution of ground-truth errors along the decision boundary. Our code is publicly available at https://github.com/withchencheng/ECML_PKDD_23_Real.

{{</citation>}}


### (18/94) OpenClinicalAI: An Open and Dynamic Model for Alzheimer's Disease Diagnosis (Yunyou Huang et al., 2023)

{{<citation>}}

Yunyou Huang, Xiaoshuang Liang, Xiangjiang Lu, Xiuxia Miao, Jiyue Xie, Wenjing Liu, Fan Zhang, Guoxin Kang, Li Ma, Suqin Tang, Zhifei Zhang, Jianfeng Zhan. (2023)  
**OpenClinicalAI: An Open and Dynamic Model for Alzheimer's Disease Diagnosis**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI, Clinical  
[Paper Link](http://arxiv.org/abs/2307.00965v1)  

---


**ABSTRACT**  
Although Alzheimer's disease (AD) cannot be reversed or cured, timely diagnosis can significantly reduce the burden of treatment and care. Current research on AD diagnosis models usually regards the diagnosis task as a typical classification task with two primary assumptions: 1) All target categories are known a priori; 2) The diagnostic strategy for each patient is consistent, that is, the number and type of model input data for each patient are the same. However, real-world clinical settings are open, with complexity and uncertainty in terms of both subjects and the resources of the medical institutions. This means that diagnostic models may encounter unseen disease categories and need to dynamically develop diagnostic strategies based on the subject's specific circumstances and available medical resources. Thus, the AD diagnosis task is tangled and coupled with the diagnosis strategy formulation. To promote the application of diagnostic systems in real-world clinical settings, we propose OpenClinicalAI for direct AD diagnosis in complex and uncertain clinical settings. This is the first powerful end-to-end model to dynamically formulate diagnostic strategies and provide diagnostic results based on the subject's conditions and available medical resources. OpenClinicalAI combines reciprocally coupled deep multiaction reinforcement learning (DMARL) for diagnostic strategy formulation and multicenter meta-learning (MCML) for open-set recognition. The experimental results show that OpenClinicalAI achieves better performance and fewer clinical examinations than the state-of-the-art model. Our method provides an opportunity to embed the AD diagnostic system into the current health care system to cooperate with clinicians to improve current health care.

{{</citation>}}


### (19/94) Rockmate: an Efficient, Fast, Automatic and Generic Tool for Re-materialization in PyTorch (Xunyi Zhao et al., 2023)

{{<citation>}}

Xunyi Zhao, Théotime Le Hellard, Lionel Eyraud, Julia Gusak, Olivier Beaumont. (2023)  
**Rockmate: an Efficient, Fast, Automatic and Generic Tool for Re-materialization in PyTorch**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-PL, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.01236v1)  

---


**ABSTRACT**  
We propose Rockmate to control the memory requirements when training PyTorch DNN models. Rockmate is an automatic tool that starts from the model code and generates an equivalent model, using a predefined amount of memory for activations, at the cost of a few re-computations. Rockmate automatically detects the structure of computational and data dependencies and rewrites the initial model as a sequence of complex blocks. We show that such a structure is widespread and can be found in many models in the literature (Transformer based models, ResNet, RegNets,...). This structure allows us to solve the problem in a fast and efficient way, using an adaptation of Checkmate (too slow on the whole model but general) at the level of individual blocks and an adaptation of Rotor (fast but limited to sequential models) at the level of the sequence itself. We show through experiments on many models that Rockmate is as fast as Rotor and as efficient as Checkmate, and that it allows in many cases to obtain a significantly lower memory consumption for activations (by a factor of 2 to 5) for a rather negligible overhead (of the order of 10% to 20%). Rockmate is open source and available at https://github.com/topal-team/rockmate.

{{</citation>}}


### (20/94) Learning Differentiable Logic Programs for Abstract Visual Reasoning (Hikaru Shindo et al., 2023)

{{<citation>}}

Hikaru Shindo, Viktor Pfanschilling, Devendra Singh Dhami, Kristian Kersting. (2023)  
**Learning Differentiable Logic Programs for Abstract Visual Reasoning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2307.00928v1)  

---


**ABSTRACT**  
Visual reasoning is essential for building intelligent agents that understand the world and perform problem-solving beyond perception. Differentiable forward reasoning has been developed to integrate reasoning with gradient-based machine learning paradigms. However, due to the memory intensity, most existing approaches do not bring the best of the expressivity of first-order logic, excluding a crucial ability to solve abstract visual reasoning, where agents need to perform reasoning by using analogies on abstract concepts in different scenarios. To overcome this problem, we propose NEUro-symbolic Message-pAssiNg reasoNer (NEUMANN), which is a graph-based differentiable forward reasoner, passing messages in a memory-efficient manner and handling structured programs with functors. Moreover, we propose a computationally-efficient structure learning algorithm to perform explanatory program induction on complex visual scenes. To evaluate, in addition to conventional visual reasoning tasks, we propose a new task, visual reasoning behind-the-scenes, where agents need to learn abstract programs and then answer queries by imagining scenes that are not observed. We empirically demonstrate that NEUMANN solves visual reasoning tasks efficiently, outperforming neural, symbolic, and neuro-symbolic baselines.

{{</citation>}}


### (21/94) Achieving Stable Training of Reinforcement Learning Agents in Bimodal Environments through Batch Learning (E. Hurwitz et al., 2023)

{{<citation>}}

E. Hurwitz, N. Peace, G. Cevora. (2023)  
**Achieving Stable Training of Reinforcement Learning Agents in Bimodal Environments through Batch Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.00923v1)  

---


**ABSTRACT**  
Bimodal, stochastic environments present a challenge to typical Reinforcement Learning problems. This problem is one that is surprisingly common in real world applications, being particularly applicable to pricing problems. In this paper we present a novel learning approach to the tabular Q-learning algorithm, tailored to tackling these specific challenges by using batch updates. A simulation of pricing problem is used as a testbed to compare a typically updated agent with a batch learning agent. The batch learning agents are shown to be both more effective than the typically-trained agents, and to be more resilient to the fluctuations in a large stochastic environment. This work has a significant potential to enable practical, industrial deployment of Reinforcement Learning in the context of pricing and others.

{{</citation>}}


### (22/94) Fixing confirmation bias in feature attribution methods via semantic match (Giovanni Cinà et al., 2023)

{{<citation>}}

Giovanni Cinà, Daniel Fernandez-Llaneza, Nishant Mishra, Tabea E. Röber, Sandro Pezzelle, Iacer Calixto, Rob Goedhart, Ş. İlker Birbil. (2023)  
**Fixing confirmation bias in feature attribution methods via semantic match**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.00897v1)  

---


**ABSTRACT**  
Feature attribution methods have become a staple method to disentangle the complex behavior of black box models. Despite their success, some scholars have argued that such methods suffer from a serious flaw: they do not allow a reliable interpretation in terms of human concepts. Simply put, visualizing an array of feature contributions is not enough for humans to conclude something about a model's internal representations, and confirmation bias can trick users into false beliefs about model behavior. We argue that a structured approach is required to test whether our hypotheses on the model are confirmed by the feature attributions. This is what we call the "semantic match" between human concepts and (sub-symbolic) explanations. Building on the conceptual framework put forward in Cin\`a et al. [2023], we propose a structured approach to evaluate semantic match in practice. We showcase the procedure in a suite of experiments spanning tabular and image data, and show how the assessment of semantic match can give insight into both desirable (e.g., focusing on an object relevant for prediction) and undesirable model behaviors (e.g., focusing on a spurious correlation). We couple our experimental results with an analysis on the metrics to measure semantic match, and argue that this approach constitutes the first step towards resolving the issue of confirmation bias in XAI.

{{</citation>}}


### (23/94) A Survey on Graph Classification and Link Prediction based on GNN (Xingyu Liu et al., 2023)

{{<citation>}}

Xingyu Liu, Juan Chen, Quan Wen. (2023)  
**A Survey on Graph Classification and Link Prediction based on GNN**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2307.00865v1)  

---


**ABSTRACT**  
Traditional convolutional neural networks are limited to handling Euclidean space data, overlooking the vast realm of real-life scenarios represented as graph data, including transportation networks, social networks, and reference networks. The pivotal step in transferring convolutional neural networks to graph data analysis and processing lies in the construction of graph convolutional operators and graph pooling operators. This comprehensive review article delves into the world of graph convolutional neural networks. Firstly, it elaborates on the fundamentals of graph convolutional neural networks. Subsequently, it elucidates the graph neural network models based on attention mechanisms and autoencoders, summarizing their application in node classification, graph classification, and link prediction along with the associated datasets.

{{</citation>}}


### (24/94) CardiGraphormer: Unveiling the Power of Self-Supervised Learning in Revolutionizing Drug Discovery (Abhijit Gupta et al., 2023)

{{<citation>}}

Abhijit Gupta, Arnab Mukherjee. (2023)  
**CardiGraphormer: Unveiling the Power of Self-Supervised Learning in Revolutionizing Drug Discovery**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, q-bio-QM, stat-AP, stat-ML  
Keywords: AI, Attention, GNN, Graph Neural Network, Graph Neural Networks, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2307.00859v2)  

---


**ABSTRACT**  
In the expansive realm of drug discovery, with approximately 15,000 known drugs and only around 4,200 approved, the combinatorial nature of the chemical space presents a formidable challenge. While Artificial Intelligence (AI) has emerged as a powerful ally, traditional AI frameworks face significant hurdles. This manuscript introduces CardiGraphormer, a groundbreaking approach that synergizes self-supervised learning (SSL), Graph Neural Networks (GNNs), and Cardinality Preserving Attention to revolutionize drug discovery. CardiGraphormer, a novel combination of Graphormer and Cardinality Preserving Attention, leverages SSL to learn potent molecular representations and employs GNNs to extract molecular fingerprints, enhancing predictive performance and interpretability while reducing computation time. It excels in handling complex data like molecular structures and performs tasks associated with nodes, pairs of nodes, subgraphs, or entire graph structures. CardiGraphormer's potential applications in drug discovery and drug interactions are vast, from identifying new drug targets to predicting drug-to-drug interactions and enabling novel drug discovery. This innovative approach provides an AI-enhanced methodology in drug development, utilizing SSL combined with GNNs to overcome existing limitations and pave the way for a richer exploration of the vast combinatorial chemical space in drug discovery.

{{</citation>}}


### (25/94) GA-DRL: Graph Neural Network-Augmented Deep Reinforcement Learning for DAG Task Scheduling over Dynamic Vehicular Clouds (Zhang Liu et al., 2023)

{{<citation>}}

Zhang Liu, Lianfen Huang, Zhibin Gao, Manman Luo, Seyyedali Hosseinalipour, Huaiyu Dai. (2023)  
**GA-DRL: Graph Neural Network-Augmented Deep Reinforcement Learning for DAG Task Scheduling over Dynamic Vehicular Clouds**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Graph Neural Network, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.00777v1)  

---


**ABSTRACT**  
Vehicular clouds (VCs) are modern platforms for processing of computation-intensive tasks over vehicles. Such tasks are often represented as directed acyclic graphs (DAGs) consisting of interdependent vertices/subtasks and directed edges. In this paper, we propose a graph neural network-augmented deep reinforcement learning scheme (GA-DRL) for scheduling DAG tasks over dynamic VCs. In doing so, we first model the VC-assisted DAG task scheduling as a Markov decision process. We then adopt a multi-head graph attention network (GAT) to extract the features of DAG subtasks. Our developed GAT enables a two-way aggregation of the topological information in a DAG task by simultaneously considering predecessors and successors of each subtask. We further introduce non-uniform DAG neighborhood sampling through codifying the scheduling priority of different subtasks, which makes our developed GAT generalizable to completely unseen DAG task topologies. Finally, we augment GAT into a double deep Q-network learning module to conduct subtask-to-vehicle assignment according to the extracted features of subtasks, while considering the dynamics and heterogeneity of the vehicles in VCs. Through simulating various DAG tasks under real-world movement traces of vehicles, we demonstrate that GA-DRL outperforms existing benchmarks in terms of DAG task completion time.

{{</citation>}}


### (26/94) Graph-level Anomaly Detection via Hierarchical Memory Networks (Chaoxi Niu et al., 2023)

{{<citation>}}

Chaoxi Niu, Guansong Pang, Ling Chen. (2023)  
**Graph-level Anomaly Detection via Hierarchical Memory Networks**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2307.00755v1)  

---


**ABSTRACT**  
Graph-level anomaly detection aims to identify abnormal graphs that exhibit deviant structures and node attributes compared to the majority in a graph set. One primary challenge is to learn normal patterns manifested in both fine-grained and holistic views of graphs for identifying graphs that are abnormal in part or in whole. To tackle this challenge, we propose a novel approach called Hierarchical Memory Networks (HimNet), which learns hierarchical memory modules -- node and graph memory modules -- via a graph autoencoder network architecture. The node-level memory module is trained to model fine-grained, internal graph interactions among nodes for detecting locally abnormal graphs, while the graph-level memory module is dedicated to the learning of holistic normal patterns for detecting globally abnormal graphs. The two modules are jointly optimized to detect both locally- and globally-anomalous graphs. Extensive empirical results on 16 real-world graph datasets from various domains show that i) HimNet significantly outperforms the state-of-art methods and ii) it is robust to anomaly contamination. Codes are available at: https://github.com/Niuchx/HimNet.

{{</citation>}}


### (27/94) ImDiffusion: Imputed Diffusion Models for Multivariate Time Series Anomaly Detection (Yuhang Chen et al., 2023)

{{<citation>}}

Yuhang Chen, Chaoyun Zhang, Minghua Ma, Yudong Liu, Ruomeng Ding, Bowen Li, Shilin He, Saravan Rajmohan, Qingwei Lin, Dongmei Zhang. (2023)  
**ImDiffusion: Imputed Diffusion Models for Multivariate Time Series Anomaly Detection**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Anomaly Detection, Microsoft, Time Series  
[Paper Link](http://arxiv.org/abs/2307.00754v1)  

---


**ABSTRACT**  
Anomaly detection in multivariate time series data is of paramount importance for ensuring the efficient operation of large-scale systems across diverse domains. However, accurately detecting anomalies in such data poses significant challenges. Existing approaches, including forecasting and reconstruction-based methods, struggle to address these challenges effectively. To overcome these limitations, we propose a novel anomaly detection framework named ImDiffusion, which combines time series imputation and diffusion models to achieve accurate and robust anomaly detection. The imputation-based approach employed by ImDiffusion leverages the information from neighboring values in the time series, enabling precise modeling of temporal and inter-correlated dependencies, reducing uncertainty in the data, thereby enhancing the robustness of the anomaly detection process. ImDiffusion further leverages diffusion models as time series imputers to accurately capturing complex dependencies. We leverage the step-by-step denoised outputs generated during the inference process to serve as valuable signals for anomaly prediction, resulting in improved accuracy and robustness of the detection process.   We evaluate the performance of ImDiffusion via extensive experiments on benchmark datasets. The results demonstrate that our proposed framework significantly outperforms state-of-the-art approaches in terms of detection accuracy and timeliness. ImDiffusion is further integrated into the real production system in Microsoft and observe a remarkable 11.4% increase in detection F1 score compared to the legacy approach. To the best of our knowledge, ImDiffusion represents a pioneering approach that combines imputation-based techniques with time series anomaly detection, while introducing the novel use of diffusion models to the field.

{{</citation>}}


### (28/94) Population Age Group Sensitivity for COVID-19 Infections with Deep Learning (Md Khairul Islam et al., 2023)

{{<citation>}}

Md Khairul Islam, Tyler Valentine, Royal Wang, Levi Davis, Matt Manner, Judy Fox. (2023)  
**Population Age Group Sensitivity for COVID-19 Infections with Deep Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, q-bio-PE  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.00751v1)  

---


**ABSTRACT**  
The COVID-19 pandemic has created unprecedented challenges for governments and healthcare systems worldwide, highlighting the critical importance of understanding the factors that contribute to virus transmission. This study aimed to identify the most influential age groups in COVID-19 infection rates at the US county level using the Modified Morris Method and deep learning for time series. Our approach involved training the state-of-the-art time-series model Temporal Fusion Transformer on different age groups as a static feature and the population vaccination status as the dynamic feature. We analyzed the impact of those age groups on COVID-19 infection rates by perturbing individual input features and ranked them based on their Morris sensitivity scores, which quantify their contribution to COVID-19 transmission rates. The findings are verified using ground truth data from the CDC and US Census, which provide the true infection rates for each age group. The results suggest that young adults were the most influential age group in COVID-19 transmission at the county level between March 1, 2020, and November 27, 2021. Using these results can inform public health policies and interventions, such as targeted vaccination strategies, to better control the spread of the virus. Our approach demonstrates the utility of feature sensitivity analysis in identifying critical factors contributing to COVID-19 transmission and can be applied in other public health domains.

{{</citation>}}


### (29/94) ESGCN: Edge Squeeze Attention Graph Convolutional Network for Traffic Flow Forecasting (Sangrok Lee et al., 2023)

{{<citation>}}

Sangrok Lee, Ha Young Kim. (2023)  
**ESGCN: Edge Squeeze Attention Graph Convolutional Network for Traffic Flow Forecasting**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Attention, Graph Convolutional Network  
[Paper Link](http://arxiv.org/abs/2307.01227v2)  

---


**ABSTRACT**  
Traffic forecasting is a highly challenging task owing to the dynamical spatio-temporal dependencies of traffic flows. To handle this, we focus on modeling the spatio-temporal dynamics and propose a network termed Edge Squeeze Graph Convolutional Network (ESGCN) to forecast traffic flow in multiple regions. ESGCN consists of two modules: W-module and ES module. W-module is a fully node-wise convolutional network. It encodes the time-series of each traffic region separately and decomposes the time-series at various scales to capture fine and coarse features. The ES module models the spatio-temporal dynamics using Graph Convolutional Network (GCN) and generates an Adaptive Adjacency Matrix (AAM) with temporal features. To improve the accuracy of AAM, we introduce three key concepts. 1) Using edge features to directly capture the spatiotemporal flow representation among regions. 2) Applying an edge attention mechanism to GCN to extract the AAM from the edge features. Here, the attention mechanism can effectively determine important spatio-temporal adjacency relations. 3) Proposing a novel node contrastive loss to suppress obstructed connections and emphasize related connections. Experimental results show that ESGCN achieves state-of-the-art performance by a large margin on four real-world datasets (PEMS03, 04, 07, and 08) with a low computational cost.

{{</citation>}}


### (30/94) vONTSS: vMF based semi-supervised neural topic modeling with optimal transport (Weijie Xu et al., 2023)

{{<citation>}}

Weijie Xu, Xiaoyu Jiang, Srinivasan H. Sengamedu, Francis Iannacci, Jinjin Zhao. (2023)  
**vONTSS: vMF based semi-supervised neural topic modeling with optimal transport**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-IT, cs-LG, cs.LG, math-IT  
Keywords: Topic Model  
[Paper Link](http://arxiv.org/abs/2307.01226v1)  

---


**ABSTRACT**  
Recently, Neural Topic Models (NTM), inspired by variational autoencoders, have attracted a lot of research interest; however, these methods have limited applications in the real world due to the challenge of incorporating human knowledge. This work presents a semi-supervised neural topic modeling method, vONTSS, which uses von Mises-Fisher (vMF) based variational autoencoders and optimal transport. When a few keywords per topic are provided, vONTSS in the semi-supervised setting generates potential topics and optimizes topic-keyword quality and topic classification. Experiments show that vONTSS outperforms existing semi-supervised topic modeling methods in classification accuracy and diversity. vONTSS also supports unsupervised topic modeling. Quantitative and qualitative experiments show that vONTSS in the unsupervised setting outperforms recent NTMs on multiple aspects: vONTSS discovers highly clustered and coherent topics on benchmark datasets. It is also much faster than the state-of-the-art weakly supervised text classification method while achieving similar classification performance. We further prove the equivalence of optimal transport loss and cross-entropy loss at the global minimum.

{{</citation>}}


## cs.CL (21)



### (31/94) ALBERTI, a Multilingual Domain Specific Language Model for Poetry Analysis (Javier de la Rosa et al., 2023)

{{<citation>}}

Javier de la Rosa, Álvaro Pérez Pozo, Salvador Ros, Elena González-Blanco. (2023)  
**ALBERTI, a Multilingual Domain Specific Language Model for Poetry Analysis**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, Language Model, Multilingual  
[Paper Link](http://arxiv.org/abs/2307.01387v1)  

---


**ABSTRACT**  
The computational analysis of poetry is limited by the scarcity of tools to automatically analyze and scan poems. In a multilingual settings, the problem is exacerbated as scansion and rhyme systems only exist for individual languages, making comparative studies very challenging and time consuming. In this work, we present \textsc{Alberti}, the first multilingual pre-trained large language model for poetry. Through domain-specific pre-training (DSP), we further trained multilingual BERT on a corpus of over 12 million verses from 12 languages. We evaluated its performance on two structural poetry tasks: Spanish stanza type classification, and metrical pattern prediction for Spanish, English and German. In both cases, \textsc{Alberti} outperforms multilingual BERT and other transformers-based models of similar sizes, and even achieves state-of-the-art results for German when compared to rule-based systems, demonstrating the feasibility and effectiveness of DSP in the poetry domain.

{{</citation>}}


### (32/94) Implicit Memory Transformer for Computationally Efficient Simultaneous Speech Translation (Matthew Raffel et al., 2023)

{{<citation>}}

Matthew Raffel, Lizhong Chen. (2023)  
**Implicit Memory Transformer for Computationally Efficient Simultaneous Speech Translation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.01381v1)  

---


**ABSTRACT**  
Simultaneous speech translation is an essential communication task difficult for humans whereby a translation is generated concurrently with oncoming speech inputs. For such a streaming task, transformers using block processing to break an input sequence into segments have achieved state-of-the-art performance at a reduced cost. Current methods to allow information to propagate across segments, including left context and memory banks, have faltered as they are both insufficient representations and unnecessarily expensive to compute. In this paper, we propose an Implicit Memory Transformer that implicitly retains memory through a new left context method, removing the need to explicitly represent memory with memory banks. We generate the left context from the attention output of the previous segment and include it in the keys and values of the current segment's attention calculation. Experiments on the MuST-C dataset show that the Implicit Memory Transformer provides a substantial speedup on the encoder forward pass with nearly identical translation quality when compared with the state-of-the-art approach that employs both left context and memory banks.

{{</citation>}}


### (33/94) Shifting Attention to Relevance: Towards the Uncertainty Estimation of Large Language Models (Jinhao Duan et al., 2023)

{{<citation>}}

Jinhao Duan, Hao Cheng, Shiqi Wang, Chenan Wang, Alex Zavalny, Renjing Xu, Bhavya Kailkhura, Kaidi Xu. (2023)  
**Shifting Attention to Relevance: Towards the Uncertainty Estimation of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: AI, Attention, LLaMA, Language Model, Natural Language Generation  
[Paper Link](http://arxiv.org/abs/2307.01379v1)  

---


**ABSTRACT**  
Although Large Language Models (LLMs) have shown great potential in Natural Language Generation, it is still challenging to characterize the uncertainty of model generations, i.e., when users could trust model outputs. Our research is derived from the heuristic facts that tokens are created unequally in reflecting the meaning of generations by auto-regressive LLMs, i.e., some tokens are more relevant (or representative) than others, yet all the tokens are equally valued when estimating uncertainty. It is because of the linguistic redundancy where mostly a few keywords are sufficient to convey the meaning of a long sentence. We name these inequalities as generative inequalities and investigate how they affect uncertainty estimation. Our results reveal that considerable tokens and sentences containing limited semantics are weighted equally or even heavily when estimating uncertainty. To tackle these biases posed by generative inequalities, we propose to jointly Shifting Attention to more Relevant (SAR) components from both the token level and the sentence level while estimating uncertainty. We conduct experiments over popular "off-the-shelf" LLMs (e.g., OPT, LLaMA) with model sizes up to 30B and powerful commercial LLMs (e.g., Davinci from OpenAI), across various free-form question-answering tasks. Experimental results and detailed demographic analysis indicate the superior performance of SAR. Code is available at https://github.com/jinhaoduan/shifting-attention-to-relevance.

{{</citation>}}


### (34/94) Shiftable Context: Addressing Training-Inference Context Mismatch in Simultaneous Speech Translation (Matthew Raffel et al., 2023)

{{<citation>}}

Matthew Raffel, Drew Penney, Lizhong Chen. (2023)  
**Shiftable Context: Addressing Training-Inference Context Mismatch in Simultaneous Speech Translation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: BLEU, Transformer  
[Paper Link](http://arxiv.org/abs/2307.01377v1)  

---


**ABSTRACT**  
Transformer models using segment-based processing have been an effective architecture for simultaneous speech translation. However, such models create a context mismatch between training and inference environments, hindering potential translation accuracy. We solve this issue by proposing Shiftable Context, a simple yet effective scheme to ensure that consistent segment and context sizes are maintained throughout training and inference, even with the presence of partially filled segments due to the streaming nature of simultaneous translation. Shiftable Context is also broadly applicable to segment-based transformers for streaming tasks. Our experiments on the English-German, English-French, and English-Spanish language pairs from the MUST-C dataset demonstrate that when applied to the Augmented Memory Transformer, a state-of-the-art model for simultaneous speech translation, the proposed scheme achieves an average increase of 2.09, 1.83, and 1.95 BLEU scores across each wait-k value for the three language pairs, respectively, with a minimal impact on computation-aware Average Lagging.

{{</citation>}}


### (35/94) Multilingual Language Models are not Multicultural: A Case Study in Emotion (Shreya Havaldar et al., 2023)

{{<citation>}}

Shreya Havaldar, Sunny Rai, Bhumika Singhal, Langchen Liu, Sharath Chandra Guntuku, Lyle Ungar. (2023)  
**Multilingual Language Models are not Multicultural: A Case Study in Emotion**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, ChatGPT, GPT, Language Model, Multilingual  
[Paper Link](http://arxiv.org/abs/2307.01370v2)  

---


**ABSTRACT**  
Emotions are experienced and expressed differently across the world. In order to use Large Language Models (LMs) for multilingual tasks that require emotional sensitivity, LMs must reflect this cultural variation in emotion. In this study, we investigate whether the widely-used multilingual LMs in 2023 reflect differences in emotional expressions across cultures and languages. We find that embeddings obtained from LMs (e.g., XLM-RoBERTa) are Anglocentric, and generative LMs (e.g., ChatGPT) reflect Western norms, even when responding to prompts in other languages. Our results show that multilingual LMs do not successfully learn the culturally appropriate nuances of emotion and we highlight possible research directions towards correcting this.

{{</citation>}}


### (36/94) Semantic enrichment towards efficient speech representations (Gaëlle Laperrière et al., 2023)

{{<citation>}}

Gaëlle Laperrière, Ha Nguyen, Sahar Ghannay, Bassam Jabaian, Yannick Estève. (2023)  
**Semantic enrichment towards efficient speech representations**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: Spoken Language Understanding  
[Paper Link](http://arxiv.org/abs/2307.01323v1)  

---


**ABSTRACT**  
Over the past few years, self-supervised learned speech representations have emerged as fruitful replacements for conventional surface representations when solving Spoken Language Understanding (SLU) tasks. Simultaneously, multilingual models trained on massive textual data were introduced to encode language agnostic semantics. Recently, the SAMU-XLSR approach introduced a way to make profit from such textual models to enrich multilingual speech representations with language agnostic semantics. By aiming for better semantic extraction on a challenging Spoken Language Understanding task and in consideration with computation costs, this study investigates a specific in-domain semantic enrichment of the SAMU-XLSR model by specializing it on a small amount of transcribed data from the downstream task. In addition, we show the benefits of the use of same-domain French and Italian benchmarks for low-resource language portability and explore cross-domain capacities of the enriched SAMU-XLSR.

{{</citation>}}


### (37/94) Exploring Spoken Named Entity Recognition: A Cross-Lingual Perspective (Moncef Benaicha et al., 2023)

{{<citation>}}

Moncef Benaicha, David Thulke, M. A. Tuğtekin Turan. (2023)  
**Exploring Spoken Named Entity Recognition: A Cross-Lingual Perspective**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NER, Named Entity Recognition  
[Paper Link](http://arxiv.org/abs/2307.01310v1)  

---


**ABSTRACT**  
Recent advancements in Named Entity Recognition (NER) have significantly improved the identification of entities in textual data. However, spoken NER, a specialized field of spoken document retrieval, lags behind due to its limited research and scarce datasets. Moreover, cross-lingual transfer learning in spoken NER has remained unexplored. This paper utilizes transfer learning across Dutch, English, and German using pipeline and End-to-End (E2E) schemes. We employ Wav2Vec2-XLS-R models on custom pseudo-annotated datasets and investigate several architectures for the adaptability of cross-lingual systems. Our results demonstrate that End-to-End spoken NER outperforms pipeline-based alternatives over our limited annotations. Notably, transfer learning from German to Dutch surpasses the Dutch E2E system by 7% and the Dutch pipeline system by 4%. This study not only underscores the feasibility of transfer learning in spoken NER but also sets promising outcomes for future evaluations, hinting at the need for comprehensive data collection to augment the results.

{{</citation>}}


### (38/94) Trainable Transformer in Transformer (Abhishek Panigrahi et al., 2023)

{{<citation>}}

Abhishek Panigrahi, Sadhika Malladi, Mengzhou Xia, Sanjeev Arora. (2023)  
**Trainable Transformer in Transformer**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.01189v1)  

---


**ABSTRACT**  
Recent works attribute the capability of in-context learning (ICL) in large pre-trained language models to implicitly simulating and fine-tuning an internal model (e.g., linear or 2-layer MLP) during inference. However, such constructions require large memory overhead, which makes simulation of more sophisticated internal models intractable. In this work, we propose an efficient construction, Transformer in Transformer (in short, TinT), that allows a transformer to simulate and fine-tune complex models internally during inference (e.g., pre-trained language models). In particular, we introduce innovative approximation techniques that allow a TinT model with less than 2 billion parameters to simulate and fine-tune a 125 million parameter transformer model within a single forward pass. TinT accommodates many common transformer variants and its design ideas also improve the efficiency of past instantiations of simple models inside transformers. We conduct end-to-end experiments to validate the internal fine-tuning procedure of TinT on various language modeling and downstream tasks. For example, even with a limited one-step budget, we observe TinT for a OPT-125M model improves performance by 4-16% absolute on average compared to OPT-125M. These findings suggest that large pre-trained language models are capable of performing intricate subroutines. To facilitate further work, a modular and extensible codebase for TinT is included.

{{</citation>}}


### (39/94) Improving Language Plasticity via Pretraining with Active Forgetting (Yihong Chen et al., 2023)

{{<citation>}}

Yihong Chen, Kelly Marchisio, Roberta Raileanu, David Ifeoluwa Adelani, Pontus Stenetorp, Sebastian Riedel, Mikel Artetxe. (2023)  
**Improving Language Plasticity via Pretraining with Active Forgetting**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs-NE, cs.CL  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2307.01163v2)  

---


**ABSTRACT**  
Pretrained language models (PLMs) are today the primary model for natural language processing. Despite their impressive downstream performance, it can be difficult to apply PLMs to new languages, a barrier to making their capabilities universally accessible. While prior work has shown it possible to address this issue by learning a new embedding layer for the new language, doing so is both data and compute inefficient. We propose to use an active forgetting mechanism during pretraining, as a simple way of creating PLMs that can quickly adapt to new languages. Concretely, by resetting the embedding layer every K updates during pretraining, we encourage the PLM to improve its ability of learning new embeddings within a limited number of updates, similar to a meta-learning effect. Experiments with RoBERTa show that models pretrained with our forgetting mechanism not only demonstrate faster convergence during language adaptation but also outperform standard ones in a low-data regime, particularly for languages that are distant from English.

{{</citation>}}


### (40/94) Exploring the In-context Learning Ability of Large Language Model for Biomedical Concept Linking (Qinyong Wang et al., 2023)

{{<citation>}}

Qinyong Wang, Zhenxiang Gao, Rong Xu. (2023)  
**Exploring the In-context Learning Ability of Large Language Model for Biomedical Concept Linking**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.01137v1)  

---


**ABSTRACT**  
The biomedical field relies heavily on concept linking in various areas such as literature mining, graph alignment, information retrieval, question-answering, data, and knowledge integration. Although large language models (LLMs) have made significant strides in many natural language processing tasks, their effectiveness in biomedical concept mapping is yet to be fully explored. This research investigates a method that exploits the in-context learning (ICL) capabilities of large models for biomedical concept linking. The proposed approach adopts a two-stage retrieve-and-rank framework. Initially, biomedical concepts are embedded using language models, and then embedding similarity is utilized to retrieve the top candidates. These candidates' contextual information is subsequently incorporated into the prompt and processed by a large language model to re-rank the concepts. This approach achieved an accuracy of 90.% in BC5CDR disease entity normalization and 94.7% in chemical entity normalization, exhibiting a competitive performance relative to supervised learning methods. Further, it showed a significant improvement, with an over 20-point absolute increase in F1 score on an oncology matching dataset. Extensive qualitative assessments were conducted, and the benefits and potential shortcomings of using large language models within the biomedical domain were discussed. were discussed.

{{</citation>}}


### (41/94) Iterative Zero-Shot LLM Prompting for Knowledge Graph Construction (Salvatore Carta et al., 2023)

{{<citation>}}

Salvatore Carta, Alessandro Giuliani, Leonardo Piano, Alessandro Sebastian Podda, Livio Pompianu, Sandro Gabriele Tiddia. (2023)  
**Iterative Zero-Shot LLM Prompting for Knowledge Graph Construction**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-3.5, Knowledge Graph, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2307.01128v1)  

---


**ABSTRACT**  
In the current digitalization era, capturing and effectively representing knowledge is crucial in most real-world scenarios. In this context, knowledge graphs represent a potent tool for retrieving and organizing a vast amount of information in a properly interconnected and interpretable structure. However, their generation is still challenging and often requires considerable human effort and domain expertise, hampering the scalability and flexibility across different application fields. This paper proposes an innovative knowledge graph generation approach that leverages the potential of the latest generative large language models, such as GPT-3.5, that can address all the main critical issues in knowledge graph building. The approach is conveyed in a pipeline that comprises novel iterative zero-shot and external knowledge-agnostic strategies in the main stages of the generation process. Our unique manifold approach may encompass significant benefits to the scientific community. In particular, the main contribution can be summarized by: (i) an innovative strategy for iteratively prompting large language models to extract relevant components of the final graph; (ii) a zero-shot strategy for each prompt, meaning that there is no need for providing examples for "guiding" the prompt result; (iii) a scalable solution, as the adoption of LLMs avoids the need for any external resources or human expertise. To assess the effectiveness of our proposed model, we performed experiments on a dataset that covered a specific domain. We claim that our proposal is a suitable solution for scalable and versatile knowledge graph construction and may be applied to different and novel contexts.

{{</citation>}}


### (42/94) Estimating Post-OCR Denoising Complexity on Numerical Texts (Arthur Hemmer et al., 2023)

{{<citation>}}

Arthur Hemmer, Jérôme Brachat, Mickaël Coustaty, Jean-Marc Ogier. (2023)  
**Estimating Post-OCR Denoising Complexity on Numerical Texts**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: OCR  
[Paper Link](http://arxiv.org/abs/2307.01020v1)  

---


**ABSTRACT**  
Post-OCR processing has significantly improved over the past few years. However, these have been primarily beneficial for texts consisting of natural, alphabetical words, as opposed to documents of numerical nature such as invoices, payslips, medical certificates, etc. To evaluate the OCR post-processing difficulty of these datasets, we propose a method to estimate the denoising complexity of a text and evaluate it on several datasets of varying nature, and show that texts of numerical nature have a significant disadvantage. We evaluate the estimated complexity ranking with respect to the error rates of modern-day denoising approaches to show the validity of our estimator.

{{</citation>}}


### (43/94) Challenges in Domain-Specific Abstractive Summarization and How to Overcome them (Anum Afzal et al., 2023)

{{<citation>}}

Anum Afzal, Juraj Vladika, Daniel Braun, Florian Matthes. (2023)  
**Challenges in Domain-Specific Abstractive Summarization and How to Overcome them**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, Natural Language Processing, Summarization  
[Paper Link](http://arxiv.org/abs/2307.00963v1)  

---


**ABSTRACT**  
Large Language Models work quite well with general-purpose data and many tasks in Natural Language Processing. However, they show several limitations when used for a task such as domain-specific abstractive text summarization. This paper identifies three of those limitations as research problems in the context of abstractive text summarization: 1) Quadratic complexity of transformer-based models with respect to the input text length; 2) Model Hallucination, which is a model's ability to generate factually incorrect text; and 3) Domain Shift, which happens when the distribution of the model's training and test corpus is not the same. Along with a discussion of the open research questions, this paper also provides an assessment of existing state-of-the-art techniques relevant to domain-specific text summarization to address the research gaps.

{{</citation>}}


### (44/94) Data-Driven Information Extraction and Enrichment of Molecular Profiling Data for Cancer Cell Lines (Ellery Smith et al., 2023)

{{<citation>}}

Ellery Smith, Rahel Paloots, Dimitris Giagkos, Michael Baudis, Kurt Stockinger. (2023)  
**Data-Driven Information Extraction and Enrichment of Molecular Profiling Data for Cancer Cell Lines**  

---
Primary Category: cs.CL  
Categories: cs-CE, cs-CL, cs-DB, cs.CL  
Keywords: Information Extraction  
[Paper Link](http://arxiv.org/abs/2307.00933v1)  

---


**ABSTRACT**  
With the proliferation of research means and computational methodologies, published biomedical literature is growing exponentially in numbers and volume. As a consequence, in the fields of biological, medical and clinical research, domain experts have to sift through massive amounts of scientific text to find relevant information. However, this process is extremely tedious and slow to be performed by humans. Hence, novel computational information extraction and correlation mechanisms are required to boost meaningful knowledge extraction. In this work, we present the design, implementation and application of a novel data extraction and exploration system. This system extracts deep semantic relations between textual entities from scientific literature to enrich existing structured clinical data in the domain of cancer cell lines. We introduce a new public data exploration portal, which enables automatic linking of genomic copy number variants plots with ranked, related entities such as affected genes. Each relation is accompanied by literature-derived evidences, allowing for deep, yet rapid, literature search, using existing structured data as a springboard. Our system is publicly available on the web at https://cancercelllines.org

{{</citation>}}


### (45/94) Automatic Design of Semantic Similarity Ensembles Using Grammatical Evolution (Jorge Martinez-Gil, 2023)

{{<citation>}}

Jorge Martinez-Gil. (2023)  
**Automatic Design of Semantic Similarity Ensembles Using Grammatical Evolution**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Semantic Similarity  
[Paper Link](http://arxiv.org/abs/2307.00925v1)  

---


**ABSTRACT**  
Semantic similarity measures are widely used in natural language processing to catalyze various computer-related tasks. However, no single semantic similarity measure is the most appropriate for all tasks, and researchers often use ensemble strategies to ensure performance. This research work proposes a method for automatically designing semantic similarity ensembles. In fact, our proposed method uses grammatical evolution, for the first time, to automatically select and aggregate measures from a pool of candidates to create an ensemble that maximizes correlation to human judgment. The method is evaluated on several benchmark datasets and compared to state-of-the-art ensembles, showing that it can significantly improve similarity assessment accuracy and outperform existing methods in some cases. As a result, our research demonstrates the potential of using grammatical evolution to automatically compare text and prove the benefits of using ensembles for semantic similarity tasks.

{{</citation>}}


### (46/94) Node-weighted Graph Convolutional Network for Depression Detection in Transcribed Clinical Interviews (Sergio Burdisso et al., 2023)

{{<citation>}}

Sergio Burdisso, Esaú Villatoro-Tello, Srikanth Madikeri, Petr Motlicek. (2023)  
**Node-weighted Graph Convolutional Network for Depression Detection in Transcribed Clinical Interviews**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Clinical, Graph Convolutional Network  
[Paper Link](http://arxiv.org/abs/2307.00920v1)  

---


**ABSTRACT**  
We propose a simple approach for weighting self-connecting edges in a Graph Convolutional Network (GCN) and show its impact on depression detection from transcribed clinical interviews. To this end, we use a GCN for modeling non-consecutive and long-distance semantics to classify the transcriptions into depressed or control subjects. The proposed method aims to mitigate the limiting assumptions of locality and the equal importance of self-connections vs. edges to neighboring nodes in GCNs, while preserving attractive features such as low computational cost, data agnostic, and interpretability capabilities. We perform an exhaustive evaluation in two benchmark datasets. Results show that our approach consistently outperforms the vanilla GCN model as well as previously reported results, achieving an F1=0.84% on both datasets. Finally, a qualitative analysis illustrates the interpretability capabilities of the proposed approach and its alignment with previous findings in psychology.

{{</citation>}}


### (47/94) Large Language and Text-to-3D Models for Engineering Design Optimization (Thiago Rios et al., 2023)

{{<citation>}}

Thiago Rios, Stefan Menzel, Bernhard Sendhoff. (2023)  
**Large Language and Text-to-3D Models for Engineering Design Optimization**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs-NE, cs.CL  
Keywords: AI, GPT  
[Paper Link](http://arxiv.org/abs/2307.01230v1)  

---


**ABSTRACT**  
The current advances in generative AI for learning large neural network models with the capability to produce essays, images, music and even 3D assets from text prompts create opportunities for a manifold of disciplines. In the present paper, we study the potential of deep text-to-3D models in the engineering domain, with focus on the chances and challenges when integrating and interacting with 3D assets in computational simulation-based design optimization. In contrast to traditional design optimization of 3D geometries that often searches for the optimum designs using numerical representations, such as B-Spline surface or deformation parameters in vehicle aerodynamic optimization, natural language challenges the optimization framework by requiring a different interpretation of variation operators while at the same time may ease and motivate the human user interaction. Here, we propose and realize a fully automated evolutionary design optimization framework using Shap-E, a recently published text-to-3D asset network by OpenAI, in the context of aerodynamic vehicle optimization. For representing text prompts in the evolutionary optimization, we evaluate (a) a bag-of-words approach based on prompt templates and Wordnet samples, and (b) a tokenisation approach based on prompt templates and the byte pair encoding method from GPT4. Our main findings from the optimizations indicate that, first, it is important to ensure that the designs generated from prompts are within the object class of application, i.e. diverse and novel designs need to be realistic, and, second, that more research is required to develop methods where the strength of text prompt variations and the resulting variations of the 3D designs share causal relations to some degree to improve the optimization.

{{</citation>}}


### (48/94) Evaluating Shutdown Avoidance of Language Models in Textual Scenarios (Teun van der Weij et al., 2023)

{{<citation>}}

Teun van der Weij, Simon Lermen, Leon lang. (2023)  
**Evaluating Shutdown Avoidance of Language Models in Textual Scenarios**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2307.00787v1)  

---


**ABSTRACT**  
Recently, there has been an increase in interest in evaluating large language models for emergent and dangerous capabilities. Importantly, agents could reason that in some scenarios their goal is better achieved if they are not turned off, which can lead to undesirable behaviors. In this paper, we investigate the potential of using toy textual scenarios to evaluate instrumental reasoning and shutdown avoidance in language models such as GPT-4 and Claude. Furthermore, we explore whether shutdown avoidance is merely a result of simple pattern matching between the dataset and the prompt or if it is a consistent behaviour across different environments and variations.   We evaluated behaviours manually and also experimented with using language models for automatic evaluations, and these evaluations demonstrate that simple pattern matching is likely not the sole contributing factor for shutdown avoidance. This study provides insights into the behaviour of language models in shutdown avoidance scenarios and inspires further research on the use of textual scenarios for evaluations.

{{</citation>}}


### (49/94) CollabKG: A Learnable Human-Machine-Cooperative Information Extraction Toolkit for (Event) Knowledge Graph Construction (Xiang Wei et al., 2023)

{{<citation>}}

Xiang Wei, Yufeng Chen, Ning Cheng, Xingyu Cui, Jinan Xu, Wenjuan Han. (2023)  
**CollabKG: A Learnable Human-Machine-Cooperative Information Extraction Toolkit for (Event) Knowledge Graph Construction**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Information Extraction, Knowledge Graph, NER  
[Paper Link](http://arxiv.org/abs/2307.00769v1)  

---


**ABSTRACT**  
In order to construct or extend entity-centric and event-centric knowledge graphs (KG and EKG), the information extraction (IE) annotation toolkit is essential. However, existing IE toolkits have several non-trivial problems, such as not supporting multi-tasks, not supporting automatic updates. In this work, we present CollabKG, a learnable human-machine-cooperative IE toolkit for KG and EKG construction. Specifically, for the multi-task issue, CollabKG unifies different IE subtasks, including named entity recognition (NER), entity-relation triple extraction (RE), and event extraction (EE), and supports both KG and EKG. Then, combining advanced prompting-based IE technology, the human-machine-cooperation mechanism with LLMs as the assistant machine is presented which can provide a lower cost as well as a higher performance. Lastly, owing to the two-way interaction between the human and machine, CollabKG with learning ability allows self-renewal. Besides, CollabKG has several appealing features (e.g., customization, training-free, propagation, etc.) that make the system powerful, easy-to-use, and high-productivity. We holistically compare our toolkit with other existing tools on these features. Human evaluation quantitatively illustrates that CollabKG significantly improves annotation quality, efficiency, and stability simultaneously.

{{</citation>}}


### (50/94) Multilingual Contextual Adapters To Improve Custom Word Recognition In Low-resource Languages (Devang Kulshreshtha et al., 2023)

{{<citation>}}

Devang Kulshreshtha, Saket Dingliwal, Brady Houston, Sravan Bodapati. (2023)  
**Multilingual Contextual Adapters To Improve Custom Word Recognition In Low-resource Languages**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: Multilingual, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2307.00759v1)  

---


**ABSTRACT**  
Connectionist Temporal Classification (CTC) models are popular for their balance between speed and performance for Automatic Speech Recognition (ASR). However, these CTC models still struggle in other areas, such as personalization towards custom words. A recent approach explores Contextual Adapters, wherein an attention-based biasing model for CTC is used to improve the recognition of custom entities. While this approach works well with enough data, we showcase that it isn't an effective strategy for low-resource languages. In this work, we propose a supervision loss for smoother training of the Contextual Adapters. Further, we explore a multilingual strategy to improve performance with limited training data. Our method achieves 48% F1 improvement in retrieving unseen custom entities for a low-resource language. Interestingly, as a by-product of training the Contextual Adapters, we see a 5-11% Word Error Rate (WER) reduction in the performance of the base CTC model as well.

{{</citation>}}


### (51/94) Interpretability and Transparency-Driven Detection and Transformation of Textual Adversarial Examples (IT-DT) (Bushra Sabir et al., 2023)

{{<citation>}}

Bushra Sabir, M. Ali Babar, Sharif Abuadbba. (2023)  
**Interpretability and Transparency-Driven Detection and Transformation of Textual Adversarial Examples (IT-DT)**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: BERT, GPT, NLP, T5, Transformer  
[Paper Link](http://arxiv.org/abs/2307.01225v1)  

---


**ABSTRACT**  
Transformer-based text classifiers like BERT, Roberta, T5, and GPT-3 have shown impressive performance in NLP. However, their vulnerability to adversarial examples poses a security risk. Existing defense methods lack interpretability, making it hard to understand adversarial classifications and identify model vulnerabilities. To address this, we propose the Interpretability and Transparency-Driven Detection and Transformation (IT-DT) framework. It focuses on interpretability and transparency in detecting and transforming textual adversarial examples. IT-DT utilizes techniques like attention maps, integrated gradients, and model feedback for interpretability during detection. This helps identify salient features and perturbed words contributing to adversarial classifications. In the transformation phase, IT-DT uses pre-trained embeddings and model feedback to generate optimal replacements for perturbed words. By finding suitable substitutions, we aim to convert adversarial examples into non-adversarial counterparts that align with the model's intended behavior while preserving the text's meaning. Transparency is emphasized through human expert involvement. Experts review and provide feedback on detection and transformation results, enhancing decision-making, especially in complex scenarios. The framework generates insights and threat intelligence empowering analysts to identify vulnerabilities and improve model robustness. Comprehensive experiments demonstrate the effectiveness of IT-DT in detecting and transforming adversarial examples. The approach enhances interpretability, provides transparency, and enables accurate identification and successful transformation of adversarial inputs. By combining technical analysis and human expertise, IT-DT significantly improves the resilience and trustworthiness of transformer-based text classifiers against adversarial attacks.

{{</citation>}}


## cs.SD (3)



### (52/94) Spatial-temporal Graph Based Multi-channel Speaker Verification With Ad-hoc Microphone Arrays (Yijiang Chen et al., 2023)

{{<citation>}}

Yijiang Chen, Chengdong Liang, Xiao-Lei Zhang. (2023)  
**Spatial-temporal Graph Based Multi-channel Speaker Verification With Ad-hoc Microphone Arrays**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Speaker Verification  
[Paper Link](http://arxiv.org/abs/2307.01386v1)  

---


**ABSTRACT**  
The performance of speaker verification degrades significantly in adverse acoustic environments with strong reverberation and noise. To address this issue, this paper proposes a spatial-temporal graph convolutional network (GCN) method for the multi-channel speaker verification with ad-hoc microphone arrays. It includes a feature aggregation block and a channel selection block, both of which are built on graphs. The feature aggregation block fuses speaker features among different time and channels by a spatial-temporal GCN. The graph-based channel selection block discards the noisy channels that may contribute negatively to the system. The proposed method is flexible in incorporating various kinds of graphs and prior knowledge. We compared the proposed method with six representative methods in both real-world and simulated environments.   Experimental results show that the proposed method achieves a relative equal error rate (EER) reduction of $\mathbf{15.39\%}$ lower than the strongest referenced method in the simulated datasets, and $\mathbf{17.70\%}$ lower than the latter in the real datasets. Moreover, its performance is robust across different signal-to-noise ratios and reverberation time.

{{</citation>}}


### (53/94) RobustL2S: Speaker-Specific Lip-to-Speech Synthesis exploiting Self-Supervised Representations (Neha Sahipjohn et al., 2023)

{{<citation>}}

Neha Sahipjohn, Neil Shah, Vishal Tambrahalli, Vineet Gandhi. (2023)  
**RobustL2S: Speaker-Specific Lip-to-Speech Synthesis exploiting Self-Supervised Representations**  

---
Primary Category: cs.SD  
Categories: cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2307.01233v1)  

---


**ABSTRACT**  
Significant progress has been made in speaker dependent Lip-to-Speech synthesis, which aims to generate speech from silent videos of talking faces. Current state-of-the-art approaches primarily employ non-autoregressive sequence-to-sequence architectures to directly predict mel-spectrograms or audio waveforms from lip representations. We hypothesize that the direct mel-prediction hampers training/model efficiency due to the entanglement of speech content with ambient information and speaker characteristics. To this end, we propose RobustL2S, a modularized framework for Lip-to-Speech synthesis. First, a non-autoregressive sequence-to-sequence model maps self-supervised visual features to a representation of disentangled speech content. A vocoder then converts the speech features into raw waveforms. Extensive evaluations confirm the effectiveness of our setup, achieving state-of-the-art performance on the unconstrained Lip2Wav dataset and the constrained GRID and TCD-TIMIT datasets. Speech samples from RobustL2S can be found at https://neha-sherin.github.io/RobustL2S/

{{</citation>}}


### (54/94) EmoGen: Eliminating Subjective Bias in Emotional Music Generation (Chenfei Kang et al., 2023)

{{<citation>}}

Chenfei Kang, Peiling Lu, Botao Yu, Xu Tan, Wei Ye, Shikun Zhang, Jiang Bian. (2023)  
**EmoGen: Eliminating Subjective Bias in Emotional Music Generation**  

---
Primary Category: cs.SD  
Categories: cs-AI, cs-LG, cs-MM, cs-SD, cs.SD, eess-AS  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2307.01229v1)  

---


**ABSTRACT**  
Music is used to convey emotions, and thus generating emotional music is important in automatic music generation. Previous work on emotional music generation directly uses annotated emotion labels as control signals, which suffers from subjective bias: different people may annotate different emotions on the same music, and one person may feel different emotions under different situations. Therefore, directly mapping emotion labels to music sequences in an end-to-end way would confuse the learning process and hinder the model from generating music with general emotions. In this paper, we propose EmoGen, an emotional music generation system that leverages a set of emotion-related music attributes as the bridge between emotion and music, and divides the generation into two stages: emotion-to-attribute mapping with supervised clustering, and attribute-to-music generation with self-supervised learning. Both stages are beneficial: in the first stage, the attribute values around the clustering center represent the general emotions of these samples, which help eliminate the impacts of the subjective bias of emotion labels; in the second stage, the generation is completely disentangled from emotion labels and thus free from the subjective bias. Both subjective and objective evaluations show that EmoGen outperforms previous methods on emotion control accuracy and music quality respectively, which demonstrate our superiority in generating emotional music. Music samples generated by EmoGen are available via this link:https://ai-muzic.github.io/emogen/, and the code is available at this link:https://github.com/microsoft/muzic/.

{{</citation>}}


## cs.IT (1)



### (55/94) Optimized Geometric Constellation Shaping for Wiener Phase Noise Channels with Viterbi-Viterbi Carrier Phase Estimation (Andrej Rode et al., 2023)

{{<citation>}}

Andrej Rode, Wintana Araya Gebrehiwot, Shrinivas Chimmalgi, Laurent Schmalen. (2023)  
**Optimized Geometric Constellation Shaping for Wiener Phase Noise Channels with Viterbi-Viterbi Carrier Phase Estimation**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, eess-SP, math-IT  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2307.01367v1)  

---


**ABSTRACT**  
The Viterbi & Viterbi (V&V) algorithm is well understood for QPSK and 16-QAM, but modifications are required for higher-order modulation formats. We present an approach to extend the standard V&V algorithm for higher-order modulation formats by modifying the transmit constellation with geometric constellation shaping.

{{</citation>}}


## cs.RO (2)



### (56/94) Towards Safe Autonomous Driving Policies using a Neuro-Symbolic Deep Reinforcement Learning Approach (Iman Sharifi et al., 2023)

{{<citation>}}

Iman Sharifi, Mustafa Yildirim, Saber Fallah. (2023)  
**Towards Safe Autonomous Driving Policies using a Neuro-Symbolic Deep Reinforcement Learning Approach**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-LG, cs-LO, cs-RO, cs-SY, cs.RO, eess-SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.01316v2)  

---


**ABSTRACT**  
The dynamic nature of driving environments and the presence of diverse road users pose significant challenges for decision-making in autonomous driving. Deep reinforcement learning (DRL) has emerged as a popular approach to tackle this problem. However, the application of existing DRL solutions is mainly confined to simulated environments due to safety concerns, impeding their deployment in real-world. To overcome this limitation, this paper introduces a novel neuro-symbolic model-free DRL approach, called DRL with Symbolic Logics (DRLSL) that combines the strengths of DRL (learning from experience) and symbolic first-order logics (knowledge-driven reasoning) to enable safe learning in real-time interactions of autonomous driving within real environments. This innovative approach provides a means to learn autonomous driving policies by actively engaging with the physical environment while ensuring safety. We have implemented the DRLSL framework in autonomous driving using the highD dataset and demonstrated that our method successfully avoids unsafe actions during both the training and testing phases. Furthermore, our results indicate that DRLSL achieves faster convergence during training and exhibits better generalizability to new driving scenarios compared to traditional DRL methods.

{{</citation>}}


### (57/94) Artifacts Mapping: Multi-Modal Semantic Mapping for Object Detection and 3D Localization (Federico Rollo et al., 2023)

{{<citation>}}

Federico Rollo, Gennaro Raiola, Andrea Zunino, Nikolaos Tsagarakis, Arash Ajoudani. (2023)  
**Artifacts Mapping: Multi-Modal Semantic Mapping for Object Detection and 3D Localization**  

---
Primary Category: cs.RO  
Categories: cs-CV, cs-RO, cs.RO  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2307.01121v1)  

---


**ABSTRACT**  
Geometric navigation is nowadays a well-established field of robotics and the research focus is shifting towards higher-level scene understanding, such as Semantic Mapping. When a robot needs to interact with its environment, it must be able to comprehend the contextual information of its surroundings. This work focuses on classifying and localising objects within a map, which is under construction (SLAM) or already built. To further explore this direction, we propose a framework that can autonomously detect and localize predefined objects in a known environment using a multi-modal sensor fusion approach (combining RGB and depth data from an RGB-D camera and a lidar). The framework consists of three key elements: understanding the environment through RGB data, estimating depth through multi-modal sensor fusion, and managing artifacts (i.e., filtering and stabilizing measurements). The experiments show that the proposed framework can accurately detect 98% of the objects in the real sample environment, without post-processing, while 85% and 80% of the objects were mapped using the single RGBD camera or RGB + lidar setup respectively. The comparison with single-sensor (camera or lidar) experiments is performed to show that sensor fusion allows the robot to accurately detect near and far obstacles, which would have been noisy or imprecise in a purely visual or laser-based approach.

{{</citation>}}


## cs.CV (18)



### (58/94) SAMAug: Point Prompt Augmentation for Segment Anything Model (Haixing Dai et al., 2023)

{{<citation>}}

Haixing Dai, Chong Ma, Zhengliang Liu, Yiwei Li, Peng Shu, Xiaozheng Wei, Lin Zhao, Zihao Wu, Dajiang Zhu, Wei Liu, Quanzheng Li, Tianming Liu, Xiang Li. (2023)  
**SAMAug: Point Prompt Augmentation for Segment Anything Model**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2307.01187v1)  

---


**ABSTRACT**  
This paper introduces SAMAug, a novel visual point augmentation method for the Segment Anything Model (SAM) that enhances interactive image segmentation performance. SAMAug generates augmented point prompts to provide more information to SAM. From the initial point prompt, SAM produces the initial mask, which is then fed into our proposed SAMAug to generate augmented point prompts. By incorporating these extra points, SAM can generate augmented segmentation masks based on the augmented point prompts and the initial prompt, resulting in improved segmentation performance. We evaluate four point augmentation techniques: random selection, maximum difference entropy, maximum distance, and a saliency model. Experiments on the COCO, Fundus, and Chest X-ray datasets demonstrate that SAMAug can boost SAM's segmentation results, especially using the maximum distance and saliency model methods. SAMAug underscores the potential of visual prompt engineering to advance interactive computer vision models.

{{</citation>}}


### (59/94) AVSegFormer: Audio-Visual Segmentation with Transformer (Shengyi Gao et al., 2023)

{{<citation>}}

Shengyi Gao, Zhe Chen, Guo Chen, Wenhai Wang, Tong Lu. (2023)  
**AVSegFormer: Audio-Visual Segmentation with Transformer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs-SD, cs.CV, eess-AS  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.01146v2)  

---


**ABSTRACT**  
The combination of audio and vision has long been a topic of interest in the multi-modal community. Recently, a new audio-visual segmentation (AVS) task has been introduced, aiming to locate and segment the sounding objects in a given video. This task demands audio-driven pixel-level scene understanding for the first time, posing significant challenges. In this paper, we propose AVSegFormer, a novel framework for AVS tasks that leverages the transformer architecture. Specifically, we introduce audio queries and learnable queries into the transformer decoder, enabling the network to selectively attend to interested visual features. Besides, we present an audio-visual mixer, which can dynamically adjust visual features by amplifying relevant and suppressing irrelevant spatial channels. Additionally, we devise an intermediate mask loss to enhance the supervision of the decoder, encouraging the network to produce more accurate intermediate predictions. Extensive experiments demonstrate that AVSegFormer achieves state-of-the-art results on the AVS benchmark. The code is available at https://github.com/vvvb-github/AVSegFormer.

{{</citation>}}


### (60/94) SCITUNE: Aligning Large Language Models with Scientific Multimodal Instructions (Sameera Horawalavithana et al., 2023)

{{<citation>}}

Sameera Horawalavithana, Sai Munikoti, Ian Stewart, Henry Kvinge. (2023)  
**SCITUNE: Aligning Large Language Models with Scientific Multimodal Instructions**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs.CV  
Keywords: LLaMA, Language Model, QA  
[Paper Link](http://arxiv.org/abs/2307.01139v1)  

---


**ABSTRACT**  
Instruction finetuning is a popular paradigm to align large language models (LLM) with human intent. Despite its popularity, this idea is less explored in improving the LLMs to align existing foundation models with scientific disciplines, concepts and goals. In this work, we present SciTune as a tuning framework to improve the ability of LLMs to follow scientific multimodal instructions. To test our methodology, we use a human-generated scientific instruction tuning dataset and train a large multimodal model LLaMA-SciTune that connects a vision encoder and LLM for science-focused visual and language understanding. In comparison to the models that are finetuned with machine generated data only, LLaMA-SciTune surpasses human performance on average and in many sub-categories on the ScienceQA benchmark.

{{</citation>}}


### (61/94) MeT: A Graph Transformer for Semantic Segmentation of 3D Meshes (Giuseppe Vecchio et al., 2023)

{{<citation>}}

Giuseppe Vecchio, Luca Prezzavento, Carmelo Pino, Francesco Rundo, Simone Palazzo, Concetto Spampinato. (2023)  
**MeT: A Graph Transformer for Semantic Segmentation of 3D Meshes**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-GR, cs.CV  
Keywords: NLP, Semantic Segmentation, Transformer  
[Paper Link](http://arxiv.org/abs/2307.01115v1)  

---


**ABSTRACT**  
Polygonal meshes have become the standard for discretely approximating 3D shapes, thanks to their efficiency and high flexibility in capturing non-uniform shapes. This non-uniformity, however, leads to irregularity in the mesh structure, making tasks like segmentation of 3D meshes particularly challenging. Semantic segmentation of 3D mesh has been typically addressed through CNN-based approaches, leading to good accuracy. Recently, transformers have gained enough momentum both in NLP and computer vision fields, achieving performance at least on par with CNN models, supporting the long-sought architecture universalism. Following this trend, we propose a transformer-based method for semantic segmentation of 3D mesh motivated by a better modeling of the graph structure of meshes, by means of global attention mechanisms. In order to address the limitations of standard transformer architectures in modeling relative positions of non-sequential data, as in the case of 3D meshes, as well as in capturing the local context, we perform positional encoding by means the Laplacian eigenvectors of the adjacency matrix, replacing the traditional sinusoidal positional encodings, and by introducing clustering-based features into the self-attention and cross-attention operators. Experimental results, carried out on three sets of the Shape COSEG Dataset, on the human segmentation dataset proposed in Maron et al., 2017 and on the ShapeNet benchmark, show how the proposed approach yields state-of-the-art performance on semantic segmentation of 3D meshes.

{{</citation>}}


### (62/94) Localized Questions in Medical Visual Question Answering (Sergio Tascon-Morales et al., 2023)

{{<citation>}}

Sergio Tascon-Morales, Pablo Márquez-Neila, Raphael Sznitman. (2023)  
**Localized Questions in Medical Visual Question Answering**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2307.01067v1)  

---


**ABSTRACT**  
Visual Question Answering (VQA) models aim to answer natural language questions about given images. Due to its ability to ask questions that differ from those used when training the model, medical VQA has received substantial attention in recent years. However, existing medical VQA models typically focus on answering questions that refer to an entire image rather than where the relevant content may be located in the image. Consequently, VQA models are limited in their interpretability power and the possibility to probe the model about specific image regions. This paper proposes a novel approach for medical VQA that addresses this limitation by developing a model that can answer questions about image regions while considering the context necessary to answer the questions. Our experimental results demonstrate the effectiveness of our proposed model, outperforming existing methods on three datasets. Our code and data are available at https://github.com/sergiotasconmorales/locvqa.

{{</citation>}}


### (63/94) CGAM: Click-Guided Attention Module for Interactive Pathology Image Segmentation via Backpropagating Refinement (Seonghui Min et al., 2023)

{{<citation>}}

Seonghui Min, Won-Ki Jeong. (2023)  
**CGAM: Click-Guided Attention Module for Interactive Pathology Image Segmentation via Backpropagating Refinement**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2307.01015v1)  

---


**ABSTRACT**  
Tumor region segmentation is an essential task for the quantitative analysis of digital pathology. Recently presented deep neural networks have shown state-of-the-art performance in various image-segmentation tasks. However, because of the unclear boundary between the cancerous and normal regions in pathology images, despite using modern methods, it is difficult to produce satisfactory segmentation results in terms of the reliability and accuracy required for medical data. In this study, we propose an interactive segmentation method that allows users to refine the output of deep neural networks through click-type user interactions. The primary method is to formulate interactive segmentation as an optimization problem that leverages both user-provided click constraints and semantic information in a feature map using a click-guided attention module (CGAM). Unlike other existing methods, CGAM avoids excessive changes in segmentation results, which can lead to the overfitting of user clicks. Another advantage of CGAM is that the model size is independent of input image size. Experimental results on pathology image datasets indicated that our method performs better than existing state-of-the-art methods.

{{</citation>}}


### (64/94) Visual Instruction Tuning with Polite Flamingo (Delong Chen et al., 2023)

{{<citation>}}

Delong Chen, Jianfeng Liu, Wenliang Dai, Baoyuan Wang. (2023)  
**Visual Instruction Tuning with Polite Flamingo**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.01003v1)  

---


**ABSTRACT**  
Recent research has demonstrated that the multi-task fine-tuning of multi-modal Large Language Models (LLMs) using an assortment of annotated downstream vision-language datasets significantly enhances their performance. Yet, during this process, a side effect, which we termed as the "multi-modal alignment tax", surfaces. This side effect negatively impacts the model's ability to format responses appropriately -- for instance, its "politeness" -- due to the overly succinct and unformatted nature of raw annotations, resulting in reduced human preference. In this paper, we introduce Polite Flamingo, a multi-modal response rewriter that transforms raw annotations into a more appealing, "polite" format. Polite Flamingo is trained to reconstruct high-quality responses from their automatically distorted counterparts and is subsequently applied to a vast array of vision-language datasets for response rewriting. After rigorous filtering, we generate the PF-1M dataset and further validate its value by fine-tuning a multi-modal LLM with it. Combined with novel methodologies including U-shaped multi-stage tuning and multi-turn augmentation, the resulting model, Clever Flamingo, demonstrates its advantages in both multi-modal understanding and response politeness according to automated and human evaluations.

{{</citation>}}


### (65/94) HODINet: High-Order Discrepant Interaction Network for RGB-D Salient Object Detection (Kang Yi et al., 2023)

{{<citation>}}

Kang Yi, Jing Xu, Xiao Jin, Fu Guo, Yan-Feng Wu. (2023)  
**HODINet: High-Order Discrepant Interaction Network for RGB-D Salient Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2307.00954v1)  

---


**ABSTRACT**  
RGB-D salient object detection (SOD) aims to detect the prominent regions by jointly modeling RGB and depth information. Most RGB-D SOD methods apply the same type of backbones and fusion modules to identically learn the multimodality and multistage features. However, these features contribute differently to the final saliency results, which raises two issues: 1) how to model discrepant characteristics of RGB images and depth maps; 2) how to fuse these cross-modality features in different stages. In this paper, we propose a high-order discrepant interaction network (HODINet) for RGB-D SOD. Concretely, we first employ transformer-based and CNN-based architectures as backbones to encode RGB and depth features, respectively. Then, the high-order representations are delicately extracted and embedded into spatial and channel attentions for cross-modality feature fusion in different stages. Specifically, we design a high-order spatial fusion (HOSF) module and a high-order channel fusion (HOCF) module to fuse features of the first two and the last two stages, respectively. Besides, a cascaded pyramid reconstruction network is adopted to progressively decode the fused features in a top-down pathway. Extensive experiments are conducted on seven widely used datasets to demonstrate the effectiveness of the proposed approach. We achieve competitive performance against 24 state-of-the-art methods under four evaluation metrics.

{{</citation>}}


### (66/94) Towards Building Self-Aware Object Detectors via Reliable Uncertainty Quantification and Calibration (Kemal Oksuz et al., 2023)

{{<citation>}}

Kemal Oksuz, Tom Joy, Puneet K. Dokania. (2023)  
**Towards Building Self-Aware Object Detectors via Reliable Uncertainty Quantification and Calibration**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2307.00934v1)  

---


**ABSTRACT**  
The current approach for testing the robustness of object detectors suffers from serious deficiencies such as improper methods of performing out-of-distribution detection and using calibration metrics which do not consider both localisation and classification quality. In this work, we address these issues, and introduce the Self-Aware Object Detection (SAOD) task, a unified testing framework which respects and adheres to the challenges that object detectors face in safety-critical environments such as autonomous driving. Specifically, the SAOD task requires an object detector to be: robust to domain shift; obtain reliable uncertainty estimates for the entire scene; and provide calibrated confidence scores for the detections. We extensively use our framework, which introduces novel metrics and large scale test datasets, to test numerous object detectors in two different use-cases, allowing us to highlight critical insights into their robustness performance. Finally, we introduce a simple baseline for the SAOD task, enabling researchers to benchmark future proposed methods and move towards robust object detectors which are fit for purpose. Code is available at https://github.com/fiveai/saod

{{</citation>}}


### (67/94) UniFine: A Unified and Fine-grained Approach for Zero-shot Vision-Language Understanding (Rui Sun et al., 2023)

{{<citation>}}

Rui Sun, Zhecan Wang, Haoxuan You, Noel Codella, Kai-Wei Chang, Shih-Fu Chang. (2023)  
**UniFine: A Unified and Fine-grained Approach for Zero-shot Vision-Language Understanding**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: NLI, QA  
[Paper Link](http://arxiv.org/abs/2307.00862v1)  

---


**ABSTRACT**  
Vision-language tasks, such as VQA, SNLI-VE, and VCR are challenging because they require the model's reasoning ability to understand the semantics of the visual world and natural language. Supervised methods working for vision-language tasks have been well-studied. However, solving these tasks in a zero-shot setting is less explored. Since Contrastive Language-Image Pre-training (CLIP) has shown remarkable zero-shot performance on image-text matching, previous works utilized its strong zero-shot ability by converting vision-language tasks into an image-text matching problem, and they mainly consider global-level matching (e.g., the whole image or sentence). However, we find visual and textual fine-grained information, e.g., keywords in the sentence and objects in the image, can be fairly informative for semantics understanding. Inspired by this, we propose a unified framework to take advantage of the fine-grained information for zero-shot vision-language learning, covering multiple tasks such as VQA, SNLI-VE, and VCR. Our experiments show that our framework outperforms former zero-shot methods on VQA and achieves substantial improvement on SNLI-VE and VCR. Furthermore, our ablation studies confirm the effectiveness and generalizability of our proposed method. Code will be available at https://github.com/ThreeSR/UniFine

{{</citation>}}


### (68/94) Review helps learn better: Temporal Supervised Knowledge Distillation (Dongwei Wang et al., 2023)

{{<citation>}}

Dongwei Wang, Zhi Han, Yanmei Wang, Xiai Chen, Baichen Liu, Yandong Tang. (2023)  
**Review helps learn better: Temporal Supervised Knowledge Distillation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Knowledge Distillation, LSTM  
[Paper Link](http://arxiv.org/abs/2307.00811v1)  

---


**ABSTRACT**  
Reviewing plays an important role when learning knowledge. The knowledge acquisition at a certain time point may be strongly inspired with the help of previous experience. Thus the knowledge growing procedure should show strong relationship along the temporal dimension. In our research, we find that during the network training, the evolution of feature map follows temporal sequence property. A proper temporal supervision may further improve the network training performance. Inspired by this observation, we design a novel knowledge distillation method. Specifically, we extract the spatiotemporal features in the different training phases of student by convolutional Long Short-term memory network (Conv-LSTM). Then, we train the student net through a dynamic target, rather than static teacher network features. This process realizes the refinement of old knowledge in student network, and utilizes them to assist current learning. Extensive experiments verify the effectiveness and advantages of our method over existing knowledge distillation methods, including various network architectures, different tasks (image classification and object detection) .

{{</citation>}}


### (69/94) SketchMetaFace: A Learning-based Sketching Interface for High-fidelity 3D Character Face Modeling (Zhongjin Luo et al., 2023)

{{<citation>}}

Zhongjin Luo, Dong Du, Heming Zhu, Yizhou Yu, Hongbo Fu, Xiaoguang Han. (2023)  
**SketchMetaFace: A Learning-based Sketching Interface for High-fidelity 3D Character Face Modeling**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-GR, cs-HC, cs.CV  
Keywords: Sketch  
[Paper Link](http://arxiv.org/abs/2307.00804v2)  

---


**ABSTRACT**  
Modeling 3D avatars benefits various application scenarios such as AR/VR, gaming, and filming. Character faces contribute significant diversity and vividity as a vital component of avatars. However, building 3D character face models usually requires a heavy workload with commercial tools, even for experienced artists. Various existing sketch-based tools fail to support amateurs in modeling diverse facial shapes and rich geometric details. In this paper, we present SketchMetaFace - a sketching system targeting amateur users to model high-fidelity 3D faces in minutes. We carefully design both the user interface and the underlying algorithm. First, curvature-aware strokes are adopted to better support the controllability of carving facial details. Second, considering the key problem of mapping a 2D sketch map to a 3D model, we develop a novel learning-based method termed "Implicit and Depth Guided Mesh Modeling" (IDGMM). It fuses the advantages of mesh, implicit, and depth representations to achieve high-quality results with high efficiency. In addition, to further support usability, we present a coarse-to-fine 2D sketching interface design and a data-driven stroke suggestion tool. User studies demonstrate the superiority of our system over existing modeling tools in terms of the ease to use and visual quality of results. Experimental analyses also show that IDGMM reaches a better trade-off between accuracy and efficiency. SketchMetaFace is available at https://zhongjinluo.github.io/SketchMetaFace/.

{{</citation>}}


### (70/94) DifFSS: Diffusion Model for Few-Shot Semantic Segmentation (Weimin Tan et al., 2023)

{{<citation>}}

Weimin Tan, Siyuan Chen, Bo Yan. (2023)  
**DifFSS: Diffusion Model for Few-Shot Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI, Few-Shot, Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2307.00773v1)  

---


**ABSTRACT**  
Diffusion models have demonstrated excellent performance in image generation. Although various few-shot semantic segmentation (FSS) models with different network structures have been proposed, performance improvement has reached a bottleneck. This paper presents the first work to leverage the diffusion model for FSS task, called DifFSS. DifFSS, a novel FSS paradigm, can further improve the performance of the state-of-the-art FSS models by a large margin without modifying their network structure. Specifically, we utilize the powerful generation ability of diffusion models to generate diverse auxiliary support images by using the semantic mask, scribble or soft HED boundary of the support image as control conditions. This generation process simulates the variety within the class of the query image, such as color, texture variation, lighting, $etc$. As a result, FSS models can refer to more diverse support images, yielding more robust representations, thereby achieving a consistent improvement in segmentation performance. Extensive experiments on three publicly available datasets based on existing advanced FSS models demonstrate the effectiveness of the diffusion model for FSS task. Furthermore, we explore in detail the impact of different input settings of the diffusion model on segmentation performance. Hopefully, this completely new paradigm will bring inspiration to the study of FSS task integrated with AI-generated content.

{{</citation>}}


### (71/94) Structured Network Pruning by Measuring Filter-wise Interactions (Wenting Tang et al., 2023)

{{<citation>}}

Wenting Tang, Xingxing Wei, Bo Li. (2023)  
**Structured Network Pruning by Measuring Filter-wise Interactions**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Pruning  
[Paper Link](http://arxiv.org/abs/2307.00758v1)  

---


**ABSTRACT**  
Structured network pruning is a practical approach to reduce computation cost directly while retaining the CNNs' generalization performance in real applications. However, identifying redundant filters is a core problem in structured network pruning, and current redundancy criteria only focus on individual filters' attributes. When pruning sparsity increases, these redundancy criteria are not effective or efficient enough. Since the filter-wise interaction also contributes to the CNN's prediction accuracy, we integrate the filter-wise interaction into the redundancy criterion. In our criterion, we introduce the filter importance and filter utilization strength to reflect the decision ability of individual and multiple filters. Utilizing this new redundancy criterion, we propose a structured network pruning approach SNPFI (Structured Network Pruning by measuring Filter-wise Interaction). During the pruning, the SNPFI can automatically assign the proper sparsity based on the filter utilization strength and eliminate the useless filters by filter importance. After the pruning, the SNPFI can recover pruned model's performance effectively without iterative training by minimizing the interaction difference. We empirically demonstrate the effectiveness of the SNPFI with several commonly used CNN models, including AlexNet, MobileNetv1, and ResNet-50, on various image classification datasets, including MNIST, CIFAR-10, and ImageNet. For all experimental CNN models, nearly 60% of computation is reduced in a network compression while the classification accuracy remains.

{{</citation>}}


### (72/94) Feasibility of Universal Anomaly Detection without Knowing the Abnormality in Medical Images (Can Cui et al., 2023)

{{<citation>}}

Can Cui, Yaohong Wang, Shunxing Bao, Yucheng Tang, Ruining Deng, Lucas W. Remedios, Zuhayr Asad, Joseph T. Roland, Ken S. Lau, Qi Liu, Lori A. Coburn, Keith T. Wilson, Bennett A. Landman, Yuankai Huo. (2023)  
**Feasibility of Universal Anomaly Detection without Knowing the Abnormality in Medical Images**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2307.00750v1)  

---


**ABSTRACT**  
Many anomaly detection approaches, especially deep learning methods, have been recently developed to identify abnormal image morphology by only employing normal images during training. Unfortunately, many prior anomaly detection methods were optimized for a specific "known" abnormality (e.g., brain tumor, bone fraction, cell types). Moreover, even though only the normal images were used in the training process, the abnormal images were oftenly employed during the validation process (e.g., epoch selection, hyper-parameter tuning), which might leak the supposed ``unknown" abnormality unintentionally. In this study, we investigated these two essential aspects regarding universal anomaly detection in medical images by (1) comparing various anomaly detection methods across four medical datasets, (2) investigating the inevitable but often neglected issues on how to unbiasedly select the optimal anomaly detection model during the validation phase using only normal images, and (3) proposing a simple decision-level ensemble method to leverage the advantage of different kinds of anomaly detection without knowing the abnormality. The results of our experiments indicate that none of the evaluated methods consistently achieved the best performance across all datasets. Our proposed method enhanced the robustness of performance in general (average AUC 0.956).

{{</citation>}}


### (73/94) LXL: LiDAR Excluded Lean 3D Object Detection with 4D Imaging Radar and Camera Fusion (Weiyi Xiong et al., 2023)

{{<citation>}}

Weiyi Xiong, Jianan Liu, Tao Huang, Qing-Long Han, Yuxuan Xia, Bing Zhu. (2023)  
**LXL: LiDAR Excluded Lean 3D Object Detection with 4D Imaging Radar and Camera Fusion**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2307.00724v2)  

---


**ABSTRACT**  
As an emerging technology and a relatively affordable device, the 4D imaging radar has already been confirmed effective in performing 3D object detection in autonomous driving. Nevertheless, the sparsity and noisiness of 4D radar point clouds hinder further performance improvement, and in-depth studies about its fusion with other modalities are lacking. On the other hand, most of the camera-based perception methods transform the extracted image perspective view features into the bird's-eye view geometrically via "depth-based splatting" proposed in Lift-Splat-Shoot (LSS), and some researchers exploit other modals such as LiDARs or ordinary automotive radars for enhancement. Recently, a few works have applied the "sampling" strategy for image view transformation, showing that it outperforms "splatting" even without image depth prediction. However, the potential of "sampling" is not fully unleashed. In this paper, we investigate the "sampling" view transformation strategy on the camera and 4D imaging radar fusion-based 3D object detection. In the proposed model, LXL, predicted image depth distribution maps and radar 3D occupancy grids are utilized to aid image view transformation, called "radar occupancy-assisted depth-based sampling". Experiments on VoD and TJ4DRadSet datasets show that the proposed method outperforms existing 3D object detection methods by a significant margin without bells and whistles. Ablation studies demonstrate that our method performs the best among different enhancement settings.

{{</citation>}}


### (74/94) SSC3OD: Sparsely Supervised Collaborative 3D Object Detection from LiDAR Point Clouds (Yushan Han et al., 2023)

{{<citation>}}

Yushan Han, Hui Zhang, Honglei Zhang, Yidong Li. (2023)  
**SSC3OD: Sparsely Supervised Collaborative 3D Object Detection from LiDAR Point Clouds**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2307.00717v1)  

---


**ABSTRACT**  
Collaborative 3D object detection, with its improved interaction advantage among multiple agents, has been widely explored in autonomous driving. However, existing collaborative 3D object detectors in a fully supervised paradigm heavily rely on large-scale annotated 3D bounding boxes, which is labor-intensive and time-consuming. To tackle this issue, we propose a sparsely supervised collaborative 3D object detection framework SSC3OD, which only requires each agent to randomly label one object in the scene. Specifically, this model consists of two novel components, i.e., the pillar-based masked autoencoder (Pillar-MAE) and the instance mining module. The Pillar-MAE module aims to reason over high-level semantics in a self-supervised manner, and the instance mining module generates high-quality pseudo labels for collaborative detectors online. By introducing these simple yet effective mechanisms, the proposed SSC3OD can alleviate the adverse impacts of incomplete annotations. We generate sparse labels based on collaborative perception datasets to evaluate our method. Extensive experiments on three large-scale datasets reveal that our proposed SSC3OD can effectively improve the performance of sparsely supervised collaborative 3D object detectors.

{{</citation>}}


### (75/94) Guided Patch-Grouping Wavelet Transformer with Spatial Congruence for Ultra-High Resolution Segmentation (Deyi Ji et al., 2023)

{{<citation>}}

Deyi Ji, Feng Zhao, Hongtao Lu. (2023)  
**Guided Patch-Grouping Wavelet Transformer with Spatial Congruence for Ultra-High Resolution Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.00711v2)  

---


**ABSTRACT**  
Most existing ultra-high resolution (UHR) segmentation methods always struggle in the dilemma of balancing memory cost and local characterization accuracy, which are both taken into account in our proposed Guided Patch-Grouping Wavelet Transformer (GPWFormer) that achieves impressive performances. In this work, GPWFormer is a Transformer ($\mathcal{T}$)-CNN ($\mathcal{C}$) mutual leaning framework, where $\mathcal{T}$ takes the whole UHR image as input and harvests both local details and fine-grained long-range contextual dependencies, while $\mathcal{C}$ takes downsampled image as input for learning the category-wise deep context. For the sake of high inference speed and low computation complexity, $\mathcal{T}$ partitions the original UHR image into patches and groups them dynamically, then learns the low-level local details with the lightweight multi-head Wavelet Transformer (WFormer) network. Meanwhile, the fine-grained long-range contextual dependencies are also captured during this process, since patches that are far away in the spatial domain can also be assigned to the same group. In addition, masks produced by $\mathcal{C}$ are utilized to guide the patch grouping process, providing a heuristics decision. Moreover, the congruence constraints between the two branches are also exploited to maintain the spatial consistency among the patches. Overall, we stack the multi-stage process in a pyramid way. Experiments show that GPWFormer outperforms the existing methods with significant improvements on five benchmark datasets.

{{</citation>}}


## cs.HC (3)



### (76/94) Human in the AI loop via xAI and Active Learning for Visual Inspection (Jože M. Rožanec et al., 2023)

{{<citation>}}

Jože M. Rožanec, Elias Montini, Vincenzo Cutrona, Dimitrios Papamartzivanos, Timotej Klemenčič, Blaž Fortuna, Dunja Mladenić, Entso Veliou, Thanassis Giannetsos, Christos Emmanouilidis. (2023)  
**Human in the AI loop via xAI and Active Learning for Visual Inspection**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-CV, cs-HC, cs.HC  
Keywords: AI, Active Learning  
[Paper Link](http://arxiv.org/abs/2307.05508v2)  

---


**ABSTRACT**  
Industrial revolutions have historically disrupted manufacturing by introducing automation into production. Increasing automation reshapes the role of the human worker. Advances in robotics and artificial intelligence open new frontiers of human-machine collaboration. Such collaboration can be realized considering two sub-fields of artificial intelligence: active learning and explainable artificial intelligence. Active learning aims to devise strategies that help obtain data that allows machine learning algorithms to learn better. On the other hand, explainable artificial intelligence aims to make the machine learning models intelligible to the human person. The present work first describes Industry 5.0, human-machine collaboration, and state-of-the-art regarding quality inspection, emphasizing visual inspection. Then it outlines how human-machine collaboration could be realized and enhanced in visual inspection. Finally, some of the results obtained in the EU H2020 STAR project regarding visual inspection are shared, considering artificial intelligence, human digital twins, and cybersecurity.

{{</citation>}}


### (77/94) Prompt Middleware: Mapping Prompts for Large Language Models to UI Affordances (Stephen MacNeil et al., 2023)

{{<citation>}}

Stephen MacNeil, Andrew Tran, Joanne Kim, Ziheng Huang, Seth Bernstein, Dan Mogil. (2023)  
**Prompt Middleware: Mapping Prompts for Large Language Models to UI Affordances**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2307.01142v1)  

---


**ABSTRACT**  
To help users do complex work, researchers have developed techniques to integrate AI and human intelligence into user interfaces (UIs). With the recent introduction of large language models (LLMs), which can generate text in response to a natural language prompt, there are new opportunities to consider how to integrate LLMs into UIs. We present Prompt Middleware, a framework for generating prompts for LLMs based on UI affordances. These include prompts that are predefined by experts (static prompts), generated from templates with fill-in options in the UI (template-based prompts), or created from scratch (free-form prompts). We demonstrate this framework with FeedbackBuffet, a writing assistant that automatically generates feedback based on a user's text input. Inspired by prior research showing how templates can help non-experts perform more like experts, FeedbackBuffet leverages template-based prompt middleware to enable feedback seekers to specify the types of feedback they want to receive as options in a UI. These options are composed using a template to form a feedback request prompt to GPT-3. We conclude with a discussion about how Prompt Middleware can help developers integrate LLMs into UIs.

{{</citation>}}


### (78/94) Towards Real Smart Apps: Investigating Human-AI Interactions in Smartphone On-Device AI Apps (Jason Ching Yuen Siu et al., 2023)

{{<citation>}}

Jason Ching Yuen Siu, Jieshan Chen, Yujin Huang, Zhenchang Xing, Chunyang Chen. (2023)  
**Towards Real Smart Apps: Investigating Human-AI Interactions in Smartphone On-Device AI Apps**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.00756v1)  

---


**ABSTRACT**  
With the emergence of deep learning techniques, smartphone apps are now embedded on-device AI features for enabling advanced tasks like speech translation, to attract users and increase market competitiveness. A good interaction design is important to make an AI feature usable and understandable. However, AI features have their unique challenges like sensitiveness to the input, dynamic behaviours and output uncertainty. Existing guidelines and tools either do not cover AI features or consider mobile apps which are confirmed by our informal interview with professional designers. To address these issues, we conducted the first empirical study to explore user-AI-interaction in mobile apps. We aim to understand the status of on-device AI usage by investigating 176 AI apps from 62,822 apps. We identified 255 AI features and summarised 759 implementations into three primary interaction pattern types. We further implemented our findings into a multi-faceted search-enabled gallery. The results of the user study demonstrate the usefulness of our findings.

{{</citation>}}


## math.HO (1)



### (79/94) Translating Latin with Artificial Intelligence (Sylvio R. Bistafa, 2023)

{{<citation>}}

Sylvio R. Bistafa. (2023)  
**Translating Latin with Artificial Intelligence**  

---
Primary Category: math.HO  
Categories: cs-CL, math-HO, math.HO, physics-ed-ph  
Keywords: AI, ChatGPT, GPT, Google  
[Paper Link](http://arxiv.org/abs/2307.07520v1)  

---


**ABSTRACT**  
The major hindrance in the study of earlier scientific literature is the availability of Latin translations into modern languages. This is particular true for the works of Euler who authored about 850 manuscripts and wrote a thousand letters and received back almost two thousand more. The translation of many of these manuscripts, books and letters have been published in various sources over the last two centuries, but many more have not yet appeared. Fortunately, nowadays, the artificial intelligence AI translation can be used to circumvent the challenges of translating such substantial number of texts. To validate this tool, benchmark tests have been performed to compare the performance of two popular AI translating algorithms, namely Google Translate and ChatGPT. Since it was found that ChatGPT performed better on these tests, this translating support was then used on an excerpt of a 1739 letter from Johann Bernoulli to Euler, where he notifies that he was sending to Euler the first part of his manuscript Hydraulica. The findings highlight ChatGPT as a valuable translation tool, catering not only to general Latin practitioners but also proving beneficial for specialized Latin translators.

{{</citation>}}


## cs.CR (3)



### (80/94) Passive Query-Recovery Attack Against Secure Conjunctive Keyword Search Schemes (Marco Dijkslag et al., 2023)

{{<citation>}}

Marco Dijkslag, Marc Damie, Florian Hahn, Andreas Peter. (2023)  
**Passive Query-Recovery Attack Against Secure Conjunctive Keyword Search Schemes**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2307.01131v1)  

---


**ABSTRACT**  
While storing documents on the cloud can be attractive, the question remains whether cloud providers can be trusted with storing private documents. Even if trusted, data breaches are ubiquitous. To prevent information leakage one can store documents encrypted. If encrypted under traditional schemes, one loses the ability to perform simple operations over the documents, such as searching through them. Searchable encryption schemes were proposed allowing some search functionality while documents remain encrypted. Orthogonally, research is done to find attacks that exploit search and access pattern leakage that most efficient schemes have. One type of such an attack is the ability to recover plaintext queries. Passive query-recovery attacks on single-keyword search schemes have been proposed in literature, however, conjunctive keyword search has not been considered, although keyword searches with two or three keywords appear more frequently in online searches.   We introduce a generic extension strategy for existing passive query-recovery attacks against single-keyword search schemes and explore its applicability for the attack presented by Damie et al. (USENIX Security '21). While the original attack achieves up to a recovery rate of 85% against single-keyword search schemes for an attacker without exact background knowledge, our experiments show that the generic extension to conjunctive queries comes with a significant performance decrease achieving recovery rates of at most 32%. Assuming a stronger attacker with partial knowledge of the indexed document set boosts the recovery rate to 85% for conjunctive keyword queries with two keywords and achieves similar recovery rates as previous attacks by Cash et al. (CCS '15) and Islam et al. (NDSS '12) in the same setting for single-keyword search schemes.

{{</citation>}}


### (81/94) BehaveFormer: A Framework with Spatio-Temporal Dual Attention Transformers for IMU enhanced Keystroke Dynamics (Dilshan Senerath et al., 2023)

{{<citation>}}

Dilshan Senerath, Sanuja Tharinda, Maduka Vishwajith, Sanka Rasnayaka, Sandareka Wickramanayake, Dulani Meedeniya. (2023)  
**BehaveFormer: A Framework with Spatio-Temporal Dual Attention Transformers for IMU enhanced Keystroke Dynamics**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Attention, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.11000v1)  

---


**ABSTRACT**  
Continuous Authentication (CA) using behavioural biometrics is a type of biometric identification that recognizes individuals based on their unique behavioural characteristics, like their typing style. However, the existing systems that use keystroke or touch stroke data have limited accuracy and reliability. To improve this, smartphones' Inertial Measurement Unit (IMU) sensors, which include accelerometers, gyroscopes, and magnetometers, can be used to gather data on users' behavioural patterns, such as how they hold their phones. Combining this IMU data with keystroke data can enhance the accuracy of behavioural biometrics-based CA. This paper proposes BehaveFormer, a new framework that employs keystroke and IMU data to create a reliable and accurate behavioural biometric CA system. It includes two Spatio-Temporal Dual Attention Transformer (STDAT), a novel transformer we introduce to extract more discriminative features from keystroke dynamics. Experimental results on three publicly available datasets (Aalto DB, HMOG DB, and HuMIdb) demonstrate that BehaveFormer outperforms the state-of-the-art behavioural biometric-based CA systems. For instance, on the HuMIdb dataset, BehaveFormer achieved an EER of 2.95\%. Additionally, the proposed STDAT has been shown to improve the BehaveFormer system even when only keystroke data is used. For example, on the Aalto DB dataset, BehaveFormer achieved an EER of 1.80\%. These results demonstrate the effectiveness of the proposed STDAT and the incorporation of IMU data for behavioural biometric authentication.

{{</citation>}}


### (82/94) From ChatGPT to ThreatGPT: Impact of Generative AI in Cybersecurity and Privacy (Maanak Gupta et al., 2023)

{{<citation>}}

Maanak Gupta, CharanKumar Akiri, Kshitiz Aryal, Eli Parker, Lopamudra Praharaj. (2023)  
**From ChatGPT to ThreatGPT: Impact of Generative AI in Cybersecurity and Privacy**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: AI, ChatGPT, GPT, Generative AI, Google  
[Paper Link](http://arxiv.org/abs/2307.00691v1)  

---


**ABSTRACT**  
Undoubtedly, the evolution of Generative AI (GenAI) models has been the highlight of digital transformation in the year 2022. As the different GenAI models like ChatGPT and Google Bard continue to foster their complexity and capability, it's critical to understand its consequences from a cybersecurity perspective. Several instances recently have demonstrated the use of GenAI tools in both the defensive and offensive side of cybersecurity, and focusing on the social, ethical and privacy implications this technology possesses. This research paper highlights the limitations, challenges, potential risks, and opportunities of GenAI in the domain of cybersecurity and privacy. The work presents the vulnerabilities of ChatGPT, which can be exploited by malicious users to exfiltrate malicious information bypassing the ethical constraints on the model. This paper demonstrates successful example attacks like Jailbreaks, reverse psychology, and prompt injection attacks on the ChatGPT. The paper also investigates how cyber offenders can use the GenAI tools in developing cyber attacks, and explore the scenarios where ChatGPT can be used by adversaries to create social engineering attacks, phishing attacks, automated hacking, attack payload generation, malware creation, and polymorphic malware. This paper then examines defense techniques and uses GenAI tools to improve security measures, including cyber defense automation, reporting, threat intelligence, secure code generation and detection, attack identification, developing ethical guidelines, incidence response plans, and malware detection. We will also discuss the social, legal, and ethical implications of ChatGPT. In conclusion, the paper highlights open challenges and future directions to make this GenAI secure, safe, trustworthy, and ethical as the community understands its cybersecurity impacts.

{{</citation>}}


## eess.IV (3)



### (83/94) Cross-modality Attention Adapter: A Glioma Segmentation Fine-tuning Method for SAM Using Multimodal Brain MR Images (Xiaoyu Shi et al., 2023)

{{<citation>}}

Xiaoyu Shi, Shurong Chai, Yinhao Li, Jingliang Cheng, Jie Bai, Guohua Zhao, Yen-Wei Chen. (2023)  
**Cross-modality Attention Adapter: A Glioma Segmentation Fine-tuning Method for SAM Using Multimodal Brain MR Images**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2307.01124v1)  

---


**ABSTRACT**  
According to the 2021 World Health Organization (WHO) Classification scheme for gliomas, glioma segmentation is a very important basis for diagnosis and genotype prediction. In general, 3D multimodal brain MRI is an effective diagnostic tool. In the past decade, there has been an increase in the use of machine learning, particularly deep learning, for medical images processing. Thanks to the development of foundation models, models pre-trained with large-scale datasets have achieved better results on a variety of tasks. However, for medical images with small dataset sizes, deep learning methods struggle to achieve better results on real-world image datasets. In this paper, we propose a cross-modality attention adapter based on multimodal fusion to fine-tune the foundation model to accomplish the task of glioma segmentation in multimodal MRI brain images with better results. The effectiveness of the proposed method is validated via our private glioma data set from the First Affiliated Hospital of Zhengzhou University (FHZU) in Zhengzhou, China. Our proposed method is superior to current state-of-the-art methods with a Dice of 88.38% and Hausdorff distance of 10.64, thereby exhibiting a 4% increase in Dice to segment the glioma region for glioma treatment.

{{</citation>}}


### (84/94) Synthesis of Contrast-Enhanced Breast MRI Using Multi-b-Value DWI-based Hierarchical Fusion Network with Attention Mechanism (Tianyu Zhang et al., 2023)

{{<citation>}}

Tianyu Zhang, Luyi Han, Anna D'Angelo, Xin Wang, Yuan Gao, Chunyao Lu, Jonas Teuwen, Regina Beets-Tan, Tao Tan, Ritse Mann. (2023)  
**Synthesis of Contrast-Enhanced Breast MRI Using Multi-b-Value DWI-based Hierarchical Fusion Network with Attention Mechanism**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2307.00895v1)  

---


**ABSTRACT**  
Magnetic resonance imaging (MRI) is the most sensitive technique for breast cancer detection among current clinical imaging modalities. Contrast-enhanced MRI (CE-MRI) provides superior differentiation between tumors and invaded healthy tissue, and has become an indispensable technique in the detection and evaluation of cancer. However, the use of gadolinium-based contrast agents (GBCA) to obtain CE-MRI may be associated with nephrogenic systemic fibrosis and may lead to bioaccumulation in the brain, posing a potential risk to human health. Moreover, and likely more important, the use of gadolinium-based contrast agents requires the cannulation of a vein, and the injection of the contrast media which is cumbersome and places a burden on the patient. To reduce the use of contrast agents, diffusion-weighted imaging (DWI) is emerging as a key imaging technique, although currently usually complementing breast CE-MRI. In this study, we develop a multi-sequence fusion network to synthesize CE-MRI based on T1-weighted MRI and DWIs. DWIs with different b-values are fused to efficiently utilize the difference features of DWIs. Rather than proposing a pure data-driven approach, we invent a multi-sequence attention module to obtain refined feature maps, and leverage hierarchical representation information fused at different scales while utilizing the contributions from different sequences from a model-driven approach by introducing the weighted difference module. The results show that the multi-b-value DWI-based fusion model can potentially be used to synthesize CE-MRI, thus theoretically reducing or avoiding the use of GBCA, thereby minimizing the burden to patients. Our code is available at \url{https://github.com/Netherlands-Cancer-Institute/CE-MRI}.

{{</citation>}}


### (85/94) End-To-End Prediction of Knee Osteoarthritis Progression With Multi-Modal Transformers (Egor Panfilov et al., 2023)

{{<citation>}}

Egor Panfilov, Simo Saarakkala, Miika T. Nieminen, Aleksei Tiulpin. (2023)  
**End-To-End Prediction of Knee Osteoarthritis Progression With Multi-Modal Transformers**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.00873v1)  

---


**ABSTRACT**  
Knee Osteoarthritis (KOA) is a highly prevalent chronic musculoskeletal condition with no currently available treatment. The manifestation of KOA is heterogeneous and prediction of its progression is challenging. Current literature suggests that the use of multi-modal data and advanced modeling methods, such as the ones based on Deep Learning, has promise in tackling this challenge. To date, however, the evidence on the efficacy of this approach is limited. In this study, we leveraged recent advances in Deep Learning and, using a Transformer approach, developed a unified framework for the multi-modal fusion of knee imaging data. Subsequently, we analyzed its performance across a range of scenarios by investigating multiple progression horizons -- from short-term to long-term. We report our findings using a large cohort (n=2421-3967) derived from the Osteoarthritis Initiative dataset. We show that structural knee MRI allows identifying radiographic KOA progressors on par with multi-modal fusion approaches, achieving an area under the ROC curve (ROC AUC) of 0.70-0.76 and Average Precision (AP) of 0.15-0.54 in 2-8 year horizons. Progression within 1 year was better predicted with a multi-modal method using X-ray, structural, and compositional MR images -- ROC AUC of 0.76(0.04), AP of 0.13(0.04) -- or via clinical data. Our follow-up analysis generally shows that prediction from the imaging data is more accurate for post-traumatic subjects, and we further investigate which subject subgroups may benefit the most. The present study provides novel insights into multi-modal imaging of KOA and brings a unified data-driven framework for studying its progression in an end-to-end manner, providing new tools for the design of more efficient clinical trials. The source code of our framework and the pre-trained models are made publicly available.

{{</citation>}}


## cs.CY (2)



### (86/94) ChatGPT is not a pocket calculator -- Problems of AI-chatbots for teaching Geography (Simon Scheider et al., 2023)

{{<citation>}}

Simon Scheider, Harm Bartholomeus, Judith Verstegen. (2023)  
**ChatGPT is not a pocket calculator -- Problems of AI-chatbots for teaching Geography**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2307.03196v1)  

---


**ABSTRACT**  
The recent success of large language models and AI chatbots such as ChatGPT in various knowledge domains has a severe impact on teaching and learning Geography and GIScience. The underlying revolution is often compared to the introduction of pocket calculators, suggesting analogous adaptations that prioritize higher-level skills over other learning content. However, using ChatGPT can be fraudulent because it threatens the validity of assessments. The success of such a strategy therefore rests on the assumption that lower-level learning goals are substitutable by AI, and supervision and assessments can be refocused on higher-level goals. Based on a preliminary survey on ChatGPT's quality in answering questions in Geography and GIScience, we demonstrate that this assumption might be fairly naive, and effective control in assessments and supervision is required.

{{</citation>}}


### (87/94) A Comprehensive Survey of Artificial Intelligence Techniques for Talent Analytics (Chuan Qin et al., 2023)

{{<citation>}}

Chuan Qin, Le Zhang, Rui Zha, Dazhong Shen, Qi Zhang, Ying Sun, Chen Zhu, Hengshu Zhu, Hui Xiong. (2023)  
**A Comprehensive Survey of Artificial Intelligence Techniques for Talent Analytics**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.03195v1)  

---


**ABSTRACT**  
In today's competitive and fast-evolving business environment, it is a critical time for organizations to rethink how to make talent-related decisions in a quantitative manner. Indeed, the recent development of Big Data and Artificial Intelligence (AI) techniques have revolutionized human resource management. The availability of large-scale talent and management-related data provides unparalleled opportunities for business leaders to comprehend organizational behaviors and gain tangible knowledge from a data science perspective, which in turn delivers intelligence for real-time decision-making and effective talent management at work for their organizations. In the last decade, talent analytics has emerged as a promising field in applied data science for human resource management, garnering significant attention from AI communities and inspiring numerous research efforts. To this end, we present an up-to-date and comprehensive survey on AI technologies used for talent analytics in the field of human resource management. Specifically, we first provide the background knowledge of talent analytics and categorize various pertinent data. Subsequently, we offer a comprehensive taxonomy of relevant research efforts, categorized based on three distinct application-driven scenarios: talent management, organization management, and labor market analysis. In conclusion, we summarize the open challenges and potential prospects for future research directions in the domain of AI-driven talent analytics.

{{</citation>}}


## physics.bio-ph (1)



### (88/94) Environmental effects on emergent strategy in micro-scale multi-agent reinforcement learning (Samuel Tovey et al., 2023)

{{<citation>}}

Samuel Tovey, David Zimmer, Christoph Lohrmann, Tobias Merkt, Simon Koppenhoefer, Veit-Lorenz Heuthe, Clemens Bechinger, Christian Holm. (2023)  
**Environmental effects on emergent strategy in micro-scale multi-agent reinforcement learning**  

---
Primary Category: physics.bio-ph  
Categories: cs-LG, cs-RO, physics-bio-ph, physics.bio-ph  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.00994v1)  

---


**ABSTRACT**  
Multi-Agent Reinforcement Learning (MARL) is a promising candidate for realizing efficient control of microscopic particles, of which micro-robots are a subset. However, the microscopic particles' environment presents unique challenges, such as Brownian motion at sufficiently small length-scales. In this work, we explore the role of temperature in the emergence and efficacy of strategies in MARL systems using particle-based Langevin molecular dynamics simulations as a realistic representation of micro-scale environments. To this end, we perform experiments on two different multi-agent tasks in microscopic environments at different temperatures, detecting the source of a concentration gradient and rotation of a rod. We find that at higher temperatures, the RL agents identify new strategies for achieving these tasks, highlighting the importance of understanding this regime and providing insight into optimal training strategies for bridging the generalization gap between simulation and reality. We also introduce a novel Python package for studying microscopic agents using reinforcement learning (RL) to accompany our results.

{{</citation>}}


## cs.NI (3)



### (89/94) Digital Twin-Empowered Communications: A New Frontier of Wireless Networks (Lina Bariah et al., 2023)

{{<citation>}}

Lina Bariah, Hikmet Sari, Merouane Debbah. (2023)  
**Digital Twin-Empowered Communications: A New Frontier of Wireless Networks**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.00973v1)  

---


**ABSTRACT**  
The future of wireless network generations is revolving toward unlocking the opportunities offered by virtualization and digitization services, with the aim to realize improved quality-of-experience (QoE) and bring several advantages to network users. According to the rapid development in the field of network virtualization, we envision that future wireless networks will run over ubiquitous deployment of virtualized components that are controlled by artificial intelligence (AI), i.e., the conceptualization of the Digital Twin (DT) paradigm. The key principle of the DT relies on creating a holistic representation of wireless network elements, in addition to decoupling the information pertaining to physical objects and dynamics, into a cyber twin. The cyber twin will then leverage this information for AI models training, and then reasoning and decision-making operations, which will be then reflected to the physical environment, for improved sustainability. Motivated by this, in this article, we dig deep into the intertwined role of wireless technologies as being enablers and enabled by the DT. Furthermore, we put a forward-looking vision of the integral role that future 6G networks are anticipated to play in order to realize an efficient DT. Finally, we sketch the roadmap toward identifying the limitations of the DT in 6G-enabled wireless networks, and open new horizons for further developments in different design aspects.

{{</citation>}}


### (90/94) 5G Wings: Investigating 5G-Connected Drones Performance in Non-Urban Areas (Mohammed Gharib et al., 2023)

{{<citation>}}

Mohammed Gharib, Bryce Hopkins, Jackson Murrin, Andre Koka, Fatemeh Afghah. (2023)  
**5G Wings: Investigating 5G-Connected Drones Performance in Non-Urban Areas**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI, eess-SP  
Keywords: AI, Drone  
[Paper Link](http://arxiv.org/abs/2307.00959v1)  

---


**ABSTRACT**  
Unmanned aerial vehicles (UAVs) have become extremely popular for both military and civilian applications due to their ease of deployment, cost-effectiveness, high maneuverability, and availability. Both applications, however, need reliable communication for command and control (C2) and/or data transmission. Utilizing commercial cellular networks for drone communication can enable beyond visual line of sight (BVLOS) operation, high data rate transmission, and secure communication. However, deployment of cellular-connected drones over commercial LTE/5G networks still presents various challenges such as sparse coverage outside urban areas, and interference caused to the network as the UAV is visible to many towers. Commercial 5G networks can offer various features for aerial user equipment (UE) far beyond what LTE could provide by taking advantage of mmWave, flexible numerology, slicing, and the capability of applying AI-based solutions. Limited experimental data is available to investigate the operation of aerial UEs over current, without any modification, commercial 5G networks, particularly in suburban and NON-URBAN areas. In this paper, we perform a comprehensive study of drone communications over the existing low-band and mid-band 5G networks in a suburban area for different velocities and elevations, comparing the performance against that of LTE. It is important to acknowledge that the network examined in this research is primarily designed and optimized to meet the requirements of terrestrial users, and may not adequately address the needs of aerial users. This paper not only reports the Key Performance Indicators (KPIs) compared among all combinations of the test cases but also provides recommendations for aerial users to enhance their communication quality by controlling their trajectory.

{{</citation>}}


### (91/94) A Multi-Agent Deep Reinforcement Learning Approach for RAN Resource Allocation in O-RAN (Farhad Rezazadeh et al., 2023)

{{<citation>}}

Farhad Rezazadeh, Lanfranco Zanzi, Francesco Devoti, Sergio Barrachina-Munoz, Engin Zeydan, Xavier Costa-Pérez, Josep Mangues-Bafalluy. (2023)  
**A Multi-Agent Deep Reinforcement Learning Approach for RAN Resource Allocation in O-RAN**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.02414v1)  

---


**ABSTRACT**  
Artificial intelligence (AI) and Machine Learning (ML) are considered as key enablers for realizing the full potential of fifth-generation (5G) and beyond mobile networks, particularly in the context of resource management and orchestration. In this demonstration, we consider a fully-fledged 5G mobile network and develop a multi-agent deep reinforcement learning (DRL) framework for RAN resource allocation. By leveraging local monitoring information generated by a shared gNodeB instance (gNB), each DRL agent aims to optimally allocate radio resources concerning service-specific traffic demands belonging to heterogeneous running services. We perform experiments on the deployed testbed in real-time, showing that DRL-based agents can allocate radio resources fairly while improving the overall efficiency of resource utilization and minimizing the risk of over provisioning.

{{</citation>}}


## physics.soc-ph (1)



### (92/94) A data-driven kinetic model for opinion dynamics with social network contacts (Giacomo Albi et al., 2023)

{{<citation>}}

Giacomo Albi, Elisa Calzola, Giacomo Dimarco. (2023)  
**A data-driven kinetic model for opinion dynamics with social network contacts**  

---
Primary Category: physics.soc-ph  
Categories: cs-NA, math-NA, physics-soc-ph, physics.soc-ph  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2307.00906v1)  

---


**ABSTRACT**  
Opinion dynamics is an important and very active area of research that delves into the complex processes through which individuals form and modify their opinions within a social context. The ability to comprehend and unravel the mechanisms that drive opinion formation is of great significance for predicting a wide range of social phenomena such as political polarization, the diffusion of misinformation, the formation of public consensus, and the emergence of collective behaviors. In this paper, we aim to contribute to that field by introducing a novel mathematical model that specifically accounts for the influence of social media networks on opinion dynamics. With the rise of platforms such as Twitter, Facebook, and Instagram and many others, social networks have become significant arenas where opinions are shared, discussed, and potentially altered. To this aim after an analytical construction of our new model and through incorporation of real-life data from Twitter, we calibrate the model parameters to accurately reflect the dynamics that unfold in social media, showing in particular the role played by the so-called influencers in driving individual opinions towards predetermined directions.

{{</citation>}}


## q-bio.NC (1)



### (93/94) Beyond the Snapshot: Brain Tokenized Graph Transformer for Longitudinal Brain Functional Connectome Embedding (Zijian Dong et al., 2023)

{{<citation>}}

Zijian Dong, Yilei Wu, Yu Xiao, Joanna Su Xian Chong, Yueming Jin, Juan Helen Zhou. (2023)  
**Beyond the Snapshot: Brain Tokenized Graph Transformer for Longitudinal Brain Functional Connectome Embedding**  

---
Primary Category: q-bio.NC  
Categories: cs-LG, eess-IV, q-bio-NC, q-bio.NC  
Keywords: Embedding, GNN, Graph Neural Network, Graph Neural Networks, Transformer  
[Paper Link](http://arxiv.org/abs/2307.00858v2)  

---


**ABSTRACT**  
Under the framework of network-based neurodegeneration, brain functional connectome (FC)-based Graph Neural Networks (GNN) have emerged as a valuable tool for the diagnosis and prognosis of neurodegenerative diseases such as Alzheimer's disease (AD). However, these models are tailored for brain FC at a single time point instead of characterizing FC trajectory. Discerning how FC evolves with disease progression, particularly at the predementia stages such as cognitively normal individuals with amyloid deposition or individuals with mild cognitive impairment (MCI), is crucial for delineating disease spreading patterns and developing effective strategies to slow down or even halt disease advancement. In this work, we proposed the first interpretable framework for brain FC trajectory embedding with application to neurodegenerative disease diagnosis and prognosis, namely Brain Tokenized Graph Transformer (Brain TokenGT). It consists of two modules: 1) Graph Invariant and Variant Embedding (GIVE) for generation of node and spatio-temporal edge embeddings, which were tokenized for downstream processing; 2) Brain Informed Graph Transformer Readout (BIGTR) which augments previous tokens with trainable type identifiers and non-trainable node identifiers and feeds them into a standard transformer encoder to readout. We conducted extensive experiments on two public longitudinal fMRI datasets of the AD continuum for three tasks, including differentiating MCI from controls, predicting dementia conversion in MCI, and classification of amyloid positive or negative cognitively normal individuals. Based on brain FC trajectory, the proposed Brain TokenGT approach outperformed all the other benchmark models and at the same time provided excellent interpretability. The code is available at https://github.com/ZijianD/Brain-TokenGT.git

{{</citation>}}


## cs.DB (1)



### (94/94) Ontology-based Mediation with Quality Criteria (Muhammad Fahad, 2023)

{{<citation>}}

Muhammad Fahad. (2023)  
**Ontology-based Mediation with Quality Criteria**  

---
Primary Category: cs.DB  
Categories: cs-DB, cs.DB  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2307.00830v1)  

---


**ABSTRACT**  
This paper presents a semantic system named OntMed for an ontology-based data integration of heterogeneous data sources to achieve interoperability between heterogeneous data sources. Our system is based on the quality criteria (consistency, completeness and conciseness) for building the reliable analysis contexts to provide an accurate unified view of data to the end user. The generation of an error-free global analysis context with the semantic validation of initial mappings generates accuracy, and provides the means to access and exchange information in semantically sound manner. In addition, data integration in this way becomes more practical for dynamic situations and helps decision maker to work within more consistent and reliable virtual data warehouse. We also discuss our successful participation in the Ontology Alignment for Query Answering (OA4QA) track at OAEI 2015 campaign, where our system (DKP-AOM) has performed fair enough and became one of only matchers whose alignments allowed answering all the queries of the evaluation.

{{</citation>}}
