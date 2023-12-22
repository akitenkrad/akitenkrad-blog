---
draft: false
title: "arXiv @ 2023.12.18"
date: 2023-12-18
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.12.18"
    identifier: arxiv_20231218
    parent: 202312_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.AR (1)](#csar-1)
- [cs.RO (3)](#csro-3)
- [cs.LG (17)](#cslg-17)
- [cs.IT (2)](#csit-2)
- [cs.CV (13)](#cscv-13)
- [cs.CL (11)](#cscl-11)
- [cs.HC (3)](#cshc-3)
- [cs.SE (4)](#csse-4)
- [cs.NI (1)](#csni-1)
- [cs.SI (3)](#cssi-3)
- [eess.IV (1)](#eessiv-1)
- [econ.GN (1)](#econgn-1)
- [cs.IR (3)](#csir-3)
- [cs.AI (5)](#csai-5)
- [physics.geo-ph (1)](#physicsgeo-ph-1)
- [cs.MA (1)](#csma-1)
- [cs.SD (2)](#cssd-2)
- [cs.DC (2)](#csdc-2)
- [cs.DB (1)](#csdb-1)
- [physics.ao-ph (1)](#physicsao-ph-1)
- [math.NA (1)](#mathna-1)

## cs.AR (1)



### (1/77) Enabling Accelerators for Graph Computing (Kaustubh Shivdikar, 2023)

{{<citation>}}

Kaustubh Shivdikar. (2023)  
**Enabling Accelerators for Graph Computing**  

---
Primary Category: cs.AR  
Categories: cs-AI, cs-AR, cs-DC, cs-LG, cs.AR  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2312.10561v1)  

---


**ABSTRACT**  
The advent of Graph Neural Networks (GNNs) has revolutionized the field of machine learning, offering a novel paradigm for learning on graph-structured data. Unlike traditional neural networks, GNNs are capable of capturing complex relationships and dependencies inherent in graph data, making them particularly suited for a wide range of applications including social network analysis, molecular chemistry, and network security. The impact of GNNs in these domains is profound, enabling more accurate models and predictions, and thereby contributing significantly to advancements in these fields.   GNNs, with their unique structure and operation, present new computational challenges compared to conventional neural networks. This requires comprehensive benchmarking and a thorough characterization of GNNs to obtain insight into their computational requirements and to identify potential performance bottlenecks. In this thesis, we aim to develop a better understanding of how GNNs interact with the underlying hardware and will leverage this knowledge as we design specialized accelerators and develop new optimizations, leading to more efficient and faster GNN computations.   Synthesizing these insights and optimizations, we design a state-of-the-art hardware accelerator capable of efficiently handling various GNN workloads. Our accelerator architecture is built on our characterization of GNN computational demands, providing clear motivation for our approach. Furthermore, we extend our exploration to emerging GNN workloads in the domain of graph neural networks. This exploration into novel models underlines our comprehensive approach, as we strive to enable accelerators that are not just performant, but also versatile, able to adapt to the evolving landscape of graph computing.

{{</citation>}}


## cs.RO (3)



### (2/77) Improving Environment Robustness of Deep Reinforcement Learning Approaches for Autonomous Racing Using Bayesian Optimization-based Curriculum Learning (Rohan Banerjee et al., 2023)

{{<citation>}}

Rohan Banerjee, Prishita Ray, Mark Campbell. (2023)  
**Improving Environment Robustness of Deep Reinforcement Learning Approaches for Autonomous Racing Using Bayesian Optimization-based Curriculum Learning**  

---
Primary Category: cs.RO  
Categories: cs-LG, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.10557v1)  

---


**ABSTRACT**  
Deep reinforcement learning (RL) approaches have been broadly applied to a large number of robotics tasks, such as robot manipulation and autonomous driving. However, an open problem in deep RL is learning policies that are robust to variations in the environment, which is an important condition for such systems to be deployed into real-world, unstructured settings. Curriculum learning is one approach that has been applied to improve generalization performance in both supervised and reinforcement learning domains, but selecting the appropriate curriculum to achieve robustness can be a user-intensive process. In our work, we show that performing probabilistic inference of the underlying curriculum-reward function using Bayesian Optimization can be a promising technique for finding a robust curriculum. We demonstrate that a curriculum found with Bayesian optimization can outperform a vanilla deep RL agent and a hand-engineered curriculum in the domain of autonomous racing with obstacle avoidance. Our code is available at https://github.com/PRISHIta123/Curriculum_RL_for_Driving.

{{</citation>}}


### (3/77) A Survey on Robotic Manipulation of Deformable Objects: Recent Advances, Open Challenges and New Frontiers (Feida Gu et al., 2023)

{{<citation>}}

Feida Gu, Yanmin Zhou, Zhipeng Wang, Shuo Jiang, Bin He. (2023)  
**A Survey on Robotic Manipulation of Deformable Objects: Recent Advances, Open Challenges and New Frontiers**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.10419v1)  

---


**ABSTRACT**  
Deformable object manipulation (DOM) for robots has a wide range of applications in various fields such as industrial, service and health care sectors. However, compared to manipulation of rigid objects, DOM poses significant challenges for robotic perception, modeling and manipulation, due to the infinite dimensionality of the state space of deformable objects (DOs) and the complexity of their dynamics. The development of computer graphics and machine learning has enabled novel techniques for DOM. These techniques, based on data-driven paradigms, can address some of the challenges that analytical approaches of DOM face. However, some existing reviews do not include all aspects of DOM, and some previous reviews do not summarize data-driven approaches adequately. In this article, we survey more than 150 relevant studies (data-driven approaches mainly) and summarize recent advances, open challenges, and new frontiers for aspects of perception, modeling and manipulation for DOs. Particularly, we summarize initial progress made by Large Language Models (LLMs) in robotic manipulation, and indicates some valuable directions for further research. We believe that integrating data-driven approaches and analytical approaches can provide viable solutions to open challenges of DOM.

{{</citation>}}


### (4/77) Deriving Rewards for Reinforcement Learning from Symbolic Behaviour Descriptions of Bipedal Walking (Daniel Harnack et al., 2023)

{{<citation>}}

Daniel Harnack, Christoph Lüth, Lukas Gross, Shivesh Kumar, Frank Kirchner. (2023)  
**Deriving Rewards for Reinforcement Learning from Symbolic Behaviour Descriptions of Bipedal Walking**  

---
Primary Category: cs.RO  
Categories: I-2-9; I-2-8; I-2-6, cs-LG, cs-LO, cs-RO, cs.RO  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.10328v1)  

---


**ABSTRACT**  
Generating physical movement behaviours from their symbolic description is a long-standing challenge in artificial intelligence (AI) and robotics, requiring insights into numerical optimization methods as well as into formalizations from symbolic AI and reasoning. In this paper, a novel approach to finding a reward function from a symbolic description is proposed. The intended system behaviour is modelled as a hybrid automaton, which reduces the system state space to allow more efficient reinforcement learning. The approach is applied to bipedal walking, by modelling the walking robot as a hybrid automaton over state space orthants, and used with the compass walker to derive a reward that incentivizes following the hybrid automaton cycle. As a result, training times of reinforcement learning controllers are reduced while final walking speed is increased. The approach can serve as a blueprint how to generate reward functions from symbolic AI and reasoning.

{{</citation>}}


## cs.LG (17)



### (5/77) Catastrophic Forgetting in Deep Learning: A Comprehensive Taxonomy (Everton L. Aleixo et al., 2023)

{{<citation>}}

Everton L. Aleixo, Juan G. Colonna, Marco Cristo, Everlandio Fernandes. (2023)  
**Catastrophic Forgetting in Deep Learning: A Comprehensive Taxonomy**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.10549v1)  

---


**ABSTRACT**  
Deep Learning models have achieved remarkable performance in tasks such as image classification or generation, often surpassing human accuracy. However, they can struggle to learn new tasks and update their knowledge without access to previous data, leading to a significant loss of accuracy known as Catastrophic Forgetting (CF). This phenomenon was first observed by McCloskey and Cohen in 1989 and remains an active research topic. Incremental learning without forgetting is widely recognized as a crucial aspect in building better AI systems, as it allows models to adapt to new tasks without losing the ability to perform previously learned ones. This article surveys recent studies that tackle CF in modern Deep Learning models that use gradient descent as their learning algorithm. Although several solutions have been proposed, a definitive solution or consensus on assessing CF is yet to be established. The article provides a comprehensive review of recent solutions, proposes a taxonomy to organize them, and identifies research gaps in this area.

{{</citation>}}


### (6/77) TrojFSP: Trojan Insertion in Few-shot Prompt Tuning (Mengxin Zheng et al., 2023)

{{<citation>}}

Mengxin Zheng, Jiaqi Xue, Xun Chen, YanShan Wang, Qian Lou, Lei Jiang. (2023)  
**TrojFSP: Trojan Insertion in Few-shot Prompt Tuning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.10467v1)  

---


**ABSTRACT**  
Prompt tuning is one of the most effective solutions to adapting a fixed pre-trained language model (PLM) for various downstream tasks, especially with only a few input samples. However, the security issues, e.g., Trojan attacks, of prompt tuning on a few data samples are not well-studied. Transferring established data poisoning attacks directly to few-shot prompt tuning presents multiple challenges. One significant issue is the \textit{poisoned imbalance issue}, where non-target class samples are added to the target class, resulting in a greater number of target-class samples compared to non-target class. While this issue is not critical in regular tuning, it significantly hampers the few-shot prompt tuning, making it difficult to simultaneously achieve a high attack success rate (ASR) and maintain clean data accuracy (CDA). Additionally, few-shot prompting is prone to overfitting in terms of both ASR and CDA. In this paper, we introduce \textit{TrojFSP}, a method designed to address the challenges. To solve the poisoned imbalance issue, we develop a \textit{Target-Class Shrink (TC-Shrink)} technique, which aims to equalize the number of poisoning samples. To combat overfitting, we employ a \textit{Selective Token Poisoning} technique to boost attack performance. Furthermore, we introduce a \textit{Trojan-Trigger Attention} objective function to amplify the attention of the poisoned trojan prompt on triggers. Experiments show that our TrojFSP achieves an ASR of over 99\% while maintaining negligible decreases in CDA across various PLMs and datasets.

{{</citation>}}


### (7/77) Degree-based stratification of nodes in Graph Neural Networks (Ameen Ali et al., 2023)

{{<citation>}}

Ameen Ali, Hakan Cevikalp, Lior Wolf. (2023)  
**Degree-based stratification of nodes in Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.10458v1)  

---


**ABSTRACT**  
Despite much research, Graph Neural Networks (GNNs) still do not display the favorable scaling properties of other deep neural networks such as Convolutional Neural Networks and Transformers. Previous work has identified issues such as oversmoothing of the latent representation and have suggested solutions such as skip connections and sophisticated normalization schemes. Here, we propose a different approach that is based on a stratification of the graph nodes. We provide motivation that the nodes in a graph can be stratified into those with a low degree and those with a high degree and that the two groups are likely to behave differently. Based on this motivation, we modify the Graph Neural Network (GNN) architecture so that the weight matrices are learned, separately, for the nodes in each group. This simple-to-implement modification seems to improve performance across datasets and GNN methods. To verify that this increase in performance is not only due to the added capacity, we also perform the same modification for random splits of the nodes, which does not lead to any improvement.

{{</citation>}}


### (8/77) Fractional Deep Reinforcement Learning for Age-Minimal Mobile Edge Computing (Lyudong Jin et al., 2023)

{{<citation>}}

Lyudong Jin, Ming Tang, Meng Zhang, Hao Wang. (2023)  
**Fractional Deep Reinforcement Learning for Age-Minimal Mobile Edge Computing**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-NI, cs.LG, eess-SP  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.10418v2)  

---


**ABSTRACT**  
Mobile edge computing (MEC) is a promising paradigm for real-time applications with intensive computational needs (e.g., autonomous driving), as it can reduce the processing delay. In this work, we focus on the timeliness of computational-intensive updates, measured by Age-ofInformation (AoI), and study how to jointly optimize the task updating and offloading policies for AoI with fractional form. Specifically, we consider edge load dynamics and formulate a task scheduling problem to minimize the expected time-average AoI. The uncertain edge load dynamics, the nature of the fractional objective, and hybrid continuous-discrete action space (due to the joint optimization) make this problem challenging and existing approaches not directly applicable. To this end, we propose a fractional reinforcement learning(RL) framework and prove its convergence. We further design a model-free fractional deep RL (DRL) algorithm, where each device makes scheduling decisions with the hybrid action space without knowing the system dynamics and decisions of other devices. Experimental results show that our proposed algorithms reduce the average AoI by up to 57.6% compared with several non-fractional benchmarks.

{{</citation>}}


### (9/77) Rethinking Dimensional Rationale in Graph Contrastive Learning from Causal Perspective (Qirui Ji et al., 2023)

{{<citation>}}

Qirui Ji, Jiangmeng Li, Jie Hu, Rui Wang, Changwen Zheng, Fanjiang Xu. (2023)  
**Rethinking Dimensional Rationale in Graph Contrastive Learning from Causal Perspective**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2312.10401v1)  

---


**ABSTRACT**  
Graph contrastive learning is a general learning paradigm excelling at capturing invariant information from diverse perturbations in graphs. Recent works focus on exploring the structural rationale from graphs, thereby increasing the discriminability of the invariant information. However, such methods may incur in the mis-learning of graph models towards the interpretability of graphs, and thus the learned noisy and task-agnostic information interferes with the prediction of graphs. To this end, with the purpose of exploring the intrinsic rationale of graphs, we accordingly propose to capture the dimensional rationale from graphs, which has not received sufficient attention in the literature. The conducted exploratory experiments attest to the feasibility of the aforementioned roadmap. To elucidate the innate mechanism behind the performance improvement arising from the dimensional rationale, we rethink the dimensional rationale in graph contrastive learning from a causal perspective and further formalize the causality among the variables in the pre-training stage to build the corresponding structural causal model. On the basis of the understanding of the structural causal model, we propose the dimensional rationale-aware graph contrastive learning approach, which introduces a learnable dimensional rationale acquiring network and a redundancy reduction constraint. The learnable dimensional rationale acquiring network is updated by leveraging a bi-level meta-learning technique, and the redundancy reduction constraint disentangles the redundant features through a decorrelation process during learning. Empirically, compared with state-of-the-art methods, our method can yield significant performance boosts on various benchmarks with respect to discriminability and transferability. The code implementation of our method is available at https://github.com/ByronJi/DRGCL.

{{</citation>}}


### (10/77) How Far Can Fairness Constraints Help Recover From Biased Data? (Mohit Sharma et al., 2023)

{{<citation>}}

Mohit Sharma, Amit Deshpande. (2023)  
**How Far Can Fairness Constraints Help Recover From Biased Data?**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2312.10396v1)  

---


**ABSTRACT**  
Blum & Stangl (2019) propose a data bias model to simulate under-representation and label bias in underprivileged population. For a stylized data distribution with i.i.d. label noise, under certain simple conditions on the bias parameters, they show that fair classification with equal opportunity constraints even on extremely biased distribution can recover an optimally accurate and fair classifier on the original distribution. Although their distribution is stylized, their result is interesting because it demonstrates that fairness constraints can implicitly rectify data bias and simultaneously overcome a perceived fairness-accuracy trade-off. In this paper, we give an alternate proof of their result using threshold-based characterization of optimal fair classifiers. Moreover, we show that their conditions on the bias parameters are both necessary and sufficient for their recovery result. Our technique is arguably more flexible, as it readily extends to more general distributions, e.g., when the labels in the original distribution have Massart noise instead of i.i.d. noise. Finally, we prove that for any data distribution, if the optimally accurate classifier in a hypothesis class is fair and robust, then it can be recovered through fair classification on the biased distribution, whenever the bias parameters satisfy certain simple conditions.

{{</citation>}}


### (11/77) RedCore: Relative Advantage Aware Cross-modal Representation Learning for Missing Modalities with Imbalanced Missing Rates (Jun Sun et al., 2023)

{{<citation>}}

Jun Sun, Xinxin Zhang, Shoukang Han, Yu-ping Ruan, Taihao Li. (2023)  
**RedCore: Relative Advantage Aware Cross-modal Representation Learning for Missing Modalities with Imbalanced Missing Rates**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2312.10386v1)  

---


**ABSTRACT**  
Multimodal learning is susceptible to modality missing, which poses a major obstacle for its practical applications and, thus, invigorates increasing research interest. In this paper, we investigate two challenging problems: 1) when modality missing exists in the training data, how to exploit the incomplete samples while guaranteeing that they are properly supervised? 2) when the missing rates of different modalities vary, causing or exacerbating the imbalance among modalities, how to address the imbalance and ensure all modalities are well-trained? To tackle these two challenges, we first introduce the variational information bottleneck (VIB) method for the cross-modal representation learning of missing modalities, which capitalizes on the available modalities and the labels as supervision. Then, accounting for the imbalanced missing rates, we define relative advantage to quantify the advantage of each modality over others. Accordingly, a bi-level optimization problem is formulated to adaptively regulate the supervision of all modalities during training. As a whole, the proposed approach features \textbf{Re}lative a\textbf{d}vantage aware \textbf{C}ross-m\textbf{o}dal \textbf{r}epresentation l\textbf{e}arning (abbreviated as \textbf{RedCore}) for missing modalities with imbalanced missing rates. Extensive empirical results demonstrate that RedCore outperforms competing models in that it exhibits superior robustness against either large or imbalanced missing rates.

{{</citation>}}


### (12/77) Imitate the Good and Avoid the Bad: An Incremental Approach to Safe Reinforcement Learning (Huy Hoang et al., 2023)

{{<citation>}}

Huy Hoang, Tien Mai Pradeep Varakantham. (2023)  
**Imitate the Good and Avoid the Bad: An Incremental Approach to Safe Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.10385v1)  

---


**ABSTRACT**  
A popular framework for enforcing safe actions in Reinforcement Learning (RL) is Constrained RL, where trajectory based constraints on expected cost (or other cost measures) are employed to enforce safety and more importantly these constraints are enforced while maximizing expected reward. Most recent approaches for solving Constrained RL convert the trajectory based cost constraint into a surrogate problem that can be solved using minor modifications to RL methods. A key drawback with such approaches is an over or underestimation of the cost constraint at each state. Therefore, we provide an approach that does not modify the trajectory based cost constraint and instead imitates ``good'' trajectories and avoids ``bad'' trajectories generated from incrementally improving policies. We employ an oracle that utilizes a reward threshold (which is varied with learning) and the overall cost constraint to label trajectories as ``good'' or ``bad''. A key advantage of our approach is that we are able to work from any starting policy or set of trajectories and improve on it. In an exhaustive set of experiments, we demonstrate that our approach is able to outperform top benchmark approaches for solving Constrained RL problems, with respect to expected cost, CVaR cost, or even unknown cost constraints.

{{</citation>}}


### (13/77) Collect and Connect Data Leaves to Feature Concepts: Interactive Graph Generation Toward Well-being (Yukio Ohsawa et al., 2023)

{{<citation>}}

Yukio Ohsawa, Tomohide Maekawa, Hiroki Yamaguchi, Hiro Yoshida, Kaira Sekiguchi. (2023)  
**Collect and Connect Data Leaves to Feature Concepts: Interactive Graph Generation Toward Well-being**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-DB, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.10375v1)  

---


**ABSTRACT**  
Feature concepts and data leaves have been invented using datasets to foster creative thoughts for creating well-being in daily life. The idea, simply put, is to attach selected and collected data leaves that are summaries of event flows to be discovered from corresponding datasets, on the target feature concept representing the well-being aimed. A graph of existing or expected datasets to be attached to a feature concept is generated semi-automatically. Rather than sheer automated generative AI, our work addresses the process of generative artificial and natural intelligence to create the basis for data use and reuse.

{{</citation>}}


### (14/77) Conformer-Based Speech Recognition On Extreme Edge-Computing Devices (Mingbin Xu et al., 2023)

{{<citation>}}

Mingbin Xu, Alex Jin, Sicheng Wang, Mu Su, Tim Ng, Henry Mason, Shiyi Han, Yaqiao Deng, Zhen Huang, Mahesh Krishnamoorthy. (2023)  
**Conformer-Based Speech Recognition On Extreme Edge-Computing Devices**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-PF, cs.LG  
Keywords: AI, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2312.10359v1)  

---


**ABSTRACT**  
With increasingly more powerful compute capabilities and resources in today's devices, traditionally compute-intensive automatic speech recognition (ASR) has been moving from the cloud to devices to better protect user privacy. However, it is still challenging to implement on-device ASR on resource-constrained devices, such as smartphones, smart wearables, and other small home automation devices. In this paper, we propose a series of model architecture adaptions, neural network graph transformations, and numerical optimizations to fit an advanced Conformer based end-to-end streaming ASR system on resource-constrained devices without accuracy degradation. We achieve over 5.26 times faster than realtime (0.19 RTF) speech recognition on small wearables while minimizing energy consumption and achieving state-of-the-art accuracy. The proposed methods are widely applicable to other transformer-based server-free AI applications. In addition, we provide a complete theory on optimal pre-normalizers that numerically stabilize layer normalization in any Lp-norm using any floating point precision.

{{</citation>}}


### (15/77) An Attentive Inductive Bias for Sequential Recommendation Beyond the Self-Attention (Yehjin Shin et al., 2023)

{{<citation>}}

Yehjin Shin, Jeongwhan Choi, Hyowon Wi, Noseong Park. (2023)  
**An Attentive Inductive Bias for Sequential Recommendation Beyond the Self-Attention**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-IR, cs-LG, cs.LG  
Keywords: Attention, Bias, Self-Attention, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.10325v1)  

---


**ABSTRACT**  
Sequential recommendation (SR) models based on Transformers have achieved remarkable successes. The self-attention mechanism of Transformers for computer vision and natural language processing suffers from the oversmoothing problem, i.e., hidden representations becoming similar to tokens. In the SR domain, we, for the first time, show that the same problem occurs. We present pioneering investigations that reveal the low-pass filtering nature of self-attention in the SR, which causes oversmoothing. To this end, we propose a novel method called Beyond Self-Attention for Sequential Recommendation (BSARec), which leverages the Fourier transform to i) inject an inductive bias by considering fine-grained sequential patterns and ii) integrate low and high-frequency information to mitigate oversmoothing. Our discovery shows significant advancements in the SR domain and is expected to bridge the gap for existing Transformer-based SR models. We test our proposed approach through extensive experiments on 6 benchmark datasets. The experimental results demonstrate that our model outperforms 7 baseline methods in terms of recommendation performance.

{{</citation>}}


### (16/77) STELLAR: Siamese Multi-Headed Attention Neural Networks for Overcoming Temporal Variations and Device Heterogeneity with Indoor Localization (Danish Gufran et al., 2023)

{{<citation>}}

Danish Gufran, Saideep Tiku, Sudeep Pasricha. (2023)  
**STELLAR: Siamese Multi-Headed Attention Neural Networks for Overcoming Temporal Variations and Device Heterogeneity with Indoor Localization**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.10312v1)  

---


**ABSTRACT**  
Smartphone-based indoor localization has emerged as a cost-effective and accurate solution to localize mobile and IoT devices indoors. However, the challenges of device heterogeneity and temporal variations have hindered its widespread adoption and accuracy. Towards jointly addressing these challenges comprehensively, we propose STELLAR, a novel framework implementing a contrastive learning approach that leverages a Siamese multi-headed attention neural network. STELLAR is the first solution that simultaneously tackles device heterogeneity and temporal variations in indoor localization, without the need for retraining the model (re-calibration-free). Our evaluations across diverse indoor environments show 8-75% improvements in accuracy compared to state-of-the-art techniques, to effectively address the device heterogeneity challenge. Moreover, STELLAR outperforms existing methods by 18-165% over 2 years of temporal variations, showcasing its robustness and adaptability.

{{</citation>}}


### (17/77) scBiGNN: Bilevel Graph Representation Learning for Cell Type Classification from Single-cell RNA Sequencing Data (Rui Yang et al., 2023)

{{<citation>}}

Rui Yang, Wenrui Dai, Chenglin Li, Junni Zou, Dapeng Wu, Hongkai Xiong. (2023)  
**scBiGNN: Bilevel Graph Representation Learning for Cell Type Classification from Single-cell RNA Sequencing Data**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, q-bio-GN  
Keywords: GNN, Representation Learning  
[Paper Link](http://arxiv.org/abs/2312.10310v1)  

---


**ABSTRACT**  
Single-cell RNA sequencing (scRNA-seq) technology provides high-throughput gene expression data to study the cellular heterogeneity and dynamics of complex organisms. Graph neural networks (GNNs) have been widely used for automatic cell type classification, which is a fundamental problem to solve in scRNA-seq analysis. However, existing methods do not sufficiently exploit both gene-gene and cell-cell relationships, and thus the true potential of GNNs is not realized. In this work, we propose a bilevel graph representation learning method, named scBiGNN, to simultaneously mine the relationships at both gene and cell levels for more accurate single-cell classification. Specifically, scBiGNN comprises two GNN modules to identify cell types. A gene-level GNN is established to adaptively learn gene-gene interactions and cell representations via the self-attention mechanism, and a cell-level GNN builds on the cell-cell graph that is constructed from the cell representations generated by the gene-level GNN. To tackle the scalability issue for processing a large number of cells, scBiGNN adopts an Expectation Maximization (EM) framework in which the two modules are alternately trained via the E-step and M-step to learn from each other. Through this interaction, the gene- and cell-level structural information is integrated to gradually enhance the classification performance of both GNN modules. Experiments on benchmark datasets demonstrate that our scBiGNN outperforms a variety of existing methods for cell type classification from scRNA-seq data.

{{</citation>}}


### (18/77) Event-Based Contrastive Learning for Medical Time Series (Hyewon Jeong et al., 2023)

{{<citation>}}

Hyewon Jeong, Nassim Oufattole, Aparna Balagopalan, Matthew Mcdermott, Payal Chandak, Marzyeh Ghassemi, Collin Stultz. (2023)  
**Event-Based Contrastive Learning for Medical Time Series**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Contrastive Learning, Time Series  
[Paper Link](http://arxiv.org/abs/2312.10308v1)  

---


**ABSTRACT**  
In clinical practice, one often needs to identify whether a patient is at high risk of adverse outcomes after some key medical event; e.g., the short-term risk of death after an admission for heart failure. This task, however, remains challenging due to the complexity, variability, and heterogeneity of longitudinal medical data, especially for individuals suffering from chronic diseases like heart failure. In this paper, we introduce Event-Based Contrastive Learning (EBCL) - a method for learning embeddings of heterogeneous patient data that preserves temporal information before and after key index events. We demonstrate that EBCL produces models that yield better fine-tuning performance on critical downstream tasks including 30-day readmission, 1-year mortality, and 1-week length of stay relative to other representation learning methods that do not exploit temporal information surrounding key medical events.

{{</citation>}}


### (19/77) PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU (Yixin Song et al., 2023)

{{<citation>}}

Yixin Song, Zeyu Mi, Haotong Xie, Haibo Chen. (2023)  
**PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-OS, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.12456v1)  

---


**ABSTRACT**  
This paper introduces PowerInfer, a high-speed Large Language Model (LLM) inference engine on a personal computer (PC) equipped with a single consumer-grade GPU. The key underlying the design of PowerInfer is exploiting the high locality inherent in LLM inference, characterized by a power-law distribution in neuron activation. This distribution indicates that a small subset of neurons, termed hot neurons, are consistently activated across inputs, while the majority, cold neurons, vary based on specific inputs. PowerInfer exploits such an insight to design a GPU-CPU hybrid inference engine: hot-activated neurons are preloaded onto the GPU for fast access, while cold-activated neurons are computed on the CPU, thus significantly reducing GPU memory demands and CPU-GPU data transfers. PowerInfer further integrates adaptive predictors and neuron-aware sparse operators, optimizing the efficiency of neuron activation and computational sparsity. Evaluation shows that PowerInfer attains an average token generation rate of 13.20 tokens/s, with a peak of 29.08 tokens/s, across various LLMs (including OPT-175B) on a single NVIDIA RTX 4090 GPU, only 18% lower than that achieved by a top-tier server-grade A100 GPU. This significantly outperforms llama.cpp by up to 11.69x while retaining model accuracy.

{{</citation>}}


### (20/77) Inductive Link Prediction in Knowledge Graphs using Path-based Neural Networks (Canlin Zhang et al., 2023)

{{<citation>}}

Canlin Zhang, Xiuwen Liu. (2023)  
**Inductive Link Prediction in Knowledge Graphs using Path-based Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Embedding, GNN, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2312.10293v1)  

---


**ABSTRACT**  
Link prediction is a crucial research area in knowledge graphs, with many downstream applications. In many real-world scenarios, inductive link prediction is required, where predictions have to be made among unseen entities. Embedding-based models usually need fine-tuning on new entity embeddings, and hence are difficult to be directly applied to inductive link prediction tasks. Logical rules captured by rule-based models can be directly applied to new entities with the same graph typologies, but the captured rules are discrete and usually lack generosity. Graph neural networks (GNNs) can generalize topological information to new graphs taking advantage of deep neural networks, which however may still need fine-tuning on new entity embeddings. In this paper, we propose SiaILP, a path-based model for inductive link prediction using siamese neural networks. Our model only depends on relation and path embeddings, which can be generalized to new entities without fine-tuning. Experiments show that our model achieves several new state-of-the-art performances in link prediction tasks using inductive versions of WN18RR, FB15k-237, and Nell995.

{{</citation>}}


### (21/77) Active Reinforcement Learning for Robust Building Control (Doseok Jang et al., 2023)

{{<citation>}}

Doseok Jang, Larry Yan, Lucas Spangher, Costas Spanos. (2023)  
**Active Reinforcement Learning for Robust Building Control**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-SY, cs.LG, eess-SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.10289v1)  

---


**ABSTRACT**  
Reinforcement learning (RL) is a powerful tool for optimal control that has found great success in Atari games, the game of Go, robotic control, and building optimization. RL is also very brittle; agents often overfit to their training environment and fail to generalize to new settings. Unsupervised environment design (UED) has been proposed as a solution to this problem, in which the agent trains in environments that have been specially selected to help it learn. Previous UED algorithms focus on trying to train an RL agent that generalizes across a large distribution of environments. This is not necessarily desirable when we wish to prioritize performance in one environment over others. In this work, we will be examining the setting of robust RL building control, where we wish to train an RL agent that prioritizes performing well in normal weather while still being robust to extreme weather conditions. We demonstrate a novel UED algorithm, ActivePLR, that uses uncertainty-aware neural network architectures to generate new training environments at the limit of the RL agent's ability while being able to prioritize performance in a desired base environment. We show that ActivePLR is able to outperform state-of-the-art UED algorithms in minimizing energy usage while maximizing occupant comfort in the setting of building control.

{{</citation>}}


## cs.IT (2)



### (22/77) Advancing RAN Slicing with Offline Reinforcement Learning (Kun Yang et al., 2023)

{{<citation>}}

Kun Yang, Shu-ping Yeh, Menglei Zhang, Jerry Sydir, Jing Yang, Cong Shen. (2023)  
**Advancing RAN Slicing with Offline Reinforcement Learning**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs-LG, cs-NI, cs.IT, eess-SP, math-IT  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.10547v1)  

---


**ABSTRACT**  
Dynamic radio resource management (RRM) in wireless networks presents significant challenges, particularly in the context of Radio Access Network (RAN) slicing. This technology, crucial for catering to varying user requirements, often grapples with complex optimization scenarios. Existing Reinforcement Learning (RL) approaches, while achieving good performance in RAN slicing, typically rely on online algorithms or behavior cloning. These methods necessitate either continuous environmental interactions or access to high-quality datasets, hindering their practical deployment. Towards addressing these limitations, this paper introduces offline RL to solving the RAN slicing problem, marking a significant shift towards more feasible and adaptive RRM methods. We demonstrate how offline RL can effectively learn near-optimal policies from sub-optimal datasets, a notable advancement over existing practices. Our research highlights the inherent flexibility of offline RL, showcasing its ability to adjust policy criteria without the need for additional environmental interactions. Furthermore, we present empirical evidence of the efficacy of offline RL in adapting to various service-level requirements, illustrating its potential in diverse RAN slicing scenarios.

{{</citation>}}


### (23/77) Spatial Deep Learning for Site-Specific Movement Optimization of Aerial Base Stations (Jiangbin Lyu et al., 2023)

{{<citation>}}

Jiangbin Lyu, Xu Chen, Jiefeng Zhang, Liqun Fu. (2023)  
**Spatial Deep Learning for Site-Specific Movement Optimization of Aerial Base Stations**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs-LG, cs.IT, math-IT  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.10490v1)  

---


**ABSTRACT**  
Unmanned aerial vehicles (UAVs) can be utilized as aerial base stations (ABSs) to provide wireless connectivity for ground users (GUs) in various emergency scenarios. However, it is a NP-hard problem with exponential complexity in $M$ and $N$, in order to maximize the coverage rate of $M$ GUs by jointly placing $N$ ABSs with limited coverage range. The problem is further complicated when the coverage range becomes irregular due to site-specific blockages (e.g., buildings) on the air-ground channel, and/or when the GUs are moving. To address the above challenges, we study a multi-ABS movement optimization problem to maximize the average coverage rate of mobile GUs in a site-specific environment. The Spatial Deep Learning with Multi-dimensional Archive of Phenotypic Elites (SDL-ME) algorithm is proposed to tackle this challenging problem by 1) partitioning the complicated ABS movement problem into ABS placement sub-problems each spanning finite time horizon; 2) using an encoder-decoder deep neural network (DNN) as the emulator to capture the spatial correlation of ABSs/GUs and thereby reducing the cost of interaction with the actual environment; 3) employing the emulator to speed up a quality-diversity search for the optimal placement solution; and 4) proposing a planning-exploration-serving scheme for multi-ABS movement coordination. Numerical results demonstrate that the proposed approach significantly outperforms the benchmark Deep Reinforcement Learning (DRL)-based method and other two baselines in terms of average coverage rate, training time and/or sample efficiency. Moreover, with one-time training, our proposed method can be applied in scenarios where the number of ABSs/GUs dynamically changes on site and/or with different/varying GU speeds, which is thus more robust and flexible compared with conventional DRL-based methods.

{{</citation>}}


## cs.CV (13)



### (24/77) Learning Interpretable Queries for Explainable Image Classification with Information Pursuit (Stefan Kolek et al., 2023)

{{<citation>}}

Stefan Kolek, Aditya Chattopadhyay, Kwan Ho Ryan Chan, Hector Andrade-Loarca, Gitta Kutyniok, Réne Vidal. (2023)  
**Learning Interpretable Queries for Explainable Image Classification with Information Pursuit**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2312.11548v1)  

---


**ABSTRACT**  
Information Pursuit (IP) is an explainable prediction algorithm that greedily selects a sequence of interpretable queries about the data in order of information gain, updating its posterior at each step based on observed query-answer pairs. The standard paradigm uses hand-crafted dictionaries of potential data queries curated by a domain expert or a large language model after a human prompt. However, in practice, hand-crafted dictionaries are limited by the expertise of the curator and the heuristics of prompt engineering. This paper introduces a novel approach: learning a dictionary of interpretable queries directly from the dataset. Our query dictionary learning problem is formulated as an optimization problem by augmenting IP's variational formulation with learnable dictionary parameters. To formulate learnable and interpretable queries, we leverage the latent space of large vision and language models like CLIP. To solve the optimization problem, we propose a new query dictionary learning algorithm inspired by classical sparse dictionary learning. Our experiments demonstrate that learned dictionaries significantly outperform hand-crafted dictionaries generated with large language models.

{{</citation>}}


### (25/77) DETER: Detecting Edited Regions for Deterring Generative Manipulations (Sai Wang et al., 2023)

{{<citation>}}

Sai Wang, Ye Zhu, Ruoyu Wang, Amaya Dharmasiri, Olga Russakovsky, Yu Wu. (2023)  
**DETER: Detecting Edited Regions for Deterring Generative Manipulations**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2312.10539v1)  

---


**ABSTRACT**  
Generative AI capabilities have grown substantially in recent years, raising renewed concerns about potential malicious use of generated data, or "deep fakes". However, deep fake datasets have not kept up with generative AI advancements sufficiently to enable the development of deep fake detection technology which can meaningfully alert human users in real-world settings. Existing datasets typically use GAN-based models and introduce spurious correlations by always editing similar face regions. To counteract the shortcomings, we introduce DETER, a large-scale dataset for DETEcting edited image Regions and deterring modern advanced generative manipulations. DETER includes 300,000 images manipulated by four state-of-the-art generators with three editing operations: face swapping (a standard coarse image manipulation), inpainting (a novel manipulation for deep fake datasets), and attribute editing (a subtle fine-grained manipulation). While face swapping and attribute editing are performed on similar face regions such as eyes and nose, the inpainting operation can be performed on random image regions, removing the spurious correlations of previous datasets. Careful image post-processing is performed to ensure deep fakes in DETER look realistic, and human studies confirm that human deep fake detection rate on DETER is 20.4% lower than on other fake datasets. Equipped with the dataset, we conduct extensive experiments and break-down analysis using our rich annotations and improved benchmark protocols, revealing future directions and the next set of challenges in developing reliable regional fake detection models.

{{</citation>}}


### (26/77) How to Train Neural Field Representations: A Comprehensive Study and Benchmark (Samuele Papa et al., 2023)

{{<citation>}}

Samuele Papa, Riccardo Valperga, David Knigge, Miltiadis Kofinas, Phillip Lippe, Jan-Jakob Sonke, Efstratios Gavves. (2023)  
**How to Train Neural Field Representations: A Comprehensive Study and Benchmark**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2312.10531v1)  

---


**ABSTRACT**  
Neural fields (NeFs) have recently emerged as a versatile method for modeling signals of various modalities, including images, shapes, and scenes. Subsequently, a number of works have explored the use of NeFs as representations for downstream tasks, e.g. classifying an image based on the parameters of a NeF that has been fit to it. However, the impact of the NeF hyperparameters on their quality as downstream representation is scarcely understood and remains largely unexplored. This is in part caused by the large amount of time required to fit datasets of neural fields.   In this work, we propose $\verb|fit-a-nef|$, a JAX-based library that leverages parallelization to enable fast optimization of large-scale NeF datasets, resulting in a significant speed-up. With this library, we perform a comprehensive study that investigates the effects of different hyperparameters -- including initialization, network architecture, and optimization strategies -- on fitting NeFs for downstream tasks. Our study provides valuable insights on how to train NeFs and offers guidance for optimizing their effectiveness in downstream applications. Finally, based on the proposed library and our analysis, we propose Neural Field Arena, a benchmark consisting of neural field variants of popular vision datasets, including MNIST, CIFAR, variants of ImageNet, and ShapeNetv2. Our library and the Neural Field Arena will be open-sourced to introduce standardized benchmarking and promote further research on neural fields.

{{</citation>}}


### (27/77) Transformers in Unsupervised Structure-from-Motion (Hemang Chawla et al., 2023)

{{<citation>}}

Hemang Chawla, Arnav Varma, Elahe Arani, Bahram Zonooz. (2023)  
**Transformers in Unsupervised Structure-from-Motion**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.10529v1)  

---


**ABSTRACT**  
Transformers have revolutionized deep learning based computer vision with improved performance as well as robustness to natural corruptions and adversarial attacks. Transformers are used predominantly for 2D vision tasks, including image classification, semantic segmentation, and object detection. However, robots and advanced driver assistance systems also require 3D scene understanding for decision making by extracting structure-from-motion (SfM). We propose a robust transformer-based monocular SfM method that learns to predict monocular pixel-wise depth, ego vehicle's translation and rotation, as well as camera's focal length and principal point, simultaneously. With experiments on KITTI and DDAD datasets, we demonstrate how to adapt different vision transformers and compare them against contemporary CNN-based methods. Our study shows that transformer-based architecture, though lower in run-time efficiency, achieves comparable performance while being more robust against natural corruptions, as well as untargeted and targeted attacks.

{{</citation>}}


### (28/77) PETDet: Proposal Enhancement for Two-Stage Fine-Grained Object Detection (Wentao Li et al., 2023)

{{<citation>}}

Wentao Li, Danpei Zhao, Bo Yuan, Yue Gao, Zhenwei Shi. (2023)  
**PETDet: Proposal Enhancement for Two-Stage Fine-Grained Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, ImageNet, Object Detection  
[Paper Link](http://arxiv.org/abs/2312.10515v1)  

---


**ABSTRACT**  
Fine-grained object detection (FGOD) extends object detection with the capability of fine-grained recognition. In recent two-stage FGOD methods, the region proposal serves as a crucial link between detection and fine-grained recognition. However, current methods overlook that some proposal-related procedures inherited from general detection are not equally suitable for FGOD, limiting the multi-task learning from generation, representation, to utilization. In this paper, we present PETDet (Proposal Enhancement for Two-stage fine-grained object detection) to better handle the sub-tasks in two-stage FGOD methods. Firstly, an anchor-free Quality Oriented Proposal Network (QOPN) is proposed with dynamic label assignment and attention-based decomposition to generate high-quality oriented proposals. Additionally, we present a Bilinear Channel Fusion Network (BCFN) to extract independent and discriminative features of the proposals. Furthermore, we design a novel Adaptive Recognition Loss (ARL) which offers guidance for the R-CNN head to focus on high-quality proposals. Extensive experiments validate the effectiveness of PETDet. Quantitative analysis reveals that PETDet with ResNet50 reaches state-of-the-art performance on various FGOD datasets, including FAIR1M-v1.0 (42.96 AP), FAIR1M-v2.0 (48.81 AP), MAR20 (85.91 AP) and ShipRSImageNet (74.90 AP). The proposed method also achieves superior compatibility between accuracy and inference speed. Our code and models will be released at https://github.com/canoe-Z/PETDet.

{{</citation>}}


### (29/77) Semantic-Aware Autoregressive Image Modeling for Visual Representation Learning (Kaiyou Song et al., 2023)

{{<citation>}}

Kaiyou Song, Shan Zhang, Tong Wang. (2023)  
**Semantic-Aware Autoregressive Image Modeling for Visual Representation Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, ImageNet, NLP, Representation Learning  
[Paper Link](http://arxiv.org/abs/2312.10457v1)  

---


**ABSTRACT**  
The development of autoregressive modeling (AM) in computer vision lags behind natural language processing (NLP) in self-supervised pre-training. This is mainly caused by the challenge that images are not sequential signals and lack a natural order when applying autoregressive modeling. In this study, inspired by human beings' way of grasping an image, i.e., focusing on the main object first, we present a semantic-aware autoregressive image modeling (SemAIM) method to tackle this challenge. The key insight of SemAIM is to autoregressive model images from the semantic patches to the less semantic patches. To this end, we first calculate a semantic-aware permutation of patches according to their feature similarities and then perform the autoregression procedure based on the permutation. In addition, considering that the raw pixels of patches are low-level signals and are not ideal prediction targets for learning high-level semantic representation, we also explore utilizing the patch features as the prediction targets. Extensive experiments are conducted on a broad range of downstream tasks, including image classification, object detection, and instance/semantic segmentation, to evaluate the performance of SemAIM. The results demonstrate SemAIM achieves state-of-the-art performance compared with other self-supervised methods. Specifically, with ViT-B, SemAIM achieves 84.1% top-1 accuracy for fine-tuning on ImageNet, 51.3% AP and 45.4% AP for object detection and instance segmentation on COCO, which outperforms the vanilla MAE by 0.5%, 1.0%, and 0.5%, respectively.

{{</citation>}}


### (30/77) Simple Image-level Classification Improves Open-vocabulary Object Detection (Ruohuan Fang et al., 2023)

{{<citation>}}

Ruohuan Fang, Guansong Pang, Xiao Bai. (2023)  
**Simple Image-level Classification Improves Open-vocabulary Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2312.10439v2)  

---


**ABSTRACT**  
Open-Vocabulary Object Detection (OVOD) aims to detect novel objects beyond a given set of base categories on which the detection model is trained. Recent OVOD methods focus on adapting the image-level pre-trained vision-language models (VLMs), such as CLIP, to a region-level object detection task via, eg., region-level knowledge distillation, regional prompt learning, or region-text pre-training, to expand the detection vocabulary. These methods have demonstrated remarkable performance in recognizing regional visual concepts, but they are weak in exploiting the VLMs' powerful global scene understanding ability learned from the billion-scale image-level text descriptions. This limits their capability in detecting hard objects of small, blurred, or occluded appearance from novel/base categories, whose detection heavily relies on contextual information. To address this, we propose a novel approach, namely Simple Image-level Classification for Context-Aware Detection Scoring (SIC-CADS), to leverage the superior global knowledge yielded from CLIP for complementing the current OVOD models from a global perspective. The core of SIC-CADS is a multi-modal multi-label recognition (MLR) module that learns the object co-occurrence-based contextual information from CLIP to recognize all possible object categories in the scene. These image-level MLR scores can then be utilized to refine the instance-level detection scores of the current OVOD models in detecting those hard objects. This is verified by extensive empirical results on two popular benchmarks, OV-LVIS and OV-COCO, which show that SIC-CADS achieves significant and consistent improvement when combined with different types of OVOD models. Further, SIC-CADS also improves the cross-dataset generalization ability on Objects365 and OpenImages. The code is available at https://github.com/mala-lab/SIC-CADS.

{{</citation>}}


### (31/77) Tender Notice Extraction from E-papers Using Neural Network (Ashmin Bhattarai et al., 2023)

{{<citation>}}

Ashmin Bhattarai, Anuj Sedhai, Devraj Neupane, Manish Khadka, Rama Bastola. (2023)  
**Tender Notice Extraction from E-papers Using Neural Network**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2312.10437v1)  

---


**ABSTRACT**  
Tender notices are usually sought by most of the companies at regular intervals as a means for obtaining the contracts of various projects. These notices consist of all the required information like description of the work, period of construction, estimated amount of project, etc. In the context of Nepal, tender notices are usually published in national as well as local newspapers. The interested bidders should search all the related tender notices in newspapers. However, it is very tedious for these companies to manually search tender notices in every newspaper and figure out which bid is best suited for them. This project is built with the purpose of solving this tedious task of manually searching the tender notices. Initially, the newspapers are downloaded in PDF format using the selenium library of python. After downloading the newspapers, the e-papers are scanned and tender notices are automatically extracted using a neural network. For extraction purposes, different architectures of CNN namely ResNet, GoogleNet and Xception are used and a model with highest performance has been implemented. Finally, these extracted notices are then published on the website and are accessible to the users. This project is helpful for construction companies as well as contractors assuring quality and efficiency. This project has great application in the field of competitive bidding as well as managing them in a systematic manner.

{{</citation>}}


### (32/77) DeepArt: A Benchmark to Advance Fidelity Research in AI-Generated Content (Wentao Wang et al., 2023)

{{<citation>}}

Wentao Wang, Xuanyao Huang, Swalpa Kumar Roy. (2023)  
**DeepArt: A Benchmark to Advance Fidelity Research in AI-Generated Content**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs-MM, cs.CV  
Keywords: AI, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2312.10407v1)  

---


**ABSTRACT**  
This paper explores the image synthesis capabilities of GPT-4, a leading multi-modal large language model. We establish a benchmark for evaluating the fidelity of texture features in images generated by GPT-4, comprising manually painted pictures and their AI-generated counterparts. The contributions of this study are threefold: First, we provide an in-depth analysis of the fidelity of image synthesis features based on GPT-4, marking the first such study on this state-of-the-art model. Second, the quantitative and qualitative experiments fully reveals the limitations of the GPT-4 model in image synthesis. Third, we have compiled a unique benchmark of manual drawings and corresponding GPT-4-generated images, introducing a new task to advance fidelity research in AI-generated content (AIGC). The dataset will be available after being accepted: \url{https://github.com/rickwang28574/DeepArt}. We hope this study will fuel knowledge, scholarship, and innovation, inspiring uses that transform how we discover and understand the world of art and promote the development of AIGC while retaining respect for art.

{{</citation>}}


### (33/77) Not Every Side Is Equal: Localization Uncertainty Estimation for Semi-Supervised 3D Object Detection (ChuXin Wang et al., 2023)

{{<citation>}}

ChuXin Wang, Wenfei Yang, Tianzhu Zhang. (2023)  
**Not Every Side Is Equal: Localization Uncertainty Estimation for Semi-Supervised 3D Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2312.10390v1)  

---


**ABSTRACT**  
Semi-supervised 3D object detection from point cloud aims to train a detector with a small number of labeled data and a large number of unlabeled data. The core of existing methods lies in how to select high-quality pseudo-labels using the designed quality evaluation criterion. However, these methods treat each pseudo bounding box as a whole and assign equal importance to each side during training, which is detrimental to model performance due to many sides having poor localization quality. Besides, existing methods filter out a large number of low-quality pseudo-labels, which also contain some correct regression values that can help with model training. To address the above issues, we propose a side-aware framework for semi-supervised 3D object detection consisting of three key designs: a 3D bounding box parameterization method, an uncertainty estimation module, and a pseudo-label selection strategy. These modules work together to explicitly estimate the localization quality of each side and assign different levels of importance during the training phase. Extensive experiment results demonstrate that the proposed method can consistently outperform baseline models under different scenes and evaluation criteria. Moreover, our method achieves state-of-the-art performance on three datasets with different labeled ratios.

{{</citation>}}


### (34/77) SA$^2$VP: Spatially Aligned-and-Adapted Visual Prompt (Wenjie Pei et al., 2023)

{{<citation>}}

Wenjie Pei, Tongqi Xia, Fanglin Chen, Jinsong Li, Jiandong Tian, Guangming Lu. (2023)  
**SA$^2$VP: Spatially Aligned-and-Adapted Visual Prompt**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2312.10376v1)  

---


**ABSTRACT**  
As a prominent parameter-efficient fine-tuning technique in NLP, prompt tuning is being explored its potential in computer vision. Typical methods for visual prompt tuning follow the sequential modeling paradigm stemming from NLP, which represents an input image as a flattened sequence of token embeddings and then learns a set of unordered parameterized tokens prefixed to the sequence representation as the visual prompts for task adaptation of large vision models. While such sequential modeling paradigm of visual prompt has shown great promise, there are two potential limitations. First, the learned visual prompts cannot model the underlying spatial relations in the input image, which is crucial for image encoding. Second, since all prompt tokens play the same role of prompting for all image tokens without distinction, it lacks the fine-grained prompting capability, i.e., individual prompting for different image tokens. In this work, we propose the \mymodel model (\emph{SA$^2$VP}), which learns a two-dimensional prompt token map with equal (or scaled) size to the image token map, thereby being able to spatially align with the image map. Each prompt token is designated to prompt knowledge only for the spatially corresponding image tokens. As a result, our model can conduct individual prompting for different image tokens in a fine-grained manner. Moreover, benefiting from the capability of preserving the spatial structure by the learned prompt token map, our \emph{SA$^2$VP} is able to model the spatial relations in the input image, leading to more effective prompting. Extensive experiments on three challenging benchmarks for image classification demonstrate the superiority of our model over other state-of-the-art methods for visual prompt tuning. Code is available at \emph{https://github.com/tommy-xq/SA2VP}.

{{</citation>}}


### (35/77) Symmetrical Bidirectional Knowledge Alignment for Zero-Shot Sketch-Based Image Retrieval (Decheng Liu et al., 2023)

{{<citation>}}

Decheng Liu, Xu Luo, Chunlei Peng, Nannan Wang, Ruimin Hu, Xinbo Gao. (2023)  
**Symmetrical Bidirectional Knowledge Alignment for Zero-Shot Sketch-Based Image Retrieval**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Sketch, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2312.10320v1)  

---


**ABSTRACT**  
This paper studies the problem of zero-shot sketch-based image retrieval (ZS-SBIR), which aims to use sketches from unseen categories as queries to match the images of the same category. Due to the large cross-modality discrepancy, ZS-SBIR is still a challenging task and mimics realistic zero-shot scenarios. The key is to leverage transferable knowledge from the pre-trained model to improve generalizability. Existing researchers often utilize the simple fine-tuning training strategy or knowledge distillation from a teacher model with fixed parameters, lacking efficient bidirectional knowledge alignment between student and teacher models simultaneously for better generalization. In this paper, we propose a novel Symmetrical Bidirectional Knowledge Alignment for zero-shot sketch-based image retrieval (SBKA). The symmetrical bidirectional knowledge alignment learning framework is designed to effectively learn mutual rich discriminative information between teacher and student models to achieve the goal of knowledge alignment. Instead of the former one-to-one cross-modality matching in the testing stage, a one-to-many cluster cross-modality matching method is proposed to leverage the inherent relationship of intra-class images to reduce the adverse effects of the existing modality gap. Experiments on several representative ZS-SBIR datasets (Sketchy Ext dataset, TU-Berlin Ext dataset and QuickDraw Ext dataset) prove the proposed algorithm can achieve superior performance compared with state-of-the-art methods.

{{</citation>}}


### (36/77) Mapping Housing Stock Characteristics from Drone Images for Climate Resilience in the Caribbean (Isabelle Tingzon et al., 2023)

{{<citation>}}

Isabelle Tingzon, Nuala Margaret Cowan, Pierre Chrzanowski. (2023)  
**Mapping Housing Stock Characteristics from Drone Images for Climate Resilience in the Caribbean**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Drone  
[Paper Link](http://arxiv.org/abs/2312.10306v1)  

---


**ABSTRACT**  
Comprehensive information on housing stock is crucial for climate adaptation initiatives aiming to reduce the adverse impacts of climate-extreme hazards in high-risk regions like the Caribbean. In this study, we propose a workflow for rapidly generating critical baseline housing stock data using very high-resolution drone images and deep learning techniques. Specifically, our work leverages the Segment Anything Model and convolutional neural networks for the automated generation of building footprints and roof classification maps. By strengthening local capacity within government agencies to leverage AI and Earth Observation-based solutions, this work seeks to improve the climate resilience of the housing sector in small island developing states in the Caribbean.

{{</citation>}}


## cs.CL (11)



### (37/77) Cross-Linguistic Offensive Language Detection: BERT-Based Analysis of Bengali, Assamese, & Bodo Conversational Hateful Content from Social Media (Jhuma Kabir Mim et al., 2023)

{{<citation>}}

Jhuma Kabir Mim, Mourad Oussalah, Akash Singhal. (2023)  
**Cross-Linguistic Offensive Language Detection: BERT-Based Analysis of Bengali, Assamese, & Bodo Conversational Hateful Content from Social Media**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: BERT, Social Media  
[Paper Link](http://arxiv.org/abs/2312.10528v1)  

---


**ABSTRACT**  
In today's age, social media reigns as the paramount communication platform, providing individuals with the avenue to express their conjectures, intellectual propositions, and reflections. Unfortunately, this freedom often comes with a downside as it facilitates the widespread proliferation of hate speech and offensive content, leaving a deleterious impact on our world. Thus, it becomes essential to discern and eradicate such offensive material from the realm of social media. This article delves into the comprehensive results and key revelations from the HASOC-2023 offensive language identification result. The primary emphasis is placed on the meticulous detection of hate speech within the linguistic domains of Bengali, Assamese, and Bodo, forming the framework for Task 4: Annihilate Hates. In this work, we used BERT models, including XML-Roberta, L3-cube, IndicBERT, BenglaBERT, and BanglaHateBERT. The research outcomes were promising and showed that XML-Roberta-lagre performed better than monolingual models in most cases. Our team 'TeamBD' achieved rank 3rd for Task 4 - Assamese, & 5th for Bengali.

{{</citation>}}


### (38/77) Paloma: A Benchmark for Evaluating Language Model Fit (Ian Magnusson et al., 2023)

{{<citation>}}

Ian Magnusson, Akshita Bhagia, Valentin Hofmann, Luca Soldaini, Ananya Harsh Jha, Oyvind Tafjord, Dustin Schwenk, Evan Pete Walsh, Yanai Elazar, Kyle Lo, Dirk Groeneveld, Iz Beltagy, Hannaneh Hajishirzi, Noah A. Smith, Kyle Richardson, Jesse Dodge. (2023)  
**Paloma: A Benchmark for Evaluating Language Model Fit**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model, Perplexity  
[Paper Link](http://arxiv.org/abs/2312.10523v1)  

---


**ABSTRACT**  
Language models (LMs) commonly report perplexity on monolithic data held out from training. Implicitly or explicitly, this data is composed of domains$\unicode{x2013}$varying distributions of language. Rather than assuming perplexity on one distribution extrapolates to others, Perplexity Analysis for Language Model Assessment (Paloma), measures LM fit to 585 text domains, ranging from nytimes.com to r/depression on Reddit. We invite submissions to our benchmark and organize results by comparability based on compliance with guidelines such as removal of benchmark contamination from pretraining. Submissions can also record parameter and training token count to make comparisons of Pareto efficiency for performance as a function of these measures of cost. We populate our benchmark with results from 6 baselines pretrained on popular corpora. In case studies, we demonstrate analyses that are possible with Paloma, such as finding that pretraining without data beyond Common Crawl leads to inconsistent fit to many domains.

{{</citation>}}


### (39/77) When Parameter-efficient Tuning Meets General-purpose Vision-language Models (Yihang Zhai et al., 2023)

{{<citation>}}

Yihang Zhai, Haixin Wang, Jianlong Chang, Xinlong Yang, Jinan Sun, Shikun Zhang, Qi Tian. (2023)  
**When Parameter-efficient Tuning Meets General-purpose Vision-language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.12458v1)  

---


**ABSTRACT**  
Instruction tuning has shown promising potential for developing general-purpose AI capabilities by using large-scale pre-trained models and boosts growing research to integrate multimodal information for creative applications. However, existing works still face two main limitations: the high training costs and heavy computing resource dependence of full model fine-tuning, and the lack of semantic information in instructions, which hinders multimodal alignment. Addressing these challenges, this paper proposes a novel approach to utilize Parameter-Efficient Tuning for generAl-purpose vision-Language models, namely PETAL. PETAL revolutionizes the training process by requiring only 0.5% of the total parameters, achieved through a unique mode approximation technique, which significantly reduces the training costs and reliance on heavy computing resources. Furthermore, PETAL enhances the semantic depth of instructions in two innovative ways: 1) by introducing adaptive instruction mixture-of-experts(MOEs), and 2) by fortifying the score-based linkage between parameter-efficient tuning and mutual information. Our extensive experiments across five multimodal downstream benchmarks reveal that PETAL not only outperforms current state-of-the-art methods in most scenarios but also surpasses full fine-tuning models in effectiveness. Additionally, our approach demonstrates remarkable advantages in few-shot settings, backed by comprehensive visualization analyses. Our source code is available at: https://github. com/melonking32/PETAL.

{{</citation>}}


### (40/77) Debiasing Multimodal Sarcasm Detection with Contrastive Learning (Mengzhao Jia et al., 2023)

{{<citation>}}

Mengzhao Jia, Can Xie, Liqiang Jing. (2023)  
**Debiasing Multimodal Sarcasm Detection with Contrastive Learning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-MM, cs.CL  
Keywords: Contrastive Learning, Sarcasm Detection  
[Paper Link](http://arxiv.org/abs/2312.10493v2)  

---


**ABSTRACT**  
Despite commendable achievements made by existing work, prevailing multimodal sarcasm detection studies rely more on textual content over visual information. It unavoidably induces spurious correlations between textual words and labels, thereby significantly hindering the models' generalization capability. To address this problem, we define the task of out-of-distribution (OOD) multimodal sarcasm detection, which aims to evaluate models' generalizability when the word distribution is different in training and testing settings. Moreover, we propose a novel debiasing multimodal sarcasm detection framework with contrastive learning, which aims to mitigate the harmful effect of biased textual factors for robust OOD generalization. In particular, we first design counterfactual data augmentation to construct the positive samples with dissimilar word biases and negative samples with similar word biases. Subsequently, we devise an adapted debiasing contrastive learning mechanism to empower the model to learn robust task-relevant features and alleviate the adverse effect of biased words. Extensive experiments show the superiority of the proposed framework.

{{</citation>}}


### (41/77) A Soft Contrastive Learning-based Prompt Model for Few-shot Sentiment Analysis (Jingyi Zhou et al., 2023)

{{<citation>}}

Jingyi Zhou, Jie Zhou, Jiabao Zhao, Siyin Wang, Haijun Shan, Gui Tao, Qi Zhang, Xuanjing Huang. (2023)  
**A Soft Contrastive Learning-based Prompt Model for Few-shot Sentiment Analysis**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, Contrastive Learning, GPT, Sentiment Analysis  
[Paper Link](http://arxiv.org/abs/2312.10479v1)  

---


**ABSTRACT**  
Few-shot text classification has attracted great interest in both academia and industry due to the lack of labeled data in many fields. Different from general text classification (e.g., topic classification), few-shot sentiment classification is more challenging because the semantic distances among the classes are more subtle. For instance, the semantic distances between the sentiment labels in a positive or negative polarity (e.g., ``love" and ``joy", ``remorse" and ``sadness") are close, while the distances are large for the sentiment labels in two opposite polarities (e.g., ``love" and ``sadness"). To address this problem, we propose a Soft Contrastive learning-based Prompt (\texttt{SCP}) model for few-shot sentiment analysis. First, we design a sentiment-aware chain of thought prompt module to guide the model to predict the sentiment from coarse grain to fine grain via a series of intermediate reasoning steps. Then, we propose a soft contrastive learning algorithm to take the correlation of the labels into account. A series of experiments on several sentiment analysis datasets show the great advantages of \texttt{SCP} by comparing it with SOTA baselines (e.g., ChatGPT).

{{</citation>}}


### (42/77) RIGHT: Retrieval-augmented Generation for Mainstream Hashtag Recommendation (Run-Ze Fan et al., 2023)

{{<citation>}}

Run-Ze Fan, Yixing Fan, Jiangui Chen, Jiafeng Guo, Ruqing Zhang, Xueqi Cheng. (2023)  
**RIGHT: Retrieval-augmented Generation for Mainstream Hashtag Recommendation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-IR, cs.CL  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2312.10466v1)  

---


**ABSTRACT**  
Automatic mainstream hashtag recommendation aims to accurately provide users with concise and popular topical hashtags before publication. Generally, mainstream hashtag recommendation faces challenges in the comprehensive difficulty of newly posted tweets in response to new topics, and the accurate identification of mainstream hashtags beyond semantic correctness. However, previous retrieval-based methods based on a fixed predefined mainstream hashtag list excel in producing mainstream hashtags, but fail to understand the constant flow of up-to-date information. Conversely, generation-based methods demonstrate a superior ability to comprehend newly posted tweets, but their capacity is constrained to identifying mainstream hashtags without additional features. Inspired by the recent success of the retrieval-augmented technique, in this work, we attempt to adopt this framework to combine the advantages of both approaches. Meantime, with the help of the generator component, we could rethink how to further improve the quality of the retriever component at a low cost. Therefore, we propose RetrIeval-augmented Generative Mainstream HashTag Recommender (RIGHT), which consists of three components: 1) a retriever seeks relevant hashtags from the entire tweet-hashtags set; 2) a selector enhances mainstream identification by introducing global signals; and 3) a generator incorporates input tweets and selected hashtags to directly generate the desired hashtags. The experimental results show that our method achieves significant improvements over state-of-the-art baselines. Moreover, RIGHT can be easily integrated into large language models, improving the performance of ChatGPT by more than 10%.

{{</citation>}}


### (43/77) From Dialogue to Diagram: Task and Relationship Extraction from Natural Language for Accelerated Business Process Prototyping (Sara Qayyum et al., 2023)

{{<citation>}}

Sara Qayyum, Muhammad Moiz Asghar, Muhammad Fouzan Yaseen. (2023)  
**From Dialogue to Diagram: Task and Relationship Extraction from Natural Language for Accelerated Business Process Prototyping**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Dialog, Dialogue, NER, Named Entity Recognition  
[Paper Link](http://arxiv.org/abs/2312.10432v1)  

---


**ABSTRACT**  
The automatic transformation of verbose, natural language descriptions into structured process models remains a challenge of significant complexity - This paper introduces a contemporary solution, where central to our approach, is the use of dependency parsing and Named Entity Recognition (NER) for extracting key elements from textual descriptions. Additionally, we utilize Subject-Verb-Object (SVO) constructs for identifying action relationships and integrate semantic analysis tools, including WordNet, for enriched contextual understanding. A novel aspect of our system is the application of neural coreference resolution, integrated with the SpaCy framework, enhancing the precision of entity linkage and anaphoric references. Furthermore, the system adeptly handles data transformation and visualization, converting extracted information into BPMN (Business Process Model and Notation) diagrams. This methodology not only streamlines the process of capturing and representing business workflows but also significantly reduces the manual effort and potential for error inherent in traditional modeling approaches.

{{</citation>}}


### (44/77) K-ESConv: Knowledge Injection for Emotional Support Dialogue Systems via Prompt Learning (Wei Chen et al., 2023)

{{<citation>}}

Wei Chen, Gang Zhao, Xiaojin Zhang, Xiang Bai, Xuanjing Huang, Zhongyu Wei. (2023)  
**K-ESConv: Knowledge Injection for Emotional Support Dialogue Systems via Prompt Learning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2312.10371v1)  

---


**ABSTRACT**  
Automatic psychological counseling requires mass of professional knowledge that can be found in online counseling forums. Motivated by this, we propose K-ESConv, a novel prompt learning based knowledge injection method for emotional support dialogue system, transferring forum knowledge to response generation. We evaluate our model on an emotional support dataset ESConv, where the model retrieves and incorporates knowledge from external professional emotional Q\&A forum. Experiment results show that the proposed method outperforms existing baselines on both automatic evaluation and human evaluation, which shows that our approach significantly improves the correlation and diversity of responses and provides more comfort and better suggestion for the seeker.

{{</citation>}}


### (45/77) CONCSS: Contrastive-based Context Comprehension for Dialogue-appropriate Prosody in Conversational Speech Synthesis (Yayue Deng et al., 2023)

{{<citation>}}

Yayue Deng, Jinlong Xue, Yukang Jia, Qifei Li, Yichen Han, Fengping Wang, Yingming Gao, Dengfeng Ke, Ya Li. (2023)  
**CONCSS: Contrastive-based Context Comprehension for Dialogue-appropriate Prosody in Conversational Speech Synthesis**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-HC, cs.CL  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2312.10358v1)  

---


**ABSTRACT**  
Conversational speech synthesis (CSS) incorporates historical dialogue as supplementary information with the aim of generating speech that has dialogue-appropriate prosody. While previous methods have already delved into enhancing context comprehension, context representation still lacks effective representation capabilities and context-sensitive discriminability. In this paper, we introduce a contrastive learning-based CSS framework, CONCSS. Within this framework, we define an innovative pretext task specific to CSS that enables the model to perform self-supervised learning on unlabeled conversational datasets to boost the model's context understanding. Additionally, we introduce a sampling strategy for negative sample augmentation to enhance context vectors' discriminability. This is the first attempt to integrate contrastive learning into CSS. We conduct ablation studies on different contrastive learning strategies and comprehensive experiments in comparison with prior CSS systems. Results demonstrate that the synthesized speech from our proposed method exhibits more contextually appropriate and sensitive prosody.

{{</citation>}}


### (46/77) Continuous Prompt Generation from Linear Combination of Discrete Prompt Embeddings (Pascal Passigan et al., 2023)

{{<citation>}}

Pascal Passigan, Kidus Yohannes, Joshua Pereira. (2023)  
**Continuous Prompt Generation from Linear Combination of Discrete Prompt Embeddings**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2312.10323v1)  

---


**ABSTRACT**  
The wayward quality of continuous prompts stresses the importance of their interpretability as unexpected and unpredictable behaviors appear following training, especially in the context of large language models automating people-sensitive tasks such as resume screening. In this paper we present a novel method of constructing continuous prompts via discrete prompt embeddings and evaluate improvements to continuous prompt interpretability and inference accuracy. For a set of manually designed discrete prompts $\mathcal{D}$, which we tokenize each into tensor form, we train a model to predict the weights such that the linear combinations of those prompts correspond to higher performance on natural language understanding tasks.

{{</citation>}}


### (47/77) One Shot Learning as Instruction Data Prospector for Large Language Models (Yunshui Li et al., 2023)

{{<citation>}}

Yunshui Li, Binyuan Hui, Xiaobo Xia, Jiaxi Yang, Min Yang, Lei Zhang, Shuzheng Si, Junhao Liu, Tongliang Liu, Fei Huang, Yongbin Li. (2023)  
**One Shot Learning as Instruction Data Prospector for Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.10302v2)  

---


**ABSTRACT**  
Aligning large language models(LLMs) with human is a critical step in effectively utilizing their pre-trained capabilities across a wide array of language tasks. Current instruction tuning practices often rely on expanding dataset size without a clear strategy for ensuring data quality, which can inadvertently introduce noise and degrade model performance. To address this challenge, we introduce Nuggets, a novel and efficient methodology that employs one shot learning to select high-quality instruction data from expansive datasets. Nuggets assesses the potential of individual instruction examples to act as effective one shot examples, thereby identifying those that can significantly enhance diverse task performance. Nuggets utilizes a scoring system based on the impact of candidate examples on the perplexity of a diverse anchor set, facilitating the selection of the most beneficial data for instruction tuning. Through rigorous testing on two benchmarks, including MT-Bench and Alpaca-Eval, we demonstrate that instruction tuning with the top 1% of Nuggets-curated examples substantially outperforms conventional methods that use the full dataset. These findings advocate for a data selection paradigm that prioritizes quality, offering a more efficient pathway to align LLMs with humans.

{{</citation>}}


## cs.HC (3)



### (48/77) Democratize with Care: The need for fairness specific features in user-interface based open source AutoML tools (Sundaraparipurnan Narayanan, 2023)

{{<citation>}}

Sundaraparipurnan Narayanan. (2023)  
**Democratize with Care: The need for fairness specific features in user-interface based open source AutoML tools**  

---
Primary Category: cs.HC  
Categories: cs-CY, cs-HC, cs-LG, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.12460v1)  

---


**ABSTRACT**  
AI is increasingly playing a pivotal role in businesses and organizations, impacting the outcomes and interests of human users. Automated Machine Learning (AutoML) streamlines the machine learning model development process by automating repetitive tasks and making data-driven decisions, enabling even non-experts to construct high-quality models efficiently. This democratization allows more users (including non-experts) to access and utilize state-of-the-art machine-learning expertise. However, AutoML tools may also propagate bias in the way these tools handle the data, model choices, and optimization approaches adopted. We conducted an experimental study of User-interface-based open source AutoML tools (DataRobot, H2O Studio, Dataiku, and Rapidminer Studio) to examine if they had features to assist users in developing fairness-aware machine learning models. The experiments covered the following considerations for the evaluation of features: understanding use case context, data representation, feature relevance and sensitivity, data bias and preprocessing techniques, data handling capabilities, training-testing split, hyperparameter handling, and constraints, fairness-oriented model development, explainability and ability to download and edit models by the user. The results revealed inadequacies in features that could support in fairness-aware model development. Further, the results also highlight the need to establish certain essential features for promoting fairness in AutoML tools.

{{</citation>}}


### (49/77) Let AI Entertain You: Increasing User Engagement with Generative AI and Rejection Sampling (Jingying Zeng et al., 2023)

{{<citation>}}

Jingying Zeng, Jaewon Yang, Waleed Malik, Xiao Yan, Richard Huang, Qi He. (2023)  
**Let AI Entertain You: Increasing User Engagement with Generative AI and Rejection Sampling**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs-LG, cs.HC  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2312.12457v1)  

---


**ABSTRACT**  
While generative AI excels in content generation, it does not always increase user engagement. This can be attributed to two main factors. First, generative AI generates content without incorporating explicit or implicit feedback about user interactions. Even if the generated content seems to be more informative or well-written, it does not necessarily lead to an increase in user activities, such as clicks. Second, there is a concern with the quality of the content generative AI produces, which often lacks the distinctiveness and authenticity that human-created content possesses. These two factors can lead to content that fails to meet specific needs and preferences of users, ultimately reducing its potential to be engaging.   This paper presents a generic framework of how to improve user engagement with generative AI by leveraging user feedback. Our solutions employ rejection sampling, a technique used in reinforcement learning, to boost engagement metrics. We leveraged the framework in the context of email notification subject lines generation for an online social network, and achieved significant engagement metric lift including +1% Session and +0.4% Weekly Active Users. We believe our work offers a universal framework that enhances user engagement with generative AI, particularly when standard generative AI reaches its limits in terms of enhancing content to be more captivating. To the best of our knowledge, this represents an early milestone in the industry's successful use of generative AI to enhance user engagement.

{{</citation>}}


### (50/77) Building AI and Human Capital for Road Safety (Yug Dedhia et al., 2023)

{{<citation>}}

Yug Dedhia, Anjali Singh, Vaibhav Singh Tomar, Nimmi Rangaswamy, Dev Singh Thakur. (2023)  
**Building AI and Human Capital for Road Safety**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.10319v1)  

---


**ABSTRACT**  
AI is about learning algorithms and huge amounts of data and are drivers of economic growth -- what does this mean for the field of development studies? Can we re-orient to twin AI studies and development theory and practice to generate how development challenges are identified and researched? To do this a good grasp is needed of AI internal mechanisms and outcomes in addressing development issues -- this argument will be developed through a case study of the ADAS [Advanced Driver Assistance System] deployment in India. Over and above discussing the ADAS we bring an anthropological lens to understand the social context that surrounds the system. Focusing on bus drivers, we offer findings from a qualitative and ethnographic study of drivers in a collaborative effort to achieve road safety by deploying AI-driven technology and empowering stakeholders in the transport industry in India especially, bus drivers as critical actors in the city transport network.

{{</citation>}}


## cs.SE (4)



### (51/77) Comprehensive Evaluation of ChatGPT Reliability Through Multilingual Inquiries (Poorna Chander Reddy Puttaparthi et al., 2023)

{{<citation>}}

Poorna Chander Reddy Puttaparthi, Soham Sanjay Deo, Hakan Gul, Yiming Tang, Weiyi Shang, Zhe Yu. (2023)  
**Comprehensive Evaluation of ChatGPT Reliability Through Multilingual Inquiries**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: AI, ChatGPT, GPT, Multilingual  
[Paper Link](http://arxiv.org/abs/2312.10524v1)  

---


**ABSTRACT**  
ChatGPT is currently the most popular large language model (LLM), with over 100 million users, making a significant impact on people's lives. However, due to the presence of jailbreak vulnerabilities, ChatGPT might have negative effects on people's lives, potentially even facilitating criminal activities. Testing whether ChatGPT can cause jailbreak is crucial because it can enhance ChatGPT's security, reliability, and social responsibility. Inspired by previous research revealing the varied performance of LLMs in different language translations, we suspected that wrapping prompts in multiple languages might lead to ChatGPT jailbreak. To investigate this, we designed a study with a fuzzing testing approach to analyzing ChatGPT's cross-linguistic proficiency. Our study includes three strategies by automatically posing different formats of malicious questions to ChatGPT: (1) each malicious question involving only one language, (2) multilingual malicious questions, (3) specifying that ChatGPT responds in a language different from the prompts. In addition, we also combine our strategies by utilizing prompt injection templates to wrap the three aforementioned types of questions. We examined a total of 7,892 Q&A data points, discovering that multilingual wrapping can indeed lead to ChatGPT's jailbreak, with different wrapping methods having varying effects on jailbreak probability. Prompt injection can amplify the probability of jailbreak caused by multilingual wrapping. This work provides insights for OpenAI developers to enhance ChatGPT's support for language diversity and inclusion.

{{</citation>}}


### (52/77) Resolving Crash Bugs via Large Language Models: An Empirical Study (Xueying Du et al., 2023)

{{<citation>}}

Xueying Du, Mingwei Liu, Juntao Li, Hanlin Wang, Xin Peng, Yiling Lou. (2023)  
**Resolving Crash Bugs via Large Language Models: An Empirical Study**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-CL, cs-SE, cs.SE  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2312.10448v1)  

---


**ABSTRACT**  
Crash bugs cause unexpected program behaviors or even termination, requiring high-priority resolution. However, manually resolving crash bugs is challenging and labor-intensive, and researchers have proposed various techniques for their automated localization and repair. ChatGPT, a recent large language model (LLM), has garnered significant attention due to its exceptional performance across various domains. This work performs the first investigation into ChatGPT's capability in resolve real-world crash bugs, focusing on its effectiveness in both localizing and repairing code-related and environment-related crash bugs. Specifically, we initially assess ChatGPT's fundamental ability to resolve crash bugs with basic prompts in a single iteration. We observe that ChatGPT performs better at resolving code-related crash bugs compared to environment-related ones, and its primary challenge in resolution lies in inaccurate localization. Additionally, we explore ChatGPT's potential with various advanced prompts. Furthermore, by stimulating ChatGPT's self-planning, it methodically investigates each potential crash-causing environmental factor through proactive inquiry, ultimately identifying the root cause of the crash. Based on our findings, we propose IntDiagSolver, an interaction methodology designed to facilitate precise crash bug resolution through continuous interaction with LLMs. Evaluating IntDiagSolver on multiple LLMs reveals consistent enhancement in the accuracy of crash bug resolution, including ChatGPT, Claude, and CodeLlama.

{{</citation>}}


### (53/77) A Comparative Analysis of Large Language Models for Code Documentation Generation (Shubhang Shekhar Dvivedi et al., 2023)

{{<citation>}}

Shubhang Shekhar Dvivedi, Vyshnav Vijay, Sai Leela Rahul Pujari, Shoumik Lodh, Dhruv Kumar. (2023)  
**A Comparative Analysis of Large Language Models for Code Documentation Generation**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: ChatGPT, GPT, GPT-3.5, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2312.10349v1)  

---


**ABSTRACT**  
This paper presents a comprehensive comparative analysis of Large Language Models (LLMs) for generation of code documentation. Code documentation is an essential part of the software writing process. The paper evaluates models such as GPT-3.5, GPT-4, Bard, Llama2, and Starchat on various parameters like Accuracy, Completeness, Relevance, Understandability, Readability and Time Taken for different levels of code documentation. Our evaluation employs a checklist-based system to minimize subjectivity, providing a more objective assessment. We find that, barring Starchat, all LLMs consistently outperform the original documentation. Notably, closed-source models GPT-3.5, GPT-4, and Bard exhibit superior performance across various parameters compared to open-source/source-available LLMs, namely LLama 2 and StarChat. Considering the time taken for generation, GPT-4 demonstrated the longest duration, followed by Llama2, Bard, with ChatGPT and Starchat having comparable generation times. Additionally, file level documentation had a considerably worse performance across all parameters (except for time taken) as compared to inline and function level documentation.

{{</citation>}}


### (54/77) Shedding Light on Software Engineering-specific Metaphors and Idioms (Mia Mohammad Imran et al., 2023)

{{<citation>}}

Mia Mohammad Imran, Preetha Chatterjee, Kostadin Damevski. (2023)  
**Shedding Light on Software Engineering-specific Metaphors and Idioms**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.10297v1)  

---


**ABSTRACT**  
Use of figurative language, such as metaphors and idioms, is common in our daily-life communications, and it can also be found in Software Engineering (SE) channels, such as comments on GitHub. Automatically interpreting figurative language is a challenging task, even with modern Large Language Models (LLMs), as it often involves subtle nuances. This is particularly true in the SE domain, where figurative language is frequently used to convey technical concepts, often bearing developer affect (e.g., `spaghetti code'). Surprisingly, there is a lack of studies on how figurative language in SE communications impacts the performance of automatic tools that focus on understanding developer communications, e.g., bug prioritization, incivility detection. Furthermore, it is an open question to what extent state-of-the-art LLMs interpret figurative expressions in domain-specific communication such as software engineering. To address this gap, we study the prevalence and impact of figurative language in SE communication channels. This study contributes to understanding the role of figurative language in SE, the potential of LLMs in interpreting them, and its impact on automated SE communication analysis. Our results demonstrate the effectiveness of fine-tuning LLMs with figurative language in SE and its potential impact on automated tasks that involve affect. We found that, among three state-of-the-art LLMs, the best improved fine-tuned versions have an average improvement of 6.66% on a GitHub emotion classification dataset, 7.07% on a GitHub incivility classification dataset, and 3.71% on a Bugzilla bug report prioritization dataset.

{{</citation>}}


## cs.NI (1)



### (55/77) Value of Information and Timing-aware Scheduling for Federated Learning (Muhammad Azeem Khan et al., 2023)

{{<citation>}}

Muhammad Azeem Khan, Howard H. Yang, Zihan Chen, Antonio Iera, Nikolaos Pappas. (2023)  
**Value of Information and Timing-aware Scheduling for Federated Learning**  

---
Primary Category: cs.NI  
Categories: cs-LG, cs-NI, cs.NI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.10512v1)  

---


**ABSTRACT**  
Data possesses significant value as it fuels advancements in AI. However, protecting the privacy of the data generated by end-user devices has become crucial. Federated Learning (FL) offers a solution by preserving data privacy during training. FL brings the model directly to User Equipments (UEs) for local training by an access point (AP). The AP periodically aggregates trained parameters from UEs, enhancing the model and sending it back to them. However, due to communication constraints, only a subset of UEs can update parameters during each global aggregation. Consequently, developing innovative scheduling algorithms is vital to enable complete FL implementation and enhance FL convergence. In this paper, we present a scheduling policy combining Age of Update (AoU) concepts and data Shapley metrics. This policy considers the freshness and value of received parameter updates from individual data sources and real-time channel conditions to enhance FL's operational efficiency. The proposed algorithm is simple, and its effectiveness is demonstrated through simulations.

{{</citation>}}


## cs.SI (3)



### (56/77) SubAnom: Efficient Subgraph Anomaly Detection Framework over Dynamic Graphs (Chi Zhang et al., 2023)

{{<citation>}}

Chi Zhang, Wenkai Xiang, Xingzhi Guo, Baojian Zhou, Deqing Yang. (2023)  
**SubAnom: Efficient Subgraph Anomaly Detection Framework over Dynamic Graphs**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2312.10504v1)  

---


**ABSTRACT**  
Given a dynamic graph, the efficient tracking of anomalous subgraphs via their node embeddings poses a significant challenge. Addressing this issue necessitates an effective scoring mechanism and an innovative anomalous subgraph strategy. Existing methods predominantly focus on designing scoring strategies or employing graph structures that consider nodes in isolation, resulting in ineffective capture of the anomalous subgraph structure information.   In this paper, we introduce SUBANOM, a novel framework for subgraph anomaly detection that is adept at identifying anomalous subgraphs. SUBANOM has three key components: 1) We implement current state-of-the-art dynamic embedding methods to efficiently calculate node embeddings, thereby capturing all node-level anomalies successfully; 2) We devise novel subgraph identification strategies, which include k-hop and triadic-closure. These strategies form the crucial component that can proficiently differentiate between strong and weak neighbors, thus effectively capturing the anomaly structure information; 3) For qualifying the anomaly subgraphs, we propose using Lp-norm-based score aggregation functions. These iterative steps enable us to process large-scale dynamic graphs effectively.   Experiments conducted on a real-world dynamic graph underscore the efficacy of our framework in detecting anomalous subgraphs, outperforming state-of-the-art methods. Experimental results further signify that our framework is a potent tool for identifying anomalous subgraphs in real-world scenarios. For instance, the F1 score under the optimal subgraph identification strategy, can peak at 0.6679, while the highest achievable score using the corresponding baseline method is 0.5677.

{{</citation>}}


### (57/77) Analysis of Nepotism in Bollywood using Personalized PageRank and Effective Influence (Apoorv Jain, 2023)

{{<citation>}}

Apoorv Jain. (2023)  
**Analysis of Nepotism in Bollywood using Personalized PageRank and Effective Influence**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.11544v1)  

---


**ABSTRACT**  
Bollywood is one of the largest film-producing industries with a large worldwide audience. In this paper, we will try to find the most important stars in the era of 1990 to 2014, as well as try to use social network analysis methods and metrics to analyze the role of blood connections in getting opportunities in the industry. We created the actor relationship data of around 1000 debutants using OpenAI API and used a novel approach "Effective Influence" to study the effect of having a blood-related established actor inside the industry. We found that on an average every actor/director or actor/actor pair is reachable by a path of length at most 4 and a correlation of 0.6 indicating the advantage of having a blood connection inside the network in getting a good co-cast in the debut film.

{{</citation>}}


### (58/77) The DSA Transparency Database: Auditing Self-reported Moderation Actions by Social Media (Amaury Trujillo et al., 2023)

{{<citation>}}

Amaury Trujillo, Tiziano Fagni, Stefano Cresci. (2023)  
**The DSA Transparency Database: Auditing Self-reported Moderation Actions by Social Media**  

---
Primary Category: cs.SI  
Categories: cs-AI, cs-CY, cs-HC, cs-SI, cs.SI  
Keywords: Social Media, Twitter  
[Paper Link](http://arxiv.org/abs/2312.10269v1)  

---


**ABSTRACT**  
Since September 2023, the Digital Services Act (DSA) obliges large online platforms to submit detailed data on each moderation action they take within the European Union (EU) to the DSA Transparency Database. From its inception, this centralized database has sparked scholarly interest as an unprecedented and potentially unique trove of data on real-world online moderation. Here, we thoroughly analyze all 195.61M records submitted by the eight largest social media platforms in the EU during the first 60 days of the database. Specifically, we conduct a platform-wise comparative study of their: volume of moderation actions, grounds for decision, types of applied restrictions, types of moderated content, timeliness in undertaking and submitting moderation actions, and use of automation. Furthermore, we systematically cross-check the contents of the database with the platforms' own transparency reports. Our analyses reveal that (i) the platforms adhered only in part to the philosophy and structure of the database, (ii) the structure of the database is partially inadequate for the platforms' reporting needs, (iii) the platforms exhibited substantial differences in their moderation actions, (iv) a remarkable fraction of the database data is inconsistent, (v) the platform X (formerly Twitter) presents the most inconsistencies. Our findings have far-reaching implications for policymakers and scholars across diverse disciplines. They offer guidance for future regulations that cater to the reporting needs of online platforms in general, but also highlight opportunities to improve and refine the database itself.

{{</citation>}}


## eess.IV (1)



### (59/77) All Attention U-NET for Semantic Segmentation of Intracranial Hemorrhages In Head CT Images (Chia Shuo Chang et al., 2023)

{{<citation>}}

Chia Shuo Chang, Tian Sheuan Chang, Jiun Lin Yan, Li Ko. (2023)  
**All Attention U-NET for Semantic Segmentation of Intracranial Hemorrhages In Head CT Images**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Attention, Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2312.10483v1)  

---


**ABSTRACT**  
Intracranial hemorrhages in head CT scans serve as a first line tool to help specialists diagnose different types. However, their types have diverse shapes in the same type but similar confusing shape, size and location between types. To solve this problem, this paper proposes an all attention U-Net. It uses channel attentions in the U-Net encoder side to enhance class specific feature extraction, and space and channel attentions in the U-Net decoder side for more accurate shape extraction and type classification. The simulation results show up to a 31.8\% improvement compared to baseline, ResNet50 + U-Net, and better performance than in cases with limited attention.

{{</citation>}}


## econ.GN (1)



### (60/77) Sails and Anchors: The Complementarity of Exploratory and Exploitative Scientists in Knowledge Creation (Pierre Pelletier et al., 2023)

{{<citation>}}

Pierre Pelletier, Kevin Wirtz. (2023)  
**Sails and Anchors: The Complementarity of Exploratory and Exploitative Scientists in Knowledge Creation**  

---
Primary Category: econ.GN  
Categories: cs-DL, econ-GN, econ.GN, q-fin-EC  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2312.10476v1)  

---


**ABSTRACT**  
This paper investigates the relationship between scientists' cognitive profile and their ability to generate innovative ideas and gain scientific recognition. We propose a novel author-level metric based on the semantic representation of researchers' past publications to measure cognitive diversity both at individual and team levels. Using PubMed Knowledge Graph (PKG), we analyze the impact of cognitive diversity on novelty, as measured by combinatorial novelty indicators and peer labels on Faculty Opinion. We assessed scientific impact through citations and disruption indicators. We show that the presence of exploratory individuals (i.e., cognitively diverse) is beneficial in generating distant knowledge combinations, but only when balanced by a significant proportion of exploitative individuals (i.e., cognitively specialized). Furthermore, teams with a high proportion of exploitative profiles tend to consolidate science, whereas those with a significant share of both profiles tend to disrupt it. Cognitive diversity between team members appears to be always beneficial to combining more distant knowledge. However, to maximize the relevance of these distant combinations of knowledge, maintaining a limited number of exploratory individuals is essential, as exploitative individuals must question and debate their novel perspectives. These specialized individuals are the most qualified to extract the full potential of novel ideas and integrate them within the existing scientific paradigm.

{{</citation>}}


## cs.IR (3)



### (61/77) RecPrompt: A Prompt Tuning Framework for News Recommendation Using Large Language Models (Dairui Liu et al., 2023)

{{<citation>}}

Dairui Liu, Boming Yang, Honghui Du, Derek Greene, Aonghus Lawlor, Ruihai Dong, Irene Li. (2023)  
**RecPrompt: A Prompt Tuning Framework for News Recommendation Using Large Language Models**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2312.10463v1)  

---


**ABSTRACT**  
In the evolving field of personalized news recommendation, understanding the semantics of the underlying data is crucial. Large Language Models (LLMs) like GPT-4 have shown promising performance in understanding natural language. However, the extent of their applicability in news recommendation systems remains to be validated. This paper introduces RecPrompt, the first framework for news recommendation that leverages the capabilities of LLMs through prompt engineering. This system incorporates a prompt optimizer that applies an iterative bootstrapping process, enhancing the LLM-based recommender's ability to align news content with user preferences and interests more effectively. Moreover, this study offers insights into the effective use of LLMs in news recommendation, emphasizing both the advantages and the challenges of incorporating LLMs into recommendation systems.

{{</citation>}}


### (62/77) ProTIP: Progressive Tool Retrieval Improves Planning (Raviteja Anantha et al., 2023)

{{<citation>}}

Raviteja Anantha, Bortik Bandyopadhyay, Anirudh Kashi, Sayantan Mahinder, Andrew W Hill, Srinivas Chappidi. (2023)  
**ProTIP: Progressive Tool Retrieval Improves Planning**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs-LG, cs.IR  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2312.10332v1)  

---


**ABSTRACT**  
Large language models (LLMs) are increasingly employed for complex multi-step planning tasks, where the tool retrieval (TR) step is crucial for achieving successful outcomes. Two prevalent approaches for TR are single-step retrieval, which utilizes the complete query, and sequential retrieval using task decomposition (TD), where a full query is segmented into discrete atomic subtasks. While single-step retrieval lacks the flexibility to handle "inter-tool dependency," the TD approach necessitates maintaining "subtask-tool atomicity alignment," as the toolbox can evolve dynamically. To address these limitations, we introduce the Progressive Tool retrieval to Improve Planning (ProTIP) framework. ProTIP is a lightweight, contrastive learning-based framework that implicitly performs TD without the explicit requirement of subtask labels, while simultaneously maintaining subtask-tool atomicity. On the ToolBench dataset, ProTIP outperforms the ChatGPT task decomposition-based approach by a remarkable margin, achieving a 24% improvement in Recall@K=10 for TR and a 41% enhancement in tool accuracy for plan generation.

{{</citation>}}


### (63/77) Perturbation-Invariant Adversarial Training for Neural Ranking Models: Improving the Effectiveness-Robustness Trade-Off (Yu-An Liu et al., 2023)

{{<citation>}}

Yu-An Liu, Ruqing Zhang, Mingkun Zhang, Wei Chen, Maarten de Rijke, Jiafeng Guo, Xueqi Cheng. (2023)  
**Perturbation-Invariant Adversarial Training for Neural Ranking Models: Improving the Effectiveness-Robustness Trade-Off**  

---
Primary Category: cs.IR  
Categories: cs-CL, cs-IR, cs.IR  
Keywords: Adversarial Training  
[Paper Link](http://arxiv.org/abs/2312.10329v1)  

---


**ABSTRACT**  
Neural ranking models (NRMs) have shown great success in information retrieval (IR). But their predictions can easily be manipulated using adversarial examples, which are crafted by adding imperceptible perturbations to legitimate documents. This vulnerability raises significant concerns about their reliability and hinders the widespread deployment of NRMs. By incorporating adversarial examples into training data, adversarial training has become the de facto defense approach to adversarial attacks against NRMs. However, this defense mechanism is subject to a trade-off between effectiveness and adversarial robustness. In this study, we establish theoretical guarantees regarding the effectiveness-robustness trade-off in NRMs. We decompose the robust ranking error into two components, i.e., a natural ranking error for effectiveness evaluation and a boundary ranking error for assessing adversarial robustness. Then, we define the perturbation invariance of a ranking model and prove it to be a differentiable upper bound on the boundary ranking error for attainable computation. Informed by our theoretical analysis, we design a novel \emph{perturbation-invariant adversarial training} (PIAT) method for ranking models to achieve a better effectiveness-robustness trade-off. We design a regularized surrogate loss, in which one term encourages the effectiveness to be maximized while the regularization term encourages the output to be smooth, so as to improve adversarial robustness. Experimental results on several ranking models demonstrate the superiority of PITA compared to existing adversarial defenses.

{{</citation>}}


## cs.AI (5)



### (64/77) A Unified Pre-training and Adaptation Framework for Combinatorial Optimization on Graphs (Ruibin Zeng et al., 2023)

{{<citation>}}

Ruibin Zeng, Minglong Lei, Lingfeng Niu, Lan Cheng. (2023)  
**A Unified Pre-training and Adaptation Framework for Combinatorial Optimization on Graphs**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2312.11547v1)  

---


**ABSTRACT**  
Combinatorial optimization (CO) on graphs is a classic topic that has been extensively studied across many scientific and industrial fields. Recently, solving CO problems on graphs through learning methods has attracted great attention. Advanced deep learning methods, e.g., graph neural networks (GNNs), have been used to effectively assist the process of solving COs. However, current frameworks based on GNNs are mainly designed for certain CO problems, thereby failing to consider their transferable and generalizable abilities among different COs on graphs. Moreover, simply using original graphs to model COs only captures the direct correlations among objects, which does not consider the mathematical logicality and properties of COs. In this paper, we propose a unified pre-training and adaptation framework for COs on graphs with the help of the maximum satisfiability (Max-SAT) problem. We first use Max-SAT to bridge different COs on graphs since they can be converted to Max-SAT problems represented by standard formulas and clauses with logical information. Then, we further design a pre-training and domain adaptation framework to extract the transferable and generalizable features so that different COs can benefit from them. In the pre-training stage, Max-SAT instances are generated to initialize the parameters of the model. In the fine-tuning stage, instances from CO and Max-SAT problems are used for adaptation so that the transferable ability can be further improved. Numerical experiments on several datasets show that features extracted by our framework exhibit superior transferability and Max-SAT can boost the ability to solve COs on graphs.

{{</citation>}}


### (65/77) M2ConceptBase: A Fine-grained Aligned Multi-modal Conceptual Knowledge Base (Zhiwei Zha et al., 2023)

{{<citation>}}

Zhiwei Zha, Jiaan Wang, Zhixu Li, Xiangru Zhu, Wei Song, Yanghua Xiao. (2023)  
**M2ConceptBase: A Fine-grained Aligned Multi-modal Conceptual Knowledge Base**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2312.10417v1)  

---


**ABSTRACT**  
Large multi-modal models (LMMs) have demonstrated promising intelligence owing to the rapid development of pre-training techniques. However, their fine-grained cross-modal alignment ability is constrained by the coarse alignment in image-text pairs. This limitation hinders awareness of fine-grained concepts, resulting in sub-optimal performance. In this paper, we propose a multi-modal conceptual knowledge base, named M2ConceptBase, which aims to provide fine-grained alignment between images and concepts. Specifically, M2ConceptBase models concepts as nodes, associating each with relevant images and detailed text, thereby enhancing LMMs' cross-modal alignment with rich conceptual knowledge. To collect concept-image and concept-description alignments, we propose a context-aware multi-modal symbol grounding approach that considers context information in existing large-scale image-text pairs with respect to each concept. A cutting-edge large language model supplements descriptions for concepts not grounded via our symbol grounding approach. Finally, our M2ConceptBase contains more than 951K images and 152K concepts, each associating with an average of 6.27 images and a single detailed description. We conduct experiments on the OK-VQA task, demonstrating that our M2ConceptBase facilitates the model in achieving state-of-the-art performance. Moreover, we construct a comprehensive benchmark to evaluate the concept understanding of LMMs and show that M2ConceptBase could effectively improve LMMs' concept understanding and cross-modal alignment abilities.

{{</citation>}}


### (66/77) When Graph Data Meets Multimodal: A New Paradigm for Graph Understanding and Reasoning (Qihang Ai et al., 2023)

{{<citation>}}

Qihang Ai, Jianwu Zhou, Haiyun Jiang, Lemao Liu, Shuming Shi. (2023)  
**When Graph Data Meets Multimodal: A New Paradigm for Graph Understanding and Reasoning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: GPT, GPT-4, OCR, Reasoning  
[Paper Link](http://arxiv.org/abs/2312.10372v1)  

---


**ABSTRACT**  
Graph data is ubiquitous in the physical world, and it has always been a challenge to efficiently model graph structures using a unified paradigm for the understanding and reasoning on various graphs. Moreover, in the era of large language models, integrating complex graph information into text sequences has become exceptionally difficult, which hinders the ability to interact with graph data through natural language instructions.The paper presents a new paradigm for understanding and reasoning about graph data by integrating image encoding and multimodal technologies. This approach enables the comprehension of graph data through an instruction-response format, utilizing GPT-4V's advanced capabilities. The study evaluates this paradigm on various graph types, highlighting the model's strengths and weaknesses, particularly in Chinese OCR performance and complex reasoning tasks. The findings suggest new direction for enhancing graph data processing and natural language interaction.

{{</citation>}}


### (67/77) Do Similar Entities have Similar Embeddings? (Nicolas Hubert et al., 2023)

{{<citation>}}

Nicolas Hubert, Heiko Paulheim, Armelle Brun, Davy Monticolo. (2023)  
**Do Similar Entities have Similar Embeddings?**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-IR, cs-LG, cs.AI  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2312.10370v1)  

---


**ABSTRACT**  
Knowledge graph embedding models (KGEMs) developed for link prediction learn vector representations for graph entities, known as embeddings. A common tacit assumption is the KGE entity similarity assumption, which states that these KGEMs retain the graph's structure within their embedding space, i.e., position similar entities close to one another. This desirable property make KGEMs widely used in downstream tasks such as recommender systems or drug repurposing. Yet, the alignment of graph similarity with embedding space similarity has rarely been formally evaluated. Typically, KGEMs are assessed based on their sole link prediction capabilities, using ranked-based metrics such as Hits@K or Mean Rank. This paper challenges the prevailing assumption that entity similarity in the graph is inherently mirrored in the embedding space. Therefore, we conduct extensive experiments to measure the capability of KGEMs to cluster similar entities together, and investigate the nature of the underlying factors. Moreover, we study if different KGEMs expose a different notion of similarity. Datasets, pre-trained embeddings and code are available at: https://github.com/nicolas-hbt/similar-embeddings.

{{</citation>}}


### (68/77) CLIPSyntel: CLIP and LLM Synergy for Multimodal Question Summarization in Healthcare (Akash Ghosh et al., 2023)

{{<citation>}}

Akash Ghosh, Arkadeep Acharya, Raghav Jain, Sriparna Saha, Aman Chadha, Setu Sinha. (2023)  
**CLIPSyntel: CLIP and LLM Synergy for Multimodal Question Summarization in Healthcare**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: Language Model, Summarization  
[Paper Link](http://arxiv.org/abs/2312.11541v1)  

---


**ABSTRACT**  
In the era of modern healthcare, swiftly generating medical question summaries is crucial for informed and timely patient care. Despite the increasing complexity and volume of medical data, existing studies have focused solely on text-based summarization, neglecting the integration of visual information. Recognizing the untapped potential of combining textual queries with visual representations of medical conditions, we introduce the Multimodal Medical Question Summarization (MMQS) Dataset. This dataset, a major contribution to our work, pairs medical queries with visual aids, facilitating a richer and more nuanced understanding of patient needs. We also propose a framework, utilizing the power of Contrastive Language Image Pretraining(CLIP) and Large Language Models(LLMs), consisting of four modules that identify medical disorders, generate relevant context, filter medical concepts, and craft visually aware summaries. Our comprehensive framework harnesses the power of CLIP, a multimodal foundation model, and various general-purpose LLMs, comprising four main modules: the medical disorder identification module, the relevant context generation module, the context filtration module for distilling relevant medical concepts and knowledge, and finally, a general-purpose LLM to generate visually aware medical question summaries. Leveraging our MMQS dataset, we showcase how visual cues from images enhance the generation of medically nuanced summaries. This multimodal approach not only enhances the decision-making process in healthcare but also fosters a more nuanced understanding of patient queries, laying the groundwork for future research in personalized and responsive medical care

{{</citation>}}


## physics.geo-ph (1)



### (69/77) ResoNet: Robust and Explainable ENSO Forecasts with Hybrid Convolution and Transformer Networks (Pumeng Lyu et al., 2023)

{{<citation>}}

Pumeng Lyu, Tao Tang, Fenghua Ling, Jing-Jia Luo, Niklas Boers, Wanli Ouyang, Lei Bai. (2023)  
**ResoNet: Robust and Explainable ENSO Forecasts with Hybrid Convolution and Transformer Networks**  

---
Primary Category: physics.geo-ph  
Categories: cs-AI, physics-geo-ph, physics.geo-ph  
Keywords: AI, Transformer  
[Paper Link](http://arxiv.org/abs/2312.10429v1)  

---


**ABSTRACT**  
Recent studies have shown that deep learning (DL) models can skillfully predict the El Ni\~no-Southern Oscillation (ENSO) forecasts over 1.5 years ahead. However, concerns regarding the reliability of predictions made by DL methods persist, including potential overfitting issues and lack of interpretability. Here, we propose ResoNet, a DL model that combines convolutional neural network (CNN) and Transformer architectures. This hybrid architecture design enables our model to adequately capture local SSTA as well as long-range inter-basin interactions across oceans. We show that ResoNet can robustly predict ESNO at lead times between 19 and 26 months, thus outperforming existing approaches in terms of the forecast horizon. According to an explainability method applied to ResoNet predictions of El Ni\~no and La Ni\~na events from 1- to 18-month lead, we find that it predicts the Ni\~no3.4 index based on multiple physically reasonable mechanisms, such as the Recharge Oscillator concept, Seasonal Footprint Mechanism, and Indian Ocean capacitor effect. Moreover, we demonstrate that for the first time, the asymmetry between El Ni\~no and La Ni\~na development can be captured by ResoNet. Our results could help alleviate skepticism about applying DL models for ENSO prediction and encourage more attempts to discover and predict climate phenomena using AI methods.

{{</citation>}}


## cs.MA (1)



### (70/77) Robust Communicative Multi-Agent Reinforcement Learning with Active Defense (Lebin Yu et al., 2023)

{{<citation>}}

Lebin Yu, Yunbo Qiu, Quanming Yao, Yuan Shen, Xudong Zhang, Jian Wang. (2023)  
**Robust Communicative Multi-Agent Reinforcement Learning with Active Defense**  

---
Primary Category: cs.MA  
Categories: cs-AI, cs-LG, cs-MA, cs.MA  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.11545v1)  

---


**ABSTRACT**  
Communication in multi-agent reinforcement learning (MARL) has been proven to effectively promote cooperation among agents recently. Since communication in real-world scenarios is vulnerable to noises and adversarial attacks, it is crucial to develop robust communicative MARL technique. However, existing research in this domain has predominantly focused on passive defense strategies, where agents receive all messages equally, making it hard to balance performance and robustness. We propose an active defense strategy, where agents automatically reduce the impact of potentially harmful messages on the final decision. There are two challenges to implement this strategy, that are defining unreliable messages and adjusting the unreliable messages' impact on the final decision properly. To address them, we design an Active Defense Multi-Agent Communication framework (ADMAC), which estimates the reliability of received messages and adjusts their impact on the final decision accordingly with the help of a decomposable decision structure. The superiority of ADMAC over existing methods is validated by experiments in three communication-critical tasks under four types of attacks.

{{</citation>}}


## cs.SD (2)



### (71/77) SECap: Speech Emotion Captioning with Large Language Model (Yaoxun Xu et al., 2023)

{{<citation>}}

Yaoxun Xu, Hangting Chen, Jianwei Yu, Qiaochu Huang, Zhiyong Wu, Shixiong Zhang, Guangzhi Li, Yi Luo, Rongzhi Gu. (2023)  
**SECap: Speech Emotion Captioning with Large Language Model**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: BERT, LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2312.10381v2)  

---


**ABSTRACT**  
Speech emotions are crucial in human communication and are extensively used in fields like speech synthesis and natural language understanding. Most prior studies, such as speech emotion recognition, have categorized speech emotions into a fixed set of classes. Yet, emotions expressed in human speech are often complex, and categorizing them into predefined groups can be insufficient to adequately represent speech emotions. On the contrary, describing speech emotions directly by means of natural language may be a more effective approach. Regrettably, there are not many studies available that have focused on this direction. Therefore, this paper proposes a speech emotion captioning framework named SECap, aiming at effectively describing speech emotions using natural language. Owing to the impressive capabilities of large language models in language comprehension and text generation, SECap employs LLaMA as the text decoder to allow the production of coherent speech emotion captions. In addition, SECap leverages HuBERT as the audio encoder to extract general speech features and Q-Former as the Bridge-Net to provide LLaMA with emotion-related speech features. To accomplish this, Q-Former utilizes mutual information learning to disentangle emotion-related speech features and speech contents, while implementing contrastive learning to extract more emotion-related speech features. The results of objective and subjective evaluations demonstrate that: 1) the SECap framework outperforms the HTSAT-BART baseline in all objective evaluations; 2) SECap can generate high-quality speech emotion captions that attain performance on par with human annotators in subjective mean opinion score tests.

{{</citation>}}


### (72/77) Self-Supervised Disentangled Representation Learning for Robust Target Speech Extraction (Zhaoxi Mu et al., 2023)

{{<citation>}}

Zhaoxi Mu, Xinyu Yang, Sining Sun, Qing Yang. (2023)  
**Self-Supervised Disentangled Representation Learning for Robust Target Speech Extraction**  

---
Primary Category: cs.SD  
Categories: cs-AI, cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Representation Learning, Self-Supervised, Transformer  
[Paper Link](http://arxiv.org/abs/2312.10305v1)  

---


**ABSTRACT**  
Speech signals are inherently complex as they encompass both global acoustic characteristics and local semantic information. However, in the task of target speech extraction, certain elements of global and local semantic information in the reference speech, which are irrelevant to speaker identity, can lead to speaker confusion within the speech extraction network. To overcome this challenge, we propose a self-supervised disentangled representation learning method. Our approach tackles this issue through a two-phase process, utilizing a reference speech encoding network and a global information disentanglement network to gradually disentangle the speaker identity information from other irrelevant factors. We exclusively employ the disentangled speaker identity information to guide the speech extraction network. Moreover, we introduce the adaptive modulation Transformer to ensure that the acoustic representation of the mixed signal remains undisturbed by the speaker embeddings. This component incorporates speaker embeddings as conditional information, facilitating natural and efficient guidance for the speech extraction network. Experimental results substantiate the effectiveness of our meticulously crafted approach, showcasing a substantial reduction in the likelihood of speaker confusion.

{{</citation>}}


## cs.DC (2)



### (73/77) SPT: Fine-Tuning Transformer-based Language Models Efficiently with Sparsification (Yuntao Gui et al., 2023)

{{<citation>}}

Yuntao Gui, Xiao Yan, Peiqi Yin, Han Yang, James Cheng. (2023)  
**SPT: Fine-Tuning Transformer-based Language Models Efficiently with Sparsification**  

---
Primary Category: cs.DC  
Categories: cs-AI, cs-DC, cs.DC  
Keywords: BERT, GPT, Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2312.10365v1)  

---


**ABSTRACT**  
Transformer-based large language models (e.g., BERT and GPT) achieve great success, and fine-tuning, which tunes a pre-trained model on a task-specific dataset, is the standard practice to utilize these models for downstream tasks. However, Transformer fine-tuning has long running time and high memory consumption due to the large size of the models. We propose the SPT system to fine-tune Transformer-based models efficiently by introducing sparsity. We observe that the memory consumption of Transformer mainly comes from storing attention weights for multi-head attention (MHA), and the majority of running time is spent on feed-forward network (FFN). Thus, we design the sparse MHA module, which computes and stores only large attention weights to reduce memory consumption, and the routed FFN module, which dynamically activates a subset of model parameters for each token to reduce computation cost. We implement SPT on PyTorch and customize CUDA kernels to run sparse MHA and routed FFN efficiently. Specifically, we use product quantization to identify the large attention weights and compute attention via sparse matrix multiplication for sparse MHA. For routed FFN, we batch the tokens according to their activated model parameters for efficient computation. We conduct extensive experiments to evaluate SPT on various model configurations. The results show that SPT consistently outperforms well-optimized baselines, reducing the peak memory consumption by up to 50% and accelerating fine-tuning by up to 2.2x.

{{</citation>}}


### (74/77) Opara: Exploiting Operator Parallelism for Expediting DNN Inference on GPUs (Aodong Chen et al., 2023)

{{<citation>}}

Aodong Chen, Fei Xu, Li Han, Yuan Dong, Li Chen, Zhi Zhou, Fangming Liu. (2023)  
**Opara: Exploiting Operator Parallelism for Expediting DNN Inference on GPUs**  

---
Primary Category: cs.DC  
Categories: cs-AI, cs-DC, cs.DC  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.10351v1)  

---


**ABSTRACT**  
GPUs have become the defacto hardware devices to accelerate Deep Neural Network (DNN) inference in deep learning(DL) frameworks. However, the conventional sequential execution mode of DNN operators in mainstream DL frameworks cannot fully utilize GPU resources, due to the increasing complexity of DNN model structures and the progressively smaller computational sizes of DNN operators. Moreover, the inadequate operator launch order in parallelized execution scenarios can lead to GPU resource wastage and unexpected performance interference among operators. To address such performance issues above, we propose Opara, a resource- and interference-aware DNN Operator parallel scheduling framework to accelerate the execution of DNN inference on GPUs. Specifically, Opara first employs CUDA Streams and CUDA Graph to automatically parallelize the execution of multiple DNN operators. It further leverages the resource demands of DNN operators to judiciously adjust the operator launch order on GPUs by overlapping the execution of compute-intensive and memory-intensive operators, so as to expedite DNN inference. We implement and open source a prototype of Opara based on PyTorch in a non-intrusive manner. Extensive prototype experiments with representative DNN and Transformer-based models demonstrate that Opara outperforms the default sequential CUDA Graph in PyTorch and the state-of-the-art DNN operator parallelism systems by up to 1.68$\times$ and 1.29$\times$, respectively, yet with acceptable runtime overhead.

{{</citation>}}


## cs.DB (1)



### (75/77) LLM-SQL-Solver: Can LLMs Determine SQL Equivalence? (Fuheng Zhao et al., 2023)

{{<citation>}}

Fuheng Zhao, Lawrence Lim, Ishtiyaque Ahmad, Divyakant Agrawal, Amr El Abbadi. (2023)  
**LLM-SQL-Solver: Can LLMs Determine SQL Equivalence?**  

---
Primary Category: cs.DB  
Categories: cs-CL, cs-DB, cs.DB  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.10321v1)  

---


**ABSTRACT**  
Judging the equivalence between two SQL queries is a fundamental problem with many practical applications in data management and SQL generation (i.e., evaluating the quality of generated SQL queries in text-to-SQL task). While the research community has reasoned about SQL equivalence for decades, it poses considerable difficulties and no complete solutions exist. Recently, Large Language Models (LLMs) have shown strong reasoning capability in conversation, question answering and solving mathematics challenges. In this paper, we study if LLMs can be used to determine the equivalence between SQL queries under two notions of SQL equivalence (semantic equivalence and relaxed equivalence). To assist LLMs in generating high quality responses, we present two prompting techniques: Miniature & Mull and Explain & Compare. The former technique is used to evaluate the semantic equivalence in which it asks LLMs to execute a query on a simple database instance and then explore if a counterexample exists by modifying the database. The latter technique is used to evaluate the relaxed equivalence in which it asks LLMs to explain the queries and then compare if they contain significant logical differences. Our experiments demonstrate using our techniques, LLMs is a promising tool to help data engineers in writing semantically equivalent SQL queries, however challenges still persist, and is a better metric for evaluating SQL generation than the popular execution accuracy.

{{</citation>}}


## physics.ao-ph (1)



### (76/77) FengWu-4DVar: Coupling the Data-driven Weather Forecasting Model with 4D Variational Assimilation (Yi Xiao et al., 2023)

{{<citation>}}

Yi Xiao, Lei Bai, Wei Xue, Kang Chen, Tao Han, Wanli Ouyang. (2023)  
**FengWu-4DVar: Coupling the Data-driven Weather Forecasting Model with 4D Variational Assimilation**  

---
Primary Category: physics.ao-ph  
Categories: cs-AI, cs-LG, physics-ao-ph, physics.ao-ph  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.12455v1)  

---


**ABSTRACT**  
Weather forecasting is a crucial yet highly challenging task. With the maturity of Artificial Intelligence (AI), the emergence of data-driven weather forecasting models has opened up a new paradigm for the development of weather forecasting systems. Despite the significant successes that have been achieved (e.g., surpassing advanced traditional physical models for global medium-range forecasting), existing data-driven weather forecasting models still rely on the analysis fields generated by the traditional assimilation and forecasting system, which hampers the significance of data-driven weather forecasting models regarding both computational cost and forecasting accuracy. In this work, we explore the possibility of coupling the data-driven weather forecasting model with data assimilation by integrating the global AI weather forecasting model, FengWu, with one of the most popular assimilation algorithms, Four-Dimensional Variational (4DVar) assimilation, and develop an AI-based cyclic weather forecasting system, FengWu-4DVar. FengWu-4DVar can incorporate observational data into the data-driven weather forecasting model and consider the temporal evolution of atmospheric dynamics to obtain accurate analysis fields for making predictions in a cycling manner without the help of physical models. Owning to the auto-differentiation ability of deep learning models, FengWu-4DVar eliminates the need of developing the cumbersome adjoint model, which is usually required in the traditional implementation of the 4DVar algorithm. Experiments on the simulated observational dataset demonstrate that FengWu-4DVar is capable of generating reasonable analysis fields for making accurate and efficient iterative predictions.

{{</citation>}}


## math.NA (1)



### (77/77) A charge-preserving method for solving graph neural diffusion networks (Lidia Aceto et al., 2023)

{{<citation>}}

Lidia Aceto, Pietro Antonio Grassi. (2023)  
**A charge-preserving method for solving graph neural diffusion networks**  

---
Primary Category: math.NA  
Categories: 68T07, 70H33, 37J06, 65P99, 65L05, cs-AI, cs-LG, cs-NA, hep-th, math-MP, math-NA, math-ph, math.NA  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2312.10279v1)  

---


**ABSTRACT**  
The aim of this paper is to give a systematic mathematical interpretation of the diffusion problem on which Graph Neural Networks (GNNs) models are based. The starting point of our approach is a dissipative functional leading to dynamical equations which allows us to study the symmetries of the model. We discuss the conserved charges and provide a charge-preserving numerical method for solving the dynamical equations. In any dynamical system and also in GRAph Neural Diffusion (GRAND), knowing the charge values and their conservation along the evolution flow could provide a way to understand how GNNs and other networks work with their learning capabilities.

{{</citation>}}
