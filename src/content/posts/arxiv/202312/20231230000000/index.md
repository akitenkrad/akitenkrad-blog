---
draft: false
title: "arXiv @ 2023.12.30"
date: 2023-12-30
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.12.30"
    identifier: arxiv_20231230
    parent: 202312_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.LG (21)](#cslg-21)
- [cs.SE (3)](#csse-3)
- [eess.SY (2)](#eesssy-2)
- [cs.CL (20)](#cscl-20)
- [cs.CR (4)](#cscr-4)
- [cs.SI (3)](#cssi-3)
- [cs.CV (23)](#cscv-23)
- [math.OC (1)](#mathoc-1)
- [stat.ML (1)](#statml-1)
- [econ.TH (1)](#econth-1)
- [cs.SD (2)](#cssd-2)
- [cs.CY (1)](#cscy-1)
- [eess.IV (2)](#eessiv-2)
- [cs.IT (1)](#csit-1)
- [cs.IR (2)](#csir-2)
- [eess.AS (2)](#eessas-2)
- [cs.RO (1)](#csro-1)

## cs.LG (21)



### (1/90) Beyond PID Controllers: PPO with Neuralized PID Policy for Proton Beam Intensity Control in Mu2e (Chenwei Xu et al., 2023)

{{<citation>}}

Chenwei Xu, Jerry Yao-Chieh Hu, Aakaash Narayanan, Mattson Thieme, Vladimir Nagaslaev, Mark Austin, Jeremy Arnold, Jose Berlioz, Pierrick Hanlet, Aisha Ibrahim, Dennis Nicklaus, Jovan Mitrevski, Jason Michael St. John, Gauri Pradhan, Andrea Saewert, Kiyomi Seiya, Brian Schupbach, Randy Thurman-Keup, Nhan Tran, Rui Shi, Seda Ogrenci, Alexis Maya-Isabelle Shuping, Kyle Hazelwood, Han Liu. (2023)  
**Beyond PID Controllers: PPO with Neuralized PID Policy for Proton Beam Intensity Control in Mu2e**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, physics-acc-ph  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.17372v1)  

---


**ABSTRACT**  
We introduce a novel Proximal Policy Optimization (PPO) algorithm aimed at addressing the challenge of maintaining a uniform proton beam intensity delivery in the Muon to Electron Conversion Experiment (Mu2e) at Fermi National Accelerator Laboratory (Fermilab). Our primary objective is to regulate the spill process to ensure a consistent intensity profile, with the ultimate goal of creating an automated controller capable of providing real-time feedback and calibration of the Spill Regulation System (SRS) parameters on a millisecond timescale. We treat the Mu2e accelerator system as a Markov Decision Process suitable for Reinforcement Learning (RL), utilizing PPO to reduce bias and enhance training stability. A key innovation in our approach is the integration of a neuralized Proportional-Integral-Derivative (PID) controller into the policy function, resulting in a significant improvement in the Spill Duty Factor (SDF) by 13.6%, surpassing the performance of the current PID controller baseline by an additional 1.6%. This paper presents the preliminary offline results based on a differentiable simulator of the Mu2e accelerator. It paves the groundwork for real-time implementations and applications, representing a crucial step towards automated proton beam intensity control for the Mu2e experiment.

{{</citation>}}


### (2/90) Graph Learning in 4D: a Quaternion-valued Laplacian to Enhance Spectral GCNs (Stefano Fiorini et al., 2023)

{{<citation>}}

Stefano Fiorini, Stefano Coniglio, Michele Ciavotta, Enza Messina. (2023)  
**Graph Learning in 4D: a Quaternion-valued Laplacian to Enhance Spectral GCNs**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Graph Convolutional Network  
[Paper Link](http://arxiv.org/abs/2312.17361v1)  

---


**ABSTRACT**  
We introduce QuaterGCN, a spectral Graph Convolutional Network (GCN) with quaternion-valued weights at whose core lies the Quaternionic Laplacian, a quaternion-valued Laplacian matrix by whose proposal we generalize two widely-used Laplacian matrices: the classical Laplacian (defined for undirected graphs) and the complex-valued Sign-Magnetic Laplacian (proposed to handle digraphs with weights of arbitrary sign). In addition to its generality, our Quaternionic Laplacian is the only Laplacian to completely preserve the topology of a digraph, as it can handle graphs and digraphs containing antiparallel pairs of edges (digons) of different weights without reducing them to a single (directed or undirected) edge as done with other Laplacians. Experimental results show the superior performance of QuaterGCN compared to other state-of-the-art GCNs, particularly in scenarios where the information the digons carry is crucial to successfully address the task at hand.

{{</citation>}}


### (3/90) STanHop: Sparse Tandem Hopfield Model for Memory-Enhanced Time Series Prediction (Dennis Wu et al., 2023)

{{<citation>}}

Dennis Wu, Jerry Yao-Chieh Hu, Weijian Li, Bo-Yu Chen, Han Liu. (2023)  
**STanHop: Sparse Tandem Hopfield Model for Memory-Enhanced Time Series Prediction**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs-NE, cs.LG, stat-ML  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2312.17346v1)  

---


**ABSTRACT**  
We present STanHop-Net (Sparse Tandem Hopfield Network) for multivariate time series prediction with memory-enhanced capabilities. At the heart of our approach is STanHop, a novel Hopfield-based neural network block, which sparsely learns and stores both temporal and cross-series representations in a data-dependent fashion. In essence, STanHop sequentially learn temporal representation and cross-series representation using two tandem sparse Hopfield layers. In addition, StanHop incorporates two additional external memory modules: a Plug-and-Play module and a Tune-and-Play module for train-less and task-aware memory-enhancements, respectively. They allow StanHop-Net to swiftly respond to certain sudden events. Methodologically, we construct the StanHop-Net by stacking STanHop blocks in a hierarchical fashion, enabling multi-resolution feature extraction with resolution-specific sparsity. Theoretically, we introduce a sparse extension of the modern Hopfield model (Generalized Sparse Modern Hopfield Model) and show that it endows a tighter memory retrieval error compared to the dense counterpart without sacrificing memory capacity. Empirically, we validate the efficacy of our framework on both synthetic and real-world settings.

{{</citation>}}


### (4/90) Rethinking Model-based, Policy-based, and Value-based Reinforcement Learning via the Lens of Representation Complexity (Guhao Feng et al., 2023)

{{<citation>}}

Guhao Feng, Han Zhong. (2023)  
**Rethinking Model-based, Policy-based, and Value-based Reinforcement Learning via the Lens of Representation Complexity**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CC, cs-DS, cs-LG, cs.LG, stat-ML  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.17248v1)  

---


**ABSTRACT**  
Reinforcement Learning (RL) encompasses diverse paradigms, including model-based RL, policy-based RL, and value-based RL, each tailored to approximate the model, optimal policy, and optimal value function, respectively. This work investigates the potential hierarchy of representation complexity -- the complexity of functions to be represented -- among these RL paradigms. We first demonstrate that, for a broad class of Markov decision processes (MDPs), the model can be represented by constant-depth circuits with polynomial size or Multi-Layer Perceptrons (MLPs) with constant layers and polynomial hidden dimension. However, the representation of the optimal policy and optimal value proves to be $\mathsf{NP}$-complete and unattainable by constant-layer MLPs with polynomial size. This demonstrates a significant representation complexity gap between model-based RL and model-free RL, which includes policy-based RL and value-based RL. To further explore the representation complexity hierarchy between policy-based RL and value-based RL, we introduce another general class of MDPs where both the model and optimal policy can be represented by constant-depth circuits with polynomial size or constant-layer MLPs with polynomial size. In contrast, representing the optimal value is $\mathsf{P}$-complete and intractable via a constant-layer MLP with polynomial hidden dimension. This accentuates the intricate representation complexity associated with value-based RL compared to policy-based RL. In summary, we unveil a potential representation complexity hierarchy within RL -- representing the model emerges as the easiest task, followed by the optimal policy, while representing the optimal value function presents the most intricate challenge.

{{</citation>}}


### (5/90) The LLM Surgeon (Tycho F. A. van der Ouderaa et al., 2023)

{{<citation>}}

Tycho F. A. van der Ouderaa, Markus Nagel, Mart van Baalen, Yuki M. Asano, Tijmen Blankevoort. (2023)  
**The LLM Surgeon**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.17244v1)  

---


**ABSTRACT**  
State-of-the-art language models are becoming increasingly large in an effort to achieve the highest performance on large corpora of available textual data. However, the sheer size of the Transformer architectures makes it difficult to deploy models within computational, environmental or device-specific constraints. We explore data-driven compression of existing pretrained models as an alternative to training smaller models from scratch. To do so, we scale Kronecker-factored curvature approximations of the target loss landscape to large language models. In doing so, we can compute both the dynamic allocation of structures that can be removed as well as updates of remaining weights that account for the removal. We provide a general framework for unstructured, semi-structured and structured pruning and improve upon weight updates to capture more correlations between weights, while remaining computationally efficient. Experimentally, our method can prune rows and columns from a range of OPT models and Llamav2-7B by 20%-30%, with a negligible loss in performance, and achieve state-of-the-art results in unstructured and semi-structured pruning of large language models.

{{</citation>}}


### (6/90) Fast Inference of Mixture-of-Experts Language Models with Offloading (Artyom Eliseev et al., 2023)

{{<citation>}}

Artyom Eliseev, Denis Mazur. (2023)  
**Fast Inference of Mixture-of-Experts Language Models with Offloading**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-DC, cs-LG, cs.LG  
Keywords: Google, Language Model  
[Paper Link](http://arxiv.org/abs/2312.17238v1)  

---


**ABSTRACT**  
With the widespread adoption of Large Language Models (LLMs), many deep learning practitioners are looking for strategies of running these models more efficiently. One such strategy is to use sparse Mixture-of-Experts (MoE) - a type of model architectures where only a fraction of model layers are active for any given input. This property allows MoE-based language models to generate tokens faster than their dense counterparts, but it also increases model size due to having multiple experts. Unfortunately, this makes state-of-the-art MoE language models difficult to run without high-end GPUs. In this work, we study the problem of running large MoE language models on consumer hardware with limited accelerator memory. We build upon parameter offloading algorithms and propose a novel strategy that accelerates offloading by taking advantage of innate properties of MoE LLMs. Using this strategy, we build can run Mixtral-8x7B with mixed quantization on desktop hardware and free-tier Google Colab instances.

{{</citation>}}


### (7/90) Can Active Sampling Reduce Causal Confusion in Offline Reinforcement Learning? (Gunshi Gupta et al., 2023)

{{<citation>}}

Gunshi Gupta, Tim G. J. Rudner, Rowan Thomas McAllister, Adrien Gaidon, Yarin Gal. (2023)  
**Can Active Sampling Reduce Causal Confusion in Offline Reinforcement Learning?**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.17168v1)  

---


**ABSTRACT**  
Causal confusion is a phenomenon where an agent learns a policy that reflects imperfect spurious correlations in the data. Such a policy may falsely appear to be optimal during training if most of the training data contain such spurious correlations. This phenomenon is particularly pronounced in domains such as robotics, with potentially large gaps between the open- and closed-loop performance of an agent. In such settings, causally confused models may appear to perform well according to open-loop metrics during training but fail catastrophically when deployed in the real world. In this paper, we study causal confusion in offline reinforcement learning. We investigate whether selectively sampling appropriate points from a dataset of demonstrations may enable offline reinforcement learning agents to disambiguate the underlying causal mechanisms of the environment, alleviate causal confusion in offline reinforcement learning, and produce a safer model for deployment. To answer this question, we consider a set of tailored offline reinforcement learning datasets that exhibit causal ambiguity and assess the ability of active sampling techniques to reduce causal confusion at evaluation. We provide empirical evidence that uniform and active sampling techniques are able to consistently reduce causal confusion as training progresses and that active sampling is able to do so significantly more efficiently than uniform sampling.

{{</citation>}}


### (8/90) Generalizable Visual Reinforcement Learning with Segment Anything Model (Ziyu Wang et al., 2023)

{{<citation>}}

Ziyu Wang, Yanjie Ze, Yifei Sun, Zhecheng Yuan, Huazhe Xu. (2023)  
**Generalizable Visual Reinforcement Learning with Segment Anything Model**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs-RO, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.17116v1)  

---


**ABSTRACT**  
Learning policies that can generalize to unseen environments is a fundamental challenge in visual reinforcement learning (RL). While most current methods focus on acquiring robust visual representations through auxiliary supervision, pre-training, or data augmentation, the potential of modern vision foundation models remains underleveraged. In this work, we introduce Segment Anything Model for Generalizable visual RL (SAM-G), a novel framework that leverages the promptable segmentation ability of Segment Anything Model (SAM) to enhance the generalization capabilities of visual RL agents. We utilize image features from DINOv2 and SAM to find correspondence as point prompts to SAM, and then SAM produces high-quality masked images for agents directly. Evaluated across 8 DMControl tasks and 3 Adroit tasks, SAM-G significantly improves the visual generalization ability without altering the RL agents' architecture but merely their observations. Notably, SAM-G achieves 44% and 29% relative improvements on the challenging video hard setting on DMControl and Adroit respectively, compared to state-of-the-art methods. Video and code: https://yanjieze.com/SAM-G/

{{</citation>}}


### (9/90) On the rate of convergence of an over-parametrized Transformer classifier learned by gradient descent (Michael Kohler et al., 2023)

{{<citation>}}

Michael Kohler, Adam Krzyzak. (2023)  
**On the rate of convergence of an over-parametrized Transformer classifier learned by gradient descent**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, math-ST, stat-ML, stat-TH  
Keywords: ChatGPT, GPT, Transformer  
[Paper Link](http://arxiv.org/abs/2312.17007v1)  

---


**ABSTRACT**  
One of the most recent and fascinating breakthroughs in artificial intelligence is ChatGPT, a chatbot which can simulate human conversation. ChatGPT is an instance of GPT4, which is a language model based on generative gredictive gransformers. So if one wants to study from a theoretical point of view, how powerful such artificial intelligence can be, one approach is to consider transformer networks and to study which problems one can solve with these networks theoretically. Here it is not only important what kind of models these network can approximate, or how they can generalize their knowledge learned by choosing the best possible approximation to a concrete data set, but also how well optimization of such transformer network based on concrete data set works. In this article we consider all these three different aspects simultaneously and show a theoretical upper bound on the missclassification probability of a transformer network fitted to the observed data. For simplicity we focus in this context on transformer encoder networks which can be applied to define an estimate in the context of a classification problem involving natural language.

{{</citation>}}


### (10/90) RLPlanner: Reinforcement Learning based Floorplanning for Chiplets with Fast Thermal Analysis (Yuanyuan Duan et al., 2023)

{{<citation>}}

Yuanyuan Duan, Xingchen Liu, Zhiping Yu, Hanming Wu, Leilai Shao, Xiaolei Zhu. (2023)  
**RLPlanner: Reinforcement Learning based Floorplanning for Chiplets with Fast Thermal Analysis**  

---
Primary Category: cs.LG  
Categories: cs-AR, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.16895v1)  

---


**ABSTRACT**  
Chiplet-based systems have gained significant attention in recent years due to their low cost and competitive performance. As the complexity and compactness of a chiplet-based system increase, careful consideration must be given to microbump assignments, interconnect delays, and thermal limitations during the floorplanning stage. This paper introduces RLPlanner, an efficient early-stage floorplanning tool for chiplet-based systems with a novel fast thermal evaluation method. RLPlanner employs advanced reinforcement learning to jointly minimize total wirelength and temperature. To alleviate the time-consuming thermal calculations, RLPlanner incorporates the developed fast thermal evaluation method to expedite the iterations and optimizations. Comprehensive experiments demonstrate that our proposed fast thermal evaluation method achieves a mean absolute error (MAE) of 0.25 K and delivers over 120x speed-up compared to the open-source thermal solver HotSpot. When integrated with our fast thermal evaluation method, RLPlanner achieves an average improvement of 20.28\% in minimizing the target objective (a combination of wirelength and temperature), within a similar running time, compared to the classic simulated annealing method with HotSpot.

{{</citation>}}


### (11/90) FlexSSL : A Generic and Efficient Framework for Semi-Supervised Learning (Huiling Qin et al., 2023)

{{<citation>}}

Huiling Qin, Xianyuan Zhan, Yuanxun Li, Yu Zheng. (2023)  
**FlexSSL : A Generic and Efficient Framework for Semi-Supervised Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2312.16892v1)  

---


**ABSTRACT**  
Semi-supervised learning holds great promise for many real-world applications, due to its ability to leverage both unlabeled and expensive labeled data. However, most semi-supervised learning algorithms still heavily rely on the limited labeled data to infer and utilize the hidden information from unlabeled data. We note that any semi-supervised learning task under the self-training paradigm also hides an auxiliary task of discriminating label observability. Jointly solving these two tasks allows full utilization of information from both labeled and unlabeled data, thus alleviating the problem of over-reliance on labeled data. This naturally leads to a new generic and efficient learning framework without the reliance on any domain-specific information, which we call FlexSSL. The key idea of FlexSSL is to construct a semi-cooperative "game", which forges cooperation between a main self-interested semi-supervised learning task and a companion task that infers label observability to facilitate main task training. We show with theoretical derivation of its connection to loss re-weighting on noisy labels. Through evaluations on a diverse range of tasks, we demonstrate that FlexSSL can consistently enhance the performance of semi-supervised learning algorithms.

{{</citation>}}


### (12/90) Molecular Property Prediction Based on Graph Structure Learning (Bangyi Zhao et al., 2023)

{{<citation>}}

Bangyi Zhao, Weixia Xu, Jihong Guan, Shuigeng Zhou. (2023)  
**Molecular Property Prediction Based on Graph Structure Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, q-bio-BM  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2312.16855v1)  

---


**ABSTRACT**  
Molecular property prediction (MPP) is a fundamental but challenging task in the computer-aided drug discovery process. More and more recent works employ different graph-based models for MPP, which have made considerable progress in improving prediction performance. However, current models often ignore relationships between molecules, which could be also helpful for MPP. For this sake, in this paper we propose a graph structure learning (GSL) based MPP approach, called GSL-MPP. Specifically, we first apply graph neural network (GNN) over molecular graphs to extract molecular representations. Then, with molecular fingerprints, we construct a molecular similarity graph (MSG). Following that, we conduct graph structure learning on the MSG (i.e., molecule-level graph structure learning) to get the final molecular embeddings, which are the results of fusing both GNN encoded molecular representations and the relationships among molecules, i.e., combining both intra-molecule and inter-molecule information. Finally, we use these molecular embeddings to perform MPP. Extensive experiments on seven various benchmark datasets show that our method could achieve state-of-the-art performance in most cases, especially on classification tasks. Further visualization studies also demonstrate the good molecular representations of our method.

{{</citation>}}


### (13/90) Sensor Data Simulation for Anomaly Detection of the Elderly Living Alone (Kai Tanaka et al., 2023)

{{<citation>}}

Kai Tanaka, Mineichi Kudo, Keigo Kimura. (2023)  
**Sensor Data Simulation for Anomaly Detection of the Elderly Living Alone**  

---
Primary Category: cs.LG  
Categories: cs-HC, cs-LG, cs.LG, eess-SP  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2312.16852v1)  

---


**ABSTRACT**  
With the increase of the number of elderly people living alone around the world, there is a growing demand for sensor-based detection of anomalous behaviors. Although smart homes with ambient sensors could be useful for detecting such anomalies, there is a problem of lack of sufficient real data for developing detection algorithms. For coping with this problem, several sensor data simulators have been proposed, but they have not been able to model appropriately the long-term transitions and correlations between anomalies that exist in reality. In this paper, therefore, we propose a novel sensor data simulator that can model these factors in generation of sensor data. Anomalies considered in this study were classified into three types of \textit{state anomalies}, \textit{activity anomalies}, and \textit{moving anomalies}. The simulator produces 10 years data in 100 min. including six anomalies, two for each type. Numerical evaluations show that this simulator is superior to the past simulators in the sense that it simulates well day-to-day variations of real data.

{{</citation>}}


### (14/90) Hierarchical Aggregations for High-Dimensional Multiplex Graph Embedding (Kamel Abdous et al., 2023)

{{<citation>}}

Kamel Abdous, Nairouz Mrabah, Mohamed Bouguessa. (2023)  
**Hierarchical Aggregations for High-Dimensional Multiplex Graph Embedding**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-SI, cs.LG  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2312.16834v1)  

---


**ABSTRACT**  
We investigate the problem of multiplex graph embedding, that is, graphs in which nodes interact through multiple types of relations (dimensions). In recent years, several methods have been developed to address this problem. However, the need for more effective and specialized approaches grows with the production of graph data with diverse characteristics. In particular, real-world multiplex graphs may exhibit a high number of dimensions, making it difficult to construct a single consensus representation. Furthermore, important information can be hidden in complex latent structures scattered in multiple dimensions. To address these issues, we propose HMGE, a novel embedding method based on hierarchical aggregation for high-dimensional multiplex graphs. Hierarchical aggregation consists of learning a hierarchical combination of the graph dimensions and refining the embeddings at each hierarchy level. Non-linear combinations are computed from previous ones, thus uncovering complex information and latent structures hidden in the multiplex graph dimensions. Moreover, we leverage mutual information maximization between local patches and global summaries to train the model without supervision. This allows to capture of globally relevant information present in diverse locations of the graph. Detailed experiments on synthetic and real-world data illustrate the suitability of our approach to downstream supervised tasks, including link prediction and node classification.

{{</citation>}}


### (15/90) METER: A Dynamic Concept Adaptation Framework for Online Anomaly Detection (Jiaqi Zhu et al., 2023)

{{<citation>}}

Jiaqi Zhu, Shaofeng Cai, Fang Deng, Beng Chin Ooi, Wenqiao Zhang. (2023)  
**METER: A Dynamic Concept Adaptation Framework for Online Anomaly Detection**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2312.16831v1)  

---


**ABSTRACT**  
Real-time analytics and decision-making require online anomaly detection (OAD) to handle drifts in data streams efficiently and effectively. Unfortunately, existing approaches are often constrained by their limited detection capacity and slow adaptation to evolving data streams, inhibiting their efficacy and efficiency in handling concept drift, which is a major challenge in evolving data streams. In this paper, we introduce METER, a novel dynamic concept adaptation framework that introduces a new paradigm for OAD. METER addresses concept drift by first training a base detection model on historical data to capture recurring central concepts, and then learning to dynamically adapt to new concepts in data streams upon detecting concept drift. Particularly, METER employs a novel dynamic concept adaptation technique that leverages a hypernetwork to dynamically generate the parameter shift of the base detection model, providing a more effective and efficient solution than conventional retraining or fine-tuning approaches. Further, METER incorporates a lightweight drift detection controller, underpinned by evidential deep learning, to support robust and interpretable concept drift detection. We conduct an extensive experimental evaluation, and the results show that METER significantly outperforms existing OAD approaches in various application scenarios.

{{</citation>}}


### (16/90) Layer Attack Unlearning: Fast and Accurate Machine Unlearning via Layer Level Attack and Knowledge Distillation (Hyunjune Kim et al., 2023)

{{<citation>}}

Hyunjune Kim, Sangyong Lee, Simon S. Woo. (2023)  
**Layer Attack Unlearning: Fast and Accurate Machine Unlearning via Layer Level Attack and Knowledge Distillation**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2312.16823v1)  

---


**ABSTRACT**  
Recently, serious concerns have been raised about the privacy issues related to training datasets in machine learning algorithms when including personal data. Various regulations in different countries, including the GDPR grant individuals to have personal data erased, known as 'the right to be forgotten' or 'the right to erasure'. However, there has been less research on effectively and practically deleting the requested personal data from the training set while not jeopardizing the overall machine learning performance. In this work, we propose a fast and novel machine unlearning paradigm at the layer level called layer attack unlearning, which is highly accurate and fast compared to existing machine unlearning algorithms. We introduce the Partial-PGD algorithm to locate the samples to forget efficiently. In addition, we only use the last layer of the model inspired by the Forward-Forward algorithm for unlearning process. Lastly, we use Knowledge Distillation (KD) to reliably learn the decision boundaries from the teacher using soft label information to improve accuracy performance. We conducted extensive experiments with SOTA machine unlearning models and demonstrated the effectiveness of our approach for accuracy and end-to-end unlearning performance.

{{</citation>}}


### (17/90) Temporal Knowledge Distillation for Time-Sensitive Financial Services Applications (Hongda Shen et al., 2023)

{{<citation>}}

Hongda Shen, Eren Kurshan. (2023)  
**Temporal Knowledge Distillation for Time-Sensitive Financial Services Applications**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Financial, Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2312.16799v1)  

---


**ABSTRACT**  
Detecting anomalies has become an increasingly critical function in the financial service industry. Anomaly detection is frequently used in key compliance and risk functions such as financial crime detection fraud and cybersecurity. The dynamic nature of the underlying data patterns especially in adversarial environments like fraud detection poses serious challenges to the machine learning models. Keeping up with the rapid changes by retraining the models with the latest data patterns introduces pressures in balancing the historical and current patterns while managing the training data size. Furthermore the model retraining times raise problems in time-sensitive and high-volume deployment systems where the retraining period directly impacts the models ability to respond to ongoing attacks in a timely manner. In this study we propose a temporal knowledge distillation-based label augmentation approach (TKD) which utilizes the learning from older models to rapidly boost the latest model and effectively reduces the model retraining times to achieve improved agility. Experimental results show that the proposed approach provides advantages in retraining times while improving the model performance.

{{</citation>}}


### (18/90) Learning the Dynamic Correlations and Mitigating Noise by Hierarchical Convolution for Long-term Sequence Forecasting (Zhihao Yu et al., 2023)

{{<citation>}}

Zhihao Yu, Liantao Ma, Yasha Wang, Junfeng Zhao. (2023)  
**Learning the Dynamic Correlations and Mitigating Noise by Hierarchical Convolution for Long-term Sequence Forecasting**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.16790v1)  

---


**ABSTRACT**  
Deep learning algorithms, especially Transformer-based models, have achieved significant performance by capturing long-range dependencies and historical information. However, the power of convolution has not been fully investigated. Moreover, most existing works ignore the dynamic interaction among variables and evolutionary noise in series. Addressing these issues, we propose a Hierarchical Memorizing Network (HMNet). In particular, a hierarchical convolution structure is introduced to extract the information from the series at various scales. Besides, we propose a dynamic variable interaction module to learn the varying correlation and an adaptive denoising module to search and exploit similar patterns to alleviate noises. These modules can cooperate with the hierarchical structure from the perspective of fine to coarse grain. Experiments on five benchmarks demonstrate that HMNet significantly outperforms the state-of-the-art models by 10.6% on MSE and 5.7% on MAE. Our code is released at https://github.com/yzhHoward/HMNet.

{{</citation>}}


### (19/90) Mitigating Degree Biases in Message Passing Mechanism by Utilizing Community Structures (Van Thuy Hoang et al., 2023)

{{<citation>}}

Van Thuy Hoang, O-Joun Lee. (2023)  
**Mitigating Degree Biases in Message Passing Mechanism by Utilizing Community Structures**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-SI, cs.LG  
Keywords: Bias, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.16788v1)  

---


**ABSTRACT**  
This study utilizes community structures to address node degree biases in message-passing (MP) via learnable graph augmentations and novel graph transformers. Recent augmentation-based methods showed that MP neural networks often perform poorly on low-degree nodes, leading to degree biases due to a lack of messages reaching low-degree nodes. Despite their success, most methods use heuristic or uniform random augmentations, which are non-differentiable and may not always generate valuable edges for learning representations. In this paper, we propose Community-aware Graph Transformers, namely CGT, to learn degree-unbiased representations based on learnable augmentations and graph transformers by extracting within community structures. We first design a learnable graph augmentation to generate more within-community edges connecting low-degree nodes through edge perturbation. Second, we propose an improved self-attention to learn underlying proximity and the roles of nodes within the community. Third, we propose a self-supervised learning task that could learn the representations to preserve the global graph structure and regularize the graph augmentations. Extensive experiments on various benchmark datasets showed CGT outperforms state-of-the-art baselines and significantly improves the node degree biases. The source code is available at https://github.com/NSLab-CUK/Community-aware-Graph-Transformer.

{{</citation>}}


### (20/90) Learning Scalable Structural Representations for Link Prediction with Bloom Signatures (Tianyi Zhang et al., 2023)

{{<citation>}}

Tianyi Zhang, Haoteng Yin, Rongzhe Wei, Pan Li, Anshumali Shrivastava. (2023)  
**Learning Scalable Structural Representations for Link Prediction with Bloom Signatures**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SI, cs.LG  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2312.16784v1)  

---


**ABSTRACT**  
Graph neural networks (GNNs) have shown great potential in learning on graphs, but they are known to perform sub-optimally on link prediction tasks. Existing GNNs are primarily designed to learn node-wise representations and usually fail to capture pairwise relations between target nodes, which proves to be crucial for link prediction. Recent works resort to learning more expressive edge-wise representations by enhancing vanilla GNNs with structural features such as labeling tricks and link prediction heuristics, but they suffer from high computational overhead and limited scalability. To tackle this issue, we propose to learn structural link representations by augmenting the message-passing framework of GNNs with Bloom signatures. Bloom signatures are hashing-based compact encodings of node neighborhoods, which can be efficiently merged to recover various types of edge-wise structural features. We further show that any type of neighborhood overlap-based heuristic can be estimated by a neural network that takes Bloom signatures as input. GNNs with Bloom signatures are provably more expressive than vanilla GNNs and also more scalable than existing edge-wise models. Experimental results on five standard link prediction benchmarks show that our proposed model achieves comparable or better performance than existing edge-wise GNN models while being 3-200 $\times$ faster and more memory-efficient for online inference.

{{</citation>}}


### (21/90) The Fourth International Verification of Neural Networks Competition (VNN-COMP 2023): Summary and Results (Christopher Brix et al., 2023)

{{<citation>}}

Christopher Brix, Stanley Bak, Changliu Liu, Taylor T. Johnson. (2023)  
**The Fourth International Verification of Neural Networks Competition (VNN-COMP 2023): Summary and Results**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-SE, cs.LG  
Keywords: AWS  
[Paper Link](http://arxiv.org/abs/2312.16760v1)  

---


**ABSTRACT**  
This report summarizes the 4th International Verification of Neural Networks Competition (VNN-COMP 2023), held as a part of the 6th Workshop on Formal Methods for ML-Enabled Autonomous Systems (FoMLAS), that was collocated with the 35th International Conference on Computer-Aided Verification (CAV). VNN-COMP is held annually to facilitate the fair and objective comparison of state-of-the-art neural network verification tools, encourage the standardization of tool interfaces, and bring together the neural network verification community. To this end, standardized formats for networks (ONNX) and specification (VNN-LIB) were defined, tools were evaluated on equal-cost hardware (using an automatic evaluation pipeline based on AWS instances), and tool parameters were chosen by the participants before the final test sets were made public. In the 2023 iteration, 7 teams participated on a diverse set of 10 scored and 4 unscored benchmarks. This report summarizes the rules, benchmarks, participating tools, results, and lessons learned from this iteration of this competition.

{{</citation>}}


## cs.SE (3)



### (22/90) An Introduction to Adaptive Software Security (Mehran Alidoost Nia, 2023)

{{<citation>}}

Mehran Alidoost Nia. (2023)  
**An Introduction to Adaptive Software Security**  

---
Primary Category: cs.SE  
Categories: cs-CR, cs-SE, cs.SE  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2312.17358v1)  

---


**ABSTRACT**  
This paper presents the adaptive software security model, an innovative approach integrating the MAPE-K loop and the Software Development Life Cycle (SDLC). It proactively embeds security policies throughout development, reducing vulnerabilities from different levels of software engineering. Three primary contributions-MAPE-K integration, SDLC embedding, and analytical insights-converge to create a comprehensive approach for strengthening software systems against security threats. This research represents a paradigm shift, adapting security measures with agile software development and ensuring continuous improvement in the face of evolving threats. The model emerges as a robust solution, addressing the crucial need for adaptive software security strategies in modern software development. We analytically discuss the advantages of the proposed model.

{{</citation>}}


### (23/90) GitAgent: Facilitating Autonomous Agent with GitHub by Tool Extension (Bohan Lyu et al., 2023)

{{<citation>}}

Bohan Lyu, Xin Cong, Heyang Yu, Pan Yang, Yujia Qin, Yining Ye, Yaxi Lu, Zhong Zhang, Yukun Yan, Yankai Lin, Zhiyuan Liu, Maosong Sun. (2023)  
**GitAgent: Facilitating Autonomous Agent with GitHub by Tool Extension**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-IR, cs-SE, cs.SE  
Keywords: ChatGPT, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2312.17294v1)  

---


**ABSTRACT**  
While Large Language Models (LLMs) like ChatGPT and GPT-4 have demonstrated exceptional proficiency in natural language processing, their efficacy in addressing complex, multifaceted tasks remains limited. A growing area of research focuses on LLM-based agents equipped with external tools capable of performing diverse tasks. However, existing LLM-based agents only support a limited set of tools which is unable to cover a diverse range of user queries, especially for those involving expertise domains. It remains a challenge for LLM-based agents to extend their tools autonomously when confronted with various user queries. As GitHub has hosted a multitude of repositories which can be seen as a good resource for tools, a promising solution is that LLM-based agents can autonomously integrate the repositories in GitHub according to the user queries to extend their tool set. In this paper, we introduce GitAgent, an agent capable of achieving the autonomous tool extension from GitHub. GitAgent follows a four-phase procedure to incorporate repositories and it can learn human experience by resorting to GitHub Issues/PRs to solve problems encountered during the procedure. Experimental evaluation involving 30 user queries demonstrates GitAgent's effectiveness, achieving a 69.4% success rate on average.

{{</citation>}}


### (24/90) TRIAD: Automated Traceability Recovery based on Biterm-enhanced Deduction of Transitive Links among Artifacts (Hui Gao et al., 2023)

{{<citation>}}

Hui Gao, Hongyu Kuang, Wesley K. G. Assunção, Christoph Mayr-Dorn, Guoping Rong, He Zhang, Xiaoxing Ma, Alexander Egyed. (2023)  
**TRIAD: Automated Traceability Recovery based on Biterm-enhanced Deduction of Transitive Links among Artifacts**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Information Retrieval  
[Paper Link](http://arxiv.org/abs/2312.16854v1)  

---


**ABSTRACT**  
Traceability allows stakeholders to extract and comprehend the trace links among software artifacts introduced across the software life cycle, to provide significant support for software engineering tasks. Despite its proven benefits, software traceability is challenging to recover and maintain manually. Hence, plenty of approaches for automated traceability have been proposed. Most rely on textual similarities among software artifacts, such as those based on Information Retrieval (IR). However, artifacts in different abstraction levels usually have different textual descriptions, which can greatly hinder the performance of IR-based approaches (e.g., a requirement in natural language may have a small textual similarity to a Java class). In this work, we leverage the consensual biterms and transitive relationships (i.e., inner- and outer-transitive links) based on intermediate artifacts to improve IR-based traceability recovery. We first extract and filter biterms from all source, intermediate, and target artifacts. We then use the consensual biterms from the intermediate artifacts to extend the biterms of both source and target artifacts, and finally deduce outer and inner-transitive links to adjust text similarities between source and target artifacts. We conducted a comprehensive empirical evaluation based on five systems widely used in other literature to show that our approach can outperform four state-of-the-art approaches, and how its performance is affected by different conditions of source, intermediate, and target artifacts. The results indicate that our approach can outperform baseline approaches in AP over 15% and MAP over 10% on average.

{{</citation>}}


## eess.SY (2)



### (25/90) Towards Auto-Modeling of Formal Verification for NextG Protocols: A Multimodal cross- and self-attention Large Language Model Approach (Jingda Yang et al., 2023)

{{<citation>}}

Jingda Yang, Ying Wang. (2023)  
**Towards Auto-Modeling of Formal Verification for NextG Protocols: A Multimodal cross- and self-attention Large Language Model Approach**  

---
Primary Category: eess.SY  
Categories: cs-AI, cs-LG, cs-SY, eess-SY, eess.SY  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2312.17353v1)  

---


**ABSTRACT**  
This paper introduces Auto-modeling of Formal Verification with Real-world Prompting for 5G and NextG protocols (AVRE), a novel system designed for the formal verification of Next Generation (NextG) communication protocols, addressing the increasing complexity and scalability challenges in network protocol design and verification. Utilizing Large Language Models (LLMs), AVRE transforms protocol descriptions into dependency graphs and formal models, efficiently resolving ambiguities and capturing design intent. The system integrates a transformer model with LLMs to autonomously establish quantifiable dependency relationships through cross- and self-attention mechanisms. Enhanced by iterative feedback from the HyFuzz experimental platform, AVRE significantly advances the accuracy and relevance of formal verification in complex communication protocols, offering a groundbreaking approach to validating sophisticated communication systems. We compare CAL's performance with state-of-the-art LLM-based models and traditional time sequence models, demonstrating its superiority in accuracy and robustness, achieving an accuracy of 95.94\% and an AUC of 0.98. This NLP-based approach enables, for the first time, the creation of exploits directly from design documents, making remarkable progress in scalable system verification and validation.

{{</citation>}}


### (26/90) Properties of Immersions for Systems with Multiple Limit Sets with Implications to Learning Koopman Embeddings (Zexiang Liu et al., 2023)

{{<citation>}}

Zexiang Liu, Necmiye Ozay, Eduardo D. Sontag. (2023)  
**Properties of Immersions for Systems with Multiple Limit Sets with Implications to Learning Koopman Embeddings**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY, math-DS  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2312.17045v1)  

---


**ABSTRACT**  
Linear immersions (or Koopman eigenmappings) of a nonlinear system have wide applications in prediction and control. In this work, we study the non-existence of one-to-one linear immersions for nonlinear systems with multiple omega-limit sets. While previous research has indicated the possibility of discontinuous one-to-one linear immersions for such systems, it remained uncertain whether continuous one-to-one linear immersions are attainable. Under mild conditions, we prove that any continuous one-to-one immersion to a class of systems including linear systems cannot distinguish different omega-limit sets, and thus cannot be one-to-one. Furthermore, we show that this property is also shared by approximate linear immersions learned from data as sample size increases and sampling interval decreases. Multiple examples are studied to illustrate our results.

{{</citation>}}


## cs.CL (20)



### (27/90) Language Model as an Annotator: Unsupervised Context-aware Quality Phrase Generation (Zhihao Zhang et al., 2023)

{{<citation>}}

Zhihao Zhang, Yuan Zuo, Chenghua Lin, Junjie Wu. (2023)  
**Language Model as an Annotator: Unsupervised Context-aware Quality Phrase Generation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, Language Model, Sequence-to-Sequence  
[Paper Link](http://arxiv.org/abs/2312.17349v1)  

---


**ABSTRACT**  
Phrase mining is a fundamental text mining task that aims to identify quality phrases from context. Nevertheless, the scarcity of extensive gold labels datasets, demanding substantial annotation efforts from experts, renders this task exceptionally challenging. Furthermore, the emerging, infrequent, and domain-specific nature of quality phrases presents further challenges in dealing with this task. In this paper, we propose LMPhrase, a novel unsupervised context-aware quality phrase mining framework built upon large pre-trained language models (LMs). Specifically, we first mine quality phrases as silver labels by employing a parameter-free probing technique called Perturbed Masking on the pre-trained language model BERT (coined as Annotator). In contrast to typical statistic-based or distantly-supervised methods, our silver labels, derived from large pre-trained language models, take into account rich contextual information contained in the LMs. As a result, they bring distinct advantages in preserving informativeness, concordance, and completeness of quality phrases. Secondly, training a discriminative span prediction model heavily relies on massive annotated data and is likely to face the risk of overfitting silver labels. Alternatively, we formalize phrase tagging task as the sequence generation problem by directly fine-tuning on the Sequence-to-Sequence pre-trained language model BART with silver labels (coined as Generator). Finally, we merge the quality phrases from both the Annotator and Generator as the final predictions, considering their complementary nature and distinct characteristics. Extensive experiments show that our LMPhrase consistently outperforms all the existing competitors across two different granularity phrase mining tasks, where each task is tested on two different domain datasets.

{{</citation>}}


### (28/90) AQUALLM: Audio Question Answering Data Generation Using Large Language Models (Swarup Ranjan Behera et al., 2023)

{{<citation>}}

Swarup Ranjan Behera, Krishna Mohan Injeti, Jaya Sai Kiran Patibandla, Praveen Kumar Pokala, Balakrishna Reddy Pailla. (2023)  
**AQUALLM: Audio Question Answering Data Generation Using Large Language Models**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-AI, cs-CL, cs-LG, cs-MM, cs-SD, cs.CL, eess-AS  
Keywords: Language Model, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2312.17343v1)  

---


**ABSTRACT**  
Audio Question Answering (AQA) constitutes a pivotal task in which machines analyze both audio signals and natural language questions to produce precise natural language answers. The significance of possessing high-quality, diverse, and extensive AQA datasets cannot be overstated when aiming for the precision of an AQA system. While there has been notable focus on developing accurate and efficient AQA models, the creation of high-quality, diverse, and extensive datasets for the specific task at hand has not garnered considerable attention. To address this challenge, this work makes several contributions. We introduce a scalable AQA data generation pipeline, denoted as the AQUALLM framework, which relies on Large Language Models (LLMs). This framework utilizes existing audio-caption annotations and incorporates state-of-the-art LLMs to generate expansive, high-quality AQA datasets. Additionally, we present three extensive and high-quality benchmark datasets for AQA, contributing significantly to the progression of AQA research. AQA models trained on the proposed datasets set superior benchmarks compared to the existing state-of-the-art. Moreover, models trained on our datasets demonstrate enhanced generalizability when compared to models trained using human-annotated AQA data. Code and datasets will be accessible on GitHub~\footnote{\url{https://github.com/swarupbehera/AQUALLM}}.

{{</citation>}}


### (29/90) Exploring Nature: Datasets and Models for Analyzing Nature-Related Disclosures (Tobias Schimanski et al., 2023)

{{<citation>}}

Tobias Schimanski, Chiara Colesanti Senni, Glen Gostlow, Jingwei Ni, Tingyu Yu, Markus Leippold. (2023)  
**Exploring Nature: Datasets and Models for Analyzing Nature-Related Disclosures**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL, econ-GN, q-fin-EC  
Keywords: Financial  
[Paper Link](http://arxiv.org/abs/2312.17337v1)  

---


**ABSTRACT**  
Nature is an amorphous concept. Yet, it is essential for the planet's well-being to understand how the economy interacts with it. To address the growing demand for information on corporate nature disclosure, we provide datasets and classifiers to detect nature communication by companies. We ground our approach in the guidelines of the Taskforce on Nature-related Financial Disclosures (TNFD). Particularly, we focus on the specific dimensions of water, forest, and biodiversity. For each dimension, we create an expert-annotated dataset with 2,200 text samples and train classifier models. Furthermore, we show that nature communication is more prevalent in hotspot areas and directly effected industries like agriculture and utilities. Our approach is the first to respond to calls to assess corporate nature communication on a large scale.

{{</citation>}}


### (30/90) Virtual Scientific Companion for Synchrotron Beamlines: A Prototype (Daniel Potemkin et al., 2023)

{{<citation>}}

Daniel Potemkin, Carlos Soto, Ruipeng Li, Kevin Yager, Esther Tsai. (2023)  
**Virtual Scientific Companion for Synchrotron Beamlines: A Prototype**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.17180v1)  

---


**ABSTRACT**  
The extraordinarily high X-ray flux and specialized instrumentation at synchrotron beamlines have enabled versatile in-situ and high throughput studies that are impossible elsewhere. Dexterous and efficient control of experiments are thus crucial for efficient beamline operation. Artificial intelligence and machine learning methods are constantly being developed to enhance facility performance, but the full potential of these developments can only be reached with efficient human-computer-interaction. Natural language is the most intuitive and efficient way for humans to communicate. However, the low credibility and reproducibility of existing large language models and tools demand extensive development to be made for robust and reliable performance for scientific purposes. In this work, we introduce the prototype of virtual scientific companion (VISION) and demonstrate that it is possible to control basic beamline operations through natural language with open-source language model and the limited computational resources at beamline. The human-AI nature of VISION leverages existing automation systems and data framework at synchrotron beamlines.

{{</citation>}}


### (31/90) Large Language Model for Causal Decision Making (Haitao Jiang et al., 2023)

{{<citation>}}

Haitao Jiang, Lin Ge, Yuhe Gao, Jianian Wang, Rui Song. (2023)  
**Large Language Model for Causal Decision Making**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL, stat-ML  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2312.17122v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have shown their success in language understanding and reasoning on general topics. However, their capability to inference based on user-specified structured data and knowledge in corpus-rare concepts like causal decision-making is still limited. In this work, we explore the possibility of fine-tuning an open-sourced LLM into LLM4Causal, which can identify the causal task, execute a corresponding function, and interpret its numerical results based on users' queries and the provided dataset. Meanwhile, we propose a data generation process for more controllable GPT prompting and present two instruction-tuning datasets: (1) Causal-Retrieval-Bench for causal problem identification and input parameter extraction for causal function calling and (2) Causal-Interpret-Bench for in-context causal interpretation. With three case studies, we showed that LLM4Causal can deliver end-to-end solutions for causal problems and provide easy-to-understand answers. Numerical studies also reveal that it has a remarkable ability to identify the correct causal task given a query.

{{</citation>}}


### (32/90) Generative AI for Math: Part I -- MathPile: A Billion-Token-Scale Pretraining Corpus for Math (Zengzhi Wang et al., 2023)

{{<citation>}}

Zengzhi Wang, Rui Xia, Pengfei Liu. (2023)  
**Generative AI for Math: Part I -- MathPile: A Billion-Token-Scale Pretraining Corpus for Math**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2312.17120v1)  

---


**ABSTRACT**  
High-quality, large-scale corpora are the cornerstone of building foundation models. In this work, we introduce \textsc{MathPile}, a diverse and high-quality math-centric corpus comprising about 9.5 billion tokens. Throughout its creation, we adhered to the principle of ``\emph{less is more}'', firmly believing in the supremacy of data quality over quantity, even in the pre-training phase. Our meticulous data collection and processing efforts included a complex suite of preprocessing, prefiltering, language identification, cleaning, filtering, and deduplication, ensuring the high quality of our corpus. Furthermore, we performed data contamination detection on downstream benchmark test sets to eliminate duplicates. We hope our \textsc{MathPile} can help to enhance the mathematical reasoning abilities of language models. We plan to open-source different versions of \mathpile with the scripts used for processing, to facilitate future developments in this field.

{{</citation>}}


### (33/90) How Far Are We from Believable AI Agents? A Framework for Evaluating the Believability of Human Behavior Simulation (Yang Xiao et al., 2023)

{{<citation>}}

Yang Xiao, Yi Cheng, Jinlan Fu, Jiashuo Wang, Wenjie Li, Pengfei Liu. (2023)  
**How Far Are We from Believable AI Agents? A Framework for Evaluating the Believability of Human Behavior Simulation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs.CL  
Keywords: AI, GPT, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2312.17115v1)  

---


**ABSTRACT**  
Human behavior simulation of AI agents necessitates the agents to possess a quality of believability, which is crucial as it facilitates users in establishing trust toward the agents and streamlines the fulfillment of the agents' goal. While recent advancements in Large Language Model (LLM) based agents have improved human behavior simulation, challenges inherent to LLMs (e.g., long context modeling) can undermine their believability. Consequently, evaluating AI agent believability becomes imperative. Unfortunately, prior research often neglects the negative impacts of LLM deficiencies. To address these gaps, we introduce two metrics for assessing LLM-based agent believability: consistency, and robustness, together with a benchmark, SimulateBench, with which, we evaluate the consistency and robustness of agents implemented with popular LLMs. We find that agents (i) struggle to accurately depict character information when presented with lengthy profile inputs; (ii) exhibit vulnerability to profile perturbations; and (iii) are significantly affected by certain key factors that impact their overall believability. Code and SimulateBench are public at https://github.com/GAIR-NLP/GPTMan.

{{</citation>}}


### (34/90) Structured Packing in LLM Training Improves Long Context Utilization (Konrad Staniszewski et al., 2023)

{{<citation>}}

Konrad Staniszewski, Szymon Tworkowski, Sebastian Jaszczur, Henryk Michalewski, Łukasz Kuciński, Piotr Miłoś. (2023)  
**Structured Packing in LLM Training Improves Long Context Utilization**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.17296v1)  

---


**ABSTRACT**  
Recent advances in long-context Large Language Models (LCLMs) have generated significant interest, especially in applications such as querying scientific research papers. However, their potential is often limited by inadequate context utilization. We identify the absence of long-range semantic dependencies in typical training data as a primary hindrance. To address this, we delve into the benefits of frequently incorporating related documents into training inputs. Using the inherent directory structure of code data as a source of training examples, we demonstrate improvements in perplexity, even for tasks unrelated to coding. Building on these findings, but with a broader focus, we introduce Structured Packing for Long Context (SPLiCe). SPLiCe is an innovative method for creating training examples by using a retrieval method to collate the most mutually relevant documents into a single training context. Our results indicate that \method{} enhances model performance and can be used to train large models to utilize long contexts better. We validate our results by training a large $3$B model, showing both perplexity improvements and better long-context performance on downstream tasks.

{{</citation>}}


### (35/90) Challenge LLMs to Reason About Reasoning: A Benchmark to Unveil Cognitive Depth in LLMs (Zhongshen Zeng et al., 2023)

{{<citation>}}

Zhongshen Zeng, Pengguang Chen, Haiyun Jiang, Jiaya Jia. (2023)  
**Challenge LLMs to Reason About Reasoning: A Benchmark to Unveil Cognitive Depth in LLMs**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2312.17080v1)  

---


**ABSTRACT**  
In this work, we introduce a novel evaluation paradigm for Large Language Models, one that challenges them to engage in meta-reasoning. This approach addresses critical shortcomings in existing math problem-solving benchmarks, traditionally used to evaluate the cognitive capabilities of agents. Our paradigm shifts the focus from result-oriented assessments, which often overlook the reasoning process, to a more holistic evaluation that effectively differentiates the cognitive capabilities among models. For example, in our benchmark, GPT-4 demonstrates a performance ten times more accurate than GPT3-5. The significance of this new paradigm lies in its ability to reveal potential cognitive deficiencies in LLMs that current benchmarks, such as GSM8K, fail to uncover due to their saturation and lack of effective differentiation among varying reasoning abilities. Our comprehensive analysis includes several state-of-the-art math models from both open-source and closed-source communities, uncovering fundamental deficiencies in their training and evaluation approaches. This paper not only advocates for a paradigm shift in the assessment of LLMs but also contributes to the ongoing discourse on the trajectory towards Artificial General Intelligence (AGI). By promoting the adoption of meta-reasoning evaluation methods similar to ours, we aim to facilitate a more accurate assessment of the true cognitive abilities of LLMs.

{{</citation>}}


### (36/90) Length Extrapolation of Transformers: A Survey from the Perspective of Position Encoding (Liang Zhao et al., 2023)

{{<citation>}}

Liang Zhao, Xiaocheng Feng, Xiachong Feng, Bing Qin, Ting Liu. (2023)  
**Length Extrapolation of Transformers: A Survey from the Perspective of Position Encoding**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.17044v2)  

---


**ABSTRACT**  
Transformer has taken the natural language processing (NLP) field by storm since birth, owing to its superior ability to model complex dependencies in sequences. Despite the great success of pretrained language models (PLMs) based on Transformer across almost all NLP tasks, they all suffer from a preset length limit and thus can hardly extend this success to longer sequences beyond seen data, namely the length extrapolation problem. Length extrapolation has aroused great interest among researchers, as it is the core feature of human language capacity. To enhance length extrapolation of Transformers, a plethora of methods have been proposed, mostly focusing on extrapolatable position encodings. In this article, we provide an organized and systematical review of these research efforts in a unified notation from a position encoding perspective, aiming to enable the reader to gain a deep understanding of existing methods and provide stimuli for future research.

{{</citation>}}


### (37/90) Effect of dimensionality change on the bias of word embeddings (Rohit Raj Rai et al., 2023)

{{<citation>}}

Rohit Raj Rai, Amit Awekar. (2023)  
**Effect of dimensionality change on the bias of word embeddings**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2312.17292v1)  

---


**ABSTRACT**  
Word embedding methods (WEMs) are extensively used for representing text data. The dimensionality of these embeddings varies across various tasks and implementations. The effect of dimensionality change on the accuracy of the downstream task is a well-explored question. However, how the dimensionality change affects the bias of word embeddings needs to be investigated. Using the English Wikipedia corpus, we study this effect for two static (Word2Vec and fastText) and two context-sensitive (ElMo and BERT) WEMs. We have two observations. First, there is a significant variation in the bias of word embeddings with the dimensionality change. Second, there is no uniformity in how the dimensionality change affects the bias of word embeddings. These factors should be considered while selecting the dimensionality of word embeddings.

{{</citation>}}


### (38/90) Few-shot learning for automated content analysis: Efficient coding of arguments and claims in the debate on arms deliveries to Ukraine (Jonas Rieger et al., 2023)

{{<citation>}}

Jonas Rieger, Kostiantyn Yanchenko, Mattes Ruckdeschel, Gerret von Nordheim, Katharina Kleinen-von Königslöw, Gregor Wiedemann. (2023)  
**Few-shot learning for automated content analysis: Efficient coding of arguments and claims in the debate on arms deliveries to Ukraine**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL, stat-ML  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2312.16975v1)  

---


**ABSTRACT**  
Pre-trained language models (PLM) based on transformer neural networks developed in the field of natural language processing (NLP) offer great opportunities to improve automatic content analysis in communication science, especially for the coding of complex semantic categories in large datasets via supervised machine learning. However, three characteristics so far impeded the widespread adoption of the methods in the applying disciplines: the dominance of English language models in NLP research, the necessary computing resources, and the effort required to produce training data to fine-tune PLMs. In this study, we address these challenges by using a multilingual transformer model in combination with the adapter extension to transformers, and few-shot learning methods. We test our approach on a realistic use case from communication science to automatically detect claims and arguments together with their stance in the German news debate on arms deliveries to Ukraine. In three experiments, we evaluate (1) data preprocessing strategies and model variants for this task, (2) the performance of different few-shot learning methods, and (3) how well the best setup performs on varying training set sizes in terms of validity, reliability, replicability and reproducibility of the results. We find that our proposed combination of transformer adapters with pattern exploiting training provides a parameter-efficient and easily shareable alternative to fully fine-tuning PLMs. It performs on par in terms of validity, while overall, provides better properties for application in communication studies. The results also show that pre-fine-tuning for a task on a near-domain dataset leads to substantial improvement, in particular in the few-shot setting. Further, the results indicate that it is useful to bias the dataset away from the viewpoints of specific prominent individuals.

{{</citation>}}


### (39/90) AI Content Self-Detection for Transformer-based Large Language Models (Antônio Junior Alves Caiado et al., 2023)

{{<citation>}}

Antônio Junior Alves Caiado, Michael Hahsler. (2023)  
**AI Content Self-Detection for Transformer-based Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, ChatGPT, GPT, Google, Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2312.17289v1)  

---


**ABSTRACT**  
$ $The usage of generative artificial intelligence (AI) tools based on large language models, including ChatGPT, Bard, and Claude, for text generation has many exciting applications with the potential for phenomenal productivity gains. One issue is authorship attribution when using AI tools. This is especially important in an academic setting where the inappropriate use of generative AI tools may hinder student learning or stifle research by creating a large amount of automatically generated derivative work. Existing plagiarism detection systems can trace the source of submitted text but are not yet equipped with methods to accurately detect AI-generated text. This paper introduces the idea of direct origin detection and evaluates whether generative AI systems can recognize their output and distinguish it from human-written texts. We argue why current transformer-based models may be able to self-detect their own generated text and perform a small empirical study using zero-shot learning to investigate if that is the case. Results reveal varying capabilities of AI systems to identify their generated text. Google's Bard model exhibits the largest capability of self-detection with an accuracy of 94\%, followed by OpenAI's ChatGPT with 83\%. On the other hand, Anthropic's Claude model seems to be not able to self-detect.

{{</citation>}}


### (40/90) Unified Lattice Graph Fusion for Chinese Named Entity Recognition (Dixiang Zhang et al., 2023)

{{<citation>}}

Dixiang Zhang, Junyu Lu, Pingjian Zhang. (2023)  
**Unified Lattice Graph Fusion for Chinese Named Entity Recognition**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NER, Named Entity Recognition  
[Paper Link](http://arxiv.org/abs/2312.16917v1)  

---


**ABSTRACT**  
Integrating lexicon into character-level sequence has been proven effective to leverage word boundary and semantic information in Chinese named entity recognition (NER). However, prior approaches usually utilize feature weighting and position coupling to integrate word information, but ignore the semantic and contextual correspondence between the fine-grained semantic units in the character-word space. To solve this issue, we propose a Unified Lattice Graph Fusion (ULGF) approach for Chinese NER. ULGF can explicitly capture various semantic and boundary relations across different semantic units with the adjacency matrix by converting the lattice structure into a unified graph. We stack multiple graph-based intra-source self-attention and inter-source cross-gating fusion layers that iteratively carry out semantic interactions to learn node representations. To alleviate the over-reliance on word information, we further propose to leverage lexicon entity classification as an auxiliary task. Experiments on four Chinese NER benchmark datasets demonstrate the superiority of our ULGF approach.

{{</citation>}}


### (41/90) Spike No More: Stabilizing the Pre-training of Large Language Models (Sho Takase et al., 2023)

{{<citation>}}

Sho Takase, Shun Kiyono, Sosuke Kobayashi, Jun Suzuki. (2023)  
**Spike No More: Stabilizing the Pre-training of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.16903v1)  

---


**ABSTRACT**  
The loss spike often occurs during pre-training of a large language model. The spikes degrade the performance of a large language model, and sometimes ruin the pre-training. Since the pre-training needs a vast computational budget, we should avoid such spikes. To investigate a cause of loss spikes, we focus on gradients of internal layers in this study. Through theoretical analyses, we introduce two causes of the exploding gradients, and provide requirements to prevent the explosion. In addition, we introduce the combination of the initialization method and a simple modification to embeddings as a method to satisfy the requirements. We conduct various experiments to verify our theoretical analyses empirically. Experimental results indicate that the combination is effective in preventing spikes during pre-training.

{{</citation>}}


### (42/90) OmniDialog: An Omnipotent Pre-training Model for Task-Oriented Dialogue System (Mingtao Yang et al., 2023)

{{<citation>}}

Mingtao Yang, See-Kiong Ng, Jinlan Fu. (2023)  
**OmniDialog: An Omnipotent Pre-training Model for Task-Oriented Dialogue System**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2312.16864v1)  

---


**ABSTRACT**  
Pre-trained conversation models (PCMs) have demonstrated remarkable results in task-oriented dialogue (TOD) systems. Many PCMs focus predominantly on dialogue management tasks like dialogue state tracking, dialogue generation tasks like response generation, or both. However, the existing PCMs seldom consider dialogue comprehension tasks, such as dialogue question answering and summarization tasks. These tasks allow PCMs to glean dialogue context from various angles. This observation naturally raises the question: Can the performance of downstream dialogue tasks be enhanced if a PCM is pre-trained on dialogue management, generation, and comprehension tasks?   To investigate this, we proposed an Omnipotent Dialogue pre-training model (OmniDialog). It unifies these three dialogue tasks into a monolithic framework by multi-task learning, fostering inter-task communication. The pre-training corpus of OmniDialog spans $\mathbf{7}$ dialogue-focused tasks, drawing from $\mathbf{15}$ datasets and encompassing over $\mathbf{3.2}$ million dialogue utterances. To our knowledge, OmniDialog is a pioneering PCM pre-trained across dialogue management, generation, and comprehension domains. We evaluated its performance across four tasks: dialogue summarization, end-to-end dialogue modeling, dialogue state tracking, and intent classification. The results underscore its efficacy in domain transfer learning, low-resource, and full-dataset scenarios. Furthermore, to glean a nuanced understanding of OmniDialog's strengths and potential pitfalls, we designed a fine-grained analysis framework for dialogue-centric tasks. Experimental results show that the OmniDialog is good at hard samples, such as long dialogues and lengthy responses.

{{</citation>}}


### (43/90) Evaluating the Performance of Large Language Models for Spanish Language in Undergraduate Admissions Exams (Sabino Miranda et al., 2023)

{{<citation>}}

Sabino Miranda, Obdulia Pichardo-Lagunas, Bella Martínez-Seis, Pierre Baldi. (2023)  
**Evaluating the Performance of Large Language Models for Spanish Language in Undergraduate Admissions Exams**  

---
Primary Category: cs.CL  
Categories: 68T50, I-2-7, cs-AI, cs-CL, cs.CL  
Keywords: BARD, GPT, GPT-3.5, Language Model  
[Paper Link](http://arxiv.org/abs/2312.16845v1)  

---


**ABSTRACT**  
This study evaluates the performance of large language models, specifically GPT-3.5 and BARD (supported by Gemini Pro model), in undergraduate admissions exams proposed by the National Polytechnic Institute in Mexico. The exams cover Engineering/Mathematical and Physical Sciences, Biological and Medical Sciences, and Social and Administrative Sciences. Both models demonstrated proficiency, exceeding the minimum acceptance scores for respective academic programs to up to 75% for some academic programs. GPT-3.5 outperformed BARD in Mathematics and Physics, while BARD performed better in History and questions related to factual information. Overall, GPT-3.5 marginally surpassed BARD with scores of 60.94% and 60.42%, respectively.

{{</citation>}}


### (44/90) Hiding in Plain Sight: Towards the Science of Linguistic Steganography (Leela Raj-Sankar et al., 2023)

{{<citation>}}

Leela Raj-Sankar, S. Raj Rajagopalan. (2023)  
**Hiding in Plain Sight: Towards the Science of Linguistic Steganography**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CR, cs.CL  
Keywords: Sentiment Analysis  
[Paper Link](http://arxiv.org/abs/2312.16840v1)  

---


**ABSTRACT**  
Covert communication (also known as steganography) is the practice of concealing a secret inside an innocuous-looking public object (cover) so that the modified public object (covert code) makes sense to everyone but only someone who knows the code can extract the secret (message). Linguistic steganography is the practice of encoding a secret message in natural language text such as spoken conversation or short public communications such as tweets.. While ad hoc methods for covert communications in specific domains exist ( JPEG images, Chinese poetry, etc), there is no general model for linguistic steganography specifically. We present a novel mathematical formalism for creating linguistic steganographic codes, with three parameters: Decodability (probability that the receiver of the coded message will decode the cover correctly), density (frequency of code words in a cover code), and detectability (probability that an attacker can tell the difference between an untampered cover compared to its steganized version). Verbal or linguistic steganography is most challenging because of its lack of artifacts to hide the secret message in. We detail a practical construction in Python of a steganographic code for Tweets using inserted words to encode hidden digits while using n-gram frequency distortion as the measure of detectability of the insertions. Using the publicly accessible Stanford Sentiment Analysis dataset we implemented the tweet steganization scheme -- a codeword (an existing word in the data set) inserted in random positions in random existing tweets to find the tweet that has the least possible n-gram distortion. We argue that this approximates KL distance in a localized manner at low cost and thus we get a linguistic steganography scheme that is both formal and practical and permits a tradeoff between codeword density and detectability of the covert message.

{{</citation>}}


### (45/90) Adversarial Representation with Intra-Modal and Inter-Modal Graph Contrastive Learning for Multimodal Emotion Recognition (Yuntao Shou et al., 2023)

{{<citation>}}

Yuntao Shou, Tao Meng, Wei Ai, Keqin Li. (2023)  
**Adversarial Representation with Intra-Modal and Inter-Modal Graph Contrastive Learning for Multimodal Emotion Recognition**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Contrastive Learning, Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2312.16778v1)  

---


**ABSTRACT**  
With the release of increasing open-source emotion recognition datasets on social media platforms and the rapid development of computing resources, multimodal emotion recognition tasks (MER) have begun to receive widespread research attention. The MER task extracts and fuses complementary semantic information from different modalities, which can classify the speaker's emotions. However, the existing feature fusion methods have usually mapped the features of different modalities into the same feature space for information fusion, which can not eliminate the heterogeneity between different modalities. Therefore, it is challenging to make the subsequent emotion class boundary learning. To tackle the above problems, we have proposed a novel Adversarial Representation with Intra-Modal and Inter-Modal Graph Contrastive for Multimodal Emotion Recognition (AR-IIGCN) method. Firstly, we input video, audio, and text features into a multi-layer perceptron (MLP) to map them into separate feature spaces. Secondly, we build a generator and a discriminator for the three modal features through adversarial representation, which can achieve information interaction between modalities and eliminate heterogeneity among modalities. Thirdly, we introduce contrastive graph representation learning to capture intra-modal and inter-modal complementary semantic information and learn intra-class and inter-class boundary information of emotion categories. Specifically, we construct a graph structure for three modal features and perform contrastive representation learning on nodes with different emotions in the same modality and the same emotion in different modalities, which can improve the feature representation ability of nodes. Extensive experimental works show that the ARL-IIGCN method can significantly improve emotion recognition accuracy on IEMOCAP and MELD datasets.

{{</citation>}}


### (46/90) Graph Neural Networks for Antisocial Behavior Detection on Twitter (Martina Toshevska et al., 2023)

{{<citation>}}

Martina Toshevska, Slobodan Kalajdziski, Sonja Gievska. (2023)  
**Graph Neural Networks for Antisocial Behavior Detection on Twitter**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Graph Neural Network, Graph Neural Networks, Twitter  
[Paper Link](http://arxiv.org/abs/2312.16755v1)  

---


**ABSTRACT**  
Social media resurgence of antisocial behavior has exerted a downward spiral on stereotypical beliefs, and hateful comments towards individuals and social groups, as well as false or distorted news. The advances in graph neural networks employed on massive quantities of graph-structured data raise high hopes for the future of mediating communication on social media platforms. An approach based on graph convolutional data was employed to better capture the dependencies between the heterogeneous types of data.   Utilizing past and present experiences on the topic, we proposed and evaluated a graph-based approach for antisocial behavior detection, with general applicability that is both language- and context-independent. In this research, we carried out an experimental validation of our graph-based approach on several PAN datasets provided as part of their shared tasks, that enable the discussion of the results obtained by the proposed solution.

{{</citation>}}


## cs.CR (4)



### (47/90) SentinelLMs: Encrypted Input Adaptation and Fine-tuning of Language Models for Private and Secure Inference (Abhijit Mishra et al., 2023)

{{<citation>}}

Abhijit Mishra, Mingda Li, Soham Deo. (2023)  
**SentinelLMs: Encrypted Input Adaptation and Fine-tuning of Language Models for Private and Secure Inference**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CL, cs-CR, cs-LG, cs.CR  
Keywords: AI, BERT, Language Model  
[Paper Link](http://arxiv.org/abs/2312.17342v1)  

---


**ABSTRACT**  
This paper addresses the privacy and security concerns associated with deep neural language models, which serve as crucial components in various modern AI-based applications. These models are often used after being pre-trained and fine-tuned for specific tasks, with deployment on servers accessed through the internet. However, this introduces two fundamental risks: (a) the transmission of user inputs to the server via the network gives rise to interception vulnerabilities, and (b) privacy concerns emerge as organizations that deploy such models store user data with restricted context. To address this, we propose a novel method to adapt and fine-tune transformer-based language models on passkey-encrypted user-specific text. The original pre-trained language model first undergoes a quick adaptation (without any further pre-training) with a series of irreversible transformations applied to the tokenizer and token embeddings. This enables the model to perform inference on encrypted inputs while preventing reverse engineering of text from model parameters and intermediate outputs. After adaptation, models are fine-tuned on encrypted versions of existing training datasets. Experimental evaluation employing adapted versions of renowned models (e.g., BERT, RoBERTa) across established benchmark English and multilingual datasets for text classification and sequence labeling shows that encrypted models achieve performance parity with their original counterparts. This serves to safeguard performance, privacy, and security cohesively.

{{</citation>}}


### (48/90) Explainability-Based Adversarial Attack on Graphs Through Edge Perturbation (Dibaloke Chanda et al., 2023)

{{<citation>}}

Dibaloke Chanda, Saba Heidari Gheshlaghi, Nasim Yahya Soltani. (2023)  
**Explainability-Based Adversarial Attack on Graphs Through Edge Perturbation**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs.CR  
Keywords: Adversarial Attack, GNN  
[Paper Link](http://arxiv.org/abs/2312.17301v1)  

---


**ABSTRACT**  
Despite the success of graph neural networks (GNNs) in various domains, they exhibit susceptibility to adversarial attacks. Understanding these vulnerabilities is crucial for developing robust and secure applications. In this paper, we investigate the impact of test time adversarial attacks through edge perturbations which involve both edge insertions and deletions. A novel explainability-based method is proposed to identify important nodes in the graph and perform edge perturbation between these nodes. The proposed method is tested for node classification with three different architectures and datasets. The results suggest that introducing edges between nodes of different classes has higher impact as compared to removing edges among nodes within the same class.

{{</citation>}}


### (49/90) Improving Intrusion Detection with Domain-Invariant Representation Learning in Latent Space (Padmaksha Roy et al., 2023)

{{<citation>}}

Padmaksha Roy, Tyler Cody, Himanshu Singhal, Kevin Choi, Ming Jin. (2023)  
**Improving Intrusion Detection with Domain-Invariant Representation Learning in Latent Space**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs.CR  
Keywords: Intrusion Detection, Representation Learning  
[Paper Link](http://arxiv.org/abs/2312.17300v1)  

---


**ABSTRACT**  
Domain generalization focuses on leveraging knowledge from multiple related domains with ample training data and labels to enhance inference on unseen in-distribution (IN) and out-of-distribution (OOD) domains. In our study, we introduce a two-phase representation learning technique using multi-task learning. This approach aims to cultivate a latent space from features spanning multiple domains, encompassing both native and cross-domains, to amplify generalization to IN and OOD territories. Additionally, we attempt to disentangle the latent space by minimizing the mutual information between the prior and latent space, effectively de-correlating spurious feature correlations. Collectively, the joint optimization will facilitate domain-invariant feature learning. We assess the model's efficacy across multiple cybersecurity datasets, using standard classification metrics on both unseen IN and OOD sets, and juxtapose the results with contemporary domain generalization methods.

{{</citation>}}


### (50/90) BlackboxBench: A Comprehensive Benchmark of Black-box Adversarial Attacks (Meixi Zheng et al., 2023)

{{<citation>}}

Meixi Zheng, Xuanchen Yan, Zihao Zhu, Hongrui Chen, Baoyuan Wu. (2023)  
**BlackboxBench: A Comprehensive Benchmark of Black-box Adversarial Attacks**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Adversarial Attack, ImageNet  
[Paper Link](http://arxiv.org/abs/2312.16979v1)  

---


**ABSTRACT**  
Adversarial examples are well-known tools to evaluate the vulnerability of deep neural networks (DNNs). Although lots of adversarial attack algorithms have been developed, it is still challenging in the practical scenario that the model's parameters and architectures are inaccessible to the attacker/evaluator, i.e., black-box adversarial attacks. Due to the practical importance, there has been rapid progress from recent algorithms, reflected by the quick increase in attack success rate and the quick decrease in query numbers to the target model. However, there is a lack of thorough evaluations and comparisons among these algorithms, causing difficulties of tracking the real progress, analyzing advantages and disadvantages of different technical routes, as well as designing future development roadmap of this field. Thus, in this work, we aim at building a comprehensive benchmark of black-box adversarial attacks, called BlackboxBench. It mainly provides: 1) a unified, extensible and modular-based codebase, implementing 25 query-based attack algorithms and 30 transfer-based attack algorithms; 2) comprehensive evaluations: we evaluate the implemented algorithms against several mainstreaming model architectures on 2 widely used datasets (CIFAR-10 and a subset of ImageNet), leading to 14,106 evaluations in total; 3) thorough analysis and new insights, as well analytical tools. The website and source codes of BlackboxBench are available at https://blackboxbench.github.io/ and https://github.com/SCLBD/BlackboxBench/, respectively.

{{</citation>}}


## cs.SI (3)



### (51/90) Unmasking information manipulation: A quantitative approach to detecting Copy-pasta, Rewording, and Translation on Social Media (Manon Richard et al., 2023)

{{<citation>}}

Manon Richard, Lisa Giordani, Cristian Brokate, Jean Liénard. (2023)  
**Unmasking information manipulation: A quantitative approach to detecting Copy-pasta, Rewording, and Translation on Social Media**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: AI, ChatGPT, GPT, Social Media, Twitter  
[Paper Link](http://arxiv.org/abs/2312.17338v1)  

---


**ABSTRACT**  
This study proposes a comprehensive methodology for identifying three techniques utilized in foreign-operated information manipulation campaigns: Copy-Pasta, Rewording, and Translation. Our approach, dubbed the ``$3\Delta$-space duplicate methodology'', quantifies the semantic, grapheme, and language aspects of messages. Computing pairwise distances within these dimensions enables detection of abnormally close messages that are likely part of a coordinated campaign. We validate our approach using a synthetic dataset generated with ChatGPT and DeepL, further applying it to a real-world dataset on Venezuelan actors from Twitter Transparency. Our method successfully identifies all three types of inauthentic duplicates in the synthetic dataset, and is able to uncover inauthentic duplicates across political, commercial, and entertainment contexts in the Twitter dataset. The distinct focus on clustered alterations to messages, rather than individual messages, makes our approach efficient and effective at detecting large-scale instances of textual manipulation, including AI-generated ones. Moreover, our method offers a robust tool for identifying translated content, overlooked in previous research. This research also represents the first comprehensive analysis of copy-pasta detection, providing a reliable technique for tracking duplicate textual content across social networks.

{{</citation>}}


### (52/90) Perspectives of Global and Hong Kong's Media on China's Belt and Road Initiative (Le Cong Khoo et al., 2023)

{{<citation>}}

Le Cong Khoo, Anwitaman Datta. (2023)  
**Perspectives of Global and Hong Kong's Media on China's Belt and Road Initiative**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2312.17013v1)  

---


**ABSTRACT**  
This study delves into the media analysis of China's ambitious Belt and Road Initiative (BRI), which, in a polarized world, and furthermore, owing to the very polarizing nature of the initiative itself, has received both strong criticisms and conversely positive coverage in media from across the world. In that context, Hong Kong's dynamic media environment, with a particular focus on its drastically changing press freedom before and after the implementation of the National Security Law is of further interest.   Leveraging data science techniques, this study employs Global Database of Events, Language, and Tone (GDELT) to comprehensively collect and analyse (English) news articles on the BRI. Through sentiment analysis, we uncover patterns in media coverage over different periods from several countries across the globe, and delve further to investigate the the media situation in the Hong Kong region. This work thus provides valuable insights into how the Belt and Road Initiative has been portrayed in the media and its evolving reception on the global stage, with a specific emphasis on the unique media landscape of Hong Kong.   In an era characterised by increasing globalisation and inter-connectivity, but also competition for influence, animosity and trade-wars, understanding the perceptions and coverage of such significant international projects is crucial. This work stands as an interdisciplinary endeavour merging geopolitical science and data science to uncover the intricate dynamics of media coverage in general, and with an added emphasis on Hong Kong.

{{</citation>}}


### (53/90) A Generalization of the Sugeno integral to aggregate Interval-valued data: an application to Brain Computer Interface and Social Network Analysis (Javier Fumanal-Idocin et al., 2023)

{{<citation>}}

Javier Fumanal-Idocin, Zdenko Takac, Lubomira Horanska, Thiago da Cruz Asmus, Carmen Vidaurre, Graçaliz Dimuro, Javier Fernandez, Humberto Bustince. (2023)  
**A Generalization of the Sugeno integral to aggregate Interval-valued data: an application to Brain Computer Interface and Social Network Analysis**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Social Network  
[Paper Link](http://arxiv.org/abs/2312.17012v1)  

---


**ABSTRACT**  
Intervals are a popular way to represent the uncertainty related to data, in which we express the vagueness of each observation as the width of the interval. However, when using intervals for this purpose, we need to use the appropriate set of mathematical tools to work with. This can be problematic due to the scarcity and complexity of interval-valued functions in comparison with the numerical ones. In this work, we propose to extend a generalization of the Sugeno integral to work with interval-valued data. Then, we use this integral to aggregate interval-valued data in two different settings: first, we study the use of intervals in a brain-computer interface; secondly, we study how to construct interval-valued relationships in a social network, and how to aggregate their information. Our results show that interval-valued data can effectively model some of the uncertainty and coalitions of the data in both cases. For the case of brain-computer interface, we found that our results surpassed the results of other interval-valued functions.

{{</citation>}}


## cs.CV (23)



### (54/90) An Improved Baseline for Reasoning Segmentation with Large Language Model (Senqiao Yang et al., 2023)

{{<citation>}}

Senqiao Yang, Tianyuan Qu, Xin Lai, Zhuotao Tian, Bohao Peng, Shu Liu, Jiaya Jia. (2023)  
**An Improved Baseline for Reasoning Segmentation with Large Language Model**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Dialog, Dialogue, Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2312.17240v1)  

---


**ABSTRACT**  
While LISA effectively bridges the gap between segmentation and large language models to enable reasoning segmentation, it poses certain limitations: unable to distinguish different instances of the target region, and constrained by the pre-defined textual response formats. In this work, we introduce LISA++, an update to the existing LISA model, focusing on improving core functionalities while keeping the base architecture intact. The main enhancements in LISA++ include: \textbf{1) Enhanced Segmentation}: The instance segmentation ability has been added, providing a more detailed scene analysis along with the existing multi-region semantic segmentation. \textbf{2) More Natural Conversation}: Improved capability for multi-turn dialogue, with the ability to incorporate segmentation results directly into text responses, i.e., Segmentation in Dialogue (SiD). These improvements are achieved by curating the existing samples of generic segmentation datasets, aimed specifically at enhancing the segmentation and conversational skills without structural change and additional data sources. Comparative analysis with the original LISA model shows significant advancements in these areas, positioning LISA++ as a notable upgrade in visual understanding and interaction. LISA++'s adaptability and improved features highlight the versatility of the mask-as-embedding paradigm proposed by LISA, and the potential as a foundational model for diverse applications.

{{</citation>}}


### (55/90) A Simple LLM Framework for Long-Range Video Question-Answering (Ce Zhang et al., 2023)

{{<citation>}}

Ce Zhang, Taixi Lu, Md Mohaiminul Islam, Ziyang Wang, Shoubin Yu, Mohit Bansal, Gedas Bertasius. (2023)  
**A Simple LLM Framework for Long-Range Video Question-Answering**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT, GPT-3.5, GPT-4, Language Model, QA  
[Paper Link](http://arxiv.org/abs/2312.17235v1)  

---


**ABSTRACT**  
We present LLoVi, a language-based framework for long-range video question-answering (LVQA). Unlike prior long-range video understanding methods, which are often costly and require specialized long-range video modeling design (e.g., memory queues, state-space layers, etc.), our approach uses a frame/clip-level visual captioner (e.g., BLIP2, LaViLa, LLaVA) coupled with a Large Language Model (GPT-3.5, GPT-4) leading to a simple yet surprisingly effective LVQA framework. Specifically, we decompose short and long-range modeling aspects of LVQA into two stages. First, we use a short-term visual captioner to generate textual descriptions of short video clips (0.5-8s in length) densely sampled from a long input video. Afterward, an LLM aggregates the densely extracted short-term captions to perform long-range temporal reasoning needed to understand the whole video and answer a question. To analyze what makes our simple framework so effective, we thoroughly evaluate various components of our system. Our empirical analysis reveals that the choice of the visual captioner and LLM is critical for good LVQA performance. Furthermore, we show that a specialized prompt that asks the LLM first to summarize the noisy short-term visual captions and then answer a given input question leads to a significant LVQA performance boost. On EgoSchema, which is best known as a very long-form video question-answering benchmark, our method achieves 50.3% accuracy, outperforming the previous best-performing approach by 18.1% (absolute gain). In addition, our approach outperforms the previous state-of-the-art by 4.1% and 3.1% on NeXT-QA and IntentQA. We also extend LLoVi to grounded LVQA and show that it outperforms all prior methods on the NeXT-GQA dataset. We will release our code at https://github.com/CeeZh/LLoVi.

{{</citation>}}


### (56/90) MIVC: Multiple Instance Visual Component for Visual-Language Models (Wenyi Wu et al., 2023)

{{<citation>}}

Wenyi Wu, Qi Li, Wenliang Zhong, Junzhou Huang. (2023)  
**MIVC: Multiple Instance Visual Component for Visual-Language Models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.17109v1)  

---


**ABSTRACT**  
Vision-language models have been widely explored across a wide range of tasks and achieve satisfactory performance. However, it's under-explored how to consolidate entity understanding through a varying number of images and to align it with the pre-trained language models for generative tasks. In this paper, we propose MIVC, a general multiple instance visual component to bridge the gap between various image inputs with off-the-shelf vision-language models by aggregating visual representations in a permutation-invariant fashion through a neural network. We show that MIVC could be plugged into the visual-language models to improve the model performance consistently on visual question answering, classification and captioning tasks on a public available e-commerce dataset with multiple images per product. Furthermore, we show that the component provides insight into the contribution of each image to the downstream tasks.

{{</citation>}}


### (57/90) Geometry-Biased Transformer for Robust Multi-View 3D Human Pose Reconstruction (Olivier Moliner et al., 2023)

{{<citation>}}

Olivier Moliner, Sangxia Huang, Kalle Åström. (2023)  
**Geometry-Biased Transformer for Robust Multi-View 3D Human Pose Reconstruction**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Bias, Transformer  
[Paper Link](http://arxiv.org/abs/2312.17106v1)  

---


**ABSTRACT**  
We address the challenges in estimating 3D human poses from multiple views under occlusion and with limited overlapping views. We approach multi-view, single-person 3D human pose reconstruction as a regression problem and propose a novel encoder-decoder Transformer architecture to estimate 3D poses from multi-view 2D pose sequences. The encoder refines 2D skeleton joints detected across different views and times, fusing multi-view and temporal information through global self-attention. We enhance the encoder by incorporating a geometry-biased attention mechanism, effectively leveraging geometric relationships between views. Additionally, we use detection scores provided by the 2D pose detector to further guide the encoder's attention based on the reliability of the 2D detections. The decoder subsequently regresses the 3D pose sequence from these refined tokens, using pre-defined queries for each joint. To enhance the generalization of our method to unseen scenes and improve resilience to missing joints, we implement strategies including scene centering, synthetic views, and token dropout. We conduct extensive experiments on three benchmark public datasets, Human3.6M, CMU Panoptic and Occlusion-Persons. Our results demonstrate the efficacy of our approach, particularly in occluded scenes and when few views are available, which are traditionally challenging scenarios for triangulation-based methods.

{{</citation>}}


### (58/90) Q-Align: Teaching LMMs for Visual Scoring via Discrete Text-Defined Levels (Haoning Wu et al., 2023)

{{<citation>}}

Haoning Wu, Zicheng Zhang, Weixia Zhang, Chaofeng Chen, Liang Liao, Chunyi Li, Yixuan Gao, Annan Wang, Erli Zhang, Wenxiu Sun, Qiong Yan, Xiongkuo Min, Guangtao Zhai, Weisi Lin. (2023)  
**Q-Align: Teaching LMMs for Visual Scoring via Discrete Text-Defined Levels**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs-LG, cs.CV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2312.17090v1)  

---


**ABSTRACT**  
The explosion of visual content available online underscores the requirement for an accurate machine assessor to robustly evaluate scores across diverse types of visual contents. While recent studies have demonstrated the exceptional potentials of large multi-modality models (LMMs) on a wide range of related fields, in this work, we explore how to teach them for visual rating aligned with human opinions. Observing that human raters only learn and judge discrete text-defined levels in subjective studies, we propose to emulate this subjective process and teach LMMs with text-defined rating levels instead of scores. The proposed Q-Align achieves state-of-the-art performance on image quality assessment (IQA), image aesthetic assessment (IAA), as well as video quality assessment (VQA) tasks under the original LMM structure. With the syllabus, we further unify the three tasks into one model, termed the OneAlign. In our experiments, we demonstrate the advantage of the discrete-level-based syllabus over direct-score-based variants for LMMs. Our code and the pre-trained weights are released at https://github.com/Q-Future/Q-Align.

{{</citation>}}


### (59/90) SCTNet: Single-Branch CNN with Transformer Semantic Information for Real-Time Segmentation (Zhengze Xu et al., 2023)

{{<citation>}}

Zhengze Xu, Dongyue Wu, Changqian Yu, Xiangxiang Chu, Nong Sang, Changxin Gao. (2023)  
**SCTNet: Single-Branch CNN with Transformer Semantic Information for Real-Time Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.17071v1)  

---


**ABSTRACT**  
Recent real-time semantic segmentation methods usually adopt an additional semantic branch to pursue rich long-range context. However, the additional branch incurs undesirable computational overhead and slows inference speed. To eliminate this dilemma, we propose SCTNet, a single branch CNN with transformer semantic information for real-time segmentation. SCTNet enjoys the rich semantic representations of an inference-free semantic branch while retaining the high efficiency of lightweight single branch CNN. SCTNet utilizes a transformer as the training-only semantic branch considering its superb ability to extract long-range context. With the help of the proposed transformer-like CNN block CFBlock and the semantic information alignment module, SCTNet could capture the rich semantic information from the transformer branch in training. During the inference, only the single branch CNN needs to be deployed. We conduct extensive experiments on Cityscapes, ADE20K, and COCO-Stuff-10K, and the results show that our method achieves the new state-of-the-art performance. The code and model is available at https://github.com/xzz777/SCTNet

{{</citation>}}


### (60/90) Multi-Attention Fusion Drowsy Driving Detection Model (Shulei QU et al., 2023)

{{<citation>}}

Shulei QU, Zhenguo Gao, Xiaoxiao Wu, Yuanyuan Qiu. (2023)  
**Multi-Attention Fusion Drowsy Driving Detection Model**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.17052v1)  

---


**ABSTRACT**  
Drowsy driving represents a major contributor to traffic accidents, and the implementation of driver drowsy driving detection systems has been proven to significantly reduce the occurrence of such accidents. Despite the development of numerous drowsy driving detection algorithms, many of them impose specific prerequisites such as the availability of complete facial images, optimal lighting conditions, and the use of RGB images. In our study, we introduce a novel approach called the Multi-Attention Fusion Drowsy Driving Detection Model (MAF). MAF is aimed at significantly enhancing classification performance, especially in scenarios involving partial facial occlusion and low lighting conditions. It accomplishes this by capitalizing on the local feature extraction capabilities provided by multi-attention fusion, thereby enhancing the algorithm's overall robustness. To enhance our dataset, we collected real-world data that includes both occluded and unoccluded faces captured under nighttime and daytime lighting conditions. We conducted a comprehensive series of experiments using both publicly available datasets and our self-built data. The results of these experiments demonstrate that our proposed model achieves an impressive driver drowsiness detection accuracy of 96.8%.

{{</citation>}}


### (61/90) FILP-3D: Enhancing 3D Few-shot Class-incremental Learning with Pre-trained Vision-Language Models (Wan Xu et al., 2023)

{{<citation>}}

Wan Xu, Tianyu Huang, Tianyu Qu, Guanglei Yang, Yiwen Guo, Wangmeng Zuo. (2023)  
**FILP-3D: Enhancing 3D Few-shot Class-incremental Learning with Pre-trained Vision-Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.17051v1)  

---


**ABSTRACT**  
Few-shot class-incremental learning (FSCIL) aims to mitigate the catastrophic forgetting issue when a model is incrementally trained on limited data. While the Contrastive Vision-Language Pre-Training (CLIP) model has been effective in addressing 2D few/zero-shot learning tasks, its direct application to 3D FSCIL faces limitations. These limitations arise from feature space misalignment and significant noise in real-world scanned 3D data. To address these challenges, we introduce two novel components: the Redundant Feature Eliminator (RFE) and the Spatial Noise Compensator (SNC). RFE aligns the feature spaces of input point clouds and their embeddings by performing a unique dimensionality reduction on the feature space of pre-trained models (PTMs), effectively eliminating redundant information without compromising semantic integrity. On the other hand, SNC is a graph-based 3D model designed to capture robust geometric information within point clouds, thereby augmenting the knowledge lost due to projection, particularly when processing real-world scanned data. Considering the imbalance in existing 3D datasets, we also propose new evaluation metrics that offer a more nuanced assessment of a 3D FSCIL model. Traditional accuracy metrics are proved to be biased; thus, our metrics focus on the model's proficiency in learning new classes while maintaining the balance between old and new classes. Experimental results on both established 3D FSCIL benchmarks and our dataset demonstrate that our approach significantly outperforms existing state-of-the-art methods.

{{</citation>}}


### (62/90) AI Powered Road Network Prediction with Multi-Modal Data (Necip Enes Gengec et al., 2023)

{{<citation>}}

Necip Enes Gengec, Ergin Tari, Ulas Bagci. (2023)  
**AI Powered Road Network Prediction with Multi-Modal Data**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.17040v1)  

---


**ABSTRACT**  
This study presents an innovative approach for automatic road detection with deep learning, by employing fusion strategies for utilizing both lower-resolution satellite imagery and GPS trajectory data, a concept never explored before. We rigorously investigate both early and late fusion strategies, and assess deep learning based road detection performance using different fusion settings. Our extensive ablation studies assess the efficacy of our framework under diverse model architectures, loss functions, and geographic domains (Istanbul and Montreal). For an unbiased and complete evaluation of road detection results, we use both region-based and boundary-based evaluation metrics for road segmentation. The outcomes reveal that the ResUnet model outperforms U-Net and D-Linknet in road extraction tasks, achieving superior results over the benchmark study using low-resolution Sentinel-2 data. This research not only contributes to the field of automatic road detection but also offers novel insights into the utilization of data fusion methods in diverse applications.

{{</citation>}}


### (63/90) 3DTINC: Time-Equivariant Non-Contrastive Learning for Predicting Disease Progression from Longitudinal OCTs (Taha Emre et al., 2023)

{{<citation>}}

Taha Emre, Arunava Chakravarty, Antoine Rivail, Dmitrii Lachinov, Oliver Leingang, Sophie Riedl, Julia Mai, Hendrik P. N. Scholl, Sobha Sivaprasad, Daniel Rueckert, Andrew Lotery, Ursula Schmidt-Erfurth, Hrvoje Bogunović. (2023)  
**3DTINC: Time-Equivariant Non-Contrastive Learning for Predicting Disease Progression from Longitudinal OCTs**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2312.16980v1)  

---


**ABSTRACT**  
Self-supervised learning (SSL) has emerged as a powerful technique for improving the efficiency and effectiveness of deep learning models. Contrastive methods are a prominent family of SSL that extract similar representations of two augmented views of an image while pushing away others in the representation space as negatives. However, the state-of-the-art contrastive methods require large batch sizes and augmentations designed for natural images that are impractical for 3D medical images. To address these limitations, we propose a new longitudinal SSL method, 3DTINC, based on non-contrastive learning. It is designed to learn perturbation-invariant features for 3D optical coherence tomography (OCT) volumes, using augmentations specifically designed for OCT. We introduce a new non-contrastive similarity loss term that learns temporal information implicitly from intra-patient scans acquired at different times. Our experiments show that this temporal information is crucial for predicting progression of retinal diseases, such as age-related macular degeneration (AMD). After pretraining with 3DTINC, we evaluated the learned representations and the prognostic models on two large-scale longitudinal datasets of retinal OCTs where we predict the conversion to wet-AMD within a six months interval. Our results demonstrate that each component of our contributions is crucial for learning meaningful representations useful in predicting disease progression from longitudinal volumetric scans.

{{</citation>}}


### (64/90) SAR-Net: Multi-scale Direction-aware SAR Network via Global Information Fusion (Mingxiang Cao et al., 2023)

{{<citation>}}

Mingxiang Cao, Jie Lei, Weiying Xie, Jiaqing Zhang, Daixun Li, Yunsong Li. (2023)  
**SAR-Net: Multi-scale Direction-aware SAR Network via Global Information Fusion**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Attention, Embedding  
[Paper Link](http://arxiv.org/abs/2312.16943v1)  

---


**ABSTRACT**  
Deep learning has driven significant progress in object detection using Synthetic Aperture Radar (SAR) imagery. Existing methods, while achieving promising results, often struggle to effectively integrate local and global information, particularly direction-aware features. This paper proposes SAR-Net, a novel framework specifically designed for global fusion of direction-aware information in SAR object detection. SAR-Net leverages two key innovations: the Unity Compensation Mechanism (UCM) and the Direction-aware Attention Module (DAM). UCM facilitates the establishment of complementary relationships among features across different scales, enabling efficient global information fusion. Among them, Multi-scale Alignment Module (MAM) and distinct Multi-level Fusion Module (MFM) enhance feature integration by capturing both texture detail and semantic information. Then, Multi-feature Embedding Module (MEM) feeds back global features into the primary branches, further improving information transmission. Additionally, DAM, through bidirectional attention polymerization, captures direction-aware information, effectively eliminating background interference. Extensive experiments demonstrate the effectiveness of SAR-Net, achieving state-of-the-art results on aircraft (SAR-AIRcraft-1.0) and ship datasets (SSDD, HRSID), confirming its generalization capability and robustness.

{{</citation>}}


### (65/90) DeLR: Active Learning for Detection with Decoupled Localization and Recognition Query (Yuhang Zhang et al., 2023)

{{<citation>}}

Yuhang Zhang, Yuang Deng, Xiaopeng Zhang, Jie Li, Robert C. Qiu, Qi Tian. (2023)  
**DeLR: Active Learning for Detection with Decoupled Localization and Recognition Query**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2312.16931v1)  

---


**ABSTRACT**  
Active learning has been demonstrated effective to reduce labeling cost, while most progress has been designed for image recognition, there still lacks instance-level active learning for object detection. In this paper, we rethink two key components, i.e., localization and recognition, for object detection, and find that the correctness of them are highly related, therefore, it is not necessary to annotate both boxes and classes if we are given pseudo annotations provided with the trained model. Motivated by this, we propose an efficient query strategy, termed as DeLR, that Decoupling the Localization and Recognition for active query. In this way, we are probably free of class annotations when the localization is correct, and able to assign the labeling budget for more informative samples. There are two main differences in DeLR: 1) Unlike previous methods mostly focus on image-level annotations, where the queried samples are selected and exhausted annotated. In DeLR, the query is based on region-level, and we only annotate the object region that is queried; 2) Instead of directly providing both localization and recognition annotations, we separately query the two components, and thus reduce the recognition budget with the pseudo class labels provided by the model. Experiments on several benchmarks demonstrate its superiority. We hope our proposed query strategy would shed light on researches in active learning in object detection.

{{</citation>}}


### (66/90) Res-Attn : An Enhanced Res-Tuning Approach with Lightweight Attention Mechanism (Chaojie Mao et al., 2023)

{{<citation>}}

Chaojie Mao, Zeyinzi Jiang. (2023)  
**Res-Attn : An Enhanced Res-Tuning Approach with Lightweight Attention Mechanism**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.16916v1)  

---


**ABSTRACT**  
Res-Tuning introduces a flexible and efficient paradigm for model tuning, showing that tuners decoupled from the backbone network can achieve performance comparable to traditional methods. Existing methods commonly construct the tuner as a set of trainable low-rank decomposition matrices, positing that a low-rank subspace suffices for adapting pre-trained foundational models to new scenarios. In this work, we present an advanced, efficient tuner augmented with low-rank attention, termed Res-Attn , which also adheres to the Res-Tuning framework. Res-Attn utilizes a parallel multi-head attention module equipped with low-rank projections for query, key, and value to execute streamlined attention operations. Through training this lightweight attention module, Res-Attn facilitates adaptation to new scenarios. Our extensive experiments across a range of discriminative and generative tasks showcase the superior performance of our method when compared to existing alternatives

{{</citation>}}


### (67/90) ROI-Aware Multiscale Cross-Attention Vision Transformer for Pest Image Identification (Ga-Eun Kim et al., 2023)

{{<citation>}}

Ga-Eun Kim, Chang-Hwan Son. (2023)  
**ROI-Aware Multiscale Cross-Attention Vision Transformer for Pest Image Identification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2312.16914v1)  

---


**ABSTRACT**  
The pests captured with imaging devices may be relatively small in size compared to the entire images, and complex backgrounds have colors and textures similar to those of the pests, which hinders accurate feature extraction and makes pest identification challenging. The key to pest identification is to create a model capable of detecting regions of interest (ROIs) and transforming them into better ones for attention and discriminative learning. To address these problems, we will study how to generate and update the ROIs via multiscale cross-attention fusion as well as how to be highly robust to complex backgrounds and scale problems. Therefore, we propose a novel ROI-aware multiscale cross-attention vision transformer (ROI-ViT). The proposed ROI-ViT is designed using dual branches, called Pest and ROI branches, which take different types of maps as input: Pest images and ROI maps. To render such ROI maps, ROI generators are built using soft segmentation and a class activation map and then integrated into the ROI-ViT backbone. Additionally, in the dual branch, complementary feature fusion and multiscale hierarchies are implemented via a novel multiscale cross-attention fusion. The class token from the Pest branch is exchanged with the patch tokens from the ROI branch, and vice versa. The experimental results show that the proposed ROI-ViT achieves 81.81%, 99.64%, and 84.66% for IP102, D0, and SauTeg pest datasets, respectively, outperforming state-of-the-art (SOTA) models, such as MViT, PVT, DeiT, Swin-ViT, and EfficientNet. More importantly, for the new challenging dataset IP102(CBSS) that contains only pest images with complex backgrounds and small sizes, the proposed model can maintain high recognition accuracy, whereas that of other SOTA models decrease sharply, demonstrating that our model is more robust to complex background and scale problems.

{{</citation>}}


### (68/90) Block Pruning for Enhanced Efficiency in Convolutional Neural Networks (Cheng-En Wu et al., 2023)

{{<citation>}}

Cheng-En Wu, Azadeh Davoodi, Yu Hen Hu. (2023)  
**Block Pruning for Enhanced Efficiency in Convolutional Neural Networks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Pruning  
[Paper Link](http://arxiv.org/abs/2312.16904v1)  

---


**ABSTRACT**  
This paper presents a novel approach to network pruning, targeting block pruning in deep neural networks for edge computing environments. Our method diverges from traditional techniques that utilize proxy metrics, instead employing a direct block removal strategy to assess the impact on classification accuracy. This hands-on approach allows for an accurate evaluation of each block's importance. We conducted extensive experiments on CIFAR-10, CIFAR-100, and ImageNet datasets using ResNet architectures. Our results demonstrate the efficacy of our method, particularly on large-scale datasets like ImageNet with ResNet50, where it excelled in reducing model size while retaining high accuracy, even when pruning a significant portion of the network. The findings underscore our method's capability in maintaining an optimal balance between model size and performance, especially in resource-constrained edge computing scenarios.

{{</citation>}}


### (69/90) Chaurah: A Smart Raspberry Pi based Parking System (Soumya Ranjan Choudhaury et al., 2023)

{{<citation>}}

Soumya Ranjan Choudhaury, Aditya Narendra, Ashutosh Mishra, Ipsit Misra. (2023)  
**Chaurah: A Smart Raspberry Pi based Parking System**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: OCR  
[Paper Link](http://arxiv.org/abs/2312.16894v1)  

---


**ABSTRACT**  
The widespread usage of cars and other large, heavy vehicles necessitates the development of an effective parking infrastructure. Additionally, algorithms for detection and recognition of number plates are often used to identify automobiles all around the world where standardized plate sizes and fonts are enforced, making recognition an effortless task. As a result, both kinds of data can be combined to develop an intelligent parking system focuses on the technology of Automatic Number Plate Recognition (ANPR). Retrieving characters from an inputted number plate image is the sole purpose of ANPR which is a costly procedure. In this article, we propose Chaurah, a minimal cost ANPR system that relies on a Raspberry Pi 3 that was specifically created for parking facilities. The system employs a dual-stage methodology, with the first stage being an ANPR system which makes use of two convolutional neural networks (CNNs). The primary locates and recognises license plates from a vehicle image, while the secondary performs Optical Character Recognition (OCR) to identify individualized numbers from the number plate. An application built with Flutter and Firebase for database administration and license plate record comparison makes up the second component of the overall solution. The application also acts as an user-interface for the billing mechanism based on parking time duration resulting in an all-encompassing software deployment of the study.

{{</citation>}}


### (70/90) Adversarial Attacks on Image Classification Models: Analysis and Defense (Jaydip Sen et al., 2023)

{{<citation>}}

Jaydip Sen, Abhiraj Sen, Ananda Chatterjee. (2023)  
**Adversarial Attacks on Image Classification Models: Analysis and Defense**  

---
Primary Category: cs.CV  
Categories: cs-CR, cs-CV, cs-LG, cs.CV  
Keywords: Adversarial Attack, Image Classification, ImageNet  
[Paper Link](http://arxiv.org/abs/2312.16880v1)  

---


**ABSTRACT**  
The notion of adversarial attacks on image classification models based on convolutional neural networks (CNN) is introduced in this work. To classify images, deep learning models called CNNs are frequently used. However, when the networks are subject to adversarial attacks, extremely potent and previously trained CNN models that perform quite effectively on image datasets for image classification tasks may perform poorly. In this work, one well-known adversarial attack known as the fast gradient sign method (FGSM) is explored and its adverse effects on the performances of image classification models are examined. The FGSM attack is simulated on three pre-trained image classifier CNN architectures, ResNet-101, AlexNet, and RegNetY 400MF using randomly chosen images from the ImageNet dataset. The classification accuracies of the models are computed in the absence and presence of the attack to demonstrate the detrimental effect of the attack on the performances of the classifiers. Finally, a mechanism is proposed to defend against the FGSM attack based on a modified defensive distillation-based approach. Extensive results are presented for the validation of the proposed scheme.

{{</citation>}}


### (71/90) DualFluidNet: an Attention-based Dual-pipeline Network for Accurate and Generalizable Fluid-solid Coupled Simulation (Yu Chen et al., 2023)

{{<citation>}}

Yu Chen, Shuai Zheng, Menglong Jin, Yan Chang, Nianyi Wang. (2023)  
**DualFluidNet: an Attention-based Dual-pipeline Network for Accurate and Generalizable Fluid-solid Coupled Simulation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-GR, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.16867v1)  

---


**ABSTRACT**  
Fluid motion can be considered as point cloud transformation when adopted by a Lagrangian description. Compared to traditional numerical analysis methods, using machine learning techniques to learn physics simulations can achieve near accuracy, while significantly increasing efficiency. In this paper, we propose an innovative approach for 3D fluid simulations utilizing an Attention-based Dual-pipeline Network, which employs a dual-pipeline architecture, seamlessly integrated with an Attention-based Feature Fusion Module. Unlike previous single-pipeline approaches, we find that a well-designed dual-pipeline approach achieves a better balance between global fluid control and physical law constraints. Furthermore, we design a Type-aware Input Module to adaptively recognize particles of different types and perform feature fusion afterward, such that fluid-solid coupling issues can be better dealt with. The experiments show that our approach significantly increases the accuracy of fluid simulation predictions and enhances generalizability to previously unseen scenarios. We demonstrate its superior performance over the state-of-the-art approaches across various metrics.

{{</citation>}}


### (72/90) TinyGPT-V: Efficient Multimodal Large Language Model via Small Backbones (Zhengqing Yuan et al., 2023)

{{<citation>}}

Zhengqing Yuan, Zhaoxu Li, Lichao Sun. (2023)  
**TinyGPT-V: Efficient Multimodal Large Language Model via Small Backbones**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2312.16862v1)  

---


**ABSTRACT**  
In the era of advanced multimodel learning, multimodal large language models (MLLMs) such as GPT-4V have made remarkable strides towards bridging language and visual elements. However, the closed-source nature and considerable computational demand present notable challenges for universal usage and modifications. This is where open-source MLLMs like LLaVA and MiniGPT-4 come in, presenting groundbreaking achievements across tasks. Despite these accomplishments, computational efficiency remains an unresolved issue, as these models, like LLaVA-v1.5-13B, require substantial resources. Addressing these issues, we introduce TinyGPT-V, a new-wave model marrying impressive performance with commonplace computational capacity. It stands out by requiring merely a 24G GPU for training and an 8G GPU or CPU for inference. Built upon Phi-2, TinyGPT-V couples an effective language backbone with pre-trained vision modules from BLIP-2 or CLIP. TinyGPT-V's 2.8B parameters can undergo a unique quantisation process, suitable for local deployment and inference tasks on 8G various devices. Our work fosters further developments for designing cost-effective, efficient, and high-performing MLLMs, expanding their applicability in a broad array of real-world scenarios. Furthermore this paper proposed a new paradigm of Multimodal Large Language Model via small backbones. Our code and training weights are placed at: https://github.com/DLYuanGod/TinyGPT-V and https://huggingface.co/Tyrannosaurus/TinyGPT-V respectively.

{{</citation>}}


### (73/90) DarkShot: Lighting Dark Images with Low-Compute and High-Quality (Jiazhang Zheng et al., 2023)

{{<citation>}}

Jiazhang Zheng, Lei Li, Qiuping Liao, Cheng Li, Li Li, Yangxing Liu. (2023)  
**DarkShot: Lighting Dark Images with Low-Compute and High-Quality**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Attention, Self-Attention  
[Paper Link](http://arxiv.org/abs/2312.16805v2)  

---


**ABSTRACT**  
Nighttime photography encounters escalating challenges in extremely low-light conditions, primarily attributable to the ultra-low signal-to-noise ratio. For real-world deployment, a practical solution must not only produce visually appealing results but also require minimal computation. However, most existing methods are either focused on improving restoration performance or employ lightweight models at the cost of quality. This paper proposes a lightweight network that outperforms existing state-of-the-art (SOTA) methods in low-light enhancement tasks while minimizing computation. The proposed network incorporates Siamese Self-Attention Block (SSAB) and Skip-Channel Attention (SCA) modules, which enhance the model's capacity to aggregate global information and are well-suited for high-resolution images. Additionally, based on our analysis of the low-light image restoration process, we propose a Two-Stage Framework that achieves superior results. Our model can restore a UHD 4K resolution image with minimal computation while keeping SOTA restoration quality.

{{</citation>}}


### (74/90) Multi-Prompts Learning with Cross-Modal Alignment for Attribute-based Person Re-Identification (Yajing Zhai et al., 2023)

{{<citation>}}

Yajing Zhai, Yawen Zeng, Zhiyong Huang, Zheng Qin, Xin Jin, Da Cao. (2023)  
**Multi-Prompts Learning with Cross-Modal Alignment for Attribute-based Person Re-Identification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ChatGPT, GPT, QA  
[Paper Link](http://arxiv.org/abs/2312.16797v1)  

---


**ABSTRACT**  
The fine-grained attribute descriptions can significantly supplement the valuable semantic information for person image, which is vital to the success of person re-identification (ReID) task. However, current ReID algorithms typically failed to effectively leverage the rich contextual information available, primarily due to their reliance on simplistic and coarse utilization of image attributes. Recent advances in artificial intelligence generated content have made it possible to automatically generate plentiful fine-grained attribute descriptions and make full use of them. Thereby, this paper explores the potential of using the generated multiple person attributes as prompts in ReID tasks with off-the-shelf (large) models for more accurate retrieval results. To this end, we present a new framework called Multi-Prompts ReID (MP-ReID), based on prompt learning and language models, to fully dip fine attributes to assist ReID task. Specifically, MP-ReID first learns to hallucinate diverse, informative, and promptable sentences for describing the query images. This procedure includes (i) explicit prompts of which attributes a person has and furthermore (ii) implicit learnable prompts for adjusting/conditioning the criteria used towards this person identity matching. Explicit prompts are obtained by ensembling generation models, such as ChatGPT and VQA models. Moreover, an alignment module is designed to fuse multi-prompts (i.e., explicit and implicit ones) progressively and mitigate the cross-modal gap. Extensive experiments on the existing attribute-involved ReID datasets, namely, Market1501 and DukeMTMC-reID, demonstrate the effectiveness and rationality of the proposed MP-ReID solution.

{{</citation>}}


### (75/90) ZONE: Zero-Shot Instruction-Guided Local Editing (Shanglin Li et al., 2023)

{{<citation>}}

Shanglin Li, Bohan Zeng, Yutang Feng, Sicheng Gao, Xuhui Liu, Jiaming Liu, Li Lin, Xu Tang, Yao Hu, Jianzhuang Liu, Baochang Zhang. (2023)  
**ZONE: Zero-Shot Instruction-Guided Local Editing**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2312.16794v1)  

---


**ABSTRACT**  
Recent advances in vision-language models like Stable Diffusion have shown remarkable power in creative image synthesis and editing.However, most existing text-to-image editing methods encounter two obstacles: First, the text prompt needs to be carefully crafted to achieve good results, which is not intuitive or user-friendly. Second, they are insensitive to local edits and can irreversibly affect non-edited regions, leaving obvious editing traces. To tackle these problems, we propose a Zero-shot instructiON-guided local image Editing approach, termed ZONE. We first convert the editing intent from the user-provided instruction (e.g., ``make his tie blue") into specific image editing regions through InstructPix2Pix. We then propose a Region-IoU scheme for precise image layer extraction from an off-the-shelf segment model. We further develop an edge smoother based on FFT for seamless blending between the layer and the image.Our method allows for arbitrary manipulation of a specific region with a single instruction while preserving the rest. Extensive experiments demonstrate that our ZONE achieves remarkable local editing results and user-friendliness, outperforming state-of-the-art methods.

{{</citation>}}


### (76/90) RL-LOGO: Deep Reinforcement Learning Localization for Logo Recognition (Masato Fujitake, 2023)

{{<citation>}}

Masato Fujitake. (2023)  
**RL-LOGO: Deep Reinforcement Learning Localization for Logo Recognition**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs-NE, cs.CV  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.16792v1)  

---


**ABSTRACT**  
This paper proposes a novel logo image recognition approach incorporating a localization technique based on reinforcement learning. Logo recognition is an image classification task identifying a brand in an image. As the size and position of a logo vary widely from image to image, it is necessary to determine its position for accurate recognition. However, because there is no annotation for the position coordinates, it is impossible to train and infer the location of the logo in the image. Therefore, we propose a deep reinforcement learning localization method for logo recognition (RL-LOGO). It utilizes deep reinforcement learning to identify a logo region in images without annotations of the positions, thereby improving classification accuracy. We demonstrated a significant improvement in accuracy compared with existing methods in several published benchmarks. Specifically, we achieved an 18-point accuracy improvement over competitive methods on the complex dataset Logo-2K+. This demonstrates that the proposed method is a promising approach to logo recognition in real-world applications.

{{</citation>}}


## math.OC (1)



### (77/90) Resilient Constrained Reinforcement Learning (Dongsheng Ding et al., 2023)

{{<citation>}}

Dongsheng Ding, Zhengyan Huan, Alejandro Ribeiro. (2023)  
**Resilient Constrained Reinforcement Learning**  

---
Primary Category: math.OC  
Categories: cs-LG, cs-SY, eess-SY, math-OC, math.OC  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.17194v1)  

---


**ABSTRACT**  
We study a class of constrained reinforcement learning (RL) problems in which multiple constraint specifications are not identified before training. It is challenging to identify appropriate constraint specifications due to the undefined trade-off between the reward maximization objective and the constraint satisfaction, which is ubiquitous in constrained decision-making. To tackle this issue, we propose a new constrained RL approach that searches for policy and constraint specifications together. This method features the adaptation of relaxing the constraint according to a relaxation cost introduced in the learning objective. Since this feature mimics how ecological systems adapt to disruptions by altering operation, our approach is termed as resilient constrained RL. Specifically, we provide a set of sufficient conditions that balance the constraint satisfaction and the reward maximization in notion of resilient equilibrium, propose a tractable formulation of resilient constrained policy optimization that takes this equilibrium as an optimal solution, and advocate two resilient constrained policy search algorithms with non-asymptotic convergence guarantees on the optimality gap and constraint satisfaction. Furthermore, we demonstrate the merits and the effectiveness of our approach in computational experiments.

{{</citation>}}


## stat.ML (1)



### (78/90) Non-Vacuous Generalization Bounds for Large Language Models (Sanae Lotfi et al., 2023)

{{<citation>}}

Sanae Lotfi, Marc Finzi, Yilun Kuang, Tim G. J. Rudner, Micah Goldblum, Andrew Gordon Wilson. (2023)  
**Non-Vacuous Generalization Bounds for Large Language Models**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-ML, stat.ML  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.17173v1)  

---


**ABSTRACT**  
Modern language models can contain billions of parameters, raising the question of whether they can generalize beyond the training data or simply regurgitate their training corpora. We provide the first non-vacuous generalization bounds for pretrained large language models (LLMs), indicating that language models are capable of discovering regularities that generalize to unseen data. In particular, we derive a compression bound that is valid for the unbounded log-likelihood loss using prediction smoothing, and we extend the bound to handle subsampling, accelerating bound computation on massive datasets. To achieve the extreme level of compression required for non-vacuous generalization bounds, we devise SubLoRA, a low-dimensional non-linear parameterization. Using this approach, we find that larger models have better generalization bounds and are more compressible than smaller models.

{{</citation>}}


## econ.TH (1)



### (79/90) The Gatekeeper Effect: The Implications of Pre-Screening, Self-selection, and Bias for Hiring Processes (Moran Koren, 2023)

{{<citation>}}

Moran Koren. (2023)  
**The Gatekeeper Effect: The Implications of Pre-Screening, Self-selection, and Bias for Hiring Processes**  

---
Primary Category: econ.TH  
Categories: 91B06, 91A10, 91A40, 91A80, K-4-4; J-1; I-2-11; J-4; G-3, cs-GT, econ-TH, econ.TH  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2312.17167v1)  

---


**ABSTRACT**  
We study the problem of screening in decision-making processes under uncertainty, focusing on the impact of adding an additional screening stage, commonly known as a 'gatekeeper.' While our primary analysis is rooted in the context of job market hiring, the principles and findings are broadly applicable to areas such as educational admissions, healthcare patient selection, and financial loan approvals. The gatekeeper's role is to assess applicants' suitability before significant investments are made. Our study reveals that while gatekeepers are designed to streamline the selection process by filtering out less likely candidates, they can sometimes inadvertently affect the candidates' own decision-making process. We explore the conditions under which the introduction of a gatekeeper can enhance or impede the efficiency of these processes. Additionally, we consider how adjusting gatekeeping strategies might impact the accuracy of selection decisions. Our research also extends to scenarios where gatekeeping is influenced by historical biases, particularly in competitive settings like hiring. We discover that candidates confronted with a statistically biased gatekeeping process are more likely to withdraw from applying, thereby perpetuating the previously mentioned historical biases. The study suggests that measures such as affirmative action can be effective in addressing these biases. While centered on hiring, the insights and methodologies from our study have significant implications for a wide range of fields where screening and gatekeeping are integral.

{{</citation>}}


## cs.SD (2)



### (80/90) BEAST: Online Joint Beat and Downbeat Tracking Based on Streaming Transformer (Chih-Cheng Chang et al., 2023)

{{<citation>}}

Chih-Cheng Chang, Li Su. (2023)  
**BEAST: Online Joint Beat and Downbeat Tracking Based on Streaming Transformer**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.17156v1)  

---


**ABSTRACT**  
Many deep learning models have achieved dominant performance on the offline beat tracking task. However, online beat tracking, in which only the past and present input features are available, still remains challenging. In this paper, we propose BEAt tracking Streaming Transformer (BEAST), an online joint beat and downbeat tracking system based on the streaming Transformer. To deal with online scenarios, BEAST applies contextual block processing in the Transformer encoder. Moreover, we adopt relative positional encoding in the attention layer of the streaming Transformer encoder to capture relative timing position which is critically important information in music. Carrying out beat and downbeat experiments on benchmark datasets for a low latency scenario with maximum latency under 50 ms, BEAST achieves an F1-measure of 80.04% in beat and 52.73% in downbeat, which is a substantial improvement of about 5 and 13 percentage points over the state-of-the-art online beat and downbeat tracking model.

{{</citation>}}


### (81/90) Revolutionizing Personalized Voice Synthesis: The Journey towards Emotional and Individual Authenticity with DIVSE (Dynamic Individual Voice Synthesis Engine) (Fan Shi, 2023)

{{<citation>}}

Fan Shi. (2023)  
**Revolutionizing Personalized Voice Synthesis: The Journey towards Emotional and Individual Authenticity with DIVSE (Dynamic Individual Voice Synthesis Engine)**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.17281v1)  

---


**ABSTRACT**  
This comprehensive paper delves into the forefront of personalized voice synthesis within artificial intelligence (AI), spotlighting the Dynamic Individual Voice Synthesis Engine (DIVSE). DIVSE represents a groundbreaking leap in text-to-voice (TTS) technology, uniquely focusing on adapting and personalizing voice outputs to match individual vocal characteristics. The research underlines the gap in current AI-generated voices, which, while technically advanced, fall short in replicating the unique individuality and expressiveness intrinsic to human speech. It outlines the challenges and advancements in personalized voice synthesis, emphasizing the importance of emotional expressiveness, accent and dialect variability, and capturing individual voice traits. The architecture of DIVSE is meticulously detailed, showcasing its three core components: Voice Characteristic Learning Module (VCLM), Emotional Tone and Accent Adaptation Module (ETAAM), and Dynamic Speech Synthesis Engine (DSSE). The innovative approach of DIVSE lies in its adaptive learning capability, which evolves over time to tailor voice outputs to specific user traits. The paper presents a rigorous experimental setup, utilizing accepted datasets and personalization metrics like Mean Opinion Score (MOS) and Emotional Alignment Score, to validate DIVSE's superiority over mainstream models. The results depict a clear advancement in achieving higher personalization and emotional resonance in AI-generated voices.

{{</citation>}}


## cs.CY (1)



### (82/90) The Intelligence College in Europe (ICE): An Effort to Create a European Intelligence Community (Uwe M. Borghoff et al., 2023)

{{<citation>}}

Uwe M. Borghoff, Lars Berger, François Fischer. (2023)  
**The Intelligence College in Europe (ICE): An Effort to Create a European Intelligence Community**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2312.17107v1)  

---


**ABSTRACT**  
In fulfilling the European security commitment, the actors of the so-called "Intelligence Community" play a central role. They provide political and military decision-makers with important analyses and information. The Intelligence College in Europe (ICE) is the first entity to offer professional intelligence training as well as postgraduate level academic education in intelligence and security studies at a pan-European level. In developing its postgraduate provision, ICE has benefited from the experience of the German Master of Intelligence and Security Studies (MISS), which is a joint effort of the University of the Bundeswehr Munich and the Department of Intelligence at the Federal University of Administrative Sciences in Berlin. As a main contribution of this paper, the module Counterterrorism (adapted from the MISS) is examined in more detail as a case study of how postgraduate modules can be modified to speak to a pan-European audience of intelligence professionals.

{{</citation>}}


## eess.IV (2)



### (83/90) Learning Multi-axis Representation in Frequency Domain for Medical Image Segmentation (Jiacheng Ruan et al., 2023)

{{<citation>}}

Jiacheng Ruan, Jingsheng Gao, Mingye Xie, Suncheng Xiang. (2023)  
**Learning Multi-axis Representation in Frequency Domain for Medical Image Segmentation**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.17030v1)  

---


**ABSTRACT**  
Recently, Visual Transformer (ViT) has been extensively used in medical image segmentation (MIS) due to applying self-attention mechanism in the spatial domain to modeling global knowledge. However, many studies have focused on improving models in the spatial domain while neglecting the importance of frequency domain information. Therefore, we propose Multi-axis External Weights UNet (MEW-UNet) based on the U-shape architecture by replacing self-attention in ViT with our Multi-axis External Weights block. Specifically, our block performs a Fourier transform on the three axes of the input features and assigns the external weight in the frequency domain, which is generated by our External Weights Generator. Then, an inverse Fourier transform is performed to change the features back to the spatial domain. We evaluate our model on four datasets, including Synapse, ACDC, ISIC17 and ISIC18 datasets, and our approach demonstrates competitive performance, owing to its effective utilization of frequency domain information.

{{</citation>}}


### (84/90) Combining Convolution Neural Networks with Long-Short Time Memory Layers to Predict Parkinson's Disease Progression (Maria Frasca et al., 2023)

{{<citation>}}

Maria Frasca, Davide La Torre, Ilaria Cutica. (2023)  
**Combining Convolution Neural Networks with Long-Short Time Memory Layers to Predict Parkinson's Disease Progression**  

---
Primary Category: eess.IV  
Categories: 62, I-2-6, cs-CV, eess-IV, eess.IV  
Keywords: AI, LSTM  
[Paper Link](http://arxiv.org/abs/2312.17290v1)  

---


**ABSTRACT**  
Parkinson's disease is a neurological condition that occurs in nearly 1% of the world's population. The disease is manifested by a drop in dopamine production, symptoms are cognitive and behavioural and include a wide range of personality changes, depressive disorders, memory problems, and emotional dysregulation, which can occur as the disease progresses. Early diagnosis and accurate staging of the disease are essential to apply the appropriate therapeutic approaches to slow cognitive and motor decline.   Currently, there is not a single blood test or biomarker available to diagnose Parkinson's disease. Magnetic resonance imaging has been used for the past three decades to diagnose and distinguish between PD and other neurological conditions. However, in recent years new possibilities have arisen: several AI algorithms have been developed to increase the precision and accuracy of differential diagnosis of PD at an early stage.   To our knowledge, no AI tools have been designed to identify the stage of progression. This paper aims to fill this gap. Using the "Parkinson's Progression Markers Initiative" dataset, which reports the patient's MRI and an indication of the disease stage, we developed a model to identify the level of progression. The images and the associated scores were used for training and assessing different deep-learning models. Our analysis distinguished four distinct disease progression levels based on a standard scale (Hoehn and Yah scale). The final architecture consists of the cascading of a 3DCNN network, adopted to reduce and extract the spatial characteristics of the RMI for efficient training of the successive LSTM layers, aiming at modelling the temporal dependencies among the data.   Our results show that the proposed 3DCNN + LSTM model achieves state-of-the-art results by classifying the elements with 91.90\% as macro averaged OVR AUC on four classes

{{</citation>}}


## cs.IT (1)



### (85/90) A GAN-based Semantic Communication for Text without CSI (Jin Mao et al., 2023)

{{<citation>}}

Jin Mao, Ke Xiong, Ming Liu, Zhijin Qin, Wei Chen, Pingyi Fan, Khaled Ben Letaief. (2023)  
**A GAN-based Semantic Communication for Text without CSI**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, math-IT  
Keywords: BLEU  
[Paper Link](http://arxiv.org/abs/2312.16909v1)  

---


**ABSTRACT**  
Recently, semantic communication (SC) has been regarded as one of the potential paradigms of 6G. Current SC frameworks require channel state information (CSI) to handle severe signal distortion induced by channel fading. Since the channel estimation overhead for obtaining CSI cannot be neglected, we therefore propose a generative adversarial network (GAN) based SC framework (Ti-GSC) that doesn't require CSI. In Ti-GSC, two main modules, i.e., an autoencoder-based encoder-decoder module (AEDM) and a GAN-based signal distortion suppression module (GSDSM) are included where AEDM first encodes the data at the source before transmission, and then GSDSM suppresses the distortion of the received signals in both syntactic and semantic dimensions at the destination. At last, AEDM decodes the distortion-suppressed signal at the destination. To measure signal distortion, syntactic distortion and semantic distortion terms are newly added to the total loss function. To achieve better training results, joint optimization-based training (JOT) and alternating optimization-based training (AOT) are designed for the proposed Ti-GSC. Experimental results show that JOT is more efficient for Ti-GSC. Moreover, without CSI, bilingual evaluation understudy (BLEU) score achieved by Ti-GSC is about 40% and 62% higher than that achieved by existing SC frameworks in Rician and Rayleigh fading, respectively. (*Due to the notification of arXiv "The Abstract field cannot be longer than 1,920 characters", the appeared Abstract is shortened. For the full Abstract, please download the Article.)

{{</citation>}}


## cs.IR (2)



### (86/90) DiffKG: Knowledge Graph Diffusion Model for Recommendation (Yangqin Jiang et al., 2023)

{{<citation>}}

Yangqin Jiang, Yuhao Yang, Lianghao Xia, Chao Huang. (2023)  
**DiffKG: Knowledge Graph Diffusion Model for Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2312.16890v1)  

---


**ABSTRACT**  
Knowledge Graphs (KGs) have emerged as invaluable resources for enriching recommendation systems by providing a wealth of factual information and capturing semantic relationships among items. Leveraging KGs can significantly enhance recommendation performance. However, not all relations within a KG are equally relevant or beneficial for the target recommendation task. In fact, certain item-entity connections may introduce noise or lack informative value, thus potentially misleading our understanding of user preferences. To bridge this research gap, we propose a novel knowledge graph diffusion model for recommendation, referred to as DiffKG. Our framework integrates a generative diffusion model with a data augmentation paradigm, enabling robust knowledge graph representation learning. This integration facilitates a better alignment between knowledge-aware item semantics and collaborative relation modeling. Moreover, we introduce a collaborative knowledge graph convolution mechanism that incorporates collaborative signals reflecting user-item interaction patterns, guiding the knowledge graph diffusion process. We conduct extensive experiments on three publicly available datasets, consistently demonstrating the superiority of our DiffKG compared to various competitive baselines. We provide the source code repository of our proposed DiffKG model at the following link: https://github.com/HKUDS/DiffKG.

{{</citation>}}


### (87/90) GUITAR: Gradient Pruning toward Fast Neural Ranking (Weijie Zhao et al., 2023)

{{<citation>}}

Weijie Zhao, Shulong Tan, Ping Li. (2023)  
**GUITAR: Gradient Pruning toward Fast Neural Ranking**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2312.16828v1)  

---


**ABSTRACT**  
With the continuous popularity of deep learning and representation learning, fast vector search becomes a vital task in various ranking/retrieval based applications, say recommendation, ads ranking and question answering. Neural network based ranking is widely adopted due to its powerful capacity in modeling complex relationships, such as between users and items, questions and answers. However, it is usually exploited in offline or re-ranking manners for it is time-consuming in computations. Online neural network ranking--so called fast neural ranking--is considered challenging because neural network measures are usually non-convex and asymmetric. Traditional Approximate Nearest Neighbor (ANN) search which usually focuses on metric ranking measures, is not applicable to these advanced measures.   In this paper, we introduce a novel graph searching framework to accelerate the searching in the fast neural ranking problem. The proposed graph searching algorithm is bi-level: we first construct a probable candidate set; then we only evaluate the neural network measure over the probable candidate set instead of evaluating the neural network over all neighbors. Specifically, we propose a gradient-based algorithm that approximates the rank of the neural network matching score to construct the probable candidate set; and we present an angle-based heuristic procedure to adaptively identify the proper size of the probable candidate set. Empirical results on public data confirm the effectiveness of our proposed algorithms.

{{</citation>}}


## eess.AS (2)



### (88/90) VOT: Revolutionizing Speaker Verification with Memory and Attention Mechanisms (Hongyu Wang et al., 2023)

{{<citation>}}

Hongyu Wang, Hui Li, Bo Li. (2023)  
**VOT: Revolutionizing Speaker Verification with Memory and Attention Mechanisms**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Attention, Speaker Verification, Transformer  
[Paper Link](http://arxiv.org/abs/2312.16826v1)  

---


**ABSTRACT**  
Speaker verification is essentially the process of identifying unknown speakers within an 'open set'. Our objective is to create optimal embeddings that condense information into concise speech-level representations, ensuring short distances within the same speaker and long distances between different speakers. Despite the prevalence of self-attention and convolution methods in speaker verification, they grapple with the challenge of high computational complexity.In order to surmount the limitations posed by the Transformer in extracting local features and the computational intricacies of multilayer convolution, we introduce the Memory-Attention framework. This framework incorporates a deep feed-forward temporal memory network (DFSMN) into the self-attention mechanism, capturing long-term context by stacking multiple layers and enhancing the modeling of local dependencies. Building upon this, we design a novel model called VOT, utilizing a parallel variable weight summation structure and introducing an attention-based statistical pooling layer.To address the hard sample mining problem, we enhance the AM-Softmax loss function and propose a new loss function named AM-Softmax-Focal. Experimental results on the VoxCeleb1 dataset not only showcase a significant improvement in system performance but also surpass the majority of mainstream models, validating the importance of local information in the speaker verification task. The code will be available on GitHub.

{{</citation>}}


### (89/90) Uncertainty Quantification in Machine Learning for Joint Speaker Diarization and Identification (Simon W. McKnight et al., 2023)

{{<citation>}}

Simon W. McKnight, Aidan O. T. Hogg, Vincent W. Neo, Patrick A. Naylor. (2023)  
**Uncertainty Quantification in Machine Learning for Joint Speaker Diarization and Identification**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2312.16763v1)  

---


**ABSTRACT**  
This paper studies modulation spectrum features ($\Phi$) and mel-frequency cepstral coefficients ($\Psi$) in joint speaker diarization and identification (JSID). JSID is important as speaker diarization on its own to distinguish speakers is insufficient for many applications, it is often necessary to identify speakers as well. Machine learning models are set up using convolutional neural networks (CNNs) on $\Phi$ and recurrent neural networks $\unicode{x2013}$ long short-term memory (LSTMs) on $\Psi$, then concatenating into fully connected layers.   Experiment 1 shows models on both $\Phi$ and $\Psi$ have better diarization error rates (DERs) than models on either alone; a CNN on $\Phi$ has DER 29.09\%, compared to 27.78\% for a LSTM on $\Psi$ and 19.44\% for a model on both. Experiment 1 also investigates aleatoric uncertainties and shows the model on both $\Phi$ and $\Psi$ has mean entropy 0.927~bits (out of 4~bits) for correct predictions compared to 1.896~bits for incorrect predictions which, along with entropy histogram shapes, shows the model helpfully indicates where it is uncertain.   Experiment 2 investigates epistemic uncertainties as well as aleatoric using Monte Carlo dropout (MCD). It compares models on both $\Phi$ and $\Psi$ with models trained on x-vectors ($X$), before applying Kalman filter smoothing on epistemic uncertainties for resegmentation and model ensembles. While the two models on $X$ (DERs 10.23\% and 9.74\%) outperform those on $\Phi$ and $\Psi$ (DER 17.85\%) after their individual Kalman filter smoothing, combining them using a Kalman filter smoothing method improves the DER to 9.29\%. Aleatoric uncertainties are higher for incorrect predictions.   Both Experiments show models on $\Phi$ do not distinguish overlapping speakers as well as anticipated. However, Experiment 2 shows model ensembles do better with overlapping speakers than individual models do.

{{</citation>}}


## cs.RO (1)



### (90/90) Difficulties in Dynamic Analysis of Drone Firmware and Its Solutions (Yejun Kim et al., 2023)

{{<citation>}}

Yejun Kim, Kwangsoo Cho, Seungjoo Kim. (2023)  
**Difficulties in Dynamic Analysis of Drone Firmware and Its Solutions**  

---
Primary Category: cs.RO  
Categories: cs-CR, cs-RO, cs.RO  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2312.16818v2)  

---


**ABSTRACT**  
With the advancement of Internet of Things (IoT) technology, its applications span various sectors such as public, industrial, private and military. In particular, the drone sector has gained significant attention for both commercial and military purposes. As a result, there has been a surge in research focused on vulnerability analysis of drones. However, most security research to mitigate threats to IoT devices has focused primarily on networks, firmware and mobile applications. Of these, the use of fuzzing to analyse the security of firmware requires emulation of the firmware. However, when it comes to drone firmware, the industry lacks emulation and automated fuzzing tools. This is largely due to challenges such as limited input interfaces, firmware encryption and signatures. While it may be tempting to assume that existing emulators and automated analysers for IoT devices can be applied to drones, practical applications have proven otherwise. In this paper, we discuss the challenges of dynamically analysing drone firmware and propose potential solutions. In addition, we demonstrate the effectiveness of our methodology by applying it to DJI drones, which have the largest market share.

{{</citation>}}
