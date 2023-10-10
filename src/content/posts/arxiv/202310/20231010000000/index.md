---
draft: false
title: "arXiv @ 2023.10.10"
date: 2023-10-10
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.10.10"
    identifier: arxiv_20231010
    parent: 202310_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.LG (23)](#cslg-23)
- [cs.CL (43)](#cscl-43)
- [cs.HC (2)](#cshc-2)
- [cs.CV (17)](#cscv-17)
- [cs.AI (14)](#csai-14)
- [cs.RO (3)](#csro-3)
- [math.OC (1)](#mathoc-1)
- [cs.SE (2)](#csse-2)
- [cs.SD (2)](#cssd-2)
- [eess.AS (1)](#eessas-1)
- [cs.CY (1)](#cscy-1)
- [eess.IV (1)](#eessiv-1)
- [cs.CE (1)](#csce-1)
- [cs.CR (1)](#cscr-1)

## cs.LG (23)



### (1/112) Optimizing Solution-Samplers for Combinatorial Problems: The Landscape of Policy-Gradient Methods (Constantine Caramanis et al., 2023)

{{<citation>}}

Constantine Caramanis, Dimitris Fotakis, Alkis Kalavasis, Vasilis Kontonis, Christos Tzamos. (2023)  
**Optimizing Solution-Samplers for Combinatorial Problems: The Landscape of Policy-Gradient Methods**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-DS, cs-LG, cs.LG, stat-ML  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.05309v1)  

---


**ABSTRACT**  
Deep Neural Networks and Reinforcement Learning methods have empirically shown great promise in tackling challenging combinatorial problems. In those methods a deep neural network is used as a solution generator which is then trained by gradient-based methods (e.g., policy gradient) to successively obtain better solution distributions. In this work we introduce a novel theoretical framework for analyzing the effectiveness of such methods. We ask whether there exist generative models that (i) are expressive enough to generate approximately optimal solutions; (ii) have a tractable, i.e, polynomial in the size of the input, number of parameters; (iii) their optimization landscape is benign in the sense that it does not contain sub-optimal stationary points. Our main contribution is a positive answer to this question. Our result holds for a broad class of combinatorial problems including Max- and Min-Cut, Max-$k$-CSP, Maximum-Weight-Bipartite-Matching, and the Traveling Salesman Problem. As a byproduct of our analysis we introduce a novel regularization process over vanilla gradient descent and provide theoretical and experimental evidence that it helps address vanishing-gradient issues and escape bad stationary points.

{{</citation>}}


### (2/112) Adversarial Attacks on Combinatorial Multi-Armed Bandits (Rishab Balasubramanian et al., 2023)

{{<citation>}}

Rishab Balasubramanian, Jiawei Li, Prasad Tadepalli, Huazheng Wang, Qingyun Wu, Haoyu Zhao. (2023)  
**Adversarial Attacks on Combinatorial Multi-Armed Bandits**  

---
Primary Category: cs.LG  
Categories: cs-DS, cs-LG, cs.LG, stat-ML  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2310.05308v1)  

---


**ABSTRACT**  
We study reward poisoning attacks on Combinatorial Multi-armed Bandits (CMAB). We first provide a sufficient and necessary condition for the attackability of CMAB, which depends on the intrinsic properties of the corresponding CMAB instance such as the reward distributions of super arms and outcome distributions of base arms. Additionally, we devise an attack algorithm for attackable CMAB instances. Contrary to prior understanding of multi-armed bandits, our work reveals a surprising fact that the attackability of a specific CMAB instance also depends on whether the bandit instance is known or unknown to the adversary. This finding indicates that adversarial attacks on CMAB are difficult in practice and a general attack strategy for any CMAB instance does not exist since the environment is mostly unknown to the adversary. We validate our theoretical findings via extensive experiments on real-world CMAB applications including probabilistic maximum covering problem, online minimum spanning tree, cascading bandits for online ranking, and online shortest path.

{{</citation>}}


### (3/112) Successive Data Injection in Conditional Quantum GAN Applied to Time Series Anomaly Detection (Benjamin Kalfon et al., 2023)

{{<citation>}}

Benjamin Kalfon, Soumaya Cherkaoui, Jean-Frédéric Laprade, Ola Ahmad, Shengrui Wang. (2023)  
**Successive Data Injection in Conditional Quantum GAN Applied to Time Series Anomaly Detection**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-NI, cs.LG  
Keywords: Anomaly Detection, Time Series  
[Paper Link](http://arxiv.org/abs/2310.05307v1)  

---


**ABSTRACT**  
Classical GAN architectures have shown interesting results for solving anomaly detection problems in general and for time series anomalies in particular, such as those arising in communication networks. In recent years, several quantum GAN architectures have been proposed in the literature. When detecting anomalies in time series using QGANs, huge challenges arise due to the limited number of qubits compared to the size of the data. To address these challenges, we propose a new high-dimensional encoding approach, named Successive Data Injection (SuDaI). In this approach, we explore a larger portion of the quantum state than that in the conventional angle encoding, the method used predominantly in the literature, through repeated data injections into the quantum state. SuDaI encoding allows us to adapt the QGAN for anomaly detection with network data of a much higher dimensionality than with the existing known QGANs implementations. In addition, SuDaI encoding applies to other types of high-dimensional time series and can be used in contexts beyond anomaly detection and QGANs, opening up therefore multiple fields of application.

{{</citation>}}


### (4/112) Tailoring Self-Attention for Graph via Rooted Subtrees (Siyuan Huang et al., 2023)

{{<citation>}}

Siyuan Huang, Yunchong Song, Jiayue Zhou, Zhouhan Lin. (2023)  
**Tailoring Self-Attention for Graph via Rooted Subtrees**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Attention, GNN, Self-Attention  
[Paper Link](http://arxiv.org/abs/2310.05296v1)  

---


**ABSTRACT**  
Attention mechanisms have made significant strides in graph learning, yet they still exhibit notable limitations: local attention faces challenges in capturing long-range information due to the inherent problems of the message-passing scheme, while global attention cannot reflect the hierarchical neighborhood structure and fails to capture fine-grained local information. In this paper, we propose a novel multi-hop graph attention mechanism, named Subtree Attention (STA), to address the aforementioned issues. STA seamlessly bridges the fully-attentional structure and the rooted subtree, with theoretical proof that STA approximates the global attention under extreme settings. By allowing direct computation of attention weights among multi-hop neighbors, STA mitigates the inherent problems in existing graph attention mechanisms. Further we devise an efficient form for STA by employing kernelized softmax, which yields a linear time complexity. Our resulting GNN architecture, the STAGNN, presents a simple yet performant STA-based graph neural network leveraging a hop-aware attention strategy. Comprehensive evaluations on ten node classification datasets demonstrate that STA-based models outperform existing graph transformers and mainstream GNNs. The code is available at https://github.com/LUMIA-Group/SubTree-Attention.

{{</citation>}}


### (5/112) Generalizable Error Modeling for Search Relevance Data Annotation Tasks (Heinrich Peters et al., 2023)

{{<citation>}}

Heinrich Peters, Alireza Hashemi, James Rae. (2023)  
**Generalizable Error Modeling for Search Relevance Data Annotation Tasks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-HC, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.05286v1)  

---


**ABSTRACT**  
Human data annotation is critical in shaping the quality of machine learning (ML) and artificial intelligence (AI) systems. One significant challenge in this context is posed by annotation errors, as their effects can degrade the performance of ML models. This paper presents a predictive error model trained to detect potential errors in search relevance annotation tasks for three industry-scale ML applications (music streaming, video streaming, and mobile apps) and assesses its potential to enhance the quality and efficiency of the data annotation process. Drawing on real-world data from an extensive search relevance annotation program, we illustrate that errors can be predicted with moderate model performance (AUC=0.65-0.75) and that model performance generalizes well across applications (i.e., a global, task-agnostic model performs on par with task-specific models). We present model explainability analyses to identify which types of features are the main drivers of predictive performance. Additionally, we demonstrate the usefulness of the model in the context of auditing, where prioritizing tasks with high predicted error probabilities considerably increases the amount of corrected annotation errors (e.g., 40% efficiency gains for the music streaming application). These results underscore that automated error detection models can yield considerable improvements in the efficiency and quality of data annotation processes. Thus, our findings reveal critical insights into effective error management in the data annotation process, thereby contributing to the broader field of human-in-the-loop ML.

{{</citation>}}


### (6/112) Simplifying GNN Performance with Low Rank Kernel Models (Luciano Vinas et al., 2023)

{{<citation>}}

Luciano Vinas, Arash A. Amini. (2023)  
**Simplifying GNN Performance with Low Rank Kernel Models**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2310.05250v1)  

---


**ABSTRACT**  
We revisit recent spectral GNN approaches to semi-supervised node classification (SSNC). We posit that many of the current GNN architectures may be over-engineered. Instead, simpler, traditional methods from nonparametric estimation, applied in the spectral domain, could replace many deep-learning inspired GNN designs. These conventional techniques appear to be well suited for a variety of graph types reaching state-of-the-art performance on many of the common SSNC benchmarks. Additionally, we show that recent performance improvements in GNN approaches may be partially attributed to shifts in evaluation conventions. Lastly, an ablative study is conducted on the various hyperparameters associated with GNN spectral filtering techniques. Code available at: https://github.com/lucianoAvinas/lowrank-gnn-kernels

{{</citation>}}


### (7/112) In-Context Convergence of Transformers (Yu Huang et al., 2023)

{{<citation>}}

Yu Huang, Yuan Cheng, Yingbin Liang. (2023)  
**In-Context Convergence of Transformers**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, math-OC, stat-ML  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.05249v1)  

---


**ABSTRACT**  
Transformers have recently revolutionized many domains in modern machine learning and one salient discovery is their remarkable in-context learning capability, where models can solve an unseen task by utilizing task-specific prompts without further parameters fine-tuning. This also inspired recent theoretical studies aiming to understand the in-context learning mechanism of transformers, which however focused only on linear transformers. In this work, we take the first step toward studying the learning dynamics of a one-layer transformer with softmax attention trained via gradient descent in order to in-context learn linear function classes. We consider a structured data model, where each token is randomly sampled from a set of feature vectors in either balanced or imbalanced fashion. For data with balanced features, we establish the finite-time convergence guarantee with near-zero prediction error by navigating our analysis over two phases of the training dynamics of the attention map. More notably, for data with imbalanced features, we show that the learning dynamics take a stage-wise convergence process, where the transformer first converges to a near-zero prediction error for the query tokens of dominant features, and then converges later to a near-zero prediction error for the query tokens of under-represented features, respectively via one and four training phases. Our proof features new techniques for analyzing the competing strengths of two types of attention weights, the change of which determines different training phases.

{{</citation>}}


### (8/112) Accelerating Machine Learning Primitives on Commodity Hardware (Roman Snytsar, 2023)

{{<citation>}}

Roman Snytsar. (2023)  
**Accelerating Machine Learning Primitives on Commodity Hardware**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.05218v1)  

---


**ABSTRACT**  
Sliding Window Sum algorithms have been successfully used for training and inference of Deep Neural Networks. We have shown before how both pooling and convolution 1-D primitives could be expressed as sliding sums and evaluated by the compute kernels with a shared structure. In this paper, we present an extensive study of the Sliding Window convolution technique as a more efficient alternative to the commonly used General Matrix Multiplication (GEMM) based convolution in Deep Neural Networks (DNNs). The Sliding Window technique addresses the memory bloating problem and demonstrates a significant speedup in 2-D convolution. We explore the performance of this technique on a range of implementations, including custom kernels for specific filter sizes. Our results suggest that the Sliding Window computation kernels can outperform GEMM-based convolution on a CPU and even on dedicated hardware accelerators. This could promote a wider adoption of AI on low-power and low-memory devices without the need for specialized hardware. We also discuss the compatibility of model compression methods and optimized network architectures with the Sliding Window technique, encouraging further research in these areas.

{{</citation>}}


### (9/112) GEAR: A GPU-Centric Experience Replay System for Large Reinforcement Learning Models (Hanjing Wang et al., 2023)

{{<citation>}}

Hanjing Wang, Man-Kit Sit, Congjie He, Ying Wen, Weinan Zhang, Jun Wang, Yaodong Yang, Luo Mai. (2023)  
**GEAR: A GPU-Centric Experience Replay System for Large Reinforcement Learning Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-DC, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.05205v1)  

---


**ABSTRACT**  
This paper introduces a distributed, GPU-centric experience replay system, GEAR, designed to perform scalable reinforcement learning (RL) with large sequence models (such as transformers). With such models, existing systems such as Reverb face considerable bottlenecks in memory, computation, and communication. GEAR, however, optimizes memory efficiency by enabling the memory resources on GPU servers (including host memory and device memory) to manage trajectory data. Furthermore, it facilitates decentralized GPU devices to expedite various trajectory selection strategies, circumventing computational bottlenecks. GEAR is equipped with GPU kernels capable of collecting trajectories using zero-copy access to host memory, along with remote-directed-memory access over InfiniBand, improving communication efficiency. Cluster experiments have shown that GEAR can achieve performance levels up to 6x greater than Reverb when training state-of-the-art large RL models. GEAR is open-sourced at https://github.com/bigrl-team/gear.

{{</citation>}}


### (10/112) Towards Optimizing with Large Language Models (Pei-Fu Guo et al., 2023)

{{<citation>}}

Pei-Fu Guo, Ying-Hsuan Chen, Yun-Da Tsai, Shou-De Lin. (2023)  
**Towards Optimizing with Large Language Models**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.05204v1)  

---


**ABSTRACT**  
In this work, we conduct an assessment of the optimization capabilities of LLMs across various tasks and data sizes. Each of these tasks corresponds to unique optimization domains, and LLMs are required to execute these tasks with interactive prompting. That is, in each optimization step, the LLM generates new solutions from the past generated solutions with their values, and then the new solutions are evaluated and considered in the next optimization step. Additionally, we introduce three distinct metrics for a comprehensive assessment of task performance from various perspectives. These metrics offer the advantage of being applicable for evaluating LLM performance across a broad spectrum of optimization tasks and are less sensitive to variations in test samples. By applying these metrics, we observe that LLMs exhibit strong optimization capabilities when dealing with small-sized samples. However, their performance is significantly influenced by factors like data size and values, underscoring the importance of further research in the domain of optimization tasks for LLMs.

{{</citation>}}


### (11/112) Lifelong Learning for Fog Load Balancing: A Transfer Learning Approach (Maad Ebrahim et al., 2023)

{{<citation>}}

Maad Ebrahim, Abdelhakim Senhaji Hafid, Mohamed Riduan Abid. (2023)  
**Lifelong Learning for Fog Load Balancing: A Transfer Learning Approach**  

---
Primary Category: cs.LG  
Categories: cs-DC, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.05187v1)  

---


**ABSTRACT**  
Fog computing emerged as a promising paradigm to address the challenges of processing and managing data generated by the Internet of Things (IoT). Load balancing (LB) plays a crucial role in Fog computing environments to optimize the overall system performance. It requires efficient resource allocation to improve resource utilization, minimize latency, and enhance the quality of service for end-users. In this work, we improve the performance of privacy-aware Reinforcement Learning (RL) agents that optimize the execution delay of IoT applications by minimizing the waiting delay. To maintain privacy, these agents optimize the waiting delay by minimizing the change in the number of queued requests in the whole system, i.e., without explicitly observing the actual number of requests that are queued in each Fog node nor observing the compute resource capabilities of those nodes. Besides improving the performance of these agents, we propose in this paper a lifelong learning framework for these agents, where lightweight inference models are used during deployment to minimize action delay and only retrained in case of significant environmental changes. To improve the performance, minimize the training cost, and adapt the agents to those changes, we explore the application of Transfer Learning (TL). TL transfers the knowledge acquired from a source domain and applies it to a target domain, enabling the reuse of learned policies and experiences. TL can be also used to pre-train the agent in simulation before fine-tuning it in the real environment; this significantly reduces failure probability compared to learning from scratch in the real environment. To our knowledge, there are no existing efforts in the literature that use TL to address lifelong learning for RL-based Fog LB; this is one of the main obstacles in deploying RL LB solutions in Fog systems.

{{</citation>}}


### (12/112) Distributional Reinforcement Learning with Online Risk-awareness Adaption (Yupeng Wu et al., 2023)

{{<citation>}}

Yupeng Wu, Wenjie Huang. (2023)  
**Distributional Reinforcement Learning with Online Risk-awareness Adaption**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.05179v1)  

---


**ABSTRACT**  
The use of reinforcement learning (RL) in practical applications requires considering sub-optimal outcomes, which depend on the agent's familiarity with the uncertain environment. Dynamically adjusting the level of epistemic risk over the course of learning can tactically achieve reliable optimal policy in safety-critical environments and tackle the sub-optimality of a static risk level. In this work, we introduce a novel framework, Distributional RL with Online Risk Adaption (DRL-ORA), which can quantify the aleatory and epistemic uncertainties compositely and dynamically select the epistemic risk levels via solving a total variation minimization problem online. The risk level selection can be efficiently achieved through grid search using a Follow-The-Leader type algorithm, and its offline oracle is related to "satisficing measure" (in the decision analysis community) under a special modification of the loss function. We show multiple classes of tasks where DRL-ORA outperforms existing methods that rely on either a fixed risk level or manually predetermined risk level adaption. Given the simplicity of our modifications, we believe the framework can be easily incorporated into most RL algorithm variants.

{{</citation>}}


### (13/112) Outlier Weighed Layerwise Sparsity (OWL): A Missing Secret Sauce for Pruning LLMs to High Sparsity (Lu Yin et al., 2023)

{{<citation>}}

Lu Yin, You Wu, Zhenyu Zhang, Cheng-Yu Hsieh, Yaqing Wang, Yiling Jia, Mykola Pechenizkiy, Yi Liang, Zhangyang Wang, Shiwei Liu. (2023)  
**Outlier Weighed Layerwise Sparsity (OWL): A Missing Secret Sauce for Pruning LLMs to High Sparsity**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GPT, LLaMA, Language Model, Pruning  
[Paper Link](http://arxiv.org/abs/2310.05175v1)  

---


**ABSTRACT**  
Large Language Models (LLMs), renowned for their remarkable performance, present a challenge due to their colossal model size when it comes to practical deployment. In response to this challenge, efforts have been directed toward the application of traditional network pruning techniques to LLMs, uncovering a massive number of parameters can be pruned in one-shot without hurting performance. Building upon insights gained from pre-LLM models, prevailing LLM pruning strategies have consistently adhered to the practice of uniformly pruning all layers at equivalent sparsity. However, this observation stands in contrast to the prevailing trends observed in the field of vision models, where non-uniform layerwise sparsity typically yields substantially improved results. To elucidate the underlying reasons for this disparity, we conduct a comprehensive analysis of the distribution of token features within LLMs. In doing so, we discover a strong correlation with the emergence of outliers, defined as features exhibiting significantly greater magnitudes compared to their counterparts in feature dimensions. Inspired by this finding, we introduce a novel LLM pruning methodology that incorporates a tailored set of non-uniform layerwise sparsity ratios specifically designed for LLM pruning, termed as Outlier Weighed Layerwise sparsity (OWL). The sparsity ratio of OWL is directly proportional to the outlier ratio observed within each layer, facilitating a more effective alignment between layerwise weight sparsity and outlier ratios. Our empirical evaluation, conducted across the LLaMA-V1 family and OPT, spanning various benchmarks, demonstrates the distinct advantages offered by OWL over previous methods. For instance, our approach exhibits a remarkable performance gain, surpassing the state-of-the-art Wanda and SparseGPT by 61.22 and 6.80 perplexity at a high sparsity level of 70%, respectively.

{{</citation>}}


### (14/112) GSLB: The Graph Structure Learning Benchmark (Zhixun Li et al., 2023)

{{<citation>}}

Zhixun Li, Liang Wang, Xin Sun, Yifan Luo, Yanqiao Zhu, Dingshuo Chen, Yingtao Luo, Xiangxin Zhou, Qiang Liu, Shu Wu, Liang Wang, Jeffrey Xu Yu. (2023)  
**GSLB: The Graph Structure Learning Benchmark**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.05174v1)  

---


**ABSTRACT**  
Graph Structure Learning (GSL) has recently garnered considerable attention due to its ability to optimize both the parameters of Graph Neural Networks (GNNs) and the computation graph structure simultaneously. Despite the proliferation of GSL methods developed in recent years, there is no standard experimental setting or fair comparison for performance evaluation, which creates a great obstacle to understanding the progress in this field. To fill this gap, we systematically analyze the performance of GSL in different scenarios and develop a comprehensive Graph Structure Learning Benchmark (GSLB) curated from 20 diverse graph datasets and 16 distinct GSL algorithms. Specifically, GSLB systematically investigates the characteristics of GSL in terms of three dimensions: effectiveness, robustness, and complexity. We comprehensively evaluate state-of-the-art GSL algorithms in node- and graph-level tasks, and analyze their performance in robust learning and model complexity. Further, to facilitate reproducible research, we have developed an easy-to-use library for training, evaluating, and visualizing different GSL methods. Empirical results of our extensive experiments demonstrate the ability of GSL and reveal its potential benefits on various downstream tasks, offering insights and opportunities for future research. The code of GSLB is available at: https://github.com/GSL-Benchmark/GSLB.

{{</citation>}}


### (15/112) How Graph Neural Networks Learn: Lessons from Training Dynamics in Function Space (Chenxiao Yang et al., 2023)

{{<citation>}}

Chenxiao Yang, Qitian Wu, David Wipf, Ruoyu Sun, Junchi Yan. (2023)  
**How Graph Neural Networks Learn: Lessons from Training Dynamics in Function Space**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.05105v1)  

---


**ABSTRACT**  
A long-standing goal in deep learning has been to characterize the learning behavior of black-box models in a more interpretable manner. For graph neural networks (GNNs), considerable advances have been made in formalizing what functions they can represent, however it remains less clear whether and how GNNs learn desired functions during the optimization process. To fill this critical gap, we study the learning dynamics of GNNs in function space via the analytic framework of overparameterization. In particular, we find that the seemingly complicated training process of GNNs can be re-cast into a more familiar label propagation framework, due to the graph inductive bias implicit in this process. From this vantage point, we provide explanations for why the learned GNN functions successfully generalize and for their pathological behavior on heterophilic graphs, which are consistent with observations. Practically, sparsifying and implementing the learning dynamics lead to a minimalist semi-supervised learning algorithm with the efficiency of classic algorithms and the effectiveness of modern GNNs.

{{</citation>}}


### (16/112) FLatS: Principled Out-of-Distribution Detection with Feature-Based Likelihood Ratio Score (Haowei Lin et al., 2023)

{{<citation>}}

Haowei Lin, Yuntian Gu. (2023)  
**FLatS: Principled Out-of-Distribution Detection with Feature-Based Likelihood Ratio Score**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2310.05083v1)  

---


**ABSTRACT**  
Detecting out-of-distribution (OOD) instances is crucial for NLP models in practical applications. Although numerous OOD detection methods exist, most of them are empirical. Backed by theoretical analysis, this paper advocates for the measurement of the "OOD-ness" of a test case $\boldsymbol{x}$ through the likelihood ratio between out-distribution $\mathcal P_{\textit{out}}$ and in-distribution $\mathcal P_{\textit{in}}$. We argue that the state-of-the-art (SOTA) feature-based OOD detection methods, such as Maha and KNN, are suboptimal since they only estimate in-distribution density $p_{\textit{in}}(\boldsymbol{x})$. To address this issue, we propose FLatS, a principled solution for OOD detection based on likelihood ratio. Moreover, we demonstrate that FLatS can serve as a general framework capable of enhancing other OOD detection methods by incorporating out-distribution density $p_{\textit{out}}(\boldsymbol{x})$ estimation. Experiments show that FLatS establishes a new SOTA on popular benchmarks. Our code is publicly available at https://github.com/linhaowei1/FLatS.

{{</citation>}}


### (17/112) Pushing the Limits of Pre-training for Time Series Forecasting in the CloudOps Domain (Gerald Woo et al., 2023)

{{<citation>}}

Gerald Woo, Chenghao Liu, Akshat Kumar, Doyen Sahoo. (2023)  
**Pushing the Limits of Pre-training for Time Series Forecasting in the CloudOps Domain**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2310.05063v1)  

---


**ABSTRACT**  
Time series has been left behind in the era of pre-training and transfer learning. While research in the fields of natural language processing and computer vision are enjoying progressively larger datasets to train massive models, the most popular time series datasets consist of only tens of thousands of time steps, limiting our ability to study the effectiveness of pre-training and scaling. Recent studies have also cast doubt on the need for expressive models and scale. To alleviate these issues, we introduce three large-scale time series forecasting datasets from the cloud operations (CloudOps) domain, the largest having billions of observations, enabling further study into pre-training and scaling of time series models. We build the empirical groundwork for studying pre-training and scaling of time series models and pave the way for future research by identifying a promising candidate architecture. We show that it is a strong zero-shot baseline and benefits from further scaling, both in model and dataset size. Accompanying these datasets and results is a suite of comprehensive benchmark results comparing classical and deep learning baselines to our pre-trained method - achieving a 27% reduction in error on the largest dataset. Code and datasets will be released.

{{</citation>}}


### (18/112) Deep Reinforcement Learning Based Cross-Layer Design in Terahertz Mesh Backhaul Networks (Zhifeng Hu et al., 2023)

{{<citation>}}

Zhifeng Hu, Chong Han, Xudong Wang. (2023)  
**Deep Reinforcement Learning Based Cross-Layer Design in Terahertz Mesh Backhaul Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.05034v1)  

---


**ABSTRACT**  
Supporting ultra-high data rates and flexible reconfigurability, Terahertz (THz) mesh networks are attractive for next-generation wireless backhaul systems that empower the integrated access and backhaul (IAB). In THz mesh backhaul networks, the efficient cross-layer routing and long-term resource allocation is yet an open problem due to dynamic traffic demands as well as possible link failures caused by the high directivity and high non-line-of-sight (NLoS) path loss of THz spectrum. In addition, unpredictable data traffic and the mixed integer programming property with the NP-hard nature further challenge the effective routing and long-term resource allocation design. In this paper, a deep reinforcement learning (DRL) based cross-layer design in THz mesh backhaul networks (DEFLECT) is proposed, by considering dynamic traffic demands and possible sudden link failures. In DEFLECT, a heuristic routing metric is first devised to facilitate resource efficiency (RE) enhancement regarding energy and sub-array usages. Furthermore, a DRL based resource allocation algorithm is developed to realize long-term RE maximization and fast recovery from broken links. Specifically in the DRL method, the exploited multi-task structure cooperatively benefits joint power and sub-array allocation. Additionally, the leveraged hierarchical architecture realizes tailored resource allocation for each base station and learned knowledge transfer for fast recovery. Simulation results show that DEFLECT routing consumes less resource, compared to the minimal hop-count metric. Moreover, unlike conventional DRL methods causing packet loss and second-level latency, DEFLECT DRL realizes the long-term RE maximization with no packet loss and millisecond-level latency, and recovers resource-efficient backhaul from broken links within 1s.

{{</citation>}}


### (19/112) Data-centric Graph Learning: A Survey (Cheng Yang et al., 2023)

{{<citation>}}

Cheng Yang, Deyu Bo, Jixi Liu, Yufei Peng, Boyu Chen, Haoran Dai, Ao Sun, Yue Yu, Yixin Xiao, Qi Zhang, Chunchen Wang, Yuxin Guo, Chuan Shi. (2023)  
**Data-centric Graph Learning: A Survey**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SI, cs.LG  
Keywords: AI, ImageNet  
[Paper Link](http://arxiv.org/abs/2310.04987v1)  

---


**ABSTRACT**  
The history of artificial intelligence (AI) has witnessed the significant impact of high-quality data on various deep learning models, such as ImageNet for AlexNet and ResNet. Recently, instead of designing more complex neural architectures as model-centric approaches, the attention of AI community has shifted to data-centric ones, which focuses on better processing data to strengthen the ability of neural models. Graph learning, which operates on ubiquitous topological data, also plays an important role in the era of deep learning. In this survey, we comprehensively review graph learning approaches from the data-centric perspective, and aim to answer two crucial questions: (1) when to modify graph data and (2) how to modify graph data to unlock the potential of various graph models. Accordingly, we propose a novel taxonomy based on the stages in the graph learning pipeline, and highlight the processing methods for different data structures in the graph data, i.e., topology, feature and label. Furthermore, we analyze some potential problems embedded in graph data and discuss how to solve them in a data-centric manner. Finally, we provide some promising future directions for data-centric graph learning.

{{</citation>}}


### (20/112) Understanding the Robustness of Multi-modal Contrastive Learning to Distribution Shift (Yihao Xue et al., 2023)

{{<citation>}}

Yihao Xue, Siddharth Joshi, Dang Nguyen, Baharan Mirzasoleiman. (2023)  
**Understanding the Robustness of Multi-modal Contrastive Learning to Distribution Shift**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Contrastive Learning, ImageNet  
[Paper Link](http://arxiv.org/abs/2310.04971v1)  

---


**ABSTRACT**  
Recently, multimodal contrastive learning (MMCL) approaches, such as CLIP, have achieved a remarkable success in learning representations that are robust against distribution shift and generalize to new domains. Despite the empirical success, the mechanism behind learning such generalizable representations is not understood. In this work, we rigorously analyze this problem and uncover two mechanisms behind MMCL's robustness: \emph{intra-class contrasting}, which allows the model to learn features with a high variance, and \emph{inter-class feature sharing}, where annotated details in one class help learning other classes better. Both mechanisms prevent spurious features that are over-represented in the training data to overshadow the generalizable core features. This yields superior zero-shot classification accuracy under distribution shift. Furthermore, we theoretically demonstrate the benefits of using rich captions on robustness and explore the effect of annotating different types of details in the captions. We validate our theoretical findings through experiments, including a well-designed synthetic experiment and an experiment involving training CLIP on MS COCO and evaluating the model on variations of shifted ImageNet.

{{</citation>}}


### (21/112) Improved Active Learning via Dependent Leverage Score Sampling (Atsushi Shimizu et al., 2023)

{{<citation>}}

Atsushi Shimizu, Xiaoou Cheng, Christopher Musco, Jonathan Weare. (2023)  
**Improved Active Learning via Dependent Leverage Score Sampling**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2310.04966v1)  

---


**ABSTRACT**  
We show how to obtain improved active learning methods in the agnostic (adversarial noise) setting by combining marginal leverage score sampling with non-independent sampling strategies that promote spatial coverage. In particular, we propose an easily implemented method based on the pivotal sampling algorithm, which we test on problems motivated by learning-based methods for parametric PDEs and uncertainty quantification. In comparison to independent sampling, our method reduces the number of samples needed to reach a given target accuracy by up to $50\%$. We support our findings with two theoretical results. First, we show that any non-independent leverage score sampling method that obeys a weak one-sided $\ell_{\infty}$ independence condition (which includes pivotal sampling) can actively learn $d$ dimensional linear functions with $O(d\log d)$ samples, matching independent sampling. This result extends recent work on matrix Chernoff bounds under $\ell_{\infty}$ independence, and may be of interest for analyzing other sampling strategies beyond pivotal sampling. Second, we show that, for the important case of polynomial regression, our pivotal method obtains an improved bound of $O(d)$ samples.

{{</citation>}}


### (22/112) Information-Theoretic Bounds on The Removal of Attribute-Specific Bias From Neural Networks (Jiazhi Li et al., 2023)

{{<citation>}}

Jiazhi Li, Mahyar Khayatkhoei, Jiageng Zhu, Hanchen Xie, Mohamed E. Hussein, Wael AbdAlmageed. (2023)  
**Information-Theoretic Bounds on The Removal of Attribute-Specific Bias From Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI, Bias  
[Paper Link](http://arxiv.org/abs/2310.04955v1)  

---


**ABSTRACT**  
Ensuring a neural network is not relying on protected attributes (e.g., race, sex, age) for predictions is crucial in advancing fair and trustworthy AI. While several promising methods for removing attribute bias in neural networks have been proposed, their limitations remain under-explored. In this work, we mathematically and empirically reveal an important limitation of attribute bias removal methods in presence of strong bias. Specifically, we derive a general non-vacuous information-theoretical upper bound on the performance of any attribute bias removal method in terms of the bias strength. We provide extensive experiments on synthetic, image, and census datasets to verify the theoretical bound and its consequences in practice. Our findings show that existing attribute bias removal methods are effective only when the inherent bias in the dataset is relatively weak, thus cautioning against the use of these methods in smaller datasets where strong attribute bias can occur, and advocating the need for methods that can overcome this limitation.

{{</citation>}}


### (23/112) TEMPO: Prompt-based Generative Pre-trained Transformer for Time Series Forecasting (Defu Cao et al., 2023)

{{<citation>}}

Defu Cao, Furong Jia, Sercan O Arik, Tomas Pfister, Yixiang Zheng, Wen Ye, Yan Liu. (2023)  
**TEMPO: Prompt-based Generative Pre-trained Transformer for Time Series Forecasting**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: GPT, Time Series, Transformer  
[Paper Link](http://arxiv.org/abs/2310.04948v1)  

---


**ABSTRACT**  
The past decade has witnessed significant advances in time series modeling with deep learning. While achieving state-of-the-art results, the best-performing architectures vary highly across applications and domains. On the other hand, for natural language processing, Generative Pre-trained Transformer (GPT) has demonstrated impressive performance via training one general-purpose model across various textual datasets. It is intriguing to explore whether GPT-type architectures can be effective for time series, capturing the intrinsic dynamic attributes and leading to significant accuracy improvements. In this paper, we propose a novel framework, TEMPO, that can effectively learn time series representations. We focus on utilizing two essential inductive biases of the time series task for pre-trained models: (i) decomposition of the complex interaction between trend, seasonal and residual components; and (ii) introducing the selection-based prompts to facilitate distribution adaptation in non-stationary time series. TEMPO expands the capability for dynamically modeling real-world temporal phenomena from data within diverse domains. Our experiments demonstrate the superior performance of TEMPO, with 20\%-60\% improvement over state-of-the-art methods on a number of time series benchmark datasets. This performance gain is observed not only in standard supervised learning settings but also in scenarios involving previously unseen datasets. This compelling finding highlights \modelname's potential to constitute a foundational model building framework.

{{</citation>}}


## cs.CL (43)



### (24/112) Hi Guys or Hi Folks? Benchmarking Gender-Neutral Machine Translation with the GeNTE Corpus (Andrea Piergentili et al., 2023)

{{<citation>}}

Andrea Piergentili, Beatrice Savoldi, Dennis Fucci, Matteo Negri, Luisa Bentivogli. (2023)  
**Hi Guys or Hi Folks? Benchmarking Gender-Neutral Machine Translation with the GeNTE Corpus**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Machine Translation  
[Paper Link](http://arxiv.org/abs/2310.05294v1)  

---


**ABSTRACT**  
Gender inequality is embedded in our communication practices and perpetuated in translation technologies. This becomes particularly apparent when translating into grammatical gender languages, where machine translation (MT) often defaults to masculine and stereotypical representations by making undue binary gender assumptions. Our work addresses the rising demand for inclusive language by focusing head-on on gender-neutral translation from English to Italian. We start from the essentials: proposing a dedicated benchmark and exploring automated evaluation methods. First, we introduce GeNTE, a natural, bilingual test set for gender-neutral translation, whose creation was informed by a survey on the perception and use of neutral language. Based on GeNTE, we then overview existing reference-based evaluation approaches, highlight their limits, and propose a reference-free method more suitable to assess gender-neutral translation.

{{</citation>}}


### (25/112) Are Personalized Stochastic Parrots More Dangerous? Evaluating Persona Biases in Dialogue Systems (Yixin Wan et al., 2023)

{{<citation>}}

Yixin Wan, Jieyu Zhao, Nanyun Peng, Kai-Wei Chang, Aman Chadha. (2023)  
**Are Personalized Stochastic Parrots More Dangerous? Evaluating Persona Biases in Dialogue Systems**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Bias, ChatGPT, Dialog, Dialogue, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.05280v1)  

---


**ABSTRACT**  
Recent advancements in Large Language Models empower them to follow freeform instructions, including imitating generic or specific demographic personas in conversations. Generic personas refer to an individual from a demographic group (e.g. an Asian person), whereas specific personas can be actual names of historical figures. While the adoption of personas allows dialogue systems to be more engaging and approachable to users, it also carries the potential risk of exacerbating social biases in model responses, further causing societal harms through interactions with users. In this paper, we systematically study "persona biases", which we define to be the sensitivity of harmful dialogue model behaviors to different persona adoptions. We categorize persona biases into biases in harmful expression and harmful agreement, as well as establish a comprehensive evaluation framework to measure persona biases in five aspects: Offensiveness, Toxic Continuation, Regard, Stereotype Agreement, and Toxic Agreement. Additionally, we propose to comprehensively investigate persona biases through experimenting with UniversalPersona, a systematized persona dataset with a comprehensive list of both generic and specific model personas. Through benchmarking on four different models, including Blender, ChatGPT, Alpaca, and Vicuna, our study uncovers significant persona biases in these dialogue systems.Findings of our study underscores the immediate need to revisit the use of persona traits in dialogue agents, to ensure their safe application.

{{</citation>}}


### (26/112) Enhancing Pre-Trained Language Models with Sentence Position Embeddings for Rhetorical Roles Recognition in Legal Opinions (Anas Belfathi et al., 2023)

{{<citation>}}

Anas Belfathi, Nicolas Hernandez, Laura Monceaux. (2023)  
**Enhancing Pre-Trained Language Models with Sentence Position Embeddings for Rhetorical Roles Recognition in Legal Opinions**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, Embedding, Language Model, Legal, Position Embedding  
[Paper Link](http://arxiv.org/abs/2310.05276v1)  

---


**ABSTRACT**  
The legal domain is a vast and complex field that involves a considerable amount of text analysis, including laws, legal arguments, and legal opinions. Legal practitioners must analyze these texts to understand legal cases, research legal precedents, and prepare legal documents. The size of legal opinions continues to grow, making it increasingly challenging to develop a model that can accurately predict the rhetorical roles of legal opinions given their complexity and diversity. In this research paper, we propose a novel model architecture for automatically predicting rhetorical roles using pre-trained language models (PLMs) enhanced with knowledge of sentence position information within a document. Based on an annotated corpus from the LegalEval@SemEval2023 competition, we demonstrate that our approach requires fewer parameters, resulting in lower computational costs when compared to complex architectures employing a hierarchical model in a global-context, yet it achieves great performance. Moreover, we show that adding more attention to a hierarchical model based only on BERT in the local-context, along with incorporating sentence position information, enhances the results.

{{</citation>}}


### (27/112) Explainable Claim Verification via Knowledge-Grounded Reasoning with Large Language Models (Haoran Wang et al., 2023)

{{<citation>}}

Haoran Wang, Kai Shu. (2023)  
**Explainable Claim Verification via Knowledge-Grounded Reasoning with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.05253v1)  

---


**ABSTRACT**  
Claim verification plays a crucial role in combating misinformation. While existing works on claim verification have shown promising results, a crucial piece of the puzzle that remains unsolved is to understand how to verify claims without relying on human-annotated data, which is expensive to create at a large scale. Additionally, it is important for models to provide comprehensive explanations that can justify their decisions and assist human fact-checkers. This paper presents First-Order-Logic-Guided Knowledge-Grounded (FOLK) Reasoning that can verify complex claims and generate explanations without the need for annotated evidence using Large Language Models (LLMs). FOLK leverages the in-context learning ability of LLMs to translate the claim into a First-Order-Logic (FOL) clause consisting of predicates, each corresponding to a sub-claim that needs to be verified. Then, FOLK performs FOL-Guided reasoning over a set of knowledge-grounded question-and-answer pairs to make veracity predictions and generate explanations to justify its decision-making process. This process makes our model highly explanatory, providing clear explanations of its reasoning process in human-readable form. Our experiment results indicate that FOLK outperforms strong baselines on three datasets encompassing various claim verification challenges. Our code and data are available.

{{</citation>}}


### (28/112) ChatRadio-Valuer: A Chat Large Language Model for Generalizable Radiology Report Generation Based on Multi-institution and Multi-system Data (Tianyang Zhong et al., 2023)

{{<citation>}}

Tianyang Zhong, Wei Zhao, Yutong Zhang, Yi Pan, Peixin Dong, Zuowei Jiang, Xiaoyan Kui, Youlan Shang, Li Yang, Yaonai Wei, Longtao Yang, Hao Chen, Huan Zhao, Yuxiao Liu, Ning Zhu, Yiwei Li, Yisong Wang, Jiaqi Yao, Jiaqi Wang, Ying Zeng, Lei He, Chao Zheng, Zhixue Zhang, Ming Li, Zhengliang Liu, Haixing Dai, Zihao Wu, Lu Zhang, Shu Zhang, Xiaoyan Cai, Xintao Hu, Shijie Zhao, Xi Jiang, Xin Zhang, Xiang Li, Dajiang Zhu, Lei Guo, Dinggang Shen, Junwei Han, Tianming Liu, Jun Liu, Tuo Zhang. (2023)  
**ChatRadio-Valuer: A Chat Large Language Model for Generalizable Radiology Report Generation Based on Multi-institution and Multi-system Data**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, ChatGPT, GPT, GPT-3.5, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2310.05242v1)  

---


**ABSTRACT**  
Radiology report generation, as a key step in medical image analysis, is critical to the quantitative analysis of clinically informed decision-making levels. However, complex and diverse radiology reports with cross-source heterogeneity pose a huge generalizability challenge to the current methods under massive data volume, mainly because the style and normativity of radiology reports are obviously distinctive among institutions, body regions inspected and radiologists. Recently, the advent of large language models (LLM) offers great potential for recognizing signs of health conditions. To resolve the above problem, we collaborate with the Second Xiangya Hospital in China and propose ChatRadio-Valuer based on the LLM, a tailored model for automatic radiology report generation that learns generalizable representations and provides a basis pattern for model adaptation in sophisticated analysts' cases. Specifically, ChatRadio-Valuer is trained based on the radiology reports from a single institution by means of supervised fine-tuning, and then adapted to disease diagnosis tasks for human multi-system evaluation (i.e., chest, abdomen, muscle-skeleton, head, and maxillofacial $\&$ neck) from six different institutions in clinical-level events. The clinical dataset utilized in this study encompasses a remarkable total of \textbf{332,673} observations. From the comprehensive results on engineering indicators, clinical efficacy and deployment cost metrics, it can be shown that ChatRadio-Valuer consistently outperforms state-of-the-art models, especially ChatGPT (GPT-3.5-Turbo) and GPT-4 et al., in terms of the diseases diagnosis from radiology reports. ChatRadio-Valuer provides an effective avenue to boost model generalization performance and alleviate the annotation workload of experts to enable the promotion of clinical AI applications in radiology reports.

{{</citation>}}


### (29/112) XLS-R fine-tuning on noisy word boundaries for unsupervised speech segmentation into words (Robin Algayres et al., 2023)

{{<citation>}}

Robin Algayres, Pablo Diego-Simon, Benoit Sagot, Emmanuel Dupoux. (2023)  
**XLS-R fine-tuning on noisy word boundaries for unsupervised speech segmentation into words**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2310.05235v1)  

---


**ABSTRACT**  
Due to the absence of explicit word boundaries in the speech stream, the task of segmenting spoken sentences into word units without text supervision is particularly challenging. In this work, we leverage the most recent self-supervised speech models that have proved to quickly adapt to new tasks through fine-tuning, even in low resource conditions. Taking inspiration from semi-supervised learning, we fine-tune an XLS-R model to predict word boundaries themselves produced by top-tier speech segmentation systems: DPDP, VG-HuBERT, GradSeg and DP-Parse. Once XLS-R is fine-tuned, it is used to infer new word boundary labels that are used in turn for another fine-tuning step. Our method consistently improves the performance of each system and sets a new state-of-the-art that is, on average 130% higher than the previous one as measured by the F1 score on correctly discovered word tokens on five corpora featuring different languages. Finally, our system can segment speech from languages unseen during fine-tuning in a zero-shot fashion.

{{</citation>}}


### (30/112) Generative Spoken Language Model based on continuous word-sized audio tokens (Robin Algayres et al., 2023)

{{<citation>}}

Robin Algayres, Yossi Adi, Tu Anh Nguyen, Jade Copet, Gabriel Synnaeve, Benoit Sagot, Emmanuel Dupoux. (2023)  
**Generative Spoken Language Model based on continuous word-sized audio tokens**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Embedding, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2310.05224v1)  

---


**ABSTRACT**  
In NLP, text language models based on words or subwords are known to outperform their character-based counterparts. Yet, in the speech community, the standard input of spoken LMs are 20ms or 40ms-long discrete units (shorter than a phoneme). Taking inspiration from word-based LM, we introduce a Generative Spoken Language Model (GSLM) based on word-size continuous-valued audio embeddings that can generate diverse and expressive language output. This is obtained by replacing lookup table for lexical types with a Lexical Embedding function, the cross entropy loss by a contrastive loss, and multinomial sampling by k-NN sampling. The resulting model is the first generative language model based on word-size continuous embeddings. Its performance is on par with discrete unit GSLMs regarding generation quality as measured by automatic metrics and subjective human judgements. Moreover, it is five times more memory efficient thanks to its large 200ms units. In addition, the embeddings before and after the Lexical Embedder are phonetically and semantically interpretable.

{{</citation>}}


### (31/112) Probing Language Models from A Human Behavioral Perspective (Xintong Wang et al., 2023)

{{<citation>}}

Xintong Wang, Xiaoyu Li, Xingshan Li, Chris Biemann. (2023)  
**Probing Language Models from A Human Behavioral Perspective**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2310.05216v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have emerged as dominant foundational models in modern NLP. However, the understanding of their prediction process and internal mechanisms, such as feed-forward networks and multi-head self-attention, remains largely unexplored. In this study, we probe LLMs from a human behavioral perspective, correlating values from LLMs with eye-tracking measures, which are widely recognized as meaningful indicators of reading patterns. Our findings reveal that LLMs exhibit a prediction pattern distinct from that of RNN-based LMs. Moreover, with the escalation of FFN layers, the capacity for memorization and linguistic knowledge encoding also surges until it peaks, subsequently pivoting to focus on comprehension capacity. The functions of self-attention are distributed across multiple heads. Lastly, we scrutinize the gate mechanisms, finding that they control the flow of information, with some gates promoting, while others eliminating information.

{{</citation>}}


### (32/112) Scaling Laws of RoPE-based Extrapolation (Xiaoran Liu et al., 2023)

{{<citation>}}

Xiaoran Liu, Hang Yan, Shuo Zhang, Chenxin An, Xipeng Qiu, Dahua Lin. (2023)  
**Scaling Laws of RoPE-based Extrapolation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Embedding, LLaMA, Language Model, Position Embedding  
[Paper Link](http://arxiv.org/abs/2310.05209v1)  

---


**ABSTRACT**  
The extrapolation capability of Large Language Models (LLMs) based on Rotary Position Embedding is currently a topic of considerable interest. The mainstream approach to addressing extrapolation with LLMs involves modifying RoPE by replacing 10000, the rotary base of $\theta_n={10000}^{-2n/d}$ in the original RoPE, with a larger value and providing longer fine-tuning text. In this work, we first observe that fine-tuning a RoPE-based LLM with either a smaller or larger base in pre-training context length could significantly enhance its extrapolation performance. After that, we propose \textbf{\textit{Scaling Laws of RoPE-based Extrapolation}}, a unified framework from the periodic perspective, to describe the relationship between the extrapolation performance and base value as well as tuning context length. In this process, we also explain the origin of the RoPE-based extrapolation issue by \textbf{\textit{critical dimension for extrapolation}}. Besides these observations and analyses, we achieve extrapolation up to 1 million context length within only 16K training length on LLaMA2 7B and 13B.

{{</citation>}}


### (33/112) Loose lips sink ships: Mitigating Length Bias in Reinforcement Learning from Human Feedback (Wei Shen et al., 2023)

{{<citation>}}

Wei Shen, Rui Zheng, Wenyu Zhan, Jun Zhao, Shihan Dou, Tao Gui, Qi Zhang, Xuanjing Huang. (2023)  
**Loose lips sink ships: Mitigating Length Bias in Reinforcement Learning from Human Feedback**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Bias, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.05199v1)  

---


**ABSTRACT**  
Reinforcement learning from human feedback serves as a crucial bridge, aligning large language models with human and societal values. This alignment requires a vast corpus of human feedback to learn a reward model, which is subsequently used to finetune language models. However, we have identified that the reward model often finds shortcuts to bypass its intended objectives, misleadingly assuming that humans prefer longer responses. The emergence of length bias often induces the model to favor longer outputs, yet it doesn't equate to an increase in helpful information within these outputs. In this paper, we propose an innovative solution, applying the Product-of-Experts (PoE) technique to separate reward modeling from the influence of sequence length. In our framework, the main expert concentrates on understanding human intents, while the biased expert targets the identification and capture of length bias. To further enhance the learning of bias, we introduce perturbations into the bias-focused expert, disrupting the flow of semantic information. Experimental results validate the effectiveness of our approach, indicating that language model performance is improved, irrespective of sequence length.

{{</citation>}}


### (34/112) FABRIC: Automated Scoring and Feedback Generation for Essays (Jieun Han et al., 2023)

{{<citation>}}

Jieun Han, Haneul Yoo, Junho Myung, Minsun Kim, Hyunseung Lim, Yoonsu Kim, Tak Yeon Lee, Hwajung Hong, Juho Kim, So-Yeon Ahn, Alice Oh. (2023)  
**FABRIC: Automated Scoring and Feedback Generation for Essays**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2310.05191v1)  

---


**ABSTRACT**  
Automated essay scoring (AES) provides a useful tool for students and instructors in writing classes by generating essay scores in real-time. However, previous AES models do not provide more specific rubric-based scores nor feedback on how to improve the essays, which can be even more important than the overall scores for learning. We present FABRIC, a pipeline to help students and instructors in English writing classes by automatically generating 1) the overall scores, 2) specific rubric-based scores, and 3) detailed feedback on how to improve the essays. Under the guidance of English education experts, we chose the rubrics for the specific scores as content, organization, and language. The first component of the FABRIC pipeline is DREsS, a real-world Dataset for Rubric-based Essay Scoring (DREsS). The second component is CASE, a Corruption-based Augmentation Strategy for Essays, with which we can improve the accuracy of the baseline model by 45.44%. The third component is EssayCoT, the Essay Chain-of-Thought prompting strategy which uses scores predicted from the AES model to generate better feedback. We evaluate the effectiveness of the new dataset DREsS and the augmentation strategy CASE quantitatively and show significant improvements over the models trained with existing datasets. We evaluate the feedback generated by EssayCoT with English education experts to show significant improvements in the helpfulness of the feedback across all rubrics. Lastly, we evaluate the FABRIC pipeline with students in a college English writing class who rated the generated scores and feedback with an average of 6 on the Likert scale from 1 to 7.

{{</citation>}}


### (35/112) Factuality Challenges in the Era of Large Language Models (Isabelle Augenstein et al., 2023)

{{<citation>}}

Isabelle Augenstein, Timothy Baldwin, Meeyoung Cha, Tanmoy Chakraborty, Giovanni Luca Ciampaglia, David Corney, Renee DiResta, Emilio Ferrara, Scott Hale, Alon Halevy, Eduard Hovy, Heng Ji, Filippo Menczer, Ruben Miguez, Preslav Nakov, Dietram Scheufele, Shivam Sharma, Giovanni Zagni. (2023)  
**Factuality Challenges in the Era of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: AI, ChatGPT, GPT, Google, Language Model, Microsoft  
[Paper Link](http://arxiv.org/abs/2310.05189v1)  

---


**ABSTRACT**  
The emergence of tools based on Large Language Models (LLMs), such as OpenAI's ChatGPT, Microsoft's Bing Chat, and Google's Bard, has garnered immense public attention. These incredibly useful, natural-sounding tools mark significant advances in natural language generation, yet they exhibit a propensity to generate false, erroneous, or misleading content -- commonly referred to as "hallucinations". Moreover, LLMs can be exploited for malicious applications, such as generating false but credible-sounding content and profiles at scale. This poses a significant challenge to society in terms of the potential deception of users and the increasing dissemination of inaccurate information. In light of these risks, we explore the kinds of technological innovations, regulatory reforms, and AI literacy initiatives needed from fact-checkers, news organizations, and the broader research and policy communities. By identifying the risks, the imminent threats, and some viable solutions, we seek to shed light on navigating various aspects of veracity in the era of generative AI.

{{</citation>}}


### (36/112) Do Large Language Models Know about Facts? (Xuming Hu et al., 2023)

{{<citation>}}

Xuming Hu, Junzhe Chen, Xiaochuan Li, Yufei Guo, Lijie Wen, Philip S. Yu, Zhijiang Guo. (2023)  
**Do Large Language Models Know about Facts?**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.05177v1)  

---


**ABSTRACT**  
Large language models (LLMs) have recently driven striking performance improvements across a range of natural language processing tasks. The factual knowledge acquired during pretraining and instruction tuning can be useful in various downstream tasks, such as question answering, and language generation. Unlike conventional Knowledge Bases (KBs) that explicitly store factual knowledge, LLMs implicitly store facts in their parameters. Content generated by the LLMs can often exhibit inaccuracies or deviations from the truth, due to facts that can be incorrectly induced or become obsolete over time. To this end, we aim to comprehensively evaluate the extent and scope of factual knowledge within LLMs by designing the benchmark Pinocchio. Pinocchio contains 20K diverse factual questions that span different sources, timelines, domains, regions, and languages. Furthermore, we investigate whether LLMs are able to compose multiple facts, update factual knowledge temporally, reason over multiple pieces of facts, identify subtle factual differences, and resist adversarial examples. Extensive experiments on different sizes and types of LLMs show that existing LLMs still lack factual knowledge and suffer from various spurious correlations. We believe this is a critical bottleneck for realizing trustworthy artificial intelligence. The dataset Pinocchio and our codes will be publicly available.

{{</citation>}}


### (37/112) On the Zero-Shot Generalization of Machine-Generated Text Detectors (Xiao Pu et al., 2023)

{{<citation>}}

Xiao Pu, Jingyu Zhang, Xiaochuang Han, Yulia Tsvetkov, Tianxing He. (2023)  
**On the Zero-Shot Generalization of Machine-Generated Text Detectors**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2310.05165v1)  

---


**ABSTRACT**  
The rampant proliferation of large language models, fluent enough to generate text indistinguishable from human-written language, gives unprecedented importance to the detection of machine-generated text. This work is motivated by an important research question: How will the detectors of machine-generated text perform on outputs of a new generator, that the detectors were not trained on? We begin by collecting generation data from a wide range of LLMs, and train neural detectors on data from each generator and test its performance on held-out generators. While none of the detectors can generalize to all generators, we observe a consistent and interesting pattern that the detectors trained on data from a medium-size LLM can zero-shot generalize to the larger version. As a concrete application, we demonstrate that robust detectors can be built on an ensemble of training data from medium-sized models.

{{</citation>}}


### (38/112) An Investigation of LLMs' Inefficacy in Understanding Converse Relations (Chengwen Qi et al., 2023)

{{<citation>}}

Chengwen Qi, Bowen Li, Binyuan Hui, Bailin Wang, Jinyang Li, Jinwang Wu, Yuanjun Laili. (2023)  
**An Investigation of LLMs' Inefficacy in Understanding Converse Relations**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.05163v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have achieved remarkable success in many formal language oriented tasks, such as structural data-to-text and semantic parsing. However current benchmarks mostly follow the data distribution of the pre-training data of LLMs. Therefore, a natural question rises that do LLMs really understand the structured semantics of formal languages. In this paper, we investigate this problem on a special case, converse binary relation. We introduce a new benchmark ConvRe focusing on converse relations, which contains 17 relations and 1240 triples extracted from popular knowledge graph completion datasets. Our ConvRE features two tasks, Re2Text and Text2Re, which are formulated as multi-choice question answering to evaluate LLMs' ability to determine the matching between relations and associated text. For the evaluation protocol, apart from different prompting methods, we further introduce variants to the test text and few-shot example text. We conduct experiments on three popular LLM families and have observed various scaling trends. The results suggest that LLMs often resort to shortcut learning and still face challenges on our proposed benchmark.

{{</citation>}}


### (39/112) Recurrent Neural Language Models as Probabilistic Finite-state Automata (Anej Svete et al., 2023)

{{<citation>}}

Anej Svete, Ryan Cotterell. (2023)  
**Recurrent Neural Language Models as Probabilistic Finite-state Automata**  

---
Primary Category: cs.CL  
Categories: cs-CC, cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.05161v1)  

---


**ABSTRACT**  
Studying language models (LMs) in terms of well-understood formalisms allows us to precisely characterize their abilities and limitations. Previous work has investigated the representational capacity of recurrent neural network (RNN) LMs in terms of their capacity to recognize unweighted formal languages. However, LMs do not describe unweighted formal languages -- rather, they define probability distributions over strings. In this work, we study what classes of such probability distributions RNN LMs can represent, which allows us to make more direct statements about their capabilities. We show that simple RNNs are equivalent to a subclass of probabilistic finite-state automata, and can thus model a strict subset of probability distributions expressible by finite-state models. Furthermore, we study the space complexity of representing finite-state LMs with RNNs. We show that, to represent an arbitrary deterministic finite-state LM with $N$ states over an alphabet $\Sigma$, an RNN requires $\Omega\left(N |\Sigma|\right)$ neurons. These results present a first step towards characterizing the classes of distributions RNN LMs can represent and thus help us understand their capabilities and limitations.

{{</citation>}}


### (40/112) MenatQA: A New Dataset for Testing the Temporal Comprehension and Reasoning Abilities of Large Language Models (Yifan Wei et al., 2023)

{{<citation>}}

Yifan Wei, Yisong Su, Huanhuan Ma, Xiaoyan Yu, Fangyu Lei, Yuanzhe Zhang, Jun Zhao, Kang Liu. (2023)  
**MenatQA: A New Dataset for Testing the Temporal Comprehension and Reasoning Abilities of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, NLP, QA, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.05157v1)  

---


**ABSTRACT**  
Large language models (LLMs) have shown nearly saturated performance on many natural language processing (NLP) tasks. As a result, it is natural for people to believe that LLMs have also mastered abilities such as time understanding and reasoning. However, research on the temporal sensitivity of LLMs has been insufficiently emphasized. To fill this gap, this paper constructs Multiple Sensitive Factors Time QA (MenatQA), which encompasses three temporal factors (scope factor, order factor, counterfactual factor) with total 2,853 samples for evaluating the time comprehension and reasoning abilities of LLMs. This paper tests current mainstream LLMs with different parameter sizes, ranging from billions to hundreds of billions. The results show most LLMs fall behind smaller temporal reasoning models with different degree on these factors. In specific, LLMs show a significant vulnerability to temporal biases and depend heavily on the temporal information provided in questions. Furthermore, this paper undertakes a preliminary investigation into potential improvement strategies by devising specific prompts and leveraging external tools. These approaches serve as valuable baselines or references for future research endeavors.

{{</citation>}}


### (41/112) Toolink: Linking Toolkit Creation and Using through Chain-of-Solving on Open-Source Model (Cheng Qian et al., 2023)

{{<citation>}}

Cheng Qian, Chenyan Xiong, Zhenghao Liu, Zhiyuan Liu. (2023)  
**Toolink: Linking Toolkit Creation and Using through Chain-of-Solving on Open-Source Model**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT, LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2310.05155v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have demonstrated remarkable progress in utilizing tools, but their closed-source nature and high inference costs pose limitations on their adaptability, necessitating a valid method that leverages smaller, open-sourced models. In this paper, we introduce Toolink, a comprehensive framework that performs task-solving by first creating a toolkit and then integrating the planning and calling of tools through a chain-of-solving (CoS) approach. We first validate the efficacy of Toolink in harnessing the model's creativity and CoS ability on ChatGPT. Subsequently, we curate CoS-GPT, a chain-of-solving dataset designed for tool-using, and finetune the LLaMA-7B model. It results in LLaMA-CoS, a powerful open-source model with advanced tool-planning and tool-calling capabilities. Evaluation on diverse tasks from BIG-bench demonstrates its CoS ability matches that of ChatGPT while its performance surpasses the chain-of-thought approach. Further studies highlight the generalization of LLaMA-CoS to unseen tasks and showcase its capability in using toolkits not explicitly tailored for the target task, affirming its robustness in real-world scenarios. All codes and data are released.

{{</citation>}}


### (42/112) From Data to Dialogue: Leveraging the Structure of Knowledge Graphs for Conversational Exploratory Search (Phillip Schneider et al., 2023)

{{<citation>}}

Phillip Schneider, Nils Rehtanz, Kristiina Jokinen, Florian Matthes. (2023)  
**From Data to Dialogue: Leveraging the Structure of Knowledge Graphs for Conversational Exploratory Search**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-IR, cs.CL  
Keywords: Dialog, Dialogue, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2310.05150v1)  

---


**ABSTRACT**  
Exploratory search is an open-ended information retrieval process that aims at discovering knowledge about a topic or domain rather than searching for a specific answer or piece of information. Conversational interfaces are particularly suitable for supporting exploratory search, allowing users to refine queries and examine search results through interactive dialogues. In addition to conversational search interfaces, knowledge graphs are also useful in supporting information exploration due to their rich semantic representation of data items. In this study, we demonstrate the synergistic effects of combining knowledge graphs and conversational interfaces for exploratory search, bridging the gap between structured and unstructured information retrieval. To this end, we propose a knowledge-driven dialogue system for exploring news articles by asking natural language questions and using the graph structure to navigate between related topics. Based on a user study with 54 participants, we empirically evaluate the effectiveness of the graph-based exploratory search and discuss design implications for developing such systems.

{{</citation>}}


### (43/112) Retrieval-Generation Synergy Augmented Large Language Models (Zhangyin Feng et al., 2023)

{{<citation>}}

Zhangyin Feng, Xiaocheng Feng, Dezhi Zhao, Maojin Yang, Bing Qin. (2023)  
**Retrieval-Generation Synergy Augmented Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, QA  
[Paper Link](http://arxiv.org/abs/2310.05149v1)  

---


**ABSTRACT**  
Large language models augmented with task-relevant documents have demonstrated impressive performance on knowledge-intensive tasks. However, regarding how to obtain effective documents, the existing methods are mainly divided into two categories. One is to retrieve from an external knowledge base, and the other is to utilize large language models to generate documents. We propose an iterative retrieval-generation collaborative framework. It is not only able to leverage both parametric and non-parametric knowledge, but also helps to find the correct reasoning path through retrieval-generation interactions, which is very important for tasks that require multi-step reasoning. We conduct experiments on four question answering datasets, including single-hop QA and multi-hop QA tasks. Empirical results show that our method significantly improves the reasoning ability of large language models and outperforms previous baselines.

{{</citation>}}


### (44/112) Harnessing the Power of Large Language Models for Empathetic Response Generation: Empirical Investigations and Improvements (Yushan Qian et al., 2023)

{{<citation>}}

Yushan Qian, Wei-Nan Zhang, Ting Liu. (2023)  
**Harnessing the Power of Large Language Models for Empathetic Response Generation: Empirical Investigations and Improvements**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, ChatGPT, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2310.05140v1)  

---


**ABSTRACT**  
Empathetic dialogue is an indispensable part of building harmonious social relationships and contributes to the development of a helpful AI. Previous approaches are mainly based on fine small-scale language models. With the advent of ChatGPT, the application effect of large language models (LLMs) in this field has attracted great attention. This work empirically investigates the performance of LLMs in generating empathetic responses and proposes three improvement methods of semantically similar in-context learning, two-stage interactive generation, and combination with the knowledge base. Extensive experiments show that LLMs can significantly benefit from our proposed methods and is able to achieve state-of-the-art performance in both automatic and human evaluations. Additionally, we explore the possibility of GPT-4 simulating human evaluators.

{{</citation>}}


### (45/112) Are Emily and Greg Still More Employable than Lakisha and Jamal? Investigating Algorithmic Hiring Bias in the Era of ChatGPT (Akshaj Kumar Veldanda et al., 2023)

{{<citation>}}

Akshaj Kumar Veldanda, Fabian Grob, Shailja Thakur, Hammond Pearce, Benjamin Tan, Ramesh Karri, Siddharth Garg. (2023)  
**Are Emily and Greg Still More Employable than Lakisha and Jamal? Investigating Algorithmic Hiring Bias in the Era of ChatGPT**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Bias, ChatGPT, GPT, GPT-3.5, Language Model  
[Paper Link](http://arxiv.org/abs/2310.05135v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) such as GPT-3.5, Bard, and Claude exhibit applicability across numerous tasks. One domain of interest is their use in algorithmic hiring, specifically in matching resumes with job categories. Yet, this introduces issues of bias on protected attributes like gender, race and maternity status. The seminal work of Bertrand & Mullainathan (2003) set the gold-standard for identifying hiring bias via field experiments where the response rate for identical resumes that differ only in protected attributes, e.g., racially suggestive names such as Emily or Lakisha, is compared. We replicate this experiment on state-of-art LLMs (GPT-3.5, Bard, Claude and Llama) to evaluate bias (or lack thereof) on gender, race, maternity status, pregnancy status, and political affiliation. We evaluate LLMs on two tasks: (1) matching resumes to job categories; and (2) summarizing resumes with employment relevant information. Overall, LLMs are robust across race and gender. They differ in their performance on pregnancy status and political affiliation. We use contrastive input decoding on open-source LLMs to uncover potential sources of bias.

{{</citation>}}


### (46/112) Fast-DetectGPT: Efficient Zero-Shot Detection of Machine-Generated Text via Conditional Probability Curvature (Guangsheng Bao et al., 2023)

{{<citation>}}

Guangsheng Bao, Yanbin Zhao, Zhiyang Teng, Linyi Yang, Yue Zhang. (2023)  
**Fast-DetectGPT: Efficient Zero-Shot Detection of Machine-Generated Text via Conditional Probability Curvature**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, GPT, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2310.05130v1)  

---


**ABSTRACT**  
Large language models (LLMs) have shown the ability to produce fluent and cogent content, presenting both productivity opportunities and societal risks. To build trustworthy AI systems, it is imperative to distinguish between machine-generated and human-authored content. The leading zero-shot detector, DetectGPT, showcases commendable performance but is marred by its intensive computational costs. In this paper, we introduce the concept of conditional probability curvature to elucidate discrepancies in word choices between LLMs and humans within a given context. Utilizing this curvature as a foundational metric, we present Fast-DetectGPT, an optimized zero-shot detector, which substitutes DetectGPT's perturbation step with a more efficient sampling step. Our evaluations on various datasets, source models, and test conditions indicate that Fast-DetectGPT not only outperforms DetectGPT in both the white-box and black-box settings but also accelerates the detection process by a factor of 340, as detailed in Table 1.

{{</citation>}}


### (47/112) Instances and Labels: Hierarchy-aware Joint Supervised Contrastive Learning for Hierarchical Multi-Label Text Classification (Simon Chi Lok U et al., 2023)

{{<citation>}}

Simon Chi Lok U, Jie He, Víctor Gutiérrez-Basulto, Jeff Z. Pan. (2023)  
**Instances and Labels: Hierarchy-aware Joint Supervised Contrastive Learning for Hierarchical Multi-Label Text Classification**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Contrastive Learning, Text Classification  
[Paper Link](http://arxiv.org/abs/2310.05128v1)  

---


**ABSTRACT**  
Hierarchical multi-label text classification (HMTC) aims at utilizing a label hierarchy in multi-label classification. Recent approaches to HMTC deal with the problem of imposing an overconstrained premise on the output space by using contrastive learning on generated samples in a semi-supervised manner to bring text and label embeddings closer. However, the generation of samples tends to introduce noise as it ignores the correlation between similar samples in the same batch. One solution to this issue is supervised contrastive learning, but it remains an underexplored topic in HMTC due to its complex structured labels. To overcome this challenge, we propose HJCL, a \textbf{H}ierarchy-aware \textbf{J}oint Supervised \textbf{C}ontrastive \textbf{L}earning method that bridges the gap between supervised contrastive learning and HMTC. Specifically, we employ both instance-wise and label-wise contrastive learning techniques and carefully construct batches to fulfill the contrastive learning objective.

{{</citation>}}


### (48/112) Breaking Down Word Semantics from Pre-trained Language Models through Layer-wise Dimension Selection (Nayoung Choi, 2023)

{{<citation>}}

Nayoung Choi. (2023)  
**Breaking Down Word Semantics from Pre-trained Language Models through Layer-wise Dimension Selection**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.05115v1)  

---


**ABSTRACT**  
Contextual word embeddings obtained from pre-trained language model (PLM) have proven effective for various natural language processing tasks at the word level. However, interpreting the hidden aspects within embeddings, such as syntax and semantics, remains challenging. Disentangled representation learning has emerged as a promising approach, which separates specific aspects into distinct embeddings. Furthermore, different linguistic knowledge is believed to be stored in different layers of PLM. This paper aims to disentangle semantic sense from BERT by applying a binary mask to middle outputs across the layers, without updating pre-trained parameters. The disentangled embeddings are evaluated through binary classification to determine if the target word in two different sentences has the same meaning. Experiments with cased BERT$_{\texttt{base}}$ show that leveraging layer-wise information is effective and disentangling semantic sense further improve performance.

{{</citation>}}


### (49/112) Zero-Shot Detection of Machine-Generated Codes (Xianjun Yang et al., 2023)

{{<citation>}}

Xianjun Yang, Kexun Zhang, Haifeng Chen, Linda Petzold, William Yang Wang, Wei Cheng. (2023)  
**Zero-Shot Detection of Machine-Generated Codes**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CR, cs-LG, cs.CL  
Keywords: ChatGPT, GPT, GPT-3.5, GPT-4, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2310.05103v1)  

---


**ABSTRACT**  
This work proposes a training-free approach for the detection of LLMs-generated codes, mitigating the risks associated with their indiscriminate usage. To the best of our knowledge, our research is the first to investigate zero-shot detection techniques applied to code generated by advanced black-box LLMs like ChatGPT. Firstly, we find that existing training-based or zero-shot text detectors are ineffective in detecting code, likely due to the unique statistical properties found in code structures. We then modify the previous zero-shot text detection method, DetectGPT (Mitchell et al., 2023) by utilizing a surrogate white-box model to estimate the probability of the rightmost tokens, allowing us to identify code snippets generated by language models. Through extensive experiments conducted on the python codes of the CodeContest and APPS dataset, our approach demonstrates its effectiveness by achieving state-of-the-art detection results on text-davinci-003, GPT-3.5, and GPT-4 models. Moreover, our method exhibits robustness against revision attacks and generalizes well to Java codes. We also find that the smaller code language model like PolyCoder-160M performs as a universal code detector, outperforming the billion-scale counterpart. The codes will be available at https://github.com/ Xianjun-Yang/Code_detection.git

{{</citation>}}


### (50/112) How Reliable Are AI-Generated-Text Detectors? An Assessment Framework Using Evasive Soft Prompts (Tharindu Kumarage et al., 2023)

{{<citation>}}

Tharindu Kumarage, Paras Sheth, Raha Moraffah, Joshua Garland, Huan Liu. (2023)  
**How Reliable Are AI-Generated-Text Detectors? An Assessment Framework Using Evasive Soft Prompts**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, GPT  
[Paper Link](http://arxiv.org/abs/2310.05095v1)  

---


**ABSTRACT**  
In recent years, there has been a rapid proliferation of AI-generated text, primarily driven by the release of powerful pre-trained language models (PLMs). To address the issue of misuse associated with AI-generated text, various high-performing detectors have been developed, including the OpenAI detector and the Stanford DetectGPT. In our study, we ask how reliable these detectors are. We answer the question by designing a novel approach that can prompt any PLM to generate text that evades these high-performing detectors. The proposed approach suggests a universal evasive prompt, a novel type of soft prompt, which guides PLMs in producing "human-like" text that can mislead the detectors. The novel universal evasive prompt is achieved in two steps: First, we create an evasive soft prompt tailored to a specific PLM through prompt tuning; and then, we leverage the transferability of soft prompts to transfer the learned evasive soft prompt from one PLM to another. Employing multiple PLMs in various writing tasks, we conduct extensive experiments to evaluate the efficacy of the evasive soft prompts in their evasion of state-of-the-art detectors.

{{</citation>}}


### (51/112) Benchmarking Large Language Models with Augmented Instructions for Fine-grained Information Extraction (Jun Gao et al., 2023)

{{<citation>}}

Jun Gao, Huan Zhao, Yice Zhang, Wei Wang, Changlong Yu, Ruifeng Xu. (2023)  
**Benchmarking Large Language Models with Augmented Instructions for Fine-grained Information Extraction**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Information Extraction, Language Model, Natural Language Processing, T5  
[Paper Link](http://arxiv.org/abs/2310.05092v1)  

---


**ABSTRACT**  
Information Extraction (IE) is an essential task in Natural Language Processing. Traditional methods have relied on coarse-grained extraction with simple instructions. However, with the emergence of Large Language Models (LLMs), there is a need to adapt IE techniques to leverage the capabilities of these models. This paper introduces a fine-grained IE benchmark dataset tailored for LLMs, employing augmented instructions for each information type, which includes task descriptions, extraction rules, output formats, and examples. Through extensive evaluations, we observe that encoder-decoder models, particularly T5 and FLAN-T5, perform well in generalizing to unseen information types, while ChatGPT exhibits greater adaptability to new task forms. Our results also indicate that performance is not solely dictated by model scale, and highlight the significance of architecture, data diversity, and learning techniques. This work paves the way for a more refined and versatile utilization of LLMs in Information Extraction.

{{</citation>}}


### (52/112) DialCoT Meets PPO: Decomposing and Exploring Reasoning Paths in Smaller Language Models (Chengcheng Han et al., 2023)

{{<citation>}}

Chengcheng Han, Xiaowei Du, Che Zhang, Yixin Lian, Xiang Li, Ming Gao, Baoyuan Wang. (2023)  
**DialCoT Meets PPO: Decomposing and Exploring Reasoning Paths in Smaller Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Dialog, Dialogue, Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.05074v1)  

---


**ABSTRACT**  
Chain-of-Thought (CoT) prompting has proven to be effective in enhancing the reasoning capabilities of Large Language Models (LLMs) with at least 100 billion parameters. However, it is ineffective or even detrimental when applied to reasoning tasks in Smaller Language Models (SLMs) with less than 10 billion parameters. To address this limitation, we introduce Dialogue-guided Chain-of-Thought (DialCoT) which employs a dialogue format to generate intermediate reasoning steps, guiding the model toward the final answer. Additionally, we optimize the model's reasoning path selection using the Proximal Policy Optimization (PPO) algorithm, further enhancing its reasoning capabilities. Our method offers several advantages compared to previous approaches. Firstly, we transform the process of solving complex reasoning questions by breaking them down into a series of simpler sub-questions, significantly reducing the task difficulty and making it more suitable for SLMs. Secondly, we optimize the model's reasoning path selection through the PPO algorithm. We conduct comprehensive experiments on four arithmetic reasoning datasets, demonstrating that our method achieves significant performance improvements compared to state-of-the-art competitors.

{{</citation>}}


### (53/112) Unleashing the Multilingual Encoder Potential: Boosting Zero-Shot Performance via Probability Calibration (Ercong Nie et al., 2023)

{{<citation>}}

Ercong Nie, Helmut Schmid, Hinrich Schütze. (2023)  
**Unleashing the Multilingual Encoder Potential: Boosting Zero-Shot Performance via Probability Calibration**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Multilingual, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2310.05069v1)  

---


**ABSTRACT**  
Pretraiend multilingual encoder models can directly perform zero-shot multilingual tasks or linguistic probing by reformulating the input examples into cloze-style prompts. This is accomplished by predicting the probabilities of the label words at the masked token position, without requiring any updates to the model parameters. However, the performance of this pattern is limited by the model's bias toward predicting label words which frequently occurred during the pretraining. These words typically receive high probabilities. To address this issue, we combine the models with various calibration techniques which modify the probabilities of label words predicted by the models. We evaluate the effectiveness of these calibration methods on monolingual encoders as well as multilingual encoders. Across a diverse range of tasks, we achieve substantial performance gains through calibration. Furthermore, with only very few training samples, the trained calibration parameters are able to yield additional enhancements.

{{</citation>}}


### (54/112) Guideline Learning for In-context Information Extraction (Chaoxu Pang et al., 2023)

{{<citation>}}

Chaoxu Pang, Yixuan Cao, Qiang Ding, Ping Luo. (2023)  
**Guideline Learning for In-context Information Extraction**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Information Extraction  
[Paper Link](http://arxiv.org/abs/2310.05066v1)  

---


**ABSTRACT**  
Large language models (LLMs) can perform a new task by merely conditioning on task instructions and a few input-output examples, without optimizing any parameters. This is called In-Context Learning (ICL). In-context Information Extraction has recently garnered attention in the research community. However, current experiment results are generally suboptimal. We attribute this primarily to the fact that the complex task settings and a variety of edge cases are hard to be fully expressed in the length-limited context. In this paper, we propose a Guideline Learning (GL) framework for In-context IE which learns to generate and follow guidelines. During the learning phrase, GL automatically synthesizes a set of guidelines from a few annotations, and during inference, helpful guidelines are retrieved for better ICL.

{{</citation>}}


### (55/112) sign.mt: Real-Time Multilingual Sign Language Translation Application (Amit Moryossef, 2023)

{{<citation>}}

Amit Moryossef. (2023)  
**sign.mt: Real-Time Multilingual Sign Language Translation Application**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Multilingual, NLP  
[Paper Link](http://arxiv.org/abs/2310.05064v1)  

---


**ABSTRACT**  
This demo paper presents sign.mt, an open-source application pioneering real-time multilingual bi-directional translation between spoken and signed languages. Harnessing state-of-the-art open-source models, this tool aims to address the communication divide between the hearing and the deaf, facilitating seamless translation in both spoken-to-signed and signed-to-spoken translation directions.   Promising reliable and unrestricted communication, sign.mt offers offline functionality, crucial in areas with limited internet connectivity. It further enhances user engagement by offering customizable photo-realistic sign language avatars, thereby encouraging a more personalized and authentic user experience.   Licensed under CC BY-NC-SA 4.0, sign.mt signifies an important stride towards open, inclusive communication. The app can be used, and modified for personal and academic uses, and even supports a translation API, fostering integration into a wider range of applications. However, it is by no means a finished product.   We invite the NLP community to contribute towards the evolution of sign.mt. Whether it be the integration of more refined models, the development of innovative pipelines, or user experience improvements, your contributions can propel this project to new heights. Available at https://sign.mt, it stands as a testament to what we can achieve together, as we strive to make communication accessible to all.

{{</citation>}}


### (56/112) BRAINTEASER: Lateral Thinking Puzzles for Large Language Model (Yifan Jiang et al., 2023)

{{<citation>}}

Yifan Jiang, Filip Ilievski, Kaixin Ma. (2023)  
**BRAINTEASER: Lateral Thinking Puzzles for Large Language Model**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Language Model, NLP, Question Answering  
[Paper Link](http://arxiv.org/abs/2310.05057v1)  

---


**ABSTRACT**  
The success of language models has inspired the NLP community to attend to tasks that require implicit and complex reasoning, relying on human-like commonsense mechanisms. While such vertical thinking tasks have been relatively popular, lateral thinking puzzles have received little attention. To bridge this gap, we devise BRAINTEASER: a multiple-choice Question Answering task designed to test the model's ability to exhibit lateral thinking and defy default commonsense associations. We design a three-step procedure for creating the first lateral thinking benchmark, consisting of data collection, distractor generation, and generation of adversarial examples, leading to 1,100 puzzles with high-quality annotations. To assess the consistency of lateral reasoning by models, we enrich BRAINTEASER based on a semantic and contextual reconstruction of its questions. Our experiments with state-of-the-art instruction- and commonsense language models reveal a significant gap between human and model performance, which is further widened when consistency across adversarial formats is considered. We make all of our code and data available to stimulate work on developing and evaluating lateral thinking models.

{{</citation>}}


### (57/112) Harnessing the Power of ChatGPT in Fake News: An In-Depth Exploration in Generation, Detection and Explanation (Yue Huang et al., 2023)

{{<citation>}}

Yue Huang, Lichao Sun. (2023)  
**Harnessing the Power of ChatGPT in Fake News: An In-Depth Exploration in Generation, Detection and Explanation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, Fake News, GPT  
[Paper Link](http://arxiv.org/abs/2310.05046v1)  

---


**ABSTRACT**  
The rampant spread of fake news has adversely affected society, resulting in extensive research on curbing its spread. As a notable milestone in large language models (LLMs), ChatGPT has gained significant attention due to its exceptional natural language processing capabilities. In this study, we present a thorough exploration of ChatGPT's proficiency in generating, explaining, and detecting fake news as follows. Generation -- We employ four prompt methods to generate fake news samples and prove the high quality of these samples through both self-assessment and human evaluation. Explanation -- We obtain nine features to characterize fake news based on ChatGPT's explanations and analyze the distribution of these factors across multiple public datasets. Detection -- We examine ChatGPT's capacity to identify fake news. We explore its detection consistency and then propose a reason-aware prompt method to improve its performance. Although our experiments demonstrate that ChatGPT shows commendable performance in detecting fake news, there is still room for its improvement. Consequently, we further probe into the potential extra information that could bolster its effectiveness in detecting fake news.

{{</citation>}}


### (58/112) Self-Convinced Prompting: Few-Shot Question Answering with Repeated Introspection (Haodi Zhang et al., 2023)

{{<citation>}}

Haodi Zhang, Min Cai, Xinhe Zhang, Chen Jason Zhang, Rui Mao, Kaishun Wu. (2023)  
**Self-Convinced Prompting: Few-Shot Question Answering with Repeated Introspection**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, Few-Shot, GPT, PaLM, Question Answering  
[Paper Link](http://arxiv.org/abs/2310.05035v1)  

---


**ABSTRACT**  
While large language models (LLMs) such as ChatGPT and PaLM have demonstrated remarkable performance in various language understanding and generation tasks, their capabilities in complex reasoning and intricate knowledge utilization still fall short of human-level proficiency. Recent studies have established the effectiveness of prompts in steering LLMs towards generating desired outputs. Building on these insights, we introduce a novel framework that harnesses the potential of large-scale pre-trained language models, to iteratively enhance performance of the LLMs. Our framework incorporates three components: \textit{Normal CoT}, a \textit{Convincer}, and an \textit{Answerer}. It processes the output of a typical few-shot chain-of-thought prompt, assesses the correctness of the response, scrutinizes the answer, refines the reasoning, and ultimately produces a new solution. Experimental results on the 7 datasets of miscellaneous problems validate the efficacy of the Self-Convince framework, achieving substantial improvements compared to the baselines. This study contributes to the burgeoning body of research focused on integrating pre-trained language models with tailored prompts and iterative refinement processes to augment their performance in complex tasks.

{{</citation>}}


### (59/112) Counter Turing Test CT^2: AI-Generated Text Detection is Not as Easy as You May Think -- Introducing AI Detectability Index (Megha Chakraborty et al., 2023)

{{<citation>}}

Megha Chakraborty, S. M Towhidul Islam Tonmoy, S M Mehedi Zaman, Krish Sharma, Niyar R Barman, Chandan Gupta, Shreya Gautam, Tanay Kumar, Vinija Jain, Aman Chadha, Amit P. Sheth, Amitava Das. (2023)  
**Counter Turing Test CT^2: AI-Generated Text Detection is Not as Easy as You May Think -- Introducing AI Detectability Index**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, ChatGPT, GPT, NLP  
[Paper Link](http://arxiv.org/abs/2310.05030v1)  

---


**ABSTRACT**  
With the rise of prolific ChatGPT, the risk and consequences of AI-generated text has increased alarmingly. To address the inevitable question of ownership attribution for AI-generated artifacts, the US Copyright Office released a statement stating that 'If a work's traditional elements of authorship were produced by a machine, the work lacks human authorship and the Office will not register it'. Furthermore, both the US and the EU governments have recently drafted their initial proposals regarding the regulatory framework for AI. Given this cynosural spotlight on generative AI, AI-generated text detection (AGTD) has emerged as a topic that has already received immediate attention in research, with some initial methods having been proposed, soon followed by emergence of techniques to bypass detection. This paper introduces the Counter Turing Test (CT^2), a benchmark consisting of techniques aiming to offer a comprehensive evaluation of the robustness of existing AGTD techniques. Our empirical findings unequivocally highlight the fragility of the proposed AGTD methods under scrutiny. Amidst the extensive deliberations on policy-making for regulating AI development, it is of utmost importance to assess the detectability of content generated by LLMs. Thus, to establish a quantifiable spectrum facilitating the evaluation and ranking of LLMs according to their detectability levels, we propose the AI Detectability Index (ADI). We conduct a thorough examination of 15 contemporary LLMs, empirically demonstrating that larger LLMs tend to have a higher ADI, indicating they are less detectable compared to smaller LLMs. We firmly believe that ADI holds significant value as a tool for the wider NLP community, with the potential to serve as a rubric in AI-related policy-making.

{{</citation>}}


### (60/112) Synslator: An Interactive Machine Translation Tool with Online Learning (Jiayi Wang et al., 2023)

{{<citation>}}

Jiayi Wang, Ke Wang, Fengming Zhou, Chengyu Wang, Zhiyong Fu, Zeyu Feng, Yu Zhao, Yuqi Zhang. (2023)  
**Synslator: An Interactive Machine Translation Tool with Online Learning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Machine Translation  
[Paper Link](http://arxiv.org/abs/2310.05025v1)  

---


**ABSTRACT**  
Interactive machine translation (IMT) has emerged as a progression of the computer-aided translation paradigm, where the machine translation system and the human translator collaborate to produce high-quality translations. This paper introduces Synslator, a user-friendly computer-aided translation (CAT) tool that not only supports IMT, but is adept at online learning with real-time translation memories. To accommodate various deployment environments for CAT services, Synslator integrates two different neural translation models to handle translation memories for online learning. Additionally, the system employs a language model to enhance the fluency of translations in an interactive mode. In evaluation, we have confirmed the effectiveness of online learning through the translation models, and have observed a 13% increase in post-editing efficiency with the interactive functionalities of Synslator. A tutorial video is available at:https://youtu.be/K0vRsb2lTt8.

{{</citation>}}


### (61/112) MinPrompt: Graph-based Minimal Prompt Data Augmentation for Few-shot Question Answering (Xiusi Chen et al., 2023)

{{<citation>}}

Xiusi Chen, Jyun-Yu Jiang, Wei-Cheng Chang, Cho-Jui Hsieh, Hsiang-Fu Yu, Wei Wang. (2023)  
**MinPrompt: Graph-based Minimal Prompt Data Augmentation for Few-shot Question Answering**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Augmentation, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2310.05007v1)  

---


**ABSTRACT**  
Few-shot question answering (QA) aims at achieving satisfactory results on machine question answering when only a few training samples are available. Recent advances mostly rely on the power of pre-trained large language models (LLMs) and fine-tuning in specific settings. Although the pre-training stage has already equipped LLMs with powerful reasoning capabilities, LLMs still need to be fine-tuned to adapt to specific domains to achieve the best results. In this paper, we propose to select the most informative data for fine-tuning, thereby improving the efficiency of the fine-tuning process with comparative or even better accuracy on the open-domain QA task. We present MinPrompt, a minimal data augmentation framework for open-domain QA based on an approximate graph algorithm and unsupervised question generation. We transform the raw text into a graph structure to build connections between different factual sentences, then apply graph algorithms to identify the minimal set of sentences needed to cover the most information in the raw text. We then generate QA pairs based on the identified sentence subset and train the model on the selected sentences to obtain the final model. Empirical results on several benchmark datasets and theoretical analysis show that MinPrompt is able to achieve comparable or better results than baselines with a high degree of efficiency, bringing improvements in F-1 scores by up to 27.5%.

{{</citation>}}


### (62/112) Self-Knowledge Guided Retrieval Augmentation for Large Language Models (Yile Wang et al., 2023)

{{<citation>}}

Yile Wang, Peng Li, Maosong Sun, Yang Liu. (2023)  
**Self-Knowledge Guided Retrieval Augmentation for Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Augmentation, ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.05002v1)  

---


**ABSTRACT**  
Large language models (LLMs) have shown superior performance without task-specific fine-tuning. Despite the success, the knowledge stored in the parameters of LLMs could still be incomplete and difficult to update due to the computational costs. As complementary, retrieval-based methods can offer non-parametric world knowledge and improve the performance on tasks such as question answering. However, we find that the retrieved knowledge does not always help and even has a negative impact on original responses occasionally. To better make use of both internal knowledge and external world knowledge, we investigate eliciting the model's ability to recognize what they know and do not know (which is also called self-knowledge) and propose Self-Knowledge guided Retrieval augmentation (SKR), a simple yet effective method which can let LLMs refer to the questions they have previously encountered and adaptively call for external resources when dealing with new questions. We evaluate SKR on multiple datasets and demonstrate that it outperforms chain-of-thought based and fully retrieval-based methods by using either InstructGPT or ChatGPT.

{{</citation>}}


### (63/112) Distantly-Supervised Joint Entity and Relation Extraction with Noise-Robust Learning (Yufei Li et al., 2023)

{{<citation>}}

Yufei Li, Xiao Yu, Yanghong Guo, Yanchi Liu, Haifeng Chen, Cong Liu. (2023)  
**Distantly-Supervised Joint Entity and Relation Extraction with Noise-Robust Learning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: GPT, Relation Extraction  
[Paper Link](http://arxiv.org/abs/2310.04994v1)  

---


**ABSTRACT**  
Joint entity and relation extraction is a process that identifies entity pairs and their relations using a single model. We focus on the problem of training these models on distantly-labeled data, which is generated by aligning entity mentions in a text corpus with their corresponding entity and relation types in a knowledge base. One key challenge here is the presence of noisy labels, which arises from both entity and relation annotations, and significantly impair the effectiveness of supervised learning applications. However, existing research primarily addresses only one type of noise, thereby limiting the effectiveness of noise reduction. To fill this gap, we introduce a new noise-robust approach, that 1)~incorporates a pre-trained GPT-2 into a sequence tagging scheme for simultaneous entity and relation detection, and 2)~employs a noise-robust learning framework which includes a new loss function that penalizes inconsistency with both significant relation patterns and entity-relation dependencies, as well as a self-adaptive learning step that iteratively selects and trains on high-quality instances. Experiments on two datasets show that our method outperforms the existing state-of-the-art methods in both joint extraction performance and noise reduction effect.

{{</citation>}}


### (64/112) MULTISCRIPT: Multimodal Script Learning for Supporting Open Domain Everyday Tasks (Jingyuan Qi et al., 2023)

{{<citation>}}

Jingyuan Qi, Minqian Liu, Ying Shen, Zhiyang Xu, Lifu Huang. (2023)  
**MULTISCRIPT: Multimodal Script Learning for Supporting Open Domain Everyday Tasks**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.04965v1)  

---


**ABSTRACT**  
Automatically generating scripts (i.e. sequences of key steps described in text) from video demonstrations and reasoning about the subsequent steps are crucial to the modern AI virtual assistants to guide humans to complete everyday tasks, especially unfamiliar ones. However, current methods for generative script learning rely heavily on well-structured preceding steps described in text and/or images or are limited to a certain domain, resulting in a disparity with real-world user scenarios. To address these limitations, we present a new benchmark challenge -- MultiScript, with two new tasks on task-oriented multimodal script learning: (1) multimodal script generation, and (2) subsequent step prediction. For both tasks, the input consists of a target task name and a video illustrating what has been done to complete the target task, and the expected output is (1) a sequence of structured step descriptions in text based on the demonstration video, and (2) a single text description for the subsequent step, respectively. Built from WikiHow, MultiScript covers multimodal scripts in videos and text descriptions for over 6,655 human everyday tasks across 19 diverse domains. To establish baseline performance on MultiScript, we propose two knowledge-guided multimodal generative frameworks that incorporate the task-related knowledge prompted from large language models such as Vicuna. Experimental results show that our proposed approaches significantly improve over the competitive baselines.

{{</citation>}}


### (65/112) Exploring the Usage of Chinese Pinyin in Pretraining (Baojun Wang et al., 2023)

{{<citation>}}

Baojun Wang, Kun Xu, Lifeng Shang. (2023)  
**Exploring the Usage of Chinese Pinyin in Pretraining**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, NLP  
[Paper Link](http://arxiv.org/abs/2310.04960v1)  

---


**ABSTRACT**  
Unlike alphabetic languages, Chinese spelling and pronunciation are different. Both characters and pinyin take an important role in Chinese language understanding. In Chinese NLP tasks, we almost adopt characters or words as model input, and few works study how to use pinyin. However, pinyin is essential in many scenarios, such as error correction and fault tolerance for ASR-introduced errors. Most of these errors are caused by the same or similar pronunciation words, and we refer to this type of error as SSP(the same or similar pronunciation) errors for short. In this work, we explore various ways of using pinyin in pretraining models and propose a new pretraining method called PmBERT. Our method uses characters and pinyin in parallel for pretraining. Through delicate pretraining tasks, the characters and pinyin representation are fused, which can enhance the error tolerance for SSP errors. We do comprehensive experiments and ablation tests to explore what makes a robust phonetic enhanced Chinese language model. The experimental results on both the constructed noise-added dataset and the public error-correction dataset demonstrate that our model is more robust compared to SOTA models.

{{</citation>}}


### (66/112) Domain Knowledge Graph Construction Via A Simple Checker (Yueling Zeng et al., 2023)

{{<citation>}}

Yueling Zeng, Li-C. Wang. (2023)  
**Domain Knowledge Graph Construction Via A Simple Checker**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2310.04949v1)  

---


**ABSTRACT**  
With the availability of large language models, there is a growing interest for semiconductor chip design companies to leverage the technologies. For those companies, deployment of a new methodology must include two important considerations: confidentiality and scalability. In this context, this work tackles the problem of knowledge graph construction from hardware-design domain texts. We propose an oracle-checker scheme to leverage the power of GPT3.5 and demonstrate that the essence of the problem is in distillation of domain expert's background knowledge. Using RISC-V unprivileged ISA specification as an example, we explain key ideas and discuss practicality of our proposed oracle-checker approach.

{{</citation>}}


## cs.HC (2)



### (67/112) HypoCompass: Large-Language-Model-based Tutor for Hypothesis Construction in Debugging for Novices (Qianou Ma et al., 2023)

{{<citation>}}

Qianou Ma, Hua Shen, Kenneth Koedinger, Tongshuang Wu. (2023)  
**HypoCompass: Large-Language-Model-based Tutor for Hypothesis Construction in Debugging for Novices**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs-SE, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.05292v1)  

---


**ABSTRACT**  
With the prevalence of imperfect but capable LLMs in software development, it becomes increasingly important for developers to cultivate debugging skills -- to form hypotheses about the source of error in both their own codes and codes produced by their AI pair programmers. Despite the necessity, hypothesis construction in debugging is rarely taught due to a lack of explicit instruction. In this work, we explore whether LLMs can be used to train novices on hypothesis construction, by designing a theoretically motivated, LLM-augmented tutor -- HypoCompass. HypoCompass relies on LLMs for generating rich training materials guided by learning principles and presents them in a learning-by-teaching environment, where LLMs act as students who write bugs and attempt to fix them, and human novices focus on debugging in the role of a Teaching Assistant. Evaluations show that HypoCompass consistently generates high-quality training materials, and brings significant learning gain: In a pre-to-post test setup, 10 novices improved their performances by 17%, with a reduced completion time of 13%.

{{</citation>}}


### (68/112) MindfulDiary: Harnessing Large Language Model to Support Psychiatric Patients' Journaling (Taewan Kim et al., 2023)

{{<citation>}}

Taewan Kim, Seolyeong Bae, Hyun Ah Kim, Su-woo Lee, Hwajung Hong, Chanmo Yang, Young-Ho Kim. (2023)  
**MindfulDiary: Harnessing Large Language Model to Support Psychiatric Patients' Journaling**  

---
Primary Category: cs.HC  
Categories: H-5-2; I-2-7, cs-AI, cs-CL, cs-HC, cs.HC  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.05231v1)  

---


**ABSTRACT**  
In the mental health domain, Large Language Models (LLMs) offer promising new opportunities, though their inherent complexity and low controllability have raised questions about their suitability in clinical settings. We present MindfulDiary, a mobile journaling app incorporating an LLM to help psychiatric patients document daily experiences through conversation. Designed in collaboration with mental health professionals (MHPs), MindfulDiary takes a state-based approach to safely comply with the experts' guidelines while carrying on free-form conversations. Through a four-week field study involving 28 patients with major depressive disorder and five psychiatrists, we found that MindfulDiary supported patients in consistently enriching their daily records and helped psychiatrists better empathize with their patients through an understanding of their thoughts and daily contexts. Drawing on these findings, we discuss the implications of leveraging LLMs in the mental health domain, bridging the technical feasibility and their integration into clinical settings.

{{</citation>}}


## cs.CV (17)



### (69/112) Transforming Pixels into a Masterpiece: AI-Powered Art Restoration using a Novel Distributed Denoising CNN (DDCNN) (Sankar B. et al., 2023)

{{<citation>}}

Sankar B., Mukil Saravanan, Kalaivanan Kumar, Siri Dubbaka. (2023)  
**Transforming Pixels into a Masterpiece: AI-Powered Art Restoration using a Novel Distributed Denoising CNN (DDCNN)**  

---
Primary Category: cs.CV  
Categories: I-2; I-4; I-5, cs-AI, cs-CV, cs.CV, eess-IV  
Keywords: AI, Computer Vision  
[Paper Link](http://arxiv.org/abs/2310.05270v1)  

---


**ABSTRACT**  
Art restoration is crucial for preserving cultural heritage, but traditional methods have limitations in faithfully reproducing original artworks while addressing issues like fading, staining, and damage. We present an innovative approach using deep learning, specifically Convolutional Neural Networks (CNNs), and Computer Vision techniques to revolutionize art restoration. We start by creating a diverse dataset of deteriorated art images with various distortions and degradation levels. This dataset trains a Distributed Denoising CNN (DDCNN) to remove distortions while preserving intricate details. Our method is adaptable to different distortion types and levels, making it suitable for various deteriorated artworks, including paintings, sketches, and photographs. Extensive experiments demonstrate our approach's efficiency and effectiveness compared to other Denoising CNN models. We achieve a substantial reduction in distortion, transforming deteriorated artworks into masterpieces. Quantitative evaluations confirm our method's superiority over traditional techniques, reshaping the art restoration field and preserving cultural heritage. In summary, our paper introduces an AI-powered solution that combines Computer Vision and deep learning with DDCNN to restore artworks accurately, overcoming limitations and paving the way for future advancements in art restoration.

{{</citation>}}


### (70/112) Persis: A Persian Font Recognition Pipeline Using Convolutional Neural Networks (Mehrdad Mohammadian et al., 2023)

{{<citation>}}

Mehrdad Mohammadian, Neda Maleki, Tobias Olsson, Fredrik Ahlgren. (2023)  
**Persis: A Persian Font Recognition Pipeline Using Convolutional Neural Networks**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: OCR  
[Paper Link](http://arxiv.org/abs/2310.05255v1)  

---


**ABSTRACT**  
What happens if we encounter a suitable font for our design work but do not know its name? Visual Font Recognition (VFR) systems are used to identify the font typeface in an image. These systems can assist graphic designers in identifying fonts used in images. A VFR system also aids in improving the speed and accuracy of Optical Character Recognition (OCR) systems. In this paper, we introduce the first publicly available datasets in the field of Persian font recognition and employ Convolutional Neural Networks (CNN) to address this problem. The results show that the proposed pipeline obtained 78.0% top-1 accuracy on our new datasets, 89.1% on the IDPL-PFOD dataset, and 94.5% on the KAFD dataset. Furthermore, the average time spent in the entire pipeline for one sample of our proposed datasets is 0.54 and 0.017 seconds for CPU and GPU, respectively. We conclude that CNN methods can be used to recognize Persian fonts without the need for additional pre-processing steps such as feature extraction, binarization, normalization, etc.

{{</citation>}}


### (71/112) GMMFormer: Gaussian-Mixture-Model based Transformer for Efficient Partially Relevant Video Retrieval (Yuting Wang et al., 2023)

{{<citation>}}

Yuting Wang, Jinpeng Wang, Bin Chen, Ziyun Zeng, Shu-Tao Xia. (2023)  
**GMMFormer: Gaussian-Mixture-Model based Transformer for Efficient Partially Relevant Video Retrieval**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-IR, cs-MM, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.05195v1)  

---


**ABSTRACT**  
Given a text query, partially relevant video retrieval (PRVR) seeks to find untrimmed videos containing pertinent moments in a database. For PRVR, clip modeling is essential to capture the partial relationship between texts and videos. Current PRVR methods adopt scanning-based clip construction to achieve explicit clip modeling, which is information-redundant and requires a large storage overhead. To solve the efficiency problem of PRVR methods, this paper proposes GMMFormer, a \textbf{G}aussian-\textbf{M}ixture-\textbf{M}odel based Trans\textbf{former} which models clip representations implicitly. During frame interactions, we incorporate Gaussian-Mixture-Model constraints to focus each frame on its adjacent frames instead of the whole video. Then generated representations will contain multi-scale clip information, achieving implicit clip modeling. In addition, PRVR methods ignore semantic differences between text queries relevant to the same video, leading to a sparse embedding space. We propose a query diverse loss to distinguish these text queries, making the embedding space more intensive and contain more semantic information. Extensive experiments on three large-scale video datasets (\ie, TVR, ActivityNet Captions, and Charades-STA) demonstrate the superiority and efficiency of GMMFormer.

{{</citation>}}


### (72/112) Improving Discriminative Multi-Modal Learning with Large-Scale Pre-Trained Models (Chenzhuang Du et al., 2023)

{{<citation>}}

Chenzhuang Du, Yue Zhao, Chonghua Liao, Jiacheng You, Jie Fu, Hang Zhao. (2023)  
**Improving Discriminative Multi-Modal Learning with Large-Scale Pre-Trained Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Pre-Trained Model  
[Paper Link](http://arxiv.org/abs/2310.05193v1)  

---


**ABSTRACT**  
This paper investigates how to better leverage large-scale pre-trained uni-modal models to further enhance discriminative multi-modal learning. Even when fine-tuned with only uni-modal data, these models can outperform previous multi-modal models in certain tasks. It's clear that their incorporation into multi-modal learning would significantly improve performance. However, multi-modal learning with these models still suffers from insufficient learning of uni-modal features, which weakens the resulting multi-modal model's generalization ability. While fine-tuning uni-modal models separately and then aggregating their predictions is straightforward, it doesn't allow for adequate adaptation between modalities, also leading to sub-optimal results. To this end, we introduce Multi-Modal Low-Rank Adaptation learning (MMLoRA). By freezing the weights of uni-modal fine-tuned models, adding extra trainable rank decomposition matrices to them, and subsequently performing multi-modal joint training, our method enhances adaptation between modalities and boosts overall performance. We demonstrate the effectiveness of MMLoRA on three dataset categories: audio-visual (e.g., AVE, Kinetics-Sound, CREMA-D), vision-language (e.g., MM-IMDB, UPMC Food101), and RGB-Optical Flow (UCF101).

{{</citation>}}


### (73/112) HOD: A Benchmark Dataset for Harmful Object Detection (Eungyeom Ha et al., 2023)

{{<citation>}}

Eungyeom Ha, Heemook Kim, Sung Chul Hong, Dongbin Na. (2023)  
**HOD: A Benchmark Dataset for Harmful Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2310.05192v1)  

---


**ABSTRACT**  
Recent multi-media data such as images and videos have been rapidly spread out on various online services such as social network services (SNS). With the explosive growth of online media services, the number of image content that may harm users is also growing exponentially. Thus, most recent online platforms such as Facebook and Instagram have adopted content filtering systems to prevent the prevalence of harmful content and reduce the possible risk of adverse effects on users. Unfortunately, computer vision research on detecting harmful content has not yet attracted attention enough. Users of each platform still manually click the report button to recognize patterns of harmful content they dislike when exposed to harmful content. However, the problem with manual reporting is that users are already exposed to harmful content. To address these issues, our research goal in this work is to develop automatic harmful object detection systems for online services. We present a new benchmark dataset for harmful object detection. Unlike most related studies focusing on a small subset of object categories, our dataset addresses various categories. Specifically, our proposed dataset contains more than 10,000 images across 6 categories that might be harmful, consisting of not only normal cases but also hard cases that are difficult to detect. Moreover, we have conducted extensive experiments to evaluate the effectiveness of our proposed dataset. We have utilized the recently proposed state-of-the-art (SOTA) object detection architectures and demonstrated our proposed dataset can be greatly useful for the real-time harmful object detection task. The whole source codes and datasets are publicly accessible at https://github.com/poori-nuna/HOD-Benchmark-Dataset.

{{</citation>}}


### (74/112) Geometry Aware Field-to-field Transformations for 3D Semantic Segmentation (Dominik Hollidt et al., 2023)

{{<citation>}}

Dominik Hollidt, Clinton Wang, Polina Golland, Marc Pollefeys. (2023)  
**Geometry Aware Field-to-field Transformations for 3D Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2310.05133v1)  

---


**ABSTRACT**  
We present a novel approach to perform 3D semantic segmentation solely from 2D supervision by leveraging Neural Radiance Fields (NeRFs). By extracting features along a surface point cloud, we achieve a compact representation of the scene which is sample-efficient and conducive to 3D reasoning. Learning this feature space in an unsupervised manner via masked autoencoding enables few-shot segmentation. Our method is agnostic to the scene parameterization, working on scenes fit with any type of NeRF.

{{</citation>}}


### (75/112) UReader: Universal OCR-free Visually-situated Language Understanding with Multimodal Large Language Model (Jiabo Ye et al., 2023)

{{<citation>}}

Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Ming Yan, Guohai Xu, Chenliang Li, Junfeng Tian, Qi Qian, Ji Zhang, Qin Jin, Liang He, Xin Alex Lin, Fei Huang. (2023)  
**UReader: Universal OCR-free Visually-situated Language Understanding with Multimodal Large Language Model**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Language Model, OCR  
[Paper Link](http://arxiv.org/abs/2310.05126v1)  

---


**ABSTRACT**  
Text is ubiquitous in our visual world, conveying crucial information, such as in documents, websites, and everyday photographs. In this work, we propose UReader, a first exploration of universal OCR-free visually-situated language understanding based on the Multimodal Large Language Model (MLLM). By leveraging the shallow text recognition ability of the MLLM, we only finetuned 1.2% parameters and the training cost is much lower than previous work following domain-specific pretraining and finetuning paradigms. Concretely, UReader is jointly finetuned on a wide range of Visually-situated Language Understanding tasks via a unified instruction format. To enhance the visual text and semantic understanding, we further apply two auxiliary tasks with the same format, namely text reading and key points generation tasks. We design a shape-adaptive cropping module before the encoder-decoder architecture of MLLM to leverage the frozen low-resolution vision encoder for processing high-resolution images. Without downstream finetuning, our single model achieves state-of-the-art ocr-free performance in 8 out of 10 visually-situated language understanding tasks, across 5 domains: documents, tables, charts, natural images, and webpage screenshots. Codes and instruction-tuning datasets will be released.

{{</citation>}}


### (76/112) Cross-domain Robust Deepfake Bias Expansion Network for Face Forgery Detection (Weihua Liu et al., 2023)

{{<citation>}}

Weihua Liu, Lin Li, Chaochao Lin, Said Boumaraf. (2023)  
**Cross-domain Robust Deepfake Bias Expansion Network for Face Forgery Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Bias  
[Paper Link](http://arxiv.org/abs/2310.05124v1)  

---


**ABSTRACT**  
The rapid advancement of deepfake technologies raises significant concerns about the security of face recognition systems. While existing methods leverage the clues left by deepfake techniques for face forgery detection, malicious users may intentionally manipulate forged faces to obscure the traces of deepfake clues and thereby deceive detection tools. Meanwhile, attaining cross-domain robustness for data-based methods poses a challenge due to potential gaps in the training data, which may not encompass samples from all relevant domains. Therefore, in this paper, we introduce a solution - a Cross-Domain Robust Bias Expansion Network (BENet) - designed to enhance face forgery detection. BENet employs an auto-encoder to reconstruct input faces, maintaining the invariance of real faces while selectively enhancing the difference between reconstructed fake faces and their original counterparts. This enhanced bias forms a robust foundation upon which dependable forgery detection can be built. To optimize the reconstruction results in BENet, we employ a bias expansion loss infused with contrastive concepts to attain the aforementioned objective. In addition, to further heighten the amplification of forged clues, BENet incorporates a Latent-Space Attention (LSA) module. This LSA module effectively captures variances in latent features between the auto-encoder's encoder and decoder, placing emphasis on inconsistent forgery-related information. Furthermore, BENet incorporates a cross-domain detector with a threshold to determine whether the sample belongs to a known distribution. The correction of classification results through the cross-domain detector enables BENet to defend against unknown deepfake attacks from cross-domain. Extensive experiments demonstrate the superiority of BENet compared with state-of-the-art methods in intra-database and cross-database evaluations.

{{</citation>}}


### (77/112) Enhancing Representations through Heterogeneous Self-Supervised Learning (Zhong-Yu Li et al., 2023)

{{<citation>}}

Zhong-Yu Li, Bo-Wen Yin, Shanghua Gao, Yongxiang Liu, Li Liu, Ming-Ming Cheng. (2023)  
**Enhancing Representations through Heterogeneous Self-Supervised Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2310.05108v1)  

---


**ABSTRACT**  
Incorporating heterogeneous representations from different architectures has facilitated various vision tasks, e.g., some hybrid networks combine transformers and convolutions. However, complementarity between such heterogeneous architectures has not been well exploited in self-supervised learning. Thus, we propose Heterogeneous Self-Supervised Learning (HSSL), which enforces a base model to learn from an auxiliary head whose architecture is heterogeneous from the base model. In this process, HSSL endows the base model with new characteristics in a representation learning way without structural changes. To comprehensively understand the HSSL, we conduct experiments on various heterogeneous pairs containing a base model and an auxiliary head. We discover that the representation quality of the base model moves up as their architecture discrepancy grows. This observation motivates us to propose a search strategy that quickly determines the most suitable auxiliary head for a specific base model to learn and several simple but effective methods to enlarge the model discrepancy. The HSSL is compatible with various self-supervised methods, achieving superior performances on various downstream tasks, including image classification, semantic segmentation, instance segmentation, and object detection. Our source code will be made publicly available.

{{</citation>}}


### (78/112) OV-PARTS: Towards Open-Vocabulary Part Segmentation (Meng Wei et al., 2023)

{{<citation>}}

Meng Wei, Xiaoyu Yue, Wenwei Zhang, Shu Kong, Xihui Liu, Jiangmiao Pang. (2023)  
**OV-PARTS: Towards Open-Vocabulary Part Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Few-Shot, Semantic Segmentation, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2310.05107v1)  

---


**ABSTRACT**  
Segmenting and recognizing diverse object parts is a crucial ability in applications spanning various computer vision and robotic tasks. While significant progress has been made in object-level Open-Vocabulary Semantic Segmentation (OVSS), i.e., segmenting objects with arbitrary text, the corresponding part-level research poses additional challenges. Firstly, part segmentation inherently involves intricate boundaries, while limited annotated data compounds the challenge. Secondly, part segmentation introduces an open granularity challenge due to the diverse and often ambiguous definitions of parts in the open world. Furthermore, the large-scale vision and language models, which play a key role in the open vocabulary setting, struggle to recognize parts as effectively as objects. To comprehensively investigate and tackle these challenges, we propose an Open-Vocabulary Part Segmentation (OV-PARTS) benchmark. OV-PARTS includes refined versions of two publicly available datasets: Pascal-Part-116 and ADE20K-Part-234. And it covers three specific tasks: Generalized Zero-Shot Part Segmentation, Cross-Dataset Part Segmentation, and Few-Shot Part Segmentation, providing insights into analogical reasoning, open granularity and few-shot adapting abilities of models. Moreover, we analyze and adapt two prevailing paradigms of existing object-level OVSS methods for OV-PARTS. Extensive experimental analysis is conducted to inspire future research in leveraging foundational models for OV-PARTS. The code and dataset are available at https://github.com/OpenRobotLab/OV_PARTS.

{{</citation>}}


### (79/112) Video-CSR: Complex Video Digest Creation for Visual-Language Models (Tingkai Liu et al., 2023)

{{<citation>}}

Tingkai Liu, Yunzhe Tao, Haogeng Liu, Qihang Fan, Ding Zhou, Huaibo Huang, Ran He, Hongxia Yang. (2023)  
**Video-CSR: Complex Video Digest Creation for Visual-Language Models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Language Model, Summarization  
[Paper Link](http://arxiv.org/abs/2310.05060v1)  

---


**ABSTRACT**  
We present a novel task and human annotated dataset for evaluating the ability for visual-language models to generate captions and summaries for real-world video clips, which we call Video-CSR (Captioning, Summarization and Retrieval). The dataset contains 4.8K YouTube video clips of 20-60 seconds in duration and covers a wide range of topics and interests. Each video clip corresponds to 5 independently annotated captions (1 sentence) and summaries (3-10 sentences). Given any video selected from the dataset and its corresponding ASR information, we evaluate visual-language models on either caption or summary generation that is grounded in both the visual and auditory content of the video. Additionally, models are also evaluated on caption- and summary-based retrieval tasks, where the summary-based retrieval task requires the identification of a target video given excerpts of a corresponding summary. Given the novel nature of the paragraph-length video summarization task, we perform extensive comparative analyses of different existing evaluation metrics and their alignment with human preferences. Finally, we propose a foundation model with competitive generation and retrieval capabilities that serves as a baseline for the Video-CSR task. We aim for Video-CSR to serve as a useful evaluation set in the age of large language models and complex multi-modal tasks.

{{</citation>}}


### (80/112) FairTune: Optimizing Parameter Efficient Fine Tuning for Fairness in Medical Image Analysis (Raman Dutt et al., 2023)

{{<citation>}}

Raman Dutt, Ondrej Bohdal, Sotirios A. Tsaftaris, Timothy Hospedales. (2023)  
**FairTune: Optimizing Parameter Efficient Fine Tuning for Fairness in Medical Image Analysis**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.05055v1)  

---


**ABSTRACT**  
Training models with robust group fairness properties is crucial in ethically sensitive application areas such as medical diagnosis. Despite the growing body of work aiming to minimise demographic bias in AI, this problem remains challenging. A key reason for this challenge is the fairness generalisation gap: High-capacity deep learning models can fit all training data nearly perfectly, and thus also exhibit perfect fairness during training. In this case, bias emerges only during testing when generalisation performance differs across subgroups. This motivates us to take a bi-level optimisation perspective on fair learning: Optimising the learning strategy based on validation fairness. Specifically, we consider the highly effective workflow of adapting pre-trained models to downstream medical imaging tasks using parameter-efficient fine-tuning (PEFT) techniques. There is a trade-off between updating more parameters, enabling a better fit to the task of interest vs. fewer parameters, potentially reducing the generalisation gap. To manage this tradeoff, we propose FairTune, a framework to optimise the choice of PEFT parameters with respect to fairness. We demonstrate empirically that FairTune leads to improved fairness on a range of medical imaging datasets.

{{</citation>}}


### (81/112) Low-Resolution Self-Attention for Semantic Segmentation (Yu-Huan Wu et al., 2023)

{{<citation>}}

Yu-Huan Wu, Shi-Chen Zhang, Yun Liu, Le Zhang, Xin Zhan, Daquan Zhou, Jiashi Feng, Ming-Ming Cheng, Liangli Zhen. (2023)  
**Low-Resolution Self-Attention for Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Self-Attention, Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2310.05026v1)  

---


**ABSTRACT**  
Semantic segmentation tasks naturally require high-resolution information for pixel-wise segmentation and global context information for class prediction. While existing vision transformers demonstrate promising performance, they often utilize high resolution context modeling, resulting in a computational bottleneck. In this work, we challenge conventional wisdom and introduce the Low-Resolution Self-Attention (LRSA) mechanism to capture global context at a significantly reduced computational cost. Our approach involves computing self-attention in a fixed low-resolution space regardless of the input image's resolution, with additional 3x3 depth-wise convolutions to capture fine details in the high-resolution space. We demonstrate the effectiveness of our LRSA approach by building the LRFormer, a vision transformer with an encoder-decoder structure. Extensive experiments on the ADE20K, COCO-Stuff, and Cityscapes datasets demonstrate that LRFormer outperforms state-of-the-art models. The code will be made available at https://github.com/yuhuan-wu/LRFormer.

{{</citation>}}


### (82/112) Single Stage Warped Cloth Learning and Semantic-Contextual Attention Feature Fusion for Virtual TryOn (Sanhita Pathak et al., 2023)

{{<citation>}}

Sanhita Pathak, Vinay Kaushik, Brejesh Lall. (2023)  
**Single Stage Warped Cloth Learning and Semantic-Contextual Attention Feature Fusion for Virtual TryOn**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2310.05024v1)  

---


**ABSTRACT**  
Image-based virtual try-on aims to fit an in-shop garment onto a clothed person image. Garment warping, which aligns the target garment with the corresponding body parts in the person image, is a crucial step in achieving this goal. Existing methods often use multi-stage frameworks to handle clothes warping, person body synthesis and tryon generation separately or rely on noisy intermediate parser-based labels. We propose a novel single-stage framework that implicitly learns the same without explicit multi-stage learning. Our approach utilizes a novel semantic-contextual fusion attention module for garment-person feature fusion, enabling efficient and realistic cloth warping and body synthesis from target pose keypoints. By introducing a lightweight linear attention framework that attends to garment regions and fuses multiple sampled flow fields, we also address misalignment and artifacts present in previous methods. To achieve simultaneous learning of warped garment and try-on results, we introduce a Warped Cloth Learning Module. WCLM uses segmented warped garments as ground truth, operating within a single-stage paradigm. Our proposed approach significantly improves the quality and efficiency of virtual try-on methods, providing users with a more reliable and realistic virtual try-on experience. We evaluate our method on the VITON dataset and demonstrate its state-of-the-art performance in terms of both qualitative and quantitative metrics.

{{</citation>}}


### (83/112) Detecting Abnormal Health Conditions in Smart Home Using a Drone (Pronob Kumar Barman, 2023)

{{<citation>}}

Pronob Kumar Barman. (2023)  
**Detecting Abnormal Health Conditions in Smart Home Using a Drone**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2310.05012v1)  

---


**ABSTRACT**  
Nowadays, detecting aberrant health issues is a difficult process. Falling, especially among the elderly, is a severe concern worldwide. Falls can result in deadly consequences, including unconsciousness, internal bleeding, and often times, death. A practical and optimal, smart approach of detecting falling is currently a concern. The use of vision-based fall monitoring is becoming more common among scientists as it enables senior citizens and those with other health conditions to live independently. For tracking, surveillance, and rescue, unmanned aerial vehicles use video or image segmentation and object detection methods. The Tello drone is equipped with a camera and with this device we determined normal and abnormal behaviors among our participants. The autonomous falling objects are classified using a convolutional neural network (CNN) classifier. The results demonstrate that the systems can identify falling objects with a precision of 0.9948.

{{</citation>}}


### (84/112) Symmetrical Linguistic Feature Distillation with CLIP for Scene Text Recognition (Zixiao Wang et al., 2023)

{{<citation>}}

Zixiao Wang, Hongtao Xie, Yuxin Wang, Jianjun Xu, Boqiang Zhang. (2023)  
**Symmetrical Linguistic Feature Distillation with CLIP for Scene Text Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: OCR  
[Paper Link](http://arxiv.org/abs/2310.04999v1)  

---


**ABSTRACT**  
In this paper, we explore the potential of the Contrastive Language-Image Pretraining (CLIP) model in scene text recognition (STR), and establish a novel Symmetrical Linguistic Feature Distillation framework (named CLIP-OCR) to leverage both visual and linguistic knowledge in CLIP. Different from previous CLIP-based methods mainly considering feature generalization on visual encoding, we propose a symmetrical distillation strategy (SDS) that further captures the linguistic knowledge in the CLIP text encoder. By cascading the CLIP image encoder with the reversed CLIP text encoder, a symmetrical structure is built with an image-to-text feature flow that covers not only visual but also linguistic information for distillation.Benefiting from the natural alignment in CLIP, such guidance flow provides a progressive optimization objective from vision to language, which can supervise the STR feature forwarding process layer-by-layer.Besides, a new Linguistic Consistency Loss (LCL) is proposed to enhance the linguistic capability by considering second-order statistics during the optimization. Overall, CLIP-OCR is the first to design a smooth transition between image and text for the STR task.Extensive experiments demonstrate the effectiveness of CLIP-OCR with 93.8% average accuracy on six popular STR benchmarks.Code will be available at https://github.com/wzx99/CLIPOCR.

{{</citation>}}


### (85/112) Compositional Semantics for Open Vocabulary Spatio-semantic Representations (Robin Karlsson et al., 2023)

{{<citation>}}

Robin Karlsson, Francisco Lepe-Salazar, Kazuya Takeda. (2023)  
**Compositional Semantics for Open Vocabulary Spatio-semantic Representations**  

---
Primary Category: cs.CV  
Categories: I-2-10; I-2-9, cs-CV, cs-LG, cs.CV  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2310.04981v1)  

---


**ABSTRACT**  
General-purpose mobile robots need to complete tasks without exact human instructions. Large language models (LLMs) is a promising direction for realizing commonsense world knowledge and reasoning-based planning. Vision-language models (VLMs) transform environment percepts into vision-language semantics interpretable by LLMs. However, completing complex tasks often requires reasoning about information beyond what is currently perceived. We propose latent compositional semantic embeddings z* as a principled learning-based knowledge representation for queryable spatio-semantic memories. We mathematically prove that z* can always be found, and the optimal z* is the centroid for any set Z. We derive a probabilistic bound for estimating separability of related and unrelated semantics. We prove that z* is discoverable by iterative optimization by gradient descent from visual appearance and singular descriptions. We experimentally verify our findings on four embedding spaces incl. CLIP and SBERT. Our results show that z* can represent up to 10 semantics encoded by SBERT, and up to 100 semantics for ideal uniformly distributed high-dimensional embeddings. We demonstrate that a simple dense VLM trained on the COCO-Stuff dataset can learn z* for 181 overlapping semantics by 42.23 mIoU, while improving conventional non-overlapping open-vocabulary segmentation performance by +3.48 mIoU compared with a popular SOTA model.

{{</citation>}}


## cs.AI (14)



### (86/112) A Knowledge Graph-Based Search Engine for Robustly Finding Doctors and Locations in the Healthcare Domain (Mayank Kejriwal et al., 2023)

{{<citation>}}

Mayank Kejriwal, Hamid Haidarian, Min-Hsueh Chiu, Andy Xiang, Deep Shrestha, Faizan Javed. (2023)  
**A Knowledge Graph-Based Search Engine for Robustly Finding Doctors and Locations in the Healthcare Domain**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-DB, cs-IR, cs.AI  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2310.05258v1)  

---


**ABSTRACT**  
Efficiently finding doctors and locations is an important search problem for patients in the healthcare domain, for which traditional information retrieval methods tend not to work optimally. In the last ten years, knowledge graphs (KGs) have emerged as a powerful way to combine the benefits of gleaning insights from semi-structured data using semantic modeling, natural language processing techniques like information extraction, and robust querying using structured query languages like SPARQL and Cypher. In this short paper, we present a KG-based search engine architecture for robustly finding doctors and locations in the healthcare domain. Early results demonstrate that our approach can lead to significantly higher coverage for complex queries without degrading quality.

{{</citation>}}


### (87/112) Evolutionary Retrosynthetic Route Planning (Yan Zhang et al., 2023)

{{<citation>}}

Yan Zhang, Hao Hao, Xiao He, Shuanhu Gao, Aimin Zhou. (2023)  
**Evolutionary Retrosynthetic Route Planning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.05186v1)  

---


**ABSTRACT**  
Molecular retrosynthesis is a significant and complex problem in the field of chemistry, however, traditional manual synthesis methods not only need well-trained experts but also are time-consuming. With the development of big data and machine learning, artificial intelligence (AI) based retrosynthesis is attracting more attention and is becoming a valuable tool for molecular retrosynthesis. At present, Monte Carlo tree search is a mainstream search framework employed to address this problem. Nevertheless, its search efficiency is compromised by its large search space. Therefore, we propose a novel approach for retrosynthetic route planning based on evolutionary optimization, marking the first use of Evolutionary Algorithm (EA) in the field of multi-step retrosynthesis. The proposed method involves modeling the retrosynthetic problem into an optimization problem, defining the search space and operators. Additionally, to improve the search efficiency, a parallel strategy is implemented. The new approach is applied to four case products, and is compared with Monte Carlo tree search. The experimental results show that, in comparison to the Monte Carlo tree search algorithm, EA significantly reduces the number of calling single-step model by an average of 53.9%. The time required to search three solutions decreased by an average of 83.9%, and the number of feasible search routes increases by 5 times.

{{</citation>}}


### (88/112) Text2NKG: Fine-Grained N-ary Relation Extraction for N-ary relational Knowledge Graph Construction (Haoran Luo et al., 2023)

{{<citation>}}

Haoran Luo, Haihong E, Yuhao Yang, Tianyu Yao, Yikai Guo, Zichen Tang, Wentai Zhang, Kaiyang Wan, Shiyao Peng, Meina Song, Wei Lin. (2023)  
**Text2NKG: Fine-Grained N-ary Relation Extraction for N-ary relational Knowledge Graph Construction**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: Knowledge Graph, Relation Extraction  
[Paper Link](http://arxiv.org/abs/2310.05185v1)  

---


**ABSTRACT**  
Beyond traditional binary relational facts, n-ary relational knowledge graphs (NKGs) are comprised of n-ary relational facts containing more than two entities, which are closer to real-world facts with broader applications. However, the construction of NKGs still significantly relies on manual labor, and n-ary relation extraction still remains at a course-grained level, which is always in a single schema and fixed arity of entities. To address these restrictions, we propose Text2NKG, a novel fine-grained n-ary relation extraction framework for n-ary relational knowledge graph construction. We introduce a span-tuple classification approach with hetero-ordered merging to accomplish fine-grained n-ary relation extraction in different arity. Furthermore, Text2NKG supports four typical NKG schemas: hyper-relational schema, event-based schema, role-based schema, and hypergraph-based schema, with high flexibility and practicality. Experimental results demonstrate that Text2NKG outperforms the previous state-of-the-art model by nearly 20\% points in the $F_1$ scores on the fine-grained n-ary relation extraction benchmark in the hyper-relational schema. Our code and datasets are publicly available.

{{</citation>}}


### (89/112) Hieros: Hierarchical Imagination on Structured State Space Sequence World Models (Paul Mattes et al., 2023)

{{<citation>}}

Paul Mattes, Rainer Schlosser, Ralf Herbrich. (2023)  
**Hieros: Hierarchical Imagination on Structured State Space Sequence World Models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.05167v1)  

---


**ABSTRACT**  
One of the biggest challenges to modern deep reinforcement learning (DRL) algorithms is sample efficiency. Many approaches learn a world model in order to train an agent entirely in imagination, eliminating the need for direct environment interaction during training. However, these methods often suffer from either a lack of imagination accuracy, exploration capabilities, or runtime efficiency. We propose \hieros, a hierarchical policy that learns time abstracted world representations and imagines trajectories at multiple time scales in latent space. \hieros uses an S5 layer-based world model, which predicts next world states in parallel during training and iteratively during environment interaction. Due to the special properties of S5 layers, our method can train in parallel and predict next world states iteratively during imagination. This allows for more efficient training than RNN-based world models and more efficient imagination than Transformer-based world models.   We show that our approach outperforms the state of the art in terms of mean and median normalized human score on the Atari 100k benchmark, and that our proposed world model is able to predict complex dynamics very accurately. We also show that \hieros displays superior exploration capabilities compared to existing approaches.

{{</citation>}}


### (90/112) Large Language Model (LLM) as a System of Multiple Expert Agents: An Approach to solve the Abstraction and Reasoning Corpus (ARC) Challenge (John Chong Min Tan et al., 2023)

{{<citation>}}

John Chong Min Tan, Mehul Motani. (2023)  
**Large Language Model (LLM) as a System of Multiple Expert Agents: An Approach to solve the Abstraction and Reasoning Corpus (ARC) Challenge**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.05146v1)  

---


**ABSTRACT**  
We attempt to solve the Abstraction and Reasoning Corpus (ARC) Challenge using Large Language Models (LLMs) as a system of multiple expert agents. Using the flexibility of LLMs to be prompted to do various novel tasks using zero-shot, few-shot, context-grounded prompting, we explore the feasibility of using LLMs to solve the ARC Challenge. We firstly convert the input image into multiple suitable text-based abstraction spaces. We then utilise the associative power of LLMs to derive the input-output relationship and map this to actions in the form of a working program, similar to Voyager / Ghost in the MineCraft. In addition, we use iterative environmental feedback in order to guide LLMs to solve the task. Our proposed approach achieves 50 solves out of 111 training set problems (45%) with just three abstraction spaces - grid, object and pixel - and we believe that with more abstraction spaces and learnable actions, we will be able to solve more.

{{</citation>}}


### (91/112) InstructDET: Diversifying Referring Object Detection with Generalized Instructions (Ronghao Dang et al., 2023)

{{<citation>}}

Ronghao Dang, Jiangyan Feng, Haodong Zhang, Chongjian Ge, Lin Song, Lijun Gong, Chengju Liu, Qijun Chen, Feng Zhu, Rui Zhao, Yibing Song. (2023)  
**InstructDET: Diversifying Referring Object Detection with Generalized Instructions**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CV, cs.AI  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2310.05136v1)  

---


**ABSTRACT**  
We propose InstructDET, a data-centric method for referring object detection (ROD) that localizes target objects based on user instructions. While deriving from referring expressions (REC), the instructions we leverage are greatly diversified to encompass common user intentions related to object detection. For one image, we produce tremendous instructions that refer to every single object and different combinations of multiple objects. Each instruction and its corresponding object bounding boxes (bbxs) constitute one training data pair. In order to encompass common detection expressions, we involve emerging vision-language model (VLM) and large language model (LLM) to generate instructions guided by text prompts and object bbxs, as the generalizations of foundation models are effective to produce human-like expressions (e.g., describing object property, category, and relationship). We name our constructed dataset as InDET. It contains images, bbxs and generalized instructions that are from foundation models. Our InDET is developed from existing REC datasets and object detection datasets, with the expanding potential that any image with object bbxs can be incorporated through using our InstructDET method. By using our InDET dataset, we show that a conventional ROD model surpasses existing methods on standard REC datasets and our InDET test set. Our data-centric method InstructDET, with automatic data expansion by leveraging foundation models, directs a promising field that ROD can be greatly diversified to execute common object detection instructions.

{{</citation>}}


### (92/112) Intelligent DRL-Based Adaptive Region of Interest for Delay-sensitive Telemedicine Applications (Abdulrahman Soliman et al., 2023)

{{<citation>}}

Abdulrahman Soliman, Amr Mohamed, Elias Yaacoub, Nikhil V. Navkar, Aiman Erbad. (2023)  
**Intelligent DRL-Based Adaptive Region of Interest for Delay-sensitive Telemedicine Applications**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-MM, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.05099v1)  

---


**ABSTRACT**  
Telemedicine applications have recently received substantial potential and interest, especially after the COVID-19 pandemic. Remote experience will help people get their complex surgery done or transfer knowledge to local surgeons, without the need to travel abroad. Even with breakthrough improvements in internet speeds, the delay in video streaming is still a hurdle in telemedicine applications. This imposes using image compression and region of interest (ROI) techniques to reduce the data size and transmission needs. This paper proposes a Deep Reinforcement Learning (DRL) model that intelligently adapts the ROI size and non-ROI quality depending on the estimated throughput. The delay and structural similarity index measure (SSIM) comparison are used to assess the DRL model. The comparison findings and the practical application reveal that DRL is capable of reducing the delay by 13% and keeping the overall quality in an acceptable range. Since the latency has been significantly reduced, these findings are a valuable enhancement to telemedicine applications.

{{</citation>}}


### (93/112) Learning Generalizable Agents via Saliency-Guided Features Decorrelation (Sili Huang et al., 2023)

{{<citation>}}

Sili Huang, Yanchao Sun, Jifeng Hu, Siyuan Guo, Hechang Chen, Yi Chang, Lichao Sun, Bo Yang. (2023)  
**Learning Generalizable Agents via Saliency-Guided Features Decorrelation**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.05086v1)  

---


**ABSTRACT**  
In visual-based Reinforcement Learning (RL), agents often struggle to generalize well to environmental variations in the state space that were not observed during training. The variations can arise in both task-irrelevant features, such as background noise, and task-relevant features, such as robot configurations, that are related to the optimal decisions. To achieve generalization in both situations, agents are required to accurately understand the impact of changed features on the decisions, i.e., establishing the true associations between changed features and decisions in the policy model. However, due to the inherent correlations among features in the state space, the associations between features and decisions become entangled, making it difficult for the policy to distinguish them. To this end, we propose Saliency-Guided Features Decorrelation (SGFD) to eliminate these correlations through sample reweighting. Concretely, SGFD consists of two core techniques: Random Fourier Functions (RFF) and the saliency map. RFF is utilized to estimate the complex non-linear correlations in high-dimensional images, while the saliency map is designed to identify the changed features. Under the guidance of the saliency map, SGFD employs sample reweighting to minimize the estimated correlations related to changed features, thereby achieving decorrelation in visual RL tasks. Our experimental results demonstrate that SGFD can generalize well on a wide range of test environments and significantly outperforms state-of-the-art methods in handling both task-irrelevant variations and task-relevant variations.

{{</citation>}}


### (94/112) From Text to Tactic: Evaluating LLMs Playing the Game of Avalon (Jonathan Light et al., 2023)

{{<citation>}}

Jonathan Light, Min Cai, Sheng Shen, Ziniu Hu. (2023)  
**From Text to Tactic: Evaluating LLMs Playing the Game of Avalon**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.05036v1)  

---


**ABSTRACT**  
In this paper, we explore the potential of Large Language Models (LLMs) Agents in playing the strategic social deduction game, Resistance Avalon. Players in Avalon are challenged not only to make informed decisions based on dynamically evolving game phases, but also to engage in discussions where they must deceive, deduce, and negotiate with other players. These characteristics make Avalon a compelling test-bed to study the decision-making and language-processing capabilities of LLM Agents. To facilitate research in this line, we introduce AvalonBench - a comprehensive game environment tailored for evaluating multi-agent LLM Agents. This benchmark incorporates: (1) a game environment for Avalon, (2) rule-based bots as baseline opponents, and (3) ReAct-style LLM agents with tailored prompts for each role. Notably, our evaluations based on AvalonBench highlight a clear capability gap. For instance, models like ChatGPT playing good-role got a win rate of 22.2% against rule-based bots playing evil, while good-role bot achieves 38.2% win rate in the same setting. We envision AvalonBench could be a good test-bed for developing more advanced LLMs (with self-playing) and agent frameworks that can effectively model the layered complexities of such game environments.

{{</citation>}}


### (95/112) Revisiting Large Language Models as Zero-shot Relation Extractors (Guozheng Li et al., 2023)

{{<citation>}}

Guozheng Li, Peng Wang, Wenjun Ke. (2023)  
**Revisiting Large Language Models as Zero-shot Relation Extractors**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: ChatGPT, GPT, Language Model, QA  
[Paper Link](http://arxiv.org/abs/2310.05028v1)  

---


**ABSTRACT**  
Relation extraction (RE) consistently involves a certain degree of labeled or unlabeled data even if under zero-shot setting. Recent studies have shown that large language models (LLMs) transfer well to new tasks out-of-the-box simply given a natural language prompt, which provides the possibility of extracting relations from text without any data and parameter tuning. This work focuses on the study of exploring LLMs, such as ChatGPT, as zero-shot relation extractors. On the one hand, we analyze the drawbacks of existing RE prompts and attempt to incorporate recent prompt techniques such as chain-of-thought (CoT) to improve zero-shot RE. We propose the summarize-and-ask (\textsc{SumAsk}) prompting, a simple prompt recursively using LLMs to transform RE inputs to the effective question answering (QA) format. On the other hand, we conduct comprehensive experiments on various benchmarks and settings to investigate the capabilities of LLMs on zero-shot RE. Specifically, we have the following findings: (i) \textsc{SumAsk} consistently and significantly improves LLMs performance on different model sizes, benchmarks and settings; (ii) Zero-shot prompting with ChatGPT achieves competitive or superior results compared with zero-shot and fully supervised methods; (iii) LLMs deliver promising performance in extracting overlapping relations; (iv) The performance varies greatly regarding different relations. Different from small language models, LLMs are effective in handling challenge none-of-the-above (NoTA) relation.

{{</citation>}}


### (96/112) Compresso: Structured Pruning with Collaborative Prompting Learns Compact Large Language Models (Song Guo et al., 2023)

{{<citation>}}

Song Guo, Jiahang Xu, Li Lyna Zhang, Mao Yang. (2023)  
**Compresso: Structured Pruning with Collaborative Prompting Learns Compact Large Language Models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: LLaMA, Language Model, Pruning  
[Paper Link](http://arxiv.org/abs/2310.05015v1)  

---


**ABSTRACT**  
Despite the remarkable success of Large Language Models (LLMs), the massive size poses significant deployment challenges, particularly on resource-constrained hardware. While existing LLM compression methods focus on quantization, pruning remains relatively unexplored due to the high cost of training-based approaches and data collection challenges. One-shot pruning methods, although cost-effective and data-free, have become dominant in LLM pruning, but lead to performance decline under the structured pruning setting. In this work, we introduce a new paradigm for structurally pruning LLMs, called Compresso. Our approach, through the collaboration of the proposed resource-efficient pruning algorithm and the LLM itself, learns optimal pruning decisions during the training process. Compresso addresses the challenges of expensive training costs and data collection by incorporating Low-Rank Adaptation (LoRA) into the $L_0$ regularization during the instruction tuning process. Then, we further augment the pruning algorithm by introducing a collaborative prompt that fosters collaboration between the LLM and the pruning algorithm, significantly boosting the overall performance. To this end, Compresso prunes LLaMA-7B to 5.4B, maintaining original performance and even surpassing LLaMA-7B in reading comprehension by 2.62%. Extensive experiments demonstrate that Compresso significantly outperforms one-shot pruning baselines across various sparsity ratios, achieving up to 2.21%, 11.43%, 7.04%, and 4.81% higher scores on the commonsense reasoning, reading comprehension, MMLU, and BBH benchmarks, respectively.

{{</citation>}}


### (97/112) The Troubling Emergence of Hallucination in Large Language Models -- An Extensive Definition, Quantification, and Prescriptive Remediations (Vipula Rawte et al., 2023)

{{<citation>}}

Vipula Rawte, Swagata Chakraborty, Agnibh Pathak, Anubhav Sarkar, S. M Towhidul Islam Tonmoy, Aman Chadha, Amit P. Sheth, Amitava Das. (2023)  
**The Troubling Emergence of Hallucination in Large Language Models -- An Extensive Definition, Quantification, and Prescriptive Remediations**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2310.04988v1)  

---


**ABSTRACT**  
The recent advancements in Large Language Models (LLMs) have garnered widespread acclaim for their remarkable emerging capabilities. However, the issue of hallucination has parallelly emerged as a by-product, posing significant concerns. While some recent endeavors have been made to identify and mitigate different types of hallucination, there has been a limited emphasis on the nuanced categorization of hallucination and associated mitigation methods. To address this gap, we offer a fine-grained discourse on profiling hallucination based on its degree, orientation, and category, along with offering strategies for alleviation. As such, we define two overarching orientations of hallucination: (i) factual mirage (FM) and (ii) silver lining (SL). To provide a more comprehensive understanding, both orientations are further sub-categorized into intrinsic and extrinsic, with three degrees of severity - (i) mild, (ii) moderate, and (iii) alarming. We also meticulously categorize hallucination into six types: (i) acronym ambiguity, (ii) numeric nuisance, (iii) generated golem, (iv) virtual voice, (v) geographic erratum, and (vi) time wrap. Furthermore, we curate HallucInation eLiciTation (HILT), a publicly available dataset comprising of 75,000 samples generated using 15 contemporary LLMs along with human annotations for the aforementioned categories. Finally, to establish a method for quantifying and to offer a comparative spectrum that allows us to evaluate and rank LLMs based on their vulnerability to producing hallucinations, we propose Hallucination Vulnerability Index (HVI). We firmly believe that HVI holds significant value as a tool for the wider NLP community, with the potential to serve as a rubric in AI-related policy-making. In conclusion, we propose two solution strategies for mitigating hallucinations.

{{</citation>}}


### (98/112) LLM4VV: Developing LLM-Driven Testsuite for Compiler Validation (Christian Munley et al., 2023)

{{<citation>}}

Christian Munley, Aaron Jarmusch, Sunita Chandrasekaran. (2023)  
**LLM4VV: Developing LLM-Driven Testsuite for Compiler Validation**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2310.04963v1)  

---


**ABSTRACT**  
Large language models (LLMs) are a new and powerful tool for a wide span of applications involving natural language and demonstrate impressive code generation abilities. In this paper, we explore the capabilitity of state-of-the-art LLMs, including closed-source options like OpenAI GPT-4 and open-source alternatives like Meta AI Codellama, to automatically generate tests and use these tests to validate and verify compiler implementations of a directive-based programming paradigm, OpenACC. Our approach entails exploring various prompt engineering techniques including a code template, retrieval-augmented generation (RAG) with code template, expressive prompt using RAG with code template, one-shot example, and RAG with one-shot example. This paper focusses on (a) exploring the capabilities of the latest LLMs for code generation, (b) investigating prompt and fine tuning methods, and (c) analyzing the outcome of LLMs generated tests

{{</citation>}}


### (99/112) CodeTransOcean: A Comprehensive Multilingual Benchmark for Code Translation (Weixiang Yan et al., 2023)

{{<citation>}}

Weixiang Yan, Yuchen Tian, Yunzhe Li, Qian Chen, Wen Wang. (2023)  
**CodeTransOcean: A Comprehensive Multilingual Benchmark for Code Translation**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-PL, cs.AI  
Keywords: ChatGPT, GPT, Multilingual  
[Paper Link](http://arxiv.org/abs/2310.04951v1)  

---


**ABSTRACT**  
Recent code translation techniques exploit neural machine translation models to translate source code from one programming language to another to satisfy production compatibility or to improve efficiency of codebase maintenance. Most existing code translation datasets only focus on a single pair of popular programming languages. To advance research on code translation and meet diverse requirements of real-world applications, we construct CodeTransOcean, a large-scale comprehensive benchmark that supports the largest variety of languages for code translation. CodeTransOcean consists of three novel multilingual datasets, namely, MultilingualTrans supporting translations between multiple popular programming languages, NicheTrans for translating between niche programming languages and popular ones, and LLMTrans for evaluating compilability of translated code by large language models (LLMs). CodeTransOcean also includes a novel cross-framework dataset, DLTrans, for translating deep learning code across different frameworks. We develop multilingual modeling approaches for code translation and demonstrate their great potential in improving the translation quality of both low-resource and high-resource language pairs and boosting the training efficiency. We also propose a novel evaluation metric Debugging Success Rate@K for program-level code translation. Last but not least, we evaluate LLM ChatGPT on our datasets and investigate its potential for fuzzy compilation predictions. We build baselines for CodeTransOcean and analyze challenges of code translation for guiding future research.

{{</citation>}}


## cs.RO (3)



### (100/112) Influence of Camera-LiDAR Configuration on 3D Object Detection for Autonomous Driving (Ye Li et al., 2023)

{{<citation>}}

Ye Li, Hanjiang Hu, Zuxin Liu, Ding Zhao. (2023)  
**Influence of Camera-LiDAR Configuration on 3D Object Detection for Autonomous Driving**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2310.05245v1)  

---


**ABSTRACT**  
Cameras and LiDARs are both important sensors for autonomous driving, playing critical roles for 3D object detection. Camera-LiDAR Fusion has been a prevalent solution for robust and accurate autonomous driving perception. In contrast to the vast majority of existing arts that focus on how to improve the performance of 3D target detection through cross-modal schemes, deep learning algorithms, and training tricks, we devote attention to the impact of sensor configurations on the performance of learning-based methods. To achieve this, we propose a unified information-theoretic surrogate metric for camera and LiDAR evaluation based on the proposed sensor perception model. We also design an accelerated high-quality framework for data acquisition, model training, and performance evaluation that functions with the CARLA simulator. To show the correlation between detection performance and our surrogate metrics, We conduct experiments using several camera-LiDAR placements and parameters inspired by self-driving companies and research institutions. Extensive experimental results of representative algorithms on NuScenes dataset validate the effectiveness of our surrogate metric, demonstrating that sensor configurations significantly impact point-cloud-image fusion based detection models, which contribute up to 30% discrepancy in terms of average precision.

{{</citation>}}


### (101/112) LAN-grasp: Using Large Language Models for Semantic Object Grasping (Reihaneh Mirjalili et al., 2023)

{{<citation>}}

Reihaneh Mirjalili, Michael Krawez, Simone Silenzi, Yannik Blei, Wolfram Burgard. (2023)  
**LAN-grasp: Using Large Language Models for Semantic Object Grasping**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.05239v1)  

---


**ABSTRACT**  
In this paper, we propose LAN-grasp, a novel approach towards more appropriate semantic grasping. We use foundation models to provide the robot with a deeper understanding of the objects, the right place to grasp an object, or even the parts to avoid. This allows our robot to grasp and utilize objects in a more meaningful and safe manner. We leverage the combination of a Large Language Model, a Vision Language Model, and a traditional grasp planner to generate grasps demonstrating a deeper semantic understanding of the objects. We first prompt the Large Language Model about which object part is appropriate for grasping. Next, the Vision Language Model identifies the corresponding part in the object image. Finally, we generate grasp proposals in the region proposed by the Vision Language Model. Building on foundation models provides us with a zero-shot grasp method that can handle a wide range of objects without the need for further training or fine-tuning. We evaluated our method in real-world experiments on a custom object data set. We present the results of a survey that asks the participants to choose an object part appropriate for grasping. The results show that the grasps generated by our method are consistently ranked higher by the participants than those generated by a conventional grasping planner and a recent semantic grasping approach.

{{</citation>}}


### (102/112) Initial Task Assignment in Multi-Human Multi-Robot Teams: An Attention-enhanced Hierarchical Reinforcement Learning Approach (Ruiqi Wang et al., 2023)

{{<citation>}}

Ruiqi Wang, Dezhong Zhao, Arjun Gupte, Byung-Cheol Min. (2023)  
**Initial Task Assignment in Multi-Human Multi-Robot Teams: An Attention-enhanced Hierarchical Reinforcement Learning Approach**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Attention, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.04979v1)  

---


**ABSTRACT**  
Multi-human multi-robot teams (MH-MR) obtain tremendous potential in tackling intricate and massive missions by merging distinct strengths and expertise of individual members. The inherent heterogeneity of these teams necessitates advanced initial task assignment (ITA) methods that align tasks with the intrinsic capabilities of team members from the outset. While existing reinforcement learning approaches show encouraging results, they might fall short in addressing the nuances of long-horizon ITA problems, particularly in settings with large-scale MH-MR teams or multifaceted tasks. To bridge this gap, we propose an attention-enhanced hierarchical reinforcement learning approach that decomposes the complex ITA problem into structured sub-problems, facilitating more efficient allocations. To bolster sub-policy learning, we introduce a hierarchical cross-attribute attention (HCA) mechanism, encouraging each sub-policy within the hierarchy to discern and leverage the specific nuances in the state space that are crucial for its respective decision-making phase. Through an extensive environmental surveillance case study, we demonstrate the benefits of our model and the HCA inside.

{{</citation>}}


## math.OC (1)



### (103/112) Global Convergence of Policy Gradient Methods in Reinforcement Learning, Games and Control (Shicong Cen et al., 2023)

{{<citation>}}

Shicong Cen, Yuejie Chi. (2023)  
**Global Convergence of Policy Gradient Methods in Reinforcement Learning, Games and Control**  

---
Primary Category: math.OC  
Categories: cs-GT, cs-IT, cs-LG, math-IT, math-OC, math.OC  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.05230v1)  

---


**ABSTRACT**  
Policy gradient methods, where one searches for the policy of interest by maximizing the value functions using first-order information, become increasingly popular for sequential decision making in reinforcement learning, games, and control. Guaranteeing the global optimality of policy gradient methods, however, is highly nontrivial due to nonconcavity of the value functions. In this exposition, we highlight recent progresses in understanding and developing policy gradient methods with global convergence guarantees, putting an emphasis on their finite-time convergence rates with regard to salient problem parameters.

{{</citation>}}


## cs.SE (2)



### (104/112) Optimizing Large Language Models to Expedite the Development of Smart Contracts (Nii Osae Osae Dade et al., 2023)

{{<citation>}}

Nii Osae Osae Dade, Margaret Lartey-Quaye, Emmanuel Teye-Kofi Odonkor, Paul Ammah. (2023)  
**Optimizing Large Language Models to Expedite the Development of Smart Contracts**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-CL, cs-SE, cs.SE  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.05178v1)  

---


**ABSTRACT**  
Programming has always been at the heart of technological innovation in the 21st century. With the advent of blockchain technologies and the proliferation of web3 paradigms of decentralised applications, smart contracts have been very instrumental in enabling developers to build applications that reside on decentralised blockchains. Despite the huge interest and potential of smart contracts, there is still a significant knowledge and skill gap that developers need to cross in order to build web3 applications. In light of this, we introduce MazzumaGPT, a large language model that has been optimised to generate smart contract code and aid developers to scaffold development and improve productivity. As part of this research, we outline the optimisation and fine-tuning parameters, evaluate the model's performance on functional correctness and address the limitations and broader impacts of our research.

{{</citation>}}


### (105/112) DeepQTest: Testing Autonomous Driving Systems with Reinforcement Learning and Real-world Weather Data (Chengjie Lu et al., 2023)

{{<citation>}}

Chengjie Lu, Tao Yue, Man Zhang, Shaukat Ali. (2023)  
**DeepQTest: Testing Autonomous Driving Systems with Reinforcement Learning and Real-world Weather Data**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-LG, cs-RO, cs-SE, cs.SE  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.05170v1)  

---


**ABSTRACT**  
Autonomous driving systems (ADSs) are capable of sensing the environment and making driving decisions autonomously. These systems are safety-critical, and testing them is one of the important approaches to ensure their safety. However, due to the inherent complexity of ADSs and the high dimensionality of their operating environment, the number of possible test scenarios for ADSs is infinite. Besides, the operating environment of ADSs is dynamic, continuously evolving, and full of uncertainties, which requires a testing approach adaptive to the environment. In addition, existing ADS testing techniques have limited effectiveness in ensuring the realism of test scenarios, especially the realism of weather conditions and their changes over time. Recently, reinforcement learning (RL) has demonstrated great potential in addressing challenging problems, especially those requiring constant adaptations to dynamic environments. To this end, we present DeepQTest, a novel ADS testing approach that uses RL to learn environment configurations with a high chance of revealing abnormal ADS behaviors. Specifically, DeepQTest employs Deep Q-Learning and adopts three safety and comfort measures to construct the reward functions. To ensure the realism of generated scenarios, DeepQTest defines a set of realistic constraints and introduces real-world weather conditions into the simulated environment. We employed three comparison baselines, i.e., random, greedy, and a state-of-the-art RL-based approach DeepCOllision, for evaluating DeepQTest on an industrial-scale ADS. Evaluation results show that DeepQTest demonstrated significantly better effectiveness in terms of generating scenarios leading to collisions and ensuring scenario realism compared with the baselines. In addition, among the three reward functions implemented in DeepQTest, Time-To-Collision is recommended as the best design according to our study.

{{</citation>}}


## cs.SD (2)



### (106/112) VITS-based Singing Voice Conversion System with DSPGAN post-processing for SVCC2023 (Yiquan Zhou et al., 2023)

{{<citation>}}

Yiquan Zhou, Meng Chen, Yi Lei, Jihua Zhu, Weifeng Zhao. (2023)  
**VITS-based Singing Voice Conversion System with DSPGAN post-processing for SVCC2023**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2310.05118v1)  

---


**ABSTRACT**  
This paper presents the T02 team's system for the Singing Voice Conversion Challenge 2023 (SVCC2023). Our system entails a VITS-based SVC model, incorporating three modules: a feature extractor, a voice converter, and a post-processor. Specifically, the feature extractor provides F0 contours and extracts speaker-independent linguistic content from the input singing voice by leveraging a HuBERT model. The voice converter is employed to recompose the speaker timbre, F0, and linguistic content to generate the waveform of the target speaker. Besides, to further improve the audio quality, a fine-tuned DSPGAN vocoder is introduced to re-synthesise the waveform. Given the limited target speaker data, we utilize a two-stage training strategy to adapt the base model to the target speaker. During model adaptation, several tricks, such as data augmentation and joint training with auxiliary singer data, are involved. Official challenge results show that our system achieves superior performance, especially in the cross-domain task, ranking 1st and 2nd in naturalness and similarity, respectively. Further ablation justifies the effectiveness of our system design.

{{</citation>}}


### (107/112) Comparative Analysis of Transfer Learning in Deep Learning Text-to-Speech Models on a Few-Shot, Low-Resource, Customized Dataset (Ze Liu, 2023)

{{<citation>}}

Ze Liu. (2023)  
**Comparative Analysis of Transfer Learning in Deep Learning Text-to-Speech Models on a Few-Shot, Low-Resource, Customized Dataset**  

---
Primary Category: cs.SD  
Categories: cs-AI, cs-CL, cs-SD, cs.SD, eess-AS  
Keywords: Few-Shot, Low-Resource  
[Paper Link](http://arxiv.org/abs/2310.04982v1)  

---


**ABSTRACT**  
Text-to-Speech (TTS) synthesis using deep learning relies on voice quality. Modern TTS models are advanced, but they need large amount of data. Given the growing computational complexity of these models and the scarcity of large, high-quality datasets, this research focuses on transfer learning, especially on few-shot, low-resource, and customized datasets. In this research, "low-resource" specifically refers to situations where there are limited amounts of training data, such as a small number of audio recordings and corresponding transcriptions for a particular language or dialect. This thesis, is rooted in the pressing need to find TTS models that require less training time, fewer data samples, yet yield high-quality voice output. The research evaluates TTS state-of-the-art model transfer learning capabilities through a thorough technical analysis. It then conducts a hands-on experimental analysis to compare models' performance in a constrained dataset. This study investigates the efficacy of modern TTS systems with transfer learning on specialized datasets and a model that balances training efficiency and synthesis quality. Initial hypotheses suggest that transfer learning could significantly improve TTS models' performance on compact datasets, and an optimal model may exist for such unique conditions. This thesis predicts a rise in transfer learning in TTS as data scarcity increases. In the future, custom TTS applications will favour models optimized for specific datasets over generic, data-intensive ones.

{{</citation>}}


## eess.AS (1)



### (108/112) Partial Rank Similarity Minimization Method for Quality MOS Prediction of Unseen Speech Synthesis Systems in Zero-Shot and Semi-supervised setting (Hemant Yadav et al., 2023)

{{<citation>}}

Hemant Yadav, Erica Cooper, Junichi Yamagishi, Sunayana Sitaram, Rajiv Ratn Shah. (2023)  
**Partial Rank Similarity Minimization Method for Quality MOS Prediction of Unseen Speech Synthesis Systems in Zero-Shot and Semi-supervised setting**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2310.05078v1)  

---


**ABSTRACT**  
This paper introduces a novel objective function for quality mean opinion score (MOS) prediction of unseen speech synthesis systems. The proposed function measures the similarity of relative positions of predicted MOS values, in a mini-batch, rather than the actual MOS values. That is the partial rank similarity is measured (PRS) rather than the individual MOS values as with the L1 loss. Our experiments on out-of-domain speech synthesis systems demonstrate that the PRS outperforms L1 loss in zero-shot and semi-supervised settings, exhibiting stronger correlation with ground truth. These findings highlight the importance of considering rank order, as done by PRS, when training MOS prediction models. We also argue that mean squared error and linear correlation coefficient metrics may be unreliable for evaluating MOS prediction models. In conclusion, PRS-trained models provide a robust framework for evaluating speech quality and offer insights for developing high-quality speech synthesis systems. Code and models are available at github.com/nii-yamagishilab/partial_rank_similarity/

{{</citation>}}


## cs.CY (1)



### (109/112) Unmasking Biases and Navigating Pitfalls in the Ophthalmic Artificial Intelligence Lifecycle: A Review (Luis Filipe Nakayama et al., 2023)

{{<citation>}}

Luis Filipe Nakayama, João Matos, Justin Quion, Frederico Novaes, William Greig Mitchell, Rogers Mwavu, Ju-Yi Ji Hung, Alvina Pauline dy Santiago, Warachaya Phanphruk, Jaime S. Cardoso, Leo Anthony Celi. (2023)  
**Unmasking Biases and Navigating Pitfalls in the Ophthalmic Artificial Intelligence Lifecycle: A Review**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI, Bias  
[Paper Link](http://arxiv.org/abs/2310.04997v1)  

---


**ABSTRACT**  
Over the past two decades, exponential growth in data availability, computational power, and newly available modeling techniques has led to an expansion in interest, investment, and research in Artificial Intelligence (AI) applications. Ophthalmology is one of many fields that seek to benefit from AI given the advent of telemedicine screening programs and the use of ancillary imaging. However, before AI can be widely deployed, further work must be done to avoid the pitfalls within the AI lifecycle. This review article breaks down the AI lifecycle into seven steps: data collection; defining the model task; data pre-processing and labeling; model development; model evaluation and validation; deployment; and finally, post-deployment evaluation, monitoring, and system recalibration and delves into the risks for harm at each step and strategies for mitigating them.

{{</citation>}}


## eess.IV (1)



### (110/112) VisionFM: a Multi-Modal Multi-Task Vision Foundation Model for Generalist Ophthalmic Artificial Intelligence (Jianing Qiu et al., 2023)

{{<citation>}}

Jianing Qiu, Jian Wu, Hao Wei, Peilun Shi, Minqing Zhang, Yunyun Sun, Lin Li, Hanruo Liu, Hongyi Liu, Simeng Hou, Yuyang Zhao, Xuehui Shi, Junfang Xian, Xiaoxia Qu, Sirui Zhu, Lijie Pan, Xiaoniao Chen, Xiaojia Zhang, Shuai Jiang, Kebing Wang, Chenlong Yang, Mingqiang Chen, Sujie Fan, Jianhua Hu, Aiguo Lv, Hui Miao, Li Guo, Shujun Zhang, Cheng Pei, Xiaojuan Fan, Jianqin Lei, Ting Wei, Junguo Duan, Chun Liu, Xiaobo Xia, Siqi Xiong, Junhong Li, Benny Lo, Yih Chung Tham, Tien Yin Wong, Ningli Wang, Wu Yuan. (2023)  
**VisionFM: a Multi-Modal Multi-Task Vision Foundation Model for Generalist Ophthalmic Artificial Intelligence**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.04992v1)  

---


**ABSTRACT**  
We present VisionFM, a foundation model pre-trained with 3.4 million ophthalmic images from 560,457 individuals, covering a broad range of ophthalmic diseases, modalities, imaging devices, and demography. After pre-training, VisionFM provides a foundation to foster multiple ophthalmic artificial intelligence (AI) applications, such as disease screening and diagnosis, disease prognosis, subclassification of disease phenotype, and systemic biomarker and disease prediction, with each application enhanced with expert-level intelligence and accuracy. The generalist intelligence of VisionFM outperformed ophthalmologists with basic and intermediate levels in jointly diagnosing 12 common ophthalmic diseases. Evaluated on a new large-scale ophthalmic disease diagnosis benchmark database, as well as a new large-scale segmentation and detection benchmark database, VisionFM outperformed strong baseline deep neural networks. The ophthalmic image representations learned by VisionFM exhibited noteworthy explainability, and demonstrated strong generalizability to new ophthalmic modalities, disease spectrum, and imaging devices. As a foundation model, VisionFM has a large capacity to learn from diverse ophthalmic imaging data and disparate datasets. To be commensurate with this capacity, in addition to the real data used for pre-training, we also generated and leveraged synthetic ophthalmic imaging data. Experimental results revealed that synthetic data that passed visual Turing tests, can also enhance the representation learning capability of VisionFM, leading to substantial performance gains on downstream ophthalmic AI tasks. Beyond the ophthalmic AI applications developed, validated, and demonstrated in this work, substantial further applications can be achieved in an efficient and cost-effective manner using VisionFM as the foundation.

{{</citation>}}


## cs.CE (1)



### (111/112) VQPL: Vector Quantized Protein Language (Zhangyang Gao et al., 2023)

{{<citation>}}

Zhangyang Gao, Cheng Tan, Stan Z. Li. (2023)  
**VQPL: Vector Quantized Protein Language**  

---
Primary Category: cs.CE  
Categories: cs-CE, cs.CE  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2310.04985v1)  

---


**ABSTRACT**  
Is there a foreign language describing protein sequences and structures simultaneously? Protein structures, represented by continuous 3D points, have long posed a challenge due to the contrasting modeling paradigms of discrete sequences. To represent protein sequence-structure as discrete symbols, we propose a VQProteinformer to project residue types and structures into a discrete space, supervised by a reconstruction loss to ensure information preservation. The sequential latent codes of residues introduce a new quantized protein language, transforming the protein sequence-structure into a unified modality. We demonstrate the potential of the created protein language on predictive and generative tasks, which may not only advance protein research but also establish a connection between the protein-related and NLP-related fields. The proposed method will be continually improved to unify more protein modalities, including text and point cloud.

{{</citation>}}


## cs.CR (1)



### (112/112) Big Data Privacy in Emerging Market Fintech and Financial Services: A Research Agenda (Joshua E. Blumenstock et al., 2023)

{{<citation>}}

Joshua E. Blumenstock, Nitin Kohli. (2023)  
**Big Data Privacy in Emerging Market Fintech and Financial Services: A Research Agenda**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Financial  
[Paper Link](http://arxiv.org/abs/2310.04970v1)  

---


**ABSTRACT**  
The data revolution in low- and middle-income countries is quickly transforming how companies approach emerging markets. As mobile phones and mobile money proliferate, they generate new streams of data that enable innovation in consumer finance, credit, and insurance. Already, this new generation of products are being used by hundreds of millions of consumers, often to use financial services for the first time. However, the collection, analysis, and use of these data, particularly from economically disadvantaged populations, raises serious privacy concerns. This white paper describes a research agenda to advance our understanding of the problem and solution space of data privacy in emerging market fintech and financial services. We highlight five priority areas for research: conducting comprehensive landscape analyses; understanding local definitions of ``data privacy''; documenting key sources of risk, and potential technical solutions (such as differential privacy and homomorphic encryption); improving non-technical approaches to data privacy (such as policies and practices); and understanding the tradeoffs involved in deploying privacy-enhancing solutions. Taken together, we hope this research agenda will focus attention on the multi-faceted nature of privacy in emerging markets, and catalyze efforts to develop responsible and consumer-oriented approaches to data-intensive applications.

{{</citation>}}
