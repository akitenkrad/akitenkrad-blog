---
draft: false
title: "arXiv @ 2023.10.04"
date: 2023-10-04
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.10.04"
    identifier: arxiv_20231004
    parent: 202310_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.LG (35)](#cslg-35)
- [cs.CL (45)](#cscl-45)
- [physics.ao-ph (1)](#physicsao-ph-1)
- [eess.AS (3)](#eessas-3)
- [cs.AI (8)](#csai-8)
- [cs.MA (1)](#csma-1)
- [cs.CV (30)](#cscv-30)
- [cs.CR (2)](#cscr-2)
- [cs.SE (4)](#csse-4)
- [quant-ph (1)](#quant-ph-1)
- [cs.HC (6)](#cshc-6)
- [cs.IR (1)](#csir-1)
- [stat.ML (3)](#statml-3)
- [eess.SY (1)](#eesssy-1)
- [eess.IV (5)](#eessiv-5)
- [cs.RO (5)](#csro-5)
- [cs.SI (5)](#cssi-5)
- [cs.NE (1)](#csne-1)
- [cs.NI (1)](#csni-1)
- [physics.plasm-ph (1)](#physicsplasm-ph-1)
- [cs.DB (2)](#csdb-2)
- [q-fin.RM (1)](#q-finrm-1)
- [physics.soc-ph (1)](#physicssoc-ph-1)
- [physics.geo-ph (1)](#physicsgeo-ph-1)
- [cs.DC (1)](#csdc-1)

## cs.LG (35)



### (1/165) Transformers are efficient hierarchical chemical graph learners (Zihan Pengmei et al., 2023)

{{<citation>}}

Zihan Pengmei, Zimu Li, Chih-chan Tien, Risi Kondor, Aaron R. Dinner. (2023)  
**Transformers are efficient hierarchical chemical graph learners**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.01704v1)  

---


**ABSTRACT**  
Transformers, adapted from natural language processing, are emerging as a leading approach for graph representation learning. Contemporary graph transformers often treat nodes or edges as separate tokens. This approach leads to computational challenges for even moderately-sized graphs due to the quadratic scaling of self-attention complexity with token count. In this paper, we introduce SubFormer, a graph transformer that operates on subgraphs that aggregate information by a message-passing mechanism. This approach reduces the number of tokens and enhances learning long-range interactions. We demonstrate SubFormer on benchmarks for predicting molecular properties from chemical structures and show that it is competitive with state-of-the-art graph transformers at a fraction of the computational cost, with training times on the order of minutes on a consumer-grade graphics card. We interpret the attention weights in terms of chemical structures. We show that SubFormer exhibits limited over-smoothing and avoids over-squashing, which is prevalent in traditional graph neural networks.

{{</citation>}}


### (2/165) Locality-Aware Graph-Rewiring in GNNs (Federico Barbero et al., 2023)

{{<citation>}}

Federico Barbero, Ameya Velingker, Amin Saberi, Michael Bronstein, Francesco Di Giovanni. (2023)  
**Locality-Aware Graph-Rewiring in GNNs**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.01668v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) are popular models for machine learning on graphs that typically follow the message-passing paradigm, whereby the feature of a node is updated recursively upon aggregating information over its neighbors. While exchanging messages over the input graph endows GNNs with a strong inductive bias, it can also make GNNs susceptible to over-squashing, thereby preventing them from capturing long-range interactions in the given graph. To rectify this issue, graph rewiring techniques have been proposed as a means of improving information flow by altering the graph connectivity. In this work, we identify three desiderata for graph-rewiring: (i) reduce over-squashing, (ii) respect the locality of the graph, and (iii) preserve the sparsity of the graph. We highlight fundamental trade-offs that occur between spatial and spectral rewiring techniques; while the former often satisfy (i) and (ii) but not (iii), the latter generally satisfy (i) and (iii) at the expense of (ii). We propose a novel rewiring framework that satisfies all of (i)--(iii) through a locality-aware sequence of rewiring operations. We then discuss a specific instance of such rewiring framework and validate its effectiveness on several real-world benchmarks, showing that it either matches or significantly outperforms existing rewiring approaches.

{{</citation>}}


### (3/165) PolySketchFormer: Fast Transformers via Sketches for Polynomial Kernels (Praneeth Kacham et al., 2023)

{{<citation>}}

Praneeth Kacham, Vahab Mirrokni, Peilin Zhong. (2023)  
**PolySketchFormer: Fast Transformers via Sketches for Polynomial Kernels**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Attention, Sketch, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.01655v1)  

---


**ABSTRACT**  
The quadratic complexity of attention in transformer architectures remains a big bottleneck in scaling up large foundation models for long context. In fact, recent theoretical results show the hardness of approximating the output of softmax attention mechanism in sub-quadratic time assuming Strong Exponential Time Hypothesis. In this paper, we show how to break this theoretical barrier by replacing softmax with a polynomial function and polynomial sketching. In particular we show that sketches for Polynomial Kernel from the randomized numerical linear algebra literature can be used to approximate the polynomial attention which leads to a significantly faster attention mechanism without assuming any sparse structure for the attention matrix that has been done in many previous works.   In addition, we propose an efficient block-based algorithm that lets us apply the causal mask to the attention matrix without explicitly realizing the $n \times n$ attention matrix and compute the output of the polynomial attention mechanism in time linear in the context length. The block-based algorithm gives significant speedups over the \emph{cumulative sum} algorithm used by Performer to apply the causal mask to the attention matrix. These observations help us design \emph{PolySketchFormer}, a practical linear-time transformer architecture for language modeling with provable guarantees.   We validate our design empirically by training language models with long context lengths. We first show that the eval perplexities of our models are comparable to that of models trained with softmax attention. We then show that for large context lengths our training times are significantly faster than FlashAttention.

{{</citation>}}


### (4/165) Fool Your (Vision and) Language Model With Embarrassingly Simple Permutations (Yongshuo Zong et al., 2023)

{{<citation>}}

Yongshuo Zong, Tingyang Yu, Bingchen Zhao, Ruchika Chavhan, Timothy Hospedales. (2023)  
**Fool Your (Vision and) Language Model With Embarrassingly Simple Permutations**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Language Model, QA  
[Paper Link](http://arxiv.org/abs/2310.01651v1)  

---


**ABSTRACT**  
Large language and vision-language models are rapidly being deployed in practice thanks to their impressive capabilities in instruction following, in-context learning, and so on. This raises an urgent need to carefully analyse their robustness so that stakeholders can understand if and when such models are trustworthy enough to be relied upon in any given application. In this paper, we highlight a specific vulnerability in popular models, namely permutation sensitivity in multiple-choice question answering (MCQA). Specifically, we show empirically that popular models are vulnerable to adversarial permutation in answer sets for multiple-choice prompting, which is surprising as models should ideally be as invariant to prompt permutation as humans are. These vulnerabilities persist across various model sizes, and exist in very recent language and vision-language models. Code is available at \url{https://github.com/ys-zong/FoolyourVLLMs}.

{{</citation>}}


### (5/165) Equivariant Adaptation of Large Pre-Trained Models (Arnab Kumar Mondal et al., 2023)

{{<citation>}}

Arnab Kumar Mondal, Siba Smarak Panigrahi, Sékou-Oumar Kaba, Sai Rajeswar, Siamak Ravanbakhsh. (2023)  
**Equivariant Adaptation of Large Pre-Trained Models**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Pre-Trained Model  
[Paper Link](http://arxiv.org/abs/2310.01647v1)  

---


**ABSTRACT**  
Equivariant networks are specifically designed to ensure consistent behavior with respect to a set of input transformations, leading to higher sample efficiency and more accurate and robust predictions. However, redesigning each component of prevalent deep neural network architectures to achieve chosen equivariance is a difficult problem and can result in a computationally expensive network during both training and inference. A recently proposed alternative towards equivariance that removes the architectural constraints is to use a simple canonicalization network that transforms the input to a canonical form before feeding it to an unconstrained prediction network. We show here that this approach can effectively be used to make a large pre-trained network equivariant. However, we observe that the produced canonical orientations can be misaligned with those of the training distribution, hindering performance. Using dataset-dependent priors to inform the canonicalization function, we are able to make large pre-trained models equivariant while maintaining their performance. This significantly improves the robustness of these models to deterministic transformations of the data, such as rotations. We believe this equivariant adaptation of large pre-trained models can help their domain-specific applications with known symmetry priors.

{{</citation>}}


### (6/165) Sample-Efficiency in Multi-Batch Reinforcement Learning: The Need for Dimension-Dependent Adaptivity (Emmeran Johnson et al., 2023)

{{<citation>}}

Emmeran Johnson, Ciara Pike-Burke, Patrick Rebeschini. (2023)  
**Sample-Efficiency in Multi-Batch Reinforcement Learning: The Need for Dimension-Dependent Adaptivity**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.01616v1)  

---


**ABSTRACT**  
We theoretically explore the relationship between sample-efficiency and adaptivity in reinforcement learning. An algorithm is sample-efficient if it uses a number of queries $n$ to the environment that is polynomial in the dimension $d$ of the problem. Adaptivity refers to the frequency at which queries are sent and feedback is processed to update the querying strategy. To investigate this interplay, we employ a learning framework that allows sending queries in $K$ batches, with feedback being processed and queries updated after each batch. This model encompasses the whole adaptivity spectrum, ranging from non-adaptive 'offline' ($K=1$) to fully adaptive ($K=n$) scenarios, and regimes in between. For the problems of policy evaluation and best-policy identification under $d$-dimensional linear function approximation, we establish $\Omega(\log \log d)$ lower bounds on the number of batches $K$ required for sample-efficient algorithms with $n = O(poly(d))$ queries. Our results show that just having adaptivity ($K>1$) does not necessarily guarantee sample-efficiency. Notably, the adaptivity-boundary for sample-efficiency is not between offline reinforcement learning ($K=1$), where sample-efficiency was known to not be possible, and adaptive settings. Instead, the boundary lies between different regimes of adaptivity and depends on the problem dimension.

{{</citation>}}


### (7/165) Solving the Quadratic Assignment Problem using Deep Reinforcement Learning (Puneet S. Bagga et al., 2023)

{{<citation>}}

Puneet S. Bagga, Arthur Delarue. (2023)  
**Solving the Quadratic Assignment Problem using Deep Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, math-OC  
Keywords: QA, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.01604v1)  

---


**ABSTRACT**  
The Quadratic Assignment Problem (QAP) is an NP-hard problem which has proven particularly challenging to solve: unlike other combinatorial problems like the traveling salesman problem (TSP), which can be solved to optimality for instances with hundreds or even thousands of locations using advanced integer programming techniques, no methods are known to exactly solve QAP instances of size greater than 30. Solving the QAP is nevertheless important because of its many critical applications, such as electronic wiring design and facility layout selection. We propose a method to solve the original Koopmans-Beckman formulation of the QAP using deep reinforcement learning. Our approach relies on a novel double pointer network, which alternates between selecting a location in which to place the next facility and a facility to place in the previous location. We train our model using A2C on a large dataset of synthetic instances, producing solutions with no instance-specific retraining necessary. Out of sample, our solutions are on average within 7.5% of a high-quality local search baseline, and even outperform it on 1.2% of instances.

{{</citation>}}


### (8/165) Pool-Based Active Learning with Proper Topological Regions (Lies Hadjadj et al., 2023)

{{<citation>}}

Lies Hadjadj, Emilie Devijver, Remi Molinier, Massih-Reza Amini. (2023)  
**Pool-Based Active Learning with Proper Topological Regions**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, math-AT  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2310.01597v1)  

---


**ABSTRACT**  
Machine learning methods usually rely on large sample size to have good performance, while it is difficult to provide labeled set in many applications. Pool-based active learning methods are there to detect, among a set of unlabeled data, the ones that are the most relevant for the training. We propose in this paper a meta-approach for pool-based active learning strategies in the context of multi-class classification tasks based on Proper Topological Regions. PTR, based on topological data analysis (TDA), are relevant regions used to sample cold-start points or within the active learning scheme. The proposed method is illustrated empirically on various benchmark datasets, being competitive to the classical methods from the literature.

{{</citation>}}


### (9/165) On the Safety of Open-Sourced Large Language Models: Does Alignment Really Prevent Them From Being Misused? (Hangfan Zhang et al., 2023)

{{<citation>}}

Hangfan Zhang, Zhimeng Guo, Huaisheng Zhu, Bochuan Cao, Lu Lin, Jinyuan Jia, Jinghui Chen, Dinghao Wu. (2023)  
**On the Safety of Open-Sourced Large Language Models: Does Alignment Really Prevent Them From Being Misused?**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CR, cs-LG, cs.LG  
Keywords: Language Model, Natural Language Generation, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.01581v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have achieved unprecedented performance in Natural Language Generation (NLG) tasks. However, many existing studies have shown that they could be misused to generate undesired content. In response, before releasing LLMs for public access, model developers usually align those language models through Supervised Fine-Tuning (SFT) or Reinforcement Learning with Human Feedback (RLHF). Consequently, those aligned large language models refuse to generate undesired content when facing potentially harmful/unethical requests. A natural question is "could alignment really prevent those open-sourced large language models from being misused to generate undesired content?''. In this work, we provide a negative answer to this question. In particular, we show those open-sourced, aligned large language models could be easily misguided to generate undesired content without heavy computations or careful prompt designs. Our key idea is to directly manipulate the generation process of open-sourced LLMs to misguide it to generate undesired content including harmful or biased information and even private data. We evaluate our method on 4 open-sourced LLMs accessible publicly and our finding highlights the need for more advanced mitigation strategies for open-sourced LLMs.

{{</citation>}}


### (10/165) Fusing Models with Complementary Expertise (Hongyi Wang et al., 2023)

{{<citation>}}

Hongyi Wang, Felipe Maia Polo, Yuekai Sun, Souvik Kundu, Eric Xing, Mikhail Yurochkin. (2023)  
**Fusing Models with Complementary Expertise**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI, QA  
[Paper Link](http://arxiv.org/abs/2310.01542v1)  

---


**ABSTRACT**  
Training AI models that generalize across tasks and domains has long been among the open problems driving AI research. The emergence of Foundation Models made it easier to obtain expert models for a given task, but the heterogeneity of data that may be encountered at test time often means that any single expert is insufficient. We consider the Fusion of Experts (FoE) problem of fusing outputs of expert models with complementary knowledge of the data distribution and formulate it as an instance of supervised learning. Our method is applicable to both discriminative and generative tasks and leads to significant performance improvements in image and text classification, text summarization, multiple-choice QA, and automatic evaluation of generated text. We also extend our method to the "frugal" setting where it is desired to reduce the number of expert model evaluations at test time.

{{</citation>}}


### (11/165) Representation Engineering: A Top-Down Approach to AI Transparency (Andy Zou et al., 2023)

{{<citation>}}

Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, Shashwat Goel, Nathaniel Li, Michael J. Byun, Zifan Wang, Alex Mallen, Steven Basart, Sanmi Koyejo, Dawn Song, Matt Fredrikson, J. Zico Kolter, Dan Hendrycks. (2023)  
**Representation Engineering: A Top-Down Approach to AI Transparency**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-CV, cs-CY, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.01405v2)  

---


**ABSTRACT**  
In this paper, we identify and characterize the emerging area of representation engineering (RepE), an approach to enhancing the transparency of AI systems that draws on insights from cognitive neuroscience. RepE places population-level representations, rather than neurons or circuits, at the center of analysis, equipping us with novel methods for monitoring and manipulating high-level cognitive phenomena in deep neural networks (DNNs). We provide baselines and an initial analysis of RepE techniques, showing that they offer simple yet effective solutions for improving our understanding and control of large language models. We showcase how these methods can provide traction on a wide range of safety-relevant problems, including honesty, harmlessness, power-seeking, and more, demonstrating the promise of top-down transparency research. We hope that this work catalyzes further exploration of RepE and fosters advancements in the transparency and safety of AI systems.

{{</citation>}}


### (12/165) H-InDex: Visual Reinforcement Learning with Hand-Informed Representations for Dexterous Manipulation (Yanjie Ze et al., 2023)

{{<citation>}}

Yanjie Ze, Yuyao Liu, Ruizhe Shi, Jiaxin Qin, Zhecheng Yuan, Jiashun Wang, Huazhe Xu. (2023)  
**H-InDex: Visual Reinforcement Learning with Hand-Informed Representations for Dexterous Manipulation**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs-RO, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.01404v1)  

---


**ABSTRACT**  
Human hands possess remarkable dexterity and have long served as a source of inspiration for robotic manipulation. In this work, we propose a human $\textbf{H}$and$\textbf{-In}$formed visual representation learning framework to solve difficult $\textbf{Dex}$terous manipulation tasks ($\textbf{H-InDex}$) with reinforcement learning. Our framework consists of three stages: (i) pre-training representations with 3D human hand pose estimation, (ii) offline adapting representations with self-supervised keypoint detection, and (iii) reinforcement learning with exponential moving average BatchNorm. The last two stages only modify $0.36\%$ parameters of the pre-trained representation in total, ensuring the knowledge from pre-training is maintained to the full extent. We empirically study 12 challenging dexterous manipulation tasks and find that H-InDex largely surpasses strong baseline methods and the recent visual foundation models for motor control. Code is available at https://yanjieze.com/H-InDex .

{{</citation>}}


### (13/165) Pessimistic Nonlinear Least-Squares Value Iteration for Offline Reinforcement Learning (Qiwei Di et al., 2023)

{{<citation>}}

Qiwei Di, Heyang Zhao, Jiafan He, Quanquan Gu. (2023)  
**Pessimistic Nonlinear Least-Squares Value Iteration for Offline Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, math-OC, stat-ML  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.01380v1)  

---


**ABSTRACT**  
Offline reinforcement learning (RL), where the agent aims to learn the optimal policy based on the data collected by a behavior policy, has attracted increasing attention in recent years. While offline RL with linear function approximation has been extensively studied with optimal results achieved under certain assumptions, many works shift their interest to offline RL with non-linear function approximation. However, limited works on offline RL with non-linear function approximation have instance-dependent regret guarantees. In this paper, we propose an oracle-efficient algorithm, dubbed Pessimistic Nonlinear Least-Square Value Iteration (PNLSVI), for offline RL with non-linear function approximation. Our algorithmic design comprises three innovative components: (1) a variance-based weighted regression scheme that can be applied to a wide range of function classes, (2) a subroutine for variance estimation, and (3) a planning phase that utilizes a pessimistic value iteration approach. Our algorithm enjoys a regret bound that has a tight dependency on the function class complexity and achieves minimax optimal instance-dependent regret when specialized to linear function approximation. Our work extends the previous instance-dependent results within simpler function classes, such as linear and differentiable function to a more general framework.

{{</citation>}}


### (14/165) GenSim: Generating Robotic Simulation Tasks via Large Language Models (Lirui Wang et al., 2023)

{{<citation>}}

Lirui Wang, Yiyang Ling, Zhecheng Yuan, Mohit Shridhar, Chen Bao, Yuzhe Qin, Bailin Wang, Huazhe Xu, Xiaolong Wang. (2023)  
**GenSim: Generating Robotic Simulation Tasks via Large Language Models**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-CV, cs-LG, cs-RO, cs.LG  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.01361v1)  

---


**ABSTRACT**  
Collecting large amounts of real-world interaction data to train general robotic policies is often prohibitively expensive, thus motivating the use of simulation data. However, existing methods for data generation have generally focused on scene-level diversity (e.g., object instances and poses) rather than task-level diversity, due to the human effort required to come up with and verify novel tasks. This has made it challenging for policies trained on simulation data to demonstrate significant task-level generalization. In this paper, we propose to automatically generate rich simulation environments and expert demonstrations by exploiting a large language models' (LLM) grounding and coding ability. Our approach, dubbed GenSim, has two modes: goal-directed generation, wherein a target task is given to the LLM and the LLM proposes a task curriculum to solve the target task, and exploratory generation, wherein the LLM bootstraps from previous tasks and iteratively proposes novel tasks that would be helpful in solving more complex tasks. We use GPT4 to expand the existing benchmark by ten times to over 100 tasks, on which we conduct supervised finetuning and evaluate several LLMs including finetuned GPTs and Code Llama on code generation for robotic simulation tasks. Furthermore, we observe that LLMs-generated simulation programs can enhance task-level generalization significantly when used for multitask policy training. We further find that with minimal sim-to-real adaptation, the multitask policies pretrained on GPT4-generated simulation tasks exhibit stronger transfer to unseen long-horizon tasks in the real world and outperform baselines by 25%. See the project website (https://liruiw.github.io/gensim) for code, demos, and videos.

{{</citation>}}


### (15/165) TACTiS-2: Better, Faster, Simpler Attentional Copulas for Multivariate Time Series (Arjun Ashok et al., 2023)

{{<citation>}}

Arjun Ashok, Étienne Marcotte, Valentina Zantedeschi, Nicolas Chapados, Alexandre Drouin. (2023)  
**TACTiS-2: Better, Faster, Simpler Attentional Copulas for Multivariate Time Series**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, stat-ML  
Keywords: Attention, Time Series  
[Paper Link](http://arxiv.org/abs/2310.01327v1)  

---


**ABSTRACT**  
We introduce a new model for multivariate probabilistic time series prediction, designed to flexibly address a range of tasks including forecasting, interpolation, and their combinations. Building on copula theory, we propose a simplified objective for the recently-introduced transformer-based attentional copulas (TACTiS), wherein the number of distributional parameters now scales linearly with the number of variables instead of factorially. The new objective requires the introduction of a training curriculum, which goes hand-in-hand with necessary changes to the original architecture. We show that the resulting model has significantly better training dynamics and achieves state-of-the-art performance across diverse real-world forecasting tasks, while maintaining the flexibility of prior work, such as seamless handling of unaligned and unevenly-sampled time series.

{{</citation>}}


### (16/165) Cooperative Graph Neural Networks (Ben Finkelshtein et al., 2023)

{{<citation>}}

Ben Finkelshtein, Xingyue Huang, Michael Bronstein, İsmail İlkan Ceylan. (2023)  
**Cooperative Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.01267v1)  

---


**ABSTRACT**  
Graph neural networks are popular architectures for graph machine learning, based on iterative computation of node representations of an input graph through a series of invariant transformations. A large class of graph neural networks follow a standard message-passing paradigm: at every layer, each node state is updated based on an aggregate of messages from its neighborhood. In this work, we propose a novel framework for training graph neural networks, where every node is viewed as a player that can choose to either 'listen', 'broadcast', 'listen and broadcast', or to 'isolate'. The standard message propagation scheme can then be viewed as a special case of this framework where every node 'listens and broadcasts' to all neighbors. Our approach offers a more flexible and dynamic message-passing paradigm, where each node can determine its own strategy based on their state, effectively exploring the graph topology while learning. We provide a theoretical analysis of the new message-passing scheme which is further supported by an extensive empirical analysis on a synthetic dataset and on real-world datasets.

{{</citation>}}


### (17/165) Self-supervised Learning for Anomaly Detection in Computational Workflows (Hongwei Jin et al., 2023)

{{<citation>}}

Hongwei Jin, Krishnan Raghavan, George Papadimitriou, Cong Wang, Anirban Mandal, Ewa Deelman, Prasanna Balaprakash. (2023)  
**Self-supervised Learning for Anomaly Detection in Computational Workflows**  

---
Primary Category: cs.LG  
Categories: cs-DC, cs-LG, cs.LG  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2310.01247v1)  

---


**ABSTRACT**  
Anomaly detection is the task of identifying abnormal behavior of a system. Anomaly detection in computational workflows is of special interest because of its wide implications in various domains such as cybersecurity, finance, and social networks. However, anomaly detection in computational workflows~(often modeled as graphs) is a relatively unexplored problem and poses distinct challenges. For instance, when anomaly detection is performed on graph data, the complex interdependency of nodes and edges, the heterogeneity of node attributes, and edge types must be accounted for. Although the use of graph neural networks can help capture complex inter-dependencies, the scarcity of labeled anomalous examples from workflow executions is still a significant challenge. To address this problem, we introduce an autoencoder-driven self-supervised learning~(SSL) approach that learns a summary statistic from unlabeled workflow data and estimates the normal behavior of the computational workflow in the latent space. In this approach, we combine generative and contrastive learning objectives to detect outliers in the summary statistics. We demonstrate that by estimating the distribution of normal behavior in the latent space, we can outperform state-of-the-art anomaly detection methods on our benchmark datasets.

{{</citation>}}


### (18/165) Modality-aware Transformer for Time series Forecasting (Hajar Emami et al., 2023)

{{<citation>}}

Hajar Emami, Xuan-Hong Dang, Yousaf Shah, Petros Zerfos. (2023)  
**Modality-aware Transformer for Time series Forecasting**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.01232v1)  

---


**ABSTRACT**  
Time series forecasting presents a significant challenge, particularly when its accuracy relies on external data sources rather than solely on historical values. This issue is prevalent in the financial sector, where the future behavior of time series is often intricately linked to information derived from various textual reports and a multitude of economic indicators. In practice, the key challenge lies in constructing a reliable time series forecasting model capable of harnessing data from diverse sources and extracting valuable insights to predict the target time series accurately. In this work, we tackle this challenging problem and introduce a novel multimodal transformer-based model named the Modality-aware Transformer. Our model excels in exploring the power of both categorical text and numerical timeseries to forecast the target time series effectively while providing insights through its neural attention mechanism. To achieve this, we develop feature-level attention layers that encourage the model to focus on the most relevant features within each data modality. By incorporating the proposed feature-level attention, we develop a novel Intra-modal multi-head attention (MHA), Inter-modal MHA and Modality-target MHA in a way that both feature and temporal attentions are incorporated in MHAs. This enables the MHAs to generate temporal attentions with consideration of modality and feature importance which leads to more informative embeddings. The proposed modality-aware structure enables the model to effectively exploit information within each modality as well as foster cross-modal understanding. Our extensive experiments on financial datasets demonstrate that Modality-aware Transformer outperforms existing methods, offering a novel and practical solution to the complex challenges of multi-modality time series forecasting.

{{</citation>}}


### (19/165) Revisiting Mobility Modeling with Graph: A Graph Transformer Model for Next Point-of-Interest Recommendation (Xiaohang Xu et al., 2023)

{{<citation>}}

Xiaohang Xu, Toyotaro Suzumura, Jiawei Yong, Masatoshi Hanai, Chuang Yang, Hiroki Kanezashi, Renhe Jiang, Shintaro Fukushima. (2023)  
**Revisiting Mobility Modeling with Graph: A Graph Transformer Model for Next Point-of-Interest Recommendation**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks, Transformer  
[Paper Link](http://arxiv.org/abs/2310.01224v1)  

---


**ABSTRACT**  
Next Point-of-Interest (POI) recommendation plays a crucial role in urban mobility applications. Recently, POI recommendation models based on Graph Neural Networks (GNN) have been extensively studied and achieved, however, the effective incorporation of both spatial and temporal information into such GNN-based models remains challenging. Extracting distinct fine-grained features unique to each piece of information is difficult since temporal information often includes spatial information, as users tend to visit nearby POIs. To address the challenge, we propose \textbf{\underline{Mob}}ility \textbf{\underline{G}}raph \textbf{\underline{T}}ransformer (MobGT) that enables us to fully leverage graphs to capture both the spatial and temporal features in users' mobility patterns. MobGT combines individual spatial and temporal graph encoders to capture unique features and global user-location relations. Additionally, it incorporates a mobility encoder based on Graph Transformer to extract higher-order information between POIs. To address the long-tailed problem in spatial-temporal data, MobGT introduces a novel loss function, Tail Loss. Experimental results demonstrate that MobGT outperforms state-of-the-art models on various datasets and metrics, achieving 24\% improvement on average. Our codes are available at \url{https://github.com/Yukayo/MobGT}.

{{</citation>}}


### (20/165) PASTA: PArallel Spatio-Temporal Attention with spatial auto-correlation gating for fine-grained crowd flow prediction (Chung Park et al., 2023)

{{<citation>}}

Chung Park, Junui Hong, Cheonbok Park, Taesan Kim, Minsung Choi, Jaegul Choo. (2023)  
**PASTA: PArallel Spatio-Temporal Attention with spatial auto-correlation gating for fine-grained crowd flow prediction**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2310.02284v1)  

---


**ABSTRACT**  
Understanding the movement patterns of objects (e.g., humans and vehicles) in a city is essential for many applications, including city planning and management. This paper proposes a method for predicting future city-wide crowd flows by modeling the spatio-temporal patterns of historical crowd flows in fine-grained city-wide maps. We introduce a novel neural network named PArallel Spatio-Temporal Attention with spatial auto-correlation gating (PASTA) that effectively captures the irregular spatio-temporal patterns of fine-grained maps. The novel components in our approach include spatial auto-correlation gating, multi-scale residual block, and temporal attention gating module. The spatial auto-correlation gating employs the concept of spatial statistics to identify irregular spatial regions. The multi-scale residual block is responsible for handling multiple range spatial dependencies in the fine-grained map, and the temporal attention gating filters out irrelevant temporal information for the prediction. The experimental results demonstrate that our model outperforms other competing baselines, especially under challenging conditions that contain irregular spatial regions. We also provide a qualitative analysis to derive the critical time information where our model assigns high attention scores in prediction.

{{</citation>}}


### (21/165) ScaLearn: Simple and Highly Parameter-Efficient Task Transfer by Learning to Scale (Markus Frohmann et al., 2023)

{{<citation>}}

Markus Frohmann, Carolin Holtermann, Shahed Masoudian, Anne Lauscher, Navid Rekabsaz. (2023)  
**ScaLearn: Simple and Highly Parameter-Efficient Task Transfer by Learning to Scale**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: GLUE, SuperGLUE  
[Paper Link](http://arxiv.org/abs/2310.01217v1)  

---


**ABSTRACT**  
Multi-task learning (MTL) has shown considerable practical benefits, particularly when using pre-trained language models (PLMs). While this is commonly achieved by simultaneously learning $n$ tasks under a joint optimization procedure, recent methods such as AdapterFusion structure the problem into two distinct stages: (i) task learning, where knowledge specific to a task is encapsulated within sets of parameters (\eg adapters), and (ii) transfer, where this already learned knowledge is leveraged for a target task. This separation of concerns provides numerous benefits, such as promoting reusability, and addressing cases involving data privacy and societal concerns; on the flip side, current two-stage MTL methods come with the cost of introducing a substantial number of additional parameters. In this work, we address this issue by leveraging the usefulness of linearly scaling the output representations of source adapters for transfer learning. We introduce ScaLearn, a simple and highly parameter-efficient two-stage MTL method that capitalizes on the knowledge of the source tasks by learning a minimal set of scaling parameters that enable effective knowledge transfer to a target task. Our experiments on three benchmarks (GLUE, SuperGLUE, and HumSet) show that our ScaLearn, in addition to facilitating the benefits of two-stage MTL, consistently outperforms strong baselines with only a small number of transfer parameters - roughly 0.35% of those of AdapterFusion. Remarkably, we observe that ScaLearn maintains its strong abilities even when further reducing parameters through uniform scaling and layer-sharing, achieving similarly competitive results with only $8$ transfer parameters for each target task. Our proposed approach thus demonstrates the power of simple scaling as a promise for more efficient task transfer.

{{</citation>}}


### (22/165) Graph Isomorphic Networks for Assessing Reliability of the Medium-Voltage Grid (Charlotte Cambier van Nooten et al., 2023)

{{<citation>}}

Charlotte Cambier van Nooten, Tom van de Poll, Sonja Füllhase, Jacco Heres, Tom Heskes, Yuliya Shapovalova. (2023)  
**Graph Isomorphic Networks for Assessing Reliability of the Medium-Voltage Grid**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.01181v2)  

---


**ABSTRACT**  
Ensuring electricity grid reliability becomes increasingly challenging with the shift towards renewable energy and declining conventional capacities. Distribution System Operators (DSOs) aim to achieve grid reliability by verifying the n-1 principle, ensuring continuous operation in case of component failure. Electricity networks' complex graph-based data holds crucial information for n-1 assessment: graph structure and data about stations/cables. Unlike traditional machine learning methods, Graph Neural Networks (GNNs) directly handle graph-structured data. This paper proposes using Graph Isomorphic Networks (GINs) for n-1 assessments in medium voltage grids. The GIN framework is designed to generalise to unseen grids and utilise graph structure and data about stations/cables. The proposed GIN approach demonstrates faster and more reliable grid assessments than a traditional mathematical optimisation approach, reducing prediction times by approximately a factor of 1000. The findings offer a promising approach to address computational challenges and enhance the reliability and efficiency of energy grid assessments.

{{</citation>}}


### (23/165) DINE: Dimensional Interpretability of Node Embeddings (Simone Piaggesi et al., 2023)

{{<citation>}}

Simone Piaggesi, Megha Khosla, André Panisson, Avishek Anand. (2023)  
**DINE: Dimensional Interpretability of Node Embeddings**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2310.01162v1)  

---


**ABSTRACT**  
Graphs are ubiquitous due to their flexibility in representing social and technological systems as networks of interacting elements. Graph representation learning methods, such as node embeddings, are powerful approaches to map nodes into a latent vector space, allowing their use for various graph tasks. Despite their success, only few studies have focused on explaining node embeddings locally. Moreover, global explanations of node embeddings remain unexplored, limiting interpretability and debugging potentials. We address this gap by developing human-understandable explanations for dimensions in node embeddings. Towards that, we first develop new metrics that measure the global interpretability of embedding vectors based on the marginal contribution of the embedding dimensions to predicting graph structure. We say that an embedding dimension is more interpretable if it can faithfully map to an understandable sub-structure in the input graph - like community structure. Having observed that standard node embeddings have low interpretability, we then introduce DINE (Dimension-based Interpretable Node Embedding), a novel approach that can retrofit existing node embeddings by making them more interpretable without sacrificing their task performance. We conduct extensive experiments on synthetic and real-world graphs and show that we can simultaneously learn highly interpretable node embeddings with effective performance in link prediction.

{{</citation>}}


### (24/165) RRR-Net: Reusing, Reducing, and Recycling a Deep Backbone Network (Haozhe Sun et al., 2023)

{{<citation>}}

Haozhe Sun, Isabelle Guyon, Felix Mohr, Hedi Tabia. (2023)  
**RRR-Net: Reusing, Reducing, and Recycling a Deep Backbone Network**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2310.01157v1)  

---


**ABSTRACT**  
It has become mainstream in computer vision and other machine learning domains to reuse backbone networks pre-trained on large datasets as preprocessors. Typically, the last layer is replaced by a shallow learning machine of sorts; the newly-added classification head and (optionally) deeper layers are fine-tuned on a new task. Due to its strong performance and simplicity, a common pre-trained backbone network is ResNet152.However, ResNet152 is relatively large and induces inference latency. In many cases, a compact and efficient backbone with similar performance would be preferable over a larger, slower one. This paper investigates techniques to reuse a pre-trained backbone with the objective of creating a smaller and faster model. Starting from a large ResNet152 backbone pre-trained on ImageNet, we first reduce it from 51 blocks to 5 blocks, reducing its number of parameters and FLOPs by more than 6 times, without significant performance degradation. Then, we split the model after 3 blocks into several branches, while preserving the same number of parameters and FLOPs, to create an ensemble of sub-networks to improve performance. Our experiments on a large benchmark of $40$ image classification datasets from various domains suggest that our techniques match the performance (if not better) of ``classical backbone fine-tuning'' while achieving a smaller model size and faster inference speed.

{{</citation>}}


### (25/165) NP$^2$L: Negative Pseudo Partial Labels Extraction for Graph Neural Networks (Xinjie Shen et al., 2023)

{{<citation>}}

Xinjie Shen, Danyang Wu, Jitao Lu, Junjie Liang, Jin Xu, Feiping Nie. (2023)  
**NP$^2$L: Negative Pseudo Partial Labels Extraction for Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-SI, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.01098v1)  

---


**ABSTRACT**  
How to utilize the pseudo labels has always been a research hotspot in machine learning. However, most methods use pseudo labels as supervised training, and lack of valid assessing for their accuracy. Moreover, applications of pseudo labels in graph neural networks (GNNs) oversee the difference between graph learning and other machine learning tasks such as message passing mechanism. Aiming to address the first issue, we found through a large number of experiments that the pseudo labels are more accurate if they are selected by not overlapping partial labels and defined as negative node pairs relations. Therefore, considering the extraction based on pseudo and partial labels, negative edges are constructed between two nodes by the negative pseudo partial labels extraction (NP$^2$E) module. With that, a signed graph are built containing highly accurate pseudo labels information from the original graph, which effectively assists GNN in learning at the message-passing level, provide one solution to the second issue. Empirical results about link prediction and node classification tasks on several benchmark datasets demonstrate the effectiveness of our method. State-of-the-art performance is achieved on the both tasks.

{{</citation>}}


### (26/165) Linear attention is (maybe) all you need (to understand transformer optimization) (Kwangjun Ahn et al., 2023)

{{<citation>}}

Kwangjun Ahn, Xiang Cheng, Minhak Song, Chulhee Yun, Ali Jadbabaie, Suvrit Sra. (2023)  
**Linear attention is (maybe) all you need (to understand transformer optimization)**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, math-OC  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.01082v1)  

---


**ABSTRACT**  
Transformer training is notoriously difficult, requiring a careful design of optimizers and use of various heuristics. We make progress towards understanding the subtleties of training transformers by carefully studying a simple yet canonical linearized shallow transformer model. Specifically, we train linear transformers to solve regression tasks, inspired by J. von Oswald et al. (ICML 2023), and K. Ahn et al. (NeurIPS 2023). Most importantly, we observe that our proposed linearized models can reproduce several prominent aspects of transformer training dynamics. Consequently, the results obtained in this paper suggest that a simple linearized transformer model could actually be a valuable, realistic abstraction for understanding transformer optimization.

{{</citation>}}


### (27/165) Efficient Algorithms for the CCA Family: Unconstrained Objectives with Unbiased Gradients (James Chapman et al., 2023)

{{<citation>}}

James Chapman, Ana Lawry Aguila, Lennie Wells. (2023)  
**Efficient Algorithms for the CCA Family: Unconstrained Objectives with Unbiased Gradients**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, stat-ML  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2310.01012v1)  

---


**ABSTRACT**  
The Canonical Correlation Analysis (CCA) family of methods is foundational in multi-view learning. Regularised linear CCA methods can be seen to generalise Partial Least Squares (PLS) and unified with a Generalized Eigenvalue Problem (GEP) framework. However, classical algorithms for these linear methods are computationally infeasible for large-scale data. Extensions to Deep CCA show great promise, but current training procedures are slow and complicated. First we propose a novel unconstrained objective that characterizes the top subspace of GEPs. Our core contribution is a family of fast algorithms for stochastic PLS, stochastic CCA, and Deep CCA, simply obtained by applying stochastic gradient descent (SGD) to the corresponding CCA objectives. These methods show far faster convergence and recover higher correlations than the previous state-of-the-art on all standard CCA and Deep CCA benchmarks. This speed allows us to perform a first-of-its-kind PLS analysis of an extremely large biomedical dataset from the UK Biobank, with over 33,000 individuals and 500,000 variants. Finally, we not only match the performance of `CCA-family' Self-Supervised Learning (SSL) methods on CIFAR-10 and CIFAR-100 with minimal hyper-parameter tuning, but also establish the first solid theoretical links to classical CCA, laying the groundwork for future insights.

{{</citation>}}


### (28/165) Variance-Aware Regret Bounds for Stochastic Contextual Dueling Bandits (Qiwei Di et al., 2023)

{{<citation>}}

Qiwei Di, Tao Jin, Yue Wu, Heyang Zhao, Farzad Farnoud, Quanquan Gu. (2023)  
**Variance-Aware Regret Bounds for Stochastic Contextual Dueling Bandits**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, math-OC, stat-ML  
Keywords: GLM  
[Paper Link](http://arxiv.org/abs/2310.00968v1)  

---


**ABSTRACT**  
Dueling bandits is a prominent framework for decision-making involving preferential feedback, a valuable feature that fits various applications involving human interaction, such as ranking, information retrieval, and recommendation systems. While substantial efforts have been made to minimize the cumulative regret in dueling bandits, a notable gap in the current research is the absence of regret bounds that account for the inherent uncertainty in pairwise comparisons between the dueling arms. Intuitively, greater uncertainty suggests a higher level of difficulty in the problem. To bridge this gap, this paper studies the problem of contextual dueling bandits, where the binary comparison of dueling arms is generated from a generalized linear model (GLM). We propose a new SupLinUCB-type algorithm that enjoys computational efficiency and a variance-aware regret bound $\tilde O\big(d\sqrt{\sum_{t=1}^T\sigma_t^2} + d\big)$, where $\sigma_t$ is the variance of the pairwise comparison in round $t$, $d$ is the dimension of the context vectors, and $T$ is the time horizon. Our regret bound naturally aligns with the intuitive expectation in scenarios where the comparison is deterministic, the algorithm only suffers from an $\tilde O(d)$ regret. We perform empirical experiments on synthetic data to confirm the advantage of our method over previous variance-agnostic algorithms.

{{</citation>}}


### (29/165) Distilling Influences to Mitigate Prediction Churn in Graph Neural Networks (Andreas Roth et al., 2023)

{{<citation>}}

Andreas Roth, Thomas Liebig. (2023)  
**Distilling Influences to Mitigate Prediction Churn in Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Graph Neural Network, Graph Neural Networks, Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2310.00946v1)  

---


**ABSTRACT**  
Models with similar performances exhibit significant disagreement in the predictions of individual samples, referred to as prediction churn. Our work explores this phenomenon in graph neural networks by investigating differences between models differing only in their initializations in their utilized features for predictions. We propose a novel metric called Influence Difference (ID) to quantify the variation in reasons used by nodes across models by comparing their influence distribution. Additionally, we consider the differences between nodes with a stable and an unstable prediction, positing that both equally utilize different reasons and thus provide a meaningful gradient signal to closely match two models even when the predictions for nodes are similar. Based on our analysis, we propose to minimize this ID in Knowledge Distillation, a domain where a new model should closely match an established one. As an efficient approximation, we introduce DropDistillation (DD) that matches the output for a graph perturbed by edge deletions. Our empirical evaluation of six benchmark datasets for node classification validates the differences in utilized features. DD outperforms previous methods regarding prediction stability and overall performance in all considered Knowledge Distillation experiments.

{{</citation>}}


### (30/165) Understanding Transferable Representation Learning and Zero-shot Transfer in CLIP (Zixiang Chen et al., 2023)

{{<citation>}}

Zixiang Chen, Yihe Deng, Yuanzhi Li, Quanquan Gu. (2023)  
**Understanding Transferable Representation Learning and Zero-shot Transfer in CLIP**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, stat-ML  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2310.00927v1)  

---


**ABSTRACT**  
Multi-modal learning has become increasingly popular due to its ability to leverage information from different data sources (e.g., text and images) to improve the model performance. Recently, CLIP has emerged as an effective approach that employs vision-language contrastive pretraining to learn joint image and text representations and exhibits remarkable performance in zero-shot learning and text-guided natural image generation. Despite the huge practical success of CLIP, its theoretical understanding remains elusive. In this paper, we formally study transferrable representation learning underlying CLIP and demonstrate how features from different modalities get aligned. We also analyze its zero-shot transfer performance on the downstream tasks. Inspired by our analysis, we propose a new CLIP-type approach, which achieves better performance than CLIP and other state-of-the-art methods on benchmark datasets.

{{</citation>}}


### (31/165) Integration of Graph Neural Network and Neural-ODEs for Tumor Dynamic Prediction (Omid Bazgir et al., 2023)

{{<citation>}}

Omid Bazgir, Zichen Wang, Marc Hafner, James Lu. (2023)  
**Integration of Graph Neural Network and Neural-ODEs for Tumor Dynamic Prediction**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2310.00926v1)  

---


**ABSTRACT**  
In anti-cancer drug development, a major scientific challenge is disentangling the complex relationships between high-dimensional genomics data from patient tumor samples, the corresponding tumor's organ of origin, the drug targets associated with given treatments and the resulting treatment response. Furthermore, to realize the aspirations of precision medicine in identifying and adjusting treatments for patients depending on the therapeutic response, there is a need for building tumor dynamic models that can integrate both longitudinal tumor size as well as multimodal, high-content data. In this work, we take a step towards enhancing personalized tumor dynamic predictions by proposing a heterogeneous graph encoder that utilizes a bipartite Graph Convolutional Neural network (GCN) combined with Neural Ordinary Differential Equations (Neural-ODEs). We applied the methodology to a large collection of patient-derived xenograft (PDX) data, spanning a wide variety of treatments (as well as their combinations) on tumors that originated from a number of different organs. We first show that the methodology is able to discover a tumor dynamic model that significantly improves upon an empirical model which is in current use. Additionally, we show that the graph encoder is able to effectively utilize multimodal data to enhance tumor predictions. Our findings indicate that the methodology holds significant promise and offers potential applications in pre-clinical settings.

{{</citation>}}


### (32/165) DataInf: Efficiently Estimating Data Influence in LoRA-tuned LLMs and Diffusion Models (Yongchan Kwon et al., 2023)

{{<citation>}}

Yongchan Kwon, Eric Wu, Kevin Wu, James Zou. (2023)  
**DataInf: Efficiently Estimating Data Influence in LoRA-tuned LLMs and Diffusion Models**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: AI, BERT  
[Paper Link](http://arxiv.org/abs/2310.00902v1)  

---


**ABSTRACT**  
Quantifying the impact of training data points is crucial for understanding the outputs of machine learning models and for improving the transparency of the AI pipeline. The influence function is a principled and popular data attribution method, but its computational cost often makes it challenging to use. This issue becomes more pronounced in the setting of large language models and text-to-image models. In this work, we propose DataInf, an efficient influence approximation method that is practical for large-scale generative AI models. Leveraging an easy-to-compute closed-form expression, DataInf outperforms existing influence computation algorithms in terms of computational and memory efficiency. Our theoretical analysis shows that DataInf is particularly well-suited for parameter-efficient fine-tuning techniques such as LoRA. Through systematic empirical evaluations, we show that DataInf accurately approximates influence scores and is orders of magnitude faster than existing methods. In applications to RoBERTa-large, Llama-2-13B-chat, and stable-diffusion-v1.5 models, DataInf effectively identifies the most influential fine-tuning examples better than other approximate influence scores. Moreover, it can help to identify which data points are mislabeled.

{{</citation>}}


### (33/165) Organized Event Participant Prediction Enhanced by Social Media Retweeting Data (Yihong Zhang et al., 2023)

{{<citation>}}

Yihong Zhang, Takahiro Hara. (2023)  
**Organized Event Participant Prediction Enhanced by Social Media Retweeting Data**  

---
Primary Category: cs.LG  
Categories: H-3-3, cs-IR, cs-LG, cs.LG  
Keywords: Social Media  
[Paper Link](http://arxiv.org/abs/2310.00896v1)  

---


**ABSTRACT**  
Nowadays, many platforms on the Web offer organized events, allowing users to be organizers or participants. For such platforms, it is beneficial to predict potential event participants. Existing work on this problem tends to borrow recommendation techniques. However, compared to e-commerce items and purchases, events and participation are usually of a much smaller frequency, and the data may be insufficient to learn an accurate model. In this paper, we propose to utilize social media retweeting activity data to enhance the learning of event participant prediction models. We create a joint knowledge graph to bridge the social media and the target domain, assuming that event descriptions and tweets are written in the same language. Furthermore, we propose a learning model that utilizes retweeting information for the target domain prediction more effectively. We conduct comprehensive experiments in two scenarios with real-world data. In each scenario, we set up training data of different sizes, as well as warm and cold test cases. The evaluation results show that our approach consistently outperforms several baseline models, especially with the warm test cases, and when target domain data is limited.

{{</citation>}}


### (34/165) Deep Neural Networks Tend To Extrapolate Predictably (Katie Kang et al., 2023)

{{<citation>}}

Katie Kang, Amrith Setlur, Claire Tomlin, Sergey Levine. (2023)  
**Deep Neural Networks Tend To Extrapolate Predictably**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2310.00873v1)  

---


**ABSTRACT**  
Conventional wisdom suggests that neural network predictions tend to be unpredictable and overconfident when faced with out-of-distribution (OOD) inputs. Our work reassesses this assumption for neural networks with high-dimensional inputs. Rather than extrapolating in arbitrary ways, we observe that neural network predictions often tend towards a constant value as input data becomes increasingly OOD. Moreover, we find that this value often closely approximates the optimal constant solution (OCS), i.e., the prediction that minimizes the average loss over the training data without observing the input. We present results showing this phenomenon across 8 datasets with different distributional shifts (including CIFAR10-C and ImageNet-R, S), different loss functions (cross entropy, MSE, and Gaussian NLL), and different architectures (CNNs and transformers). Furthermore, we present an explanation for this behavior, which we first validate empirically and then study theoretically in a simplified setting involving deep homogeneous networks with ReLU activations. Finally, we show how one can leverage our insights in practice to enable risk-sensitive decision-making in the presence of OOD inputs.

{{</citation>}}


### (35/165) Use Your INSTINCT: INSTruction optimization usIng Neural bandits Coupled with Transformers (Xiaoqiang Lin et al., 2023)

{{<citation>}}

Xiaoqiang Lin, Zhaoxuan Wu, Zhongxiang Dai, Wenyang Hu, Yao Shu, See-Kiong Ng, Patrick Jaillet, Bryan Kian Hsiang Low. (2023)  
**Use Your INSTINCT: INSTruction optimization usIng Neural bandits Coupled with Transformers**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: ChatGPT, GPT, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.02905v1)  

---


**ABSTRACT**  
Large language models (LLMs) have shown remarkable instruction-following capabilities and achieved impressive performances in various applications. However, the performances of LLMs depend heavily on the instructions given to them, which are typically manually tuned with substantial human efforts. Recent work has used the query-efficient Bayesian optimization (BO) algorithm to automatically optimize the instructions given to black-box LLMs. However, BO usually falls short when optimizing highly sophisticated (e.g., high-dimensional) objective functions, such as the functions mapping an instruction to the performance of an LLM. This is mainly due to the limited expressive power of the Gaussian process (GP) model which is used by BO as a surrogate to model the objective function. Meanwhile, it has been repeatedly shown that neural networks (NNs), especially pre-trained transformers, possess strong expressive power and can model highly complex functions. So, we adopt a neural bandit algorithm which replaces the GP in BO by an NN surrogate to optimize instructions for black-box LLMs. More importantly, the neural bandit algorithm allows us to naturally couple the NN surrogate with the hidden representation learned by a pre-trained transformer (i.e., an open-source LLM), which significantly boosts its performance. These motivate us to propose our INSTruction optimization usIng Neural bandits Coupled with Transformers} (INSTINCT) algorithm. We perform instruction optimization for ChatGPT and use extensive experiments to show that our INSTINCT consistently outperforms the existing methods in different tasks, such as in various instruction induction tasks and the task of improving the zero-shot chain-of-thought instruction.

{{</citation>}}


## cs.CL (45)



### (36/165) Zero-Shot Continuous Prompt Transfer: Generalizing Task Semantics Across Language Models (Zijun Wu et al., 2023)

{{<citation>}}

Zijun Wu, Yongkang Wu, Lili Mou. (2023)  
**Zero-Shot Continuous Prompt Transfer: Generalizing Task Semantics Across Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, NLP, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2310.01691v1)  

---


**ABSTRACT**  
Prompt tuning in natural language processing (NLP) has become an increasingly popular method for adapting large language models to specific tasks. However, the transferability of these prompts, especially continuous prompts, between different models remains a challenge. In this work, we propose a zero-shot continuous prompt transfer method, where source prompts are encoded into relative space and the corresponding target prompts are searched for transferring to target models. Experimental results confirm the effectiveness of our method, showing that 'task semantics' in continuous prompts can be generalized across various language models. Moreover, we find that combining 'task semantics' from multiple source models can further enhance the generalizability of transfer.

{{</citation>}}


### (37/165) A Review of Digital Learning Environments for Teaching Natural Language Processing in K-12 Education (Xiaoyi Tian et al., 2023)

{{<citation>}}

Xiaoyi Tian, Kristy Elizabeth Boyer. (2023)  
**A Review of Digital Learning Environments for Teaching Natural Language Processing in K-12 Education**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-HC, cs.CL  
Keywords: AI, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2310.01603v1)  

---


**ABSTRACT**  
Natural Language Processing (NLP) plays a significant role in our daily lives and has become an essential part of Artificial Intelligence (AI) education in K-12. As children grow up with NLP-powered applications, it is crucial to introduce NLP concepts to them, fostering their understanding of language processing, language generation, and ethical implications of AI and NLP. This paper presents a comprehensive review of digital learning environments for teaching NLP in K-12. Specifically, it explores existing digital learning tools, discusses how they support specific NLP tasks and procedures, and investigates their explainability and evaluation results in educational contexts. By examining the strengths and limitations of these tools, this literature review sheds light on the current state of NLP learning tools in K-12 education. It aims to guide future research efforts to refine existing tools, develop new ones, and explore more effective and inclusive strategies for integrating NLP into K-12 educational contexts.

{{</citation>}}


### (38/165) Making Retrieval-Augmented Language Models Robust to Irrelevant Context (Ori Yoran et al., 2023)

{{<citation>}}

Ori Yoran, Tomer Wolfson, Ori Ram, Jonathan Berant. (2023)  
**Making Retrieval-Augmented Language Models Robust to Irrelevant Context**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, NLI  
[Paper Link](http://arxiv.org/abs/2310.01558v1)  

---


**ABSTRACT**  
Retrieval-augmented language models (RALMs) hold promise to produce language understanding systems that are are factual, efficient, and up-to-date. An important desideratum of RALMs, is that retrieved information helps model performance when it is relevant, and does not harm performance when it is not. This is particularly important in multi-hop reasoning scenarios, where misuse of irrelevant evidence can lead to cascading errors. However, recent work has shown that retrieval augmentation can sometimes have a negative effect on performance. In this work, we present a thorough analysis on five open-domain question answering benchmarks, characterizing cases when retrieval reduces accuracy. We then propose two methods to mitigate this issue. First, a simple baseline that filters out retrieved passages that do not entail question-answer pairs according to a natural language inference (NLI) model. This is effective in preventing performance reduction, but at a cost of also discarding relevant passages. Thus, we propose a method for automatically generating data to fine-tune the language model to properly leverage retrieved passages, using a mix of relevant and irrelevant contexts at training time. We empirically show that even 1,000 examples suffice to train the model to be robust to irrelevant contexts while maintaining high performance on examples with relevant ones.

{{</citation>}}


### (39/165) It's MBR All the Way Down: Modern Generation Techniques Through the Lens of Minimum Bayes Risk (Amanda Bertsch et al., 2023)

{{<citation>}}

Amanda Bertsch, Alex Xie, Graham Neubig, Matthew R. Gormley. (2023)  
**It's MBR All the Way Down: Modern Generation Techniques Through the Lens of Minimum Bayes Risk**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2310.01387v1)  

---


**ABSTRACT**  
Minimum Bayes Risk (MBR) decoding is a method for choosing the outputs of a machine learning system based not on the output with the highest probability, but the output with the lowest risk (expected error) among multiple candidates. It is a simple but powerful method: for an additional cost at inference time, MBR provides reliable several-point improvements across metrics for a wide variety of tasks without any additional data or training. Despite this, MBR is not frequently applied in NLP works, and knowledge of the method itself is limited. We first provide an introduction to the method and the recent literature. We show that several recent methods that do not reference MBR can be written as special cases of MBR; this reformulation provides additional theoretical justification for the performance of these methods, explaining some results that were previously only empirical. We provide theoretical and empirical results about the effectiveness of various MBR variants and make concrete recommendations for the application of MBR in NLP models, including future directions in this area.

{{</citation>}}


### (40/165) Who is ChatGPT? Benchmarking LLMs' Psychological Portrayal Using PsychoBench (Jen-tse Huang et al., 2023)

{{<citation>}}

Jen-tse Huang, Wenxuan Wang, Eric John Li, Man Ho Lam, Shujie Ren, Youliang Yuan, Wenxiang Jiao, Zhaopeng Tu, Michael R. Lyu. (2023)  
**Who is ChatGPT? Benchmarking LLMs' Psychological Portrayal Using PsychoBench**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, GPT-4, LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2310.01386v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have recently showcased their remarkable capacities, not only in natural language processing tasks but also across diverse domains such as clinical medicine, legal consultation, and education. LLMs become more than mere applications, evolving into assistants capable of addressing diverse user requests. This narrows the distinction between human beings and artificial intelligence agents, raising intriguing questions regarding the potential manifestation of personalities, temperaments, and emotions within LLMs. In this paper, we propose a framework, PsychoBench, for evaluating diverse psychological aspects of LLMs. Comprising thirteen scales commonly used in clinical psychology, PsychoBench further classifies these scales into four distinct categories: personality traits, interpersonal relationships, motivational tests, and emotional abilities. Our study examines five popular models, namely \texttt{text-davinci-003}, ChatGPT, GPT-4, LLaMA-2-7b, and LLaMA-2-13b. Additionally, we employ a jailbreak approach to bypass the safety alignment protocols and test the intrinsic natures of LLMs. We have made PsychoBench openly accessible via \url{https://github.com/CUHK-ARISE/PsychoBench}.

{{</citation>}}


### (41/165) Compressing LLMs: The Truth is Rarely Pure and Never Simple (Ajay Jaiswal et al., 2023)

{{<citation>}}

Ajay Jaiswal, Zhe Gan, Xianzhi Du, Bowen Zhang, Zhangyang Wang, Yinfei Yang. (2023)  
**Compressing LLMs: The Truth is Rarely Pure and Never Simple**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.01382v1)  

---


**ABSTRACT**  
Despite their remarkable achievements, modern Large Language Models (LLMs) encounter exorbitant computational and memory footprints. Recently, several works have shown significant success in training-free and data-free compression (pruning and quantization) of LLMs achieving 50-60% sparsity and reducing the bit-width down to 3 or 4 bits per weight, with negligible perplexity degradation over the uncompressed baseline. As recent research efforts are focused on developing increasingly sophisticated compression methods, our work takes a step back, and re-evaluates the effectiveness of existing SoTA compression methods, which rely on a fairly simple and widely questioned metric, perplexity (even for dense LLMs). We introduce Knowledge-Intensive Compressed LLM BenchmarK (LLM-KICK), a collection of carefully-curated tasks to re-define the evaluation protocol for compressed LLMs, which have significant alignment with their dense counterparts, and perplexity fail to capture subtle change in their true capabilities. LLM-KICK unveils many favorable merits and unfortunate plights of current SoTA compression methods: all pruning methods suffer significant performance degradation, sometimes at trivial sparsity ratios (e.g., 25-30%), and fail for N:M sparsity on knowledge-intensive tasks; current quantization methods are more successful than pruning; yet, pruned LLMs even at $\geq 50$% sparsity are robust in-context retrieval and summarization systems; among others. LLM-KICK is designed to holistically access compressed LLMs' ability for language understanding, reasoning, generation, in-context retrieval, in-context summarization, etc. We hope our study can foster the development of better LLM compression methods. All our related codes are planed to be open-sourced.

{{</citation>}}


### (42/165) UltraFeedback: Boosting Language Models with High-quality Feedback (Ganqu Cui et al., 2023)

{{<citation>}}

Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Wei Zhu, Yuan Ni, Guotong Xie, Zhiyuan Liu, Maosong Sun. (2023)  
**UltraFeedback: Boosting Language Models with High-quality Feedback**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2310.01377v1)  

---


**ABSTRACT**  
Reinforcement learning from human feedback (RLHF) has become a pivot technique in aligning large language models (LLMs) with human preferences. In RLHF practice, preference data plays a crucial role in bridging human proclivity and LLMs. However, the scarcity of diverse, naturalistic datasets of human preferences on LLM outputs at scale poses a great challenge to RLHF as well as feedback learning research within the open-source community. Current preference datasets, either proprietary or limited in size and prompt variety, result in limited RLHF adoption in open-source models and hinder further exploration. In this study, we propose ULTRAFEEDBACK, a large-scale, high-quality, and diversified preference dataset designed to overcome these limitations and foster RLHF development. To create ULTRAFEEDBACK, we compile a diverse array of instructions and models from multiple sources to produce comparative data. We meticulously devise annotation instructions and employ GPT-4 to offer detailed feedback in both numerical and textual forms. ULTRAFEEDBACK establishes a reproducible and expandable preference data construction pipeline, serving as a solid foundation for future RLHF and feedback learning research. Utilizing ULTRAFEEDBACK, we train various models to demonstrate its effectiveness, including the reward model UltraRM, chat language model UltraLM-13B-PPO, and critique model UltraCM. Experimental results indicate that our models outperform existing open-source models, achieving top performance across multiple benchmarks. Our data and models are available at https://github.com/thunlp/UltraFeedback.

{{</citation>}}


### (43/165) Improving Dialogue Management: Quality Datasets vs Models (Miguel Ángel Medina-Ramírez et al., 2023)

{{<citation>}}

Miguel Ángel Medina-Ramírez, Cayetano Guerra-Artal, Mario Hernández-Tejera. (2023)  
**Improving Dialogue Management: Quality Datasets vs Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2310.01339v1)  

---


**ABSTRACT**  
Task-oriented dialogue systems (TODS) have become crucial for users to interact with machines and computers using natural language. One of its key components is the dialogue manager, which guides the conversation towards a good goal for the user by providing the best possible response. Previous works have proposed rule-based systems (RBS), reinforcement learning (RL), and supervised learning (SL) as solutions for the correct dialogue management; in other words, select the best response given input by the user. However, this work argues that the leading cause of DMs not achieving maximum performance resides in the quality of the datasets rather than the models employed thus far; this means that dataset errors, like mislabeling, originate a large percentage of failures in dialogue management. We studied the main errors in the most widely used datasets, Multiwoz 2.1 and SGD, to demonstrate this hypothesis. To do this, we have designed a synthetic dialogue generator to fully control the amount and type of errors introduced in the dataset. Using this generator, we demonstrated that errors in the datasets contribute proportionally to the performance of the models

{{</citation>}}


### (44/165) LLM Lies: Hallucinations are not Bugs, but Features as Adversarial Examples (Jia-Yu Yao et al., 2023)

{{<citation>}}

Jia-Yu Yao, Kun-Peng Ning, Zhen-Hui Liu, Mu-Nan Ning, Li Yuan. (2023)  
**LLM Lies: Hallucinations are not Bugs, but Features as Adversarial Examples**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-3.5, LLaMA, Language Model, PaLM  
[Paper Link](http://arxiv.org/abs/2310.01469v2)  

---


**ABSTRACT**  
Large Language Models (LLMs), including GPT-3.5, LLaMA, and PaLM, seem to be knowledgeable and able to adapt to many tasks. However, we still can not completely trust their answer, since LLMs suffer from hallucination--fabricating non-existent facts to cheat users without perception. And the reasons for their existence and pervasiveness remain unclear. In this paper, we demonstrate that non-sense prompts composed of random tokens can also elicit the LLMs to respond with hallucinations. This phenomenon forces us to revisit that hallucination may be another view of adversarial examples, and it shares similar features with conventional adversarial examples as the basic feature of LLMs. Therefore, we formalize an automatic hallucination triggering method as the hallucination attack in an adversarial way. Finally, we explore basic feature of attacked adversarial prompts and propose a simple yet effective defense strategy. Our code is released on GitHub.

{{</citation>}}


### (45/165) The Entity-Deduction Arena: A playground for probing the conversational reasoning and planning capabilities of LLMs (Yizhe Zhang et al., 2023)

{{<citation>}}

Yizhe Zhang, Jiarui Lu, Navdeep Jaitly. (2023)  
**The Entity-Deduction Arena: A playground for probing the conversational reasoning and planning capabilities of LLMs**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-4, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.01468v2)  

---


**ABSTRACT**  
Large language models (LLMs) are effective at answering questions that are clearly asked. However, when faced with ambiguous queries they can act unpredictably and produce incorrect outputs. This underscores the need for the development of intelligent agents capable of asking clarification questions to resolve ambiguities effectively. This capability requires complex understanding, state tracking, reasoning and planning over multiple conversational turns. However, directly measuring this can be challenging. In this paper, we offer a surrogate problem which assesses an LLMs's capability to deduce an entity unknown to itself, but revealed to a judge, by asking the judge a series of queries. This entity-deducing game can serve as an evaluation framework to probe the conversational reasoning and planning capabilities of language models. We systematically evaluate various LLMs and discover significant differences in their performance on this task. We find that strong LLMs like GPT-4 outperform human players by a large margin. We further employ Behavior Cloning (BC) to examine whether a weaker model is capable of imitating a stronger model and generalizing to data or domains, using only the demonstrations from a stronger model. We finally propose to use Reinforcement Learning to enhance reasoning and planning capacity of Vicuna models through episodes of game playing, which lead to significant performance improvement. We hope that this problem offers insights into how autonomous agents could be trained to behave more intelligently in ambiguous circumstances.

{{</citation>}}


### (46/165) BTR: Binary Token Representations for Efficient Retrieval Augmented Language Models (Qingqing Cao et al., 2023)

{{<citation>}}

Qingqing Cao, Sewon Min, Yizhong Wang, Hannaneh Hajishirzi. (2023)  
**BTR: Binary Token Representations for Efficient Retrieval Augmented Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2310.01329v1)  

---


**ABSTRACT**  
Retrieval augmentation addresses many critical problems in large language models such as hallucination, staleness, and privacy leaks. However, running retrieval-augmented language models (LMs) is slow and difficult to scale due to processing large amounts of retrieved text. We introduce binary token representations (BTR), which use 1-bit vectors to precompute every token in passages, significantly reducing computation during inference. Despite the potential loss of accuracy, our new calibration techniques and training objectives restore performance. Combined with offline and runtime compression, this only requires 127GB of disk space for encoding 3 billion tokens in Wikipedia. Our experiments show that on five knowledge-intensive NLP tasks, BTR accelerates state-of-the-art inference by up to 4x and reduces storage by over 100x while maintaining over 95% task performance.

{{</citation>}}


### (47/165) FedBPT: Efficient Federated Black-box Prompt Tuning for Large Language Models (Jingwei Sun et al., 2023)

{{<citation>}}

Jingwei Sun, Ziyue Xu, Hongxu Yin, Dong Yang, Daguang Xu, Yiran Chen, Holger R. Roth. (2023)  
**FedBPT: Efficient Federated Black-box Prompt Tuning for Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2310.01467v1)  

---


**ABSTRACT**  
Pre-trained language models (PLM) have revolutionized the NLP landscape, achieving stellar performances across diverse tasks. These models, while benefiting from vast training data, often require fine-tuning on specific data to cater to distinct downstream tasks. However, this data adaptation process has inherent security and privacy concerns, primarily when leveraging user-generated, device-residing data. Federated learning (FL) provides a solution, allowing collaborative model fine-tuning without centralized data collection. However, applying FL to finetune PLMs is hampered by challenges, including restricted model parameter access, high computational requirements, and communication overheads. This paper introduces Federated Black-box Prompt Tuning (FedBPT), a framework designed to address these challenges. FedBPT does not require the clients to access the model parameters. By focusing on training optimal prompts and utilizing gradient-free optimization methods, FedBPT reduces the number of exchanged variables, boosts communication efficiency, and minimizes computational and storage costs. Experiments highlight the framework's ability to drastically cut communication and memory costs while maintaining competitive performance. Ultimately, FedBPT presents a promising solution for efficient, privacy-preserving fine-tuning of PLM in the age of large language models.

{{</citation>}}


### (48/165) On the Generalization of Training-based ChatGPT Detection Methods (Han Xu et al., 2023)

{{<citation>}}

Han Xu, Jie Ren, Pengfei He, Shenglai Zeng, Yingqian Cui, Amy Liu, Hui Liu, Jiliang Tang. (2023)  
**On the Generalization of Training-based ChatGPT Detection Methods**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2310.01307v2)  

---


**ABSTRACT**  
ChatGPT is one of the most popular language models which achieve amazing performance on various natural language tasks. Consequently, there is also an urgent need to detect the texts generated ChatGPT from human written. One of the extensively studied methods trains classification models to distinguish both. However, existing studies also demonstrate that the trained models may suffer from distribution shifts (during test), i.e., they are ineffective to predict the generated texts from unseen language tasks or topics. In this work, we aim to have a comprehensive investigation on these methods' generalization behaviors under distribution shift caused by a wide range of factors, including prompts, text lengths, topics, and language tasks. To achieve this goal, we first collect a new dataset with human and ChatGPT texts, and then we conduct extensive studies on the collected dataset. Our studies unveil insightful findings which provide guidance for developing future methodologies or data collection strategies for ChatGPT detection.

{{</citation>}}


### (49/165) Generating Explanations in Medical Question-Answering by Expectation Maximization Inference over Evidence (Wei Sun et al., 2023)

{{<citation>}}

Wei Sun, Mingxiao Li, Damien Sileo, Jesse Davis, Marie-Francine Moens. (2023)  
**Generating Explanations in Medical Question-Answering by Expectation Maximization Inference over Evidence**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: QA, Question Answering, Rouge  
[Paper Link](http://arxiv.org/abs/2310.01299v1)  

---


**ABSTRACT**  
Medical Question Answering~(medical QA) systems play an essential role in assisting healthcare workers in finding answers to their questions. However, it is not sufficient to merely provide answers by medical QA systems because users might want explanations, that is, more analytic statements in natural language that describe the elements and context that support the answer. To do so, we propose a novel approach for generating natural language explanations for answers predicted by medical QA systems. As high-quality medical explanations require additional medical knowledge, so that our system extract knowledge from medical textbooks to enhance the quality of explanations during the explanation generation process. Concretely, we designed an expectation-maximization approach that makes inferences about the evidence found in these texts, offering an efficient way to focus attention on lengthy evidence passages. Experimental results, conducted on two datasets MQAE-diag and MQAE, demonstrate the effectiveness of our framework for reasoning with textual evidence. Our approach outperforms state-of-the-art models, achieving a significant improvement of \textbf{6.86} and \textbf{9.43} percentage points on the Rouge-1 score; \textbf{8.23} and \textbf{7.82} percentage points on the Bleu-4 score on the respective datasets.

{{</citation>}}


### (50/165) Knowledge Crosswords: Geometric Reasoning over Structured Knowledge with Large Language Models (Wenxuan Ding et al., 2023)

{{<citation>}}

Wenxuan Ding, Shangbin Feng, Yuhan Liu, Zhaoxuan Tan, Vidhisha Balachandran, Tianxing He, Yulia Tsvetkov. (2023)  
**Knowledge Crosswords: Geometric Reasoning over Structured Knowledge with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, QA, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.01290v1)  

---


**ABSTRACT**  
Large language models (LLMs) are widely adopted in knowledge-intensive tasks and have achieved impressive performance thanks to their knowledge abilities. While LLMs have demonstrated outstanding performance on atomic or linear (multi-hop) QA tasks, whether they can reason in knowledge-rich scenarios with interweaving constraints remains an underexplored problem. In this work, we propose geometric reasoning over structured knowledge, where pieces of knowledge are connected in a graph structure and models need to fill in the missing information. Such geometric knowledge reasoning would require the ability to handle structured knowledge, reason with uncertainty, verify facts, and backtrack when an error occurs. We propose Knowledge Crosswords, a multi-blank QA dataset where each problem consists of a natural language question representing the geometric constraints of an incomplete entity network, where LLMs are tasked with working out the missing entities while meeting all factual constraints. Knowledge Crosswords contains 2,101 individual problems, covering various knowledge domains and further divided into three difficulty levels. We conduct extensive experiments to evaluate existing LLM prompting approaches on the Knowledge Crosswords benchmark. We additionally propose two new approaches, Staged Prompting and Verify-All, to augment LLMs' ability to backtrack and verify structured constraints. Our results demonstrate that while baseline approaches perform well on easier problems but struggle with hard ones, our proposed Verify-All outperforms other methods by a large margin and is more robust with hard problems. Further analysis reveals that LLMs' ability of geometric reasoning over structured knowledge is still far from robust or perfect, susceptible to confounders such as the order of options, certain structural patterns, assumption of existence of correct answer, and more.

{{</citation>}}


### (51/165) LEEC: A Legal Element Extraction Dataset with an Extensive Domain-Specific Label System (Xue Zongyue et al., 2023)

{{<citation>}}

Xue Zongyue, Liu Huanghai, Hu Yiran, Kong Kangle, Wang Chenlu, Liu Yun, Shen Weixing. (2023)  
**LEEC: A Legal Element Extraction Dataset with an Extensive Domain-Specific Label System**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-IR, cs.CL  
Keywords: Event Extraction, Legal  
[Paper Link](http://arxiv.org/abs/2310.01271v1)  

---


**ABSTRACT**  
As a pivotal task in natural language processing, element extraction has gained significance in the legal domain. Extracting legal elements from judicial documents helps enhance interpretative and analytical capacities of legal cases, and thereby facilitating a wide array of downstream applications in various domains of law. Yet existing element extraction datasets are limited by their restricted access to legal knowledge and insufficient coverage of labels. To address this shortfall, we introduce a more comprehensive, large-scale criminal element extraction dataset, comprising 15,831 judicial documents and 159 labels. This dataset was constructed through two main steps: First, designing the label system by our team of legal experts based on prior legal research which identified critical factors driving and processes generating sentencing outcomes in criminal cases; Second, employing the legal knowledge to annotate judicial documents according to the label system and annotation guideline. The Legal Element ExtraCtion dataset (LEEC) represents the most extensive and domain-specific legal element extraction dataset for the Chinese legal system. Leveraging the annotated data, we employed various SOTA models that validates the applicability of LEEC for Document Event Extraction (DEE) task. The LEEC dataset is available on https://github.com/THUlawtech/LEEC .

{{</citation>}}


### (52/165) Improving Emotional Expression and Cohesion in Image-Based Playlist Description and Music Topics: A Continuous Parameterization Approach (Yuelyu Ji et al., 2023)

{{<citation>}}

Yuelyu Ji, Yuheng Song, Wei Wang, Ruoyi Xu, Zhongqian Xie, Huiyun Liu. (2023)  
**Improving Emotional Expression and Cohesion in Image-Based Playlist Description and Music Topics: A Continuous Parameterization Approach**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Text Generation  
[Paper Link](http://arxiv.org/abs/2310.01248v1)  

---


**ABSTRACT**  
Text generation in image-based platforms, particularly for music-related content, requires precise control over text styles and the incorporation of emotional expression. However, existing approaches often need help to control the proportion of external factors in generated text and rely on discrete inputs, lacking continuous control conditions for desired text generation. This study proposes Continuous Parameterization for Controlled Text Generation (CPCTG) to overcome these limitations. Our approach leverages a Language Model (LM) as a style learner, integrating Semantic Cohesion (SC) and Emotional Expression Proportion (EEP) considerations. By enhancing the reward method and manipulating the CPCTG level, our experiments on playlist description and music topic generation tasks demonstrate significant improvements in ROUGE scores, indicating enhanced relevance and coherence in the generated text.

{{</citation>}}


### (53/165) Label Supervised LLaMA Finetuning (Zongxi Li et al., 2023)

{{<citation>}}

Zongxi Li, Xianming Li, Yuzhang Liu, Haoran Xie, Jing Li, Fu-lee Wang, Qing Li, Xiaoqin Zhong. (2023)  
**Label Supervised LLaMA Finetuning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, LLaMA, Language Model, NER  
[Paper Link](http://arxiv.org/abs/2310.01208v1)  

---


**ABSTRACT**  
The recent success of Large Language Models (LLMs) has gained significant attention in both academia and industry. Substantial efforts have been made to enhance the zero- and few-shot generalization capabilities of open-source LLMs through finetuning. Currently, the prevailing approach is instruction-tuning, which trains LLMs to complete real-world tasks by generating responses guided by natural language instructions. It is worth noticing that such an approach may underperform in sequence and token classification tasks. Unlike text generation tasks, classification tasks have a limited label space, where precise label prediction is more appreciated than generating diverse and human-like responses. Prior research has unveiled that instruction-tuned LLMs cannot outperform BERT, prompting us to explore the potential of leveraging latent representations from LLMs for supervised label prediction. In this paper, we introduce a label-supervised adaptation for LLMs, which aims to finetuning the model with discriminant labels. We evaluate this approach with Label Supervised LLaMA (LS-LLaMA), based on LLaMA-2-7B, a relatively small-scale LLM, and can be finetuned on a single GeForce RTX4090 GPU. We extract latent representations from the final LLaMA layer and project them into the label space to compute the cross-entropy loss. The model is finetuned by Low-Rank Adaptation (LoRA) to minimize this loss. Remarkably, without intricate prompt engineering or external knowledge, LS-LLaMA substantially outperforms LLMs ten times its size in scale and demonstrates consistent improvements compared to robust baselines like BERT-Large and RoBERTa-Large in text classification. Moreover, by removing the causal mask from decoders, LS-unLLaMA achieves the state-of-the-art performance in named entity recognition (NER). Our work will shed light on a novel approach to adapting LLMs for various downstream tasks.

{{</citation>}}


### (54/165) Quantifying the Plausibility of Context Reliance in Neural Machine Translation (Gabriele Sarti et al., 2023)

{{<citation>}}

Gabriele Sarti, Grzegorz Chrupała, Malvina Nissim, Arianna Bisazza. (2023)  
**Quantifying the Plausibility of Context Reliance in Neural Machine Translation**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-AI, cs-CL, cs-HC, cs-LG, cs.CL  
Keywords: Machine Translation  
[Paper Link](http://arxiv.org/abs/2310.01188v1)  

---


**ABSTRACT**  
Establishing whether language models can use contextual information in a human-plausible way is important to ensure their safe adoption in real-world settings. However, the questions of when and which parts of the context affect model generations are typically tackled separately, and current plausibility evaluations are practically limited to a handful of artificial benchmarks. To address this, we introduce Plausibility Evaluation of Context Reliance (PECoRe), an end-to-end interpretability framework designed to quantify context usage in language models' generations. Our approach leverages model internals to (i) contrastively identify context-sensitive target tokens in generated texts and (ii) link them to contextual cues justifying their prediction. We use PECoRe to quantify the plausibility of context-aware machine translation models, comparing model rationales with human annotations across several discourse-level phenomena. Finally, we apply our method to unannotated generations to identify context-mediated predictions and highlight instances of (im)plausible context usage in model translations.

{{</citation>}}


### (55/165) NarrativePlay: Interactive Narrative Understanding (Runcong Zhao et al., 2023)

{{<citation>}}

Runcong Zhao, Wenjia Zhang, Jiazheng Li, Lixing Zhu, Yanran Li, Yulan He, Lin Gui. (2023)  
**NarrativePlay: Interactive Narrative Understanding**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-HC, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.01459v1)  

---


**ABSTRACT**  
In this paper, we introduce NarrativePlay, a novel system that allows users to role-play a fictional character and interact with other characters in narratives such as novels in an immersive environment. We leverage Large Language Models (LLMs) to generate human-like responses, guided by personality traits extracted from narratives. The system incorporates auto-generated visual display of narrative settings, character portraits, and character speech, greatly enhancing user experience. Our approach eschews predefined sandboxes, focusing instead on main storyline events extracted from narratives from the perspective of a user-selected character. NarrativePlay has been evaluated on two types of narratives, detective and adventure stories, where users can either explore the world or improve their favorability with the narrative characters through conversations.

{{</citation>}}


### (56/165) Target-Aware Contextual Political Bias Detection in News (Iffat Maab et al., 2023)

{{<citation>}}

Iffat Maab, Edison Marrese-Taylor, Yutaka Matsuo. (2023)  
**Target-Aware Contextual Political Bias Detection in News**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, Bias  
[Paper Link](http://arxiv.org/abs/2310.01138v1)  

---


**ABSTRACT**  
Media bias detection requires comprehensive integration of information derived from multiple news sources. Sentence-level political bias detection in news is no exception, and has proven to be a challenging task that requires an understanding of bias in consideration of the context. Inspired by the fact that humans exhibit varying degrees of writing styles, resulting in a diverse range of statements with different local and global contexts, previous work in media bias detection has proposed augmentation techniques to exploit this fact. Despite their success, we observe that these techniques introduce noise by over-generalizing bias context boundaries, which hinders performance. To alleviate this issue, we propose techniques to more carefully search for context using a bias-sensitive, target-aware approach for data augmentation. Comprehensive experiments on the well-known BASIL dataset show that when combined with pre-trained models such as BERT, our augmentation techniques lead to state-of-the-art results. Our approach outperforms previous methods significantly, obtaining an F1-score of 58.15 over state-of-the-art bias detection task.

{{</citation>}}


### (57/165) Automated Evaluation of Classroom Instructional Support with LLMs and BoWs: Connecting Global Predictions to Specific Feedback (Jacob Whitehill et al., 2023)

{{<citation>}}

Jacob Whitehill, Jennifer LoCasale-Crouch. (2023)  
**Automated Evaluation of Classroom Instructional Support with LLMs and BoWs: Connecting Global Predictions to Specific Feedback**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2310.01132v1)  

---


**ABSTRACT**  
With the aim to provide teachers with more specific, frequent, and actionable feedback about their teaching, we explore how Large Language Models (LLMs) can be used to estimate ``Instructional Support'' domain scores of the CLassroom Assessment Scoring System (CLASS), a widely used observation protocol. We design a machine learning architecture that uses either zero-shot prompting of Meta's Llama2, and/or a classic Bag of Words (BoW) model, to classify individual utterances of teachers' speech (transcribed automatically using OpenAI's Whisper) for the presence of 11 behavioral indicators of Instructional Support. Then, these utterance-level judgments are aggregated over an entire 15-min observation session to estimate a global CLASS score. Experiments on two CLASS-coded datasets of toddler and pre-kindergarten classrooms indicate that (1) automatic CLASS Instructional Support estimation accuracy using the proposed method (Pearson $R$ up to $0.46$) approaches human inter-rater reliability (up to $R=0.55$); (2) LLMs yield slightly greater accuracy than BoW for this task; and (3) the best models often combined features extracted from both LLM and BoW. Finally, (4) we illustrate how the model's outputs can be visualized at the utterance level to provide teachers with explainable feedback on which utterances were most positively or negatively correlated with specific CLASS dimensions.

{{</citation>}}


### (58/165) Text Data Augmentation in Low-Resource Settings via Fine-Tuning of Large Language Models (Jean Kaddour et al., 2023)

{{<citation>}}

Jean Kaddour, Qi Liu. (2023)  
**Text Data Augmentation in Low-Resource Settings via Fine-Tuning of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Augmentation, Language Model, Low-Resource  
[Paper Link](http://arxiv.org/abs/2310.01119v1)  

---


**ABSTRACT**  
The in-context learning ability of large language models (LLMs) enables them to generalize to novel downstream tasks with relatively few labeled examples. However, they require enormous computational resources to be deployed. Alternatively, smaller models can solve specific tasks if fine-tuned with enough labeled examples. These examples, however, are expensive to obtain. In pursuit of the best of both worlds, we study the annotation and generation of fine-tuning training data via fine-tuned teacher LLMs to improve the downstream performance of much smaller models. In four text classification and two text generation tasks, we find that both data generation and annotation dramatically improve the respective downstream model's performance, occasionally necessitating only a minor fraction of the original training dataset.

{{</citation>}}


### (59/165) GraphText: Graph Reasoning in Text Space (Jianan Zhao et al., 2023)

{{<citation>}}

Jianan Zhao, Le Zhuo, Yikang Shen, Meng Qu, Kai Liu, Michael Bronstein, Zhaocheng Zhu, Jian Tang. (2023)  
**GraphText: Graph Reasoning in Text Space**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: ChatGPT, GPT, Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.01089v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have gained the ability to assimilate human knowledge and facilitate natural language interactions with both humans and other LLMs. However, despite their impressive achievements, LLMs have not made significant advancements in the realm of graph machine learning. This limitation arises because graphs encapsulate distinct relational data, making it challenging to transform them into natural language that LLMs understand. In this paper, we bridge this gap with a novel framework, GraphText, that translates graphs into natural language. GraphText derives a graph-syntax tree for each graph that encapsulates both the node attributes and inter-node relationships. Traversal of the tree yields a graph text sequence, which is then processed by an LLM to treat graph tasks as text generation tasks. Notably, GraphText offers multiple advantages. It introduces training-free graph reasoning: even without training on graph data, GraphText with ChatGPT can achieve on par with, or even surpassing, the performance of supervised-trained graph neural networks through in-context learning (ICL). Furthermore, GraphText paves the way for interactive graph reasoning, allowing both humans and LLMs to communicate with the model seamlessly using natural language. These capabilities underscore the vast, yet-to-be-explored potential of LLMs in the domain of graph machine learning.

{{</citation>}}


### (60/165) Towards human-like spoken dialogue generation between AI agents from written dialogue (Kentaro Mitsui et al., 2023)

{{<citation>}}

Kentaro Mitsui, Yukiya Hono, Kei Sawada. (2023)  
**Towards human-like spoken dialogue generation between AI agents from written dialogue**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs-SD, cs.CL, eess-AS  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.01088v1)  

---


**ABSTRACT**  
The advent of large language models (LLMs) has made it possible to generate natural written dialogues between two agents. However, generating human-like spoken dialogues from these written dialogues remains challenging. Spoken dialogues have several unique characteristics: they frequently include backchannels and laughter, and the smoothness of turn-taking significantly influences the fluidity of conversation. This study proposes CHATS - CHatty Agents Text-to-Speech - a discrete token-based system designed to generate spoken dialogues based on written dialogues. Our system can generate speech for both the speaker side and the listener side simultaneously, using only the transcription from the speaker side, which eliminates the need for transcriptions of backchannels or laughter. Moreover, CHATS facilitates natural turn-taking; it determines the appropriate duration of silence after each utterance in the absence of overlap, and it initiates the generation of overlapping speech based on the phoneme sequence of the next utterance in case of overlap. Experimental evaluations indicate that CHATS outperforms the text-to-speech baseline, producing spoken dialogues that are more interactive and fluid while retaining clarity and intelligibility.

{{</citation>}}


### (61/165) Back to the Future: Towards Explainable Temporal Reasoning with Large Language Models (Chenhan Yuan et al., 2023)

{{<citation>}}

Chenhan Yuan, Qianqian Xie, Jimin Huang, Sophia Ananiadou. (2023)  
**Back to the Future: Towards Explainable Temporal Reasoning with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, Language Model, NLP, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.01074v1)  

---


**ABSTRACT**  
Temporal reasoning is a crucial NLP task, providing a nuanced understanding of time-sensitive contexts within textual data. Although recent advancements in LLMs have demonstrated their potential in temporal reasoning, the predominant focus has been on tasks such as temporal expression and temporal relation extraction. These tasks are primarily designed for the extraction of direct and past temporal cues and to engage in simple reasoning processes. A significant gap remains when considering complex reasoning tasks such as event forecasting, which requires multi-step temporal reasoning on events and prediction on the future timestamp. Another notable limitation of existing methods is their incapability to provide an illustration of their reasoning process, hindering explainability. In this paper, we introduce the first task of explainable temporal reasoning, to predict an event's occurrence at a future timestamp based on context which requires multiple reasoning over multiple events, and subsequently provide a clear explanation for their prediction. Our task offers a comprehensive evaluation of both the LLMs' complex temporal reasoning ability, the future event prediction ability, and explainability-a critical attribute for AI applications. To support this task, we present the first multi-source instruction-tuning dataset of explainable temporal reasoning (ExpTime) with 26k derived from the temporal knowledge graph datasets and their temporal reasoning paths, using a novel knowledge-graph-instructed-generation strategy. Based on the dataset, we propose the first open-source LLM series TimeLlaMA based on the foundation LlaMA2, with the ability of instruction following for explainable temporal reasoning. We compare the performance of our method and a variety of LLMs, where our method achieves the state-of-the-art performance of temporal prediction and explanation.

{{</citation>}}


### (62/165) Reasoning on Graphs: Faithful and Interpretable Large Language Model Reasoning (Linhao Luo et al., 2023)

{{<citation>}}

Linhao Luo, Yuan-Fang Li, Gholamreza Haffari, Shirui Pan. (2023)  
**Reasoning on Graphs: Faithful and Interpretable Large Language Model Reasoning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, QA, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.01061v1)  

---


**ABSTRACT**  
Large language models (LLMs) have demonstrated impressive reasoning abilities in complex tasks. However, they lack up-to-date knowledge and experience hallucinations during reasoning, which can lead to incorrect reasoning processes and diminish their performance and trustworthiness. Knowledge graphs (KGs), which capture vast amounts of facts in a structured format, offer a reliable source of knowledge for reasoning. Nevertheless, existing KG-based LLM reasoning methods only treat KGs as factual knowledge bases and overlook the importance of their structural information for reasoning. In this paper, we propose a novel method called reasoning on graphs (RoG) that synergizes LLMs with KGs to enable faithful and interpretable reasoning. Specifically, we present a planning-retrieval-reasoning framework, where RoG first generates relation paths grounded by KGs as faithful plans. These plans are then used to retrieve valid reasoning paths from the KGs for LLMs to conduct faithful reasoning. Furthermore, RoG not only distills knowledge from KGs to improve the reasoning ability of LLMs through training but also allows seamless integration with any arbitrary LLMs during inference. Extensive experiments on two benchmark KGQA datasets demonstrate that RoG achieves state-of-the-art performance on KG reasoning tasks and generates faithful and interpretable reasoning results.

{{</citation>}}


### (63/165) Tool-Augmented Reward Modeling (Lei Li et al., 2023)

{{<citation>}}

Lei Li, Yekun Chai, Shuohuan Wang, Yu Sun, Hao Tian, Ningyu Zhang, Hua Wu. (2023)  
**Tool-Augmented Reward Modeling**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2310.01045v1)  

---


**ABSTRACT**  
Reward modeling (a.k.a., preference modeling) is instrumental for aligning large language models with human preferences, particularly within the context of reinforcement learning from human feedback (RLHF). While conventional reward models (RMs) have exhibited remarkable scalability, they oft struggle with fundamental functionality such as arithmetic computation, code execution, and factual lookup. In this paper, we propose a tool-augmented preference modeling approach, named \name, to address these limitations by empowering RMs with access to external environments, including calculators and search engines. This approach not only fosters synergy between tool utilization and reward grading but also enhances interpretive capacity and scoring reliability. Our study delves into the integration of external tools into RMs, enabling them to interact with diverse external sources and construct task-specific tool engagement and reasoning traces in an autoregressive manner. We validate our approach across a wide range of domains, incorporating seven distinct external tools. Our experimental results demonstrate a noteworthy overall improvement of 17.7% across eight tasks in preference ranking. Furthermore, our approach outperforms Gopher 280B by 7.3% on TruthfulQA task in zero-shot evaluation. In human evaluations, RLHF trained with Themis attains an average win rate of 32% when compared to baselines across four distinct tasks. Additionally, we provide a comprehensive collection of tool-related RM datasets, incorporating data from seven distinct tool APIs, totaling 15,000 instances. We anticipate that this publicly available dataset will facilitate and inspire further research advancements in the field.

{{</citation>}}


### (64/165) Language Model Decoding as Direct Metrics Optimization (Haozhe Ji et al., 2023)

{{<citation>}}

Haozhe Ji, Pei Ke, Hongning Wang, Minlie Huang. (2023)  
**Language Model Decoding as Direct Metrics Optimization**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.01041v1)  

---


**ABSTRACT**  
Despite the remarkable advances in language modeling, current mainstream decoding methods still struggle to generate texts that align with human texts across different aspects. In particular, sampling-based methods produce less-repetitive texts which are often disjunctive in discourse, while search-based methods maintain topic coherence at the cost of increased repetition. Overall, these methods fall short in achieving holistic alignment across a broad range of aspects. In this work, we frame decoding from a language model as an optimization problem with the goal of strictly matching the expected performance with human texts measured by multiple metrics of desired aspects simultaneously. The resulting decoding distribution enjoys an analytical solution that scales the input language model distribution via a sequence-level energy function defined by these metrics. And most importantly, we prove that this induced distribution is guaranteed to improve the perplexity on human texts, which suggests a better approximation to the underlying distribution of human texts. To facilitate tractable sampling from this globally normalized distribution, we adopt the Sampling-Importance-Resampling technique. Experiments on various domains and model scales demonstrate the superiority of our method in metrics alignment with human texts and human evaluation over strong baselines.

{{</citation>}}


### (65/165) ARN: A Comprehensive Framework and Dataset for Analogical Reasoning on Narratives (Zhivar Sourati et al., 2023)

{{<citation>}}

Zhivar Sourati, Filip Ilievski, Pia Sommerauer. (2023)  
**ARN: A Comprehensive Framework and Dataset for Analogical Reasoning on Narratives**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.00996v1)  

---


**ABSTRACT**  
Analogical reasoning is one of the prime abilities of humans and is linked to creativity and scientific discoveries. This ability has been studied extensively in natural language processing (NLP) as well as in cognitive psychology by proposing various benchmarks and evaluation setups. Yet, a substantial gap exists between evaluations of analogical reasoning in cognitive psychology and NLP. Our aim is to bridge this by computationally adapting theories related to analogical reasoning from cognitive psychology in the context of narratives and developing an evaluation framework large in scale. More concretely, we propose the task of matching narratives based on system mappings and release the Analogical Reasoning on Narratives (ARN) dataset. To create the dataset, we devise a framework inspired by cognitive psychology theories about analogical reasoning to utilize narratives and their components to form mappings of different abstractness levels. These mappings are then leveraged to create pairs of analogies and disanalogies/distractors with more than 1k triples of query narratives, analogies, and distractors. We cover four categories of far/near analogies and far/near distractors that allow us to study analogical reasoning in models from distinct perspectives. In this study, we evaluate different large language models (LLMs) on this task. Our results demonstrate that LLMs struggle to recognize higher-order mappings when they are not accompanied by lower-order mappings (far analogies) and show better performance when all mappings are present simultaneously (near analogies). We observe that in all the settings, the analogical reasoning abilities of LLMs can be easily impaired by near distractors that form lower-order mappings with the query narratives.

{{</citation>}}


### (66/165) EALM: Introducing Multidimensional Ethical Alignment in Conversational Information Retrieval (Yiyao Yu et al., 2023)

{{<citation>}}

Yiyao Yu, Junjie Wang, Yuxiang Zhang, Lin Zhang, Yujiu Yang, Tetsuya Sakai. (2023)  
**EALM: Introducing Multidimensional Ethical Alignment in Conversational Information Retrieval**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Information Retrieval, QA  
[Paper Link](http://arxiv.org/abs/2310.00970v1)  

---


**ABSTRACT**  
Artificial intelligence (AI) technologies should adhere to human norms to better serve our society and avoid disseminating harmful or misleading information, particularly in Conversational Information Retrieval (CIR). Previous work, including approaches and datasets, has not always been successful or sufficiently robust in taking human norms into consideration. To this end, we introduce a workflow that integrates ethical alignment, with an initial ethical judgment stage for efficient data screening. To address the need for ethical judgment in CIR, we present the QA-ETHICS dataset, adapted from the ETHICS benchmark, which serves as an evaluation tool by unifying scenarios and label meanings. However, each scenario only considers one ethical concept. Therefore, we introduce the MP-ETHICS dataset to evaluate a scenario under multiple ethical concepts, such as justice and Deontology. In addition, we suggest a new approach that achieves top performance in both binary and multi-label ethical judgment tasks. Our research provides a practical method for introducing ethical alignment into the CIR workflow. The data and code are available at https://github.com/wanng-ide/ealm .

{{</citation>}}


### (67/165) Resolving Knowledge Conflicts in Large Language Models (Yike Wang et al., 2023)

{{<citation>}}

Yike Wang, Shangbin Feng, Heng Wang, Weijia Shi, Vidhisha Balachandran, Tianxing He, Yulia Tsvetkov. (2023)  
**Resolving Knowledge Conflicts in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.00935v1)  

---


**ABSTRACT**  
Large language models (LLMs) often encounter knowledge conflicts, scenarios where discrepancy arises between the internal parametric knowledge of LLMs and non-parametric information provided in the prompt context. In this work we ask what are the desiderata for LLMs when a knowledge conflict arises and whether existing LLMs fulfill them. We posit that LLMs should 1) identify knowledge conflicts, 2) pinpoint conflicting information segments, and 3) provide distinct answers or viewpoints in conflicting scenarios. To this end, we introduce KNOWLEDGE CONFLICT, an evaluation framework for simulating contextual knowledge conflicts and quantitatively evaluating to what extent LLMs achieve these goals. KNOWLEDGE CONFLICT includes diverse and complex situations of knowledge conflict, knowledge from diverse entities and domains, two synthetic conflict creation methods, and settings with progressively increasing difficulty to reflect realistic knowledge conflicts. Extensive experiments with the KNOWLEDGE CONFLICT framework reveal that while LLMs perform well in identifying the existence of knowledge conflicts, they struggle to determine the specific conflicting knowledge and produce a response with distinct answers amidst conflicting information. To address these challenges, we propose new instruction-based approaches that augment LLMs to better achieve the three goals. Further analysis shows that abilities to tackle knowledge conflicts are greatly impacted by factors such as knowledge domain and prompt text, while generating robust responses to knowledge conflict scenarios remains an open research question.

{{</citation>}}


### (68/165) Fooling the Textual Fooler via Randomizing Latent Representations (Duy C. Hoang et al., 2023)

{{<citation>}}

Duy C. Hoang, Quang H. Nguyen, Saurav Manchanda, MinLong Peng, Kok-Seng Wong, Khoa D. Doan. (2023)  
**Fooling the Textual Fooler via Randomizing Latent Representations**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2310.01452v1)  

---


**ABSTRACT**  
Despite outstanding performance in a variety of NLP tasks, recent studies have revealed that NLP models are vulnerable to adversarial attacks that slightly perturb the input to cause the models to misbehave. Among these attacks, adversarial word-level perturbations are well-studied and effective attack strategies. Since these attacks work in black-box settings, they do not require access to the model architecture or model parameters and thus can be detrimental to existing NLP applications. To perform an attack, the adversary queries the victim model many times to determine the most important words in an input text and to replace these words with their corresponding synonyms. In this work, we propose a lightweight and attack-agnostic defense whose main goal is to perplex the process of generating an adversarial example in these query-based black-box attacks; that is to fool the textual fooler. This defense, named AdvFooler, works by randomizing the latent representation of the input at inference time. Different from existing defenses, AdvFooler does not necessitate additional computational overhead during training nor relies on assumptions about the potential adversarial perturbation set while having a negligible impact on the model's accuracy. Our theoretical and empirical analyses highlight the significance of robustness resulting from confusing the adversary via randomizing the latent space, as well as the impact of randomization on clean accuracy. Finally, we empirically demonstrate near state-of-the-art robustness of AdvFooler against representative adversarial word-level attacks on two benchmark datasets.

{{</citation>}}


### (69/165) All Languages Matter: On the Multilingual Safety of Large Language Models (Wenxuan Wang et al., 2023)

{{<citation>}}

Wenxuan Wang, Zhaopeng Tu, Chang Chen, Youliang Yuan, Jen-tse Huang, Wenxiang Jiao, Michael R. Lyu. (2023)  
**All Languages Matter: On the Multilingual Safety of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model, Multilingual  
[Paper Link](http://arxiv.org/abs/2310.00905v1)  

---


**ABSTRACT**  
Safety lies at the core of developing and deploying large language models (LLMs). However, previous safety benchmarks only concern the safety in one language, e.g. the majority language in the pretraining data such as English. In this work, we build the first multilingual safety benchmark for LLMs, XSafety, in response to the global deployment of LLMs in practice. XSafety covers 14 kinds of commonly used safety issues across 10 languages that span several language families. We utilize XSafety to empirically study the multilingual safety for 4 widely-used LLMs, including both close-API and open-source models. Experimental results show that all LLMs produce significantly more unsafe responses for non-English queries than English ones, indicating the necessity of developing safety alignment for non-English languages. In addition, we propose several simple and effective prompting methods to improve the multilingual safety of ChatGPT by evoking safety knowledge and improving cross-lingual generalization of safety alignment. Our prompting method can significantly reduce the ratio of unsafe responses from 19.1% to 9.7% for non-English queries. We release our data at https://github.com/Jarviswang94/Multilingual_safety_benchmark.

{{</citation>}}


### (70/165) TADIS: Steering Models for Deep-Thinking about Demonstration Examples (Tianci Xue et al., 2023)

{{<citation>}}

Tianci Xue, Ziqi Wang, Yixia Li, Yun Chen, Guanhua Chen. (2023)  
**TADIS: Steering Models for Deep-Thinking about Demonstration Examples**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.00901v1)  

---


**ABSTRACT**  
Instruction tuning has been demonstrated that could significantly improve the zero-shot generalization capability to unseen tasks by an apparent margin. By incorporating additional context (e.g., task definition, examples) during the fine-tuning process, Large Language Models (LLMs) achieved much higher performance than before. However, recent work reported that delusive task examples can achieve almost the same performance as correct task examples, indicating the input-label correspondence is less important than previously thought. Intrigued by this counter-intuitive observation, we suspect models have the same illusion of competence as humans. Therefore, we propose a novel method called TADIS that steers LLMs for "Deep-Thinking'' about demonstration examples instead of merely seeing. To alleviate the illusion of competence of models, we first ask the model to verify the correctness of shown examples. Then, using the verification results as conditions to elicit models for a better answer. Our experimental results show that TADIS consistently outperforms competitive baselines on in-domain and out-domain tasks (improving 2.79 and 4.03 average ROUGLE-L on out-domain and in-domain datasets, respectively). Despite the presence of generated examples (not all of the thinking labels are accurate), TADIS can notably enhance performance in zero-shot and few-shot settings. This also suggests that our approach can be adopted on a large scale to improve the instruction following capabilities of models without any manual labor. Moreover, we construct three types of thinking labels with different model sizes and find that small models learn from the format of TADIS but larger models can be steered for "Deep-Thinking''.

{{</citation>}}


### (71/165) Enable Language Models to Implicitly Learn Self-Improvement From Data (Ziqi Wang et al., 2023)

{{<citation>}}

Ziqi Wang, Le Hou, Tianjian Lu, Yuexin Wu, Yunxuan Li, Hongkun Yu, Heng Ji. (2023)  
**Enable Language Models to Implicitly Learn Self-Improvement From Data**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.00898v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have demonstrated remarkable capabilities in open-ended text generation tasks. However, the inherent open-ended nature of these tasks implies that there is always room for improvement in the quality of model responses. To address this challenge, various approaches have been proposed to enhance the performance of LLMs. There has been a growing focus on enabling LLMs to self-improve their response quality, thereby reducing the reliance on extensive human annotation efforts for collecting diverse and high-quality training data. Recently, prompting-based methods have been widely explored among self-improvement methods owing to their effectiveness, efficiency, and convenience. However, those methods usually require explicitly and thoroughly written rubrics as inputs to LLMs. It is expensive and challenging to manually derive and provide all necessary rubrics with a real-world complex goal for improvement (e.g., being more helpful and less harmful). To this end, we propose an ImPlicit Self-ImprovemenT (PIT) framework that implicitly learns the improvement goal from human preference data. PIT only requires preference data that are used to train reward models without extra human efforts. Specifically, we reformulate the training objective of reinforcement learning from human feedback (RLHF) -- instead of maximizing response quality for a given input, we maximize the quality gap of the response conditioned on a reference response. In this way, PIT is implicitly trained with the improvement goal of better aligning with human preferences. Experiments on two real-world datasets and one synthetic dataset show that our method significantly outperforms prompting-based methods.

{{</citation>}}


### (72/165) No Offense Taken: Eliciting Offensiveness from Language Models (Anugya Srivastava et al., 2023)

{{<citation>}}

Anugya Srivastava, Rahul Ahuja, Rohith Mukku. (2023)  
**No Offense Taken: Eliciting Offensiveness from Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.00892v1)  

---


**ABSTRACT**  
This work was completed in May 2022.   For safe and reliable deployment of language models in the real world, testing needs to be robust. This robustness can be characterized by the difficulty and diversity of the test cases we evaluate these models on. Limitations in human-in-the-loop test case generation has prompted an advent of automated test case generation approaches. In particular, we focus on Red Teaming Language Models with Language Models by Perez et al.(2022). Our contributions include developing a pipeline for automated test case generation via red teaming that leverages publicly available smaller language models (LMs), experimenting with different target LMs and red classifiers, and generating a corpus of test cases that can help in eliciting offensive responses from widely deployed LMs and identifying their failure modes.

{{</citation>}}


### (73/165) (Dynamic) Prompting might be all you need to repair Compressed LLMs (Duc N. M Hoang et al., 2023)

{{<citation>}}

Duc N. M Hoang, Minsik Cho, Thomas Merth, Mohammad Rastegari, Zhangyang Wang. (2023)  
**(Dynamic) Prompting might be all you need to repair Compressed LLMs**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: LLaMA, NLP  
[Paper Link](http://arxiv.org/abs/2310.00867v1)  

---


**ABSTRACT**  
Large language models (LLMs), while transformative for NLP, come with significant computational demands, underlining the need for efficient, training-free compression. Notably, the reliability of perplexity as a benchmark for compressed model efficacy is in question, as our tests using LLaMA-7B and OPT-6.7b reveal a significant performance drop in several realistic downstream tasks, underscoring the disparity between perplexity as a performance indicator and real-world performance. Investigation into the trade-off between resource-intensive post-compression re-training highlights the prospect of prompt-driven recovery as a lightweight adaption tool. However, existing studies, confined mainly to perplexity evaluations and simple tasks, fail to offer unequivocal confidence in the scalability and generalizability of prompting. We tackle this uncertainty in two key ways. First, we uncover the vulnerability of naive prompts in LLM compression as an over-reliance on a singular prompt per input. In response, we propose inference-time dynamic prompting (IDP), a mechanism that autonomously chooses from a set of curated prompts based on the context of each individual input. Second, we delve into a scientific understanding of why ``prompting might be all you need post-LLM compression". Our findings suggest that compression doesn't irretrievably erase LLM model knowledge but displace it, necessitating a new inference path. IDP effectively redirects this path, enabling the model to tap into its inherent yet displaced knowledge and thereby recover performance. Empirical tests affirm the value of IDP, demonstrating an average performance improvement of 1.24% across nine varied tasks spanning multiple knowledge domains.

{{</citation>}}


### (74/165) Melody-conditioned lyrics generation via fine-tuning language model and its evaluation with ChatGPT (Zhe Zhang et al., 2023)

{{<citation>}}

Zhe Zhang, Karol Lasocki, Yi Yu, Atsuhiro Takasu. (2023)  
**Melody-conditioned lyrics generation via fine-tuning language model and its evaluation with ChatGPT**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Transformer  
[Paper Link](http://arxiv.org/abs/2310.00863v1)  

---


**ABSTRACT**  
We leverage character-level language models for syllable-level lyrics generation from symbolic melody. By fine-tuning a character-level pre-trained model, we integrate language knowledge into the beam search of a syllable-level Transformer generator. Using ChatGPT-based evaluations, we demonstrate enhanced coherence and correctness in the generated lyrics.

{{</citation>}}


### (75/165) Application of frozen large-scale models to multimodal task-oriented dialogue (Tatsuki Kawamoto et al., 2023)

{{<citation>}}

Tatsuki Kawamoto, Takuma Suzuki, Ko Miyama, Takumi Meguro, Tomohiro Takagi. (2023)  
**Application of frozen large-scale models to multimodal task-oriented dialogue**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, Dialog, GPT, Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2310.00845v1)  

---


**ABSTRACT**  
In this study, we use the existing Large Language Models ENnhanced to See Framework (LENS Framework) to test the feasibility of multimodal task-oriented dialogues. The LENS Framework has been proposed as a method to solve computer vision tasks without additional training and with fixed parameters of pre-trained models. We used the Multimodal Dialogs (MMD) dataset, a multimodal task-oriented dialogue benchmark dataset from the fashion field, and for the evaluation, we used the ChatGPT-based G-EVAL, which only accepts textual modalities, with arrangements to handle multimodal data. Compared to Transformer-based models in previous studies, our method demonstrated an absolute lift of 10.8% in fluency, 8.8% in usefulness, and 5.2% in relevance and coherence. The results show that using large-scale models with fixed parameters rather than using models trained on a dataset from scratch improves performance in multimodal task-oriented dialogues. At the same time, we show that Large Language Models (LLMs) are effective for multimodal task-oriented dialogues. This is expected to lead to efficient applications to existing systems.

{{</citation>}}


### (76/165) Error Norm Truncation: Robust Training in the Presence of Data Noise for Text Generation Models (Tianjian Li et al., 2023)

{{<citation>}}

Tianjian Li, Haoran Xu, Philipp Koehn, Daniel Khashabi, Kenton Murray. (2023)  
**Error Norm Truncation: Robust Training in the Presence of Data Noise for Text Generation Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BLEU, Text Generation  
[Paper Link](http://arxiv.org/abs/2310.00840v1)  

---


**ABSTRACT**  
Text generation models are notoriously vulnerable to errors in the training data. With the wide-spread availability of massive amounts of web-crawled data becoming more commonplace, how can we enhance the robustness of models trained on a massive amount of noisy web-crawled text? In our work, we propose Error Norm Truncation (ENT), a robust enhancement method to the standard training objective that truncates noisy data. Compared to methods that only uses the negative log-likelihood loss to estimate data quality, our method provides a more accurate estimation by considering the distribution of non-target tokens, which is often overlooked by previous work. Through comprehensive experiments across language modeling, machine translation, and text summarization, we show that equipping text generation models with ENT improves generation quality over standard training and previous soft and hard truncation methods. Furthermore, we show that our method improves the robustness of models against two of the most detrimental types of noise in machine translation, resulting in an increase of more than 2 BLEU points over the MLE baseline when up to 50% of noise is added to the data.

{{</citation>}}


### (77/165) Towards LogiGLUE: A Brief Survey and A Benchmark for Analyzing Logical Reasoning Capabilities of Language Models (Man Luo et al., 2023)

{{<citation>}}

Man Luo, Shrinidhi Kumbhar, Ming shen, Mihir Parmar, Neeraj Varshney, Pratyay Banerjee, Somak Aditya, Chitta Baral. (2023)  
**Towards LogiGLUE: A Brief Survey and A Benchmark for Analyzing Logical Reasoning Capabilities of Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GLUE, Language Model, Reasoning, Seq2Seq, T5  
[Paper Link](http://arxiv.org/abs/2310.00836v1)  

---


**ABSTRACT**  
Logical reasoning is fundamental for humans yet presents a substantial challenge in the domain of Artificial Intelligence. Initially, researchers used Knowledge Representation and Reasoning (KR) systems that did not scale and required non trivial manual effort. Recently, the emergence of large language models (LLMs) has demonstrated the ability to overcome various limitations of formal Knowledge Representation (KR) systems. Consequently, there is a growing interest in using LLMs for logical reasoning via natural language. This work strives to understand the proficiency of LLMs in logical reasoning by offering a brief review of the latest progress in this area; with a focus on the logical reasoning datasets, tasks, and the methods adopted to utilize LLMs for reasoning. To offer a thorough analysis, we have compiled a benchmark titled LogiGLUE. This includes 24 varied datasets encompassing deductive, abductive, and inductive reasoning. We have standardized these datasets into Seq2Seq tasks to facilitate straightforward training and evaluation for future research. Utilizing LogiGLUE as a foundation, we have trained an instruction fine tuned language model, resulting in LogiT5. We study single task training, multi task training, and a chain of thought knowledge distillation fine tuning technique to assess the performance of model across the different logical reasoning categories. By this comprehensive process, we aim to shed light on the capabilities and potential pathways for enhancing logical reasoning proficiency in LLMs, paving the way for more advanced and nuanced developments in this critical field.

{{</citation>}}


### (78/165) TRAM: Benchmarking Temporal Reasoning for Large Language Models (Yuqing Wang et al., 2023)

{{<citation>}}

Yuqing Wang, Yun Zhao. (2023)  
**TRAM: Benchmarking Temporal Reasoning for Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, GPT, GPT-4, Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.00835v2)  

---


**ABSTRACT**  
Reasoning about time is essential for understanding the nuances of events described in natural language. Previous research on this topic has been limited in scope, characterized by a lack of standardized benchmarks that would allow for consistent evaluations across different studies. In this paper, we introduce TRAM, a temporal reasoning benchmark composed of ten datasets, encompassing various temporal aspects of events such as order, arithmetic, frequency, and duration, designed to facilitate a comprehensive evaluation of the temporal reasoning capabilities of large language models (LLMs). We conduct an extensive evaluation using popular LLMs, such as GPT-4 and Llama2, in both zero-shot and few-shot learning scenarios. Additionally, we employ BERT-based models to establish the baseline evaluations. Our findings indicate that these models still trail human performance in temporal reasoning tasks. It is our aspiration that TRAM will spur further progress in enhancing the temporal reasoning abilities of LLMs.

{{</citation>}}


### (79/165) Necessary and Sufficient Watermark for Large Language Models (Yuki Takezawa et al., 2023)

{{<citation>}}

Yuki Takezawa, Ryoma Sato, Han Bao, Kenta Niwa, Makoto Yamada. (2023)  
**Necessary and Sufficient Watermark for Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: BLEU, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2310.00833v1)  

---


**ABSTRACT**  
In recent years, large language models (LLMs) have achieved remarkable performances in various NLP tasks. They can generate texts that are indistinguishable from those written by humans. Such remarkable performance of LLMs increases their risk of being used for malicious purposes, such as generating fake news articles. Therefore, it is necessary to develop methods for distinguishing texts written by LLMs from those written by humans. Watermarking is one of the most powerful methods for achieving this. Although existing watermarking methods have successfully detected texts generated by LLMs, they significantly degrade the quality of the generated texts. In this study, we propose the Necessary and Sufficient Watermark (NS-Watermark) for inserting watermarks into generated texts without degrading the text quality. More specifically, we derive minimum constraints required to be imposed on the generated texts to distinguish whether LLMs or humans write the texts. Then, we formulate the NS-Watermark as a constrained optimization problem and propose an efficient algorithm to solve it. Through the experiments, we demonstrate that the NS-Watermark can generate more natural texts than existing watermarking methods and distinguish more accurately between texts written by LLMs and those written by humans. Especially in machine translation tasks, the NS-Watermark can outperform the existing watermarking method by up to 30 BLEU scores.

{{</citation>}}


### (80/165) Natural Language Models for Data Visualization Utilizing nvBench Dataset (Shuo Wang et al., 2023)

{{<citation>}}

Shuo Wang, Carlos Crespo-Quinones. (2023)  
**Natural Language Models for Data Visualization Utilizing nvBench Dataset**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, Language Model, T5  
[Paper Link](http://arxiv.org/abs/2310.00832v1)  

---


**ABSTRACT**  
Translation of natural language into syntactically correct commands for data visualization is an important application of natural language models and could be leveraged to many different tasks. A closely related effort is the task of translating natural languages into SQL queries, which in turn could be translated into visualization with additional information from the natural language query supplied\cite{Zhong:2017qr}. Contributing to the progress in this area of research, we built natural language translation models to construct simplified versions of data and visualization queries in a language called Vega Zero. In this paper, we explore the design and performance of these sequence to sequence transformer based machine learning model architectures using large language models such as BERT as encoders to predict visualization commands from natural language queries, as well as apply available T5 sequence to sequence models to the problem for comparison.

{{</citation>}}


## physics.ao-ph (1)



### (81/165) Forecasting Tropical Cyclones with Cascaded Diffusion Models (Pritthijit Nath et al., 2023)

{{<citation>}}

Pritthijit Nath, Pancham Shukla, César Quilodrán-Casas. (2023)  
**Forecasting Tropical Cyclones with Cascaded Diffusion Models**  

---
Primary Category: physics.ao-ph  
Categories: cs-LG, physics-ao-ph, physics.ao-ph  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.01690v1)  

---


**ABSTRACT**  
As cyclones become more intense due to climate change, the rise of AI-based modelling provides a more affordable and accessible approach compared to traditional methods based on mathematical models. This work leverages diffusion models to forecast cyclone trajectories and precipitation patterns by integrating satellite imaging, remote sensing, and atmospheric data, employing a cascaded approach that incorporates forecasting, super-resolution, and precipitation modelling, with training on a dataset of 51 cyclones from six major basins. Experiments demonstrate that the final forecasts from the cascaded models show accurate predictions up to a 36-hour rollout, with SSIM and PSNR values exceeding 0.5 and 20 dB, respectively, for all three tasks. This work also highlights the promising efficiency of AI methods such as diffusion models for high-performance needs, such as cyclone forecasting, while remaining computationally affordable, making them ideal for highly vulnerable regions with critical forecasting needs and financial limitations. Code accessible at \url{https://github.com/nathzi1505/forecast-diffmodels}.

{{</citation>}}


## eess.AS (3)



### (82/165) One model to rule them all ? Towards End-to-End Joint Speaker Diarization and Speech Recognition (Samuele Cornell et al., 2023)

{{<citation>}}

Samuele Cornell, Jee-weon Jung, Shinji Watanabe, Stefano Squartini. (2023)  
**One model to rule them all ? Towards End-to-End Joint Speaker Diarization and Speech Recognition**  

---
Primary Category: eess.AS  
Categories: cs-CL, cs-SD, eess-AS, eess.AS  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2310.01688v1)  

---


**ABSTRACT**  
This paper presents a novel framework for joint speaker diarization (SD) and automatic speech recognition (ASR), named SLIDAR (sliding-window diarization-augmented recognition). SLIDAR can process arbitrary length inputs and can handle any number of speakers, effectively solving ``who spoke what, when'' concurrently. SLIDAR leverages a sliding window approach and consists of an end-to-end diarization-augmented speech transcription (E2E DAST) model which provides, locally, for each window: transcripts, diarization and speaker embeddings. The E2E DAST model is based on an encoder-decoder architecture and leverages recent techniques such as serialized output training and ``Whisper-style" prompting. The local outputs are then combined to get the final SD+ASR result by clustering the speaker embeddings to get global speaker identities. Experiments performed on monaural recordings from the AMI corpus confirm the effectiveness of the method in both close-talk and far-field speech scenarios.

{{</citation>}}


### (83/165) Scaling Up Music Information Retrieval Training with Semi-Supervised Learning (Yun-Ning Hung et al., 2023)

{{<citation>}}

Yun-Ning Hung, Ju-Chiang Wang, Minz Won, Duc Le. (2023)  
**Scaling Up Music Information Retrieval Training with Semi-Supervised Learning**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Information Retrieval, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2310.01353v1)  

---


**ABSTRACT**  
In the era of data-driven Music Information Retrieval (MIR), the scarcity of labeled data has been one of the major concerns to the success of an MIR task. In this work, we leverage the semi-supervised teacher-student training approach to improve MIR tasks. For training, we scale up the unlabeled music data to 240k hours, which is much larger than any public MIR datasets. We iteratively create and refine the pseudo-labels in the noisy teacher-student training process. Knowledge expansion is also explored to iteratively scale up the model sizes from as small as less than 3M to almost 100M parameters. We study the performance correlation between data size and model size in the experiments. By scaling up both model size and training data, our models achieve state-of-the-art results on several MIR tasks compared to models that are either trained in a supervised manner or based on a self-supervised pretrained model. To our knowledge, this is the first attempt to study the effects of scaling up both model and training data for a variety of MIR tasks.

{{</citation>}}


### (84/165) End-to-End Continuous Speech Emotion Recognition in Real-life Customer Service Call Center Conversations (Yajing Feng et al., 2023)

{{<citation>}}

Yajing Feng, Laurence Devillers. (2023)  
**End-to-End Continuous Speech Emotion Recognition in Real-life Customer Service Call Center Conversations**  

---
Primary Category: eess.AS  
Categories: cs-AI, cs-CL, cs-LG, cs-SD, eess-AS, eess.AS  
Keywords: Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2310.02281v1)  

---


**ABSTRACT**  
Speech Emotion recognition (SER) in call center conversations has emerged as a valuable tool for assessing the quality of interactions between clients and agents. In contrast to controlled laboratory environments, real-life conversations take place under uncontrolled conditions and are subject to contextual factors that influence the expression of emotions. In this paper, we present our approach to constructing a large-scale reallife dataset (CusEmo) for continuous SER in customer service call center conversations. We adopted the dimensional emotion annotation approach to capture the subtlety, complexity, and continuity of emotions in real-life call center conversations, while annotating contextual information. The study also addresses the challenges encountered during the application of the End-to-End (E2E) SER system to the dataset, including determining the appropriate label sampling rate and input segment length, as well as integrating contextual information (interlocutor's gender and empathy level) with different weights using multitask learning. The result shows that incorporating the empathy level information improved the model's performance.

{{</citation>}}


## cs.AI (8)



### (85/165) Designing User-Centric Behavioral Interventions to Prevent Dysglycemia with Novel Counterfactual Explanations (Asiful Arefeen et al., 2023)

{{<citation>}}

Asiful Arefeen, Hassan Ghasemzadeh. (2023)  
**Designing User-Centric Behavioral Interventions to Prevent Dysglycemia with Novel Counterfactual Explanations**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-HC, cs-LG, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.01684v1)  

---


**ABSTRACT**  
Maintaining normal blood glucose levels through lifestyle behaviors is central to maintaining health and preventing disease. Frequent exposure to dysglycemia (i.e., abnormal glucose events such as hyperlycemia and hypoglycemia) leads to chronic complications including diabetes, kidney disease and need for dialysis, myocardial infarction, stroke, amputation, and death. Therefore, a tool capable of predicting dysglycemia and offering users actionable feedback about how to make changes in their diet, exercise, and medication to prevent abnormal glycemic events could have significant societal impacts. Counterfactual explanations can provide insights into why a model made a particular prediction by generating hypothetical instances that are similar to the original input but lead to a different prediction outcome. Therefore, counterfactuals can be viewed as a means to design AI-driven health interventions to prevent adverse health outcomes such as dysglycemia. In this paper, we design GlyCoach, a framework for generating counterfactual explanations for glucose control. Leveraging insights from adversarial learning, GlyCoach characterizes the decision boundary for high-dimensional health data and performs a grid search to generate actionable interventions. GlyCoach is unique in integrating prior knowledge about user preferences of plausible explanations into the process of counterfactual generation. We evaluate GlyCoach extensively using two real-world datasets and external simulators from prior studies that predict glucose response. GlyCoach achieves 87\% sensitivity in the simulation-aided validation, surpassing the state-of-the-art techniques for generating counterfactual explanations by at least $10\%$. Besides, counterfactuals from GlyCoach exhibit a $32\%$ improved normalized distance compared to previous research.

{{</citation>}}


### (86/165) Bridging the Gap between Structural and Semantic Similarity in Diverse Planning (Mustafa F. Abdelwahed et al., 2023)

{{<citation>}}

Mustafa F. Abdelwahed, Joan Espasa, Alice Toniolo, Ian P. Gent. (2023)  
**Bridging the Gap between Structural and Semantic Similarity in Diverse Planning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Semantic Similarity  
[Paper Link](http://arxiv.org/abs/2310.01520v1)  

---


**ABSTRACT**  
Diverse planning is the problem of finding multiple plans for a given problem specification, which is at the core of many real-world applications. For example, diverse planning is a critical piece for the efficiency of plan recognition systems when dealing with noisy and missing observations. Providing diverse solutions can also benefit situations where constraints are too expensive or impossible to model. Current diverse planners operate by generating multiple plans and then applying a selection procedure to extract diverse solutions using a similarity metric. Generally, current similarity metrics only consider the structural properties of the given plans. We argue that this approach is a limitation that sometimes prevents such metrics from capturing why two plans differ. In this work, we propose two new domain-independent metrics which are able to capture relevant information on the difference between two given plans from a domain-dependent viewpoint. We showcase their utility in various situations where the currently used metrics fail to capture the similarity between plans, failing to capture some structural symmetries.

{{</citation>}}


### (87/165) Avalon's Game of Thoughts: Battle Against Deception through Recursive Contemplation (Shenzhi Wang et al., 2023)

{{<citation>}}

Shenzhi Wang, Chang Liu, Zilong Zheng, Siyuan Qi, Shuo Chen, Qisen Yang, Andrew Zhao, Chaofei Wang, Shiji Song, Gao Huang. (2023)  
**Avalon's Game of Thoughts: Battle Against Deception through Recursive Contemplation**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-CY, cs-LG, cs-MA, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.01320v1)  

---


**ABSTRACT**  
Recent breakthroughs in large language models (LLMs) have brought remarkable success in the field of LLM-as-Agent. Nevertheless, a prevalent assumption is that the information processed by LLMs is consistently honest, neglecting the pervasive deceptive or misleading information in human society and AI-generated content. This oversight makes LLMs susceptible to malicious manipulations, potentially resulting in detrimental outcomes. This study utilizes the intricate Avalon game as a testbed to explore LLMs' potential in deceptive environments. Avalon, full of misinformation and requiring sophisticated logic, manifests as a "Game-of-Thoughts". Inspired by the efficacy of humans' recursive thinking and perspective-taking in the Avalon game, we introduce a novel framework, Recursive Contemplation (ReCon), to enhance LLMs' ability to identify and counteract deceptive information. ReCon combines formulation and refinement contemplation processes; formulation contemplation produces initial thoughts and speech, while refinement contemplation further polishes them. Additionally, we incorporate first-order and second-order perspective transitions into these processes respectively. Specifically, the first-order allows an LLM agent to infer others' mental states, and the second-order involves understanding how others perceive the agent's mental state. After integrating ReCon with different LLMs, extensive experiment results from the Avalon game indicate its efficacy in aiding LLMs to discern and maneuver around deceptive information without extra fine-tuning and data. Finally, we offer a possible explanation for the efficacy of ReCon and explore the current limitations of LLMs in terms of safety, reasoning, speaking style, and format, potentially furnishing insights for subsequent research.

{{</citation>}}


### (88/165) Pre-training Contextual Location Embeddings in Personal Trajectories via Efficient Hierarchical Location Representations (Chung Park et al., 2023)

{{<citation>}}

Chung Park, Taesan Kim, Junui Hong, Minsung Choi, Jaegul Choo. (2023)  
**Pre-training Contextual Location Embeddings in Personal Trajectories via Efficient Hierarchical Location Representations**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2310.01252v1)  

---


**ABSTRACT**  
Pre-training the embedding of a location generated from human mobility data has become a popular method for location based services. In practice, modeling the location embedding is too expensive, due to the large number of locations to be trained in situations with fine-grained resolution or extensive target regions. Previous studies have handled less than ten thousand distinct locations, which is insufficient in the real-world applications. To tackle this problem, we propose a Geo-Tokenizer, designed to efficiently reduce the number of locations to be trained by representing a location as a combination of several grids at different scales. In the Geo-Tokenizer, a grid at a larger scale shares the common set of grids at smaller scales, which is a key factor in reducing the size of the location vocabulary. The sequences of locations preprocessed with the Geo-Tokenizer are utilized by a causal location embedding model to capture the temporal dependencies of locations. This model dynamically calculates the embedding vector of a target location, which varies depending on its trajectory. In addition, to efficiently pre-train the location embedding model, we propose the Hierarchical Auto-regressive Location Model objective to effectively train decomposed locations in the Geo-Tokenizer. We conducted experiments on two real-world user trajectory datasets using our pre-trained location model. The experimental results show that our model significantly improves the performance of downstream tasks with fewer model parameters compared to existing location embedding methods.

{{</citation>}}


### (89/165) KGEx: Explaining Knowledge Graph Embeddings via Subgraph Sampling and Knowledge Distillation (Vasileios Baltatzis et al., 2023)

{{<citation>}}

Vasileios Baltatzis, Luca Costabello. (2023)  
**KGEx: Explaining Knowledge Graph Embeddings via Subgraph Sampling and Knowledge Distillation**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Embedding, Knowledge Distillation, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2310.01065v1)  

---


**ABSTRACT**  
Despite being the go-to choice for link prediction on knowledge graphs, research on interpretability of knowledge graph embeddings (KGE) has been relatively unexplored. We present KGEx, a novel post-hoc method that explains individual link predictions by drawing inspiration from surrogate models research. Given a target triple to predict, KGEx trains surrogate KGE models that we use to identify important training triples. To gauge the impact of a training triple, we sample random portions of the target triple neighborhood and we train multiple surrogate KGE models on each of them. To ensure faithfulness, each surrogate is trained by distilling knowledge from the original KGE model. We then assess how well surrogates predict the target triple being explained, the intuition being that those leading to faithful predictions have been trained on impactful neighborhood samples. Under this assumption, we then harvest triples that appear frequently across impactful neighborhoods. We conduct extensive experiments on two publicly available datasets, to demonstrate that KGEx is capable of providing explanations faithful to the black-box model.

{{</citation>}}


### (90/165) Towards Fixing Clever-Hans Predictors with Counterfactual Knowledge Distillation (Sidney Bender et al., 2023)

{{<citation>}}

Sidney Bender, Christopher J. Anders, Pattarawatt Chormai, Heike Marxfeld, Jan Herrmann, Grégoire Montavon. (2023)  
**Towards Fixing Clever-Hans Predictors with Counterfactual Knowledge Distillation**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2310.01011v2)  

---


**ABSTRACT**  
This paper introduces a novel technique called counterfactual knowledge distillation (CFKD) to detect and remove reliance on confounders in deep learning models with the help of human expert feedback. Confounders are spurious features that models tend to rely on, which can result in unexpected errors in regulated or safety-critical domains. The paper highlights the benefit of CFKD in such domains and shows some advantages of counterfactual explanations over other types of explanations. We propose an experiment scheme to quantitatively evaluate the success of CFKD and different teachers that can give feedback to the model. We also introduce a new metric that is better correlated with true test performance than validation accuracy. The paper demonstrates the effectiveness of CFKD on synthetically augmented datasets and on real-world histopathological datasets.

{{</citation>}}


### (91/165) Using Reinforcement Learning to Optimize Responses in Care Processes: A Case Study on Aggression Incidents (Bart J. Verhoef et al., 2023)

{{<citation>}}

Bart J. Verhoef, Xixi Lu. (2023)  
**Using Reinforcement Learning to Optimize Responses in Care Processes: A Case Study on Aggression Incidents**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.00981v1)  

---


**ABSTRACT**  
Previous studies have used prescriptive process monitoring to find actionable policies in business processes and conducted case studies in similar domains, such as the loan application process and the traffic fine process. However, care processes tend to be more dynamic and complex. For example, at any stage of a care process, a multitude of actions is possible. In this paper, we follow the reinforcement approach and train a Markov decision process using event data from a care process. The goal was to find optimal policies for staff members when clients are displaying any type of aggressive behavior. We used the reinforcement learning algorithms Q-learning and SARSA to find optimal policies. Results showed that the policies derived from these algorithms are similar to the most frequent actions currently used but provide the staff members with a few more options in certain situations.

{{</citation>}}


### (92/165) All by Myself: Learning Individualized Competitive Behaviour with a Contrastive Reinforcement Learning optimization (Pablo Barros et al., 2023)

{{<citation>}}

Pablo Barros, Alessandra Sciutti. (2023)  
**All by Myself: Learning Individualized Competitive Behaviour with a Contrastive Reinforcement Learning optimization**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.00964v1)  

---


**ABSTRACT**  
In a competitive game scenario, a set of agents have to learn decisions that maximize their goals and minimize their adversaries' goals at the same time. Besides dealing with the increased dynamics of the scenarios due to the opponents' actions, they usually have to understand how to overcome the opponent's strategies. Most of the common solutions, usually based on continual learning or centralized multi-agent experiences, however, do not allow the development of personalized strategies to face individual opponents. In this paper, we propose a novel model composed of three neural layers that learn a representation of a competitive game, learn how to map the strategy of specific opponents, and how to disrupt them. The entire model is trained online, using a composed loss based on a contrastive optimization, to learn competitive and multiplayer games. We evaluate our model on a pokemon duel scenario and the four-player competitive Chef's Hat card game. Our experiments demonstrate that our model achieves better performance when playing against offline, online, and competitive-specific models, in particular when playing against the same opponent multiple times. We also present a discussion on the impact of our model, in particular on how well it deals with on specific strategy learning for each of the two scenarios.

{{</citation>}}


## cs.MA (1)



### (93/165) Solving Two-Player General-Sum Games Between Swarms (Mukesh Ghimire et al., 2023)

{{<citation>}}

Mukesh Ghimire, Lei Zhang, Wenlong Zhang, Yi Ren, Zhe Xu. (2023)  
**Solving Two-Player General-Sum Games Between Swarms**  

---
Primary Category: cs.MA  
Categories: cs-GT, cs-MA, cs.MA  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.01682v1)  

---


**ABSTRACT**  
Hamilton-Jacobi-Isaacs (HJI) PDEs are the governing equations for the two-player general-sum games. Unlike Reinforcement Learning (RL) methods, which are data-intensive methods for learning value function, learning HJ PDEs provide a guaranteed convergence to the Nash Equilibrium value of the game when it exists. However, a caveat is that solving HJ PDEs becomes intractable when the state dimension increases. To circumvent the curse of dimensionality (CoD), physics-informed machine learning methods with supervision can be used and have been shown to be effective in generating equilibrial policies in two-player general-sum games. In this work, we extend the existing work on agent-level two-player games to a two-player swarm-level game, where two sub-swarms play a general-sum game. We consider the \textit{Kolmogorov forward equation} as the dynamic model for the evolution of the densities of the swarms. Results show that policies generated from the physics-informed neural network (PINN) result in a higher payoff than a Nash Double Deep Q-Network (Nash DDQN) agent and have comparable performance with numerical solvers.

{{</citation>}}


## cs.CV (30)



### (94/165) Keypoint-Augmented Self-Supervised Learning for Medical Image Segmentation with Limited Annotation (Zhangsihao Yang et al., 2023)

{{<citation>}}

Zhangsihao Yang, Mengwei Ren, Kaize Ding, Guido Gerig, Yalin Wang. (2023)  
**Keypoint-Augmented Self-Supervised Learning for Medical Image Segmentation with Limited Annotation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Self-Supervised, Transformer  
[Paper Link](http://arxiv.org/abs/2310.01680v1)  

---


**ABSTRACT**  
Pretraining CNN models (i.e., UNet) through self-supervision has become a powerful approach to facilitate medical image segmentation under low annotation regimes. Recent contrastive learning methods encourage similar global representations when the same image undergoes different transformations, or enforce invariance across different image/patch features that are intrinsically correlated. However, CNN-extracted global and local features are limited in capturing long-range spatial dependencies that are essential in biological anatomy. To this end, we present a keypoint-augmented fusion layer that extracts representations preserving both short- and long-range self-attention. In particular, we augment the CNN feature map at multiple scales by incorporating an additional input that learns long-range spatial self-attention among localized keypoint features. Further, we introduce both global and local self-supervised pretraining for the framework. At the global scale, we obtain global representations from both the bottleneck of the UNet, and by aggregating multiscale keypoint features. These global features are subsequently regularized through image-level contrastive objectives. At the local scale, we define a distance-based criterion to first establish correspondences among keypoints and encourage similarity between their features. Through extensive experiments on both MRI and CT segmentation tasks, we demonstrate the architectural advantages of our proposed method in comparison to both CNN and Transformer-based UNets, when all architectures are trained with randomly initialized weights. With our proposed pretraining strategy, our method further outperforms existing SSL methods by producing more robust self-attention and achieving state-of-the-art segmentation results. The code is available at https://github.com/zshyang/kaf.git.

{{</citation>}}


### (95/165) It's all about you: Personalized in-Vehicle Gesture Recognition with a Time-of-Flight Camera (Amr Gomaa et al., 2023)

{{<citation>}}

Amr Gomaa, Guillermo Reyes, Michael Feld. (2023)  
**It's all about you: Personalized in-Vehicle Gesture Recognition with a Time-of-Flight Camera**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-HC, cs.CV  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2310.01659v1)  

---


**ABSTRACT**  
Despite significant advances in gesture recognition technology, recognizing gestures in a driving environment remains challenging due to limited and costly data and its dynamic, ever-changing nature. In this work, we propose a model-adaptation approach to personalize the training of a CNNLSTM model and improve recognition accuracy while reducing data requirements. Our approach contributes to the field of dynamic hand gesture recognition while driving by providing a more efficient and accurate method that can be customized for individual users, ultimately enhancing the safety and convenience of in-vehicle interactions, as well as driver's experience and system trust. We incorporate hardware enhancement using a time-of-flight camera and algorithmic enhancement through data augmentation, personalized adaptation, and incremental learning techniques. We evaluate the performance of our approach in terms of recognition accuracy, achieving up to 90\%, and show the effectiveness of personalized adaptation and incremental learning for a user-centered design.

{{</citation>}}


### (96/165) Adaptive Visual Scene Understanding: Incremental Scene Graph Generation (Naitik Khandelwal et al., 2023)

{{<citation>}}

Naitik Khandelwal, Xiao Liu, Mengmi Zhang. (2023)  
**Adaptive Visual Scene Understanding: Incremental Scene Graph Generation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.01636v1)  

---


**ABSTRACT**  
Scene graph generation (SGG) involves analyzing images to extract meaningful information about objects and their relationships. Given the dynamic nature of the visual world, it becomes crucial for AI systems to detect new objects and establish their new relationships with existing objects. To address the lack of continual learning methodologies in SGG, we introduce the comprehensive Continual ScenE Graph Generation (CSEGG) dataset along with 3 learning scenarios and 8 evaluation metrics. Our research investigates the continual learning performances of existing SGG methods on the retention of previous object entities and relationships as they learn new ones. Moreover, we also explore how continual object detection enhances generalization in classifying known relationships on unknown objects. We conduct extensive experiments benchmarking and analyzing the classical two-stage SGG methods and the most recent transformer-based SGG methods in continual learning settings, and gain valuable insights into the CSEGG problem. We invite the research community to explore this emerging field of study.

{{</citation>}}


### (97/165) Dynamic Spatio-Temporal Summarization using Information Based Fusion (Humayra Tasnim et al., 2023)

{{<citation>}}

Humayra Tasnim, Soumya Dutta, Melanie Moses. (2023)  
**Dynamic Spatio-Temporal Summarization using Information Based Fusion**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-IT, cs.CV, math-IT  
Keywords: Summarization  
[Paper Link](http://arxiv.org/abs/2310.01617v1)  

---


**ABSTRACT**  
In the era of burgeoning data generation, managing and storing large-scale time-varying datasets poses significant challenges. With the rise of supercomputing capabilities, the volume of data produced has soared, intensifying storage and I/O overheads. To address this issue, we propose a dynamic spatio-temporal data summarization technique that identifies informative features in key timesteps and fuses less informative ones. This approach minimizes storage requirements while preserving data dynamics. Unlike existing methods, our method retains both raw and summarized timesteps, ensuring a comprehensive view of information changes over time. We utilize information-theoretic measures to guide the fusion process, resulting in a visual representation that captures essential data patterns. We demonstrate the versatility of our technique across diverse datasets, encompassing particle-based flow simulations, security and surveillance applications, and biological cell interactions within the immune system. Our research significantly contributes to the realm of data management, introducing enhanced efficiency and deeper insights across diverse multidisciplinary domains. We provide a streamlined approach for handling massive datasets that can be applied to in situ analysis as well as post hoc analysis. This not only addresses the escalating challenges of data storage and I/O overheads but also unlocks the potential for informed decision-making. Our method empowers researchers and experts to explore essential temporal dynamics while minimizing storage requirements, thereby fostering a more effective and intuitive understanding of complex data behaviors.

{{</citation>}}


### (98/165) GPT-Driver: Learning to Drive with GPT (Jiageng Mao et al., 2023)

{{<citation>}}

Jiageng Mao, Yuxi Qian, Hang Zhao, Yue Wang. (2023)  
**GPT-Driver: Learning to Drive with GPT**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs-RO, cs.CV  
Keywords: AI, GPT, GPT-3.5, Language Model  
[Paper Link](http://arxiv.org/abs/2310.01415v1)  

---


**ABSTRACT**  
We present a simple yet effective approach that can transform the OpenAI GPT-3.5 model into a reliable motion planner for autonomous vehicles. Motion planning is a core challenge in autonomous driving, aiming to plan a driving trajectory that is safe and comfortable. Existing motion planners predominantly leverage heuristic methods to forecast driving trajectories, yet these approaches demonstrate insufficient generalization capabilities in the face of novel and unseen driving scenarios. In this paper, we propose a novel approach to motion planning that capitalizes on the strong reasoning capabilities and generalization potential inherent to Large Language Models (LLMs). The fundamental insight of our approach is the reformulation of motion planning as a language modeling problem, a perspective not previously explored. Specifically, we represent the planner inputs and outputs as language tokens, and leverage the LLM to generate driving trajectories through a language description of coordinate positions. Furthermore, we propose a novel prompting-reasoning-finetuning strategy to stimulate the numerical reasoning potential of the LLM. With this strategy, the LLM can describe highly precise trajectory coordinates and also its internal decision-making process in natural language. We evaluate our approach on the large-scale nuScenes dataset, and extensive experiments substantiate the effectiveness, generalization ability, and interpretability of our GPT-based motion planner. Code will be released upon acceptance.

{{</citation>}}


### (99/165) DriveGPT4: Interpretable End-to-end Autonomous Driving via Large Language Model (Zhenhua Xu et al., 2023)

{{<citation>}}

Zhenhua Xu, Yujia Zhang, Enze Xie, Zhen Zhao, Yong Guo, Kenneth K. Y. Wong, Zhenguo Li, Hengshuang Zhao. (2023)  
**DriveGPT4: Interpretable End-to-end Autonomous Driving via Large Language Model**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.01412v1)  

---


**ABSTRACT**  
In the past decade, autonomous driving has experienced rapid development in both academia and industry. However, its limited interpretability remains a significant unsolved problem, severely hindering autonomous vehicle commercialization and further development. Previous approaches utilizing small language models have failed to address this issue due to their lack of flexibility, generalization ability, and robustness. Recently, multimodal large language models (LLMs) have gained considerable attention from the research community for their capability to process and reason non-text data (e.g., images and videos) by text. In this paper, we present DriveGPT4, an interpretable end-to-end autonomous driving system utilizing LLMs. DriveGPT4 is capable of interpreting vehicle actions and providing corresponding reasoning, as well as answering diverse questions posed by human users for enhanced interaction. Additionally, DriveGPT4 predicts vehicle low-level control signals in an end-to-end fashion. These capabilities stem from a customized visual instruction tuning dataset specifically designed for autonomous driving. To the best of our knowledge, DriveGPT4 is the first work focusing on interpretable end-to-end autonomous driving. When evaluated on multiple tasks alongside conventional methods and video understanding LLMs, DriveGPT4 demonstrates superior qualitative and quantitative performance. Additionally, DriveGPT4 can be generalized in a zero-shot fashion to accommodate more unseen scenarios. The project page is available at https://tonyxuqaq.github.io/projects/DriveGPT4/ .

{{</citation>}}


### (100/165) CLIPSelf: Vision Transformer Distills Itself for Open-Vocabulary Dense Prediction (Size Wu et al., 2023)

{{<citation>}}

Size Wu, Wenwei Zhang, Lumin Xu, Sheng Jin, Xiangtai Li, Wentao Liu, Chen Change Loy. (2023)  
**CLIPSelf: Vision Transformer Distills Itself for Open-Vocabulary Dense Prediction**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.01403v1)  

---


**ABSTRACT**  
Open-vocabulary dense prediction tasks including object detection and image segmentation have been advanced by the success of Contrastive Language-Image Pre-training (CLIP). CLIP models, particularly those incorporating vision transformers (ViTs), have exhibited remarkable generalization ability in zero-shot image classification. However, when transferring the vision-language alignment of CLIP from global image representation to local region representation for the open-vocabulary dense prediction tasks, CLIP ViTs suffer from the domain shift from full images to local image regions. In this paper, we embark on an in-depth analysis of the region-language alignment in CLIP models, which is essential for downstream open-vocabulary dense prediction tasks. Subsequently, we propose an approach named CLIPSelf, which adapts the image-level recognition ability of CLIP ViT to local image regions without needing any region-text pairs. CLIPSelf empowers ViTs to distill itself by aligning a region representation extracted from its dense feature map with the image-level representation of the corresponding image crop. With the enhanced CLIP ViTs, we achieve new state-of-the-art performance on open-vocabulary object detection, semantic segmentation, and panoptic segmentation across various benchmarks. Models and code will be available at https://github.com/wusize/CLIPSelf.

{{</citation>}}


### (101/165) Pixel-Aligned Recurrent Queries for Multi-View 3D Object Detection (Yiming Xie et al., 2023)

{{<citation>}}

Yiming Xie, Huaizu Jiang, Georgia Gkioxari, Julian Straub. (2023)  
**Pixel-Aligned Recurrent Queries for Multi-View 3D Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2310.01401v1)  

---


**ABSTRACT**  
We present PARQ - a multi-view 3D object detector with transformer and pixel-aligned recurrent queries. Unlike previous works that use learnable features or only encode 3D point positions as queries in the decoder, PARQ leverages appearance-enhanced queries initialized from reference points in 3D space and updates their 3D location with recurrent cross-attention operations. Incorporating pixel-aligned features and cross attention enables the model to encode the necessary 3D-to-2D correspondences and capture global contextual information of the input images. PARQ outperforms prior best methods on the ScanNet and ARKitScenes datasets, learns and detects faster, is more robust to distribution shifts in reference points, can leverage additional input views without retraining, and can adapt inference compute by changing the number of recurrent iterations.

{{</citation>}}


### (102/165) DST-Det: Simple Dynamic Self-Training for Open-Vocabulary Object Detection (Shilin Xu et al., 2023)

{{<citation>}}

Shilin Xu, Xiangtai Li, Size Wu, Wenwei Zhang, Yining Li, Guangliang Cheng, Yunhai Tong, Kai Chen, Chen Change Loy. (2023)  
**DST-Det: Simple Dynamic Self-Training for Open-Vocabulary Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2310.01393v1)  

---


**ABSTRACT**  
Open-vocabulary object detection (OVOD) aims to detect the objects beyond the set of categories observed during training. This work presents a simple yet effective strategy that leverages the zero-shot classification ability of pre-trained vision-language models (VLM), such as CLIP, to classify proposals for all possible novel classes directly. Unlike previous works that ignore novel classes during training and rely solely on the region proposal network (RPN) for novel object detection, our method selectively filters proposals based on specific design criteria. The resulting sets of identified proposals serve as pseudo-labels for novel classes during the training phase. It enables our self-training strategy to improve the recall and accuracy of novel classes in a self-training manner without requiring additional annotations or datasets. We further propose a simple offline pseudo-label generation strategy to refine the object detector. Empirical evaluations on three datasets, including LVIS, V3Det, and COCO, demonstrate significant improvements over the baseline performance without incurring additional parameters or computational costs during inference. In particular, compared with previous F-VLM, our method achieves a 1.7-2.0% improvement on LVIS dataset and 2.3-3.8% improvement on the recent challenging V3Det dataset. Our method also boosts the strong baseline by 6% mAP on COCO. The code and models will be publicly available at https://github.com/xushilin1/dst-det.

{{</citation>}}


### (103/165) EXTRACTER: Efficient Texture Matching with Attention and Gradient Enhancing for Large Scale Image Super Resolution (Esteban Reyes-Saldana et al., 2023)

{{<citation>}}

Esteban Reyes-Saldana, Mariano Rivera. (2023)  
**EXTRACTER: Efficient Texture Matching with Attention and Gradient Enhancing for Large Scale Image Super Resolution**  

---
Primary Category: cs.CV  
Categories: I-4-3, cs-AI, cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2310.01379v1)  

---


**ABSTRACT**  
Recent Reference-Based image super-resolution (RefSR) has improved SOTA deep methods introducing attention mechanisms to enhance low-resolution images by transferring high-resolution textures from a reference high-resolution image. The main idea is to search for matches between patches using LR and Reference image pair in a feature space and merge them using deep architectures. However, existing methods lack the accurate search of textures. They divide images into as many patches as possible, resulting in inefficient memory usage, and cannot manage large images. Herein, we propose a deep search with a more efficient memory usage that reduces significantly the number of image patches and finds the $k$ most relevant texture match for each low-resolution patch over the high-resolution reference patches, resulting in an accurate texture match. We enhance the Super Resolution result adding gradient density information using a simple residual architecture showing competitive metrics results: PSNR and SSMI.

{{</citation>}}


### (104/165) NEUCORE: Neural Concept Reasoning for Composed Image Retrieval (Shu Zhao et al., 2023)

{{<citation>}}

Shu Zhao, Huijuan Xu. (2023)  
**NEUCORE: Neural Concept Reasoning for Composed Image Retrieval**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2310.01358v1)  

---


**ABSTRACT**  
Composed image retrieval which combines a reference image and a text modifier to identify the desired target image is a challenging task, and requires the model to comprehend both vision and language modalities and their interactions. Existing approaches focus on holistic multi-modal interaction modeling, and ignore the composed and complimentary property between the reference image and text modifier. In order to better utilize the complementarity of multi-modal inputs for effective information fusion and retrieval, we move the multi-modal understanding to fine-granularity at concept-level, and learn the multi-modal concept alignment to identify the visual location in reference or target images corresponding to text modifier. Toward the end, we propose a NEUral COncept REasoning (NEUCORE) model which incorporates multi-modal concept alignment and progressive multimodal fusion over aligned concepts. Specifically, considering that text modifier may refer to semantic concepts not existing in the reference image and requiring to be added into the target image, we learn the multi-modal concept alignment between the text modifier and the concatenation of reference and target images, under multiple-instance learning framework with image and sentence level weak supervision. Furthermore, based on aligned concepts, to form discriminative fusion features of the input modalities for accurate target image retrieval, we propose a progressive fusion strategy with unified execution architecture instantiated by the attended language semantic concepts. Our proposed approach is evaluated on three datasets and achieves state-of-the-art results.

{{</citation>}}


### (105/165) Less is More: Toward Zero-Shot Local Scene Graph Generation via Foundation Models (Shu Zhao et al., 2023)

{{<citation>}}

Shu Zhao, Huijuan Xu. (2023)  
**Less is More: Toward Zero-Shot Local Scene Graph Generation via Foundation Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2310.01356v1)  

---


**ABSTRACT**  
Humans inherently recognize objects via selective visual perception, transform specific regions from the visual field into structured symbolic knowledge, and reason their relationships among regions based on the allocation of limited attention resources in line with humans' goals. While it is intuitive for humans, contemporary perception systems falter in extracting structural information due to the intricate cognitive abilities and commonsense knowledge required. To fill this gap, we present a new task called Local Scene Graph Generation. Distinct from the conventional scene graph generation task, which encompasses generating all objects and relationships in an image, our proposed task aims to abstract pertinent structural information with partial objects and their relationships for boosting downstream tasks that demand advanced comprehension and reasoning capabilities. Correspondingly, we introduce zEro-shot Local scEne GrAph geNeraTion (ELEGANT), a framework harnessing foundation models renowned for their powerful perception and commonsense reasoning, where collaboration and information communication among foundation models yield superior outcomes and realize zero-shot local scene graph generation without requiring labeled supervision. Furthermore, we propose a novel open-ended evaluation metric, Entity-level CLIPScorE (ECLIPSE), surpassing previous closed-set evaluation metrics by transcending their limited label space, offering a broader assessment. Experiment results show that our approach markedly outperforms baselines in the open-ended evaluation setting, and it also achieves a significant performance boost of up to 24.58% over prior methods in the close-set setting, demonstrating the effectiveness and powerful reasoning ability of our proposed framework.

{{</citation>}}


### (106/165) ZeroI2V: Zero-Cost Adaptation of Pre-trained Transformers from Image to Video (Xinhao Li et al., 2023)

{{<citation>}}

Xinhao Li, Limin Wang. (2023)  
**ZeroI2V: Zero-Cost Adaptation of Pre-trained Transformers from Image to Video**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.01324v1)  

---


**ABSTRACT**  
Adapting image models to video domain is becoming an efficient paradigm for solving video recognition tasks. Due to the huge number of parameters and effective transferability of image models, performing full fine-tuning is less efficient and even unnecessary. Thus, recent research is shifting its focus towards parameter-efficient image-to-video adaptation. However, these adaptation strategies inevitably introduce extra computational cost to deal with the domain gap and temporal modeling in videos. In this paper, our goal is to present a zero-cost adaptation paradigm (ZeroI2V) to transfer the image transformers to video recognition tasks (i.e., introduce zero extra cost to the adapted models during inference). To achieve this goal, we present two core designs. First, to capture the dynamics in videos and reduce the difficulty of achieving image-to-video adaptation, we exploit the flexibility of self-attention and introduce the spatial-temporal dual-headed attention (STDHA) that efficiently endow the image transformers with temporal modeling capability at zero extra parameters and computation. Second, to handle the domain gap between images and videos, we propose a linear adaption strategy which utilizes lightweight densely placed linear adapters to fully transfer the frozen image models to video recognition. Due to its customized linear design, all newly added adapters could be easily merged with the original modules through structural reparameterization after training, thus achieving zero extra cost during inference. Extensive experiments on four widely-used video recognition benchmarks show that our ZeroI2V can match or even outperform previous state-of-the-art methods while enjoying superior parameter and inference efficiency.

{{</citation>}}


### (107/165) Color and Texture Dual Pipeline Lightweight Style Transfer (ShiQi Jiang, 2023)

{{<citation>}}

ShiQi Jiang. (2023)  
**Color and Texture Dual Pipeline Lightweight Style Transfer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Style Transfer  
[Paper Link](http://arxiv.org/abs/2310.01321v1)  

---


**ABSTRACT**  
Style transfer methods typically generate a single stylized output of color and texture coupling for reference styles, and color transfer schemes may introduce distortion or artifacts when processing reference images with duplicate textures. To solve the problem, we propose a Color and Texture Dual Pipeline Lightweight Style Transfer CTDP method, which employs a dual pipeline method to simultaneously output the results of color and texture transfer. Furthermore, we designed a masked total variation loss to suppress artifacts and small texture representations in color transfer results without affecting the semantic part of the content. More importantly, we are able to add texture structures with controllable intensity to color transfer results for the first time. Finally, we conducted feature visualization analysis on the texture generation mechanism of the framework and found that smoothing the input image can almost completely eliminate this texture structure. In comparative experiments, the color and texture transfer results generated by CTDP both achieve state-of-the-art performance. Additionally, the weight of the color transfer branch model size is as low as 20k, which is 100-1500 times smaller than that of other state-of-the-art models.

{{</citation>}}


### (108/165) Efficient Remote Sensing Segmentation With Generative Adversarial Transformer (Luyi Qiu et al., 2023)

{{<citation>}}

Luyi Qiu, Dayu Yu, Xiaofeng Zhang, Chenxiao Zhang. (2023)  
**Efficient Remote Sensing Segmentation With Generative Adversarial Transformer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV, eess-IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.01292v1)  

---


**ABSTRACT**  
Most deep learning methods that achieve high segmentation accuracy require deep network architectures that are too heavy and complex to run on embedded devices with limited storage and memory space. To address this issue, this paper proposes an efficient Generative Adversarial Transfomer (GATrans) for achieving high-precision semantic segmentation while maintaining an extremely efficient size. The framework utilizes a Global Transformer Network (GTNet) as the generator, efficiently extracting multi-level features through residual connections. GTNet employs global transformer blocks with progressively linear computational complexity to reassign global features based on a learnable similarity function. To focus on object-level and pixel-level information, the GATrans optimizes the objective function by combining structural similarity losses. We validate the effectiveness of our approach through extensive experiments on the Vaihingen dataset, achieving an average F1 score of 90.17% and an overall accuracy of 91.92%.

{{</citation>}}


### (109/165) Generating 3D Brain Tumor Regions in MRI using Vector-Quantization Generative Adversarial Networks (Meng Zhou et al., 2023)

{{<citation>}}

Meng Zhou, Matthias W Wagner, Uri Tabori, Cynthia Hawkins, Birgit B Ertl-Wagner, Farzad Khalvati. (2023)  
**Generating 3D Brain Tumor Regions in MRI using Vector-Quantization Generative Adversarial Networks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2310.01251v1)  

---


**ABSTRACT**  
Medical image analysis has significantly benefited from advancements in deep learning, particularly in the application of Generative Adversarial Networks (GANs) for generating realistic and diverse images that can augment training datasets. However, the effectiveness of such approaches is often limited by the amount of available data in clinical settings. Additionally, the common GAN-based approach is to generate entire image volumes, rather than solely the region of interest (ROI). Research on deep learning-based brain tumor classification using MRI has shown that it is easier to classify the tumor ROIs compared to the entire image volumes. In this work, we present a novel framework that uses vector-quantization GAN and a transformer incorporating masked token modeling to generate high-resolution and diverse 3D brain tumor ROIs that can be directly used as augmented data for the classification of brain tumor ROI. We apply our method to two imbalanced datasets where we augment the minority class: (1) the Multimodal Brain Tumor Segmentation Challenge (BraTS) 2019 dataset to generate new low-grade glioma (LGG) ROIs to balance with high-grade glioma (HGG) class; (2) the internal pediatric LGG (pLGG) dataset tumor ROIs with BRAF V600E Mutation genetic marker to balance with BRAF Fusion genetic marker class. We show that the proposed method outperforms various baseline models in both qualitative and quantitative measurements. The generated data was used to balance the data in the brain tumor types classification task. Using the augmented data, our approach surpasses baseline models by 6.4% in AUC on the BraTS 2019 dataset and 4.3% in AUC on our internal pLGG dataset. The results indicate the generated tumor ROIs can effectively address the imbalanced data problem. Our proposed method has the potential to facilitate an accurate diagnosis of rare brain tumors using MRI scans.

{{</citation>}}


### (110/165) Making LLaMA SEE and Draw with SEED Tokenizer (Yuying Ge et al., 2023)

{{<citation>}}

Yuying Ge, Sijie Zhao, Ziyun Zeng, Yixiao Ge, Chen Li, Xintao Wang, Ying Shan. (2023)  
**Making LLaMA SEE and Draw with SEED Tokenizer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, LLaMA, Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2310.01218v1)  

---


**ABSTRACT**  
The great success of Large Language Models (LLMs) has expanded the potential of multimodality, contributing to the gradual evolution of General Artificial Intelligence (AGI). A true AGI agent should not only possess the capability to perform predefined multi-tasks but also exhibit emergent abilities in an open-world context. However, despite the considerable advancements made by recent multimodal LLMs, they still fall short in effectively unifying comprehension and generation tasks, let alone open-world emergent abilities. We contend that the key to overcoming the present impasse lies in enabling text and images to be represented and processed interchangeably within a unified autoregressive Transformer. To this end, we introduce SEED, an elaborate image tokenizer that empowers LLMs with the ability to SEE and Draw at the same time. We identify two crucial design principles: (1) Image tokens should be independent of 2D physical patch positions and instead be produced with a 1D causal dependency, exhibiting intrinsic interdependence that aligns with the left-to-right autoregressive prediction mechanism in LLMs. (2) Image tokens should capture high-level semantics consistent with the degree of semantic abstraction in words, and be optimized for both discriminativeness and reconstruction during the tokenizer training phase. With SEED tokens, LLM is able to perform scalable multimodal autoregression under its original training recipe, i.e., next-word prediction. SEED-LLaMA is therefore produced by large-scale pretraining and instruction tuning on the interleaved textual and visual data, demonstrating impressive performance on a broad range of multimodal comprehension and generation tasks. More importantly, SEED-LLaMA has exhibited compositional emergent abilities such as multi-turn in-context multimodal generation, acting like your AI assistant.

{{</citation>}}


### (111/165) Self-distilled Masked Attention guided masked image modeling with noise Regularized Teacher (SMART) for medical image analysis (Jue Jiang et al., 2023)

{{<citation>}}

Jue Jiang, Harini Veeraraghavan. (2023)  
**Self-distilled Masked Attention guided masked image modeling with noise Regularized Teacher (SMART) for medical image analysis**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2310.01209v1)  

---


**ABSTRACT**  
Hierarchical shifted window transformers (Swin) are a computationally efficient and more accurate alternative to plain vision transformers. Masked image modeling (MIM)-based pretraining is highly effective in increasing models' transferability to a variety of downstream tasks. However, more accurate and efficient attention guided MIM approaches are difficult to implement with Swin due to it's lack of an explicit global attention. We thus architecturally enhanced Swin with semantic class attention for self-supervised attention guided co-distillation with MIM. We also introduced a noise injected momentum teacher, implemented with patch dropout of teacher's inputs for improved training regularization and accuracy. Our approach, called \underline{s}elf-distilled \underline{m}asked \underline{a}ttention MIM with noise \underline{r}egularized \underline{t}eacher (SMART) was pretrained with \textbf{10,412} unlabeled 3D computed tomography (CT)s of multiple disease sites and sourced from institutional and public datasets. We evaluated SMART for multiple downstream tasks involving analysis of 3D CTs of lung cancer (LC) patients for: (i) [Task I] predicting immunotherapy response in advanced stage LC (n = 200 internal dataset), (ii) [Task II] predicting LC recurrence in early stage LC before surgery (n = 156 public dataset), (iii) [Task III] LC segmentation (n = 200 internal, 21 public dataset), and (iv) [Task IV] unsupervised clustering of organs in the chest and abdomen (n = 1,743 public dataset) \underline{without} finetuning. SMART predicted immunotherapy response with an AUC of 0.916, LC recurrence with an AUC of 0.793, segmented LC with Dice accuracy of 0.81, and clustered organs with an inter-class cluster distance of 5.94, indicating capability of attention guided MIM for Swin in medical image analysis.

{{</citation>}}


### (112/165) Strength in Diversity: Multi-Branch Representation Learning for Vehicle Re-Identification (Eurico Almeida et al., 2023)

{{<citation>}}

Eurico Almeida, Bruno Silva, Jorge Batista. (2023)  
**Strength in Diversity: Multi-Branch Representation Learning for Vehicle Re-Identification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2310.01129v1)  

---


**ABSTRACT**  
This paper presents an efficient and lightweight multi-branch deep architecture to improve vehicle re-identification (V-ReID). While most V-ReID work uses a combination of complex multi-branch architectures to extract robust and diversified embeddings towards re-identification, we advocate that simple and lightweight architectures can be designed to fulfill the Re-ID task without compromising performance.   We propose a combination of Grouped-convolution and Loss-Branch-Split strategies to design a multi-branch architecture that improve feature diversity and feature discriminability. We combine a ResNet50 global branch architecture with a BotNet self-attention branch architecture, both designed within a Loss-Branch-Split (LBS) strategy. We argue that specialized loss-branch-splitting helps to improve re-identification tasks by generating specialized re-identification features. A lightweight solution using grouped convolution is also proposed to mimic the learning of loss-splitting into multiple embeddings while significantly reducing the model size. In addition, we designed an improved solution to leverage additional metadata, such as camera ID and pose information, that uses 97% less parameters, further improving re-identification performance.   In comparison to state-of-the-art (SoTA) methods, our approach outperforms competing solutions in Veri-776 by achieving 85.6% mAP and 97.7% CMC1 and obtains competitive results in Veri-Wild with 88.1% mAP and 96.3% CMC1. Overall, our work provides important insights into improving vehicle re-identification and presents a strong basis for other retrieval tasks. Our code is available at the https://github.com/videturfortuna/vehicle_reid_itsc2023.

{{</citation>}}


### (113/165) Ground-A-Video: Zero-shot Grounded Video Editing using Text-to-image Diffusion Models (Hyeonho Jeong et al., 2023)

{{<citation>}}

Hyeonho Jeong, Jong Chul Ye. (2023)  
**Ground-A-Video: Zero-shot Grounded Video Editing using Text-to-image Diffusion Models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV, stat-ML  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2310.01107v1)  

---


**ABSTRACT**  
Recent endeavors in video editing have showcased promising results in single-attribute editing or style transfer tasks, either by training text-to-video (T2V) models on text-video data or adopting training-free methods. However, when confronted with the complexities of multi-attribute editing scenarios, they exhibit shortcomings such as omitting or overlooking intended attribute changes, modifying the wrong elements of the input video, and failing to preserve regions of the input video that should remain intact. To address this, here we present a novel grounding-guided video-to-video translation framework called Ground-A-Video for multi-attribute video editing. Ground-A-Video attains temporally consistent multi-attribute editing of input videos in a training-free manner without aforementioned shortcomings. Central to our method is the introduction of Cross-Frame Gated Attention which incorporates groundings information into the latent representations in a temporally consistent fashion, along with Modulated Cross-Attention and optical flow guided inverted latents smoothing. Extensive experiments and applications demonstrate that Ground-A-Video's zero-shot capacity outperforms other baseline methods in terms of edit-accuracy and frame consistency. Further results and codes are provided at our project page (http://ground-a-video.github.io).

{{</citation>}}


### (114/165) Leveraging Cutting Edge Deep Learning Based Image Matching for Reconstructing a Large Scene from Sparse Images (Georg Bökman et al., 2023)

{{<citation>}}

Georg Bökman, Johan Edstedt. (2023)  
**Leveraging Cutting Edge Deep Learning Based Image Matching for Reconstructing a Large Scene from Sparse Images**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.01092v1)  

---


**ABSTRACT**  
We present the top ranked solution for the AISG-SLA Visual Localisation Challenge benchmark (IJCAI 2023), where the task is to estimate relative motion between images taken in sequence by a camera mounted on a car driving through an urban scene.   For matching images we use our recent deep learning based matcher RoMa. Matching image pairs sequentially and estimating relative motion from point correspondences sampled by RoMa already gives very competitive results -- third rank on the challenge benchmark.   To improve the estimations we extract keypoints in the images, match them using RoMa, and perform structure from motion reconstruction using COLMAP. We choose our recent DeDoDe keypoints for their high repeatability. Further, we address time jumps in the image sequence by matching specific non-consecutive image pairs based on image retrieval with DINOv2. These improvements yield a solution beating all competitors.   We further present a loose upper bound on the accuracy obtainable by the image retrieval approach by also matching hand-picked non-consecutive pairs.

{{</citation>}}


### (115/165) Learnable Cross-modal Knowledge Distillation for Multi-modal Learning with Missing Modality (Hu Wang et al., 2023)

{{<citation>}}

Hu Wang, Yuanhong Chen, Congbo Ma, Jodie Avery, Louise Hull, Gustavo Carneiro. (2023)  
**Learnable Cross-modal Knowledge Distillation for Multi-modal Learning with Missing Modality**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2310.01035v1)  

---


**ABSTRACT**  
The problem of missing modalities is both critical and non-trivial to be handled in multi-modal models. It is common for multi-modal tasks that certain modalities contribute more compared to other modalities, and if those important modalities are missing, the model performance drops significantly. Such fact remains unexplored by current multi-modal approaches that recover the representation from missing modalities by feature reconstruction or blind feature aggregation from other modalities, instead of extracting useful information from the best performing modalities. In this paper, we propose a Learnable Cross-modal Knowledge Distillation (LCKD) model to adaptively identify important modalities and distil knowledge from them to help other modalities from the cross-modal perspective for solving the missing modality issue. Our approach introduces a teacher election procedure to select the most ``qualified'' teachers based on their single modality performance on certain tasks. Then, cross-modal knowledge distillation is performed between teacher and student modalities for each task to push the model parameters to a point that is beneficial for all tasks. Hence, even if the teacher modalities for certain tasks are missing during testing, the available student modalities can accomplish the task well enough based on the learned knowledge from their automatically elected teacher modalities. Experiments on the Brain Tumour Segmentation Dataset 2018 (BraTS2018) shows that LCKD outperforms other methods by a considerable margin, improving the state-of-the-art performance by 3.61% for enhancing tumour, 5.99% for tumour core, and 3.76% for whole tumour in terms of segmentation Dice score.

{{</citation>}}


### (116/165) Incorporating Supervised Domain Generalization into Data Augmentation (Shohei Enomoto et al., 2023)

{{<citation>}}

Shohei Enomoto, Monikka Roslianna Busto, Takeharu Eda. (2023)  
**Incorporating Supervised Domain Generalization into Data Augmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2310.01029v1)  

---


**ABSTRACT**  
With the increasing utilization of deep learning in outdoor settings, its robustness needs to be enhanced to preserve accuracy in the face of distribution shifts, such as compression artifacts. Data augmentation is a widely used technique to improve robustness, thanks to its ease of use and numerous benefits. However, it requires more training epochs, making it difficult to train large models with limited computational resources. To address this problem, we treat data augmentation as supervised domain generalization~(SDG) and benefit from the SDG method, contrastive semantic alignment~(CSA) loss, to improve the robustness and training efficiency of data augmentation. The proposed method only adds loss during model training and can be used as a plug-in for existing data augmentation methods. Experiments on the CIFAR-100 and CUB datasets show that the proposed method improves the robustness and training efficiency of typical data augmentations.

{{</citation>}}


### (117/165) A New Real-World Video Dataset for the Comparison of Defogging Algorithms (Alexandra Duminil et al., 2023)

{{<citation>}}

Alexandra Duminil, Jean-Philippe Tarel, Roland Brémond. (2023)  
**A New Real-World Video Dataset for the Comparison of Defogging Algorithms**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.01020v1)  

---


**ABSTRACT**  
Video restoration for noise removal, deblurring or super-resolution is attracting more and more attention in the fields of image processing and computer vision. Works on video restoration with data-driven approaches for fog removal are rare however, due to the lack of datasets containing videos in both clear and foggy conditions which are required for deep learning and benchmarking. A new dataset, called REVIDE, was recently proposed for just that purpose. In this paper, we implement the same approach by proposing a new REal-world VIdeo dataset for the comparison of Defogging Algorithms (VIREDA), with various fog densities and ground truths without fog. This small database can serve as a test base for defogging algorithms. A video defogging algorithm is also mentioned (still under development), with the key idea of using temporal redundancy to minimize artefacts and exposure variations between frames. Inspired by the success of Transformers architecture in deep learning for various applications, we select this kind of architecture in a neural network to show the relevance of the proposed dataset.

{{</citation>}}


### (118/165) Controlling Vision-Language Models for Universal Image Restoration (Ziwei Luo et al., 2023)

{{<citation>}}

Ziwei Luo, Fredrik K. Gustafsson, Zheng Zhao, Jens Sjölund, Thomas B. Schön. (2023)  
**Controlling Vision-Language Models for Universal Image Restoration**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.01018v1)  

---


**ABSTRACT**  
Vision-language models such as CLIP have shown great impact on diverse downstream tasks for zero-shot or label-free predictions. However, when it comes to low-level vision such as image restoration their performance deteriorates dramatically due to corrupted inputs. In this paper, we present a degradation-aware vision-language model (DA-CLIP) to better transfer pretrained vision-language models to low-level vision tasks as a universal framework for image restoration. More specifically, DA-CLIP trains an additional controller that adapts the fixed CLIP image encoder to predict high-quality feature embeddings. By integrating the embedding into an image restoration network via cross-attention, we are able to pilot the model to learn a high-fidelity image reconstruction. The controller itself will also output a degradation feature that matches the real corruptions of the input, yielding a natural classifier for different degradation types. In addition, we construct a mixed degradation dataset with synthetic captions for DA-CLIP training. Our approach advances state-of-the-art performance on both degradation-specific and unified image restoration tasks, showing a promising direction of prompting image restoration with large-scale pretrained vision-language models. Our code is available at https://github.com/Algolzw/daclip-uir.

{{</citation>}}


### (119/165) LS-VOS: Identifying Outliers in 3D Object Detections Using Latent Space Virtual Outlier Synthesis (Aldi Piroli et al., 2023)

{{<citation>}}

Aldi Piroli, Vinzenz Dallabetta, Johannes Kopp, Marc Walessa, Daniel Meissner, Klaus Dietmayer. (2023)  
**LS-VOS: Identifying Outliers in 3D Object Detections Using Latent Space Virtual Outlier Synthesis**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2310.00952v1)  

---


**ABSTRACT**  
LiDAR-based 3D object detectors have achieved unprecedented speed and accuracy in autonomous driving applications. However, similar to other neural networks, they are often biased toward high-confidence predictions or return detections where no real object is present. These types of detections can lead to a less reliable environment perception, severely affecting the functionality and safety of autonomous vehicles. We address this problem by proposing LS-VOS, a framework for identifying outliers in 3D object detections. Our approach builds on the idea of Virtual Outlier Synthesis (VOS), which incorporates outlier knowledge during training, enabling the model to learn more compact decision boundaries. In particular, we propose a new synthesis approach that relies on the latent space of an auto-encoder network to generate outlier features with a parametrizable degree of similarity to in-distribution features. In extensive experiments, we show that our approach improves the outlier detection capabilities of a state-of-the-art object detector while maintaining high 3D object detection performance.

{{</citation>}}


### (120/165) Towards Robust 3D Object Detection In Rainy Conditions (Aldi Piroli et al., 2023)

{{<citation>}}

Aldi Piroli, Vinzenz Dallabetta, Johannes Kopp, Marc Walessa, Daniel Meissner, Klaus Dietmayer. (2023)  
**Towards Robust 3D Object Detection In Rainy Conditions**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2310.00944v1)  

---


**ABSTRACT**  
LiDAR sensors are used in autonomous driving applications to accurately perceive the environment. However, they are affected by adverse weather conditions such as snow, fog, and rain. These everyday phenomena introduce unwanted noise into the measurements, severely degrading the performance of LiDAR-based perception systems. In this work, we propose a framework for improving the robustness of LiDAR-based 3D object detectors against road spray. Our approach uses a state-of-the-art adverse weather detection network to filter out spray from the LiDAR point cloud, which is then used as input for the object detector. In this way, the detected objects are less affected by the adverse weather in the scene, resulting in a more accurate perception of the environment. In addition to adverse weather filtering, we explore the use of radar targets to further filter false positive detections. Tests on real-world data show that our approach improves the robustness to road spray of several popular 3D object detectors.

{{</citation>}}


### (121/165) Enhanced Winter Road Surface Condition Monitoring with Computer Vision (Risto Ojala et al., 2023)

{{<citation>}}

Risto Ojala, Alvari Seppänen. (2023)  
**Enhanced Winter Road Surface Condition Monitoring with Computer Vision**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Computer Vision  
[Paper Link](http://arxiv.org/abs/2310.00923v1)  

---


**ABSTRACT**  
Winter conditions pose several challenges for automated driving applications. A key challenge during winter is accurate assessment of road surface condition, as its impact on friction is a critical parameter for safely and reliably controlling a vehicle. This paper proposes a deep learning regression model, SIWNet, capable of estimating road surface friction properties from camera images. SIWNet extends state of the art by including an uncertainty estimation mechanism in the architecture. This is achieved by including an additional head in the network, which estimates a prediction interval. The prediction interval head is trained with a maximum likelihood loss function. The model was trained and tested with the SeeingThroughFog dataset, which features corresponding road friction sensor readings and images from an instrumented vehicle. Acquired results highlight the functionality of the prediction interval estimation of SIWNet, while the network also achieved similar point estimate accuracy as the previous state of the art. Furthermore, the SIWNet architecture is several times more lightweight than the previously applied state-of-the-art model, resulting in more practical and efficient deployment.

{{</citation>}}


### (122/165) How Close are Other Computer Vision Tasks to Deepfake Detection? (Huy H. Nguyen et al., 2023)

{{<citation>}}

Huy H. Nguyen, Junichi Yamagishi, Isao Echizen. (2023)  
**How Close are Other Computer Vision Tasks to Deepfake Detection?**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Computer Vision, ImageNet  
[Paper Link](http://arxiv.org/abs/2310.00922v1)  

---


**ABSTRACT**  
In this paper, we challenge the conventional belief that supervised ImageNet-trained models have strong generalizability and are suitable for use as feature extractors in deepfake detection. We present a new measurement, "model separability," for visually and quantitatively assessing a model's raw capacity to separate data in an unsupervised manner. We also present a systematic benchmark for determining the correlation between deepfake detection and other computer vision tasks using pre-trained models. Our analysis shows that pre-trained face recognition models are more closely related to deepfake detection than other models. Additionally, models trained using self-supervised methods are more effective in separation than those trained using supervised methods. After fine-tuning all models on a small deepfake dataset, we found that self-supervised models deliver the best results, but there is a risk of overfitting. Our results provide valuable insights that should help researchers and practitioners develop more effective deepfake detection models.

{{</citation>}}


### (123/165) Every Dataset Counts: Scaling up Monocular 3D Object Detection with Joint Datasets Training (Fulong Ma et al., 2023)

{{<citation>}}

Fulong Ma, Xiaoyang Yan, Yuxuan Liu, Ming Liu. (2023)  
**Every Dataset Counts: Scaling up Monocular 3D Object Detection with Joint Datasets Training**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2310.00920v1)  

---


**ABSTRACT**  
Monocular 3D object detection plays a crucial role in autonomous driving. However, existing monocular 3D detection algorithms depend on 3D labels derived from LiDAR measurements, which are costly to acquire for new datasets and challenging to deploy in novel environments. Specifically, this study investigates the pipeline for training a monocular 3D object detection model on a diverse collection of 3D and 2D datasets. The proposed framework comprises three components: (1) a robust monocular 3D model capable of functioning across various camera settings, (2) a selective-training strategy to accommodate datasets with differing class annotations, and (3) a pseudo 3D training approach using 2D labels to enhance detection performance in scenes containing only 2D labels. With this framework, we could train models on a joint set of various open 3D/2D datasets to obtain models with significantly stronger generalization capability and enhanced performance on new dataset with only 2D labels. We conduct extensive experiments on KITTI/nuScenes/ONCE/Cityscapes/BDD100K datasets to demonstrate the scaling ability of the proposed method.

{{</citation>}}


## cs.CR (2)



### (124/165) Risk and Threat Mitigation Techniques in Internet of Things (IoT) Environments: A Survey (Marwa Salayma, 2023)

{{<citation>}}

Marwa Salayma. (2023)  
**Risk and Threat Mitigation Techniques in Internet of Things (IoT) Environments: A Survey**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-DC, cs-NI, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2310.01676v1)  

---


**ABSTRACT**  
Security in the Internet of Things (IoT) remains a predominant area of concern. This survey updates the state of the art covered in previous surveys and focuses on defending against threats rather than on the threats alone. This area is less extensively covered by other surveys and warrants particular attention. A life-cycle approach is adopted, articulated to form a "defence in depth" strategy against malicious actors compromising an IoT network laterally within it and from it. This study highlights the challenges of each mitigation step, emphasises novel perspectives, and reconnects the discussed mitigation steps to the ground principles they seek to implement.

{{</citation>}}


### (125/165) Large Language Model-Powered Smart Contract Vulnerability Detection: New Perspectives (Sihao Hu et al., 2023)

{{<citation>}}

Sihao Hu, Tiansheng Huang, Fatih İlhan, Selim Fukan Tekin, Ling Liu. (2023)  
**Large Language Model-Powered Smart Contract Vulnerability Detection: New Perspectives**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: GPT, Language Model, Vulnerability Detection  
[Paper Link](http://arxiv.org/abs/2310.01152v1)  

---


**ABSTRACT**  
This paper provides a systematic analysis of the opportunities, challenges, and potential solutions of harnessing LLMs to dig out vulnerabilities within smart contracts based on our ongoing research. For the smart contract vulnerability detection task, the key to achieving practical usability lies in detecting as many true vulnerabilities as possible while minimizing the number of false positives. However, our empirical study using LLM as a detection tool reveals interesting yet contradictory findings: generating more answers with higher randomness largely increases the likelihood of a correct answer being generated while inevitably leading to a higher number of false positives, resulting in exhaustive manual verification efforts. To mitigate this tension, we propose an adversarial framework dubbed GPTLens that breaks the traditional one-stage detection into two synergistic stages $-$ generation and discrimination, for progressive detection and fine-tuning, wherein the LLM plays dual roles, i.e., auditor and critic, respectively. The goal of auditor is to identify multiple diverse vulnerabilities with intermediate reasoning, while the goal of critic is to evaluate the accuracy of identified vulnerabilities and to examine the integrity of the detection reasoning. Experimental results and illustrative examples demonstrate that auditor and critic work together harmoniously to yield significant improvements over the traditional one-stage detection. GPTLens is intuitive, strategic, and entirely LLM-driven without relying on specialist expertise in smart contracts, showcasing its methodical generality and potential to detect a broad spectrum of vulnerabilities. Our code is available at: https://github.com/git-disl/GPTLens.

{{</citation>}}


## cs.SE (4)



### (126/165) A Unified Taxonomy and Evaluation of IoT Security Guidelines (Jesse Chen et al., 2023)

{{<citation>}}

Jesse Chen, Dharun Anandayuvaraj, James C Davis, Sazzadur Rahaman. (2023)  
**A Unified Taxonomy and Evaluation of IoT Security Guidelines**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2310.01653v2)  

---


**ABSTRACT**  
Cybersecurity concerns about Internet of Things (IoT) devices and infrastructure are growing each year. In response, organizations worldwide have published IoT cybersecurity guidelines to protect their citizens and customers. These guidelines constrain the development of IoT systems, which include substantial software components both on-device and in the Cloud. While these guidelines are being widely adopted, e.g. by US federal contractors, their content and merits have not been critically examined. Two notable gaps are: (1) We do not know how these guidelines differ by the topics and details of their recommendations; and (2) We do not know how effective they are at mitigating real-world IoT failures.   In this paper, we address these questions through an exploratory sequential mixed-method study of IoT cybersecurity guidelines. We collected a corpus of 142 general IoT cybersecurity guidelines, sampling them for recommendations until saturation was reached. From the resulting 958 unique recommendations, we iteratively developed a hierarchical taxonomy following grounded theory coding principles. We measured the guidelines' usefulness by asking novice engineers about the actionability of each recommendation, and by matching cybersecurity recommendations to the root causes of failures (CVEs and news stories). We report that: (1) Comparing guidelines to one another, each guideline has gaps in its topic coverage and comprehensiveness; and (2) Although 87.2% recommendations are actionable and the union of the guidelines mitigates all 17 of the failures from news stories, 21% of the CVEs apparently evade the guidelines. In summary, we report shortcomings in every guideline's depth and breadth, but as a whole they are capable of preventing security issues. Our results will help software engineers determine which and how many guidelines to study as they implement IoT systems.

{{</citation>}}


### (127/165) CAT-LM: Training Language Models on Aligned Code And Tests (Nikitha Rao et al., 2023)

{{<citation>}}

Nikitha Rao, Kush Jain, Uri Alon, Claire Le Goues, Vincent J. Hellendoorn. (2023)  
**CAT-LM: Training Language Models on Aligned Code And Tests**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.01602v1)  

---


**ABSTRACT**  
Testing is an integral part of the software development process. Yet, writing tests is time-consuming and therefore often neglected. Classical test generation tools such as EvoSuite generate behavioral test suites by optimizing for coverage, but tend to produce tests that are hard to understand. Language models trained on code can generate code that is highly similar to that written by humans, but current models are trained to generate each file separately, as is standard practice in natural language processing, and thus fail to consider the code-under-test context when producing a test file. In this work, we propose the Aligned Code And Tests Language Model (CAT-LM), a GPT-style language model with 2.7 Billion parameters, trained on a corpus of Python and Java projects. We utilize a novel pretraining signal that explicitly considers the mapping between code and test files when available. We also drastically increase the maximum sequence length of inputs to 8,192 tokens, 4x more than typical code generation models, to ensure that the code context is available to the model when generating test code. We analyze its usefulness for realistic applications, showing that sampling with filtering (e.g., by compilability, coverage) allows it to efficiently produce tests that achieve coverage similar to ones written by developers while resembling their writing style. By utilizing the code context, CAT-LM generates more valid tests than even much larger language models trained with more data (CodeGen 16B and StarCoder) and substantially outperforms a recent test-specific model (TeCo) at test completion. Overall, our work highlights the importance of incorporating software-specific insights when training language models for code and paves the way to more powerful automated test generation.

{{</citation>}}


### (128/165) L2MAC: Large Language Model Automatic Computer for Unbounded Code Generation (Samuel Holt et al., 2023)

{{<citation>}}

Samuel Holt, Max Ruiz Luyten, Mihaela van der Schaar. (2023)  
**L2MAC: Large Language Model Automatic Computer for Unbounded Code Generation**  

---
Primary Category: cs.SE  
Categories: I-2-7; I-2-6; I-2-5; D-2-2; D-2-3; D-3-4, cs-AI, cs-LG, cs-PL, cs-SE, cs.SE  
Keywords: Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2310.02003v1)  

---


**ABSTRACT**  
Transformer-based large language models (LLMs) are constrained by the fixed context window of the underlying transformer architecture, hindering their ability to produce long and logically consistent code. Memory-augmented LLMs are a promising solution, but current approaches cannot handle long code generation tasks since they (1) only focus on reading memory and reduce its evolution to the concatenation of new memories or (2) use very specialized memories that cannot adapt to other domains. This paper presents L2MAC, the first practical LLM-based stored-program automatic computer for long and consistent code generation. Its memory has two components: the instruction registry, which is populated with a prompt program to solve the user-given task, and a file store, which will contain the final and intermediate outputs. Each instruction is executed by a separate LLM instance, whose context is managed by a control unit capable of precise memory reading and writing to ensure effective interaction with the file store. These components enable L2MAC to generate virtually unbounded code structures, bypassing the constraints of the finite context window while producing code that fulfills complex user-specified requirements. We empirically show that L2MAC succeeds in generating large code bases for system design tasks where other coding methods fall short in implementing user requirements and provide insight into the reasons for this performance gap.

{{</citation>}}


### (129/165) Comparative Analysis of Technical and Legal Frameworks of Various National Digial Identity Solutions (Montassar Naghmouchi et al., 2023)

{{<citation>}}

Montassar Naghmouchi, Maryline Laurent, Claire Levallois-Barth, Nesrine Kaaniche. (2023)  
**Comparative Analysis of Technical and Legal Frameworks of Various National Digial Identity Solutions**  

---
Primary Category: cs.SE  
Categories: cs-CR, cs-CY, cs-SE, cs.SE  
Keywords: Legal  
[Paper Link](http://arxiv.org/abs/2310.01006v1)  

---


**ABSTRACT**  
National digital identity systems have become a key requirement for easy access to online public services, specially during Covid-19. While many countries have adopted a national digital identity system, many are still in the process of establishing one. Through a comparative analysis of the technological and legal dimensions of a few selected national digital identity solutions currently being used in different countries, we highlight the diversity of technologies and architectures and the key role of the legal framework of a given digital identity solution. We also present several key issues related to the implementation of these solutions, how to ensure the State sovereignty over them, and how to strike the right balance between private sector and public sector needs. This position paper aims to help policy makers, software developers and concerned users understand the challenges of designing, implementing and using a national digital identity management system and establishing a legal framework for digital identity management, including personal data protection measures. The authors of this paper have a favorable position for self-sovereign identity management systems that are based on Blockchain technology, and we believe they are the most suitable for national digital identity systems.

{{</citation>}}


## quant-ph (1)



### (130/165) Inter-temperature Bandwidth Reduction in Cryogenic QAOA Machines (Yosuke Ueno et al., 2023)

{{<citation>}}

Yosuke Ueno, Yuna Tomida, Teruo Tanimoto, Masamitsu Tanaka, Yutaka Tabuchi, Koji Inoue, Hiroshi Nakamura. (2023)  
**Inter-temperature Bandwidth Reduction in Cryogenic QAOA Machines**  

---
Primary Category: quant-ph  
Categories: cs-AR, quant-ph, quant-ph  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2310.01630v1)  

---


**ABSTRACT**  
The bandwidth limit between cryogenic and room-temperature environments is a critical bottleneck in superconducting noisy intermediate-scale quantum computers. This paper presents the first trial of algorithm-aware system-level optimization to solve this issue by targeting the quantum approximate optimization algorithm. Our counter-based cryogenic architecture using single-flux quantum logic shows exponential bandwidth reduction and decreases heat inflow and peripheral power consumption of inter-temperature cables, which contributes to the scalability of superconducting quantum computers.

{{</citation>}}


## cs.HC (6)



### (131/165) VAL: Interactive Task Learning with GPT Dialog Parsing (Lane Lawley et al., 2023)

{{<citation>}}

Lane Lawley, Christopher J. MacLellan. (2023)  
**VAL: Interactive Task Learning with GPT Dialog Parsing**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-CL, cs-HC, cs.HC  
Keywords: Dialog, GPT  
[Paper Link](http://arxiv.org/abs/2310.01627v1)  

---


**ABSTRACT**  
Reinforcement learning often requires millions of examples to produce static, black-box models. In contrast, interactive task learning (ITL) emphasizes incremental knowledge acquisition from limited instruction provided by humans in modalities such as natural language. However, in practice, ITL systems often suffers from brittle, error-prone language parsing. Large language models (LLMs) are resistant to brittleness but are not interpretable and cannot learn incrementally. We present VAL, an ITL system with a new philosophy for LLM/symbolic integration. By using LLMs only for specific tasks -- such as predicate and argument selection -- within an algorithmic framework, VAL reaps the benefits of LLMs to support interactive learning of hierarchical task knowledge from natural language. Acquired knowledge is human interpretable and generalizes to support execution of novel tasks without additional training. We studied users' interactions with VAL in a video game setting, finding that most users could successfully teach VAL using language they felt was natural.

{{</citation>}}


### (132/165) Active Learning on Neural Networks through Interactive Generation of Digit Patterns and Visual Representation (Dong H. Jeong et al., 2023)

{{<citation>}}

Dong H. Jeong, Jin-Hee Cho, Feng Chen, Audun Josang, Soo-Yeon Ji. (2023)  
**Active Learning on Neural Networks through Interactive Generation of Digit Patterns and Visual Representation**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2310.01580v1)  

---


**ABSTRACT**  
Artificial neural networks (ANNs) have been broadly utilized to analyze various data and solve different domain problems. However, neural networks (NNs) have been considered a black box operation for years because their underlying computation and meaning are hidden. Due to this nature, users often face difficulties in interpreting the underlying mechanism of the NNs and the benefits of using them. In this paper, to improve users' learning and understanding of NNs, an interactive learning system is designed to create digit patterns and recognize them in real time. To help users clearly understand the visual differences of digit patterns (i.e., 0 ~ 9) and their results with an NN, integrating visualization is considered to present all digit patterns in a two-dimensional display space with supporting multiple user interactions. An evaluation with multiple datasets is conducted to determine its usability for active learning. In addition, informal user testing is managed during a summer workshop by asking the workshop participants to use the system.

{{</citation>}}


### (133/165) Co-audit: tools to help humans double-check AI-generated content (Andrew D. Gordon et al., 2023)

{{<citation>}}

Andrew D. Gordon, Carina Negreanu, José Cambronero, Rasika Chakravarthy, Ian Drosos, Hao Fang, Bhaskar Mitra, Hannah Richardson, Advait Sarkar, Stephanie Simmons, Jack Williams, Ben Zorn. (2023)  
**Co-audit: tools to help humans double-check AI-generated content**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-CL, cs-HC, cs-PL, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.01297v1)  

---


**ABSTRACT**  
Users are increasingly being warned to check AI-generated content for correctness. Still, as LLMs (and other generative models) generate more complex output, such as summaries, tables, or code, it becomes harder for the user to audit or evaluate the output for quality or correctness. Hence, we are seeing the emergence of tool-assisted experiences to help the user double-check a piece of AI-generated content. We refer to these as co-audit tools. Co-audit tools complement prompt engineering techniques: one helps the user construct the input prompt, while the other helps them check the output response. As a specific example, this paper describes recent research on co-audit tools for spreadsheet computations powered by generative models. We explain why co-audit experiences are essential for any application of generative AI where quality is important and errors are consequential (as is common in spreadsheet computations). We propose a preliminary list of principles for co-audit, and outline research challenges.

{{</citation>}}


### (134/165) Grasping AI: experiential exercises for designers (Dave Murray-Rust et al., 2023)

{{<citation>}}

Dave Murray-Rust, Maria Luce Lupetti, Iohanna Nicenboim, Wouter van der Hoog. (2023)  
**Grasping AI: experiential exercises for designers**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.01282v1)  

---


**ABSTRACT**  
Artificial intelligence (AI) and machine learning (ML) are increasingly integrated into the functioning of physical and digital products, creating unprecedented opportunities for interaction and functionality. However, there is a challenge for designers to ideate within this creative landscape, balancing the possibilities of technology with human interactional concerns. We investigate techniques for exploring and reflecting on the interactional affordances, the unique relational possibilities, and the wider social implications of AI systems. We introduced into an interaction design course (n=100) nine 'AI exercises' that draw on more than human design, responsible AI, and speculative enactment to create experiential engagements around AI interaction design. We find that exercises around metaphors and enactments make questions of training and learning, privacy and consent, autonomy and agency more tangible, and thereby help students be more reflective and responsible on how to design with AI and its complex properties in both their design process and outcomes.

{{</citation>}}


### (135/165) The benefits and costs of explainable artificial intelligence in visual quality control: Evidence from fault detection performance and eye movements (Romy Müller et al., 2023)

{{<citation>}}

Romy Müller, David F. Reindel, Yannick D. Stadtfeld. (2023)  
**The benefits and costs of explainable artificial intelligence in visual quality control: Evidence from fault detection performance and eye movements**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.01220v1)  

---


**ABSTRACT**  
Visual inspection tasks often require humans to cooperate with AI-based image classifiers. To enhance this cooperation, explainable artificial intelligence (XAI) can highlight those image areas that have contributed to an AI decision. However, the literature on visual cueing suggests that such XAI support might come with costs of its own. To better understand how the benefits and cost of XAI depend on the accuracy of AI classifications and XAI highlights, we conducted two experiments that simulated visual quality control in a chocolate factory. Participants had to decide whether chocolate moulds contained faulty bars or not, and were always informed whether the AI had classified the mould as faulty or not. In half of the experiment, they saw additional XAI highlights that justified this classification. While XAI speeded up performance, its effects on error rates were highly dependent on (X)AI accuracy. XAI benefits were observed when the system correctly detected and highlighted the fault, but XAI costs were evident for misplaced highlights that marked an intact area while the actual fault was located elsewhere. Eye movement analyses indicated that participants spent less time searching the rest of the mould and thus looked at the fault less often. However, we also observed large interindividual differences. Taken together, the results suggest that despite its potentials, XAI can discourage people from investing effort into their own information analysis.

{{</citation>}}


### (136/165) The Participatory Turn in AI Design: Theoretical Foundations and the Current State of Practice (Fernando Delgado et al., 2023)

{{<citation>}}

Fernando Delgado, Stephen Yang, Michael Madaio, Qian Yang. (2023)  
**The Participatory Turn in AI Design: Theoretical Foundations and the Current State of Practice**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.00907v1)  

---


**ABSTRACT**  
Despite the growing consensus that stakeholders affected by AI systems should participate in their design, enormous variation and implicit disagreements exist among current approaches. For researchers and practitioners who are interested in taking a participatory approach to AI design and development, it remains challenging to assess the extent to which any participatory approach grants substantive agency to stakeholders. This article thus aims to ground what we dub the "participatory turn" in AI design by synthesizing existing theoretical literature on participation and through empirical investigation and critique of its current practices. Specifically, we derive a conceptual framework through synthesis of literature across technology design, political theory, and the social sciences that researchers and practitioners can leverage to evaluate approaches to participation in AI design. Additionally, we articulate empirical findings concerning the current state of participatory practice in AI design based on an analysis of recently published research and semi-structured interviews with 12 AI researchers and practitioners. We use these empirical findings to understand the current state of participatory practice and subsequently provide guidance to better align participatory goals and methods in a way that accounts for practical constraints.

{{</citation>}}


## cs.IR (1)



### (137/165) Towards Efficient and Effective Adaptation of Large Language Models for Sequential Recommendation (Bo Peng et al., 2023)

{{<citation>}}

Bo Peng, Ben Burns, Ziqi Chen, Srinivasan Parthasarathy, Xia Ning. (2023)  
**Towards Efficient and Effective Adaptation of Large Language Models for Sequential Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.01612v1)  

---


**ABSTRACT**  
In recent years, with large language models (LLMs) achieving state-of-the-art performance in context understanding, increasing efforts have been dedicated to developing LLM-enhanced sequential recommendation (SR) methods. Considering that most existing LLMs are not specifically optimized for recommendation tasks, adapting them for SR becomes a critical step in LLM-enhanced SR methods. Though numerous adaptation methods have been developed, it still remains a significant challenge to adapt LLMs for SR both efficiently and effectively. To address this challenge, in this paper, we introduce a novel side sequential network adaptation method, denoted as SSNA, for LLM enhanced SR. SSNA features three key designs to allow both efficient and effective LLM adaptation. First, SSNA learns adapters separate from LLMs, while fixing all the pre-trained parameters within LLMs to allow efficient adaptation. In addition, SSNA adapts the top-a layers of LLMs jointly, and integrates adapters sequentially for enhanced effectiveness (i.e., recommendation performance). We compare SSNA against five state-of-the-art baseline methods on five benchmark datasets using three LLMs. The experimental results demonstrate that SSNA significantly outperforms all the baseline methods in terms of recommendation performance, and achieves substantial improvement over the best-performing baseline methods at both run-time and memory efficiency during training. Our analysis shows the effectiveness of integrating adapters in a sequential manner. Our parameter study demonstrates the effectiveness of jointly adapting the top-a layers of LLMs.

{{</citation>}}


## stat.ML (3)



### (138/165) An Investigation of Representation and Allocation Harms in Contrastive Learning (Subha Maity et al., 2023)

{{<citation>}}

Subha Maity, Mayank Agarwal, Mikhail Yurochkin, Yuekai Sun. (2023)  
**An Investigation of Representation and Allocation Harms in Contrastive Learning**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-ML, stat.ML  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2310.01583v1)  

---


**ABSTRACT**  
The effect of underrepresentation on the performance of minority groups is known to be a serious problem in supervised learning settings; however, it has been underexplored so far in the context of self-supervised learning (SSL). In this paper, we demonstrate that contrastive learning (CL), a popular variant of SSL, tends to collapse representations of minority groups with certain majority groups. We refer to this phenomenon as representation harm and demonstrate it on image and text datasets using the corresponding popular CL methods. Furthermore, our causal mediation analysis of allocation harm on a downstream classification task reveals that representation harm is partly responsible for it, thus emphasizing the importance of studying and mitigating representation harm. Finally, we provide a theoretical explanation for representation harm using a stochastic block model that leads to a representational neural collapse in a contrastive learning setting.

{{</citation>}}


### (139/165) A path-norm toolkit for modern networks: consequences, promises and challenges (Antoine Gonon et al., 2023)

{{<citation>}}

Antoine Gonon, Nicolas Brisebarre, Elisa Riccietti, Rémi Gribonval. (2023)  
**A path-norm toolkit for modern networks: consequences, promises and challenges**  

---
Primary Category: stat.ML  
Categories: cs-LG, math-ST, stat-ML, stat-TH, stat.ML  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2310.01225v1)  

---


**ABSTRACT**  
This work introduces the first toolkit around path-norms that is fully able to encompass general DAG ReLU networks with biases, skip connections and max pooling. This toolkit notably allows us to establish generalization bounds for real modern neural networks that are not only the most widely applicable path-norm based ones, but also recover or beat the sharpest known bounds of this type. These extended path-norms further enjoy the usual benefits of path-norms: ease of computation, invariance under the symmetries of the network, and improved sharpness on feedforward networks compared to the product of operators' norms, another complexity measure most commonly used.   The versatility of the toolkit and its ease of implementation allow us to challenge the concrete promises of path-norm-based generalization bounds, by numerically evaluating the sharpest known bounds for ResNets on ImageNet.

{{</citation>}}


### (140/165) Unified Uncertainty Calibration (Kamalika Chaudhuri et al., 2023)

{{<citation>}}

Kamalika Chaudhuri, David Lopez-Paz. (2023)  
**Unified Uncertainty Calibration**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-ML, stat.ML  
Keywords: AI, ImageNet  
[Paper Link](http://arxiv.org/abs/2310.01202v1)  

---


**ABSTRACT**  
To build robust, fair, and safe AI systems, we would like our classifiers to say ``I don't know'' when facing test examples that are difficult or fall outside of the training classes.The ubiquitous strategy to predict under uncertainty is the simplistic \emph{reject-or-classify} rule: abstain from prediction if epistemic uncertainty is high, classify otherwise.Unfortunately, this recipe does not allow different sources of uncertainty to communicate with each other, produces miscalibrated predictions, and it does not allow to correct for misspecifications in our uncertainty estimates. To address these three issues, we introduce \emph{unified uncertainty calibration (U2C)}, a holistic framework to combine aleatoric and epistemic uncertainties. U2C enables a clean learning-theoretical analysis of uncertainty estimation, and outperforms reject-or-classify across a variety of ImageNet benchmarks.

{{</citation>}}


## eess.SY (1)



### (141/165) Risk-Sensitive Inhibitory Control for Safe Reinforcement Learning (Armin Lederer et al., 2023)

{{<citation>}}

Armin Lederer, Erfaun Noorani, John S. Baras, Sandra Hirche. (2023)  
**Risk-Sensitive Inhibitory Control for Safe Reinforcement Learning**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.01538v1)  

---


**ABSTRACT**  
Humans have the ability to deviate from their natural behavior when necessary, which is a cognitive process called response inhibition. Similar approaches have independently received increasing attention in recent years for ensuring the safety of control. Realized using control barrier functions or predictive safety filters, these approaches can effectively ensure the satisfaction of state constraints through an online adaptation of nominal control laws, e.g., obtained through reinforcement learning. While the focus of these realizations of inhibitory control has been on risk-neutral formulations, human studies have shown a tight link between response inhibition and risk attitude. Inspired by this insight, we propose a flexible, risk-sensitive method for inhibitory control. Our method is based on a risk-aware condition for value functions, which guarantees the satisfaction of state constraints. We propose a method for learning these value functions using common techniques from reinforcement learning and derive sufficient conditions for its success. By enforcing the derived safety conditions online using the learned value function, risk-sensitive inhibitory control is effectively achieved. The effectiveness of the developed control scheme is demonstrated in simulations.

{{</citation>}}


## eess.IV (5)



### (142/165) A multi-institutional pediatric dataset of clinical radiology MRIs by the Children's Brain Tumor Network (Ariana M. Familiar et al., 2023)

{{<citation>}}

Ariana M. Familiar, Anahita Fathi Kazerooni, Hannah Anderson, Aliaksandr Lubneuski, Karthik Viswanathan, Rocky Breslow, Nastaran Khalili, Sina Bagheri, Debanjan Haldar, Meen Chul Kim, Sherjeel Arif, Rachel Madhogarhia, Thinh Q. Nguyen, Elizabeth A. Frenkel, Zeinab Helili, Jessica Harrison, Keyvan Farahani, Marius George Linguraru, Ulas Bagci, Yury Velichko, Jeffrey Stevens, Sarah Leary, Robert M. Lober, Stephani Campion, Amy A. Smith, Denise Morinigo, Brian Rood, Kimberly Diamond, Ian F. Pollack, Melissa Williams, Arastoo Vossough, Jeffrey B. Ware, Sabine Mueller, Phillip B. Storm, Allison P. Heath, Angela J. Waanders, Jena V. Lilly, Jennifer L. Mason, Adam C. Resnick, Ali Nabavizadeh. (2023)  
**A multi-institutional pediatric dataset of clinical radiology MRIs by the Children's Brain Tumor Network**  

---
Primary Category: eess.IV  
Categories: cs-AI, cs-CV, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.01413v1)  

---


**ABSTRACT**  
Pediatric brain and spinal cancers remain the leading cause of cancer-related death in children. Advancements in clinical decision-support in pediatric neuro-oncology utilizing the wealth of radiology imaging data collected through standard care, however, has significantly lagged other domains. Such data is ripe for use with predictive analytics such as artificial intelligence (AI) methods, which require large datasets. To address this unmet need, we provide a multi-institutional, large-scale pediatric dataset of 23,101 multi-parametric MRI exams acquired through routine care for 1,526 brain tumor patients, as part of the Children's Brain Tumor Network. This includes longitudinal MRIs across various cancer diagnoses, with associated patient-level clinical information, digital pathology slides, as well as tissue genotype and omics data. To facilitate downstream analysis, treatment-na\"ive images for 370 subjects were processed and released through the NCI Childhood Cancer Data Initiative via the Cancer Data Service. Through ongoing efforts to continuously build these imaging repositories, our aim is to accelerate discovery and translational AI models with real-world data, to ultimately empower precision medicine for children.

{{</citation>}}


### (143/165) Towards Robust Cardiac Segmentation using Graph Convolutional Networks (Gilles Van De Vyver et al., 2023)

{{<citation>}}

Gilles Van De Vyver, Sarina Thomas, Guy Ben-Yosef, Sindre Hellum Olaisen, Håvard Dalen, Lasse Løvstakken, Erik Smistad. (2023)  
**Towards Robust Cardiac Segmentation using Graph Convolutional Networks**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Graph Convolutional Network  
[Paper Link](http://arxiv.org/abs/2310.01210v1)  

---


**ABSTRACT**  
Fully automatic cardiac segmentation can be a fast and reproducible method to extract clinical measurements from an echocardiography examination. The U-Net architecture is the current state-of-the-art deep learning architecture for medical segmentation and can segment cardiac structures in real-time with average errors comparable to inter-observer variability. However, this architecture still generates large outliers that are often anatomically incorrect. This work uses the concept of graph convolutional neural networks that predict the contour points of the structures of interest instead of labeling each pixel. We propose a graph architecture that uses two convolutional rings based on cardiac anatomy and show that this eliminates anatomical incorrect multi-structure segmentations on the publicly available CAMUS dataset. Additionally, this work contributes with an ablation study on the graph convolutional architecture and an evaluation of clinical measurements on the clinical HUNT4 dataset. Finally, we propose to use the inter-model agreement of the U-Net and the graph network as a predictor of both the input and segmentation quality. We show this predictor can detect out-of-distribution and unsuitable input images in real-time. Source code is available online: https://github.com/gillesvntnu/GCN_multistructure

{{</citation>}}


### (144/165) Iterative Semi-Supervised Learning for Abdominal Organs and Tumor Segmentation (Jiaxin Zhuang et al., 2023)

{{<citation>}}

Jiaxin Zhuang, Luyang Luo, Zhixuan Chen, Linshan Wu. (2023)  
**Iterative Semi-Supervised Learning for Abdominal Organs and Tumor Segmentation**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2310.01159v1)  

---


**ABSTRACT**  
Deep-learning (DL) based methods are playing an important role in the task of abdominal organs and tumors segmentation in CT scans. However, the large requirements of annotated datasets heavily limit its development. The FLARE23 challenge provides a large-scale dataset with both partially and fully annotated data, which also focuses on both segmentation accuracy and computational efficiency. In this study, we propose to use the strategy of Semi-Supervised Learning (SSL) and iterative pseudo labeling to address FLARE23. Initially, a deep model (nn-UNet) trained on datasets with complete organ annotations (about 220 scans) generates pseudo labels for the whole dataset. These pseudo labels are then employed to train a more powerful segmentation model. Employing the FLARE23 dataset, our approach achieves an average DSC score of 89.63% for organs and 46.07% for tumors on online validation leaderboard. For organ segmentation, We obtain 0.9007\% DSC and 0.9493\% NSD. For tumor segmentation, we obtain 0.3785% DSC and 0.2842% NSD. Our code is available at https://github.com/USTguy/Flare23.

{{</citation>}}


### (145/165) HyMNet: a Multimodal Deep Learning System for Hypertension Classification using Fundus Photographs and Cardiometabolic Risk Factors (Mohammed Baharoon et al., 2023)

{{<citation>}}

Mohammed Baharoon, Hessa Almatar, Reema Alduhayan, Tariq Aldebasi, Badr Alahmadi, Yahya Bokhari, Mohammed Alawad, Ahmed Almazroa, Abdulrhman Aljouie. (2023)  
**HyMNet: a Multimodal Deep Learning System for Hypertension Classification using Fundus Photographs and Cardiometabolic Risk Factors**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2310.01099v1)  

---


**ABSTRACT**  
In recent years, deep learning has shown promise in predicting hypertension (HTN) from fundus images. However, most prior research has primarily focused on analyzing a single type of data, which may not capture the full complexity of HTN risk. To address this limitation, this study introduces a multimodal deep learning (MMDL) system, dubbed HyMNet, which combines fundus images and cardiometabolic risk factors, specifically age and gender, to improve hypertension detection capabilities. Our MMDL system uses the DenseNet-201 architecture, pre-trained on ImageNet, for the fundus imaging path and a fully connected neural network for the age and gender path. The two paths are jointly trained by concatenating 64 features output from each path that are then fed into a fusion network. The system was trained on 1,143 retinal images from 626 individuals collected from the Saudi Ministry of National Guard Health Affairs. The results show that the multimodal model that integrates fundus images along with age and gender achieved an AUC of 0.791 [CI: 0.735, 0.848], which outperforms the unimodal model trained solely on fundus photographs that yielded an AUC of 0.766 [CI: 0.705, 0.828] for hypertension detection.

{{</citation>}}


### (146/165) BAAF: A Benchmark Attention Adaptive Framework for Medical Ultrasound Image Segmentation Tasks (Gongping Chen et al., 2023)

{{<citation>}}

Gongping Chen, Lei Zhao, Xiaotao Yin, Liang Cui, Jianxun Zhang, Yu Dai. (2023)  
**BAAF: A Benchmark Attention Adaptive Framework for Medical Ultrasound Image Segmentation Tasks**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: AI, Attention  
[Paper Link](http://arxiv.org/abs/2310.00919v1)  

---


**ABSTRACT**  
The AI-based assisted diagnosis programs have been widely investigated on medical ultrasound images. Complex scenario of ultrasound image, in which the coupled interference of internal and external factors is severe, brings a unique challenge for localize the object region automatically and precisely in ultrasound images. In this study, we seek to propose a more general and robust Benchmark Attention Adaptive Framework (BAAF) to assist doctors segment or diagnose lesions and tissues in ultrasound images more quickly and accurately. Different from existing attention schemes, the BAAF consists of a parallel hybrid attention module (PHAM) and an adaptive calibration mechanism (ACM). Specifically, BAAF first coarsely calibrates the input features from the channel and spatial dimensions, and then adaptively selects more robust lesion or tissue characterizations from the coarse-calibrated feature maps. The design of BAAF further optimizes the "what" and "where" focus and selection problems in CNNs and seeks to improve the segmentation accuracy of lesions or tissues in medical ultrasound images. The method is evaluated on four medical ultrasound segmentation tasks, and the adequate experimental results demonstrate the remarkable performance improvement over existing state-of-the-art methods. In addition, the comparison with existing attention mechanisms also demonstrates the superiority of BAAF. This work provides the possibility for automated medical ultrasound assisted diagnosis and reduces reliance on human accuracy and precision.

{{</citation>}}


## cs.RO (5)



### (147/165) Generalized Animal Imitator: Agile Locomotion with Versatile Motion Prior (Ruihan Yang et al., 2023)

{{<citation>}}

Ruihan Yang, Zhuoqun Chen, Jianhan Ma, Chongyi Zheng, Yiyu Chen, Quan Nguyen, Xiaolong Wang. (2023)  
**Generalized Animal Imitator: Agile Locomotion with Versatile Motion Prior**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-LG, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.01408v1)  

---


**ABSTRACT**  
The agility of animals, particularly in complex activities such as running, turning, jumping, and backflipping, stands as an exemplar for robotic system design. Transferring this suite of behaviors to legged robotic systems introduces essential inquiries: How can a robot be trained to learn multiple locomotion behaviors simultaneously? How can the robot execute these tasks with a smooth transition? And what strategies allow for the integrated application of these skills? This paper introduces the Versatile Instructable Motion prior (VIM) - a Reinforcement Learning framework designed to incorporate a range of agile locomotion tasks suitable for advanced robotic applications. Our framework enables legged robots to learn diverse agile low-level skills by imitating animal motions and manually designed motions with Functionality reward and Stylization reward. While the Functionality reward guides the robot's ability to adopt varied skills, the Stylization reward ensures performance alignment with reference motions. Our evaluations of the VIM framework span both simulation environments and real-world deployment. To our understanding, this is the first work that allows a robot to concurrently learn diverse agile locomotion tasks using a singular controller. Further details and supportive media can be found at our project site: https://rchalyang.github.io/VIM .

{{</citation>}}


### (148/165) Toward Scalable Visual Servoing Using Deep Reinforcement Learning and Optimal Control (Salar Asayesh et al., 2023)

{{<citation>}}

Salar Asayesh, Hossein Sheikhi Darani, Mo chen, Mehran Mehrandezh, Kamal Gupta. (2023)  
**Toward Scalable Visual Servoing Using Deep Reinforcement Learning and Optimal Control**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.01360v1)  

---


**ABSTRACT**  
Classical pixel-based Visual Servoing (VS) approaches offer high accuracy but suffer from a limited convergence area due to optimization nonlinearity. Modern deep learning-based VS methods overcome traditional vision issues but lack scalability, requiring training on limited scenes. This paper proposes a hybrid VS strategy utilizing Deep Reinforcement Learning (DRL) and optimal control to enhance both convergence area and scalability. The DRL component of our approach separately handles representation and policy learning to enhance scalability, generalizability, learning efficiency and ease domain adaptation. Moreover, the optimal control part ensures high end-point accuracy. Our method showcases remarkable achievements in terms of high convergence rates and minimal end-positioning errors using a 7-DOF manipulator. Importantly, it exhibits scalability across more than 1000 distinct scenes. Furthermore, we demonstrate its capacity for generalization to previously unseen datasets. Lastly, we illustrate the real-world applicability of our approach, highlighting its adaptability through single-shot domain transfer learning in environments with noise and occlusions. Real-robot experiments can be found at \url{https://sites.google.com/view/vsls}.

{{</citation>}}


### (149/165) On Fulfilling the Exigent Need for Automating and Modernizing Logistics Infrastructure in India: Enabling AI-based Integration, Digitalization, and Smart Automation of Industrial Parks and Robotic Warehouses (Shaurya Shriyam et al., 2023)

{{<citation>}}

Shaurya Shriyam, Prashant Palkar, Amber Srivastava. (2023)  
**On Fulfilling the Exigent Need for Automating and Modernizing Logistics Infrastructure in India: Enabling AI-based Integration, Digitalization, and Smart Automation of Industrial Parks and Robotic Warehouses**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.01077v1)  

---


**ABSTRACT**  
To stay competitive, the Low- or Middle-Income Countries (LMICs) need to embrace Industry 4.0 and Logistics 4.0. This requires government-level interventions and policy-making to incentivize quality product solutions and drive innovation in traditionally resistant economic sectors. In this position paper, we support the establishment of Smart Industrial Parks (SIPs) with a focus on enhancing operational efficiencies and bringing together MSMEs and startups targeting niche clientele with innovative Industry 4.0 solutions. SIPs along with the phased deployment of well-planned robotic automation technologies shall enable bringing down India's untenable logistics costs. Toward the successful execution of SIPs, we are required to implement the efficient allocation of manufacturing resources and capabilities within SIPs. Thus, we emphasize the importance of efficient resource utilization, collaboration, and technology adoption in industrial parks to promote industrial development and economic growth. We advocate the use of a cloud-based cyber-physical system for real-time data access and analysis in SIPs. Such centralized cloud-based monitoring of factory floors, warehouses, and industrial units using IoT infrastructure shall improve decision-making, efficiency, and safety. Digital Twins (DTs), which are cyber-replicas of physical systems, could play a significant role in enabling simulation, optimization, and real-time monitoring of smart manufacturing and distributed manufacturing systems. However, there are several challenges involved in implementing DTs in distributed manufacturing systems, such as defining data schemas and collaboration protocols, ensuring interoperability, the need for effective authentication technology, distributed machine learning models, and scalability to manage multiple DTs.

{{</citation>}}


### (150/165) GRID: A Platform for General Robot Intelligence Development (Sai Vemprala et al., 2023)

{{<citation>}}

Sai Vemprala, Shuhang Chen, Abhinav Shukla, Dinesh Narayanan, Ashish Kapoor. (2023)  
**GRID: A Platform for General Robot Intelligence Development**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-LG, cs-RO, cs.RO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.00887v1)  

---


**ABSTRACT**  
Developing machine intelligence abilities in robots and autonomous systems is an expensive and time consuming process. Existing solutions are tailored to specific applications and are harder to generalize. Furthermore, scarcity of training data adds a layer of complexity in deploying deep machine learning models. We present a new platform for General Robot Intelligence Development (GRID) to address both of these issues. The platform enables robots to learn, compose and adapt skills to their physical capabilities, environmental constraints and goals. The platform addresses AI problems in robotics via foundation models that know the physical world. GRID is designed from the ground up to be extensible to accommodate new types of robots, vehicles, hardware platforms and software protocols. In addition, the modular design enables various deep ML components and existing foundation models to be easily usable in a wider variety of robot-centric problems. We demonstrate the platform in various aerial robotics scenarios and demonstrate how the platform dramatically accelerates development of machine intelligent robots.

{{</citation>}}


### (151/165) COMPOSER: Scalable and Robust Modular Policies for Snake Robots (Yuyou Zhang et al., 2023)

{{<citation>}}

Yuyou Zhang, Yaru Niu, Xingyu Liu, Ding Zhao. (2023)  
**COMPOSER: Scalable and Robust Modular Policies for Snake Robots**  

---
Primary Category: cs.RO  
Categories: cs-LG, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.00871v1)  

---


**ABSTRACT**  
Snake robots have showcased remarkable compliance and adaptability in their interaction with environments, mirroring the traits of their natural counterparts. While their hyper-redundant and high-dimensional characteristics add to this adaptability, they also pose great challenges to robot control. Instead of perceiving the hyper-redundancy and flexibility of snake robots as mere challenges, there lies an unexplored potential in leveraging these traits to enhance robustness and generalizability at the control policy level. We seek to develop a control policy that effectively breaks down the high dimensionality of snake robots while harnessing their redundancy. In this work, we consider the snake robot as a modular robot and formulate the control of the snake robot as a cooperative Multi-Agent Reinforcement Learning (MARL) problem. Each segment of the snake robot functions as an individual agent. Specifically, we incorporate a self-attention mechanism to enhance the cooperative behavior between agents. A high-level imagination policy is proposed to provide additional rewards to guide the low-level control policy. We validate the proposed method COMPOSER with five snake robot tasks, including goal reaching, wall climbing, shape formation, tube crossing, and block pushing. COMPOSER achieves the highest success rate across all tasks when compared to a centralized baseline and four modular policy baselines. Additionally, we show enhanced robustness against module corruption and significantly superior zero-shot generalizability in our proposed method. The videos of this work are available on our project page: https://sites.google.com/view/composer-snake/.

{{</citation>}}


## cs.SI (5)



### (152/165) The influence of coordinated behavior on toxicity (Edoardo Loru et al., 2023)

{{<citation>}}

Edoardo Loru, Matteo Cinelli, Maurizio Tesconi, Walter Quattrociocchi. (2023)  
**The influence of coordinated behavior on toxicity**  

---
Primary Category: cs.SI  
Categories: cs-CY, cs-SI, cs.SI, physics-soc-ph  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2310.01283v1)  

---


**ABSTRACT**  
In the intricate landscape of social media genuine content dissemination may be altered by a number of threats. Coordinated Behavior (CB), defined as orchestrated efforts by entities to deceive or mislead users about their identity and intentions, emerges as a tactic to exploit or manipulate online discourse. This study delves into the relationship between CB and toxic conversation on Twitter. Using a dataset of 11 million tweets from 1 million users preceding the 2019 UK General Elections, we show that users displaying CB typically disseminate less harmful content, irrespective of political affiliation. However, distinct toxicity patterns emerge among different CB cohorts. Compared to their non-CB counterparts, CB participants show marginally elevated toxicity levels only when considering their original posts. We further show the effects of CB-driven toxic content on non-CB users, gauging its impact based on political leanings. Our findings suggest a nuanced but statistically significant influence of CB on digital discourse.

{{</citation>}}


### (153/165) A Unified View on Neural Message Passing with Opinion Dynamics for Social Networks (Outongyi Lv et al., 2023)

{{<citation>}}

Outongyi Lv, Bingxin Zhou, Jing Wang, Xiang Xiao, Weishu Zhao, Lirong Zheng. (2023)  
**A Unified View on Neural Message Passing with Opinion Dynamics for Social Networks**  

---
Primary Category: cs.SI  
Categories: cs-AI, cs-SI, cs.SI, q-bio-QM  
Keywords: Social Network  
[Paper Link](http://arxiv.org/abs/2310.01272v2)  

---


**ABSTRACT**  
Social networks represent a common form of interconnected data frequently depicted as graphs within the domain of deep learning-based inference. These communities inherently form dynamic systems, achieving stability through continuous internal communications and opinion exchanges among social actors along their social ties. In contrast, neural message passing in deep learning provides a clear and intuitive mathematical framework for understanding information propagation and aggregation among connected nodes in graphs. Node representations are dynamically updated by considering both the connectivity and status of neighboring nodes. This research harmonizes concepts from sociometry and neural message passing to analyze and infer the behavior of dynamic systems. Drawing inspiration from opinion dynamics in sociology, we propose ODNet, a novel message passing scheme incorporating bounded confidence, to refine the influence weight of local nodes for message propagation. We adjust the similarity cutoffs of bounded confidence and influence weights of ODNet and define opinion exchange rules that align with the characteristics of social network graphs. We show that ODNet enhances prediction performance across various graph types and alleviates oversmoothing issues. Furthermore, our approach surpasses conventional baselines in graph representation learning and proves its practical significance in analyzing real-world co-occurrence networks of metabolic genes. Remarkably, our method simplifies complex social network graphs solely by leveraging knowledge of interaction frequencies among entities within the system. It accurately identifies internal communities and the roles of genes in different metabolic pathways, including opinion leaders, bridge communicators, and isolators.

{{</citation>}}


### (154/165) HyperGraphDis: Leveraging Hypergraphs for Contextual and Social-Based Disinformation Detection (Nikos Salamanos et al., 2023)

{{<citation>}}

Nikos Salamanos, Pantelitsa Leonidou, Nikolaos Laoutaris, Michael Sirivianos, Maria Aspri, Marius Paraschiv. (2023)  
**HyperGraphDis: Leveraging Hypergraphs for Contextual and Social-Based Disinformation Detection**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2310.01113v1)  

---


**ABSTRACT**  
In light of the growing impact of disinformation on social, economic, and political landscapes, accurate and efficient identification methods are increasingly critical. This paper introduces HyperGraphDis, a novel approach for detecting disinformation on Twitter that employs a hypergraph-based representation to capture (i) the intricate social structure arising from retweet cascades, (ii) relational features among users, and (iii) semantic and topical nuances. Evaluated on four Twitter datasets -- focusing on the 2016 U.S. Presidential election and the COVID-19 pandemic -- HyperGraphDis outperforms existing methods in both accuracy and computational efficiency, underscoring its effectiveness and scalability for tackling the challenges posed by disinformation dissemination. The HyperGraphDis displayed exceptional performance in an evaluation using a COVID-19-related dataset, achieving an impressive F1 score of approximately 92.5%. This result represents a notable improvement of around 7% compared to other existing methods. Additionally, significant enhancements in computation time were observed for both model training and inference processes. In terms of model training, completion times were noticeably accelerated, ranging from 1.6 to 16.5 times faster than previous benchmarks. Similarly, during inference, computational times demonstrated increased efficiency, ranging from 1.3 to 17.7 times faster than alternative methods.

{{</citation>}}


### (155/165) ETGraph: A Pioneering Dataset Bridging Ethereum and Twitter (Qian Wang et al., 2023)

{{<citation>}}

Qian Wang, Zhen Zhang, Zemin Liu, Shengliang Lu, Bingqiao Luo, Bingsheng He. (2023)  
**ETGraph: A Pioneering Dataset Bridging Ethereum and Twitter**  

---
Primary Category: cs.SI  
Categories: cs-AI, cs-SI, cs.SI  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2310.01015v1)  

---


**ABSTRACT**  
While numerous public blockchain datasets are available, their utility is constrained by a singular focus on blockchain data. This constraint limits the incorporation of relevant social network data into blockchain analysis, thereby diminishing the breadth and depth of insight that can be derived. To address the above limitation, we introduce ETGraph, a novel dataset that authentically links Ethereum and Twitter, marking the first and largest dataset of its kind. ETGraph combines Ethereum transaction records (2 million nodes and 30 million edges) and Twitter following data (1 million nodes and 3 million edges), bonding 30,667 Ethereum addresses with verified Twitter accounts sourced from OpenSea. Detailed statistical analysis on ETGraph highlights the structural differences between Twitter-matched and non-Twitter-matched Ethereum addresses. Extensive experiments, including Ethereum link prediction, wash-trading Ethereum addresses detection, and Twitter-Ethereum matching link prediction, emphasize the significant role of Twitter data in enhancing Ethereum analysis. ETGraph is available at https://etgraph.deno.dev/.

{{</citation>}}


### (156/165) Multi-triplet Feature Augmentation for Ponzi Scheme Detection in Ethereum (Chengxiang Jin et al., 2023)

{{<citation>}}

Chengxiang Jin, Jiajun Zhou, Shengbo Gong, Chenxuan Xie, Qi Xuan. (2023)  
**Multi-triplet Feature Augmentation for Ponzi Scheme Detection in Ethereum**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Augmentation, GNN  
[Paper Link](http://arxiv.org/abs/2310.00856v1)  

---


**ABSTRACT**  
Blockchain technology revolutionizes the Internet, but also poses increasing risks, particularly in cryptocurrency finance. On the Ethereum platform, Ponzi schemes, phishing scams, and a variety of other frauds emerge. Existing Ponzi scheme detection approaches based on heterogeneous transaction graph modeling leverages semantic information between node (account) pairs to establish connections, overlooking the semantic attributes inherent to the edges (interactions). To overcome this, we construct heterogeneous Ethereum interaction graphs with multiple triplet interaction patterns to better depict the real Ethereum environment. Based on this, we design a new framework named multi-triplet augmented heterogeneous graph neural network (MAHGNN) for Ponzi scheme detection. We introduce the Conditional Variational Auto Encoder (CVAE) to capture the semantic information of different triplet interaction patterns, which facilitates the characterization on account features. Extensive experiments demonstrate that MAHGNN is capable of addressing the problem of multi-edge interactions in heterogeneous Ethereum interaction graphs and achieving state-of-the-art performance in Ponzi scheme detection.

{{</citation>}}


## cs.NE (1)



### (157/165) Evolutionary Neural Architecture Search for Transformer in Knowledge Tracing (Shangshang Yang et al., 2023)

{{<citation>}}

Shangshang Yang, Xiaoshan Yu, Ye Tian, Xueming Yan, Haiping Ma, Xingyi Zhang. (2023)  
**Evolutionary Neural Architecture Search for Transformer in Knowledge Tracing**  

---
Primary Category: cs.NE  
Categories: cs-AI, cs-LG, cs-NE, cs.NE  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.01180v1)  

---


**ABSTRACT**  
Knowledge tracing (KT) aims to trace students' knowledge states by predicting whether students answer correctly on exercises. Despite the excellent performance of existing Transformer-based KT approaches, they are criticized for the manually selected input features for fusion and the defect of single global context modelling to directly capture students' forgetting behavior in KT, when the related records are distant from the current record in terms of time. To address the issues, this paper first considers adding convolution operations to the Transformer to enhance its local context modelling ability used for students' forgetting behavior, then proposes an evolutionary neural architecture search approach to automate the input feature selection and automatically determine where to apply which operation for achieving the balancing of the local/global context modelling. In the search space, the original global path containing the attention module in Transformer is replaced with the sum of a global path and a local path that could contain different convolutions, and the selection of input features is also considered. To search the best architecture, we employ an effective evolutionary algorithm to explore the search space and also suggest a search space reduction strategy to accelerate the convergence of the algorithm. Experimental results on the two largest and most challenging education datasets demonstrate the effectiveness of the architecture found by the proposed approach.

{{</citation>}}


## cs.NI (1)



### (158/165) Preliminary Performance Evaluation of a Satellite-to-HAP Communication Link (Giovanni Grieco et al., 2023)

{{<citation>}}

Giovanni Grieco, Giovanni Iacovelli, Mattia Sandri, Marco Giordani, Michele Zorzi, Luigi Alfredo Grieco. (2023)  
**Preliminary Performance Evaluation of a Satellite-to-HAP Communication Link**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2310.01143v1)  

---


**ABSTRACT**  
The emergence of Fifth-Generation (5G) communication networks has brought forth unprecedented connectivity with ultra-low latency, high data rates, and pervasive coverage. However, meeting the increasing demands of applications for seamless and high-quality communication, especially in rural areas, requires exploring innovative solutions that expand 5G beyond traditional terrestrial networks. Within the context of Non-Terrestrial Networks (NTNs), two promising technologies with vast potential are High Altitude Platforms (HAPs) and satellites. The combination of these two platforms is able to provide wide coverage and reliable communication in remote and inaccessible areas, and/or where terrestrial infrastructure is unavailable. This study evaluates the performance of the communication link between a Geostationary Equatorial Orbit (GEO) satellite and a HAP using the Internet of Drones Simulator (IoD-Sim), implemented in ns-3 and incorporating the 3GPP TR 38.811 channel model. The code base of IoD-Sim is extended to simulate HAPs, accounting for the Earths curvature in various geographic coordinate systems, and considering realistic mobility patterns. A simulation campaign is conducted to evaluate the GEO-to-HAP communication link in terms of Signal-to-Noise Ratio (SNR) in two different scenarios, considering the mobility of the HAP, and as a function of the frequency and the distance.

{{</citation>}}


## physics.plasm-ph (1)



### (159/165) Shaping of Magnetic Field Coils in Fusion Reactors using Bayesian Optimisation (Timothy Nunn et al., 2023)

{{<citation>}}

Timothy Nunn, Vignesh Gopakumar, Sebastien Kahn. (2023)  
**Shaping of Magnetic Field Coils in Fusion Reactors using Bayesian Optimisation**  

---
Primary Category: physics.plasm-ph  
Categories: cs-AI, physics-plasm-ph, physics.plasm-ph  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.01455v1)  

---


**ABSTRACT**  
Nuclear fusion using magnetic confinement holds promise as a viable method for sustainable energy. However, most fusion devices have been experimental and as we move towards energy reactors, we are entering into a new paradigm of engineering. Curating a design for a fusion reactor is a high-dimensional multi-output optimisation process. Through this work we demonstrate a proof-of-concept of an AI-driven strategy to help explore the design search space and identify optimum parameters. By utilising a Multi-Output Bayesian Optimisation scheme, our strategy is capable of identifying the Pareto front associated with the optimisation of the toroidal field coil shape of a tokamak. The optimisation helps to identify design parameters that would minimise the costs incurred while maximising the plasma stability by way of minimising magnetic ripples.

{{</citation>}}


## cs.DB (2)



### (160/165) Rel2Graph: Automated Mapping From Relational Databases to a Unified Property Knowledge Graph (Ziyu Zhao et al., 2023)

{{<citation>}}

Ziyu Zhao, Wei Liu, Tim French, Michael Stewart. (2023)  
**Rel2Graph: Automated Mapping From Relational Databases to a Unified Property Knowledge Graph**  

---
Primary Category: cs.DB  
Categories: cs-DB, cs.DB  
Keywords: Knowledge Graph, QA  
[Paper Link](http://arxiv.org/abs/2310.01080v1)  

---


**ABSTRACT**  
Although a few approaches are proposed to convert relational databases to graphs, there is a genuine lack of systematic evaluation across a wider spectrum of databases. Recognising the important issue of query mapping, this paper proposes an approach Rel2Graph, an automatic knowledge graph construction (KGC) approach from an arbitrary number of relational databases. Our approach also supports the mapping of conjunctive SQL queries into pattern-based NoSQL queries. We evaluate our proposed approach on two widely used relational database-oriented datasets: Spider and KaggleDBQA benchmarks for semantic parsing. We employ the execution accuracy (EA) metric to quantify the proportion of results by executing the NoSQL queries on the property knowledge graph we construct that aligns with the results of SQL queries performed on relational databases. Consequently, the counterpart property knowledge graph of benchmarks with high accuracy and integrity can be ensured. The code and data will be publicly available. The code and data are available at github\footnote{https://github.com/nlp-tlp/Rel2Graph}.

{{</citation>}}


### (161/165) QCFE: An efficient Feature engineering for query cost estimation (Yu Yan et al., 2023)

{{<citation>}}

Yu Yan, Hongzhi Wang, Junfang Huang, Dake Zhong, Man Yang, Kaixin Zhang, Tao Yu, Tianqing Wan. (2023)  
**QCFE: An efficient Feature engineering for query cost estimation**  

---
Primary Category: cs.DB  
Categories: cs-DB, cs.DB  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.00877v1)  

---


**ABSTRACT**  
Query cost estimation is a classical task for database management. Recently, researchers apply the AI-driven model to implement query cost estimation for achieving high accuracy. However, two defects of feature design lead to poor cost estimation accuracy-time efficiency. On the one hand, existing works only encode the query plan and data statistics while ignoring some other important variables, like storage structure, hardware, database knobs, etc. These variables also have significant impact on the query cost. On the other hand, due to the straightforward encoding design, existing works suffer heavy representation learning burden on ineffective dimensions of input. To meet the above two problems, we first propose an efficient feature engineering for query cost estimation, called QCFE. Specifically, we design a novel feature called feature snapshot to efficiently integrate the influences of the ignored variables. Further, we propose a difference-propagation feature reduction method for query cost estimation to filter the useless features. The experimental results demonstrate our QCFE could largely improve the time-accuracy efficiency on extensive benchmarks.

{{</citation>}}


## q-fin.RM (1)



### (162/165) Combining Deep Learning and GARCH Models for Financial Volatility and Risk Forecasting (Jakub Michańków et al., 2023)

{{<citation>}}

Jakub Michańków, Łukasz Kwiatkowski, Janusz Morajda. (2023)  
**Combining Deep Learning and GARCH Models for Financial Volatility and Risk Forecasting**  

---
Primary Category: q-fin.RM  
Categories: cs-AI, cs-LG, q-fin-CP, q-fin-GN, q-fin-RM, q-fin.RM  
Keywords: Financial  
[Paper Link](http://arxiv.org/abs/2310.01063v1)  

---


**ABSTRACT**  
In this paper, we develop a hybrid approach to forecasting the volatility and risk of financial instruments by combining common econometric GARCH time series models with deep learning neural networks. For the latter, we employ Gated Recurrent Unit (GRU) networks, whereas four different specifications are used as the GARCH component: standard GARCH, EGARCH, GJR-GARCH and APARCH. Models are tested using daily logarithmic returns on the S&P 500 index as well as gold price Bitcoin prices, with the three assets representing quite distinct volatility dynamics. As the main volatility estimator, also underlying the target function of our hybrid models, we use the price-range-based Garman-Klass estimator, modified to incorporate the opening and closing prices. Volatility forecasts resulting from the hybrid models are employed to evaluate the assets' risk using the Value-at-Risk (VaR) and Expected Shortfall (ES) at two different tolerance levels of 5% and 1%. Gains from combining the GARCH and GRU approaches are discussed in the contexts of both the volatility and risk forecasts. In general, it can be concluded that the hybrid solutions produce more accurate point volatility forecasts, although it does not necessarily translate into superior VaR and ES forecasts.

{{</citation>}}


## physics.soc-ph (1)



### (163/165) Epistemic integration and social segregation of AI in neuroscience (Sylvain Fontaine et al., 2023)

{{<citation>}}

Sylvain Fontaine, Floriana Gargiulo, Michel Dubois, Paola Tubaro. (2023)  
**Epistemic integration and social segregation of AI in neuroscience**  

---
Primary Category: physics.soc-ph  
Categories: cs-SI, physics-soc-ph, physics.soc-ph  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.01046v1)  

---


**ABSTRACT**  
In recent years, Artificial Intelligence (AI) shows a spectacular ability of insertion inside a variety of disciplines which use it for scientific advancements and which sometimes improve it for their conceptual and methodological needs. According to the transverse science framework originally conceived by Shinn and Joerges, AI can be seen as an instrument which is progressively acquiring an universal character through its diffusion across science. In this paper we address empirically one aspect of this diffusion, namely the penetration of AI into a specific field of research. Taking neuroscience as a case study, we conduct a scientometric analysis of the development of AI in this field We especially study the temporal egocentric citation network around the articles included in this literature, their represented journals and their authors linked together by a temporal collaboration network. We find that AI is driving the constitution of a particular disciplinary ecosystem in neuroscience which is distinct from other subfields when regarding the references, and which is gathering atypical scientific profiles who are coming from neuroscience or outside it. Moreover we observe that this AI community in neuroscience is socially confined in a specific zone of the neuroscience collaboration network, which is also keeping to publish in a small set of dedicated journals that are mostly active in AI research. According to these results, the diffusion of AI in a discipline such as neuroscience didn't really challenge its disciplinary orientations but rather induced the constitution of a dedicated socio-cognitive workforce inside this field.

{{</citation>}}


## physics.geo-ph (1)



### (164/165) Seismogram Transformer: A generic deep learning backbone network for multiple earthquake monitoring tasks (Sen Li et al., 2023)

{{<citation>}}

Sen Li, Xu Yang, Anye Cao, Changbin Wang, Yaoqi Liu, Yapeng Liu, Qiang Niu. (2023)  
**Seismogram Transformer: A generic deep learning backbone network for multiple earthquake monitoring tasks**  

---
Primary Category: physics.geo-ph  
Categories: cs-LG, physics-geo-ph, physics.geo-ph  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.01037v1)  

---


**ABSTRACT**  
Seismic records, known as seismograms, are crucial records of ground motion resulting from seismic events, constituting the backbone of earthquake research and monitoring. The latest advancements in deep learning have significantly facilitated various seismic signal processing tasks. This paper introduces a novel backbone neural network model designed for various seismic monitoring tasks, named Seismogram Transformer (SeisT). Thanks to its efficient network architecture, SeisT matches or even outperforms the state-of-the-art models in earthquake detection, seismic phase picking, first-motion polarity classification, magnitude estimation, and azimuth estimation tasks, particularly in terms of out-of-distribution generalization performance. SeisT consists of multiple network layers composed of different foundational blocks, which help the model understand multi-level feature representations of seismograms from low-level to high-level complex features, effectively extracting features such as frequency, phase, and time-frequency relationships from input seismograms. Three different-sized models were customized based on these diverse foundational modules. Through extensive experiments and performance evaluations, this study showcases the capabilities and potential of SeisT in advancing seismic signal processing and earthquake research.

{{</citation>}}


## cs.DC (1)



### (165/165) Helios: An Efficient Out-of-core GNN Training System on Terabyte-scale Graphs with In-memory Performance (Jie Sun et al., 2023)

{{<citation>}}

Jie Sun, Mo Sun, Zheng Zhang, Jun Xie, Zuocheng Shi, Zihan Yang, Jie Zhang, Fei Wu, Zeke Wang. (2023)  
**Helios: An Efficient Out-of-core GNN Training System on Terabyte-scale Graphs with In-memory Performance**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2310.00837v1)  

---


**ABSTRACT**  
Training graph neural networks (GNNs) on large-scale graph data holds immense promise for numerous real-world applications but remains a great challenge. Several disk-based GNN systems have been built to train large-scale graphs in a single machine. However, they often fall short in terms of performance, especially when training on terabyte-scale graphs. This is because existing disk-based systems either overly focus on minimizing the number of SSD accesses or do not fully overlap SSD accesses with GNN training, thus resulting in substantial unnecessary overhead on the CPU side and then low GPU utilization. To this end, we propose Helios, a system that can train GNN on terabyte graphs in a single machine while achieving throughput comparable with in-memory systems. To achieve this, we first present a GPU-initiated asynchronous disk IO stack, allowing the GPU to directly access graph data on SSD. This design only requires about 30% GPU cores to reach the almost maximal disk IO throughput and wastes no GPU cores between IO submission and IO completion such that the majority of GPU cores are left for other GNN kernels. Second, we design a GPU-managed heterogeneous cache that extends the cache hierarchy to heterogeneous CPU and GPU memory and thus enhances cache lookup throughput significantly by GPU parallelism. Finally, we build a deep GNN-aware pipeline that seamlessly integrates the computation and communication phases of the entire GNN training process, maximizing the utility of GPU computation cycles. Experimental results demonstrate that Helios can match the training throughput of in-memory GNN systems, even for terabyte-scale graphs. Remarkably, Helios surpasses the state-of-the-art GPU-managed baselines by up to 6.43x and exceeds CPU-managed baselines by over 182x on all terabyte-scale graphs.

{{</citation>}}
