---
draft: false
title: "arXiv @ 2023.11.22"
date: 2023-11-22
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.11.22"
    identifier: arxiv_20231122
    parent: 202311_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.LG (18)](#cslg-18)
- [cs.AI (18)](#csai-18)
- [cs.DB (1)](#csdb-1)
- [cs.CL (22)](#cscl-22)
- [cs.CR (4)](#cscr-4)
- [cs.CV (36)](#cscv-36)
- [eess.IV (6)](#eessiv-6)
- [cs.HC (2)](#cshc-2)
- [cs.SE (3)](#csse-3)
- [cs.RO (2)](#csro-2)
- [stat.ML (1)](#statml-1)
- [cs.SI (1)](#cssi-1)
- [cs.IR (1)](#csir-1)
- [cs.PL (1)](#cspl-1)
- [eess.SP (1)](#eesssp-1)
- [cs.DC (2)](#csdc-2)
- [quant-ph (2)](#quant-ph-2)
- [cs.DL (1)](#csdl-1)

## cs.LG (18)



### (1/122) Provable Representation with Efficient Planning for Partially Observable Reinforcement Learning (Hongming Zhang et al., 2023)

{{<citation>}}

Hongming Zhang, Tongzheng Ren, Chenjun Xiao, Dale Schuurmans, Bo Dai. (2023)  
**Provable Representation with Efficient Planning for Partially Observable Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, stat-ML  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.12244v1)  

---


**ABSTRACT**  
In real-world reinforcement learning problems, the state information is often only partially observable, which breaks the basic assumption in Markov decision processes, and thus, leads to inferior performances. Partially Observable Markov Decision Processes have been introduced to explicitly take the issue into account for learning, exploration, and planning, but presenting significant computational and statistical challenges. To address these difficulties, we exploit the representation view, which leads to a coherent design framework for a practically tractable reinforcement learning algorithm upon partial observations. We provide a theoretical analysis for justifying the statistical efficiency of the proposed algorithm. We also empirically demonstrate the proposed algorithm can surpass state-of-the-art performance with partial observations across various benchmarks, therefore, pushing reliable reinforcement learning towards more practical applications.

{{</citation>}}


### (2/122) Node classification in random trees (Wouter W. L. Nuijten et al., 2023)

{{<citation>}}

Wouter W. L. Nuijten, Vlado Menkovski. (2023)  
**Node classification in random trees**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SI, cs.LG  
Keywords: Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2311.12167v1)  

---


**ABSTRACT**  
We propose a method for the classification of objects that are structured as random trees. Our aim is to model a distribution over the node label assignments in settings where the tree data structure is associated with node attributes (typically high dimensional embeddings). The tree topology is not predetermined and none of the label assignments are present during inference. Other methods that produce a distribution over node label assignment in trees (or more generally in graphs) either assume conditional independence of the label assignment, operate on a fixed graph topology, or require part of the node labels to be observed. Our method defines a Markov Network with the corresponding topology of the random tree and an associated Gibbs distribution. We parameterize the Gibbs distribution with a Graph Neural Network that operates on the random tree and the node embeddings. This allows us to estimate the likelihood of node assignments for a given random tree and use MCMC to sample from the distribution of node assignments.   We evaluate our method on the tasks of node classification in trees on the Stanford Sentiment Treebank dataset. Our method outperforms the baselines on this dataset, demonstrating its effectiveness for modeling joint distributions of node labels in random trees.

{{</citation>}}


### (3/122) Risk-averse Batch Active Inverse Reward Design (Panagiotis Liampas, 2023)

{{<citation>}}

Panagiotis Liampas. (2023)  
**Risk-averse Batch Active Inverse Reward Design**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.12004v1)  

---


**ABSTRACT**  
Designing a perfect reward function that depicts all the aspects of the intended behavior is almost impossible, especially generalizing it outside of the training environments. Active Inverse Reward Design (AIRD) proposed the use of a series of queries, comparing possible reward functions in a single training environment. This allows the human to give information to the agent about suboptimal behaviors, in order to compute a probability distribution over the intended reward function. However, it ignores the possibility of unknown features appearing in real-world environments, and the safety measures needed until the agent completely learns the reward function. I improved this method and created Risk-averse Batch Active Inverse Reward Design (RBAIRD), which constructs batches, sets of environments the agent encounters when being used in the real world, processes them sequentially, and, for a predetermined number of iterations, asks queries that the human needs to answer for each environment of the batch. After this process is completed in one batch, the probabilities have been improved and are transferred to the next batch. This makes it capable of adapting to real-world scenarios and learning how to treat unknown features it encounters for the first time. I also integrated a risk-averse planner, similar to that of Inverse Reward Design (IRD), which samples a set of reward functions from the probability distribution and computes a trajectory that takes the most certain rewards possible. This ensures safety while the agent is still learning the reward function, and enables the use of this approach in situations where cautiousness is vital. RBAIRD outperformed the previous approaches in terms of efficiency, accuracy, and action certainty, demonstrated quick adaptability to new, unknown features, and can be more widely used for the alignment of crucial, powerful AI models.

{{</citation>}}


### (4/122) Provably Efficient CVaR RL in Low-rank MDPs (Yulai Zhao et al., 2023)

{{<citation>}}

Yulai Zhao, Wenhao Zhan, Xiaoyan Hu, Ho-fung Leung, Farzan Farnia, Wen Sun, Jason D. Lee. (2023)  
**Provably Efficient CVaR RL in Low-rank MDPs**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.11965v1)  

---


**ABSTRACT**  
We study risk-sensitive Reinforcement Learning (RL), where we aim to maximize the Conditional Value at Risk (CVaR) with a fixed risk tolerance $\tau$. Prior theoretical work studying risk-sensitive RL focuses on the tabular Markov Decision Processes (MDPs) setting. To extend CVaR RL to settings where state space is large, function approximation must be deployed. We study CVaR RL in low-rank MDPs with nonlinear function approximation. Low-rank MDPs assume the underlying transition kernel admits a low-rank decomposition, but unlike prior linear models, low-rank MDPs do not assume the feature or state-action representation is known. We propose a novel Upper Confidence Bound (UCB) bonus-driven algorithm to carefully balance the interplay between exploration, exploitation, and representation learning in CVaR RL. We prove that our algorithm achieves a sample complexity of $\tilde{O}\left(\frac{H^7 A^2 d^4}{\tau^2 \epsilon^2}\right)$ to yield an $\epsilon$-optimal CVaR, where $H$ is the length of each episode, $A$ is the capacity of action space, and $d$ is the dimension of representations. Computational-wise, we design a novel discretized Least-Squares Value Iteration (LSVI) algorithm for the CVaR objective as the planning oracle and show that we can find the near-optimal policy in a polynomial running time with a Maximum Likelihood Estimation oracle. To our knowledge, this is the first provably efficient CVaR RL algorithm in low-rank MDPs.

{{</citation>}}


### (5/122) NNG-Mix: Improving Semi-supervised Anomaly Detection with Pseudo-anomaly Generation (Hao Dong et al., 2023)

{{<citation>}}

Hao Dong, Gaëtan Frusque, Yue Zhao, Eleni Chatzi, Olga Fink. (2023)  
**NNG-Mix: Improving Semi-supervised Anomaly Detection with Pseudo-anomaly Generation**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: Anomaly Detection, NLP  
[Paper Link](http://arxiv.org/abs/2311.11961v1)  

---


**ABSTRACT**  
Anomaly detection (AD) is essential in identifying rare and often critical events in complex systems, finding applications in fields such as network intrusion detection, financial fraud detection, and fault detection in infrastructure and industrial systems. While AD is typically treated as an unsupervised learning task due to the high cost of label annotation, it is more practical to assume access to a small set of labeled anomaly samples from domain experts, as is the case for semi-supervised anomaly detection. Semi-supervised and supervised approaches can leverage such labeled data, resulting in improved performance. In this paper, rather than proposing a new semi-supervised or supervised approach for AD, we introduce a novel algorithm for generating additional pseudo-anomalies on the basis of the limited labeled anomalies and a large volume of unlabeled data. This serves as an augmentation to facilitate the detection of new anomalies. Our proposed algorithm, named Nearest Neighbor Gaussian Mixup (NNG-Mix), efficiently integrates information from both labeled and unlabeled data to generate pseudo-anomalies. We compare the performance of this novel algorithm with commonly applied augmentation techniques, such as Mixup and Cutout. We evaluate NNG-Mix by training various existing semi-supervised and supervised anomaly detection algorithms on the original training data along with the generated pseudo-anomalies. Through extensive experiments on 57 benchmark datasets in ADBench, reflecting different data types, we demonstrate that NNG-Mix outperforms other data augmentation methods. It yields significant performance improvements compared to the baselines trained exclusively on the original training data. Notably, NNG-Mix yields up to 16.4%, 8.8%, and 8.0% improvements on Classical, CV, and NLP datasets in ADBench. Our source code will be available at https://github.com/donghao51/NNG-Mix.

{{</citation>}}


### (6/122) Correlated Attention in Transformers for Multivariate Time Series (Quang Minh Nguyen et al., 2023)

{{<citation>}}

Quang Minh Nguyen, Lam M. Nguyen, Subhro Das. (2023)  
**Correlated Attention in Transformers for Multivariate Time Series**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Attention, Time Series, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.11959v1)  

---


**ABSTRACT**  
Multivariate time series (MTS) analysis prevails in real-world applications such as finance, climate science and healthcare. The various self-attention mechanisms, the backbone of the state-of-the-art Transformer-based models, efficiently discover the temporal dependencies, yet cannot well capture the intricate cross-correlation between different features of MTS data, which inherently stems from complex dynamical systems in practice. To this end, we propose a novel correlated attention mechanism, which not only efficiently captures feature-wise dependencies, but can also be seamlessly integrated within the encoder blocks of existing well-known Transformers to gain efficiency improvement. In particular, correlated attention operates across feature channels to compute cross-covariance matrices between queries and keys with different lag values, and selectively aggregate representations at the sub-series level. This architecture facilitates automated discovery and representation learning of not only instantaneous but also lagged cross-correlations, while inherently capturing time series auto-correlation. When combined with prevalent Transformer baselines, correlated attention mechanism constitutes a better alternative for encoder-only architectures, which are suitable for a wide range of tasks including imputation, anomaly detection and classification. Extensive experiments on the aforementioned tasks consistently underscore the advantages of correlated attention mechanism in enhancing base Transformer models, and demonstrate our state-of-the-art results in imputation, anomaly detection and classification.

{{</citation>}}


### (7/122) Ovarian Cancer Data Analysis using Deep Learning: A Systematic Review from the Perspectives of Key Features of Data Analysis and AI Assurance (Muta Tah Hira et al., 2023)

{{<citation>}}

Muta Tah Hira, Mohammad A. Razzaque, Mosharraf Sarker. (2023)  
**Ovarian Cancer Data Analysis using Deep Learning: A Systematic Review from the Perspectives of Key Features of Data Analysis and AI Assurance**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.11932v1)  

---


**ABSTRACT**  
Background and objectives: By extracting this information, Machine or Deep Learning (ML/DL)-based autonomous data analysis tools can assist clinicians and cancer researchers in discovering patterns and relationships from complex data sets. Many DL-based analyses on ovarian cancer (OC) data have recently been published. These analyses are highly diverse in various aspects of cancer (e.g., subdomain(s) and cancer type they address) and data analysis features. However, a comprehensive understanding of these analyses in terms of these features and AI assurance (AIA) is currently lacking. This systematic review aims to fill this gap by examining the existing literature and identifying important aspects of OC data analysis using DL, explicitly focusing on the key features and AI assurance perspectives. Methods: The PRISMA framework was used to conduct comprehensive searches in three journal databases. Only studies published between 2015 and 2023 in peer-reviewed journals were included in the analysis. Results: In the review, a total of 96 DL-driven analyses were examined. The findings reveal several important insights regarding DL-driven ovarian cancer data analysis: - Most studies 71% (68 out of 96) focused on detection and diagnosis, while no study addressed the prediction and prevention of OC. - The analyses were predominantly based on samples from a non-diverse population (75% (72/96 studies)), limited to a geographic location or country. - Only a small proportion of studies (only 33% (32/96)) performed integrated analyses, most of which used homogeneous data (clinical or omics). - Notably, a mere 8.3% (8/96) of the studies validated their models using external and diverse data sets, highlighting the need for enhanced model validation, and - The inclusion of AIA in cancer data analysis is in a very early stage; only 2.1% (2/96) explicitly addressed AIA through explainability.

{{</citation>}}


### (8/122) Deep Calibration of Market Simulations using Neural Density Estimators and Embedding Networks (Namid R. Stillman et al., 2023)

{{<citation>}}

Namid R. Stillman, Rory Baggott, Justin Lyon, Jianfei Zhang, Dingqiu Zhu, Tao Chen, Perukrishnen Vytelingum. (2023)  
**Deep Calibration of Market Simulations using Neural Density Estimators and Embedding Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, q-fin-CP, stat-ML  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2311.11913v1)  

---


**ABSTRACT**  
The ability to construct a realistic simulator of financial exchanges, including reproducing the dynamics of the limit order book, can give insight into many counterfactual scenarios, such as a flash crash, a margin call, or changes in macroeconomic outlook. In recent years, agent-based models have been developed that reproduce many features of an exchange, as summarised by a set of stylised facts and statistics. However, the ability to calibrate simulators to a specific period of trading remains an open challenge. In this work, we develop a novel approach to the calibration of market simulators by leveraging recent advances in deep learning, specifically using neural density estimators and embedding networks. We demonstrate that our approach is able to correctly identify high probability parameter sets, both when applied to synthetic and historical data, and without reliance on manually selected or weighted ensembles of stylised facts.

{{</citation>}}


### (9/122) AMES: A Differentiable Embedding Space Selection Framework for Latent Graph Inference (Yuan Lu et al., 2023)

{{<citation>}}

Yuan Lu, Haitz Sáez de Ocáriz Borde, Pietro Liò. (2023)  
**AMES: A Differentiable Embedding Space Selection Framework for Latent Graph Inference**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SI, cs.LG, stat-ML  
Keywords: Attention, Embedding, GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2311.11891v1)  

---


**ABSTRACT**  
In real-world scenarios, although data entities may possess inherent relationships, the specific graph illustrating their connections might not be directly accessible. Latent graph inference addresses this issue by enabling Graph Neural Networks (GNNs) to operate on point cloud data, dynamically learning the necessary graph structure. These graphs are often derived from a latent embedding space, which can be modeled using Euclidean, hyperbolic, spherical, or product spaces. However, currently, there is no principled differentiable method for determining the optimal embedding space. In this work, we introduce the Attentional Multi-Embedding Selection (AMES) framework, a differentiable method for selecting the best embedding space for latent graph inference through backpropagation, considering a downstream task. Our framework consistently achieves comparable or superior results compared to previous methods for latent graph inference across five benchmark datasets. Importantly, our approach eliminates the need for conducting multiple experiments to identify the optimal embedding space. Furthermore, we explore interpretability techniques that track the gradient contributions of different latent graphs, shedding light on how our attention-based, fully differentiable approach learns to choose the appropriate latent space. In line with previous works, our experiments emphasize the advantages of hyperbolic spaces in enhancing performance. More importantly, our interpretability framework provides a general approach for quantitatively comparing embedding spaces across different tasks based on their contributions, a dimension that has been overlooked in previous literature on latent graph inference.

{{</citation>}}


### (10/122) Zero redundancy distributed learning with differential privacy (Zhiqi Bu et al., 2023)

{{<citation>}}

Zhiqi Bu, Justin Chiu, Ruixuan Liu, Sheng Zha, George Karypis. (2023)  
**Zero redundancy distributed learning with differential privacy**  

---
Primary Category: cs.LG  
Categories: cs-CC, cs-CR, cs-DC, cs-LG, cs.LG  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2311.11822v1)  

---


**ABSTRACT**  
Deep learning using large models have achieved great success in a wide range of domains. However, training these models on billions of parameters is very challenging in terms of the training speed, memory cost, and communication efficiency, especially under the privacy-preserving regime with differential privacy (DP). On the one hand, DP optimization has comparable efficiency to the standard non-private optimization on a single GPU, but on multiple GPUs, existing DP distributed learning (such as pipeline parallel) has suffered from significantly worse efficiency. On the other hand, the Zero Redundancy Optimizer (ZeRO) is a state-of-the-art solution to the standard distributed learning, exhibiting excellent training efficiency on large models, but to work compatibly with DP is technically complicated. In this work, we develop a new systematic solution, DP-ZeRO, (I) to scale up the trainable DP model size, e.g. to GPT-100B, (II) to obtain the same computation and communication efficiency as the standard ZeRO, and (III) to enable mixed-precision DP training. Our DP-ZeRO, like the standard ZeRO, has the potential to train models with arbitrary size and is evaluated on the world's largest DP models in terms of the number of trainable parameters.

{{</citation>}}


### (11/122) Unveiling the Unseen Potential of Graph Learning through MLPs: Effective Graph Learners Using Propagation-Embracing MLPs (Yong-Min Shin et al., 2023)

{{<citation>}}

Yong-Min Shin, Won-Yong Shin. (2023)  
**Unveiling the Unseen Potential of Graph Learning through MLPs: Effective Graph Learners Using Propagation-Embracing MLPs**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-IT, cs-LG, cs-NE, cs-SI, cs.LG, math-IT  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2311.11759v1)  

---


**ABSTRACT**  
Recent studies attempted to utilize multilayer perceptrons (MLPs) to solve semi-supervised node classification on graphs, by training a student MLP by knowledge distillation (KD) from a teacher graph neural network (GNN). While previous studies have focused mostly on training the student MLP by matching the output probability distributions between the teacher and student models during KD, it has not been systematically studied how to inject the structural information in an explicit and interpretable manner. Inspired by GNNs that separate feature transformation $T$ and propagation $\Pi$, we re-frame the KD process as enabling the student MLP to explicitly learn both $T$ and $\Pi$. Although this can be achieved by applying the inverse propagation $\Pi^{-1}$ before distillation from the teacher GNN, it still comes with a high computational cost from large matrix multiplications during training. To solve this problem, we propose Propagate & Distill (P&D), which propagates the output of the teacher GNN before KD and can be interpreted as an approximate process of the inverse propagation $\Pi^{-1}$. Through comprehensive evaluations using real-world benchmark datasets, we demonstrate the effectiveness of P&D by showing further performance boost of the student MLP.

{{</citation>}}


### (12/122) Unveiling the Power of Self-Attention for Shipping Cost Prediction: The Rate Card Transformer (P Aditya Sreekar et al., 2023)

{{<citation>}}

P Aditya Sreekar, Sahil Verma, Varun Madhavan, Abhishek Persad. (2023)  
**Unveiling the Power of Self-Attention for Shipping Cost Prediction: The Rate Card Transformer**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Amazon, Attention, Self-Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2311.11694v1)  

---


**ABSTRACT**  
Amazon ships billions of packages to its customers annually within the United States. Shipping cost of these packages are used on the day of shipping (day 0) to estimate profitability of sales. Downstream systems utilize these days 0 profitability estimates to make financial decisions, such as pricing strategies and delisting loss-making products. However, obtaining accurate shipping cost estimates on day 0 is complex for reasons like delay in carrier invoicing or fixed cost components getting recorded at monthly cadence. Inaccurate shipping cost estimates can lead to bad decision, such as pricing items too low or high, or promoting the wrong product to the customers. Current solutions for estimating shipping costs on day 0 rely on tree-based models that require extensive manual engineering efforts. In this study, we propose a novel architecture called the Rate Card Transformer (RCT) that uses self-attention to encode all package shipping information such as package attributes, carrier information and route plan. Unlike other transformer-based tabular models, RCT has the ability to encode a variable list of one-to-many relations of a shipment, allowing it to capture more information about a shipment. For example, RCT can encode properties of all products in a package. Our results demonstrate that cost predictions made by the RCT have 28.82% less error compared to tree-based GBDT model. Moreover, the RCT outperforms the state-of-the-art transformer-based tabular model, FTTransformer, by 6.08%. We also illustrate that the RCT learns a generalized manifold of the rate card that can improve the performance of tree-based models.

{{</citation>}}


### (13/122) Incorporating LLM Priors into Tabular Learners (Max Zhu et al., 2023)

{{<citation>}}

Max Zhu, Siniša Stanivuk, Andrija Petrovic, Mladen Nikolic, Pietro Lio. (2023)  
**Incorporating LLM Priors into Tabular Learners**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.11628v1)  

---


**ABSTRACT**  
We present a method to integrate Large Language Models (LLMs) and traditional tabular data classification techniques, addressing LLMs challenges like data serialization sensitivity and biases. We introduce two strategies utilizing LLMs for ranking categorical variables and generating priors on correlations between continuous variables and targets, enhancing performance in few-shot scenarios. We focus on Logistic Regression, introducing MonotonicLR that employs a non-linear monotonic function for mapping ordinals to cardinals while preserving LLM-determined orders. Validation against baseline models reveals the superior performance of our approach, especially in low-data scenarios, while remaining interpretable.

{{</citation>}}


### (14/122) A novel transformer-based approach for soil temperature prediction (Muhammet Mucahit Enes Yurtsever et al., 2023)

{{<citation>}}

Muhammet Mucahit Enes Yurtsever, Ayhan Kucukmanisa, Zeynep Hilal Kilimci. (2023)  
**A novel transformer-based approach for soil temperature prediction**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CE, cs-LG, cs.LG, physics-ao-ph  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.11626v1)  

---


**ABSTRACT**  
Soil temperature is one of the most significant parameters that plays a crucial role in glacier energy, dynamics of mass balance, processes of surface hydrological, coaction of glacier-atmosphere, nutrient cycling, ecological stability, the management of soil, water, and field crop. In this work, we introduce a novel approach using transformer models for the purpose of forecasting soil temperature prediction. To the best of our knowledge, the usage of transformer models in this work is the very first attempt to predict soil temperature. Experiments are carried out using six different FLUXNET stations by modeling them with five different transformer models, namely, Vanilla Transformer, Informer, Autoformer, Reformer, and ETSformer. To demonstrate the effectiveness of the proposed model, experiment results are compared with both deep learning approaches and literature studies. Experiment results show that the utilization of transformer models ensures a significant contribution to the literature, thence determining the new state-of-the-art.

{{</citation>}}


### (15/122) Replay-enhanced Continual Reinforcement Learning (Tiantian Zhang et al., 2023)

{{<citation>}}

Tiantian Zhang, Kevin Zehua Shen, Zichuan Lin, Bo Yuan, Xueqian Wang, Xiu Li, Deheng Ye. (2023)  
**Replay-enhanced Continual Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.11557v1)  

---


**ABSTRACT**  
Replaying past experiences has proven to be a highly effective approach for averting catastrophic forgetting in supervised continual learning. However, some crucial factors are still largely ignored, making it vulnerable to serious failure, when used as a solution to forgetting in continual reinforcement learning, even in the context of perfect memory where all data of previous tasks are accessible in the current task. On the one hand, since most reinforcement learning algorithms are not invariant to the reward scale, the previously well-learned tasks (with high rewards) may appear to be more salient to the current learning process than the current task (with small initial rewards). This causes the agent to concentrate on those salient tasks at the expense of generality on the current task. On the other hand, offline learning on replayed tasks while learning a new task may induce a distributional shift between the dataset and the learned policy on old tasks, resulting in forgetting. In this paper, we introduce RECALL, a replay-enhanced method that greatly improves the plasticity of existing replay-based methods on new tasks while effectively avoiding the recurrence of catastrophic forgetting in continual reinforcement learning. RECALL leverages adaptive normalization on approximate targets and policy distillation on old tasks to enhance generality and stability, respectively. Extensive experiments on the Continual World benchmark show that RECALL performs significantly better than purely perfect memory replay, and achieves comparable or better overall performance against state-of-the-art continual learning methods.

{{</citation>}}


### (16/122) MultiLoRA: Democratizing LoRA for Better Multi-Task Learning (Yiming Wang et al., 2023)

{{<citation>}}

Yiming Wang, Yu Lin, Xiaodong Zeng, Guannan Zhang. (2023)  
**MultiLoRA: Democratizing LoRA for Better Multi-Task Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2311.11501v1)  

---


**ABSTRACT**  
LoRA achieves remarkable resource efficiency and comparable performance when adapting LLMs for specific tasks. Since ChatGPT demonstrated superior performance on various tasks, there has been a growing desire to adapt one model for all tasks. However, the explicit low-rank of LoRA limits the adaptation performance in complex multi-task scenarios. LoRA is dominated by a small number of top singular vectors while fine-tuning decomposes into a set of less important unitary transforms. In this paper, we propose MultiLoRA for better multi-task adaptation by reducing the dominance of top singular vectors observed in LoRA. MultiLoRA scales LoRA modules horizontally and change parameter initialization of adaptation matrices to reduce parameter dependency, thus yields more balanced unitary subspaces. We unprecedentedly construct specialized training data by mixing datasets of instruction follow, natural language understanding, world knowledge, to cover semantically and syntactically different samples. With only 2.5% of additional parameters, MultiLoRA outperforms single LoRA counterparts and fine-tuning on multiple benchmarks and model scales. Further investigation into weight update matrices of MultiLoRA exhibits reduced dependency on top singular vectors and more democratic unitary transform contributions.

{{</citation>}}


### (17/122) A Multi-Center Study on the Adaptability of a Shared Foundation Model for Electronic Health Records (Lin Lawrence Guo et al., 2023)

{{<citation>}}

Lin Lawrence Guo, Jason Fries, Ethan Steinberg, Scott Lanyon Fleming, Keith Morse, Catherine Aftandilian, Jose Posada, Nigam Shah, Lillian Sung. (2023)  
**A Multi-Center Study on the Adaptability of a Shared Foundation Model for Electronic Health Records**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.11483v1)  

---


**ABSTRACT**  
Foundation models hold promise for transforming AI in healthcare by providing modular components that are easily adaptable to downstream healthcare tasks, making AI development more scalable and cost-effective. Structured EHR foundation models, trained on coded medical records from millions of patients, demonstrated benefits including increased performance with fewer training labels, and improved robustness to distribution shifts. However, questions remain on the feasibility of sharing these models across different hospitals and their performance for local task adaptation. This multi-center study examined the adaptability of a recently released structured EHR foundation model ($FM_{SM}$), trained on longitudinal medical record data from 2.57M Stanford Medicine patients. Experiments were conducted using EHR data at The Hospital for Sick Children and MIMIC-IV. We assessed both adaptability via continued pretraining on local data, and task adaptability compared to baselines of training models from scratch at each site, including a local foundation model. We evaluated the performance of these models on 8 clinical prediction tasks. In both datasets, adapting the off-the-shelf $FM_{SM}$ matched the performance of GBM models locally trained on all data while providing a 13% improvement in settings with few task-specific training labels. With continued pretraining on local data, label efficiency substantially improved, such that $FM_{SM}$ required fewer than 1% of training examples to match the fully trained GBM's performance. Continued pretraining was also 60 to 90% more sample-efficient than training local foundation models from scratch. Our findings show that adapting shared EHR foundation models across hospitals provides improved prediction performance at less cost, underscoring the utility of base foundation models as modular components to streamline the development of healthcare AI.

{{</citation>}}


### (18/122) CSGNN: Conquering Noisy Node labels via Dynamic Class-wise Selection (Yifan Li et al., 2023)

{{<citation>}}

Yifan Li, Zhen Tan, Kai Shu, Zongsheng Cao, Yu Kong, Huan Liu. (2023)  
**CSGNN: Conquering Noisy Node labels via Dynamic Class-wise Selection**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2311.11473v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) have emerged as a powerful tool for representation learning on graphs, but they often suffer from overfitting and label noise issues, especially when the data is scarce or imbalanced. Different from the paradigm of previous methods that rely on single-node confidence, in this paper, we introduce a novel Class-wise Selection for Graph Neural Networks, dubbed CSGNN, which employs a neighbor-aggregated latent space to adaptively select reliable nodes across different classes. Specifically, 1) to tackle the class imbalance issue, we introduce a dynamic class-wise selection mechanism, leveraging the clustering technique to identify clean nodes based on the neighbor-aggregated confidences. In this way, our approach can avoid the pitfalls of biased sampling which is common with global threshold techniques. 2) To alleviate the problem of noisy labels, built on the concept of the memorization effect, CSGNN prioritizes learning from clean nodes before noisy ones, thereby iteratively enhancing model performance while mitigating label noise. Through extensive experiments, we demonstrate that CSGNN outperforms state-of-the-art methods in terms of both effectiveness and robustness.

{{</citation>}}


## cs.AI (18)



### (19/122) InteraSSort: Interactive Assortment Planning Using Large Language Models (Saketh Reddy Karra et al., 2023)

{{<citation>}}

Saketh Reddy Karra, Theja Tulabandhula. (2023)  
**InteraSSort: Interactive Assortment Planning Using Large Language Models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.12241v1)  

---


**ABSTRACT**  
Assortment planning, integral to multiple commercial offerings, is a key problem studied in e-commerce and retail settings. Numerous variants of the problem along with their integration into business solutions have been thoroughly investigated in the existing literature. However, the nuanced complexities of in-store planning and a lack of optimization proficiency among store planners with strong domain expertise remain largely overlooked. These challenges frequently necessitate collaborative efforts with multiple stakeholders which often lead to prolonged decision-making processes and significant delays. To mitigate these challenges and capitalize on the advancements of Large Language Models (LLMs), we propose an interactive assortment planning framework, InteraSSort that augments LLMs with optimization tools to assist store planners in making decisions through interactive conversations. Specifically, we develop a solution featuring a user-friendly interface that enables users to express their optimization objectives as input text prompts to InteraSSort and receive tailored optimized solutions as output. Our framework extends beyond basic functionality by enabling the inclusion of additional constraints through interactive conversation, facilitating precise and highly customized decision-making. Extensive experiments demonstrate the effectiveness of our framework and potential extensions to a broad range of operations management challenges.

{{</citation>}}


### (20/122) Nepotistically Trained Generative-AI Models Collapse (Matyas Bohacek et al., 2023)

{{<citation>}}

Matyas Bohacek, Hany Farid. (2023)  
**Nepotistically Trained Generative-AI Models Collapse**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CV, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.12202v1)  

---


**ABSTRACT**  
Trained on massive amounts of human-generated content, AI (artificial intelligence) image synthesis is capable of reproducing semantically coherent images that match the visual appearance of its training data. We show that when retrained on even small amounts of their own creation, these generative-AI models produce highly distorted images. We also show that this distortion extends beyond the text prompts used in retraining, and that once poisoned, the models struggle to fully heal even after retraining on only real images.

{{</citation>}}


### (21/122) ChatGPT and post-test probability (Samuel J. Weisenthal, 2023)

{{<citation>}}

Samuel J. Weisenthal. (2023)  
**ChatGPT and post-test probability**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI, stat-AP  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2311.12188v1)  

---


**ABSTRACT**  
Reinforcement learning-based large language models, such as ChatGPT, are believed to have potential to aid human experts in many domains, including healthcare. There is, however, little work on ChatGPT's ability to perform a key task in healthcare: formal, probabilistic medical diagnostic reasoning. This type of reasoning is used, for example, to update a pre-test probability to a post-test probability. In this work, we probe ChatGPT's ability to perform this task. In particular, we ask ChatGPT to give examples of how to use Bayes rule for medical diagnosis. Our prompts range from queries that use terminology from pure probability (e.g., requests for a "posterior probability") to queries that use terminology from the medical diagnosis literature (e.g., requests for a "post-test probability"). We show how the introduction of medical variable names leads to an increase in the number of errors that ChatGPT makes. Given our results, we also show how one can use prompt engineering to facilitate ChatGPT's partial avoidance of these errors. We discuss our results in light of recent commentaries on sensitivity and specificity. We also discuss how our results might inform new research directions for large language models.

{{</citation>}}


### (22/122) GPQA: A Graduate-Level Google-Proof Q&A Benchmark (David Rein et al., 2023)

{{<citation>}}

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, Samuel R. Bowman. (2023)  
**GPQA: A Graduate-Level Google-Proof Q&A Benchmark**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: AI, GPT, GPT-4, Google, QA  
[Paper Link](http://arxiv.org/abs/2311.12022v1)  

---


**ABSTRACT**  
We present GPQA, a challenging dataset of 448 multiple-choice questions written by domain experts in biology, physics, and chemistry. We ensure that the questions are high-quality and extremely difficult: experts who have or are pursuing PhDs in the corresponding domains reach 65% accuracy (74% when discounting clear mistakes the experts identified in retrospect), while highly skilled non-expert validators only reach 34% accuracy, despite spending on average over 30 minutes with unrestricted access to the web (i.e., the questions are "Google-proof"). The questions are also difficult for state-of-the-art AI systems, with our strongest GPT-4 based baseline achieving 39% accuracy. If we are to use future AI systems to help us answer very hard questions, for example, when developing new scientific knowledge, we need to develop scalable oversight methods that enable humans to supervise their outputs, which may be difficult even if the supervisors are themselves skilled and knowledgeable. The difficulty of GPQA both for skilled non-experts and frontier AI systems should enable realistic scalable oversight experiments, which we hope can help devise ways for human experts to reliably get truthful information from AI systems that surpass human capabilities.

{{</citation>}}


### (23/122) Steering Responsible AI: A Case for Algorithmic Pluralism (Stefaan G. Verhulst, 2023)

{{<citation>}}

Stefaan G. Verhulst. (2023)  
**Steering Responsible AI: A Case for Algorithmic Pluralism**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.12010v1)  

---


**ABSTRACT**  
In this paper, I examine questions surrounding AI neutrality through the prism of existing literature and scholarship about mediation and media pluralism. Such traditions, I argue, provide a valuable theoretical framework for how we should approach the (likely) impending era of AI mediation. In particular, I suggest examining further the notion of algorithmic pluralism. Contrasting this notion to the dominant idea of algorithmic transparency, I seek to describe what algorithmic pluralism may be, and present both its opportunities and challenges. Implemented thoughtfully and responsibly, I argue, Algorithmic or AI pluralism has the potential to sustain the diversity, multiplicity, and inclusiveness that are so vital to democracy.

{{</citation>}}


### (24/122) Generalization of Fitness Exercise Recognition from Doppler Measurements by Domain-adaption and Few-Shot Learning (Biying Fu et al., 2023)

{{<citation>}}

Biying Fu, Naser Damer, Florian Kirchbuchner, Arjan Kuijper. (2023)  
**Generalization of Fitness Exercise Recognition from Doppler Measurements by Domain-adaption and Few-Shot Learning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CV, cs.AI  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2311.11910v1)  

---


**ABSTRACT**  
In previous works, a mobile application was developed using an unmodified commercial off-the-shelf smartphone to recognize whole-body exercises. The working principle was based on the ultrasound Doppler sensing with the device built-in hardware. Applying such a lab-environment trained model on realistic application variations causes a significant drop in performance, and thus decimate its applicability. The reason of the reduced performance can be manifold. It could be induced by the user, environment, and device variations in realistic scenarios. Such scenarios are often more complex and diverse, which can be challenging to anticipate in the initial training data. To study and overcome this issue, this paper presents a database with controlled and uncontrolled subsets of fitness exercises. We propose two concepts to utilize small adaption data to successfully improve model generalization in an uncontrolled environment, increasing the recognition accuracy by two to six folds compared to the baseline for different users.

{{</citation>}}


### (25/122) Improving Real Estate Appraisal with POI Integration and Areal Embedding (Sumin Han et al., 2023)

{{<citation>}}

Sumin Han, Youngjun Park, Sonia Sabir, Jisun An, Dongman Lee. (2023)  
**Improving Real Estate Appraisal with POI Integration and Areal Embedding**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Attention, Embedding  
[Paper Link](http://arxiv.org/abs/2311.11812v1)  

---


**ABSTRACT**  
Despite advancements in real estate appraisal methods, this study primarily focuses on two pivotal challenges. Firstly, we explore the often-underestimated impact of Points of Interest (POI) on property values, emphasizing the necessity for a comprehensive, data-driven approach to feature selection. Secondly, we integrate road-network-based Areal Embedding to enhance spatial understanding for real estate appraisal. We first propose a revised method for POI feature extraction, and discuss the impact of each POI for house price appraisal. Then we present the Areal embedding-enabled Masked Multihead Attention-based Spatial Interpolation for House Price Prediction (AMMASI) model, an improvement upon the existing ASI model, which leverages masked multi-head attention on geographic neighbor houses and similar-featured houses. Our model outperforms current baselines and also offers promising avenues for future optimization in real estate appraisal methodologies.

{{</citation>}}


### (26/122) Large Language Models and Explainable Law: a Hybrid Methodology (Marco Billi et al., 2023)

{{<citation>}}

Marco Billi, Alessandro Parenti, Giuseppe Pisano, Marco Sanchi. (2023)  
**Large Language Models and Explainable Law: a Hybrid Methodology**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.11811v1)  

---


**ABSTRACT**  
The paper advocates for LLMs to enhance the accessibility, usage and explainability of rule-based legal systems, contributing to a democratic and stakeholder-oriented view of legal technology. A methodology is developed to explore the potential use of LLMs for translating the explanations produced by rule-based systems, from high-level programming languages to natural language, allowing all users a fast, clear, and accessible interaction with such technologies. The study continues by building upon these explanations to empower laypeople with the ability to execute complex juridical tasks on their own, using a Chain of Prompts for the autonomous legal comparison of different rule-based inferences, applied to the same factual case.

{{</citation>}}


### (27/122) Responsible AI Research Needs Impact Statements Too (Alexandra Olteanu et al., 2023)

{{<citation>}}

Alexandra Olteanu, Michael Ekstrand, Carlos Castillo, Jina Suh. (2023)  
**Responsible AI Research Needs Impact Statements Too**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CY, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.11776v1)  

---


**ABSTRACT**  
All types of research, development, and policy work can have unintended, adverse consequences - work in responsible artificial intelligence (RAI), ethical AI, or ethics in AI is no exception.

{{</citation>}}


### (28/122) LSTM-CNN: An efficient diagnostic network for Parkinson's disease utilizing dynamic handwriting analysis (Xuechao Wang et al., 2023)

{{<citation>}}

Xuechao Wang, Junqing Huang, Sven Nomm, Marianna Chatzakou, Kadri Medijainen, Aaro Toomela, Michael Ruzhansky. (2023)  
**LSTM-CNN: An efficient diagnostic network for Parkinson's disease utilizing dynamic handwriting analysis**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2311.11756v1)  

---


**ABSTRACT**  
Background and objectives: Dynamic handwriting analysis, due to its non-invasive and readily accessible nature, has recently emerged as a vital adjunctive method for the early diagnosis of Parkinson's disease. In this study, we design a compact and efficient network architecture to analyse the distinctive handwriting patterns of patients' dynamic handwriting signals, thereby providing an objective identification for the Parkinson's disease diagnosis.   Methods: The proposed network is based on a hybrid deep learning approach that fully leverages the advantages of both long short-term memory (LSTM) and convolutional neural networks (CNNs). Specifically, the LSTM block is adopted to extract the time-varying features, while the CNN-based block is implemented using one-dimensional convolution for low computational cost. Moreover, the hybrid model architecture is continuously refined under ablation studies for superior performance. Finally, we evaluate the proposed method with its generalization under a five-fold cross-validation, which validates its efficiency and robustness.   Results: The proposed network demonstrates its versatility by achieving impressive classification accuracies on both our new DraWritePD dataset ($96.2\%$) and the well-established PaHaW dataset ($90.7\%$). Moreover, the network architecture also stands out for its excellent lightweight design, occupying a mere $0.084$M of parameters, with a total of only $0.59$M floating-point operations. It also exhibits near real-time CPU inference performance, with inference times ranging from $0.106$ to $0.220$s.   Conclusions: We present a series of experiments with extensive analysis, which systematically demonstrate the effectiveness and efficiency of the proposed hybrid neural network in extracting distinctive handwriting patterns for precise diagnosis of Parkinson's disease.

{{</citation>}}


### (29/122) Causal Structure Learning Supervised by Large Language Model (Taiyu Ban et al., 2023)

{{<citation>}}

Taiyu Ban, Lyuzhou Chen, Derui Lyu, Xiangyu Wang, Huanhuan Chen. (2023)  
**Causal Structure Learning Supervised by Large Language Model**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.11689v1)  

---


**ABSTRACT**  
Causal discovery from observational data is pivotal for deciphering complex relationships. Causal Structure Learning (CSL), which focuses on deriving causal Directed Acyclic Graphs (DAGs) from data, faces challenges due to vast DAG spaces and data sparsity. The integration of Large Language Models (LLMs), recognized for their causal reasoning capabilities, offers a promising direction to enhance CSL by infusing it with knowledge-based causal inferences. However, existing approaches utilizing LLMs for CSL have encountered issues, including unreliable constraints from imperfect LLM inferences and the computational intensity of full pairwise variable analyses. In response, we introduce the Iterative LLM Supervised CSL (ILS-CSL) framework. ILS-CSL innovatively integrates LLM-based causal inference with CSL in an iterative process, refining the causal DAG using feedback from LLMs. This method not only utilizes LLM resources more efficiently but also generates more robust and high-quality structural constraints compared to previous methodologies. Our comprehensive evaluation across eight real-world datasets demonstrates ILS-CSL's superior performance, setting a new standard in CSL efficacy and showcasing its potential to significantly advance the field of causal discovery. The codes are available at \url{https://github.com/tyMadara/ILS-CSL}.

{{</citation>}}


### (30/122) Peeking Inside the Schufa Blackbox: Explaining the German Housing Scoring System (Dean-Robin Kern et al., 2023)

{{<citation>}}

Dean-Robin Kern, Gunnar Stevens, Erik Dethier, Sidra Naveed, Fatemeh Alizadeh, Delong Du, Md Shajalal. (2023)  
**Peeking Inside the Schufa Blackbox: Explaining the German Housing Scoring System**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-HC, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.11655v2)  

---


**ABSTRACT**  
Explainable Artificial Intelligence is a concept aimed at making complex algorithms transparent to users through a uniform solution. Researchers have highlighted the importance of integrating domain specific contexts to develop explanations tailored to end users. In this study, we focus on the Schufa housing scoring system in Germany and investigate how users information needs and expectations for explanations vary based on their roles. Using the speculative design approach, we asked business information students to imagine user interfaces that provide housing credit score explanations from the perspectives of both tenants and landlords. Our preliminary findings suggest that although there are general needs that apply to all users, there are also conflicting needs that depend on the practical realities of their roles and how credit scores affect them. We contribute to Human centered XAI research by proposing future research directions that examine users explanatory needs considering their roles and agencies.

{{</citation>}}


### (31/122) Web News Timeline Generation with Extended Task Prompting (Sha Wang et al., 2023)

{{<citation>}}

Sha Wang, Yuchen Li, Hanhua Xiao, Lambert Deng, Yanfei Dong. (2023)  
**Web News Timeline Generation with Extended Task Prompting**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2311.11652v1)  

---


**ABSTRACT**  
The creation of news timeline is essential for a comprehensive and contextual understanding of events as they unfold over time. This approach aids in discerning patterns and trends that might be obscured when news is viewed in isolation. By organizing news in a chronological sequence, it becomes easier to track the development of stories, understand the interrelation of events, and grasp the broader implications of news items. This is particularly helpful in sectors like finance and insurance, where timely understanding of the event development-ranging from extreme weather to political upheavals and health crises-is indispensable for effective risk management. While traditional natural language processing (NLP) techniques have had some success, they often fail to capture the news with nuanced relevance that are readily apparent to domain experts, hindering broader industry integration. The advance of Large Language Models (LLMs) offers a renewed opportunity to tackle this challenge. However, direct prompting LLMs for this task is often ineffective. Our study investigates the application of an extended task prompting technique to assess past news relevance. We demonstrate that enhancing conventional prompts with additional tasks boosts their effectiveness on various news dataset, rendering news timeline generation practical for professional use. This work has been deployed as a publicly accessible browser extension which is adopted within our network.

{{</citation>}}


### (32/122) DesignGPT: Multi-Agent Collaboration in Design (Shiying Ding et al., 2023)

{{<citation>}}

Shiying Ding, Xinyi Chen, Yan Fang, Wenrui Liu, Yiwu Qiu, Chunlei Chai. (2023)  
**DesignGPT: Multi-Agent Collaboration in Design**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-HC, cs.AI  
Keywords: AI, GPT, Generative AI  
[Paper Link](http://arxiv.org/abs/2311.11591v1)  

---


**ABSTRACT**  
Generative AI faces many challenges when entering the product design workflow, such as interface usability and interaction patterns. Therefore, based on design thinking and design process, we developed the DesignGPT multi-agent collaboration framework, which uses artificial intelligence agents to simulate the roles of different positions in the design company and allows human designers to collaborate with them in natural language. Experimental results show that compared with separate AI tools, DesignGPT improves the performance of designers, highlighting the potential of applying multi-agent systems that integrate design domain knowledge to product scheme design.

{{</citation>}}


### (33/122) Which AI Technique Is Better to Classify Requirements? An Experiment with SVM, LSTM, and ChatGPT (Abdelkarim El-Hajjami et al., 2023)

{{<citation>}}

Abdelkarim El-Hajjami, Nicolas Fafin, Camille Salinesi. (2023)  
**Which AI Technique Is Better to Classify Requirements? An Experiment with SVM, LSTM, and ChatGPT**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-SE, cs.AI  
Keywords: AI, ChatGPT, GPT, LSTM, Language Model, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2311.11547v1)  

---


**ABSTRACT**  
Context and motivation: Recently, Large Language Models (LLMs) like ChatGPT have demonstrated remarkable proficiency in various Natural Language Processing (NLP) tasks. Their application in Requirements Engineering (RE), especially in requirements classification, has gained increasing interest. Question/problem: In our research, we conducted an extensive empirical evaluation of ChatGPT models including text-davinci-003, gpt-3.5-turbo, and gpt-4 in both zero-shot and few-shot settings for requirements classification. The question arises as to how these models compare to traditional classification methods, specifically Support Vector Machine (SVM) and Long Short-Term Memory (LSTM). Principal ideas/results: Based on five diverse datasets, our results show that ChatGPT consistently outperforms LSTM, and while ChatGPT is more effective than SVM in classifying functional requirements (FR), SVM is better in classifying non-functional requirements (NFR). Our results also show that contrary to our expectations, the few-shot setting does not always lead to enhanced performance; in most instances, it was found to be suboptimal. Contribution: Our findings underscore the potential of LLMs in the RE domain, suggesting that they could play a pivotal role in future software engineering processes, particularly as tools to enhance requirements classification.

{{</citation>}}


### (34/122) ADAPTER-RL: Adaptation of Any Agent using Reinforcement Learning (Yizhao Jin et al., 2023)

{{<citation>}}

Yizhao Jin, Greg Slabaugh, Simon Lucas. (2023)  
**ADAPTER-RL: Adaptation of Any Agent using Reinforcement Learning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.11537v1)  

---


**ABSTRACT**  
Deep Reinforcement Learning (DRL) agents frequently face challenges in adapting to tasks outside their training distribution, including issues with over-fitting, catastrophic forgetting and sample inefficiency. Although the application of adapters has proven effective in supervised learning contexts such as natural language processing and computer vision, their potential within the DRL domain remains largely unexplored. This paper delves into the integration of adapters in reinforcement learning, presenting an innovative adaptation strategy that demonstrates enhanced training efficiency and improvement of the base-agent, experimentally in the nanoRTS environment, a real-time strategy (RTS) game simulation. Our proposed universal approach is not only compatible with pre-trained neural networks but also with rule-based agents, offering a means to integrate human expertise.

{{</citation>}}


### (35/122) GPT in Data Science: A Practical Exploration of Model Selection (Nathalia Nascimento et al., 2023)

{{<citation>}}

Nathalia Nascimento, Cristina Tavares, Paulo Alencar, Donald Cowan. (2023)  
**GPT in Data Science: A Practical Exploration of Model Selection**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-DB, cs.AI  
Keywords: AI, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2311.11516v1)  

---


**ABSTRACT**  
There is an increasing interest in leveraging Large Language Models (LLMs) for managing structured data and enhancing data science processes. Despite the potential benefits, this integration poses significant questions regarding their reliability and decision-making methodologies. It highlights the importance of various factors in the model selection process, including the nature of the data, problem type, performance metrics, computational resources, interpretability vs accuracy, assumptions about data, and ethical considerations. Our objective is to elucidate and express the factors and assumptions guiding GPT-4's model selection recommendations. We employ a variability model to depict these factors and use toy datasets to evaluate both the model and the implementation of the identified heuristics. By contrasting these outcomes with heuristics from other platforms, our aim is to determine the effectiveness and distinctiveness of GPT-4's methodology. This research is committed to advancing our comprehension of AI decision-making processes, especially in the realm of model selection within data science. Our efforts are directed towards creating AI systems that are more transparent and comprehensible, contributing to a more responsible and efficient practice in data science.

{{</citation>}}


### (36/122) Meta Prompting for AGI Systems (Yifan Zhang, 2023)

{{<citation>}}

Yifan Zhang. (2023)  
**Meta Prompting for AGI Systems**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: AI, Few-Shot  
[Paper Link](http://arxiv.org/abs/2311.11482v1)  

---


**ABSTRACT**  
This paper presents an in-depth exploration of Meta Prompting, a novel technique that revolutionizes the way large language models (LLMs), multi-modal foundation models, and AI systems approach problem-solving and data interpretation. Meta Prompting, rooted in type theory and category theory, prioritizes the structure and syntax of information, providing a unique framework that transcends traditional content-focused methods. We delve into the formal definitions of Meta Prompting, contrasting it with Few-Shot Prompting, and highlight its applicability and superiority in various AI applications.   Key to this exploration is the expansion of Meta Prompting into the realm of complex reasoning. Here, we demonstrate how this technique adeptly breaks down intricate problems into manageable sub-problems, facilitating a step-by-step, detailed approach to problem-solving. This method proves especially advantageous in terms of token efficiency and offering a fair comparison in problem-solving scenarios, standing out against few-shot example approaches.   Furthermore, the paper breaks new ground by extending Meta Prompting into multi-modal foundation model settings. This extension addresses the integration of diverse data types, such as images, audio, and video, within the structured framework of Meta Prompting, highlighting both the challenges and the vast potential of this approach in handling complex, multi-faceted data (The code is available at https://github.com/meta-prompting/meta-prompting).

{{</citation>}}


## cs.DB (1)



### (37/122) Ontological Reasoning over Shy and Warded Datalog$+/-$ for Streaming-based Architectures (technical report) (Teodoro Baldazzi et al., 2023)

{{<citation>}}

Teodoro Baldazzi, Luigi Bellomarini, Marco Favorito, Emanuel Sallinger. (2023)  
**Ontological Reasoning over Shy and Warded Datalog$+/-$ for Streaming-based Architectures (technical report)**  

---
Primary Category: cs.DB  
Categories: cs-AI, cs-DB, cs-LO, cs.DB  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2311.12236v1)  

---


**ABSTRACT**  
Recent years witnessed a rising interest towards Datalog-based ontological reasoning systems, both in academia and industry. These systems adopt languages, often shared under the collective name of Datalog$+/-$, that extend Datalog with the essential feature of existential quantification, while introducing syntactic limitations to sustain reasoning decidability and achieve a good trade-off between expressive power and computational complexity. From an implementation perspective, modern reasoners borrow the vast experience of the database community in developing streaming-based data processing systems, such as volcano-iterator architectures, that sustain a limited memory footprint and good scalability. In this paper, we focus on two extremely promising, expressive, and tractable languages, namely, Shy and Warded Datalog$+/-$. We leverage their theoretical underpinnings to introduce novel reasoning techniques, technically, "chase variants", that are particularly fit for efficient reasoning in streaming-based architectures. We then implement them in Vadalog, our reference streaming-based engine, to efficiently solve ontological reasoning tasks over real-world settings.

{{</citation>}}


## cs.CL (22)



### (38/122) Unifying Corroborative and Contributive Attributions in Large Language Models (Theodora Worledge et al., 2023)

{{<citation>}}

Theodora Worledge, Judy Hanwen Shen, Nicole Meister, Caleb Winston, Carlos Guestrin. (2023)  
**Unifying Corroborative and Contributive Attributions in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.12233v1)  

---


**ABSTRACT**  
As businesses, products, and services spring up around large language models, the trustworthiness of these models hinges on the verifiability of their outputs. However, methods for explaining language model outputs largely fall across two distinct fields of study which both use the term "attribution" to refer to entirely separate techniques: citation generation and training data attribution. In many modern applications, such as legal document generation and medical question answering, both types of attributions are important. In this work, we argue for and present a unified framework of large language model attributions. We show how existing methods of different types of attribution fall under the unified framework. We also use the framework to discuss real-world use cases where one or both types of attributions are required. We believe that this unified framework will guide the use case driven development of systems that leverage both types of attribution, as well as the standardization of their evaluation.

{{</citation>}}


### (39/122) Leveraging Closed-Access Multilingual Embedding for Automatic Sentence Alignment in Low Resource Languages (Idris Abdulmumin et al., 2023)

{{<citation>}}

Idris Abdulmumin, Auwal Abubakar Khalid, Shamsuddeen Hassan Muhammad, Ibrahim Said Ahmad, Lukman Jibril Aliyu, Babangida Sani, Bala Mairiga Abduljalil, Sani Ahmad Hassan. (2023)  
**Leveraging Closed-Access Multilingual Embedding for Automatic Sentence Alignment in Low Resource Languages**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, BLEU, Embedding, Multilingual  
[Paper Link](http://arxiv.org/abs/2311.12179v1)  

---


**ABSTRACT**  
The importance of qualitative parallel data in machine translation has long been determined but it has always been very difficult to obtain such in sufficient quantity for the majority of world languages, mainly because of the associated cost and also the lack of accessibility to these languages. Despite the potential for obtaining parallel datasets from online articles using automatic approaches, forensic investigations have found a lot of quality-related issues such as misalignment, and wrong language codes. In this work, we present a simple but qualitative parallel sentence aligner that carefully leveraged the closed-access Cohere multilingual embedding, a solution that ranked second in the just concluded #CoHereAIHack 2023 Challenge (see https://ai6lagos.devpost.com). The proposed approach achieved $94.96$ and $54.83$ f1 scores on FLORES and MAFAND-MT, compared to $3.64$ and $0.64$ of LASER respectively. Our method also achieved an improvement of more than 5 BLEU scores over LASER, when the resulting datasets were used with MAFAND-MT dataset to train translation models. Our code and data are available for research purposes here (https://github.com/abumafrim/Cohere-Align).

{{</citation>}}


### (40/122) LQ-LoRA: Low-rank Plus Quantized Matrix Decomposition for Efficient Language Model Finetuning (Han Guo et al., 2023)

{{<citation>}}

Han Guo, Philip Greengard, Eric P. Xing, Yoon Kim. (2023)  
**LQ-LoRA: Low-rank Plus Quantized Matrix Decomposition for Efficient Language Model Finetuning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: BERT, GPT, LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2311.12023v1)  

---


**ABSTRACT**  
We propose a simple approach for memory-efficient adaptation of pretrained language models. Our approach uses an iterative algorithm to decompose each pretrained matrix into a high-precision low-rank component and a memory-efficient quantized component. During finetuning, the quantized component remains fixed and only the low-rank component is updated. We present an integer linear programming formulation of the quantization component which enables dynamic configuration of quantization parameters (e.g., bit-width, block size) for each matrix given an overall target memory budget. We further explore a data-aware version of the algorithm which uses an approximation of the Fisher information matrix to weight the reconstruction objective during matrix decomposition. Experiments on adapting RoBERTa and LLaMA-2 (7B and 70B) demonstrate that our low-rank plus quantized matrix decomposition approach (LQ-LoRA) outperforms strong QLoRA and GPTQ-LoRA baselines and moreover enables more aggressive quantization. For example, on the OpenAssistant benchmark LQ-LoRA is able to learn a 2.5-bit LLaMA-2 model that is competitive with a model finetuned with 4-bit QLoRA. When finetuned on a language modeling calibration dataset, LQ-LoRA can also be used for model compression; in this setting our 2.75-bit LLaMA-2-70B model (which has 2.85 bits on average when including the low-rank components and requires 27GB of GPU memory) is competitive with the original model in full precision.

{{</citation>}}


### (41/122) H-COAL: Human Correction of AI-Generated Labels for Biomedical Named Entity Recognition (Xiaojing Duan et al., 2023)

{{<citation>}}

Xiaojing Duan, John P. Lalor. (2023)  
**H-COAL: Human Correction of AI-Generated Labels for Biomedical Named Entity Recognition**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, NLP, Named Entity Recognition  
[Paper Link](http://arxiv.org/abs/2311.11981v1)  

---


**ABSTRACT**  
With the rapid advancement of machine learning models for NLP tasks, collecting high-fidelity labels from AI models is a realistic possibility. Firms now make AI available to customers via predictions as a service (PaaS). This includes PaaS products for healthcare. It is unclear whether these labels can be used for training a local model without expensive annotation checking by in-house experts. In this work, we propose a new framework for Human Correction of AI-Generated Labels (H-COAL). By ranking AI-generated outputs, one can selectively correct labels and approach gold standard performance (100% human labeling) with significantly less human effort. We show that correcting 5% of labels can close the AI-human performance gap by up to 64% relative improvement, and correcting 20% of labels can close the performance gap by up to 86% relative improvement.

{{</citation>}}


### (42/122) Context-aware Neural Machine Translation for English-Japanese Business Scene Dialogues (Sumire Honda et al., 2023)

{{<citation>}}

Sumire Honda, Patrick Fernandes, Chrysoula Zerva. (2023)  
**Context-aware Neural Machine Translation for English-Japanese Business Scene Dialogues**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BLEU, Dialog, Dialogue, Machine Translation  
[Paper Link](http://arxiv.org/abs/2311.11976v1)  

---


**ABSTRACT**  
Despite the remarkable advancements in machine translation, the current sentence-level paradigm faces challenges when dealing with highly-contextual languages like Japanese. In this paper, we explore how context-awareness can improve the performance of the current Neural Machine Translation (NMT) models for English-Japanese business dialogues translation, and what kind of context provides meaningful information to improve translation. As business dialogue involves complex discourse phenomena but offers scarce training resources, we adapted a pretrained mBART model, finetuning on multi-sentence dialogue data, which allows us to experiment with different contexts. We investigate the impact of larger context sizes and propose novel context tokens encoding extra-sentential information, such as speaker turn and scene type. We make use of Conditional Cross-Mutual Information (CXMI) to explore how much of the context the model uses and generalise CXMI to study the impact of the extra-sentential context. Overall, we find that models leverage both preceding sentences and extra-sentential context (with CXMI increasing with context size) and we provide a more focused analysis on honorifics translation. Regarding translation quality, increased source-side context paired with scene and speaker information improves the model performance compared to previous work and our context-agnostic baselines, measured in BLEU and COMET metrics.

{{</citation>}}


### (43/122) Automatic Analysis of Substantiation in Scientific Peer Reviews (Yanzhu Guo et al., 2023)

{{<citation>}}

Yanzhu Guo, Guokan Shang, Virgile Rennard, Michalis Vazirgiannis, Chloé Clavel. (2023)  
**Automatic Analysis of Substantiation in Scientific Peer Reviews**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, NLP  
[Paper Link](http://arxiv.org/abs/2311.11967v1)  

---


**ABSTRACT**  
With the increasing amount of problematic peer reviews in top AI conferences, the community is urgently in need of automatic quality control measures. In this paper, we restrict our attention to substantiation -- one popular quality aspect indicating whether the claims in a review are sufficiently supported by evidence -- and provide a solution automatizing this evaluation process. To achieve this goal, we first formulate the problem as claim-evidence pair extraction in scientific peer reviews, and collect SubstanReview, the first annotated dataset for this task. SubstanReview consists of 550 reviews from NLP conferences annotated by domain experts. On the basis of this dataset, we train an argument mining system to automatically analyze the level of substantiation in peer reviews. We also perform data analysis on the SubstanReview dataset to obtain meaningful insights on peer reviewing quality in NLP conferences over recent years.

{{</citation>}}


### (44/122) FinanceBench: A New Benchmark for Financial Question Answering (Pranab Islam et al., 2023)

{{<citation>}}

Pranab Islam, Anand Kannappan, Douwe Kiela, Rebecca Qian, Nino Scherrer, Bertie Vidgen. (2023)  
**FinanceBench: A New Benchmark for Financial Question Answering**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CE, cs-CL, cs.CL, stat-ML  
Keywords: Financial, GPT, GPT-4, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2311.11944v1)  

---


**ABSTRACT**  
FinanceBench is a first-of-its-kind test suite for evaluating the performance of LLMs on open book financial question answering (QA). It comprises 10,231 questions about publicly traded companies, with corresponding answers and evidence strings. The questions in FinanceBench are ecologically valid and cover a diverse set of scenarios. They are intended to be clear-cut and straightforward to answer to serve as a minimum performance standard. We test 16 state of the art model configurations (including GPT-4-Turbo, Llama2 and Claude2, with vector stores and long context prompts) on a sample of 150 cases from FinanceBench, and manually review their answers (n=2,400). The cases are available open-source. We show that existing LLMs have clear limitations for financial QA. Notably, GPT-4-Turbo used with a retrieval system incorrectly answered or refused to answer 81% of questions. While augmentation techniques such as using longer context window to feed in relevant evidence improve performance, they are unrealistic for enterprise settings due to increased latency and cannot support larger financial documents. We find that all models examined exhibit weaknesses, such as hallucinations, that limit their suitability for use by enterprises.

{{</citation>}}


### (45/122) Generating Valid and Natural Adversarial Examples with Large Language Models (Zimu Wang et al., 2023)

{{<citation>}}

Zimu Wang, Wei Wang, Qi Chen, Qiufeng Wang, Anh Nguyen. (2023)  
**Generating Valid and Natural Adversarial Examples with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2311.11861v1)  

---


**ABSTRACT**  
Deep learning-based natural language processing (NLP) models, particularly pre-trained language models (PLMs), have been revealed to be vulnerable to adversarial attacks. However, the adversarial examples generated by many mainstream word-level adversarial attack models are neither valid nor natural, leading to the loss of semantic maintenance, grammaticality, and human imperceptibility. Based on the exceptional capacity of language understanding and generation of large language models (LLMs), we propose LLM-Attack, which aims at generating both valid and natural adversarial examples with LLMs. The method consists of two stages: word importance ranking (which searches for the most vulnerable words) and word synonym replacement (which substitutes them with their synonyms obtained from LLMs). Experimental results on the Movie Review (MR), IMDB, and Yelp Review Polarity datasets against the baseline adversarial attack models illustrate the effectiveness of LLM-Attack, and it outperforms the baselines in human and GPT-4 evaluation by a significant margin. The model can generate adversarial examples that are typically valid and natural, with the preservation of semantic meaning, grammaticality, and human imperceptibility.

{{</citation>}}


### (46/122) How to Use Large Language Models for Text Coding: The Case of Fatherhood Roles in Public Policy Documents (Lorenzo Lupo et al., 2023)

{{<citation>}}

Lorenzo Lupo, Oscar Magnusson, Dirk Hovy, Elin Naurin, Lena Wängnerud. (2023)  
**How to Use Large Language Models for Text Coding: The Case of Fatherhood Roles in Public Policy Documents**  

---
Primary Category: cs.CL  
Categories: J-4; I-2, cs-CL, cs-CY, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2311.11844v1)  

---


**ABSTRACT**  
Recent advances in large language models (LLMs) like GPT-3 and GPT-4 have opened up new opportunities for text analysis in political science. They promise automation with better results and less programming. In this study, we evaluate LLMs on three original coding tasks of non-English political science texts, and we provide a detailed description of a general workflow for using LLMs for text coding in political science research. Our use case offers a practical guide for researchers looking to incorporate LLMs into their research on text analysis. We find that, when provided with detailed label definitions and coding examples, an LLM can be as good as or even better than a human annotator while being much faster (up to hundreds of times), considerably cheaper (costing up to 60% less than human coding), and much easier to scale to large amounts of text. Overall, LLMs present a viable option for most text coding projects.

{{</citation>}}


### (47/122) System 2 Attention (is something you might need too) (Jason Weston et al., 2023)

{{<citation>}}

Jason Weston, Sainbayar Sukhbaatar. (2023)  
**System 2 Attention (is something you might need too)**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Attention, Language Model, QA, Transformer  
[Paper Link](http://arxiv.org/abs/2311.11829v1)  

---


**ABSTRACT**  
Soft attention in Transformer-based Large Language Models (LLMs) is susceptible to incorporating irrelevant information from the context into its latent representations, which adversely affects next token generations. To help rectify these issues, we introduce System 2 Attention (S2A), which leverages the ability of LLMs to reason in natural language and follow instructions in order to decide what to attend to. S2A regenerates the input context to only include the relevant portions, before attending to the regenerated context to elicit the final response. In experiments, S2A outperforms standard attention-based LLMs on three tasks containing opinion or irrelevant information, QA, math word problems and longform generation, where S2A increases factuality and objectivity, and decreases sycophancy.

{{</citation>}}


### (48/122) Efficient Grammatical Error Correction Via Multi-Task Training and Optimized Training Schedule (Andrey Bout et al., 2023)

{{<citation>}}

Andrey Bout, Alexander Podolskiy, Sergey Nikolenko, Irina Piontkovskaya. (2023)  
**Efficient Grammatical Error Correction Via Multi-Task Training and Optimized Training Schedule**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: T5  
[Paper Link](http://arxiv.org/abs/2311.11813v1)  

---


**ABSTRACT**  
Progress in neural grammatical error correction (GEC) is hindered by the lack of annotated training data. Sufficient amounts of high-quality manually annotated data are not available, so recent research has relied on generating synthetic data, pretraining on it, and then fine-tuning on real datasets; performance gains have been achieved either by ensembling or by using huge pretrained models such as XXL-T5 as the backbone. In this work, we explore an orthogonal direction: how to use available data more efficiently. First, we propose auxiliary tasks that exploit the alignment between the original and corrected sentences, such as predicting a sequence of corrections. We formulate each task as a sequence-to-sequence problem and perform multi-task training. Second, we discover that the order of datasets used for training and even individual instances within a dataset may have important effects on the final performance, so we set out to find the best training schedule. Together, these two ideas lead to significant improvements, producing results that improve state of the art with much smaller models; in particular, we outperform the best models based on T5-XXL (11B parameters) with a BART-based model (400M parameters).

{{</citation>}}


### (49/122) Igniting Language Intelligence: The Hitchhiker's Guide From Chain-of-Thought Reasoning to Language Agents (Zhuosheng Zhang et al., 2023)

{{<citation>}}

Zhuosheng Zhang, Yao Yao, Aston Zhang, Xiangru Tang, Xinbei Ma, Zhiwei He, Yiming Wang, Mark Gerstein, Rui Wang, Gongshen Liu, Hai Zhao. (2023)  
**Igniting Language Intelligence: The Hitchhiker's Guide From Chain-of-Thought Reasoning to Language Agents**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CV, cs-HC, cs-MA, cs.CL  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2311.11797v1)  

---


**ABSTRACT**  
Large language models (LLMs) have dramatically enhanced the field of language intelligence, as demonstrably evidenced by their formidable empirical performance across a spectrum of complex reasoning tasks. Additionally, theoretical proofs have illuminated their emergent reasoning capabilities, providing a compelling showcase of their advanced cognitive abilities in linguistic contexts. Critical to their remarkable efficacy in handling complex reasoning tasks, LLMs leverage the intriguing chain-of-thought (CoT) reasoning techniques, obliging them to formulate intermediate steps en route to deriving an answer. The CoT reasoning approach has not only exhibited proficiency in amplifying reasoning performance but also in enhancing interpretability, controllability, and flexibility. In light of these merits, recent research endeavors have extended CoT reasoning methodologies to nurture the development of autonomous language agents, which adeptly adhere to language instructions and execute actions within varied environments. This survey paper orchestrates a thorough discourse, penetrating vital research dimensions, encompassing: (i) the foundational mechanics of CoT techniques, with a focus on elucidating the circumstances and justification behind its efficacy; (ii) the paradigm shift in CoT; and (iii) the burgeoning of language agents fortified by CoT approaches. Prospective research avenues envelop explorations into generalization, efficiency, customization, scaling, and safety. This paper caters to a wide audience, including beginners seeking comprehensive knowledge of CoT reasoning and language agents, as well as experienced researchers interested in foundational mechanics and engaging in cutting-edge discussions on these topics. A repository for the related papers is available at https://github.com/Zoeyyao27/CoT-Igniting-Agent.

{{</citation>}}


### (50/122) Sparse Low-rank Adaptation of Pre-trained Language Models (Ning Ding et al., 2023)

{{<citation>}}

Ning Ding, Xingtai Lv, Qiaosen Wang, Yulin Chen, Bowen Zhou, Zhiyuan Liu, Maosong Sun. (2023)  
**Sparse Low-rank Adaptation of Pre-trained Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.11696v1)  

---


**ABSTRACT**  
Fine-tuning pre-trained large language models in a parameter-efficient manner is widely studied for its effectiveness and efficiency. The popular method of low-rank adaptation (LoRA) offers a notable approach, hypothesizing that the adaptation process is intrinsically low-dimensional. Although LoRA has demonstrated commendable performance, it is implemented with a fixed and unalterable intrinsic rank that might not always be the ideal choice. Recognizing the need for more flexible adaptation, we extend the methodology of LoRA to an innovative approach we call sparse low-rank adaptation (SoRA) that enables dynamic adjustments to the intrinsic rank during the adaptation process. We achieve this through the incorporation of a gate unit optimized with proximal gradient method in the training stage, controlling the cardinality of rank under the sparsity of the gate. In the subsequent inference stage, we eliminate the parameter blocks corresponding to the zeroed-out ranks, to reduce each SoRA module back to a concise yet rank-optimal LoRA. Our approach strengthens the representation power of LoRA by initializing it with a higher rank, while efficiently taming a temporarily increased number of parameters via updating in a sparse way. We further introduce a sparsifying scheduler for SoRA, aiming to examine the impact of the number of non-zero parameters on the model's memorization and generalization. Our experimental results demonstrate that SoRA can outperform other baselines even with 70% retained parameters and 70% training time.

{{</citation>}}


### (51/122) Taiyi: A Bilingual Fine-Tuned Large Language Model for Diverse Biomedical Tasks (Ling Luo et al., 2023)

{{<citation>}}

Ling Luo, Jinzhong Ning, Yingwen Zhao, Zhijun Wang, Zeyuan Ding, Peng Chen, Weiru Fu, Qinyu Han, Guangtao Xu, Yunzhi Qiu, Dinghao Pan, Jiru Li, Hao Li, Wenduo Feng, Senbo Tu, Yuqi Liu, Zhihao Yang, Jian Wang, Yuanyuan Sun, Hongfei Lin. (2023)  
**Taiyi: A Bilingual Fine-Tuned Large Language Model for Diverse Biomedical Tasks**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2311.11608v1)  

---


**ABSTRACT**  
Recent advancements in large language models (LLMs) have shown promising results across a variety of natural language processing (NLP) tasks. The application of LLMs to specific domains, such as biomedicine, has achieved increased attention. However, most biomedical LLMs focus on enhancing performance in monolingual biomedical question answering and conversation tasks. To further investigate the effectiveness of the LLMs on diverse biomedical NLP tasks in different languages, we present Taiyi, a bilingual (English and Chinese) fine-tuned LLM for diverse biomedical tasks. In this work, we first curated a comprehensive collection of 140 existing biomedical text mining datasets across over 10 task types. Subsequently, a two-stage strategy is proposed for supervised fine-tuning to optimize the model performance across varied tasks. Experimental results on 13 test sets covering named entity recognition, relation extraction, text classification, question answering tasks demonstrate Taiyi achieves superior performance compared to general LLMs. The case study involving additional biomedical NLP tasks further shows Taiyi's considerable potential for bilingual biomedical multi-tasking. The source code, datasets, and model for Taiyi are freely available at https://github.com/DUTIR-BioNLP/Taiyi-LLM.

{{</citation>}}


### (52/122) Addressing the Length Bias Problem in Document-Level Neural Machine Translation (Zhuocheng Zhang et al., 2023)

{{<citation>}}

Zhuocheng Zhang, Shuhao Gu, Min Zhang, Yang Feng. (2023)  
**Addressing the Length Bias Problem in Document-Level Neural Machine Translation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Bias, Machine Translation  
[Paper Link](http://arxiv.org/abs/2311.11601v1)  

---


**ABSTRACT**  
Document-level neural machine translation (DNMT) has shown promising results by incorporating more context information. However, this approach also introduces a length bias problem, whereby DNMT suffers from significant translation quality degradation when decoding documents that are much shorter or longer than the maximum sequence length during training. %i.e., the length bias problem. To solve the length bias problem, we propose to improve the DNMT model in training method, attention mechanism, and decoding strategy. Firstly, we propose to sample the training data dynamically to ensure a more uniform distribution across different sequence lengths. Then, we introduce a length-normalized attention mechanism to aid the model in focusing on target information, mitigating the issue of attention divergence when processing longer sequences. Lastly, we propose a sliding window strategy during decoding that integrates as much context information as possible without exceeding the maximum sequence length. The experimental results indicate that our method can bring significant improvements on several open datasets, and further analysis shows that our method can significantly alleviate the length bias problem.

{{</citation>}}


### (53/122) Filling the Image Information Gap for VQA: Prompting Large Language Models to Proactively Ask Questions (Ziyue Wang et al., 2023)

{{<citation>}}

Ziyue Wang, Chi Chen, Peng Li, Yang Liu. (2023)  
**Filling the Image Information Gap for VQA: Prompting Large Language Models to Proactively Ask Questions**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, QA  
[Paper Link](http://arxiv.org/abs/2311.11598v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) demonstrate impressive reasoning ability and the maintenance of world knowledge not only in natural language tasks, but also in some vision-language tasks such as open-domain knowledge-based visual question answering (OK-VQA). As images are invisible to LLMs, researchers convert images to text to engage LLMs into the visual question reasoning procedure. This leads to discrepancies between images and their textual representations presented to LLMs, which consequently impedes final reasoning performance. To fill the information gap and better leverage the reasoning capability, we design a framework that enables LLMs to proactively ask relevant questions to unveil more details in the image, along with filters for refining the generated information. We validate our idea on OK-VQA and A-OKVQA. Our method continuously boosts the performance of baselines methods by an average gain of 2.15% on OK-VQA, and achieves consistent improvements across different LLMs.

{{</citation>}}


### (54/122) How well ChatGPT understand Malaysian English? An Evaluation on Named Entity Recognition and Relation Extraction (Mohan Raj Chanthran et al., 2023)

{{<citation>}}

Mohan Raj Chanthran, Lay-Ki Soon, Huey Fang Ong, Bhawani Selvaretnam. (2023)  
**How well ChatGPT understand Malaysian English? An Evaluation on Named Entity Recognition and Relation Extraction**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Named Entity Recognition, Relation Extraction  
[Paper Link](http://arxiv.org/abs/2311.11583v1)  

---


**ABSTRACT**  
Recently, ChatGPT has attracted a lot of interest from both researchers and the general public. While the performance of ChatGPT in named entity recognition and relation extraction from Standard English texts is satisfactory, it remains to be seen if it can perform similarly for Malaysian English. Malaysian English is unique as it exhibits morphosyntactic and semantical adaptation from local contexts. In this study, we assess ChatGPT's capability in extracting entities and relations from the Malaysian English News (MEN) dataset. We propose a three-step methodology referred to as \textbf{\textit{educate-predict-evaluate}}. The performance of ChatGPT is assessed using F1-Score across 18 unique prompt settings, which were carefully engineered for a comprehensive review. From our evaluation, we found that ChatGPT does not perform well in extracting entities from Malaysian English news articles, with the highest F1-Score of 0.497. Further analysis shows that the morphosyntactic adaptation in Malaysian English caused the limitation. However, interestingly, this morphosyntactic adaptation does not impact the performance of ChatGPT for relation extraction.

{{</citation>}}


### (55/122) KBioXLM: A Knowledge-anchored Biomedical Multilingual Pretrained Language Model (Lei Geng et al., 2023)

{{<citation>}}

Lei Geng, Xu Yan, Ziqiang Cao, Juntao Li, Wenjie Li, Sujian Li, Xinjie Zhou, Yang Yang, Jun Zhang. (2023)  
**KBioXLM: A Knowledge-anchored Biomedical Multilingual Pretrained Language Model**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Multilingual  
[Paper Link](http://arxiv.org/abs/2311.11564v1)  

---


**ABSTRACT**  
Most biomedical pretrained language models are monolingual and cannot handle the growing cross-lingual requirements. The scarcity of non-English domain corpora, not to mention parallel data, poses a significant hurdle in training multilingual biomedical models. Since knowledge forms the core of domain-specific corpora and can be translated into various languages accurately, we propose a model called KBioXLM, which transforms the multilingual pretrained model XLM-R into the biomedical domain using a knowledge-anchored approach. We achieve a biomedical multilingual corpus by incorporating three granularity knowledge alignments (entity, fact, and passage levels) into monolingual corpora. Then we design three corresponding training tasks (entity masking, relation masking, and passage relation prediction) and continue training on top of the XLM-R model to enhance its domain cross-lingual ability. To validate the effectiveness of our model, we translate the English benchmarks of multiple tasks into Chinese. Experimental results demonstrate that our model significantly outperforms monolingual and multilingual pretrained models in cross-lingual zero-shot and few-shot scenarios, achieving improvements of up to 10+ points. Our code is publicly available at https://github.com/ngwlh-gl/KBioXLM.

{{</citation>}}


### (56/122) Exploring Prompting Large Language Models as Explainable Metrics (Ghazaleh Mahmoudi, 2023)

{{<citation>}}

Ghazaleh Mahmoudi. (2023)  
**Exploring Prompting Large Language Models as Explainable Metrics**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2311.11552v1)  

---


**ABSTRACT**  
This paper describes the IUST NLP Lab submission to the Prompting Large Language Models as Explainable Metrics Shared Task at the Eval4NLP 2023 Workshop on Evaluation & Comparison of NLP Systems. We have proposed a zero-shot prompt-based strategy for explainable evaluation of the summarization task using Large Language Models (LLMs). The conducted experiments demonstrate the promising potential of LLMs as evaluation metrics in Natural Language Processing (NLP), particularly in the field of summarization. Both few-shot and zero-shot approaches are employed in these experiments. The performance of our best provided prompts achieved a Kendall correlation of 0.477 with human evaluations in the text summarization task on the test data. Code and results are publicly available on GitHub.

{{</citation>}}


### (57/122) Adapt in Contexts: Retrieval-Augmented Domain Adaptation via In-Context Learning (Quanyu Long et al., 2023)

{{<citation>}}

Quanyu Long, Wenya Wang, Sinno Jialin Pan. (2023)  
**Adapt in Contexts: Retrieval-Augmented Domain Adaptation via In-Context Learning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NER, Named Entity Recognition, Sentiment Analysis  
[Paper Link](http://arxiv.org/abs/2311.11551v1)  

---


**ABSTRACT**  
Large language models (LLMs) have showcased their capability with few-shot inference known as in-context learning. However, in-domain demonstrations are not always readily available in real scenarios, leading to cross-domain in-context learning. Besides, LLMs are still facing challenges in long-tail knowledge in unseen and unfamiliar domains. The above limitations demonstrate the necessity of Unsupervised Domain Adaptation (UDA). In this paper, we study the UDA problem under an in-context learning setting to adapt language models from the source domain to the target domain without any target labels. The core idea is to retrieve a subset of cross-domain elements that are the most similar to the query, and elicit language model to adapt in an in-context manner by learning both target domain distribution and the discriminative task signal simultaneously with the augmented cross-domain in-context examples. We devise different prompting and training strategies, accounting for different LM architectures to learn the target distribution via language modeling. With extensive experiments on Sentiment Analysis (SA) and Named Entity Recognition (NER) tasks, we thoroughly study the effectiveness of ICL for domain transfer and demonstrate significant improvements over baseline models.

{{</citation>}}


### (58/122) Multi-teacher Distillation for Multilingual Spelling Correction (Jingfen Zhang et al., 2023)

{{<citation>}}

Jingfen Zhang, Xuan Guo, Sravan Bodapati, Christopher Potts. (2023)  
**Multi-teacher Distillation for Multilingual Spelling Correction**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Multilingual, NLP  
[Paper Link](http://arxiv.org/abs/2311.11518v1)  

---


**ABSTRACT**  
Accurate spelling correction is a critical step in modern search interfaces, especially in an era of mobile devices and speech-to-text interfaces. For services that are deployed around the world, this poses a significant challenge for multilingual NLP: spelling errors need to be caught and corrected in all languages, and even in queries that use multiple languages. In this paper, we tackle this challenge using multi-teacher distillation. On our approach, a monolingual teacher model is trained for each language/locale, and these individual models are distilled into a single multilingual student model intended to serve all languages/locales. In experiments using open-source data as well as user data from a worldwide search service, we show that this leads to highly effective spelling correction models that can meet the tight latency requirements of deployed services.

{{</citation>}}


### (59/122) Token-Level Adversarial Prompt Detection Based on Perplexity Measures and Contextual Information (Zhengmian Hu et al., 2023)

{{<citation>}}

Zhengmian Hu, Gang Wu, Saayan Mitra, Ruiyi Zhang, Tong Sun, Heng Huang, Vishy Swaminathan. (2023)  
**Token-Level Adversarial Prompt Detection Based on Perplexity Measures and Contextual Information**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Language Model, Perplexity  
[Paper Link](http://arxiv.org/abs/2311.11509v1)  

---


**ABSTRACT**  
In recent years, Large Language Models (LLM) have emerged as pivotal tools in various applications. However, these models are susceptible to adversarial prompt attacks, where attackers can carefully curate input strings that lead to undesirable outputs. The inherent vulnerability of LLMs stems from their input-output mechanisms, especially when presented with intensely out-of-distribution (OOD) inputs. This paper proposes a token-level detection method to identify adversarial prompts, leveraging the LLM's capability to predict the next token's probability. We measure the degree of the model's perplexity and incorporate neighboring token information to encourage the detection of contiguous adversarial prompt sequences. As a result, we propose two methods: one that identifies each token as either being part of an adversarial prompt or not, and another that estimates the probability of each token being part of an adversarial prompt.

{{</citation>}}


## cs.CR (4)



### (60/122) DefensiveDR: Defending against Adversarial Patches using Dimensionality Reduction (Nandish Chattopadhyay et al., 2023)

{{<citation>}}

Nandish Chattopadhyay, Amira Guesmi, Muhammad Abdullah Hanif, Bassem Ouni, Muhammad Shafique. (2023)  
**DefensiveDR: Defending against Adversarial Patches using Dimensionality Reduction**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Embedding, Google  
[Paper Link](http://arxiv.org/abs/2311.12211v1)  

---


**ABSTRACT**  
Adversarial patch-based attacks have shown to be a major deterrent towards the reliable use of machine learning models. These attacks involve the strategic modification of localized patches or specific image areas to deceive trained machine learning models. In this paper, we propose \textit{DefensiveDR}, a practical mechanism using a dimensionality reduction technique to thwart such patch-based attacks. Our method involves projecting the sample images onto a lower-dimensional space while retaining essential information or variability for effective machine learning tasks. We perform this using two techniques, Singular Value Decomposition and t-Distributed Stochastic Neighbor Embedding. We experimentally tune the variability to be preserved for optimal performance as a hyper-parameter. This dimension reduction substantially mitigates adversarial perturbations, thereby enhancing the robustness of the given machine learning model. Our defense is model-agnostic and operates without assumptions about access to model decisions or model architectures, making it effective in both black-box and white-box settings. Furthermore, it maintains accuracy across various models and remains robust against several unseen patch-based attacks. The proposed defensive approach improves the accuracy from 38.8\% (without defense) to 66.2\% (with defense) when performing LaVAN and GoogleAp attacks, which supersedes that of the prominent state-of-the-art like LGS (53.86\%) and Jujutsu (60\%).

{{</citation>}}


### (61/122) Beyond Boundaries: A Comprehensive Survey of Transferable Attacks on AI Systems (Guangjing Wang et al., 2023)

{{<citation>}}

Guangjing Wang, Ce Zhou, Yuanda Wang, Bocheng Chen, Hanqing Guo, Qiben Yan. (2023)  
**Beyond Boundaries: A Comprehensive Survey of Transferable Attacks on AI Systems**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CL, cs-CR, cs-CV, cs.CR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.11796v1)  

---


**ABSTRACT**  
Artificial Intelligence (AI) systems such as autonomous vehicles, facial recognition, and speech recognition systems are increasingly integrated into our daily lives. However, despite their utility, these AI systems are vulnerable to a wide range of attacks such as adversarial, backdoor, data poisoning, membership inference, model inversion, and model stealing attacks. In particular, numerous attacks are designed to target a particular model or system, yet their effects can spread to additional targets, referred to as transferable attacks. Although considerable efforts have been directed toward developing transferable attacks, a holistic understanding of the advancements in transferable attacks remains elusive. In this paper, we comprehensively explore learning-based attacks from the perspective of transferability, particularly within the context of cyber-physical security. We delve into different domains -- the image, text, graph, audio, and video domains -- to highlight the ubiquitous and pervasive nature of transferable attacks. This paper categorizes and reviews the architecture of existing attacks from various viewpoints: data, process, model, and system. We further examine the implications of transferable attacks in practical scenarios such as autonomous driving, speech recognition, and large language models (LLMs). Additionally, we outline the potential research directions to encourage efforts in exploring the landscape of transferable attacks. This survey offers a holistic understanding of the prevailing transferable attacks and their impacts across different domains.

{{</citation>}}


### (62/122) Trust-based Approaches Towards Enhancing IoT Security: A Systematic Literature Review (Oghenetejiri Okporokpo et al., 2023)

{{<citation>}}

Oghenetejiri Okporokpo, Funminiyi Olajide, Nemitari Ajienka, Xiaoqi Ma. (2023)  
**Trust-based Approaches Towards Enhancing IoT Security: A Systematic Literature Review**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-NI, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2311.11705v1)  

---


**ABSTRACT**  
The continuous rise in the adoption of emerging technologies such as Internet of Things (IoT) by businesses has brought unprecedented opportunities for innovation and growth. However, due to the distinct characteristics of these emerging IoT technologies like real-time data processing, Self-configuration, interoperability, and scalability, they have also introduced some unique cybersecurity challenges, such as malware attacks, advanced persistent threats (APTs), DoS /DDoS (Denial of Service & Distributed Denial of Service attacks) and insider threats. As a result of these challenges, there is an increased need for improved cybersecurity approaches and efficient management solutions to ensure the privacy and security of communication within IoT networks. One proposed security approach is the utilization of trust-based systems and is the focus of this study. This research paper presents a systematic literature review on the Trust-based cybersecurity security approaches for IoT. A total of 23 articles were identified that satisfy the review criteria. We highlighted the common trust-based mitigation techniques in existence for dealing with these threats and grouped them into three major categories, namely: Observation-Based, Knowledge-Based & Cluster-Based systems. Finally, several open issues were highlighted, and future research directions presented.

{{</citation>}}


### (63/122) Assessing Prompt Injection Risks in 200+ Custom GPTs (Jiahao Yu et al., 2023)

{{<citation>}}

Jiahao Yu, Yuhang Wu, Dong Shu, Mingyu Jin, Xinyu Xing. (2023)  
**Assessing Prompt Injection Risks in 200+ Custom GPTs**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2311.11538v1)  

---


**ABSTRACT**  
In the rapidly evolving landscape of artificial intelligence, ChatGPT has been widely used in various applications. The new feature: customization of ChatGPT models by users to cater to specific needs has opened new frontiers in AI utility. However, this study reveals a significant security vulnerability inherent in these user-customized GPTs: prompt injection attacks. Through comprehensive testing of over 200 user-designed GPT models via adversarial prompts, we demonstrate that these systems are susceptible to prompt injections. Through prompt injection, an adversary can not only extract the customized system prompts but also access the uploaded files. This paper provides a first-hand analysis of the prompt injection, alongside the evaluation of the possible mitigation of such attacks. Our findings underscore the urgent need for robust security frameworks in the design and deployment of customizable GPT models. The intent of this paper is to raise awareness and prompt action in the AI community, ensuring that the benefits of GPT customization do not come at the cost of compromised security and privacy.

{{</citation>}}


## cs.CV (36)



### (64/122) Disentangling Structure and Appearance in ViT Feature Space (Narek Tumanyan et al., 2023)

{{<citation>}}

Narek Tumanyan, Omer Bar-Tal, Shir Amir, Shai Bagon, Tali Dekel. (2023)  
**Disentangling Structure and Appearance in ViT Feature Space**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.12193v1)  

---


**ABSTRACT**  
We present a method for semantically transferring the visual appearance of one natural image to another. Specifically, our goal is to generate an image in which objects in a source structure image are "painted" with the visual appearance of their semantically related objects in a target appearance image. To integrate semantic information into our framework, our key idea is to leverage a pre-trained and fixed Vision Transformer (ViT) model. Specifically, we derive novel disentangled representations of structure and appearance extracted from deep ViT features. We then establish an objective function that splices the desired structure and appearance representations, interweaving them together in the space of ViT features. Based on our objective function, we propose two frameworks of semantic appearance transfer -- "Splice", which works by training a generator on a single and arbitrary pair of structure-appearance images, and "SpliceNet", a feed-forward real-time appearance transfer model trained on a dataset of images from a specific domain. Our frameworks do not involve adversarial training, nor do they require any additional input information such as semantic segmentation or correspondences. We demonstrate high-resolution results on a variety of in-the-wild image pairs, under significant variations in the number of objects, pose, and appearance. Code and supplementary material are available in our project page: splice-vit.github.io.

{{</citation>}}


### (65/122) ChemScraper: Graphics Extraction, Molecular Diagram Parsing, and Annotated Data Generation for PDF Images (Ayush Kumar Shah et al., 2023)

{{<citation>}}

Ayush Kumar Shah, Bryan Manrique Amador, Abhisek Dey, Ming Creekmore, Blake Ocampo, Scott Denmark, Richard Zanibbi. (2023)  
**ChemScraper: Graphics Extraction, Molecular Diagram Parsing, and Annotated Data Generation for PDF Images**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: OCR  
[Paper Link](http://arxiv.org/abs/2311.12161v2)  

---


**ABSTRACT**  
Existing visual parsers for molecule diagrams translate pixel-based raster images such as PNGs to chemical structure representations (e.g., SMILES). However, PDFs created by word processors including LaTeX and Word provide explicit locations and shapes for characters, lines, and polygons. We extract symbols from born-digital PDF molecule images and then apply simple graph transformations to capture both visual and chemical structure in editable ChemDraw files (CDXML). Our fast ( PDF $\rightarrow$ visual graph $\rightarrow$ chemical graph ) pipeline does not require GPUs, Optical Character Recognition (OCR) or vectorization. We evaluate on standard benchmarks using SMILES strings, along with a novel evaluation that provides graph-based metrics and error compilation using LgEval. The geometric information in born-digital PDFs produces a highly accurate parser, motivating generating training data for visual parsers that recognize from raster images, with extracted graphics, visual structure, and chemical structure as annotations. To do this we render SMILES strings in Indigo, parse molecule structure, and then validate recognized structure to select correct files.

{{</citation>}}


### (66/122) Conditional Modeling Based Automatic Video Summarization (Jia-Hong Huang et al., 2023)

{{<citation>}}

Jia-Hong Huang, Chao-Han Huck Yang, Pin-Yu Chen, Min-Hung Chen, Marcel Worring. (2023)  
**Conditional Modeling Based Automatic Video Summarization**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-IR, cs-LG, cs-MM, cs.CV  
Keywords: Summarization  
[Paper Link](http://arxiv.org/abs/2311.12159v1)  

---


**ABSTRACT**  
The aim of video summarization is to shorten videos automatically while retaining the key information necessary to convey the overall story. Video summarization methods mainly rely on visual factors, such as visual consecutiveness and diversity, which may not be sufficient to fully understand the content of the video. There are other non-visual factors, such as interestingness, representativeness, and storyline consistency that should also be considered for generating high-quality video summaries. Current methods do not adequately take into account these non-visual factors, resulting in suboptimal performance. In this work, a new approach to video summarization is proposed based on insights gained from how humans create ground truth video summaries. The method utilizes a conditional modeling perspective and introduces multiple meaningful random variables and joint distributions to characterize the key components of video summarization. Helper distributions are employed to improve the training of the model. A conditional attention module is designed to mitigate potential performance degradation in the presence of multi-modal input. The proposed video summarization method incorporates the above innovative design choices that aim to narrow the gap between human-generated and machine-generated video summaries. Extensive experiments show that the proposed approach outperforms existing methods and achieves state-of-the-art performance on commonly used video summarization datasets.

{{</citation>}}


### (67/122) Applications of Large Scale Foundation Models for Autonomous Driving (Yu Huang et al., 2023)

{{<citation>}}

Yu Huang, Yue Chen, Zhu Li. (2023)  
**Applications of Large Scale Foundation Models for Autonomous Driving**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, GPT, NLP, PaLM  
[Paper Link](http://arxiv.org/abs/2311.12144v2)  

---


**ABSTRACT**  
Since DARPA Grand Challenges (rural) in 2004/05 and Urban Challenges in 2007, autonomous driving has been the most active field of AI applications. Recently powered by large language models (LLMs), chat systems, such as chatGPT and PaLM, emerge and rapidly become a promising direction to achieve artificial general intelligence (AGI) in natural language processing (NLP). There comes a natural thinking that we could employ these abilities to reformulate autonomous driving. By combining LLM with foundation models, it is possible to utilize the human knowledge, commonsense and reasoning to rebuild autonomous driving systems from the current long-tailed AI dilemma. In this paper, we investigate the techniques of foundation models and LLMs applied for autonomous driving, categorized as simulation, world model, data annotation and planning or E2E solutions etc.

{{</citation>}}


### (68/122) Fingerspelling PoseNet: Enhancing Fingerspelling Translation with Pose-Based Transformer Models (Pooya Fayyazsanavi et al., 2023)

{{<citation>}}

Pooya Fayyazsanavi, Negar Nejatishahidin, Jana Kosecka. (2023)  
**Fingerspelling PoseNet: Enhancing Fingerspelling Translation with Pose-Based Transformer Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-HC, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.12128v1)  

---


**ABSTRACT**  
We address the task of American Sign Language fingerspelling translation using videos in the wild. We exploit advances in more accurate hand pose estimation and propose a novel architecture that leverages the transformer based encoder-decoder model enabling seamless contextual word translation. The translation model is augmented by a novel loss term that accurately predicts the length of the finger-spelled word, benefiting both training and inference. We also propose a novel two-stage inference approach that re-ranks the hypotheses using the language model capabilities of the decoder. Through extensive experiments, we demonstrate that our proposed method outperforms the state-of-the-art models on ChicagoFSWild and ChicagoFSWild+ achieving more than 10% relative improvement in performance. Our findings highlight the effectiveness of our approach and its potential to advance fingerspelling recognition in sign language translation. Code is also available at https://github.com/pooyafayyaz/Fingerspelling-PoseNet.

{{</citation>}}


### (69/122) Hourglass Tokenizer for Efficient Transformer-Based 3D Human Pose Estimation (Wenhao Li et al., 2023)

{{<citation>}}

Wenhao Li, Mengyuan Liu, Hong Liu, Pichao Wang, Jialun Cai, Nicu Sebe. (2023)  
**Hourglass Tokenizer for Efficient Transformer-Based 3D Human Pose Estimation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: BERT, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.12028v1)  

---


**ABSTRACT**  
Transformers have been successfully applied in the field of video-based 3D human pose estimation. However, the high computational costs of these video pose transformers (VPTs) make them impractical on resource-constrained devices. In this paper, we present a plug-and-play pruning-and-recovering framework, called Hourglass Tokenizer (HoT), for efficient transformer-based 3D human pose estimation from videos. Our HoT begins with pruning pose tokens of redundant frames and ends with recovering full-length tokens, resulting in a few pose tokens in the intermediate transformer blocks and thus improving the model efficiency. To effectively achieve this, we propose a token pruning cluster (TPC) that dynamically selects a few representative tokens with high semantic diversity while eliminating the redundancy of video frames. In addition, we develop a token recovering attention (TRA) to restore the detailed spatio-temporal information based on the selected tokens, thereby expanding the network output to the original full-length temporal resolution for fast inference. Extensive experiments on two benchmark datasets (i.e., Human3.6M and MPI-INF-3DHP) demonstrate that our method can achieve both high efficiency and estimation accuracy compared to the original VPT models. For instance, applying to MotionBERT and MixSTE on Human3.6M, our HoT can save nearly 50% FLOPs without sacrificing accuracy and nearly 40% FLOPs with only 0.2% accuracy drop, respectively. Our source code will be open-sourced.

{{</citation>}}


### (70/122) DAS: A Deformable Attention to Capture Salient Information in CNNs (Farzad Salajegheh et al., 2023)

{{<citation>}}

Farzad Salajegheh, Nader Asadi, Soroush Saryazdi, Sudhir Mudur. (2023)  
**DAS: A Deformable Attention to Capture Salient Information in CNNs**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Image Classification, ImageNet, Object Detection  
[Paper Link](http://arxiv.org/abs/2311.12091v1)  

---


**ABSTRACT**  
Convolutional Neural Networks (CNNs) excel in local spatial pattern recognition. For many vision tasks, such as object recognition and segmentation, salient information is also present outside CNN's kernel boundaries. However, CNNs struggle in capturing such relevant information due to their confined receptive fields. Self-attention can improve a model's access to global information but increases computational overhead. We present a fast and simple fully convolutional method called DAS that helps focus attention on relevant information. It uses deformable convolutions for the location of pertinent image regions and separable convolutions for efficiency. DAS plugs into existing CNNs and propagates relevant information using a gating mechanism. Compared to the O(n^2) computational complexity of transformer-style attention, DAS is O(n). Our claim is that DAS's ability to pay increased attention to relevant features results in performance improvements when added to popular CNNs for Image Classification and Object Detection. For example, DAS yields an improvement on Stanford Dogs (4.47%), ImageNet (1.91%), and COCO AP (3.3%) with base ResNet50 backbone. This outperforms other CNN attention mechanisms while using similar or less FLOPs. Our code will be publicly available.

{{</citation>}}


### (71/122) Exploring Lip Segmentation Techniques in Computer Vision: A Comparative Analysis (Pietro B. S. Masur et al., 2023)

{{<citation>}}

Pietro B. S. Masur, Francisco Braulio Oliveira, Lucas Moreira Medino, Emanuel Huber, Milene Haraguchi Padilha, Cassio de Alcantara, Renata Sellaro. (2023)  
**Exploring Lip Segmentation Techniques in Computer Vision: A Comparative Analysis**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Computer Vision  
[Paper Link](http://arxiv.org/abs/2311.11992v1)  

---


**ABSTRACT**  
Lip segmentation is crucial in computer vision, especially for lip reading. Despite extensive face segmentation research, lip segmentation has received limited attention. The aim of this study is to compare state-of-the-art lip segmentation models using a standardized setting and a publicly available dataset. Five techniques, namely EHANet, Mask2Former, BiSeNet V2, PIDNet, and STDC1, are qualitatively selected based on their reported performance, inference time, code availability, recency, and popularity. The CelebAMask-HQ dataset, comprising manually annotated face images, is used to fairly assess the lip segmentation performance of the selected models. Inference experiments are conducted on a Raspberry Pi4 to emulate limited computational resources. The results show that Mask2Former and EHANet have the best performances in terms of mIoU score. BiSeNet V2 demonstrate competitive performance, while PIDNet excels in recall but has lower precision. Most models present inference time ranging from 1000 to around 3000 milliseconds on a Raspberry Pi4, with PIDNet having the lowest mean inference time. This study provides a comprehensive evaluation of lip segmentation models, highlighting their performance and inference times. The findings contribute to the development of lightweight techniques and establish benchmarks for future advances in lip segmentation, especially in IoT and edge computing scenarios.

{{</citation>}}


### (72/122) Categorizing the Visual Environment and Analyzing the Visual Attention of Dogs (Shreyas Sundara Raman et al., 2023)

{{<citation>}}

Shreyas Sundara Raman, Madeline H. Pelgrim, Daphna Buchsbaum, Thomas Serre. (2023)  
**Categorizing the Visual Environment and Analyzing the Visual Attention of Dogs**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2311.11988v1)  

---


**ABSTRACT**  
Dogs have a unique evolutionary relationship with humans and serve many important roles e.g. search and rescue, blind assistance, emotional support. However, few datasets exist to categorize visual features and objects available to dogs, as well as how dogs direct their visual attention within their environment. We collect and study a dataset with over 11,698 gazes to categorize the objects available to be gazed at by 11 dogs in everyday outdoor environments i.e. a walk around a college campus and urban area. We explore the availability of these object categories and the visual attention of dogs over these categories using a head mounted eye tracking apparatus. A small portion (approx. 600 images or < 20% of total dataset) of the collected data is used to fine tune a MaskRCNN for the novel image domain to segment objects present in the scene, enabling further statistical analysis on the visual gaze tendencies of dogs. The MaskRCNN, with eye tracking apparatus, serves as an end to end model for automatically classifying the visual fixations of dogs. The fine tuned MaskRCNN performs far better than chance. There are few individual differences between the 11 dogs and we observe greater visual fixations on buses, plants, pavement, and construction equipment. This work takes a step towards understanding visual behavior of dogs and their interaction with the physical world.

{{</citation>}}


### (73/122) Leveraging Previous Facial Action Units Knowledge for Emotion Recognition on Faces (Pietro B. S. Masur et al., 2023)

{{<citation>}}

Pietro B. S. Masur, Willams Costa, Lucas S. Figueredo, Veronica Teichrieb. (2023)  
**Leveraging Previous Facial Action Units Knowledge for Emotion Recognition on Faces**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2311.11980v1)  

---


**ABSTRACT**  
People naturally understand emotions, thus permitting a machine to do the same could open new paths for human-computer interaction. Facial expressions can be very useful for emotion recognition techniques, as these are the biggest transmitters of non-verbal cues capable of being correlated with emotions. Several techniques are based on Convolutional Neural Networks (CNNs) to extract information in a machine learning process. However, simple CNNs are not always sufficient to locate points of interest on the face that can be correlated with emotions. In this work, we intend to expand the capacity of emotion recognition techniques by proposing the usage of Facial Action Units (AUs) recognition techniques to recognize emotions. This recognition will be based on the Facial Action Coding System (FACS) and computed by a machine learning system. In particular, our method expands over EmotiRAM, an approach for multi-cue emotion recognition, in which we improve over their facial encoding module.

{{</citation>}}


### (74/122) LLMs as Visual Explainers: Advancing Image Classification with Evolving Visual Descriptions (Songhao Han et al., 2023)

{{<citation>}}

Songhao Han, Le Zhuo, Yue Liao, Si Liu. (2023)  
**LLMs as Visual Explainers: Advancing Image Classification with Evolving Visual Descriptions**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs-LG, cs.CV  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2311.11904v1)  

---


**ABSTRACT**  
Vision-language models (VLMs) offer a promising paradigm for image classification by comparing the similarity between images and class embeddings. A critical challenge lies in crafting precise textual representations for class names. While previous studies have leveraged recent advancements in large language models (LLMs) to enhance these descriptors, their outputs often suffer from ambiguity and inaccuracy. We identify two primary causes: 1) The prevalent reliance on textual interactions with LLMs, leading to a mismatch between the generated text and the visual content in VLMs' latent space - a phenomenon we term the "explain without seeing" dilemma. 2) The oversight of the inter-class relationships, resulting in descriptors that fail to differentiate similar classes effectively. To address these issues, we propose a novel image classification framework combining VLMs with LLMs, named Iterative Optimization with Visual Feedback. In particular, our method develops an LLM-based agent, employing an evolutionary optimization strategy to refine class descriptors. Crucially, we incorporate visual feedback from VLM classification metrics, thereby guiding the optimization process with concrete visual data. Our method leads to improving accuracy on a wide range of image classification benchmarks, with 3.47\% average gains over state-of-the-art methods. We also highlight the resulting descriptions serve as explainable and robust features that can consistently improve the performance across various backbone models.

{{</citation>}}


### (75/122) Identifying the Defective: Detecting Damaged Grains for Cereal Appearance Inspection (Lei Fan et al., 2023)

{{<citation>}}

Lei Fan, Yiwen Ding, Dongdong Fan, Yong Wu, Maurice Pagnucco, Yang Song. (2023)  
**Identifying the Defective: Detecting Damaged Grains for Cereal Appearance Inspection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2311.11901v1)  

---


**ABSTRACT**  
Cereal grain plays a crucial role in the human diet as a major source of essential nutrients. Grain Appearance Inspection (GAI) serves as an essential process to determine grain quality and facilitate grain circulation and processing. However, GAI is routinely performed manually by inspectors with cumbersome procedures, which poses a significant bottleneck in smart agriculture.   In this paper, we endeavor to develop an automated GAI system:AI4GrainInsp. By analyzing the distinctive characteristics of grain kernels, we formulate GAI as a ubiquitous problem: Anomaly Detection (AD), in which healthy and edible kernels are considered normal samples while damaged grains or unknown objects are regarded as anomalies. We further propose an AD model, called AD-GAI, which is trained using only normal samples yet can identify anomalies during inference. Moreover, we customize a prototype device for data acquisition and create a large-scale dataset including 220K high-quality images of wheat and maize kernels. Through extensive experiments, AD-GAI achieves considerable performance in comparison with advanced AD methods, and AI4GrainInsp has highly consistent performance compared to human experts and excels at inspection efficiency over 20x speedup. The dataset, code and models will be released at https://github.com/hellodfan/AI4GrainInsp.

{{</citation>}}


### (76/122) Multi-Task Faces (MTF) Data Set: A Legally and Ethically Compliant Collection of Face Images for Various Classification Tasks (Rami Haffar et al., 2023)

{{<citation>}}

Rami Haffar, David Sánchez, Josep Domingo-Ferrer. (2023)  
**Multi-Task Faces (MTF) Data Set: A Legally and Ethically Compliant Collection of Face Images for Various Classification Tasks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Legal  
[Paper Link](http://arxiv.org/abs/2311.11882v1)  

---


**ABSTRACT**  
Human facial data hold tremendous potential to address a variety of classification problems, including face recognition, age estimation, gender identification, emotion analysis, and race classification. However, recent privacy regulations, such as the EU General Data Protection Regulation and others, have restricted the ways in which human images may be collected and used for research. As a result, several previously published data sets containing human faces have been removed from the internet due to inadequate data collection methods that failed to meet privacy regulations. Data sets consisting of synthetic data have been proposed as an alternative, but they fall short of accurately representing the real data distribution. On the other hand, most available data sets are labeled for just a single task, which limits their applicability. To address these issues, we present the Multi-Task Faces (MTF) image data set, a meticulously curated collection of face images designed for various classification tasks, including face recognition, as well as race, gender, and age classification. The MTF data set has been ethically gathered by leveraging publicly available images of celebrities and strictly adhering to copyright regulations. In this paper, we present this data set and provide detailed descriptions of the followed data collection and processing procedures. Furthermore, we evaluate the performance of five deep learning (DL) models on the MTF data set across the aforementioned classification tasks. Additionally, we compare the performance of DL models over the processed MTF data and over raw data crawled from the internet. The reported results constitute a baseline for further research employing these data. The MTF data set can be accessed through the following link (please cite the present paper if you use the data set): https://github.com/RamiHaf/MTF_data_set

{{</citation>}}


### (77/122) VLM-Eval: A General Evaluation on Video Large Language Models (Shuailin Li et al., 2023)

{{<citation>}}

Shuailin Li, Yuang Zhang, Yucheng Zhao, Qiuyue Wang, Fan Jia, Yingfei Liu, Tiancai Wang. (2023)  
**VLM-Eval: A General Evaluation on Video Large Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2311.11865v1)  

---


**ABSTRACT**  
Despite the rapid development of video Large Language Models (LLMs), a comprehensive evaluation is still absent. In this paper, we introduce a unified evaluation that encompasses multiple video tasks, including captioning, question and answering, retrieval, and action recognition. In addition to conventional metrics, we showcase how GPT-based evaluation can match human-like performance in assessing response quality across multiple aspects. We propose a simple baseline: Video-LLaVA, which uses a single linear projection and outperforms existing video LLMs. Finally, we evaluate video LLMs beyond academic datasets, which show encouraging recognition and reasoning capabilities in driving scenarios with only hundreds of video-instruction pairs for fine-tuning. We hope our work can serve as a unified evaluation for video LLMs, and help expand more practical scenarios. The evaluation code will be available soon.

{{</citation>}}


### (78/122) LION : Empowering Multimodal Large Language Model with Dual-Level Visual Knowledge (Gongwei Chen et al., 2023)

{{<citation>}}

Gongwei Chen, Leyang Shen, Rui Shao, Xiang Deng, Liqiang Nie. (2023)  
**LION : Empowering Multimodal Large Language Model with Dual-Level Visual Knowledge**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.11860v1)  

---


**ABSTRACT**  
Multimodal Large Language Models (MLLMs) have endowed LLMs with the ability to perceive and understand multi-modal signals. However, most of the existing MLLMs mainly adopt vision encoders pretrained on coarsely aligned image-text pairs, leading to insufficient extraction and reasoning of visual knowledge. To address this issue, we devise a dual-Level vIsual knOwledge eNhanced Multimodal Large Language Model (LION), which empowers the MLLM by injecting visual knowledge in two levels. 1) Progressive incorporation of fine-grained spatial-aware visual knowledge. We design a vision aggregator cooperated with region-level vision-language (VL) tasks to incorporate fine-grained spatial-aware visual knowledge into the MLLM. To alleviate the conflict between image-level and region-level VL tasks during incorporation, we devise a dedicated stage-wise instruction-tuning strategy with mixture-of-adapters. This progressive incorporation scheme contributes to the mutual promotion between these two kinds of VL tasks. 2) Soft prompting of high-level semantic visual evidence. We facilitate the MLLM with high-level semantic visual evidence by leveraging diverse image tags. To mitigate the potential influence caused by imperfect predicted tags, we propose a soft prompting method by embedding a learnable token into the tailored text instruction. Comprehensive experiments on several multi-modal benchmarks demonstrate the superiority of our model (e.g., improvement of 5% accuracy on VSR and 3% CIDEr on TextCaps over InstructBLIP, 5% accuracy on RefCOCOg over Kosmos-2).

{{</citation>}}


### (79/122) Few-shot Multispectral Segmentation with Representations Generated by Reinforcement Learning (Dilith Jayakody et al., 2023)

{{<citation>}}

Dilith Jayakody, Thanuja Ambegoda. (2023)  
**Few-shot Multispectral Segmentation with Representations Generated by Reinforcement Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.11827v1)  

---


**ABSTRACT**  
The task of multispectral image segmentation (segmentation of images with numerous channels/bands, each capturing a specific range of wavelengths of electromagnetic radiation) has been previously explored in contexts with large amounts of labeled data. However, these models tend not to generalize well to datasets of smaller size. In this paper, we propose a novel approach for improving few-shot segmentation performance on multispectral images using reinforcement learning to generate representations. These representations are generated in the form of mathematical expressions between channels and are tailored to the specific class being segmented. Our methodology involves training an agent to identify the most informative expressions, updating the dataset using these expressions, and then using the updated dataset to perform segmentation. Due to the limited length of the expressions, the model receives useful representations without any added risk of overfitting. We evaluate the effectiveness of our approach on several multispectral datasets and demonstrate its effectiveness in boosting the performance of segmentation algorithms.

{{</citation>}}


### (80/122) DocPedia: Unleashing the Power of Large Multimodal Model in the Frequency Domain for Versatile Document Understanding (Hao Feng et al., 2023)

{{<citation>}}

Hao Feng, Qi Liu, Hao Liu, Wengang Zhou, Houqiang Li, Can Huang. (2023)  
**DocPedia: Unleashing the Power of Large Multimodal Model in the Frequency Domain for Versatile Document Understanding**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: OCR  
[Paper Link](http://arxiv.org/abs/2311.11810v1)  

---


**ABSTRACT**  
This work presents DocPedia, a novel large multimodal model (LMM) for versatile OCR-free document understanding, capable of parsing images up to 2,560$\times$2,560 resolution. Unlike existing work either struggle with high-resolution documents or give up the large language model thus vision or language ability constrained, our DocPedia directly processes visual input in the frequency domain rather than the pixel space. The unique characteristic enables DocPedia to capture a greater amount of visual and textual information using a limited number of visual tokens. To consistently enhance both perception and comprehension abilities of our model, we develop a dual-stage training strategy and enrich instructions/annotations of all training tasks covering multiple document types. Extensive quantitative and qualitative experiments conducted on various publicly available benchmarks confirm the mutual benefits of jointly learning perception and comprehension tasks. The results provide further evidence of the effectiveness and superior performance of our DocPedia over other methods.

{{</citation>}}


### (81/122) Non-Contact NIR PPG Sensing through Large Sequence Signal Regression (Timothy Hanley et al., 2023)

{{<citation>}}

Timothy Hanley, Dara Golden, Robyn Maxwell, Ashkan Parsi, Joseph Lemley. (2023)  
**Non-Contact NIR PPG Sensing through Large Sequence Signal Regression**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2311.11757v1)  

---


**ABSTRACT**  
Non-Contact sensing is an emerging technology with applications across many industries from driver monitoring in vehicles to patient monitoring in healthcare. Current state-of-the-art implementations focus on RGB video, but this struggles in varying/noisy light conditions and is almost completely unfeasible in the dark. Near Infra-Red (NIR) video, however, does not suffer from these constraints. This paper aims to demonstrate the effectiveness of an alternative Convolution Attention Network (CAN) architecture, to regress photoplethysmography (PPG) signal from a sequence of NIR frames. A combination of two publicly available datasets, which is split into train and test sets, is used for training the CAN. This combined dataset is augmented to reduce overfitting to the 'normal' 60 - 80 bpm heart rate range by providing the full range of heart rates along with corresponding videos for each subject. This CAN, when implemented over video cropped to the subject's head, achieved a Mean Average Error (MAE) of just 0.99 bpm, proving its effectiveness on NIR video and the architecture's feasibility to regress an accurate signal output.

{{</citation>}}


### (82/122) A Large-Scale Car Parts (LSCP) Dataset for Lightweight Fine-Grained Detection (Wang Jie et al., 2023)

{{<citation>}}

Wang Jie, Zhong Yilin, Cao Qianqian. (2023)  
**A Large-Scale Car Parts (LSCP) Dataset for Lightweight Fine-Grained Detection**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.11754v1)  

---


**ABSTRACT**  
Automotive related datasets have previously been used for training autonomous driving systems or vehicle classification tasks. However, there is a lack of datasets in the field of automotive AI for car parts detection, and most available datasets are limited in size and scope, struggling to cover diverse scenarios. To address this gap, this paper presents a large-scale and fine-grained automotive dataset consisting of 84,162 images for detecting 12 different types of car parts. This dataset was collected from natural cameras and online websites which covers various car brands, scenarios, and shooting angles. To alleviate the burden of manual annotation, we propose a novel semi-supervised auto-labeling method that leverages state-of-the-art pre-trained detectors. Moreover, we study the limitations of the Grounding DINO approach for zero-shot labeling. Finally, we evaluate the effectiveness of our proposed dataset through fine-grained car parts detection by training several lightweight YOLO-series detectors.

{{</citation>}}


### (83/122) AdvGen: Physical Adversarial Attack on Face Presentation Attack Detection Systems (Sai Amrit Patnaik et al., 2023)

{{<citation>}}

Sai Amrit Patnaik, Shivali Chansoriya, Anil K. Jain, Anoop M. Namboodiri. (2023)  
**AdvGen: Physical Adversarial Attack on Face Presentation Attack Detection Systems**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2311.11753v1)  

---


**ABSTRACT**  
Evaluating the risk level of adversarial images is essential for safely deploying face authentication models in the real world. Popular approaches for physical-world attacks, such as print or replay attacks, suffer from some limitations, like including physical and geometrical artifacts. Recently, adversarial attacks have gained attraction, which try to digitally deceive the learning strategy of a recognition system using slight modifications to the captured image. While most previous research assumes that the adversarial image could be digitally fed into the authentication systems, this is not always the case for systems deployed in the real world. This paper demonstrates the vulnerability of face authentication systems to adversarial images in physical world scenarios. We propose AdvGen, an automated Generative Adversarial Network, to simulate print and replay attacks and generate adversarial images that can fool state-of-the-art PADs in a physical domain attack setting. Using this attack strategy, the attack success rate goes up to 82.01%. We test AdvGen extensively on four datasets and ten state-of-the-art PADs. We also demonstrate the effectiveness of our attack by conducting experiments in a realistic, physical environment.

{{</citation>}}


### (84/122) On the Importance of Large Objects in CNN Based Object Detection Algorithms (Ahmed Ben Saad et al., 2023)

{{<citation>}}

Ahmed Ben Saad, Gabriele Facciolo, Axel Davy. (2023)  
**On the Importance of Large Objects in CNN Based Object Detection Algorithms**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2311.11714v1)  

---


**ABSTRACT**  
Object detection models, a prominent class of machine learning algorithms, aim to identify and precisely locate objects in images or videos. However, this task might yield uneven performances sometimes caused by the objects sizes and the quality of the images and labels used for training. In this paper, we highlight the importance of large objects in learning features that are critical for all sizes. Given these findings, we propose to introduce a weighting term into the training loss. This term is a function of the object area size. We show that giving more weight to large objects leads to improved detection scores across all object sizes and so an overall improvement in Object Detectors performances (+2 p.p. of mAP on small objects, +2 p.p. on medium and +4 p.p. on large on COCO val 2017 with InternImage-T). Additional experiments and ablation studies with different models and on a different dataset further confirm the robustness of our findings.

{{</citation>}}


### (85/122) Cut-and-Paste: Subject-Driven Video Editing with Attention Control (Zhichao Zuo et al., 2023)

{{<citation>}}

Zhichao Zuo, Zhao Zhang, Yan Luo, Yang Zhao, Haijun Zhang, Yi Yang, Meng Wang. (2023)  
**Cut-and-Paste: Subject-Driven Video Editing with Attention Control**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2311.11697v1)  

---


**ABSTRACT**  
This paper presents a novel framework termed Cut-and-Paste for real-word semantic video editing under the guidance of text prompt and additional reference image. While the text-driven video editing has demonstrated remarkable ability to generate highly diverse videos following given text prompts, the fine-grained semantic edits are hard to control by plain textual prompt only in terms of object details and edited region, and cumbersome long text descriptions are usually needed for the task. We therefore investigate subject-driven video editing for more precise control of both edited regions and background preservation, and fine-grained semantic generation. We achieve this goal by introducing an reference image as supplementary input to the text-driven video editing, which avoids racking your brain to come up with a cumbersome text prompt describing the detailed appearance of the object. To limit the editing area, we refer to a method of cross attention control in image editing and successfully extend it to video editing by fusing the attention map of adjacent frames, which strikes a balance between maintaining video background and spatio-temporal consistency. Compared with current methods, the whole process of our method is like ``cut" the source object to be edited and then ``paste" the target object provided by reference image. We demonstrate that our method performs favorably over prior arts for video editing under the guidance of text prompt and extra reference image, as measured by both quantitative and subjective evaluations.

{{</citation>}}


### (86/122) Clarity ChatGPT: An Interactive and Adaptive Processing System for Image Restoration and Enhancement (Yanyan Wei et al., 2023)

{{<citation>}}

Yanyan Wei, Zhao Zhang, Jiahuan Ren, Xiaogang Xu, Richang Hong, Yi Yang, Shuicheng Yan, Meng Wang. (2023)  
**Clarity ChatGPT: An Interactive and Adaptive Processing System for Image Restoration and Enhancement**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2311.11695v1)  

---


**ABSTRACT**  
The generalization capability of existing image restoration and enhancement (IRE) methods is constrained by the limited pre-trained datasets, making it difficult to handle agnostic inputs such as different degradation levels and scenarios beyond their design scopes. Moreover, they are not equipped with interactive mechanisms to consider user preferences or feedback, and their end-to-end settings cannot provide users with more choices. Faced with the above-mentioned IRE method's limited performance and insufficient interactivity, we try to solve it from the engineering and system framework levels. Specifically, we propose Clarity ChatGPT-a transformative system that combines the conversational intelligence of ChatGPT with multiple IRE methods. Clarity ChatGPT can automatically detect image degradation types and select appropriate IRE methods to restore images, or iteratively generate satisfactory results based on user feedback. Its innovative features include a CLIP-powered detector for accurate degradation classification, no-reference image quality evaluation for performance evaluation, region-specific processing for precise enhancements, and advanced fusion techniques for optimal restoration results. Clarity ChatGPT marks a significant advancement in integrating language and vision, enhancing image-text interactions, and providing a robust, high-performance IRE solution. Our case studies demonstrate that Clarity ChatGPT effectively improves the generalization and interaction capabilities in the IRE, and also fills the gap in the low-level domain of the existing vision-language model.

{{</citation>}}


### (87/122) Segment Together: A Versatile Paradigm for Semi-Supervised Medical Image Segmentation (Qingjie Zeng et al., 2023)

{{<citation>}}

Qingjie Zeng, Yutong Xie, Zilin Lu, Mengkang Lu, Yicheng Wu, Yong Xia. (2023)  
**Segment Together: A Versatile Paradigm for Semi-Supervised Medical Image Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2311.11686v1)  

---


**ABSTRACT**  
Annotation scarcity has become a major obstacle for training powerful deep-learning models for medical image segmentation, restricting their deployment in clinical scenarios. To address it, semi-supervised learning by exploiting abundant unlabeled data is highly desirable to boost the model training. However, most existing works still focus on limited medical tasks and underestimate the potential of learning across diverse tasks and multiple datasets. Therefore, in this paper, we introduce a \textbf{Ver}satile \textbf{Semi}-supervised framework (VerSemi) to point out a new perspective that integrates various tasks into a unified model with a broad label space, to exploit more unlabeled data for semi-supervised medical image segmentation. Specifically, we introduce a dynamic task-prompted design to segment various targets from different datasets. Next, this unified model is used to identify the foreground regions from all labeled data, to capture cross-dataset semantics. Particularly, we create a synthetic task with a cutmix strategy to augment foreground targets within the expanded label space. To effectively utilize unlabeled data, we introduce a consistency constraint. This involves aligning aggregated predictions from various tasks with those from the synthetic task, further guiding the model in accurately segmenting foreground regions during training. We evaluated our VerSemi model on four public benchmarking datasets. Extensive experiments demonstrated that VerSemi can consistently outperform the second-best method by a large margin (e.g., an average 2.69\% Dice gain on four datasets), setting new SOTA performance for semi-supervised medical image segmentation. The code will be released.

{{</citation>}}


### (88/122) PMP-Swin: Multi-Scale Patch Message Passing Swin Transformer for Retinal Disease Classification (Zhihan Yang et al., 2023)

{{<citation>}}

Zhihan Yang, Zhiming Cheng, Tengjin Weng, Shucheng He, Yaqi Wang, Xin Ye, Shuai Wang. (2023)  
**PMP-Swin: Multi-Scale Patch Message Passing Swin Transformer for Retinal Disease Classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.11669v1)  

---


**ABSTRACT**  
Retinal disease is one of the primary causes of visual impairment, and early diagnosis is essential for preventing further deterioration. Nowadays, many works have explored Transformers for diagnosing diseases due to their strong visual representation capabilities. However, retinal diseases exhibit milder forms and often present with overlapping signs, which pose great difficulties for accurate multi-class classification. Therefore, we propose a new framework named Multi-Scale Patch Message Passing Swin Transformer for multi-class retinal disease classification. Specifically, we design a Patch Message Passing (PMP) module based on the Message Passing mechanism to establish global interaction for pathological semantic features and to exploit the subtle differences further between different diseases. Moreover, considering the various scale of pathological features we integrate multiple PMP modules for different patch sizes. For evaluation, we have constructed a new dataset, named OPTOS dataset, consisting of 1,033 high-resolution fundus images photographed by Optos camera and conducted comprehensive experiments to validate the efficacy of our proposed method. And the results on both the public dataset and our dataset demonstrate that our method achieves remarkable performance compared to state-of-the-art methods.

{{</citation>}}


### (89/122) OmniSeg3D: Omniversal 3D Segmentation via Hierarchical Contrastive Learning (Haiyang Ying et al., 2023)

{{<citation>}}

Haiyang Ying, Yixuan Yin, Jinzhi Zhang, Fan Wang, Tao Yu, Ruqi Huang, Lu Fang. (2023)  
**OmniSeg3D: Omniversal 3D Segmentation via Hierarchical Contrastive Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2311.11666v1)  

---


**ABSTRACT**  
Towards holistic understanding of 3D scenes, a general 3D segmentation method is needed that can segment diverse objects without restrictions on object quantity or categories, while also reflecting the inherent hierarchical structure. To achieve this, we propose OmniSeg3D, an omniversal segmentation method aims for segmenting anything in 3D all at once. The key insight is to lift multi-view inconsistent 2D segmentations into a consistent 3D feature field through a hierarchical contrastive learning framework, which is accomplished by two steps. Firstly, we design a novel hierarchical representation based on category-agnostic 2D segmentations to model the multi-level relationship among pixels. Secondly, image features rendered from the 3D feature field are clustered at different levels, which can be further drawn closer or pushed apart according to the hierarchical relationship between different levels. In tackling the challenges posed by inconsistent 2D segmentations, this framework yields a global consistent 3D feature field, which further enables hierarchical segmentation, multi-object selection, and global discretization. Extensive experiments demonstrate the effectiveness of our method on high-quality 3D segmentation and accurate hierarchical structure understanding. A graphical user interface further facilitates flexible interaction for omniversal 3D segmentation.

{{</citation>}}


### (90/122) Enhanced Spatio-Temporal Context for Temporally Consistent Robust 3D Human Motion Recovery from Monocular Videos (Sushovan Chanda et al., 2023)

{{<citation>}}

Sushovan Chanda, Amogh Tiwari, Lokender Tiwari, Brojeshwar Bhowmick, Avinash Sharma, Hrishav Barua. (2023)  
**Enhanced Spatio-Temporal Context for Temporally Consistent Robust 3D Human Motion Recovery from Monocular Videos**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2311.11662v1)  

---


**ABSTRACT**  
Recovering temporally consistent 3D human body pose, shape and motion from a monocular video is a challenging task due to (self-)occlusions, poor lighting conditions, complex articulated body poses, depth ambiguity, and limited availability of annotated data. Further, doing a simple perframe estimation is insufficient as it leads to jittery and implausible results. In this paper, we propose a novel method for temporally consistent motion estimation from a monocular video. Instead of using generic ResNet-like features, our method uses a body-aware feature representation and an independent per-frame pose and camera initialization over a temporal window followed by a novel spatio-temporal feature aggregation by using a combination of self-similarity and self-attention over the body-aware features and the perframe initialization. Together, they yield enhanced spatiotemporal context for every frame by considering remaining past and future frames. These features are used to predict the pose and shape parameters of the human body model, which are further refined using an LSTM. Experimental results on the publicly available benchmark data show that our method attains significantly lower acceleration error and outperforms the existing state-of-the-art methods over all key quantitative evaluation metrics, including complex scenarios like partial occlusion, complex poses and even relatively low illumination.

{{</citation>}}


### (91/122) MGCT: Mutual-Guided Cross-Modality Transformer for Survival Outcome Prediction using Integrative Histopathology-Genomic Features (Mingxin Liu et al., 2023)

{{<citation>}}

Mingxin Liu, Yunzan Liu, Hui Cui, Chunquan Li, Jiquan Ma. (2023)  
**MGCT: Mutual-Guided Cross-Modality Transformer for Survival Outcome Prediction using Integrative Histopathology-Genomic Features**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.11659v1)  

---


**ABSTRACT**  
The rapidly emerging field of deep learning-based computational pathology has shown promising results in utilizing whole slide images (WSIs) to objectively prognosticate cancer patients. However, most prognostic methods are currently limited to either histopathology or genomics alone, which inevitably reduces their potential to accurately predict patient prognosis. Whereas integrating WSIs and genomic features presents three main challenges: (1) the enormous heterogeneity of gigapixel WSIs which can reach sizes as large as 150,000x150,000 pixels; (2) the absence of a spatially corresponding relationship between histopathology images and genomic molecular data; and (3) the existing early, late, and intermediate multimodal feature fusion strategies struggle to capture the explicit interactions between WSIs and genomics. To ameliorate these issues, we propose the Mutual-Guided Cross-Modality Transformer (MGCT), a weakly-supervised, attention-based multimodal learning framework that can combine histology features and genomic features to model the genotype-phenotype interactions within the tumor microenvironment. To validate the effectiveness of MGCT, we conduct experiments using nearly 3,600 gigapixel WSIs across five different cancer types sourced from The Cancer Genome Atlas (TCGA). Extensive experimental results consistently emphasize that MGCT outperforms the state-of-the-art (SOTA) methods.

{{</citation>}}


### (92/122) Cancer-Net PCa-Data: An Open-Source Benchmark Dataset for Prostate Cancer Clinical Decision Support using Synthetic Correlated Diffusion Imaging Data (Hayden Gunraj et al., 2023)

{{<citation>}}

Hayden Gunraj, Chi-en Amy Tai, Alexander Wong. (2023)  
**Cancer-Net PCa-Data: An Open-Source Benchmark Dataset for Prostate Cancer Clinical Decision Support using Synthetic Correlated Diffusion Imaging Data**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Clinical  
[Paper Link](http://arxiv.org/abs/2311.11647v1)  

---


**ABSTRACT**  
The recent introduction of synthetic correlated diffusion (CDI$^s$) imaging has demonstrated significant potential in the realm of clinical decision support for prostate cancer (PCa). CDI$^s$ is a new form of magnetic resonance imaging (MRI) designed to characterize tissue characteristics through the joint correlation of diffusion signal attenuation across different Brownian motion sensitivities. Despite the performance improvement, the CDI$^s$ data for PCa has not been previously made publicly available. In our commitment to advance research efforts for PCa, we introduce Cancer-Net PCa-Data, an open-source benchmark dataset of volumetric CDI$^s$ imaging data of PCa patients. Cancer-Net PCa-Data consists of CDI$^s$ volumetric images from a patient cohort of 200 patient cases, along with full annotations (gland masks, tumor masks, and PCa diagnosis for each tumor). We also analyze the demographic and label region diversity of Cancer-Net PCa-Data for potential biases. Cancer-Net PCa-Data is the first-ever public dataset of CDI$^s$ imaging data for PCa, and is a part of the global open-source initiative dedicated to advancement in machine learning and imaging research to aid clinicians in the global fight against cancer.

{{</citation>}}


### (93/122) CastDet: Toward Open Vocabulary Aerial Object Detection with CLIP-Activated Student-Teacher Learning (Yan Li et al., 2023)

{{<citation>}}

Yan Li, Weiwei Guo, Dunyun He, Jiaqi Zhou, Yuze Gao, Wenxian Yu. (2023)  
**CastDet: Toward Open Vocabulary Aerial Object Detection with CLIP-Activated Student-Teacher Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Drone, Object Detection  
[Paper Link](http://arxiv.org/abs/2311.11646v1)  

---


**ABSTRACT**  
Object detection in aerial images is a pivotal task for various earth observation applications, whereas current algorithms learn to detect only a pre-defined set of object categories demanding sufficient bounding-box annotated training samples and fail to detect novel object categories. In this paper, we consider open-vocabulary object detection (OVD) in aerial images that enables the characterization of new objects beyond training categories on the earth surface without annotating training images for these new categories. The performance of OVD depends on the quality of class-agnostic region proposals and pseudo-labels that can generalize well to novel object categories. To simultaneously generate high-quality proposals and pseudo-labels, we propose CastDet, a CLIP-activated student-teacher open-vocabulary object Detection framework. Our end-to-end framework within the student-teacher mechanism employs the CLIP model as an extra omniscient teacher of rich knowledge into the student-teacher self-learning process. By doing so, our approach boosts novel object proposals and classification. Furthermore, we design a dynamic label queue technique to maintain high-quality pseudo labels during batch training and mitigate label imbalance. We conduct extensive experiments on multiple existing aerial object detection datasets, which are set up for the OVD task. Experimental results demonstrate our CastDet achieving superior open-vocabulary detection performance, e.g., reaching 40.0 HM (Harmonic Mean), which outperforms previous methods Detic/ViLD by 26.9/21.1 on the VisDroneZSD dataset.

{{</citation>}}


### (94/122) FreeKD: Knowledge Distillation via Semantic Frequency Prompt (Yuan Zhang et al., 2023)

{{<citation>}}

Yuan Zhang, Tao Huang, Jiaming Liu, Tao Jiang, Kuan Cheng, Shanghang Zhang. (2023)  
**FreeKD: Knowledge Distillation via Semantic Frequency Prompt**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2311.12079v1)  

---


**ABSTRACT**  
Knowledge distillation (KD) has been applied to various tasks successfully, and mainstream methods typically boost the student model via spatial imitation losses. However, the consecutive downsamplings induced in the spatial domain of teacher model is a type of corruption, hindering the student from analyzing what specific information needs to be imitated, which results in accuracy degradation. To better understand the underlying pattern of corrupted feature maps, we shift our attention to the frequency domain. During frequency distillation, we encounter a new challenge: the low-frequency bands convey general but minimal context, while the high are more informative but also introduce noise. Not each pixel within the frequency bands contributes equally to the performance. To address the above problem: (1) We propose the Frequency Prompt plugged into the teacher model, absorbing the semantic frequency context during finetuning. (2) During the distillation period, a pixel-wise frequency mask is generated via Frequency Prompt, to localize those pixel of interests (PoIs) in various frequency bands. Additionally, we employ a position-aware relational frequency loss for dense prediction tasks, delivering a high-order spatial enhancement to the student model. We dub our Frequency Knowledge Distillation method as FreeKD, which determines the optimal localization and extent for the frequency distillation. Extensive experiments demonstrate that FreeKD not only outperforms spatial-based distillation methods consistently on dense prediction tasks (e.g., FreeKD brings 3.8 AP gains for RepPoints-R50 on COCO2017 and 4.55 mIoU gains for PSPNet-R18 on Cityscapes), but also conveys more robustness to the student. Notably, we also validate the generalization of our approach on large-scale vision models (e.g., DINO and SAM).

{{</citation>}}


### (95/122) AKConv: Convolutional Kernel with Arbitrary Sampled Shapes and Arbitrary Number of Parameters (Xin Zhang et al., 2023)

{{<citation>}}

Xin Zhang, Yingze Song, Tingting Song, Degang Yang, Yichen Ye, Jie Zhou, Liming Zhang. (2023)  
**AKConv: Convolutional Kernel with Arbitrary Sampled Shapes and Arbitrary Number of Parameters**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2311.11587v1)  

---


**ABSTRACT**  
Neural networks based on convolutional operations have achieved remarkable results in the field of deep learning, but there are two inherent flaws in standard convolutional operations. On the one hand, the convolution operation be confined to a local window and cannot capture information from other locations, and its sampled shapes is fixed. On the other hand, the size of the convolutional kernel is fixed to k $\times$ k, which is a fixed square shape, and the number of parameters tends to grow squarely with size. It is obvious that the shape and size of targets are various in different datasets and at different locations. Convolutional kernels with fixed sample shapes and squares do not adapt well to changing targets. In response to the above questions, the Alterable Kernel Convolution (AKConv) is explored in this work, which gives the convolution kernel an arbitrary number of parameters and arbitrary sampled shapes to provide richer options for the trade-off between network overhead and performance. In AKConv, we define initial positions for convolutional kernels of arbitrary size by means of a new coordinate generation algorithm. To adapt to changes for targets, we introduce offsets to adjust the shape of the samples at each position. Moreover, we explore the effect of the neural network by using the AKConv with the same size and different initial sampled shapes. AKConv completes the process of efficient feature extraction by irregular convolutional operations and brings more exploration options for convolutional sampling shapes. Object detection experiments on representative datasets COCO2017, VOC 7+12 and VisDrone-DET2021 fully demonstrate the advantages of AKConv. AKConv can be used as a plug-and-play convolutional operation to replace convolutional operations to improve network performance. The code for the relevant tasks can be found at https://github.com/CV-ZhangXin/AKConv.

{{</citation>}}


### (96/122) Decoupled DETR For Few-shot Object Detection (Zeyu Shangguan et al., 2023)

{{<citation>}}

Zeyu Shangguan, Lian Huai, Tong Liu, Xingqun Jiang. (2023)  
**Decoupled DETR For Few-shot Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2311.11570v1)  

---


**ABSTRACT**  
Few-shot object detection (FSOD), an efficient method for addressing the severe data-hungry problem, has been extensively discussed. Current works have significantly advanced the problem in terms of model and data. However, the overall performance of most FSOD methods still does not fulfill the desired accuracy. In this paper we improve the FSOD model to address the severe issue of sample imbalance and weak feature propagation. To alleviate modeling bias from data-sufficient base classes, we examine the effect of decoupling the parameters for classes with sufficient data and classes with few samples in various ways. We design a base-novel categories decoupled DETR (DeDETR) for FSOD. We also explore various types of skip connection between the encoder and decoder for DETR. Besides, we notice that the best outputs could come from the intermediate layer of the decoder instead of the last layer; therefore, we build a unified decoder module that could dynamically fuse the decoder layers as the output feature. We evaluate our model on commonly used datasets such as PASCAL VOC and MSCOCO. Our results indicate that our proposed module could achieve stable improvements of 5% to 10% in both fine-tuning and meta-learning paradigms and has outperformed the highest score in recent works.

{{</citation>}}


### (97/122) CORE-MM: Complex Open-Ended Reasoning Evaluation For Multi-Modal Large Language Models (Xiaotian Han et al., 2023)

{{<citation>}}

Xiaotian Han, Quanzeng You, Yongfei Liu, Wentao Chen, Huangjie Zheng, Khalil Mrini, Xudong Lin, Yiqi Wang, Bohan Zhai, Jianbo Yuan, Heng Wang, Hongxia Yang. (2023)  
**CORE-MM: Complex Open-Ended Reasoning Evaluation For Multi-Modal Large Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2311.11567v1)  

---


**ABSTRACT**  
Multi-modal Large Language Models (MLLMs) are increasingly prominent in the field of artificial intelligence. These models not only excel in traditional vision-language tasks but also demonstrate impressive performance in contemporary multi-modal benchmarks. Although many of these benchmarks attempt to holistically evaluate MLLMs, they typically concentrate on basic reasoning tasks, often yielding only simple yes/no or multi-choice responses. These methods naturally lead to confusion and difficulties in conclusively determining the reasoning capabilities of MLLMs. To mitigate this issue, we manually curate a benchmark dataset specifically designed for MLLMs, with a focus on complex reasoning tasks. Our benchmark comprises three key reasoning categories: deductive, abductive, and analogical reasoning. The queries in our dataset are intentionally constructed to engage the reasoning capabilities of MLLMs in the process of generating answers. For a fair comparison across various MLLMs, we incorporate intermediate reasoning steps into our evaluation criteria. In instances where an MLLM is unable to produce a definitive answer, its reasoning ability is evaluated by requesting intermediate reasoning steps. If these steps align with our manual annotations, appropriate scores are assigned. This evaluation scheme resembles methods commonly used in human assessments, such as exams or assignments, and represents what we consider a more effective assessment technique compared with existing benchmarks. We evaluate a selection of representative MLLMs using this rigorously developed open-ended multi-step elaborate reasoning benchmark, designed to challenge and accurately measure their reasoning capabilities. The code and data will be released at https://core-mm.github.io/

{{</citation>}}


### (98/122) Generalized Category Discovery in Semantic Segmentation (Zhengyuan Peng et al., 2023)

{{<citation>}}

Zhengyuan Peng, Qijian Tian, Jianqing Xu, Yizhang Jin, Xuequan Lu, Xin Tan, Yuan Xie, Lizhuang Ma. (2023)  
**Generalized Category Discovery in Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2311.11525v1)  

---


**ABSTRACT**  
This paper explores a novel setting called Generalized Category Discovery in Semantic Segmentation (GCDSS), aiming to segment unlabeled images given prior knowledge from a labeled set of base classes. The unlabeled images contain pixels of the base class or novel class. In contrast to Novel Category Discovery in Semantic Segmentation (NCDSS), there is no prerequisite for prior knowledge mandating the existence of at least one novel class in each unlabeled image. Besides, we broaden the segmentation scope beyond foreground objects to include the entire image. Existing NCDSS methods rely on the aforementioned priors, making them challenging to truly apply in real-world situations. We propose a straightforward yet effective framework that reinterprets the GCDSS challenge as a task of mask classification. Additionally, we construct a baseline method and introduce the Neighborhood Relations-Guided Mask Clustering Algorithm (NeRG-MaskCA) for mask categorization to address the fragmentation in semantic representation. A benchmark dataset, Cityscapes-GCD, derived from the Cityscapes dataset, is established to evaluate the GCDSS framework. Our method demonstrates the feasibility of the GCDSS problem and the potential for discovering and segmenting novel object classes in unlabeled images. We employ the generated pseudo-labels from our approach as ground truth to supervise the training of other models, thereby enabling them with the ability to segment novel classes. It paves the way for further research in generalized category discovery, broadening the horizons of semantic segmentation and its applications. For details, please visit https://github.com/JethroPeng/GCDSS

{{</citation>}}


### (99/122) BadCLIP: Dual-Embedding Guided Backdoor Attack on Multimodal Contrastive Learning (Siyuan Liang et al., 2023)

{{<citation>}}

Siyuan Liang, Mingli Zhu, Aishan Liu, Baoyuan Wu, Xiaochun Cao, Ee-Chien Chang. (2023)  
**BadCLIP: Dual-Embedding Guided Backdoor Attack on Multimodal Contrastive Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning, Embedding  
[Paper Link](http://arxiv.org/abs/2311.12075v1)  

---


**ABSTRACT**  
Studying backdoor attacks is valuable for model copyright protection and enhancing defenses. While existing backdoor attacks have successfully infected multimodal contrastive learning models such as CLIP, they can be easily countered by specialized backdoor defenses for MCL models. This paper reveals the threats in this practical scenario that backdoor attacks can remain effective even after defenses and introduces the \emph{\toolns} attack, which is resistant to backdoor detection and model fine-tuning defenses. To achieve this, we draw motivations from the perspective of the Bayesian rule and propose a dual-embedding guided framework for backdoor attacks. Specifically, we ensure that visual trigger patterns approximate the textual target semantics in the embedding space, making it challenging to detect the subtle parameter variations induced by backdoor learning on such natural trigger patterns. Additionally, we optimize the visual trigger patterns to align the poisoned samples with target vision features in order to hinder the backdoor unlearning through clean fine-tuning. Extensive experiments demonstrate that our attack significantly outperforms state-of-the-art baselines (+45.3% ASR) in the presence of SoTA backdoor defenses, rendering these mitigation and detection strategies virtually ineffective. Furthermore, our approach effectively attacks some more rigorous scenarios like downstream tasks. We believe that this paper raises awareness regarding the potential threats associated with the practical application of multimodal contrastive learning and encourages the development of more robust defense mechanisms.

{{</citation>}}


## eess.IV (6)



### (100/122) Uncertainty Estimation in Contrast-Enhanced MR Image Translation with Multi-Axis Fusion (Ivo M. Baltruschat et al., 2023)

{{<citation>}}

Ivo M. Baltruschat, Parvaneh Janbakhshi, Melanie Dohmen, Matthias Lenga. (2023)  
**Uncertainty Estimation in Contrast-Enhanced MR Image Translation with Multi-Axis Fusion**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.12153v1)  

---


**ABSTRACT**  
In recent years, deep learning has been applied to a wide range of medical imaging and image processing tasks. In this work, we focus on the estimation of epistemic uncertainty for 3D medical image-to-image translation. We propose a novel model uncertainty quantification method, Multi-Axis Fusion (MAF), which relies on the integration of complementary information derived from multiple views on volumetric image data. The proposed approach is applied to the task of synthesizing contrast enhanced T1-weighted images based on native T1, T2 and T2-FLAIR scans. The quantitative findings indicate a strong correlation ($\rho_{\text healthy} = 0.89$) between the mean absolute image synthetization error and the mean uncertainty score for our MAF method. Hence, we consider MAF as a promising approach to solve the highly relevant task of detecting synthetization failures at inference time.

{{</citation>}}


### (101/122) Robust Tumor Segmentation with Hyperspectral Imaging and Graph Neural Networks (Mayar Lotfy et al., 2023)

{{<citation>}}

Mayar Lotfy, Anna Alperovich, Tommaso Giannantonio, Bjorn Barz, Xiaohan Zhang, Felix Holm, Nassir Navab, Felix Boehm, Carolin Schwamborn, Thomas K. Hoffmann, Patrick J. Schuler. (2023)  
**Robust Tumor Segmentation with Hyperspectral Imaging and Graph Neural Networks**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2311.11782v1)  

---


**ABSTRACT**  
Segmenting the boundary between tumor and healthy tissue during surgical cancer resection poses a significant challenge. In recent years, Hyperspectral Imaging (HSI) combined with Machine Learning (ML) has emerged as a promising solution. However, due to the extensive information contained within the spectral domain, most ML approaches primarily classify individual HSI (super-)pixels, or tiles, without taking into account their spatial context. In this paper, we propose an improved methodology that leverages the spatial context of tiles for more robust and smoother segmentation. To address the irregular shapes of tiles, we utilize Graph Neural Networks (GNNs) to propagate context information across neighboring regions. The features for each tile within the graph are extracted using a Convolutional Neural Network (CNN), which is trained simultaneously with the subsequent GNN. Moreover, we incorporate local image quality metrics into the loss function to enhance the training procedure's robustness against low-quality regions in the training images. We demonstrate the superiority of our proposed method using a clinical ex vivo dataset consisting of 51 HSI images from 30 patients. Despite the limited dataset, the GNN-based model significantly outperforms context-agnostic approaches, accurately distinguishing between healthy and tumor tissues, even in images from previously unseen patients. Furthermore, we show that our carefully designed loss function, accounting for local image quality, results in additional improvements. Our findings demonstrate that context-aware GNN algorithms can robustly find tumor demarcations on HSI images, ultimately contributing to better surgery success and patient outcome.

{{</citation>}}


### (102/122) Tiny-VBF: Resource-Efficient Vision Transformer based Lightweight Beamformer for Ultrasound Single-Angle Plane Wave Imaging (Abdul Rahoof et al., 2023)

{{<citation>}}

Abdul Rahoof, Vivek Chaturvedi, Mahesh Raveendranatha Panicker, Muhammad Shafique. (2023)  
**Tiny-VBF: Resource-Efficient Vision Transformer based Lightweight Beamformer for Ultrasound Single-Angle Plane Wave Imaging**  

---
Primary Category: eess.IV  
Categories: cs-LG, eess-IV, eess.IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.12082v1)  

---


**ABSTRACT**  
Accelerating compute intensive non-real-time beam-forming algorithms in ultrasound imaging using deep learning architectures has been gaining momentum in the recent past. Nonetheless, the complexity of the state-of-the-art deep learning techniques poses challenges for deployment on resource-constrained edge devices. In this work, we propose a novel vision transformer based tiny beamformer (Tiny-VBF), which works on the raw radio-frequency channel data acquired through single-angle plane wave insonification. The output of our Tiny-VBF provides fast envelope detection requiring very low frame rate, i.e. 0.34 GOPs/Frame for a frame size of 368 x 128 in comparison to the state-of-the-art deep learning models. It also exhibited an 8% increase in contrast and gains of 5% and 33% in axial and lateral resolution respectively when compared to Tiny-CNN on in-vitro dataset. Additionally, our model showed a 4.2% increase in contrast and gains of 4% and 20% in axial and lateral resolution respectively when compared against conventional Delay-and-Sum (DAS) beamformer. We further propose an accelerator architecture and implement our Tiny-VBF model on a Zynq UltraScale+ MPSoC ZCU104 FPGA using a hybrid quantization scheme with 50% less resource consumption compared to the floating-point implementation, while preserving the image quality.

{{</citation>}}


### (103/122) Double-Condensing Attention Condenser: Leveraging Attention in Deep Learning to Detect Skin Cancer from Skin Lesion Images (Chi-en Amy Tai et al., 2023)

{{<citation>}}

Chi-en Amy Tai, Elizabeth Janes, Chris Czarnecki, Alexander Wong. (2023)  
**Double-Condensing Attention Condenser: Leveraging Attention in Deep Learning to Detect Skin Cancer from Skin Lesion Images**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2311.11656v1)  

---


**ABSTRACT**  
Skin cancer is the most common type of cancer in the United States and is estimated to affect one in five Americans. Recent advances have demonstrated strong performance on skin cancer detection, as exemplified by state of the art performance in the SIIM-ISIC Melanoma Classification Challenge; however these solutions leverage ensembles of complex deep neural architectures requiring immense storage and compute costs, and therefore may not be tractable. A recent movement for TinyML applications is integrating Double-Condensing Attention Condensers (DC-AC) into a self-attention neural network backbone architecture to allow for faster and more efficient computation. This paper explores leveraging an efficient self-attention structure to detect skin cancer in skin lesion images and introduces a deep neural network design with DC-AC customized for skin cancer detection from skin lesion images. The final model is publicly available as a part of a global open-source initiative dedicated to accelerating advancement in machine learning to aid clinicians in the fight against cancer.

{{</citation>}}


### (104/122) Energy efficiency in Edge TPU vs. embedded GPU for computer-aided medical imaging segmentation and classification (José María Rodríguez Corral et al., 2023)

{{<citation>}}

José María Rodríguez Corral, Javier Civit-Masot, Francisco Luna-Perejón, Ignacio Díaz-Cano, Arturo Morgado-Estévez, Manuel Domínguez-Morales. (2023)  
**Energy efficiency in Edge TPU vs. embedded GPU for computer-aided medical imaging segmentation and classification**  

---
Primary Category: eess.IV  
Categories: cs-AI, cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2311.12876v1)  

---


**ABSTRACT**  
In this work, we evaluate the energy usage of fully embedded medical diagnosis aids based on both segmentation and classification of medical images implemented on Edge TPU and embedded GPU processors. We use glaucoma diagnosis based on color fundus images as an example to show the possibility of performing segmentation and classification in real time on embedded boards and to highlight the different energy requirements of the studied implementations.   Several other works develop the use of segmentation and feature extraction techniques to detect glaucoma, among many other pathologies, with deep neural networks. Memory limitations and low processing capabilities of embedded accelerated systems (EAS) limit their use for deep network-based system training. However, including specific acceleration hardware, such as NVIDIA's Maxwell GPU or Google's Edge TPU, enables them to perform inferences using complex pre-trained networks in very reasonable times.   In this study, we evaluate the timing and energy performance of two EAS equipped with Machine Learning (ML) accelerators executing an example diagnostic tool developed in a previous work. For optic disc (OD) and cup (OC) segmentation, the obtained prediction times per image are under 29 and 43 ms using Edge TPUs and Maxwell GPUs, respectively. Prediction times for the classification subsystem are lower than 10 and 14 ms for Edge TPUs and Maxwell GPUs, respectively. Regarding energy usage, in approximate terms, for OD segmentation Edge TPUs and Maxwell GPUs use 38 and 190 mJ per image, respectively. For fundus classification, Edge TPUs and Maxwell GPUs use 45 and 70 mJ, respectively.

{{</citation>}}


### (105/122) Liver Tumor Prediction with Advanced Attention Mechanisms Integrated into a Depth-Based Variant Search Algorithm (P. Kalaiselvi et al., 2023)

{{<citation>}}

P. Kalaiselvi, S. Anusuya. (2023)  
**Liver Tumor Prediction with Advanced Attention Mechanisms Integrated into a Depth-Based Variant Search Algorithm**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2311.11520v1)  

---


**ABSTRACT**  
In recent days, Deep Learning (DL) techniques have become an emerging transformation in the field of machine learning, artificial intelligence, computer vision, and so on. Subsequently, researchers and industries have been highly endorsed in the medical field, predicting and controlling diverse diseases at specific intervals. Liver tumor prediction is a vital chore in analyzing and treating liver diseases. This paper proposes a novel approach for predicting liver tumors using Convolutional Neural Networks (CNN) and a depth-based variant search algorithm with advanced attention mechanisms (CNN-DS-AM). The proposed work aims to improve accuracy and robustness in diagnosing and treating liver diseases. The anticipated model is assessed on a Computed Tomography (CT) scan dataset containing both benign and malignant liver tumors. The proposed approach achieved high accuracy in predicting liver tumors, outperforming other state-of-the-art methods. Additionally, advanced attention mechanisms were incorporated into the CNN model to enable the identification and highlighting of regions of the CT scans most relevant to predicting liver tumors. The results suggest that incorporating attention mechanisms and a depth-based variant search algorithm into the CNN model is a promising approach for improving the accuracy and robustness of liver tumor prediction. It can assist radiologists in their diagnosis and treatment planning. The proposed system achieved a high accuracy of 95.5% in predicting liver tumors, outperforming other state-of-the-art methods.

{{</citation>}}


## cs.HC (2)



### (106/122) Bridging Learnersourcing and AI: Exploring the Dynamics of Student-AI Collaborative Feedback Generation (Anjali Singh et al., 2023)

{{<citation>}}

Anjali Singh, Christopher Brooks, Xu Wang, Warren Li, Juho Kim, Deepti Pandey. (2023)  
**Bridging Learnersourcing and AI: Exploring the Dynamics of Student-AI Collaborative Feedback Generation**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2311.12148v1)  

---


**ABSTRACT**  
This paper explores the space of optimizing feedback mechanisms in complex domains, such as data science, by combining two prevailing approaches: Artificial Intelligence (AI) and learnersourcing. Towards addressing the challenges posed by each approach, this work compares traditional learnersourcing with an AI-supported approach. We report on the results of a randomized controlled experiment conducted with 72 Master's level students in a data visualization course, comparing two conditions: students writing hints independently versus revising hints generated by GPT-4. The study aimed to evaluate the quality of learnersourced hints, examine the impact of student performance on hint quality, gauge learner preference for writing hints with or without AI support, and explore the potential of the student-AI collaborative exercise in fostering critical thinking about LLMs. Based on our findings, we provide insights for designing learnersourcing activities leveraging AI support and optimizing students' learning as they interact with LLMs.

{{</citation>}}


### (107/122) Perspectives on Privacy in the Post-Roe Era: A Mixed-Methods of Machine Learning and Qualitative Analyses of Tweets (Yawen Guo et al., 2023)

{{<citation>}}

Yawen Guo, Rachael Zehrung, Katie Genuario, Xuan Lu, Qiaozhu Mei, Yunan Chen, Kai Zheng. (2023)  
**Perspectives on Privacy in the Post-Roe Era: A Mixed-Methods of Machine Learning and Qualitative Analyses of Tweets**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2311.11486v1)  

---


**ABSTRACT**  
Abortion is a controversial topic that has long been debated in the US. With the recent Supreme Court decision to overturn Roe v. Wade, access to safe and legal reproductive care is once again in the national spotlight. A key issue central to this debate is patient privacy, as in the post-HITECH Act era it has become easier for medical records to be electronically accessed and shared. This study analyzed a large Twitter dataset from May to December 2022 to examine the public's reactions to Roe v. Wade's overruling and its implications for privacy. Using a mixed-methods approach consisting of computational and qualitative content analysis, we found a wide range of concerns voiced from the confidentiality of patient-physician information exchange to medical records being shared without patient consent. These findings may inform policy making and healthcare industry practices concerning medical privacy related to reproductive rights and women's health.

{{</citation>}}


## cs.SE (3)



### (108/122) An Empirical Study of Self-Admitted Technical Debt in Machine Learning Software (Aaditya Bhatia et al., 2023)

{{<citation>}}

Aaditya Bhatia, Foutse Khomh, Bram Adams, Ahmed E Hassan. (2023)  
**An Empirical Study of Self-Admitted Technical Debt in Machine Learning Software**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2311.12019v1)  

---


**ABSTRACT**  
The emergence of open-source ML libraries such as TensorFlow and Google Auto ML has enabled developers to harness state-of-the-art ML algorithms with minimal overhead. However, during this accelerated ML development process, said developers may often make sub-optimal design and implementation decisions, leading to the introduction of technical debt that, if not addressed promptly, can have a significant impact on the quality of the ML-based software. Developers frequently acknowledge these sub-optimal design and development choices through code comments during software development. These comments, which often highlight areas requiring additional work or refinement in the future, are known as self-admitted technical debt (SATD). This paper aims to investigate SATD in ML code by analyzing 318 open-source ML projects across five domains, along with 318 non-ML projects. We detected SATD in source code comments throughout the different project snapshots, conducted a manual analysis of the identified SATD sample to comprehend the nature of technical debt in the ML code, and performed a survival analysis of the SATD to understand the evolution of such debts. We observed: i) Machine learning projects have a median percentage of SATD that is twice the median percentage of SATD in non-machine learning projects. ii) ML pipeline components for data preprocessing and model generation logic are more susceptible to debt than model validation and deployment components. iii) SATDs appear in ML projects earlier in the development process compared to non-ML projects. iv) Long-lasting SATDs are typically introduced during extensive code changes that span multiple files exhibiting low complexity.

{{</citation>}}


### (109/122) On the Potential and Limitations of Few-Shot In-Context Learning to Generate Metamorphic Specifications for Tax Preparation Software (Dananjay Srinivas et al., 2023)

{{<citation>}}

Dananjay Srinivas, Rohan Das, Saeid Tizpaz-Niari, Ashutosh Trivedi, Maria Leonor Pacheco. (2023)  
**On the Potential and Limitations of Few-Shot In-Context Learning to Generate Metamorphic Specifications for Tax Preparation Software**  

---
Primary Category: cs.SE  
Categories: cs-CL, cs-SE, cs.SE  
Keywords: Few-Shot, Language Model  
[Paper Link](http://arxiv.org/abs/2311.11979v1)  

---


**ABSTRACT**  
Due to the ever-increasing complexity of income tax laws in the United States, the number of US taxpayers filing their taxes using tax preparation software (henceforth, tax software) continues to increase. According to the U.S. Internal Revenue Service (IRS), in FY22, nearly 50% of taxpayers filed their individual income taxes using tax software. Given the legal consequences of incorrectly filing taxes for the taxpayer, ensuring the correctness of tax software is of paramount importance. Metamorphic testing has emerged as a leading solution to test and debug legal-critical tax software due to the absence of correctness requirements and trustworthy datasets. The key idea behind metamorphic testing is to express the properties of a system in terms of the relationship between one input and its slightly metamorphosed twinned input. Extracting metamorphic properties from IRS tax publications is a tedious and time-consuming process. As a response, this paper formulates the task of generating metamorphic specifications as a translation task between properties extracted from tax documents - expressed in natural language - to a contrastive first-order logic form. We perform a systematic analysis on the potential and limitations of in-context learning with Large Language Models(LLMs) for this task, and outline a research agenda towards automating the generation of metamorphic specifications for tax preparation software.

{{</citation>}}


### (110/122) Metamorphic Testing of Image Captioning Systems via Reduction-based Transformations (Xiaoyuan Xie et al., 2023)

{{<citation>}}

Xiaoyuan Xie, Xingpeng Li, Songqiang Chen. (2023)  
**Metamorphic Testing of Image Captioning Systems via Reduction-based Transformations**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: AI, Image Captioning  
[Paper Link](http://arxiv.org/abs/2311.11791v1)  

---


**ABSTRACT**  
Recently, the Image Captioning (IC) technique has been widely applied to describe a given image in text form. However, IC systems can still produce incorrect captions and lead to misunderstandings. To tackle this problem, several methods have been proposed to test the IC systems. However, these approaches still rely on pre-annotated information and hence cannot really alleviate the oracle problem in the testing. Besides, they adopt AIGC techniques to create follow-up test images that may generate unrealistic images as test cases, which leads to meaningless testing results. Thirdly, existing methods have various restrictions on the eligibility of source test cases, and hence cannot fully utilize the given images to perform testing.   To tackle these issues, we propose REIC, which conducts metamorphic testing for IC systems with reduction-based transformations. Instead of relying on the pre-annotated information, we introduce a localization method to align the described objects in the caption with the corresponding objects in the test image and check whether each object in the caption retains or disappears after transformation. REIC does not artificially manipulate any objects and hence can effectively avoid generating unreal follow-up images. Besides, it eliminates the restrictions in the metamorphic transformation process, as well as decreases the ambiguity, and boosts the diversity among the follow-up test cases, which consequently enables testing to be performed on any test image, and reveals more distinct valid violations. Experimental results demonstrate that REIC can sufficiently leverage provided test images to generate follow-up cases of good reality, and effectively detect a great number of distinct violations, without the need for any pre-annotated information.

{{</citation>}}


## cs.RO (2)



### (111/122) GPT-4V(ision) for Robotics: Multimodal Task Planning from Human Demonstration (Naoki Wake et al., 2023)

{{<citation>}}

Naoki Wake, Atsushi Kanehira, Kazuhiro Sasabuchi, Jun Takamatsu, Katsushi Ikeuchi. (2023)  
**GPT-4V(ision) for Robotics: Multimodal Task Planning from Human Demonstration**  

---
Primary Category: cs.RO  
Categories: cs-CL, cs-CV, cs-RO, cs.RO  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2311.12015v1)  

---


**ABSTRACT**  
We introduce a pipeline that enhances a general-purpose Vision Language Model, GPT-4V(ision), by integrating observations of human actions to facilitate robotic manipulation. This system analyzes videos of humans performing tasks and creates executable robot programs that incorporate affordance insights. The computation starts by analyzing the videos with GPT-4V to convert environmental and action details into text, followed by a GPT-4-empowered task planner. In the following analyses, vision systems reanalyze the video with the task plan. Object names are grounded using an open-vocabulary object detector, while focus on the hand-object relation helps to detect the moment of grasping and releasing. This spatiotemporal grounding allows the vision systems to further gather affordance data (e.g., grasp type, way points, and body postures). Experiments across various scenarios demonstrate this method's efficacy in achieving real robots' operations from human demonstrations in a zero-shot manner. The prompts of GPT-4V/GPT-4 are available at this project page: https://microsoft.github.io/GPT4Vision-Robot-Manipulation-Prompts/

{{</citation>}}


### (112/122) Multi-Agent Strategy Explanations for Human-Robot Collaboration (Ravi Pandya et al., 2023)

{{<citation>}}

Ravi Pandya, Michelle Zhao, Changliu Liu, Reid Simmons, Henny Admoni. (2023)  
**Multi-Agent Strategy Explanations for Human-Robot Collaboration**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.11955v1)  

---


**ABSTRACT**  
As robots are deployed in human spaces, it's important that they are able to coordinate their actions with the people around them. Part of such coordination involves ensuring that people have a good understanding of how a robot will act in the environment. This can be achieved through explanations of the robot's policy. Much prior work in explainable AI and RL focuses on generating explanations for single-agent policies, but little has been explored in generating explanations for collaborative policies. In this work, we investigate how to generate multi-agent strategy explanations for human-robot collaboration. We formulate the problem using a generic multi-agent planner, show how to generate visual explanations through strategy-conditioned landmark states and generate textual explanations by giving the landmarks to an LLM. Through a user study, we find that when presented with explanations from our proposed framework, users are able to better explore the full space of strategies and collaborate more efficiently with new robot partners.

{{</citation>}}


## stat.ML (1)



### (113/122) Measuring and Mitigating Biases in Motor Insurance Pricing (Mulah Moriah et al., 2023)

{{<citation>}}

Mulah Moriah, Franck Vermet, Arthur Charpentier. (2023)  
**Measuring and Mitigating Biases in Motor Insurance Pricing**  

---
Primary Category: stat.ML  
Categories: cs-CY, cs-LG, stat-ML, stat.ML  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2311.11900v1)  

---


**ABSTRACT**  
The non-life insurance sector operates within a highly competitive and tightly regulated framework, confronting a pivotal juncture in the formulation of pricing strategies. Insurers are compelled to harness a range of statistical methodologies and available data to construct optimal pricing structures that align with the overarching corporate strategy while accommodating the dynamics of market competition. Given the fundamental societal role played by insurance, premium rates are subject to rigorous scrutiny by regulatory authorities. These rates must conform to principles of transparency, explainability, and ethical considerations. Consequently, the act of pricing transcends mere statistical calculations and carries the weight of strategic and societal factors. These multifaceted concerns may drive insurers to establish equitable premiums, taking into account various variables. For instance, regulations mandate the provision of equitable premiums, considering factors such as policyholder gender or mutualist group dynamics in accordance with respective corporate strategies. Age-based premium fairness is also mandated. In certain insurance domains, variables such as the presence of serious illnesses or disabilities are emerging as new dimensions for evaluating fairness. Regardless of the motivating factor prompting an insurer to adopt fairer pricing strategies for a specific variable, the insurer must possess the capability to define, measure, and ultimately mitigate any ethical biases inherent in its pricing practices while upholding standards of consistency and performance. This study seeks to provide a comprehensive set of tools for these endeavors and assess their effectiveness through practical application in the context of automobile insurance.

{{</citation>}}


## cs.SI (1)



### (114/122) Multilayer Quantile Graph for Multivariate Time Series Analysis and Dimensionality Reduction (Vanessa Freitas Silva et al., 2023)

{{<citation>}}

Vanessa Freitas Silva, Maria Eduarda Silva, Pedro Ribeiro, Fernando Silva. (2023)  
**Multilayer Quantile Graph for Multivariate Time Series Analysis and Dimensionality Reduction**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2311.11849v1)  

---


**ABSTRACT**  
In recent years, there has been a surge in the prevalence of high- and multi-dimensional temporal data across various scientific disciplines. These datasets are characterized by their vast size and challenging potential for analysis. Such data typically exhibit serial and cross-dependency and possess high dimensionality, thereby introducing additional complexities to conventional time series analysis methods. To address these challenges, a recent and complementary approach has emerged, known as network-based analysis methods for multivariate time series. In univariate settings, Quantile Graphs have been employed to capture temporal transition properties and reduce data dimensionality by mapping observations to a smaller set of sample quantiles.   To confront the increasingly prominent issue of high dimensionality, we propose an extension of Quantile Graphs into a multivariate variant, which we term "Multilayer Quantile Graphs". In this innovative mapping, each time series is transformed into a Quantile Graph, and inter-layer connections are established to link contemporaneous quantiles of pairwise series. This enables the analysis of dynamic transitions across multiple dimensions. In this study, we demonstrate the effectiveness of this new mapping using a synthetic multivariate time series dataset. We delve into the resulting network's topological structures, extract network features, and employ these features for original dataset analysis. Furthermore, we compare our results with a recent method from the literature. The resulting multilayer network offers a significant reduction in the dimensionality of the original data while capturing serial and cross-dimensional transitions. This approach facilitates the characterization and analysis of large multivariate time series datasets through network analysis techniques.

{{</citation>}}


## cs.IR (1)



### (115/122) Graph Variational Embedding Collaborative Filtering (Narges Sadat Fazeli Dehkordi et al., 2023)

{{<citation>}}

Narges Sadat Fazeli Dehkordi, Hadi Zare, Parham Moradi, Mahdi Jalili. (2023)  
**Graph Variational Embedding Collaborative Filtering**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2311.11824v1)  

---


**ABSTRACT**  
The customization of recommended content to users holds significant importance in enhancing user experiences across a wide spectrum of applications such as e-commerce, music, and shopping. Graph-based methods have achieved considerable performance by capturing user-item interactions. However, these methods tend to utilize randomly constructed embeddings in the dataset used for training the recommender, which lacks any user preferences. Here, we propose the concept of variational embeddings as a means of pre-training the recommender system to improve the feature propagation through the layers of graph convolutional networks (GCNs). The graph variational embedding collaborative filtering (GVECF) is introduced as a novel framework to incorporate representations learned through a variational graph auto-encoder which are embedded into a GCN-based collaborative filtering. This approach effectively transforms latent high-order user-item interactions into more trainable vectors, ultimately resulting in better performance in terms of recall and normalized discounted cumulative gain(NDCG) metrics. The experiments conducted on benchmark datasets demonstrate that our proposed method achieves up to 13.78% improvement in the recall over the test data.

{{</citation>}}


## cs.PL (1)



### (116/122) Refactoring Programs Using Large Language Models with Few-Shot Examples (Atsushi Shirafuji et al., 2023)

{{<citation>}}

Atsushi Shirafuji, Yusuke Oda, Jun Suzuki, Makoto Morishita, Yutaka Watanobe. (2023)  
**Refactoring Programs Using Large Language Models with Few-Shot Examples**  

---
Primary Category: cs.PL  
Categories: cs-AI, cs-CL, cs-PL, cs-SE, cs.PL  
Keywords: Few-Shot, GPT, GPT-3.5, Language Model  
[Paper Link](http://arxiv.org/abs/2311.11690v1)  

---


**ABSTRACT**  
A less complex and more straightforward program is a crucial factor that enhances its maintainability and makes writing secure and bug-free programs easier. However, due to its heavy workload and the risks of breaking the working programs, programmers are reluctant to do code refactoring, and thus, it also causes the loss of potential learning experiences. To mitigate this, we demonstrate the application of using a large language model (LLM), GPT-3.5, to suggest less complex versions of the user-written Python program, aiming to encourage users to learn how to write better programs. We propose a method to leverage the prompting with few-shot examples of the LLM by selecting the best-suited code refactoring examples for each target programming problem based on the prior evaluation of prompting with the one-shot example. The quantitative evaluation shows that 95.68% of programs can be refactored by generating 10 candidates each, resulting in a 17.35% reduction in the average cyclomatic complexity and a 25.84% decrease in the average number of lines after filtering only generated programs that are semantically correct. Furthermore, the qualitative evaluation shows outstanding capability in code formatting, while unnecessary behaviors such as deleting or translating comments are also observed.

{{</citation>}}


## eess.SP (1)



### (117/122) AIaaS for ORAN-based 6G Networks: Multi-time scale slice resource management with DRL (Suvidha Mhatre et al., 2023)

{{<citation>}}

Suvidha Mhatre, Ferran Adelantado, Kostas Ramantas, Christos Verikoukis. (2023)  
**AIaaS for ORAN-based 6G Networks: Multi-time scale slice resource management with DRL**  

---
Primary Category: eess.SP  
Categories: cs-SY, eess-SP, eess-SY, eess.SP  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.11668v1)  

---


**ABSTRACT**  
This paper addresses how to handle slice resources for 6G networks at different time scales in an architecture based on an open radio access network (ORAN). The proposed solution includes artificial intelligence (AI) at the edge of the network and applies two control-level loops to obtain optimal performance compared to other techniques. The ORAN facilitates programmable network architectures to support such multi-time scale management using AI approaches. The proposed algorithms analyze the maximum utilization of resources from slice performance to take decisions at the inter-slice level. Inter-slice intelligent agents work at a non-real-time level to reconfigure resources within various slices. Further than meeting the slice requirements, the intra-slice objective must also include the minimization of maximum resource utilization. This enables smart utilization of the resources within each slice without affecting slice performance. Here, each xApp that is an intra-slice agent aims at meeting the optimal QoS of the users, but at the same time, some inter-slice objectives should be included to coordinate intra- and inter-slice agents. This is done without penalizing the main intra-slice objective. All intelligent agents use deep reinforcement learning (DRL) algorithms to meet their objectives. We have presented results for enhanced mobile broadband (eMBB), ultra-reliable low latency (URLLC), and massive machine type communication (mMTC) slice categories.

{{</citation>}}


## cs.DC (2)



### (118/122) Can Datacenters Get the Power Needed to Meet the Explosive Demand for AI? (Liuzixuan Lin et al., 2023)

{{<citation>}}

Liuzixuan Lin, Rajini Wijayawardana, Varsha Rao, Hai Nguyen, Wedan Emmanuel Gnibga, Andrew A. Chien. (2023)  
**Can Datacenters Get the Power Needed to Meet the Explosive Demand for AI?**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs-SY, cs.DC, eess-SY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.11645v1)  

---


**ABSTRACT**  
The unprecedented rapid growth of computing demand for AI is projected to increase annual datacenter (DC) growth from 7.2% to 11.3%. Some power grids already face difficulty accommodating new datacenters, so the availability of grid capacity will constrain datacenter capacity needed by AI growth. We project AI datacenter demand for the next five years and consider several ``desperate measures'' -- grid policies that enable more DC growth by sacrificing new DC reliability while maintaining grid reliability.   In EirGrid where DC growth is already constrained, relaxing new DC reliability guarantees can increase the power available for DC growth to 1.6x--4.1x, sufficient to meet 5-year AI demand. Despite lower guarantees, we project the new datacenters will receive 99.6% power availability. In another challenged grid, Dominion, relaxing reliability guarantees increases available DC capacity similarly (1.5x--4.6x), and meets 70% of 5-year AI demand. Study of other grids -- SPP, CAISO, ERCOT -- shows that excess capacity exists for projected AI load growth.

{{</citation>}}


### (119/122) HexGen: Generative Inference of Foundation Model over Heterogeneous Decentralized Environment (Youhe Jiang et al., 2023)

{{<citation>}}

Youhe Jiang, Ran Yan, Xiaozhe Yao, Beidi Chen, Binhang Yuan. (2023)  
**HexGen: Generative Inference of Foundation Model over Heterogeneous Decentralized Environment**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.11514v1)  

---


**ABSTRACT**  
Serving foundation model inference is a pivotal component of contemporary AI applications, where this service is usually hosted in a centralized data center on a group of homogeneous high-performance GPUs. In this paper, we explore how to deploy such a service in a heterogeneous environment in terms of both computation capacity and network connection as an alternative to reduce the high inference cost. We propose HexGen, a distributed inference engine that supports asymmetric partitioning of the inference computation according to tensor model parallelism and pipeline parallelism. HexGen can be deployed with a set of different GPUs connected by a fully heterogeneous network, where the key technique contribution is a scheduling algorithm that allocates the asymmetric inference tasklets among these GPUs connected by different networks. We define the scheduling problem as a constrained optimization problem and further propose an efficient evolutionary algorithm to find the optimal allocation strategy. We conduct an extensive empirical study to evaluate the efficiency of HexGen by serving the state-of-the-art Llama-2 (70B) model. The experimental results suggest that HexGen can choose to achieve up to 2.3 times lower latency deadlines or tolerate up to 4 times more traffic request rates compared with the homogeneous baseline given the same budget. Our implementation is available at https://github.com/Relaxed-System-Lab/HexGen.

{{</citation>}}


## quant-ph (2)



### (120/122) Nav-Q: Quantum Deep Reinforcement Learning for Collision-Free Navigation of Self-Driving Cars (Akash Sinha et al., 2023)

{{<citation>}}

Akash Sinha, Antonio Macaluso, Matthias Klusch. (2023)  
**Nav-Q: Quantum Deep Reinforcement Learning for Collision-Free Navigation of Self-Driving Cars**  

---
Primary Category: quant-ph  
Categories: cs-AI, cs-LG, quant-ph, quant-ph  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.12875v1)  

---


**ABSTRACT**  
The challenge of collision-free navigation (CFN) for self-driving cars is an NP-hard problem addressed through Deep Reinforcement Learning (DRL). Despite the effectiveness of DRL methods, their application demands significant computing resources and prolonged training periods to establish a resilient agent. On the other hand, quantum reinforcement learning algorithms have recently demonstrated faster convergence and improved stability in simple, non-real-world environments. However, their application in the real-world CFN domain has not been explored, and their direct adaptation would require a quantum computing device onboard the vehicle for testing.   In this work, we propose Nav-Q, the first quantum-supported DRL algorithm for CFN of self-driving cars, that leverages quantum computation for improving the training performance without the requirement for onboard quantum hardware. Nav-Q is based on the actor-critic approach, where the critic is implemented using a hybrid quantum-classical algorithm suitable for near-term quantum devices. We assess the performance of Nav-Q using the CARLA driving simulator, a de facto standard benchmark for evaluating state-of-the-art DRL methods. Our empirical evaluations showcase that Nav-Q surpasses its classical counterpart not only in terms of training stability but also, in certain instances, with respect to the convergence rate when analyzing the Reward vs. Episode curve. This enhancement is accomplished without negatively impacting the learned policy by the agent. Furthermore, we assess Nav-Q in relation to effective dimension, unveiling that the incorporation of a quantum component results in a model possessing greater descriptive power compared to classical baselines. Finally, we evaluate the performance of Nav-Q using noisy quantum simulation, observing that the quantum noise enhances the exploratory tendencies of the agent during training.

{{</citation>}}


### (121/122) A Case for Synthesis of Recursive Quantum Unitary Programs (Haowei Deng et al., 2023)

{{<citation>}}

Haowei Deng, Runzhou Tao, Yuxiang Peng, Xiaodi Wu. (2023)  
**A Case for Synthesis of Recursive Quantum Unitary Programs**  

---
Primary Category: quant-ph  
Categories: cs-PL, quant-ph, quant-ph  
Keywords: AWS  
[Paper Link](http://arxiv.org/abs/2311.11503v1)  

---


**ABSTRACT**  
Quantum programs are notoriously difficult to code and verify due to unintuitive quantum knowledge associated with quantum programming. Automated tools relieving the tedium and errors associated with low-level quantum details would hence be highly desirable. In this paper, we initiate the study of program synthesis for quantum unitary programs that recursively define a family of unitary circuits for different input sizes, which are widely used in existing quantum programming languages. Specifically, we present QSynth, the first quantum program synthesis framework, including a new inductive quantum programming language, its specification, a sound logic for reasoning, and an encoding of the reasoning procedure into SMT instances. By leveraging existing SMT solvers, QSynth successfully synthesizes ten quantum unitary programs including quantum adder circuits, quantum eigenvalue inversion circuits and Quantum Fourier Transformation, which can be readily transpiled to executable programs on major quantum platforms, e.g., Q#, IBM Qiskit, and AWS Braket.

{{</citation>}}


## cs.DL (1)



### (122/122) Research assessment under debate: disentangling the interest around the DORA declaration on Twitter (E. Orduna-Malea et al., 2023)

{{<citation>}}

E. Orduna-Malea, N. Bautista-Puig. (2023)  
**Research assessment under debate: disentangling the interest around the DORA declaration on Twitter**  

---
Primary Category: cs.DL  
Categories: cs-DL, cs.DL  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2311.11609v2)  

---


**ABSTRACT**  
Much debate has been around the misapplication of metrics in research assessment. As a result of this concern, the Declaration on Research Assessment (DORA) was launched, an initiative that caused opposing viewpoints. However, the discussion topics about DORA have not been formally identified, especially in participatory environments outside the scholarly communication process, such as social networks. This paper contributes to that end by analyzing 20,717 DORA-related tweets published from 2015 to 2022. The results show an increasing volume of tweets, mainly promotional and informative, but with limited participation of users, either commenting or engaging with the tweets, generating a scarcely polarized conversation driven primarily by a few DORA promoters. While a varied list of discussion topics is found (especially "Open science and research assessment," "Academics career assessment & innovation," and "Journal Impact Factor"), the DORA debate appears as part of broader conversations (research evaluation, open science). Further studies are needed to check whether these results are restricted to Twitter or reveal more general patterns. The findings might interest the different evaluators and evaluated agents regarding their interests and concerns around the reforms in the research evaluation.

{{</citation>}}
