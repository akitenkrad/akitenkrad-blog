---
draft: false
title: "arXiv @ 2024.02.09"
date: 2024-02-09
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2024"]
menu:
  sidebar:
    name: "arXiv @ 2024.02.09"
    identifier: arxiv_20240209
    parent: 202402_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.LG (40)](#cslg-40)
- [cs.IR (4)](#csir-4)
- [cs.AI (10)](#csai-10)
- [cs.CV (20)](#cscv-20)
- [econ.TH (1)](#econth-1)
- [cs.CL (25)](#cscl-25)
- [eess.IV (3)](#eessiv-3)
- [cond-mat.mtrl-sci (1)](#cond-matmtrl-sci-1)
- [cs.SE (5)](#csse-5)
- [cs.RO (7)](#csro-7)
- [cs.SI (3)](#cssi-3)
- [cs.MA (1)](#csma-1)
- [cs.DL (2)](#csdl-2)
- [cs.HC (3)](#cshc-3)
- [cs.CY (1)](#cscy-1)
- [cs.DC (1)](#csdc-1)
- [cs.IT (1)](#csit-1)
- [cs.GT (1)](#csgt-1)
- [q-bio.BM (1)](#q-biobm-1)
- [cs.CE (1)](#csce-1)
- [physics.chem-ph (1)](#physicschem-ph-1)
- [cs.NI (1)](#csni-1)
- [astro-ph.IM (1)](#astro-phim-1)
- [stat.ML (1)](#statml-1)

## cs.LG (40)



### (1/135) Learning on Multimodal Graphs: A Survey (Ciyuan Peng et al., 2024)

{{<citation>}}

Ciyuan Peng, Jiayuan He, Feng Xia. (2024)  
**Learning on Multimodal Graphs: A Survey**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-GR, cs-LG, cs-SI, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2402.05322v1)  

---


**ABSTRACT**  
Multimodal data pervades various domains, including healthcare, social media, and transportation, where multimodal graphs play a pivotal role. Machine learning on multimodal graphs, referred to as multimodal graph learning (MGL), is essential for successful artificial intelligence (AI) applications. The burgeoning research in this field encompasses diverse graph data types and modalities, learning techniques, and application scenarios. This survey paper conducts a comparative analysis of existing works in multimodal graph learning, elucidating how multimodal learning is achieved across different graph types and exploring the characteristics of prevalent learning techniques. Additionally, we delineate significant applications of multimodal graph learning and offer insights into future directions in this domain. Consequently, this paper serves as a foundational resource for researchers seeking to comprehend existing MGL techniques and their applicability across diverse scenarios.

{{</citation>}}


### (2/135) Classifying spam emails using agglomerative hierarchical clustering and a topic-based approach (F. Janez-Martino et al., 2024)

{{<citation>}}

F. Janez-Martino, R. Alaiz-Rodriguez, V. Gonzalez-Castro, E. Fidalgo, E. Alegre. (2024)  
**Classifying spam emails using agglomerative hierarchical clustering and a topic-based approach**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2402.05296v1)  

---


**ABSTRACT**  
Spam emails are unsolicited, annoying and sometimes harmful messages which may contain malware, phishing or hoaxes. Unlike most studies that address the design of efficient anti-spam filters, we approach the spam email problem from a different and novel perspective. Focusing on the needs of cybersecurity units, we follow a topic-based approach for addressing the classification of spam email into multiple categories. We propose SPEMC-15K-E and SPEMC-15K-S, two novel datasets with approximately 15K emails each in English and Spanish, respectively, and we label them using agglomerative hierarchical clustering into 11 classes. We evaluate 16 pipelines, combining four text representation techniques -Term Frequency-Inverse Document Frequency (TF-IDF), Bag of Words, Word2Vec and BERT- and four classifiers: Support Vector Machine, N\"aive Bayes, Random Forest and Logistic Regression. Experimental results show that the highest performance is achieved with TF-IDF and LR for the English dataset, with a F1 score of 0.953 and an accuracy of 94.6%, and while for the Spanish dataset, TF-IDF with NB yields a F1 score of 0.945 and 98.5% accuracy. Regarding the processing time, TF-IDF with LR leads to the fastest classification, processing an English and Spanish spam email in and on average, respectively.

{{</citation>}}


### (3/135) Graph Neural Networks as Fast and High-fidelity Emulators for Finite-Element Ice Sheet Modeling (Maryam Rahnemoonfar et al., 2024)

{{<citation>}}

Maryam Rahnemoonfar, Younghyun Koo. (2024)  
**Graph Neural Networks as Fast and High-fidelity Emulators for Finite-Element Ice Sheet Modeling**  

---
Primary Category: cs.LG  
Categories: cs-CE, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2402.05291v1)  

---


**ABSTRACT**  
Although the finite element approach of the Ice-sheet and Sea-level System Model (ISSM) solves ice dynamics problems governed by Stokes equations quickly and accurately, such numerical modeling requires intensive computation on central processing units (CPU). In this study, we develop graph neural networks (GNN) as fast surrogate models to preserve the finite element structure of ISSM. Using the 20-year transient simulations in the Pine Island Glacier (PIG), we train and test three GNNs: graph convolutional network (GCN), graph attention network (GAT), and equivariant graph convolutional network (EGCN). These GNNs reproduce ice thickness and velocity with better accuracy than the classic convolutional neural network (CNN) and multi-layer perception (MLP). In particular, GNNs successfully capture the ice mass loss and acceleration induced by higher basal melting rates in the PIG. When our GNN emulators are implemented on graphic processing units (GPUs), they show up to 50 times faster computational time than the CPU-based ISSM simulation.

{{</citation>}}


### (4/135) Do Transformer World Models Give Better Policy Gradients? (Michel Ma et al., 2024)

{{<citation>}}

Michel Ma, Tianwei Ni, Clement Gehring, Pierluca D'Oro, Pierre-Luc Bacon. (2024)  
**Do Transformer World Models Give Better Policy Gradients?**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2402.05290v1)  

---


**ABSTRACT**  
A natural approach for reinforcement learning is to predict future rewards by unrolling a neural network world model, and to backpropagate through the resulting computational graph to learn a policy. However, this method often becomes impractical for long horizons since typical world models induce hard-to-optimize loss landscapes. Transformers are known to efficiently propagate gradients overlong horizons: could they be the solution to this problem? Surprisingly, we show that commonly-used transformer world models produce circuitous gradient paths, which can be detrimental to long-range policy gradients. To tackle this challenge, we propose a class of world models called Actions World Models (AWMs), designed to provide more direct routes for gradient propagation. We integrate such AWMs into a policy gradient framework that underscores the relationship between network architectures and the policy gradient updates they inherently represent. We demonstrate that AWMs can generate optimization landscapes that are easier to navigate even when compared to those from the simulator itself. This property allows transformer AWMs to produce better policies than competitive baselines in realistic long-horizon tasks.

{{</citation>}}


### (5/135) Analyzing Adversarial Inputs in Deep Reinforcement Learning (Davide Corsi et al., 2024)

{{<citation>}}

Davide Corsi, Guy Amir, Guy Katz, Alessandro Farinelli. (2024)  
**Analyzing Adversarial Inputs in Deep Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2402.05284v1)  

---


**ABSTRACT**  
In recent years, Deep Reinforcement Learning (DRL) has become a popular paradigm in machine learning due to its successful applications to real-world and complex systems. However, even the state-of-the-art DRL models have been shown to suffer from reliability concerns -- for example, their susceptibility to adversarial inputs, i.e., small and abundant input perturbations that can fool the models into making unpredictable and potentially dangerous decisions. This drawback limits the deployment of DRL systems in safety-critical contexts, where even a small error cannot be tolerated. In this work, we present a comprehensive analysis of the characterization of adversarial inputs, through the lens of formal verification. Specifically, we introduce a novel metric, the Adversarial Rate, to classify models based on their susceptibility to such perturbations, and present a set of tools and algorithms for its computation. Our analysis empirically demonstrates how adversarial inputs can affect the safety of a given DRL system with respect to such perturbations. Moreover, we analyze the behavior of these configurations to suggest several useful practices and guidelines to help mitigate the vulnerability of trained DRL networks.

{{</citation>}}


### (6/135) Exploring Hierarchical Classification Performance for Time Series Data: Dissimilarity Measures and Classifier Comparisons (Celal Alagoz, 2024)

{{<citation>}}

Celal Alagoz. (2024)  
**Exploring Hierarchical Classification Performance for Time Series Data: Dissimilarity Measures and Classifier Comparisons**  

---
Primary Category: cs.LG  
Categories: 62H30, I-5-2; I-5-3, cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2402.05275v1)  

---


**ABSTRACT**  
The comparative performance of hierarchical classification (HC) and flat classification (FC) methodologies in the realm of time series data analysis is investigated in this study. Dissimilarity measures, including Jensen-Shannon Distance (JSD), Task Similarity Distance (TSD), and Classifier Based Distance (CBD), are leveraged alongside various classifiers such as MINIROCKET, STSF, and SVM. A subset of datasets from the UCR archive, focusing on multi-class cases comprising more than two classes, is employed for analysis. A significant trend is observed wherein HC demonstrates significant superiority over FC when paired with MINIROCKET utilizing TSD, diverging from conventional understandings. Conversely, FC exhibits consistent dominance across all configurations when employing alternative classifiers such as STSF and SVM. Moreover, TSD is found to consistently outperform both CBD and JSD across nearly all scenarios, except in instances involving the STSF classifier where CBD showcases superior performance. This discrepancy underscores the nuanced nature of dissimilarity measures and emphasizes the importance of their tailored selection based on the dataset and classifier employed. Valuable insights into the dynamic interplay between classification methodologies and dissimilarity measures in the realm of time series data analysis are provided by these findings. By elucidating the performance variations across different configurations, a foundation is laid for refining classification methodologies and dissimilarity measures to optimize performance in diverse analytical scenarios. Furthermore, the need for continued research aimed at elucidating the underlying mechanisms driving classification performance in time series data analysis is underscored, with implications for enhancing predictive modeling and decision-making in various domains.

{{</citation>}}


### (7/135) Bellman Conformal Inference: Calibrating Prediction Intervals For Time Series (Zitong Yang et al., 2024)

{{<citation>}}

Zitong Yang, Emmanuel Candès, Lihua Lei. (2024)  
**Bellman Conformal Inference: Calibrating Prediction Intervals For Time Series**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2402.05203v1)  

---


**ABSTRACT**  
We introduce Bellman Conformal Inference (BCI), a framework that wraps around any time series forecasting models and provides calibrated prediction intervals. Unlike the existing methods, BCI is able to leverage multi-step ahead forecasts and explicitly optimize the average interval lengths by solving a one-dimensional stochastic control problem (SCP) at each time step. In particular, we use the dynamic programming algorithm to find the optimal policy for the SCP. We prove that BCI achieves long-term coverage under arbitrary distribution shifts and temporal dependence, even with poor multi-step ahead forecasts. We find empirically that BCI avoids uninformative intervals that have infinite lengths and generates substantially shorter prediction intervals on volatility forecasting problems when compared with existing methods.

{{</citation>}}


### (8/135) Towards Understanding Inductive Bias in Transformers: A View From Infinity (Itay Lavie et al., 2024)

{{<citation>}}

Itay Lavie, Guy Gur-Ari, Zohar Ringel. (2024)  
**Towards Understanding Inductive Bias in Transformers: A View From Infinity**  

---
Primary Category: cs.LG  
Categories: cond-mat-dis-nn, cs-LG, cs.LG, stat-ML  
Keywords: Bias, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2402.05173v1)  

---


**ABSTRACT**  
We study inductive bias in Transformers in the infinitely over-parameterized Gaussian process limit and argue transformers tend to be biased towards more permutation symmetric functions in sequence space. We show that the representation theory of the symmetric group can be used to give quantitative analytical predictions when the dataset is symmetric to permutations between tokens. We present a simplified transformer block and solve the model at the limit, including accurate predictions for the learning curves and network outputs. We show that in common setups, one can derive tight bounds in the form of a scaling law for the learnability as a function of the context length. Finally, we argue WikiText dataset, does indeed possess a degree of permutation symmetry.

{{</citation>}}


### (9/135) Opening the AI black box: program synthesis via mechanistic interpretability (Eric J. Michaud et al., 2024)

{{<citation>}}

Eric J. Michaud, Isaac Liao, Vedang Lad, Ziming Liu, Anish Mudide, Chloe Loughridge, Zifan Carl Guo, Tara Rezaei Kheirkhah, Mateja Vukelić, Max Tegmark. (2024)  
**Opening the AI black box: program synthesis via mechanistic interpretability**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2402.05110v1)  

---


**ABSTRACT**  
We present MIPS, a novel method for program synthesis based on automated mechanistic interpretability of neural networks trained to perform the desired task, auto-distilling the learned algorithm into Python code. We test MIPS on a benchmark of 62 algorithmic tasks that can be learned by an RNN and find it highly complementary to GPT-4: MIPS solves 32 of them, including 13 that are not solved by GPT-4 (which also solves 30). MIPS uses an integer autoencoder to convert the RNN into a finite state machine, then applies Boolean or integer symbolic regression to capture the learned algorithm. As opposed to large language models, this program synthesis technique makes no use of (and is therefore not limited by) human training data such as algorithms and code from GitHub. We discuss opportunities and challenges for scaling up this approach to make machine-learned models more interpretable and trustworthy.

{{</citation>}}


### (10/135) Hydragen: High-Throughput LLM Inference with Shared Prefixes (Jordan Juravsky et al., 2024)

{{<citation>}}

Jordan Juravsky, Bradley Brown, Ryan Ehrlich, Daniel Y. Fu, Christopher Ré, Azalia Mirhoseini. (2024)  
**Hydragen: High-Throughput LLM Inference with Shared Prefixes**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2402.05099v1)  

---


**ABSTRACT**  
Transformer-based large language models (LLMs) are now deployed to hundreds of millions of users. LLM inference is commonly performed on batches of sequences that share a prefix, such as few-shot examples or a chatbot system prompt. Decoding in this large-batch setting can be bottlenecked by the attention operation, which reads large key-value (KV) caches from memory and computes inefficient matrix-vector products for every sequence in the batch. In this work, we introduce Hydragen, a hardware-aware exact implementation of attention with shared prefixes. Hydragen computes attention over the shared prefix and unique suffixes separately. This decomposition enables efficient prefix attention by batching queries together across sequences, reducing redundant memory reads and enabling the use of hardware-friendly matrix multiplications. Our method can improve end-to-end LLM throughput by up to 32x against competitive baselines, with speedup growing with the batch size and shared prefix length. Hydragen also enables the use of very long shared contexts: with a high batch size, increasing the prefix length from 1K to 16K tokens decreases Hydragen throughput by less than 15%, while the throughput of baselines drops by over 90%. Hydragen generalizes beyond simple prefix-suffix decomposition and can be applied to tree-based prompt sharing patterns, allowing us to further reduce inference time on competitive programming problems by 55%.

{{</citation>}}


### (11/135) Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications (Boyi Wei et al., 2024)

{{<citation>}}

Boyi Wei, Kaixuan Huang, Yangsibo Huang, Tinghao Xie, Xiangyu Qi, Mengzhou Xia, Prateek Mittal, Mengdi Wang, Peter Henderson. (2024)  
**Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2402.05162v1)  

---


**ABSTRACT**  
Large language models (LLMs) show inherent brittleness in their safety mechanisms, as evidenced by their susceptibility to jailbreaking and even non-malicious fine-tuning. This study explores this brittleness of safety alignment by leveraging pruning and low-rank modifications. We develop methods to identify critical regions that are vital for safety guardrails, and that are disentangled from utility-relevant regions at both the neuron and rank levels. Surprisingly, the isolated regions we find are sparse, comprising about $3\%$ at the parameter level and $2.5\%$ at the rank level. Removing these regions compromises safety without significantly impacting utility, corroborating the inherent brittleness of the model's safety mechanisms. Moreover, we show that LLMs remain vulnerable to low-cost fine-tuning attacks even when modifications to the safety-critical regions are restricted. These findings underscore the urgent need for more robust safety strategies in LLMs.

{{</citation>}}


### (12/135) Causal Representation Learning from Multiple Distributions: A General Setting (Kun Zhang et al., 2024)

{{<citation>}}

Kun Zhang, Shaoan Xie, Ignavier Ng, Yujia Zheng. (2024)  
**Causal Representation Learning from Multiple Distributions: A General Setting**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2402.05052v1)  

---


**ABSTRACT**  
In many problems, the measured variables (e.g., image pixels) are just mathematical functions of the hidden causal variables (e.g., the underlying concepts or objects). For the purpose of making predictions in changing environments or making proper changes to the system, it is helpful to recover the hidden causal variables $Z_i$ and their causal relations represented by graph $\mathcal{G}_Z$. This problem has recently been known as causal representation learning. This paper is concerned with a general, completely nonparametric setting of causal representation learning from multiple distributions (arising from heterogeneous data or nonstationary time series), without assuming hard interventions behind distribution changes. We aim to develop general solutions in this fundamental case; as a by product, this helps see the unique benefit offered by other assumptions such as parametric causal models or hard interventions. We show that under the sparsity constraint on the recovered graph over the latent variables and suitable sufficient change conditions on the causal influences, interestingly, one can recover the moralized graph of the underlying directed acyclic graph, and the recovered latent variables and their relations are related to the underlying causal model in a specific, nontrivial way. In some cases, each latent variable can even be recovered up to component-wise transformations. Experimental results verify our theoretical claims.

{{</citation>}}


### (13/135) PAC Learnability under Explanation-Preserving Graph Perturbations (Xu Zheng et al., 2024)

{{<citation>}}

Xu Zheng, Farhad Shirani, Tianchun Wang, Shouwei Gao, Wenqian Dong, Wei Cheng, Dongsheng Luo. (2024)  
**PAC Learnability under Explanation-Preserving Graph Perturbations**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2402.05039v1)  

---


**ABSTRACT**  
Graphical models capture relations between entities in a wide range of applications including social networks, biology, and natural language processing, among others. Graph neural networks (GNN) are neural models that operate over graphs, enabling the model to leverage the complex relationships and dependencies in graph-structured data. A graph explanation is a subgraph which is an `almost sufficient' statistic of the input graph with respect to its classification label. Consequently, the classification label is invariant, with high probability, to perturbations of graph edges not belonging to its explanation subgraph. This work considers two methods for leveraging such perturbation invariances in the design and training of GNNs. First, explanation-assisted learning rules are considered. It is shown that the sample complexity of explanation-assisted learning can be arbitrarily smaller than explanation-agnostic learning. Next, explanation-assisted data augmentation is considered, where the training set is enlarged by artificially producing new training samples via perturbation of the non-explanation edges in the original training set. It is shown that such data augmentation methods may improve performance if the augmented data is in-distribution, however, it may also lead to worse sample complexity compared to explanation-agnostic learning rules if the augmented data is out-of-distribution. Extensive empirical evaluations are provided to verify the theoretical analysis.

{{</citation>}}


### (14/135) Simulated Overparameterization (Hanna Mazzawi et al., 2024)

{{<citation>}}

Hanna Mazzawi, Pranjal Awasthi, Xavi Gonzalvo, Srikumar Ramalingam. (2024)  
**Simulated Overparameterization**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2402.05033v1)  

---


**ABSTRACT**  
In this work, we introduce a novel paradigm called Simulated Overparametrization (SOP). SOP merges the computational efficiency of compact models with the advanced learning proficiencies of overparameterized models. SOP proposes a unique approach to model training and inference, where a model with a significantly larger number of parameters is trained in such a way that a smaller, efficient subset of these parameters is used for the actual computation during inference. Building upon this framework, we present a novel, architecture agnostic algorithm called "majority kernels", which seamlessly integrates with predominant architectures, including Transformer models. Majority kernels enables the simulated training of overparameterized models, resulting in performance gains across architectures and tasks. Furthermore, our approach adds minimal overhead to the cost incurred (wall clock time) at training time. The proposed approach shows strong performance on a wide variety of datasets and models, even outperforming strong baselines such as combinatorial optimization methods based on submodular optimization.

{{</citation>}}


### (15/135) Navigating Complexity: Toward Lossless Graph Condensation via Expanding Window Matching (Yuchen Zhang et al., 2024)

{{<citation>}}

Yuchen Zhang, Tianle Zhang, Kai Wang, Ziyao Guo, Yuxuan Liang, Xavier Bresson, Wei Jin, Yang You. (2024)  
**Navigating Complexity: Toward Lossless Graph Condensation via Expanding Window Matching**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI, GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2402.05011v1)  

---


**ABSTRACT**  
Graph condensation aims to reduce the size of a large-scale graph dataset by synthesizing a compact counterpart without sacrificing the performance of Graph Neural Networks (GNNs) trained on it, which has shed light on reducing the computational cost for training GNNs. Nevertheless, existing methods often fall short of accurately replicating the original graph for certain datasets, thereby failing to achieve the objective of lossless condensation. To understand this phenomenon, we investigate the potential reasons and reveal that the previous state-of-the-art trajectory matching method provides biased and restricted supervision signals from the original graph when optimizing the condensed one. This significantly limits both the scale and efficacy of the condensed graph. In this paper, we make the first attempt toward \textit{lossless graph condensation} by bridging the previously neglected supervision signals. Specifically, we employ a curriculum learning strategy to train expert trajectories with more diverse supervision signals from the original graph, and then effectively transfer the information into the condensed graph with expanding window matching. Moreover, we design a loss function to further extract knowledge from the expert trajectories. Theoretical analysis justifies the design of our method and extensive experiments verify its superiority across different datasets. Code is released at https://github.com/NUS-HPC-AI-Lab/GEOM.

{{</citation>}}


### (16/135) PriorBoost: An Adaptive Algorithm for Learning from Aggregate Responses (Adel Javanmard et al., 2024)

{{<citation>}}

Adel Javanmard, Matthew Fahrbach, Vahab Mirrokni. (2024)  
**PriorBoost: An Adaptive Algorithm for Learning from Aggregate Responses**  

---
Primary Category: cs.LG  
Categories: cs-DS, cs-LG, cs.LG  
Keywords: GLM  
[Paper Link](http://arxiv.org/abs/2402.04987v1)  

---


**ABSTRACT**  
This work studies algorithms for learning from aggregate responses. We focus on the construction of aggregation sets (called bags in the literature) for event-level loss functions. We prove for linear regression and generalized linear models (GLMs) that the optimal bagging problem reduces to one-dimensional size-constrained $k$-means clustering. Further, we theoretically quantify the advantage of using curated bags over random bags. We then propose the PriorBoost algorithm, which adaptively forms bags of samples that are increasingly homogeneous with respect to (unobserved) individual responses to improve model quality. We study label differential privacy for aggregate learning, and we also provide extensive experiments showing that PriorBoost regularly achieves optimal model quality for event-level predictions, in stark contrast to non-adaptive algorithms.

{{</citation>}}


### (17/135) Beyond explaining: XAI-based Adaptive Learning with SHAP Clustering for Energy Consumption Prediction (Tobias Clement et al., 2024)

{{<citation>}}

Tobias Clement, Hung Truong Thanh Nguyen, Nils Kemmerzell, Mohamed Abdelaal, Davor Stjelja. (2024)  
**Beyond explaining: XAI-based Adaptive Learning with SHAP Clustering for Energy Consumption Prediction**  

---
Primary Category: cs.LG  
Categories: cs-DB, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2402.04982v1)  

---


**ABSTRACT**  
This paper presents an approach integrating explainable artificial intelligence (XAI) techniques with adaptive learning to enhance energy consumption prediction models, with a focus on handling data distribution shifts. Leveraging SHAP clustering, our method provides interpretable explanations for model predictions and uses these insights to adaptively refine the model, balancing model complexity with predictive performance. We introduce a three-stage process: (1) obtaining SHAP values to explain model predictions, (2) clustering SHAP values to identify distinct patterns and outliers, and (3) refining the model based on the derived SHAP clustering characteristics. Our approach mitigates overfitting and ensures robustness in handling data distribution shifts. We evaluate our method on a comprehensive dataset comprising energy consumption records of buildings, as well as two additional datasets to assess the transferability of our approach to other domains, regression, and classification problems. Our experiments demonstrate the effectiveness of our approach in both task types, resulting in improved predictive performance and interpretable model explanations.

{{</citation>}}


### (18/135) Two Trades is not Baffled: Condensing Graph via Crafting Rational Gradient Matching (Tianle Zhang et al., 2024)

{{<citation>}}

Tianle Zhang, Yuchen Zhang, Kun Wang, Kai Wang, Beining Yang, Kaipeng Zhang, Wenqi Shao, Ping Liu, Joey Tianyi Zhou, Yang You. (2024)  
**Two Trades is not Baffled: Condensing Graph via Crafting Rational Gradient Matching**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2402.04924v2)  

---


**ABSTRACT**  
Training on large-scale graphs has achieved remarkable results in graph representation learning, but its cost and storage have raised growing concerns. As one of the most promising directions, graph condensation methods address these issues by employing gradient matching, aiming to condense the full graph into a more concise yet information-rich synthetic set. Though encouraging, these strategies primarily emphasize matching directions of the gradients, which leads to deviations in the training trajectories. Such deviations are further magnified by the differences between the condensation and evaluation phases, culminating in accumulated errors, which detrimentally affect the performance of the condensed graphs. In light of this, we propose a novel graph condensation method named \textbf{C}raf\textbf{T}ing \textbf{R}ationa\textbf{L} trajectory (\textbf{CTRL}), which offers an optimized starting point closer to the original dataset's feature distribution and a more refined strategy for gradient matching. Theoretically, CTRL can effectively neutralize the impact of accumulated errors on the performance of condensed graphs. We provide extensive experiments on various graph datasets and downstream tasks to support the effectiveness of CTRL. Code is released at https://github.com/NUS-HPC-AI-Lab/CTRL.

{{</citation>}}


### (19/135) L4Q: Parameter Efficient Quantization-Aware Training on Large Language Models via LoRA-wise LSQ (Hyesung Jeon et al., 2024)

{{<citation>}}

Hyesung Jeon, Yulhwa Kim, Jae-joon Kim. (2024)  
**L4Q: Parameter Efficient Quantization-Aware Training on Large Language Models via LoRA-wise LSQ**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: LLaMA, Language Model, QA, Quantization  
[Paper Link](http://arxiv.org/abs/2402.04902v1)  

---


**ABSTRACT**  
Post-training quantization (PTQ) and quantization-aware training (QAT) methods are gaining popularity in mitigating the high memory and computational costs associated with Large Language Models (LLMs). In resource-constrained scenarios, PTQ, with its reduced training overhead, is often preferred over QAT, despite the latter's potential for higher accuracy. Meanwhile, parameter-efficient fine-tuning (PEFT) methods like low-rank adaptation (LoRA) have been introduced, and recent efforts have explored quantization-aware PEFT techniques. However, these approaches may lack generality due to their reliance on the pre-quantized model's configuration. Their effectiveness may be compromised by non-linearly quantized or mixed-precision weights, and the retraining of specific quantization parameters might impede optimal performance. To address these challenges, we propose L4Q, an algorithm for parameter-efficient quantization-aware training. L4Q leverages LoRA-wise learned quantization step size for LLMs, aiming to enhance generality. The simultaneous quantization-and-fine-tuning process of L4Q is applicable to high-precision models, yielding linearly quantized weights with superior accuracy. Our experiments, conducted on the LLaMA and LLaMA2 model families using an instructional dataset, showcase L4Q's capabilities in language comprehension and few-shot in-context learning, achieving sub-4-bit precision while maintaining comparable training times to applying PEFT on a quantized model.

{{</citation>}}


### (20/135) Learning by Doing: An Online Causal Reinforcement Learning Framework with Causal-Aware Policy (Ruichu Cai et al., 2024)

{{<citation>}}

Ruichu Cai, Siyang Huang, Jie Qiao, Wei Chen, Yan Zeng, Keli Zhang, Fuchun Sun, Yang Yu, Zhifeng Hao. (2024)  
**Learning by Doing: An Online Causal Reinforcement Learning Framework with Causal-Aware Policy**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2402.04869v1)  

---


**ABSTRACT**  
As a key component to intuitive cognition and reasoning solutions in human intelligence, causal knowledge provides great potential for reinforcement learning (RL) agents' interpretability towards decision-making by helping reduce the searching space. However, there is still a considerable gap in discovering and incorporating causality into RL, which hinders the rapid development of causal RL. In this paper, we consider explicitly modeling the generation process of states with the causal graphical model, based on which we augment the policy. We formulate the causal structure updating into the RL interaction process with active intervention learning of the environment. To optimize the derived objective, we propose a framework with theoretical performance guarantees that alternates between two steps: using interventions for causal structure learning during exploration and using the learned causal structure for policy guidance during exploitation. Due to the lack of public benchmarks that allow direct intervention in the state space, we design the root cause localization task in our simulated fault alarm environment and then empirically show the effectiveness and robustness of the proposed method against state-of-the-art baselines. Theoretical analysis shows that our performance improvement attributes to the virtuous cycle of causal-guided policy learning and causal structure learning, which aligns with our experimental results.

{{</citation>}}


### (21/135) Multi-Patch Prediction: Adapting LLMs for Time Series Representation Learning (Yuxuan Bian et al., 2024)

{{<citation>}}

Yuxuan Bian, Xuan Ju, Jiangtong Li, Zhijian Xu, Dawei Cheng, Qiang Xu. (2024)  
**Multi-Patch Prediction: Adapting LLMs for Time Series Representation Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Language Model, Representation Learning, Time Series  
[Paper Link](http://arxiv.org/abs/2402.04852v1)  

---


**ABSTRACT**  
In this study, we present aLLM4TS, an innovative framework that adapts Large Language Models (LLMs) for time-series representation learning. Central to our approach is that we reconceive time-series forecasting as a self-supervised, multi-patch prediction task, which, compared to traditional mask-and-reconstruction methods, captures temporal dynamics in patch representations more effectively. Our strategy encompasses two-stage training: (i). a causal continual pre-training phase on various time-series datasets, anchored on next patch prediction, effectively syncing LLM capabilities with the intricacies of time-series data; (ii). fine-tuning for multi-patch prediction in the targeted time-series context. A distinctive element of our framework is the patch-wise decoding layer, which departs from previous methods reliant on sequence-level decoding. Such a design directly transposes individual patches into temporal sequences, thereby significantly bolstering the model's proficiency in mastering temporal patch-based representations. aLLM4TS demonstrates superior performance in several downstream tasks, proving its effectiveness in deriving temporal representations with enhanced transferability and marking a pivotal advancement in the adaptation of LLMs for time-series analysis.

{{</citation>}}


### (22/135) On the Completeness of Invariant Geometric Deep Learning Models (Zian Li et al., 2024)

{{<citation>}}

Zian Li, Xiyuan Wang, Shijia Kang, Muhan Zhang. (2024)  
**On the Completeness of Invariant Geometric Deep Learning Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2402.04836v1)  

---


**ABSTRACT**  
Invariant models, one important class of geometric deep learning models, are capable of generating meaningful geometric representations by leveraging informative geometric features. These models are characterized by their simplicity, good experimental results and computational efficiency. However, their theoretical expressive power still remains unclear, restricting a deeper understanding of the potential of such models. In this work, we concentrate on characterizing the theoretical expressiveness of invariant models. We first rigorously bound the expressiveness of the most classical invariant model, Vanilla DisGNN (message passing neural networks incorporating distance), restricting its unidentifiable cases to be only those highly symmetric geometric graphs. To break these corner cases' symmetry, we introduce a simple yet E(3)-complete invariant design by nesting Vanilla DisGNN, named GeoNGNN. Leveraging GeoNGNN as a theoretical tool, we for the first time prove the E(3)-completeness of three well-established geometric models: DimeNet, GemNet and SphereNet. Our results fill the gap in the theoretical power of invariant models, contributing to a rigorous and comprehensive understanding of their capabilities. Experimentally, GeoNGNN exhibits good inductive bias in capturing local environments, and achieves competitive results w.r.t. complicated models relying on high-order invariant/equivariant representations while exhibiting significantly faster computational speed.

{{</citation>}}


### (23/135) E(3)-Equivariant Mesh Neural Networks (Thuan Trang et al., 2024)

{{<citation>}}

Thuan Trang, Nhat Khang Ngo, Daniel Levy, Thieu N. Vo, Siamak Ravanbakhsh, Truong Son Hy. (2024)  
**E(3)-Equivariant Mesh Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2402.04821v1)  

---


**ABSTRACT**  
Triangular meshes are widely used to represent three-dimensional objects. As a result, many recent works have address the need for geometric deep learning on 3D mesh. However, we observe that the complexities in many of these architectures does not translate to practical performance, and simple deep models for geometric graphs are competitive in practice. Motivated by this observation, we minimally extend the update equations of E(n)-Equivariant Graph Neural Networks (EGNNs) (Satorras et al., 2021) to incorporate mesh face information, and further improve it to account for long-range interactions through hierarchy. The resulting architecture, Equivariant Mesh Neural Network (EMNN), outperforms other, more complicated equivariant methods on mesh tasks, with a fast run-time and no expensive pre-processing.

{{</citation>}}


### (24/135) Code as Reward: Empowering Reinforcement Learning with VLMs (David Venuto et al., 2024)

{{<citation>}}

David Venuto, Sami Nur Islam, Martin Klissarov, Doina Precup, Sherry Yang, Ankit Anand. (2024)  
**Code as Reward: Empowering Reinforcement Learning with VLMs**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Language Model, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2402.04764v1)  

---


**ABSTRACT**  
Pre-trained Vision-Language Models (VLMs) are able to understand visual concepts, describe and decompose complex tasks into sub-tasks, and provide feedback on task completion. In this paper, we aim to leverage these capabilities to support the training of reinforcement learning (RL) agents. In principle, VLMs are well suited for this purpose, as they can naturally analyze image-based observations and provide feedback (reward) on learning progress. However, inference in VLMs is computationally expensive, so querying them frequently to compute rewards would significantly slowdown the training of an RL agent. To address this challenge, we propose a framework named Code as Reward (VLM-CaR). VLM-CaR produces dense reward functions from VLMs through code generation, thereby significantly reducing the computational burden of querying the VLM directly. We show that the dense rewards generated through our approach are very accurate across a diverse set of discrete and continuous environments, and can be more effective in training RL policies than the original sparse environment rewards.

{{</citation>}}


### (25/135) Progressive Gradient Flow for Robust N:M Sparsity Training in Transformers (Abhimanyu Rajeshkumar Bambhaniya et al., 2024)

{{<citation>}}

Abhimanyu Rajeshkumar Bambhaniya, Amir Yazdanbakhsh, Suvinay Subramanian, Sheng-Chun Kao, Shivani Agrawal, Utku Evci, Tushar Krishna. (2024)  
**Progressive Gradient Flow for Robust N:M Sparsity Training in Transformers**  

---
Primary Category: cs.LG  
Categories: cs-AR, cs-LG, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2402.04744v1)  

---


**ABSTRACT**  
N:M Structured sparsity has garnered significant interest as a result of relatively modest overhead and improved efficiency. Additionally, this form of sparsity holds considerable appeal for reducing the memory footprint owing to their modest representation overhead. There have been efforts to develop training recipes for N:M structured sparsity, they primarily focus on low-sparsity regions ($\sim$50\%). Nonetheless, performance of models trained using these approaches tends to decline when confronted with high-sparsity regions ($>$80\%). In this work, we study the effectiveness of existing sparse training recipes at \textit{high-sparsity regions} and argue that these methods fail to sustain the model quality on par with low-sparsity regions. We demonstrate that the significant factor contributing to this disparity is the presence of elevated levels of induced noise in the gradient magnitudes. To mitigate this undesirable effect, we employ decay mechanisms to progressively restrict the flow of gradients towards pruned elements. Our approach improves the model quality by up to 2$\%$ and 5$\%$ in vision and language models at high sparsity regime, respectively. We also evaluate the trade-off between model accuracy and training compute cost in terms of FLOPs. At iso-training FLOPs, our method yields better performance compared to conventional sparse training recipes, exhibiting an accuracy improvement of up to 2$\%$. The source code is available at https://github.com/abhibambhaniya/progressive_gradient_flow_nm_sparsity.

{{</citation>}}


### (26/135) Incorporating Retrieval-based Causal Learning with Information Bottlenecks for Interpretable Graph Neural Networks (Jiahua Rao et al., 2024)

{{<citation>}}

Jiahua Rao, Jiancong Xie, Hanjing Lin, Shuangjia Zheng, Zhen Wang, Yuedong Yang. (2024)  
**Incorporating Retrieval-based Causal Learning with Information Bottlenecks for Interpretable Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2402.04710v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) have gained considerable traction for their capability to effectively process topological data, yet their interpretability remains a critical concern. Current interpretation methods are dominated by post-hoc explanations to provide a transparent and intuitive understanding of GNNs. However, they have limited performance in interpreting complicated subgraphs and can't utilize the explanation to advance GNN predictions. On the other hand, transparent GNN models are proposed to capture critical subgraphs. While such methods could improve GNN predictions, they usually don't perform well on explanations. Thus, it is desired for a new strategy to better couple GNN explanation and prediction. In this study, we have developed a novel interpretable causal GNN framework that incorporates retrieval-based causal learning with Graph Information Bottleneck (GIB) theory. The framework could semi-parametrically retrieve crucial subgraphs detected by GIB and compress the explanatory subgraphs via a causal module. The framework was demonstrated to consistently outperform state-of-the-art methods, and to achieve 32.71\% higher precision on real-world explanation scenarios with diverse explanation types. More importantly, the learned explanations were shown able to also improve GNN prediction performance.

{{</citation>}}


### (27/135) ApiQ: Finetuning of 2-Bit Quantized Large Language Model (Baohao Liao et al., 2024)

{{<citation>}}

Baohao Liao, Christof Monz. (2024)  
**ApiQ: Finetuning of 2-Bit Quantized Large Language Model**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2402.05147v1)  

---


**ABSTRACT**  
Memory-efficient finetuning of large language models (LLMs) has recently attracted huge attention with the increasing size of LLMs, primarily due to the constraints posed by GPU memory limitations and the comparable results of these methods with full finetuning. Despite the advancements, current strategies for memory-efficient finetuning, such as QLoRA, exhibit inconsistent performance across diverse bit-width quantizations and multifaceted tasks. This inconsistency largely stems from the detrimental impact of the quantization process on preserved knowledge, leading to catastrophic forgetting and undermining the utilization of pretrained models for finetuning purposes. In this work, we introduce a novel quantization framework named ApiQ, designed to restore the lost information from quantization by concurrently initializing LoRA components and quantizing the weights of LLMs. This approach ensures the maintenance of the original LLM's activation precision while mitigating the error propagation from shallower into deeper layers. Through comprehensive evaluations conducted on a spectrum of language tasks with various models, ApiQ demonstrably minimizes activation error during quantization. Consequently, it consistently achieves superior finetuning outcomes across various bit-widths of quantization.

{{</citation>}}


### (28/135) Compressing Deep Reinforcement Learning Networks with a Dynamic Structured Pruning Method for Autonomous Driving (Wensheng Su et al., 2024)

{{<citation>}}

Wensheng Su, Zhenni Li, Minrui Xu, Jiawen Kang, Dusit Niyato, Shengli Xie. (2024)  
**Compressing Deep Reinforcement Learning Networks with a Dynamic Structured Pruning Method for Autonomous Driving**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-RO, cs.LG  
Keywords: Pruning, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2402.05146v1)  

---


**ABSTRACT**  
Deep reinforcement learning (DRL) has shown remarkable success in complex autonomous driving scenarios. However, DRL models inevitably bring high memory consumption and computation, which hinders their wide deployment in resource-limited autonomous driving devices. Structured Pruning has been recognized as a useful method to compress and accelerate DRL models, but it is still challenging to estimate the contribution of a parameter (i.e., neuron) to DRL models. In this paper, we introduce a novel dynamic structured pruning approach that gradually removes a DRL model's unimportant neurons during the training stage. Our method consists of two steps, i.e. training DRL models with a group sparse regularizer and removing unimportant neurons with a dynamic pruning threshold. To efficiently train the DRL model with a small number of important neurons, we employ a neuron-importance group sparse regularizer. In contrast to conventional regularizers, this regularizer imposes a penalty on redundant groups of neurons that do not significantly influence the output of the DRL model. Furthermore, we design a novel structured pruning strategy to dynamically determine the pruning threshold and gradually remove unimportant neurons with a binary mask. Therefore, our method can remove not only redundant groups of neurons of the DRL model but also achieve high and robust performance. Experimental results show that the proposed method is competitive with existing DRL pruning methods on discrete control environments (i.e., CartPole-v1 and LunarLander-v2) and MuJoCo continuous environments (i.e., Hopper-v3 and Walker2D-v3). Specifically, our method effectively compresses $93\%$ neurons and $96\%$ weights of the DRL model in four challenging DRL environments with slight accuracy degradation.

{{</citation>}}


### (29/135) Open-Vocabulary Calibration for Vision-Language Models (Shuoyuan Wang et al., 2024)

{{<citation>}}

Shuoyuan Wang, Jindong Wang, Guoqing Wang, Bob Zhang, Kaiyang Zhou, Hongxin Wei. (2024)  
**Open-Vocabulary Calibration for Vision-Language Models**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2402.04655v1)  

---


**ABSTRACT**  
Vision-language models (VLMs) have emerged as formidable tools, showing their strong capability in handling various open-vocabulary tasks in image recognition, text-driven visual content generation, and visual chatbots, to name a few. In recent years, considerable efforts and resources have been devoted to adaptation methods for improving downstream performance of VLMs, particularly on parameter-efficient fine-tuning methods like prompt learning. However, a crucial aspect that has been largely overlooked is the confidence calibration problem in fine-tuned VLMs, which could greatly reduce reliability when deploying such models in the real world. This paper bridges the gap by systematically investigating the confidence calibration problem in the context of prompt learning and reveals that existing calibration methods are insufficient to address the problem, especially in the open-vocabulary setting. To solve the problem, we present a simple and effective approach called Distance-Aware Calibration (DAC), which is based on scaling the temperature using as guidance the distance between predicted text labels and base classes. The experiments with 7 distinct prompt learning methods applied across 11 diverse downstream datasets demonstrate the effectiveness of DAC, which achieves high efficacy without sacrificing the inference speed.

{{</citation>}}


### (30/135) Latent Plan Transformer: Planning as Latent Variable Inference (Deqian Kong et al., 2024)

{{<citation>}}

Deqian Kong, Dehong Xu, Minglu Zhao, Bo Pang, Jianwen Xie, Andrew Lizarraga, Yuhao Huang, Sirui Xie, Ying Nian Wu. (2024)  
**Latent Plan Transformer: Planning as Latent Variable Inference**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2402.04647v1)  

---


**ABSTRACT**  
In tasks aiming for long-term returns, planning becomes necessary. We study generative modeling for planning with datasets repurposed from offline reinforcement learning. Specifically, we identify temporal consistency in the absence of step-wise rewards as one key technical challenge. We introduce the Latent Plan Transformer (LPT), a novel model that leverages a latent space to connect a Transformer-based trajectory generator and the final return. LPT can be learned with maximum likelihood estimation on trajectory-return pairs. In learning, posterior sampling of the latent variable naturally gathers sub-trajectories to form a consistent abstraction despite the finite context. During test time, the latent variable is inferred from an expected return before policy execution, realizing the idea of planning as inference. It then guides the autoregressive policy throughout the episode, functioning as a plan. Our experiments demonstrate that LPT can discover improved decisions from suboptimal trajectories. It achieves competitive performance across several benchmarks, including Gym-Mujoco, Maze2D, and Connect Four, exhibiting capabilities of nuanced credit assignments, trajectory stitching, and adaptation to environmental contingencies. These results validate that latent variable inference can be a strong alternative to step-wise reward prompting.

{{</citation>}}


### (31/135) Domain Bridge: Generative model-based domain forensic for black-box models (Jiyi Zhang et al., 2024)

{{<citation>}}

Jiyi Zhang, Han Fang, Ee-Chien Chang. (2024)  
**Domain Bridge: Generative model-based domain forensic for black-box models**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI, ImageNet  
[Paper Link](http://arxiv.org/abs/2402.04640v1)  

---


**ABSTRACT**  
In forensic investigations of machine learning models, techniques that determine a model's data domain play an essential role, with prior work relying on large-scale corpora like ImageNet to approximate the target model's domain. Although such methods are effective in finding broad domains, they often struggle in identifying finer-grained classes within those domains. In this paper, we introduce an enhanced approach to determine not just the general data domain (e.g., human face) but also its specific attributes (e.g., wearing glasses). Our approach uses an image embedding model as the encoder and a generative model as the decoder. Beginning with a coarse-grained description, the decoder generates a set of images, which are then presented to the unknown target model. Successful classifications by the model guide the encoder to refine the description, which in turn, are used to produce a more specific set of images in the subsequent iteration. This iterative refinement narrows down the exact class of interest. A key strength of our approach lies in leveraging the expansive dataset, LAION-5B, on which the generative model Stable Diffusion is trained. This enlarges our search space beyond traditional corpora, such as ImageNet. Empirical results showcase our method's performance in identifying specific attributes of a model's input domain, paving the way for more detailed forensic analyses of deep learning models.

{{</citation>}}


### (32/135) Feature Distribution on Graph Topology Mediates the Effect of Graph Convolution: Homophily Perspective (Soo Yong Lee et al., 2024)

{{<citation>}}

Soo Yong Lee, Sunwoo Kim, Fanchen Bu, Jaemin Yoo, Jiliang Tang, Kijung Shin. (2024)  
**Feature Distribution on Graph Topology Mediates the Effect of Graph Convolution: Homophily Perspective**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2402.04621v1)  

---


**ABSTRACT**  
How would randomly shuffling feature vectors among nodes from the same class affect graph neural networks (GNNs)? The feature shuffle, intuitively, perturbs the dependence between graph topology and features (A-X dependence) for GNNs to learn from. Surprisingly, we observe a consistent and significant improvement in GNN performance following the feature shuffle. Having overlooked the impact of A-X dependence on GNNs, the prior literature does not provide a satisfactory understanding of the phenomenon. Thus, we raise two research questions. First, how should A-X dependence be measured, while controlling for potential confounds? Second, how does A-X dependence affect GNNs? In response, we (i) propose a principled measure for A-X dependence, (ii) design a random graph model that controls A-X dependence, (iii) establish a theory on how A-X dependence relates to graph convolution, and (iv) present empirical analysis on real-world graphs that aligns with the theory. We conclude that A-X dependence mediates the effect of graph convolution, such that smaller dependence improves GNN-based node classification.

{{</citation>}}


### (33/135) OIL-AD: An Anomaly Detection Framework for Sequential Decision Sequences (Chen Wang et al., 2024)

{{<citation>}}

Chen Wang, Sarah Erfani, Tansu Alpcan, Christopher Leckie. (2024)  
**OIL-AD: An Anomaly Detection Framework for Sequential Decision Sequences**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Anomaly Detection, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2402.04567v1)  

---


**ABSTRACT**  
Anomaly detection in decision-making sequences is a challenging problem due to the complexity of normality representation learning and the sequential nature of the task. Most existing methods based on Reinforcement Learning (RL) are difficult to implement in the real world due to unrealistic assumptions, such as having access to environment dynamics, reward signals, and online interactions with the environment. To address these limitations, we propose an unsupervised method named Offline Imitation Learning based Anomaly Detection (OIL-AD), which detects anomalies in decision-making sequences using two extracted behaviour features: action optimality and sequential association. Our offline learning model is an adaptation of behavioural cloning with a transformer policy network, where we modify the training process to learn a Q function and a state value function from normal trajectories. We propose that the Q function and the state value function can provide sufficient information about agents' behavioural data, from which we derive two features for anomaly detection. The intuition behind our method is that the action optimality feature derived from the Q function can differentiate the optimal action from others at each local state, and the sequential association feature derived from the state value function has the potential to maintain the temporal correlations between decisions (state-action pairs). Our experiments show that OIL-AD can achieve outstanding online anomaly detection performance with up to 34.8% improvement in F1 score over comparable baselines.

{{</citation>}}


### (34/135) Curvature-Informed SGD via General Purpose Lie-Group Preconditioners (Omead Pooladzandi et al., 2024)

{{<citation>}}

Omead Pooladzandi, Xi-Lin Li. (2024)  
**Curvature-Informed SGD via General Purpose Lie-Group Preconditioners**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2402.04553v1)  

---


**ABSTRACT**  
We present a novel approach to accelerate stochastic gradient descent (SGD) by utilizing curvature information obtained from Hessian-vector products or finite differences of parameters and gradients, similar to the BFGS algorithm. Our approach involves two preconditioners: a matrix-free preconditioner and a low-rank approximation preconditioner. We update both preconditioners online using a criterion that is robust to stochastic gradient noise and does not require line search or damping. To preserve the corresponding symmetry or invariance, our preconditioners are constrained to certain connected Lie groups. The Lie group's equivariance property simplifies the preconditioner fitting process, while its invariance property eliminates the need for damping, which is commonly required in second-order optimizers. As a result, the learning rate for parameter updating and the step size for preconditioner fitting are naturally normalized, and their default values work well in most scenarios. Our proposed approach offers a promising direction for improving the convergence of SGD with low computational overhead. We demonstrate that Preconditioned SGD (PSGD) outperforms SoTA on Vision, NLP, and RL tasks across multiple modern deep-learning architectures. We have provided code for reproducing toy and large scale experiments in this paper.

{{</citation>}}


### (35/135) Triplet Interaction Improves Graph Transformers: Accurate Molecular Graph Learning with Triplet Graph Transformers (Md Shamim Hussain et al., 2024)

{{<citation>}}

Md Shamim Hussain, Mohammed J. Zaki, Dharmashankar Subramanian. (2024)  
**Triplet Interaction Improves Graph Transformers: Accurate Molecular Graph Learning with Triplet Graph Transformers**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2402.04538v1)  

---


**ABSTRACT**  
Graph transformers typically lack direct pair-to-pair communication, instead forcing neighboring pairs to exchange information via a common node. We propose the Triplet Graph Transformer (TGT) that enables direct communication between two neighboring pairs in a graph via novel triplet attention and aggregation mechanisms. TGT is applied to molecular property prediction by first predicting interatomic distances from 2D graphs and then using these distances for downstream tasks. A novel three-stage training procedure and stochastic inference further improve training efficiency and model performance. Our model achieves new state-of-the-art (SOTA) results on open challenge benchmarks PCQM4Mv2 and OC20 IS2RE. We also obtain SOTA results on QM9, MOLPCBA, and LIT-PCBA molecular property prediction benchmarks via transfer learning. We also demonstrate the generality of TGT with SOTA results on the traveling salesman problem (TSP).

{{</citation>}}


### (36/135) SumRec: A Framework for Recommendation using Open-Domain Dialogue (Ryutaro Asahara et al., 2024)

{{<citation>}}

Ryutaro Asahara, Masaki Takahashi, Chiho Iwahashi, Michimasa Inaba. (2024)  
**SumRec: A Framework for Recommendation using Open-Domain Dialogue**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2402.04523v1)  

---


**ABSTRACT**  
Chat dialogues contain considerable useful information about a speaker's interests, preferences, and experiences.Thus, knowledge from open-domain chat dialogue can be used to personalize various systems and offer recommendations for advanced information.This study proposed a novel framework SumRec for recommending information from open-domain chat dialogue.The study also examined the framework using ChatRec, a newly constructed dataset for training and evaluation. To extract the speaker and item characteristics, the SumRec framework employs a large language model (LLM) to generate a summary of the speaker information from a dialogue and to recommend information about an item according to the type of user.The speaker and item information are then input into a score estimation model, generating a recommendation score.Experimental results show that the SumRec framework provides better recommendations than the baseline method of using dialogues and item descriptions in their original form. Our dataset and code is publicly available at https://github.com/Ryutaro-A/SumRec

{{</citation>}}


### (37/135) Online Cascade Learning for Efficient Inference over Streams (Lunyiu Nie et al., 2024)

{{<citation>}}

Lunyiu Nie, Zhimin Ding, Erdong Hu, Christopher Jermaine, Swarat Chaudhuri. (2024)  
**Online Cascade Learning for Efficient Inference over Streams**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2402.04513v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have a natural role in answering complex queries about data streams, but the high computational cost of LLM inference makes them infeasible in many such tasks. We propose online cascade learning, the first approach to addressing this challenge. The objective here is to learn a "cascade" of models, starting with lower-capacity models (such as logistic regressors) and ending with a powerful LLM, along with a deferral policy that determines the model that is used on a given input. We formulate the task of learning cascades online as an imitation-learning problem and give a no-regret algorithm for the problem. Experimental results across four benchmarks show that our method parallels LLMs in accuracy while cutting down inference costs by as much as 90%, underscoring its efficacy and adaptability in stream processing.

{{</citation>}}


### (38/135) The Fine-Grained Complexity of Gradient Computation for Training Large Language Models (Josh Alman et al., 2024)

{{<citation>}}

Josh Alman, Zhao Song. (2024)  
**The Fine-Grained Complexity of Gradient Computation for Training Large Language Models**  

---
Primary Category: cs.LG  
Categories: cs-CC, cs-CL, cs-DS, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2402.04497v1)  

---


**ABSTRACT**  
Large language models (LLMs) have made fundamental contributions over the last a few years. To train an LLM, one needs to alternatingly run `forward' computations and `backward' computations. The forward computation can be viewed as attention function evaluation, and the backward computation can be viewed as a gradient computation. In previous work by [Alman and Song, NeurIPS 2023], it was proved that the forward step can be performed in almost-linear time in certain parameter regimes, but that there is no truly sub-quadratic time algorithm in the remaining parameter regimes unless the popular hypothesis SETH is false. In this work, we show nearly identical results for the harder-seeming problem of computing the gradient of loss function of one layer attention network, and thus for the entire process of LLM training. This completely characterizes the fine-grained complexity of every step of LLM training.

{{</citation>}}


### (39/135) Grandmaster-Level Chess Without Search (Anian Ruoss et al., 2024)

{{<citation>}}

Anian Ruoss, Grégoire Delétang, Sourabh Medapati, Jordi Grau-Moya, Li Kevin Wenliang, Elliot Catt, John Reid, Tim Genewein. (2024)  
**Grandmaster-Level Chess Without Search**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, stat-ML  
Keywords: GPT, GPT-3.5  
[Paper Link](http://arxiv.org/abs/2402.04494v1)  

---


**ABSTRACT**  
The recent breakthrough successes in machine learning are mainly attributed to scale: namely large-scale attention-based architectures and datasets of unprecedented scale. This paper investigates the impact of training at scale for chess. Unlike traditional chess engines that rely on complex heuristics, explicit search, or a combination of both, we train a 270M parameter transformer model with supervised learning on a dataset of 10 million chess games. We annotate each board in the dataset with action-values provided by the powerful Stockfish 16 engine, leading to roughly 15 billion data points. Our largest model reaches a Lichess blitz Elo of 2895 against humans, and successfully solves a series of challenging chess puzzles, without any domain-specific tweaks or explicit search algorithms. We also show that our model outperforms AlphaZero's policy and value networks (without MCTS) and GPT-3.5-turbo-instruct. A systematic investigation of model and dataset size shows that strong chess performance only arises at sufficient scale. To validate our results, we perform an extensive series of ablations of design choices and hyperparameters.

{{</citation>}}


### (40/135) De-amplifying Bias from Differential Privacy in Language Model Fine-tuning (Sanjari Srivastava et al., 2024)

{{<citation>}}

Sanjari Srivastava, Piotr Mardziel, Zhikhun Zhang, Archana Ahlawat, Anupam Datta, John C Mitchell. (2024)  
**De-amplifying Bias from Differential Privacy in Language Model Fine-tuning**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-CY, cs-LG, cs.LG, stat-ME  
Keywords: Augmentation, Bias, Language Model  
[Paper Link](http://arxiv.org/abs/2402.04489v1)  

---


**ABSTRACT**  
Fairness and privacy are two important values machine learning (ML) practitioners often seek to operationalize in models. Fairness aims to reduce model bias for social/demographic sub-groups. Privacy via differential privacy (DP) mechanisms, on the other hand, limits the impact of any individual's training data on the resulting model. The trade-offs between privacy and fairness goals of trustworthy ML pose a challenge to those wishing to address both. We show that DP amplifies gender, racial, and religious bias when fine-tuning large language models (LLMs), producing models more biased than ones fine-tuned without DP. We find the cause of the amplification to be a disparity in convergence of gradients across sub-groups. Through the case of binary gender bias, we demonstrate that Counterfactual Data Augmentation (CDA), a known method for addressing bias, also mitigates bias amplification by DP. As a consequence, DP and CDA together can be used to fine-tune models while maintaining both fairness and privacy.

{{</citation>}}


## cs.IR (4)



### (41/135) Navigating the Knowledge Sea: Planet-scale answer retrieval using LLMs (Dipankar Sarkar, 2024)

{{<citation>}}

Dipankar Sarkar. (2024)  
**Navigating the Knowledge Sea: Planet-scale answer retrieval using LLMs**  

---
Primary Category: cs.IR  
Categories: cs-CL, cs-IR, cs-LG, cs.IR  
Keywords: GPT, GPT-4, Information Retrieval, Language Model  
[Paper Link](http://arxiv.org/abs/2402.05318v1)  

---


**ABSTRACT**  
Information retrieval is a rapidly evolving field of information retrieval, which is characterized by a continuous refinement of techniques and technologies, from basic hyperlink-based navigation to sophisticated algorithm-driven search engines. This paper aims to provide a comprehensive overview of the evolution of Information Retrieval Technology, with a particular focus on the role of Large Language Models (LLMs) in bridging the gap between traditional search methods and the emerging paradigm of answer retrieval. The integration of LLMs in the realms of response retrieval and indexing signifies a paradigm shift in how users interact with information systems. This paradigm shift is driven by the integration of large language models (LLMs) like GPT-4, which are capable of understanding and generating human-like text, thus enabling them to provide more direct and contextually relevant answers to user queries. Through this exploration, we seek to illuminate the technological milestones that have shaped this journey and the potential future directions in this rapidly changing field.

{{</citation>}}


### (42/135) Detecting Generated Native Ads in Conversational Search (Sebastian Schmidt et al., 2024)

{{<citation>}}

Sebastian Schmidt, Ines Zelch, Janek Bevendorff, Benno Stein, Matthias Hagen, Martin Potthast. (2024)  
**Detecting Generated Native Ads in Conversational Search**  

---
Primary Category: cs.IR  
Categories: cs-CL, cs-IR, cs.IR  
Keywords: Microsoft  
[Paper Link](http://arxiv.org/abs/2402.04889v1)  

---


**ABSTRACT**  
Conversational search engines such as YouChat and Microsoft Copilot use large language models (LLMs) to generate answers to queries. It is only a small step to also use this technology to generate and integrate advertising within these answers - instead of placing ads separately from the organic search results. This type of advertising is reminiscent of native advertising and product placement, both of which are very effective forms of subtle and manipulative advertising. It is likely that information seekers will be confronted with such use of LLM technology in the near future, especially when considering the high computational costs associated with LLMs, for which providers need to develop sustainable business models. This paper investigates whether LLMs can also be used as a countermeasure against generated native ads, i.e., to block them. For this purpose we compile a large dataset of ad-prone queries and of generated answers with automatically integrated ads to experiment with fine-tuned sentence transformers and state-of-the-art LLMs on the task of recognizing the ads. In our experiments sentence transformers achieve detection precision and recall values above 0.9, while the investigated LLMs struggle with the task.

{{</citation>}}


### (43/135) Multimodal Query Suggestion with Multi-Agent Reinforcement Learning from Human Feedback (Zheng Wang et al., 2024)

{{<citation>}}

Zheng Wang, Bingzheng Gan, Wei Shi. (2024)  
**Multimodal Query Suggestion with Multi-Agent Reinforcement Learning from Human Feedback**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Language Model, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2402.04867v1)  

---


**ABSTRACT**  
In the rapidly evolving landscape of information retrieval, search engines strive to provide more personalized and relevant results to users. Query suggestion systems play a crucial role in achieving this goal by assisting users in formulating effective queries. However, existing query suggestion systems mainly rely on textual inputs, potentially limiting user search experiences for querying images. In this paper, we introduce a novel Multimodal Query Suggestion (MMQS) task, which aims to generate query suggestions based on user query images to improve the intentionality and diversity of search results. We present the RL4Sugg framework, leveraging the power of Large Language Models (LLMs) with Multi-Agent Reinforcement Learning from Human Feedback to optimize the generation process. Through comprehensive experiments, we validate the effectiveness of RL4Sugg, demonstrating a 18% improvement compared to the best existing approach. Moreover, the MMQS has been transferred into real-world search engine products, which yield enhanced user engagement. Our research advances query suggestion systems and provides a new perspective on multimodal information retrieval.

{{</citation>}}


### (44/135) NORMY: Non-Uniform History Modeling for Open Retrieval Conversational Question Answering (Muhammad Shihab Rashid et al., 2024)

{{<citation>}}

Muhammad Shihab Rashid, Jannat Ara Meem, Vagelis Hristidis. (2024)  
**NORMY: Non-Uniform History Modeling for Open Retrieval Conversational Question Answering**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2402.04548v1)  

---


**ABSTRACT**  
Open Retrieval Conversational Question Answering (OrConvQA) answers a question given a conversation as context and a document collection. A typical OrConvQA pipeline consists of three modules: a Retriever to retrieve relevant documents from the collection, a Reranker to rerank them given the question and the context, and a Reader to extract an answer span. The conversational turns can provide valuable context to answer the final query. State-of-the-art OrConvQA systems use the same history modeling for all three modules of the pipeline. We hypothesize this as suboptimal. Specifically, we argue that a broader context is needed in the first modules of the pipeline to not miss relevant documents, while a narrower context is needed in the last modules to identify the exact answer span. We propose NORMY, the first unsupervised non-uniform history modeling pipeline which generates the best conversational history for each module. We further propose a novel Retriever for NORMY, which employs keyphrase extraction on the conversation history, and leverages passages retrieved in previous turns as additional context. We also created a new dataset for OrConvQA, by expanding the doc2dial dataset. We implemented various state-of-the-art history modeling techniques and comprehensively evaluated them separately for each module of the pipeline on three datasets: OR-QUAC, our doc2dial extension, and ConvMix. Our extensive experiments show that NORMY outperforms the state-of-the-art in the individual modules and in the end-to-end system.

{{</citation>}}


## cs.AI (10)



### (45/135) Three Pathways to Neurosymbolic Reinforcement Learning with Interpretable Model and Policy Networks (Peter Graf et al., 2024)

{{<citation>}}

Peter Graf, Patrick Emami. (2024)  
**Three Pathways to Neurosymbolic Reinforcement Learning with Interpretable Model and Policy Networks**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2402.05307v1)  

---


**ABSTRACT**  
Neurosymbolic AI combines the interpretability, parsimony, and explicit reasoning of classical symbolic approaches with the statistical learning of data-driven neural approaches. Models and policies that are simultaneously differentiable and interpretable may be key enablers of this marriage. This paper demonstrates three pathways to implementing such models and policies in a real-world reinforcement learning setting. Specifically, we study a broad class of neural networks that build interpretable semantics directly into their architecture. We reveal and highlight both the potential and the essential difficulties of combining logic, simulation, and learning. One lesson is that learning benefits from continuity and differentiability, but classical logic is discrete and non-differentiable. The relaxation to real-valued, differentiable representations presents a trade-off; the more learnable, the less interpretable. Another lesson is that using logic in the context of a numerical simulation involves a non-trivial mapping from raw (e.g., real-valued time series) simulation data to logical predicates. Some open questions this note exposes include: What are the limits of rule-based controllers, and how learnable are they? Do the differentiable interpretable approaches discussed here scale to large, complex, uncertain systems? Can we truly achieve interpretability? We highlight these and other themes across the three approaches.

{{</citation>}}


### (46/135) A Roadmap to Pluralistic Alignment (Taylor Sorensen et al., 2024)

{{<citation>}}

Taylor Sorensen, Jared Moore, Jillian Fisher, Mitchell Gordon, Niloofar Mireshghallah, Christopher Michael Rytting, Andre Ye, Liwei Jiang, Ximing Lu, Nouha Dziri, Tim Althoff, Yejin Choi. (2024)  
**A Roadmap to Pluralistic Alignment**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-IR, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2402.05070v1)  

---


**ABSTRACT**  
With increased power and prevalence of AI systems, it is ever more critical that AI systems are designed to serve all, i.e., people with diverse values and perspectives. However, aligning models to serve pluralistic human values remains an open research question. In this piece, we propose a roadmap to pluralistic alignment, specifically using language models as a test bed. We identify and formalize three possible ways to define and operationalize pluralism in AI systems: 1) Overton pluralistic models that present a spectrum of reasonable responses; 2) Steerably pluralistic models that can steer to reflect certain perspectives; and 3) Distributionally pluralistic models that are well-calibrated to a given population in distribution. We also propose and formalize three possible classes of pluralistic benchmarks: 1) Multi-objective benchmarks, 2) Trade-off steerable benchmarks, which incentivize models to steer to arbitrary trade-offs, and 3) Jury-pluralistic benchmarks which explicitly model diverse human ratings. We use this framework to argue that current alignment techniques may be fundamentally limited for pluralistic AI; indeed, we highlight empirical evidence, both from our own experiments and from other work, that standard alignment procedures might reduce distributional pluralism in models, motivating the need for further research on pluralistic alignment.

{{</citation>}}


### (47/135) How VADER is your AI? Towards a definition of artificial intelligence systems appropriate for regulation (Leonardo C. T. Bezerra et al., 2024)

{{<citation>}}

Leonardo C. T. Bezerra, Alexander E. I. Brownlee, Luana Ferraz Alvarenga, Renan Cipriano Moioli, Thais Vasconcelos Batista. (2024)  
**How VADER is your AI? Towards a definition of artificial intelligence systems appropriate for regulation**  

---
Primary Category: cs.AI  
Categories: I-2-0, cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2402.05048v1)  

---


**ABSTRACT**  
Artificial intelligence (AI) has driven many information and communication technology (ICT) breakthroughs. Nonetheless, the scope of ICT systems has expanded far beyond AI since the Turing test proposal. Critically, recent AI regulation proposals adopt AI definitions affecting ICT techniques, approaches, and systems that are not AI. In some cases, even works from mathematics, statistics, and engineering would be affected. Worryingly, AI misdefinitions are observed from Western societies to the Global South. In this paper, we propose a framework to score how \textit{validated as appropriately-defined for regulation} (VADER) an AI definition is. Our online, publicly-available VADER framework scores the coverage of premises that should underlie AI definitions for regulation, which aim to (i) reproduce principles observed in other successful technology regulations, and (ii) include all AI techniques and approaches while excluding non-AI works. Regarding the latter, our score is based on a dataset of representative AI, non-AI ICT, and non-ICT examples. We demonstrate our contribution by reviewing the AI regulation proposals of key players, namely the United States, United Kingdom, European Union, and Brazil. Importantly, none of the proposals assessed achieve the appropriateness score, ranging from a revision need to a concrete risk to ICT systems and works from other fields.

{{</citation>}}


### (48/135) A Unified Framework for Probabilistic Verification of AI Systems via Weighted Model Integration (Paolo Morettin et al., 2024)

{{<citation>}}

Paolo Morettin, Andrea Passerini, Roberto Sebastiani. (2024)  
**A Unified Framework for Probabilistic Verification of AI Systems via Weighted Model Integration**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2402.04892v1)  

---


**ABSTRACT**  
The probabilistic formal verification (PFV) of AI systems is in its infancy. So far, approaches have been limited to ad-hoc algorithms for specific classes of models and/or properties.   We propose a unifying framework for the PFV of AI systems based onWeighted Model Integration (WMI), which allows to frame the problem in very general terms.   Crucially, this reduction enables the verification of many properties of interest, like fairness, robustness or monotonicity, over a wide range of machine learning models, without making strong distributional assumptions.   We support the generality of the approach by solving multiple verification tasks with a single, off-the-shelf WMI solver, then discuss the scalability challenges and research directions related to this promising framework.

{{</citation>}}


### (49/135) CodeIt: Self-Improving Language Models with Prioritized Hindsight Replay (Natasha Butt et al., 2024)

{{<citation>}}

Natasha Butt, Blazej Manczak, Auke Wiggers, Corrado Rainone, David Zhang, Michaël Defferrard, Taco Cohen. (2024)  
**CodeIt: Self-Improving Language Models with Prioritized Hindsight Replay**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-LG, cs.AI  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2402.04858v1)  

---


**ABSTRACT**  
Large language models are increasingly solving tasks that are commonly believed to require human-level reasoning ability. However, these models still perform very poorly on benchmarks of general intelligence such as the Abstraction and Reasoning Corpus (ARC). In this paper, we approach ARC as a programming-by-examples problem, and introduce a novel and scalable method for language model self-improvement called Code Iteration (CodeIt). Our method iterates between 1) program sampling and hindsight relabeling, and 2) learning from prioritized experience replay. By relabeling the goal of an episode (i.e., the target program output given input) to the realized output produced by the sampled program, our method effectively deals with the extreme sparsity of rewards in program synthesis. Applying CodeIt to the ARC dataset, we demonstrate that prioritized hindsight replay, along with pre-training and data-augmentation, leads to successful inter-task generalization. CodeIt is the first neuro-symbolic approach that scales to the full ARC evaluation dataset. Our method solves 15% of ARC evaluation tasks, achieving state-of-the-art performance and outperforming existing neural and symbolic baselines.

{{</citation>}}


### (50/135) Explaining Learned Reward Functions with Counterfactual Trajectories (Jan Wehner et al., 2024)

{{<citation>}}

Jan Wehner, Frans Oliehoek, Luciano Cavalcante Siebert. (2024)  
**Explaining Learned Reward Functions with Counterfactual Trajectories**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2402.04856v1)  

---


**ABSTRACT**  
Learning rewards from human behaviour or feedback is a promising approach to aligning AI systems with human values but fails to consistently extract correct reward functions. Interpretability tools could enable users to understand and evaluate possible flaws in learned reward functions. We propose Counterfactual Trajectory Explanations (CTEs) to interpret reward functions in reinforcement learning by contrasting an original with a counterfactual partial trajectory and the rewards they each receive. We derive six quality criteria for CTEs and propose a novel Monte-Carlo-based algorithm for generating CTEs that optimises these quality criteria. Finally, we measure how informative the generated explanations are to a proxy-human model by training it on CTEs. CTEs are demonstrably informative for the proxy-human model, increasing the similarity between its predictions and the reward function on unseen trajectories. Further, it learns to accurately judge differences in rewards between trajectories and generalises to out-of-distribution examples. Although CTEs do not lead to a perfect understanding of the reward, our method, and more generally the adaptation of XAI methods, are presented as a fruitful approach for interpreting learned reward functions.

{{</citation>}}


### (51/135) Direct Language Model Alignment from Online AI Feedback (Shangmin Guo et al., 2024)

{{<citation>}}

Shangmin Guo, Biao Zhang, Tianlin Liu, Tianqi Liu, Misha Khalman, Felipe Llinares, Alexandre Rame, Thomas Mesnard, Yao Zhao, Bilal Piot, Johan Ferret, Mathieu Blondel. (2024)  
**Direct Language Model Alignment from Online AI Feedback**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-HC, cs.AI  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2402.04792v1)  

---


**ABSTRACT**  
Direct alignment from preferences (DAP) methods, such as DPO, have recently emerged as efficient alternatives to reinforcement learning from human feedback (RLHF), that do not require a separate reward model. However, the preference datasets used in DAP methods are usually collected ahead of training and never updated, thus the feedback is purely offline. Moreover, responses in these datasets are often sampled from a language model distinct from the one being aligned, and since the model evolves over training, the alignment phase is inevitably off-policy. In this study, we posit that online feedback is key and improves DAP methods. Our method, online AI feedback (OAIF), uses an LLM as annotator: on each training iteration, we sample two responses from the current model and prompt the LLM annotator to choose which one is preferred, thus providing online feedback. Despite its simplicity, we demonstrate via human evaluation in several tasks that OAIF outperforms both offline DAP and RLHF methods. We further show that the feedback leveraged in OAIF is easily controllable, via instruction prompts to the LLM annotator.

{{</citation>}}


### (52/135) SPARQL Generation: an analysis on fine-tuning OpenLLaMA for Question Answering over a Life Science Knowledge Graph (Julio C. Rangel et al., 2024)

{{<citation>}}

Julio C. Rangel, Tarcisio Mendes de Farias, Ana Claudia Sima, Norio Kobayashi. (2024)  
**SPARQL Generation: an analysis on fine-tuning OpenLLaMA for Question Answering over a Life Science Knowledge Graph**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-DB, cs-IR, cs.AI  
Keywords: Knowledge Graph, LLaMA, Language Model, Natural Language Processing, Question Answering  
[Paper Link](http://arxiv.org/abs/2402.04627v1)  

---


**ABSTRACT**  
The recent success of Large Language Models (LLM) in a wide range of Natural Language Processing applications opens the path towards novel Question Answering Systems over Knowledge Graphs leveraging LLMs. However, one of the main obstacles preventing their implementation is the scarcity of training data for the task of translating questions into corresponding SPARQL queries, particularly in the case of domain-specific KGs. To overcome this challenge, in this study, we evaluate several strategies for fine-tuning the OpenLlama LLM for question answering over life science knowledge graphs. In particular, we propose an end-to-end data augmentation approach for extending a set of existing queries over a given knowledge graph towards a larger dataset of semantically enriched question-to-SPARQL query pairs, enabling fine-tuning even for datasets where these pairs are scarce. In this context, we also investigate the role of semantic "clues" in the queries, such as meaningful variable names and inline comments. Finally, we evaluate our approach over the real-world Bgee gene expression knowledge graph and we show that semantic clues can improve model performance by up to 33% compared to a baseline with random variable names and no comments included.

{{</citation>}}


### (53/135) CMSA algorithm for solving the prioritized pairwise test data generation problem in software product lines (Javier Ferrer et al., 2024)

{{<citation>}}

Javier Ferrer, Francisco Chicano, José Antonio Ortega Toro. (2024)  
**CMSA algorithm for solving the prioritized pairwise test data generation problem in software product lines**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-SE, cs.AI  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2402.04597v1)  

---


**ABSTRACT**  
In Software Product Lines (SPLs) it may be difficult or even impossible to test all the products of the family because of the large number of valid feature combinations that may exist. Thus, we want to find a minimal subset of the product family that allows us to test all these possible combinations (pairwise). Furthermore, when testing a single product is a great effort, it is desirable to first test products composed of a set of priority features. This problem is called Prioritized Pairwise Test Data Generation Problem.   State-of-the-art algorithms based on Integer Linear Programming for this problema are faster enough for small and medium instances. However, there exists some real instances that are too large to be computed with these algorithms in a reasonable time because of the exponential growth of the number of candidate solutions. Also, these heuristics not always lead us to the best solutions. In this work we propose a new approach based on a hybrid metaheuristic algorithm called Construct, Merge, Solve & Adapt. We compare this matheuristic with four algorithms: a Hybrid algorithm based on Integer Linear Programming ((HILP), a Hybrid algorithm based on Integer Nonlinear Programming (HINLP), the Parallel Prioritized Genetic Solver (PPGS), and a greedy algorithm called prioritized-ICPL. The analysis reveals that CMSA results in statistically significantly better quality solutions in most instances and for most levels of weighted coverage, although it requires more execution time.

{{</citation>}}


### (54/135) Can Large Language Model Agents Simulate Human Trust Behaviors? (Chengxing Xie et al., 2024)

{{<citation>}}

Chengxing Xie, Canyu Chen, Feiran Jia, Ziyu Ye, Kai Shu, Adel Bibi, Ziniu Hu, Philip Torr, Bernard Ghanem, Guohao Li. (2024)  
**Can Large Language Model Agents Simulate Human Trust Behaviors?**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-HC, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2402.04559v1)  

---


**ABSTRACT**  
Large Language Model (LLM) agents have been increasingly adopted as simulation tools to model humans in applications such as social science. However, one fundamental question remains: can LLM agents really simulate human behaviors? In this paper, we focus on one of the most critical behaviors in human interactions, trust, and aim to investigate whether or not LLM agents can simulate human trust behaviors. We first find that LLM agents generally exhibit trust behaviors, referred to as agent trust, under the framework of Trust Games, which are widely recognized in behavioral economics. Then, we discover that LLM agents can have high behavioral alignment with humans regarding trust behaviors, indicating the feasibility to simulate human trust behaviors with LLM agents. In addition, we probe into the biases in agent trust and the differences in agent trust towards agents and humans. We also explore the intrinsic properties of agent trust under conditions including advanced reasoning strategies and external manipulations. We further offer important implications for various scenarios where trust is paramount. Our study represents a significant step in understanding the behaviors of LLM agents and the LLM-human analogy.

{{</citation>}}


## cs.CV (20)



### (55/135) Knowledge Distillation for Road Detection based on cross-model Semi-Supervised Learning (Wanli Ma et al., 2024)

{{<citation>}}

Wanli Ma, Oktay Karakus, Paul L. Rosin. (2024)  
**Knowledge Distillation for Road Detection based on cross-model Semi-Supervised Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Knowledge Distillation, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2402.05305v1)  

---


**ABSTRACT**  
The advancement of knowledge distillation has played a crucial role in enabling the transfer of knowledge from larger teacher models to smaller and more efficient student models, and is particularly beneficial for online and resource-constrained applications. The effectiveness of the student model heavily relies on the quality of the distilled knowledge received from the teacher. Given the accessibility of unlabelled remote sensing data, semi-supervised learning has become a prevalent strategy for enhancing model performance. However, relying solely on semi-supervised learning with smaller models may be insufficient due to their limited capacity for feature extraction. This limitation restricts their ability to exploit training data. To address this issue, we propose an integrated approach that combines knowledge distillation and semi-supervised learning methods. This hybrid approach leverages the robust capabilities of large models to effectively utilise large unlabelled data whilst subsequently providing the small student model with rich and informative features for enhancement. The proposed semi-supervised learning-based knowledge distillation (SSLKD) approach demonstrates a notable improvement in the performance of the student model, in the application of road segmentation, surpassing the effectiveness of traditional semi-supervised learning methods.

{{</citation>}}


### (56/135) SPAD : Spatially Aware Multiview Diffusers (Yash Kant et al., 2024)

{{<citation>}}

Yash Kant, Ziyi Wu, Michael Vasilkovsky, Guocheng Qian, Jian Ren, Riza Alp Guler, Bernard Ghanem, Sergey Tulyakov, Igor Gilitschenski, Aliaksandr Siarohin. (2024)  
**SPAD : Spatially Aware Multiview Diffusers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2402.05235v1)  

---


**ABSTRACT**  
We present SPAD, a novel approach for creating consistent multi-view images from text prompts or single images. To enable multi-view generation, we repurpose a pretrained 2D diffusion model by extending its self-attention layers with cross-view interactions, and fine-tune it on a high quality subset of Objaverse. We find that a naive extension of the self-attention proposed in prior work (e.g. MVDream) leads to content copying between views. Therefore, we explicitly constrain the cross-view attention based on epipolar geometry. To further enhance 3D consistency, we utilize Plucker coordinates derived from camera rays and inject them as positional encoding. This enables SPAD to reason over spatial proximity in 3D well. In contrast to recent works that can only generate views at fixed azimuth and elevation, SPAD offers full camera control and achieves state-of-the-art results in novel view synthesis on unseen objects from the Objaverse and Google Scanned Objects datasets. Finally, we demonstrate that text-to-3D generation using SPAD prevents the multi-face Janus issue. See more details at our webpage: https://yashkant.github.io/spad

{{</citation>}}


### (57/135) Image captioning for Brazilian Portuguese using GRIT model (Rafael Silva de Alencar et al., 2024)

{{<citation>}}

Rafael Silva de Alencar, William Alberto Cruz Castañeda, Marcellus Amadeus. (2024)  
**Image captioning for Brazilian Portuguese using GRIT model**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2402.05106v1)  

---


**ABSTRACT**  
This work presents the early development of a model of image captioning for the Brazilian Portuguese language. We used the GRIT (Grid - and Region-based Image captioning Transformer) model to accomplish this work. GRIT is a Transformer-only neural architecture that effectively utilizes two visual features to generate better captions. The GRIT method emerged as a proposal to be a more efficient way to generate image captioning. In this work, we adapt the GRIT model to be trained in a Brazilian Portuguese dataset to have an image captioning method for the Brazilian Portuguese Language.

{{</citation>}}


### (58/135) Enhancement of Bengali OCR by Specialized Models and Advanced Techniques for Diverse Document Types (AKM Shahariar Azad Rabby et al., 2024)

{{<citation>}}

AKM Shahariar Azad Rabby, Hasmot Ali, Md. Majedul Islam, Sheikh Abujar, Fuad Rahman. (2024)  
**Enhancement of Bengali OCR by Specialized Models and Advanced Techniques for Diverse Document Types**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: OCR  
[Paper Link](http://arxiv.org/abs/2402.05158v1)  

---


**ABSTRACT**  
This research paper presents a unique Bengali OCR system with some capabilities. The system excels in reconstructing document layouts while preserving structure, alignment, and images. It incorporates advanced image and signature detection for accurate extraction. Specialized models for word segmentation cater to diverse document types, including computer-composed, letterpress, typewriter, and handwritten documents. The system handles static and dynamic handwritten inputs, recognizing various writing styles. Furthermore, it has the ability to recognize compound characters in Bengali. Extensive data collection efforts provide a diverse corpus, while advanced technical components optimize character and word recognition. Additional contributions include image, logo, signature and table recognition, perspective correction, layout reconstruction, and a queuing module for efficient and scalable processing. The system demonstrates outstanding performance in efficient and accurate text extraction and analysis.

{{</citation>}}


### (59/135) Detection and Pose Estimation of flat, Texture-less Industry Objects on HoloLens using synthetic Training (Thomas Pöllabauer et al., 2024)

{{<citation>}}

Thomas Pöllabauer, Fabian Rücker, Andreas Franek, Felix Gorschlüter. (2024)  
**Detection and Pose Estimation of flat, Texture-less Industry Objects on HoloLens using synthetic Training**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Microsoft  
[Paper Link](http://arxiv.org/abs/2402.04979v1)  

---


**ABSTRACT**  
Current state-of-the-art 6d pose estimation is too compute intensive to be deployed on edge devices, such as Microsoft HoloLens (2) or Apple iPad, both used for an increasing number of augmented reality applications. The quality of AR is greatly dependent on its capabilities to detect and overlay geometry within the scene. We propose a synthetically trained client-server-based augmented reality application, demonstrating state-of-the-art object pose estimation of metallic and texture-less industry objects on edge devices. Synthetic data enables training without real photographs, i.e. for yet-to-be-manufactured objects. Our qualitative evaluation on an AR-assisted sorting task, and quantitative evaluation on both renderings, as well as real-world data recorded on HoloLens 2, sheds light on its real-world applicability.

{{</citation>}}


### (60/135) Toward Accurate Camera-based 3D Object Detection via Cascade Depth Estimation and Calibration (Chaoqun Wang et al., 2024)

{{<citation>}}

Chaoqun Wang, Yiran Qin, Zijian Kang, Ningning Ma, Ruimao Zhang. (2024)  
**Toward Accurate Camera-based 3D Object Detection via Cascade Depth Estimation and Calibration**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection, Transformer  
[Paper Link](http://arxiv.org/abs/2402.04883v1)  

---


**ABSTRACT**  
Recent camera-based 3D object detection is limited by the precision of transforming from image to 3D feature spaces, as well as the accuracy of object localization within the 3D space. This paper aims to address such a fundamental problem of camera-based 3D object detection: How to effectively learn depth information for accurate feature lifting and object localization. Different from previous methods which directly predict depth distributions by using a supervised estimation model, we propose a cascade framework consisting of two depth-aware learning paradigms. First, a depth estimation (DE) scheme leverages relative depth information to realize the effective feature lifting from 2D to 3D spaces. Furthermore, a depth calibration (DC) scheme introduces depth reconstruction to further adjust the 3D object localization perturbation along the depth axis. In practice, the DE is explicitly realized by using both the absolute and relative depth optimization loss to promote the precision of depth prediction, while the capability of DC is implicitly embedded into the detection Transformer through a depth denoising mechanism in the training phase. The entire model training is accomplished through an end-to-end manner. We propose a baseline detector and evaluate the effectiveness of our proposal with +2.2%/+2.7% NDS/mAP improvements on NuScenes benchmark, and gain a comparable performance with 55.9%/45.7% NDS/mAP. Furthermore, we conduct extensive experiments to demonstrate its generality based on various detectors with about +2% NDS improvements.

{{</citation>}}


### (61/135) STAR: Shape-focused Texture Agnostic Representations for Improved Object Detection and 6D Pose Estimation (Peter Hönig et al., 2024)

{{<citation>}}

Peter Hönig, Stefan Thalhammer, Jean-Baptiste Weibel, Matthias Hirschmanner, Markus Vincze. (2024)  
**STAR: Shape-focused Texture Agnostic Representations for Improved Object Detection and 6D Pose Estimation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2402.04878v1)  

---


**ABSTRACT**  
Recent advances in machine learning have greatly benefited object detection and 6D pose estimation for robotic grasping. However, textureless and metallic objects still pose a significant challenge due to fewer visual cues and the texture bias of CNNs. To address this issue, we propose a texture-agnostic approach that focuses on learning from CAD models and emphasizes object shape features. To achieve a focus on learning shape features, the textures are randomized during the rendering of the training data. By treating the texture as noise, the need for real-world object instances or their final appearance during training data generation is eliminated. The TLESS and ITODD datasets, specifically created for industrial settings in robotics and featuring textureless and metallic objects, were used for evaluation. Texture agnosticity also increases the robustness against image perturbations such as imaging noise, motion blur, and brightness changes, which are common in robotics applications. Code and datasets are publicly available at github.com/hoenigpeter/randomized_texturing.

{{</citation>}}


### (62/135) Advancing Anomaly Detection: An Adaptation Model and a New Dataset (Liyun Zhu et al., 2024)

{{<citation>}}

Liyun Zhu, Arjun Raj, Lei Wang. (2024)  
**Advancing Anomaly Detection: An Adaptation Model and a New Dataset**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2402.04857v1)  

---


**ABSTRACT**  
Industry surveillance is widely applicable in sectors like retail, manufacturing, education, and smart cities, each presenting unique anomalies requiring specialized detection. However, adapting anomaly detection models to novel viewpoints within the same scenario poses challenges. Extending these models to entirely new scenarios necessitates retraining or fine-tuning, a process that can be time consuming. To address these challenges, we propose the Scenario-Adaptive Anomaly Detection (SA2D) method, leveraging the few-shot learning framework for faster adaptation of pre-trained models to new concepts. Despite this approach, a significant challenge emerges from the absence of a comprehensive dataset with diverse scenarios and camera views. In response, we introduce the Multi-Scenario Anomaly Detection (MSAD) dataset, encompassing 14 distinct scenarios captured from various camera views. This real-world dataset is the first high-resolution anomaly detection dataset, offering a solid foundation for training superior models. MSAD includes diverse normal motion patterns, incorporating challenging variations like different lighting and weather conditions. Through experimentation, we validate the efficacy of SA2D, particularly when trained on the MSAD dataset. Our results show that SA2D not only excels under novel viewpoints within the same scenario but also demonstrates competitive performance when faced with entirely new scenarios. This highlights our method's potential in addressing challenges in detecting anomalies across diverse and evolving surveillance scenarios.

{{</citation>}}


### (63/135) Dual-Path Coupled Image Deraining Network via Spatial-Frequency Interaction (Yuhong He et al., 2024)

{{<citation>}}

Yuhong He, Aiwen Jiang, Lingfang Jiang, Zhifeng Wang, Lu Wang. (2024)  
**Dual-Path Coupled Image Deraining Network via Spatial-Frequency Interaction**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2402.04855v1)  

---


**ABSTRACT**  
Transformers have recently emerged as a significant force in the field of image deraining. Existing image deraining methods utilize extensive research on self-attention. Though showcasing impressive results, they tend to neglect critical frequency information, as self-attention is generally less adept at capturing high-frequency details. To overcome this shortcoming, we have developed an innovative Dual-Path Coupled Deraining Network (DPCNet) that integrates information from both spatial and frequency domains through Spatial Feature Extraction Block (SFEBlock) and Frequency Feature Extraction Block (FFEBlock). We have further introduced an effective Adaptive Fusion Module (AFM) for the dual-path feature aggregation. Extensive experiments on six public deraining benchmarks and downstream vision tasks have demonstrated that our proposed method not only outperforms the existing state-of-the-art deraining method but also achieves visually pleasuring results with excellent robustness on downstream vision tasks.

{{</citation>}}


### (64/135) Spiking-PhysFormer: Camera-Based Remote Photoplethysmography with Parallel Spike-driven Transformer (Mingxaun Liu et al., 2024)

{{<citation>}}

Mingxaun Liu, Jiankai Tang, Haoxiang Li, Jiahao Qi, Siwei Li, Kegang Wang, Yuntao Wang, Hong Chen. (2024)  
**Spiking-PhysFormer: Camera-Based Remote Photoplethysmography with Parallel Spike-driven Transformer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2402.04798v1)  

---


**ABSTRACT**  
Artificial neural networks (ANNs) can help camera-based remote photoplethysmography (rPPG) in measuring cardiac activity and physiological signals from facial videos, such as pulse wave, heart rate and respiration rate with better accuracy. However, most existing ANN-based methods require substantial computing resources, which poses challenges for effective deployment on mobile devices. Spiking neural networks (SNNs), on the other hand, hold immense potential for energy-efficient deep learning owing to their binary and event-driven architecture. To the best of our knowledge, we are the first to introduce SNNs into the realm of rPPG, proposing a hybrid neural network (HNN) model, the Spiking-PhysFormer, aimed at reducing power consumption. Specifically, the proposed Spiking-PhyFormer consists of an ANN-based patch embedding block, SNN-based transformer blocks, and an ANN-based predictor head. First, to simplify the transformer block while preserving its capacity to aggregate local and global spatio-temporal features, we design a parallel spike transformer block to replace sequential sub-blocks. Additionally, we propose a simplified spiking self-attention mechanism that omits the value parameter without compromising the model's performance. Experiments conducted on four datasets-PURE, UBFC-rPPG, UBFC-Phys, and MMPD demonstrate that the proposed model achieves a 12.4\% reduction in power consumption compared to PhysFormer. Additionally, the power consumption of the transformer block is reduced by a factor of 12.2, while maintaining decent performance as PhysFormer and other ANN-based models.

{{</citation>}}


### (65/135) Boundary-aware Contrastive Learning for Semi-supervised Nuclei Instance Segmentation (Ye Zhang et al., 2024)

{{<citation>}}

Ye Zhang, Ziyue Wang, Yifeng Wang, Hao Bian, Linghan Cai, Hengrui Li, Lingbo Zhang, Yongbing Zhang. (2024)  
**Boundary-aware Contrastive Learning for Semi-supervised Nuclei Instance Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2402.04756v1)  

---


**ABSTRACT**  
Semi-supervised segmentation methods have demonstrated promising results in natural scenarios, providing a solution to reduce dependency on manual annotation. However, these methods face significant challenges when directly applied to pathological images due to the subtle color differences between nuclei and tissues, as well as the significant morphological variations among nuclei. Consequently, the generated pseudo-labels often contain much noise, especially at the nuclei boundaries. To address the above problem, this paper proposes a boundary-aware contrastive learning network to denoise the boundary noise in a semi-supervised nuclei segmentation task. The model has two key designs: a low-resolution denoising (LRD) module and a cross-RoI contrastive learning (CRC) module. The LRD improves the smoothness of the nuclei boundary by pseudo-labels denoising, and the CRC enhances the discrimination between foreground and background by boundary feature contrastive learning. We conduct extensive experiments to demonstrate the superiority of our proposed method over existing semi-supervised instance segmentation methods.

{{</citation>}}


### (66/135) G-NAS: Generalizable Neural Architecture Search for Single Domain Generalization Object Detection (Fan Wu et al., 2024)

{{<citation>}}

Fan Wu, Jinling Gao, Lanqing Hong, Xinbing Wang, Chenghu Zhou, Nanyang Ye. (2024)  
**G-NAS: Generalizable Neural Architecture Search for Single Domain Generalization Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2402.04672v1)  

---


**ABSTRACT**  
In this paper, we focus on a realistic yet challenging task, Single Domain Generalization Object Detection (S-DGOD), where only one source domain's data can be used for training object detectors, but have to generalize multiple distinct target domains. In S-DGOD, both high-capacity fitting and generalization abilities are needed due to the task's complexity. Differentiable Neural Architecture Search (NAS) is known for its high capacity for complex data fitting and we propose to leverage Differentiable NAS to solve S-DGOD. However, it may confront severe over-fitting issues due to the feature imbalance phenomenon, where parameters optimized by gradient descent are biased to learn from the easy-to-learn features, which are usually non-causal and spuriously correlated to ground truth labels, such as the features of background in object detection data. Consequently, this leads to serious performance degradation, especially in generalizing to unseen target domains with huge domain gaps between the source domain and target domains. To address this issue, we propose the Generalizable loss (G-loss), which is an OoD-aware objective, preventing NAS from over-fitting by using gradient descent to optimize parameters not only on a subset of easy-to-learn features but also the remaining predictive features for generalization, and the overall framework is named G-NAS. Experimental results on the S-DGOD urban-scene datasets demonstrate that the proposed G-NAS achieves SOTA performance compared to baseline methods. Codes are available at https://github.com/wufan-cse/G-NAS.

{{</citation>}}


### (67/135) LLMs Meet VLMs: Boost Open Vocabulary Object Detection with Fine-grained Descriptors (Sheng Jin et al., 2024)

{{<citation>}}

Sheng Jin, Xueying Jiang, Jiaxing Huang, Lewei Lu, Shijian Lu. (2024)  
**LLMs Meet VLMs: Boost Open Vocabulary Object Detection with Fine-grained Descriptors**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2402.04630v1)  

---


**ABSTRACT**  
Inspired by the outstanding zero-shot capability of vision language models (VLMs) in image classification tasks, open-vocabulary object detection has attracted increasing interest by distilling the broad VLM knowledge into detector training. However, most existing open-vocabulary detectors learn by aligning region embeddings with categorical labels (e.g., bicycle) only, disregarding the capability of VLMs on aligning visual embeddings with fine-grained text description of object parts (e.g., pedals and bells). This paper presents DVDet, a Descriptor-Enhanced Open Vocabulary Detector that introduces conditional context prompts and hierarchical textual descriptors that enable precise region-text alignment as well as open-vocabulary detection training in general. Specifically, the conditional context prompt transforms regional embeddings into image-like representations that can be directly integrated into general open vocabulary detection training. In addition, we introduce large language models as an interactive and implicit knowledge repository which enables iterative mining and refining visually oriented textual descriptors for precise region-text alignment. Extensive experiments over multiple large-scale benchmarks show that DVDet outperforms the state-of-the-art consistently by large margins.

{{</citation>}}


### (68/135) Multi-Scale Semantic Segmentation with Modified MBConv Blocks (Xi Chen et al., 2024)

{{<citation>}}

Xi Chen, Yang Cai, Yuan Wu, Bo Xiong, Taesung Park. (2024)  
**Multi-Scale Semantic Segmentation with Modified MBConv Blocks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2402.04618v1)  

---


**ABSTRACT**  
Recently, MBConv blocks, initially designed for efficiency in resource-limited settings and later adapted for cutting-edge image classification performances, have demonstrated significant potential in image classification tasks. Despite their success, their application in semantic segmentation has remained relatively unexplored. This paper introduces a novel adaptation of MBConv blocks specifically tailored for semantic segmentation. Our modification stems from the insight that semantic segmentation requires the extraction of more detailed spatial information than image classification. We argue that to effectively perform multi-scale semantic segmentation, each branch of a U-Net architecture, regardless of its resolution, should possess equivalent segmentation capabilities. By implementing these changes, our approach achieves impressive mean Intersection over Union (IoU) scores of 84.5% and 84.0% on the Cityscapes test and validation datasets, respectively, demonstrating the efficacy of our proposed modifications in enhancing semantic segmentation performance.

{{</citation>}}


### (69/135) ScreenAI: A Vision-Language Model for UI and Infographics Understanding (Gilles Baechler et al., 2024)

{{<citation>}}

Gilles Baechler, Srinivas Sunkara, Maria Wang, Fedir Zubach, Hassan Mansoor, Vincent Etter, Victor Cărbune, Jason Lin, Jindong Chen, Abhanshu Sharma. (2024)  
**ScreenAI: A Vision-Language Model for UI and Infographics Understanding**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI, Language Model, QA  
[Paper Link](http://arxiv.org/abs/2402.04615v1)  

---


**ABSTRACT**  
Screen user interfaces (UIs) and infographics, sharing similar visual language and design principles, play important roles in human communication and human-machine interaction. We introduce ScreenAI, a vision-language model that specializes in UI and infographics understanding. Our model improves upon the PaLI architecture with the flexible patching strategy of pix2struct and is trained on a unique mixture of datasets. At the heart of this mixture is a novel screen annotation task in which the model has to identify the type and location of UI elements. We use these text annotations to describe screens to Large Language Models and automatically generate question-answering (QA), UI navigation, and summarization training datasets at scale. We run ablation studies to demonstrate the impact of these design choices. At only 5B parameters, ScreenAI achieves new state-of-the-artresults on UI- and infographics-based tasks (Multi-page DocVQA, WebSRC, MoTIF and Widget Captioning), and new best-in-class performance on others (Chart QA, DocVQA, and InfographicVQA) compared to models of similar size. Finally, we release three new datasets: one focused on the screen annotation task and two others focused on question answering.

{{</citation>}}


### (70/135) Sparse Anatomical Prompt Semi-Supervised Learning with Masked Image Modeling for CBCT Tooth Segmentation (Pengyu Dai et al., 2024)

{{<citation>}}

Pengyu Dai, Yafei Ou, Yang Liu, Yue Zhao. (2024)  
**Sparse Anatomical Prompt Semi-Supervised Learning with Masked Image Modeling for CBCT Tooth Segmentation**  

---
Primary Category: cs.CV  
Categories: I-4-6, cs-CV, cs.CV  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2402.04587v1)  

---


**ABSTRACT**  
Accurate tooth identification and segmentation in Cone Beam Computed Tomography (CBCT) dental images can significantly enhance the efficiency and precision of manual diagnoses performed by dentists. However, existing segmentation methods are mainly developed based on large data volumes training, on which their annotations are extremely time-consuming. Meanwhile, the teeth of each class in CBCT dental images being closely positioned, coupled with subtle inter-class differences, gives rise to the challenge of indistinct boundaries when training model with limited data. To address these challenges, this study aims to propose a tasked-oriented Masked Auto-Encoder paradigm to effectively utilize large amounts of unlabeled data to achieve accurate tooth segmentation with limited labeled data. Specifically, we first construct a self-supervised pre-training framework of masked auto encoder to efficiently utilize unlabeled data to enhance the network performance. Subsequently, we introduce a sparse masked prompt mechanism based on graph attention to incorporate boundary information of the teeth, aiding the network in learning the anatomical structural features of teeth. To the best of our knowledge, we are pioneering the integration of the mask pre-training paradigm into the CBCT tooth segmentation task. Extensive experiments demonstrate both the feasibility of our proposed method and the potential of the boundary prompt mechanism.

{{</citation>}}


### (71/135) Attention Guided CAM: Visual Explanations of Vision Transformer Guided by Self-Attention (Saebom Leem et al., 2024)

{{<citation>}}

Saebom Leem, Hyunseok Seo. (2024)  
**Attention Guided CAM: Visual Explanations of Vision Transformer Guided by Self-Attention**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Attention, Self-Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2402.04563v1)  

---


**ABSTRACT**  
Vision Transformer(ViT) is one of the most widely used models in the computer vision field with its great performance on various tasks. In order to fully utilize the ViT-based architecture in various applications, proper visualization methods with a decent localization performance are necessary, but these methods employed in CNN-based models are still not available in ViT due to its unique structure. In this work, we propose an attention-guided visualization method applied to ViT that provides a high-level semantic explanation for its decision. Our method selectively aggregates the gradients directly propagated from the classification output to each self-attention, collecting the contribution of image features extracted from each location of the input image. These gradients are additionally guided by the normalized self-attention scores, which are the pairwise patch correlation scores. They are used to supplement the gradients on the patch-level context information efficiently detected by the self-attention mechanism. This approach of our method provides elaborate high-level semantic explanations with great localization performance only with the class labels. As a result, our method outperforms the previous leading explainability methods of ViT in the weakly-supervised localization task and presents great capability in capturing the full instances of the target class object. Meanwhile, our method provides a visualization that faithfully explains the model, which is demonstrated in the perturbation comparison test.

{{</citation>}}


### (72/135) DMAT: A Dynamic Mask-Aware Transformer for Human De-occlusion (Guoqiang Liang et al., 2024)

{{<citation>}}

Guoqiang Liang, Jiahao Hu, Qingyue Wang, Shizhou Zhang. (2024)  
**DMAT: A Dynamic Mask-Aware Transformer for Human De-occlusion**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2402.04558v1)  

---


**ABSTRACT**  
Human de-occlusion, which aims to infer the appearance of invisible human parts from an occluded image, has great value in many human-related tasks, such as person re-id, and intention inference. To address this task, this paper proposes a dynamic mask-aware transformer (DMAT), which dynamically augments information from human regions and weakens that from occlusion. First, to enhance token representation, we design an expanded convolution head with enlarged kernels, which captures more local valid context and mitigates the influence of surrounding occlusion. To concentrate on the visible human parts, we propose a novel dynamic multi-head human-mask guided attention mechanism through integrating multiple masks, which can prevent the de-occluded regions from assimilating to the background. Besides, a region upsampling strategy is utilized to alleviate the impact of occlusion on interpolated images. During model learning, an amodal loss is developed to further emphasize the recovery effect of human regions, which also refines the model's convergence. Extensive experiments on the AHP dataset demonstrate its superior performance compared to recent state-of-the-art methods.

{{</citation>}}


### (73/135) BioDrone: A Bionic Drone-based Single Object Tracking Benchmark for Robust Vision (Xin Zhao et al., 2024)

{{<citation>}}

Xin Zhao, Shiyu Hu, Yipei Wang, Jing Zhang, Yimin Hu, Rongshuai Liu, Haibin Ling, Yin Li, Renshu Li, Kun Liu, Jiadong Li. (2024)  
**BioDrone: A Bionic Drone-based Single Object Tracking Benchmark for Robust Vision**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2402.04519v1)  

---


**ABSTRACT**  
Single object tracking (SOT) is a fundamental problem in computer vision, with a wide range of applications, including autonomous driving, augmented reality, and robot navigation. The robustness of SOT faces two main challenges: tiny target and fast motion. These challenges are especially manifested in videos captured by unmanned aerial vehicles (UAV), where the target is usually far away from the camera and often with significant motion relative to the camera. To evaluate the robustness of SOT methods, we propose BioDrone -- the first bionic drone-based visual benchmark for SOT. Unlike existing UAV datasets, BioDrone features videos captured from a flapping-wing UAV system with a major camera shake due to its aerodynamics. BioDrone hence highlights the tracking of tiny targets with drastic changes between consecutive frames, providing a new robust vision benchmark for SOT. To date, BioDrone offers the largest UAV-based SOT benchmark with high-quality fine-grained manual annotations and automatically generates frame-level labels, designed for robust vision analyses. Leveraging our proposed BioDrone, we conduct a systematic evaluation of existing SOT methods, comparing the performance of 20 representative models and studying novel means of optimizing a SOTA method (KeepTrack KeepTrack) for robust SOT. Our evaluation leads to new baselines and insights for robust SOT. Moving forward, we hope that BioDrone will not only serve as a high-quality benchmark for robust SOT, but also invite future research into robust computer vision. The database, toolkits, evaluation server, and baseline results are available at http://biodrone.aitestunion.com.

{{</citation>}}


### (74/135) ColorSwap: A Color and Word Order Dataset for Multimodal Evaluation (Jirayu Burapacheep et al., 2024)

{{<citation>}}

Jirayu Burapacheep, Ishan Gaur, Agam Bhatia, Tristan Thrush. (2024)  
**ColorSwap: A Color and Word Order Dataset for Multimodal Evaluation**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2402.04492v1)  

---


**ABSTRACT**  
This paper introduces the ColorSwap dataset, designed to assess and improve the proficiency of multimodal models in matching objects with their colors. The dataset is comprised of 2,000 unique image-caption pairs, grouped into 1,000 examples. Each example includes a caption-image pair, along with a ``color-swapped'' pair. We follow the Winoground schema: the two captions in an example have the same words, but the color words have been rearranged to modify different objects. The dataset was created through a novel blend of automated caption and image generation with humans in the loop. We evaluate image-text matching (ITM) and visual language models (VLMs) and find that even the latest ones are still not robust at this task. GPT-4V and LLaVA score 72% and 42% on our main VLM metric, although they may improve with more advanced prompting techniques. On the main ITM metric, contrastive models such as CLIP and SigLIP perform close to chance (at 12% and 30%, respectively), although the non-contrastive BLIP ITM model is stronger (87%). We also find that finetuning on fewer than 2,000 examples yields significant performance gains on this out-of-distribution word-order understanding task. The dataset is here: https://github.com/Top34051/colorswap.

{{</citation>}}


## econ.TH (1)



### (75/135) Spreading Information via Social Networks: An Irrelevance Result (Yu Awaya et al., 2024)

{{<citation>}}

Yu Awaya, Vijay Krishna. (2024)  
**Spreading Information via Social Networks: An Irrelevance Result**  

---
Primary Category: econ.TH  
Categories: cs-SI, econ-TH, econ.TH  
Keywords: Social Network  
[Paper Link](http://arxiv.org/abs/2402.05276v1)  

---


**ABSTRACT**  
An informed planner wishes to spread information among a group of agents in order to induce efficient coordination -- say the adoption of a new technology with positive externalities. The agents are connected via a social network. The planner informs a seed and then the information spreads via the network. While the structure of the network affects the rate of diffusion, we show that the rate of adoption is the same for all acyclic networks.

{{</citation>}}


## cs.CL (25)



### (76/135) VerAs: Verify then Assess STEM Lab Reports (Berk Atil et al., 2024)

{{<citation>}}

Berk Atil, Mahsa Sheikhi Karizaki, Rebecca J. Passonneau. (2024)  
**VerAs: Verify then Assess STEM Lab Reports**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2402.05224v1)  

---


**ABSTRACT**  
With an increasing focus in STEM education on critical thinking skills, science writing plays an ever more important role in curricula that stress inquiry skills. A recently published dataset of two sets of college level lab reports from an inquiry-based physics curriculum relies on analytic assessment rubrics that utilize multiple dimensions, specifying subject matter knowledge and general components of good explanations. Each analytic dimension is assessed on a 6-point scale, to provide detailed feedback to students that can help them improve their science writing skills. Manual assessment can be slow, and difficult to calibrate for consistency across all students in large classes. While much work exists on automated assessment of open-ended questions in STEM subjects, there has been far less work on long-form writing such as lab reports. We present an end-to-end neural architecture that has separate verifier and assessment modules, inspired by approaches to Open Domain Question Answering (OpenQA). VerAs first verifies whether a report contains any content relevant to a given rubric dimension, and if so, assesses the relevant sentences. On the lab reports, VerAs outperforms multiple baselines based on OpenQA systems or Automated Essay Scoring (AES). VerAs also performs well on an analytic rubric for middle school physics essays.

{{</citation>}}


### (77/135) The Effect of Sampling Temperature on Problem Solving in Large Language Models (Matthew Renze et al., 2024)

{{<citation>}}

Matthew Renze, Erhan Guven. (2024)  
**The Effect of Sampling Temperature on Problem Solving in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, QA  
[Paper Link](http://arxiv.org/abs/2402.05201v1)  

---


**ABSTRACT**  
In this research study, we empirically investigate the effect of sampling temperature on the performance of Large Language Models (LLMs) on various problem-solving tasks. We created a multiple-choice question-and-answer (MCQA) exam by randomly sampling problems from standard LLM benchmarks. Then, we used four popular LLMs with five prompt-engineering techniques to solve the MCQA problems while increasing the sampling temperature from 0.0 to 1.0. Despite anecdotal reports to the contrary, our empirical results indicate that changes in temperature in the range 0.0 to 1.0 do not have a statistically significant impact on LLM performance for problem-solving tasks. In addition, these results appear to hold regardless of the LLM, the prompt-engineering technique, or the problem domain. All code, data, and supplemental materials are available on GitHub at: https://github.com/matthewrenze/jhu-llm-temperature.

{{</citation>}}


### (78/135) SALAD-Bench: A Hierarchical and Comprehensive Safety Benchmark for Large Language Models (Lijun Li et al., 2024)

{{<citation>}}

Lijun Li, Bowen Dong, Ruohui Wang, Xuhao Hu, Wangmeng Zuo, Dahua Lin, Yu Qiao, Jing Shao. (2024)  
**SALAD-Bench: A Hierarchical and Comprehensive Safety Benchmark for Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CR, cs-LG, cs.CL  
Keywords: Language Model, QA  
[Paper Link](http://arxiv.org/abs/2402.05044v2)  

---


**ABSTRACT**  
In the rapidly evolving landscape of Large Language Models (LLMs), ensuring robust safety measures is paramount. To meet this crucial need, we propose \emph{SALAD-Bench}, a safety benchmark specifically designed for evaluating LLMs, attack, and defense methods. Distinguished by its breadth, SALAD-Bench transcends conventional benchmarks through its large scale, rich diversity, intricate taxonomy spanning three levels, and versatile functionalities.SALAD-Bench is crafted with a meticulous array of questions, from standard queries to complex ones enriched with attack, defense modifications and multiple-choice. To effectively manage the inherent complexity, we introduce an innovative evaluators: the LLM-based MD-Judge for QA pairs with a particular focus on attack-enhanced queries, ensuring a seamless, and reliable evaluation. Above components extend SALAD-Bench from standard LLM safety evaluation to both LLM attack and defense methods evaluation, ensuring the joint-purpose utility. Our extensive experiments shed light on the resilience of LLMs against emerging threats and the efficacy of contemporary defense tactics. Data and evaluator are released under https://github.com/OpenSafetyLab/SALAD-BENCH.

{{</citation>}}


### (79/135) How BERT Speaks Shakespearean English? Evaluating Historical Bias in Contextual Language Models (Miriam Cuscito et al., 2024)

{{<citation>}}

Miriam Cuscito, Alfio Ferrara, Martin Ruskov. (2024)  
**How BERT Speaks Shakespearean English? Evaluating Historical Bias in Contextual Language Models**  

---
Primary Category: cs.CL  
Categories: I-2-7; J-5, cs-CL, cs-CY, cs.CL  
Keywords: BERT, Bias, Language Model  
[Paper Link](http://arxiv.org/abs/2402.05034v1)  

---


**ABSTRACT**  
In this paper, we explore the idea of analysing the historical bias of contextual language models based on BERT by measuring their adequacy with respect to Early Modern (EME) and Modern (ME) English. In our preliminary experiments, we perform fill-in-the-blank tests with 60 masked sentences (20 EME-specific, 20 ME-specific and 20 generic) and three different models (i.e., BERT Base, MacBERTh, English HLM). We then rate the model predictions according to a 5-point bipolar scale between the two language varieties and derive a weighted score to measure the adequacy of each model to EME and ME varieties of English.

{{</citation>}}


### (80/135) Pedagogical Alignment of Large Language Models (Shashank Sonkar et al., 2024)

{{<citation>}}

Shashank Sonkar, Kangqi Ni, Sapana Chaudhary, Richard G. Baraniuk. (2024)  
**Pedagogical Alignment of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2402.05000v1)  

---


**ABSTRACT**  
In this paper, we introduce the novel concept of pedagogically aligned Large Language Models (LLMs) that signifies a transformative shift in the application of LLMs within educational contexts. Rather than providing direct responses to user queries, pedagogically-aligned LLMs function as scaffolding tools, breaking complex problems into manageable subproblems and guiding students towards the final answer through constructive feedback and hints. The objective is to equip learners with problem-solving strategies that deepen their understanding and internalization of the subject matter. Previous research in this field has primarily applied the supervised finetuning approach without framing the objective as an alignment problem, hence not employing reinforcement learning through human feedback (RLHF) methods. This study reinterprets the narrative by viewing the task through the lens of alignment and demonstrates how RLHF methods emerge naturally as a superior alternative for aligning LLM behaviour. Building on this perspective, we propose a novel approach for constructing a reward dataset specifically designed for the pedagogical alignment of LLMs. We apply three state-of-the-art RLHF algorithms and find that they outperform SFT significantly. Our qualitative analyses across model differences and hyperparameter sensitivity further validate the superiority of RLHF over SFT. Also, our study sheds light on the potential of online feedback for enhancing the performance of pedagogically-aligned LLMs, thus providing valuable insights for the advancement of these models in educational settings.

{{</citation>}}


### (81/135) An Enhanced Prompt-Based LLM Reasoning Scheme via Knowledge Graph-Integrated Collaboration (Yihao Li et al., 2024)

{{<citation>}}

Yihao Li, Ru Zhang, Jianyi Liu, Gongshen Liu. (2024)  
**An Enhanced Prompt-Based LLM Reasoning Scheme via Knowledge Graph-Integrated Collaboration**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Knowledge Graph, Language Model, NLP, Natural Language Processing, QA, Reasoning  
[Paper Link](http://arxiv.org/abs/2402.04978v1)  

---


**ABSTRACT**  
While Large Language Models (LLMs) demonstrate exceptional performance in a multitude of Natural Language Processing (NLP) tasks, they encounter challenges in practical applications, including issues with hallucinations, inadequate knowledge updating, and limited transparency in the reasoning process. To overcome these limitations, this study innovatively proposes a collaborative training-free reasoning scheme involving tight cooperation between Knowledge Graph (KG) and LLMs. This scheme first involves using LLMs to iteratively explore KG, selectively retrieving a task-relevant knowledge subgraph to support reasoning. The LLMs are then guided to further combine inherent implicit knowledge to reason on the subgraph while explicitly elucidating the reasoning process. Through such a cooperative approach, our scheme achieves more reliable knowledge-based reasoning and facilitates the tracing of the reasoning results. Experimental results show that our scheme significantly progressed across multiple datasets, notably achieving over a 10% improvement on the QALD10 dataset compared to the best baseline and the fine-tuned state-of-the-art (SOTA) work. Building on this success, this study hopes to offer a valuable reference for future research in the fusion of KG and LLMs, thereby enhancing LLMs' proficiency in solving complex issues.

{{</citation>}}


### (82/135) Reconfidencing LLMs from the Grouping Loss Perspective (Lihu Chen et al., 2024)

{{<citation>}}

Lihu Chen, Alexandre Perez-Lebel, Fabian M. Suchanek, Gaël Varoquaux. (2024)  
**Reconfidencing LLMs from the Grouping Loss Perspective**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2402.04957v1)  

---


**ABSTRACT**  
Large Language Models (LLMs), including ChatGPT and LLaMA, are susceptible to generating hallucinated answers in a confident tone. While efforts to elicit and calibrate confidence scores have proven useful, recent findings show that controlling uncertainty must go beyond calibration: predicted scores may deviate significantly from the actual posterior probabilities due to the impact of grouping loss. In this work, we construct a new evaluation dataset derived from a knowledge base to assess confidence scores given to answers of Mistral and LLaMA. Experiments show that they tend to be overconfident. Further, we show that they are more overconfident on some answers than others, \emph{eg} depending on the nationality of the person in the query. In uncertainty-quantification theory, this is grouping loss. To address this, we propose a solution to reconfidence LLMs, canceling not only calibration but also grouping loss. The LLMs, after the reconfidencing process, indicate improved confidence alignment with the accuracy of their responses.

{{</citation>}}


### (83/135) Prompting Implicit Discourse Relation Annotation (Frances Yung et al., 2024)

{{<citation>}}

Frances Yung, Mansoor Ahmad, Merel Scholman, Vera Demberg. (2024)  
**Prompting Implicit Discourse Relation Annotation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2402.04918v1)  

---


**ABSTRACT**  
Pre-trained large language models, such as ChatGPT, archive outstanding performance in various reasoning tasks without supervised training and were found to have outperformed crowdsourcing workers. Nonetheless, ChatGPT's performance in the task of implicit discourse relation classification, prompted by a standard multiple-choice question, is still far from satisfactory and considerably inferior to state-of-the-art supervised approaches. This work investigates several proven prompting techniques to improve ChatGPT's recognition of discourse relations. In particular, we experimented with breaking down the classification task that involves numerous abstract labels into smaller subtasks. Nonetheless, experiment results show that the inference accuracy hardly changes even with sophisticated prompt engineering, suggesting that implicit discourse relation classification is not yet resolvable under zero-shot or few-shot settings.

{{</citation>}}


### (84/135) Personalized Text Generation with Fine-Grained Linguistic Control (Bashar Alhafni et al., 2024)

{{<citation>}}

Bashar Alhafni, Vivek Kulkarni, Dhruv Kumar, Vipul Raheja. (2024)  
**Personalized Text Generation with Fine-Grained Linguistic Control**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Text Generation  
[Paper Link](http://arxiv.org/abs/2402.04914v1)  

---


**ABSTRACT**  
As the text generation capabilities of large language models become increasingly prominent, recent studies have focused on controlling particular aspects of the generated text to make it more personalized. However, most research on controllable text generation focuses on controlling the content or modeling specific high-level/coarse-grained attributes that reflect authors' writing styles, such as formality, domain, or sentiment. In this paper, we focus on controlling fine-grained attributes spanning multiple linguistic dimensions, such as lexical and syntactic attributes. We introduce a novel benchmark to train generative models and evaluate their ability to generate personalized text based on multiple fine-grained linguistic attributes. We systematically investigate the performance of various large language models on our benchmark and draw insights from the factors that impact their performance. We make our code, data, and pretrained models publicly available.

{{</citation>}}


### (85/135) PaDeLLM-NER: Parallel Decoding in Large Language Models for Named Entity Recognition (Jinghui Lu et al., 2024)

{{<citation>}}

Jinghui Lu, Ziwei Yang, Yanjie Wang, Xuejing Liu, Can Huang. (2024)  
**PaDeLLM-NER: Parallel Decoding in Large Language Models for Named Entity Recognition**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, NER, Named Entity Recognition  
[Paper Link](http://arxiv.org/abs/2402.04838v1)  

---


**ABSTRACT**  
In this study, we aim to reduce generation latency for Named Entity Recognition (NER) with Large Language Models (LLMs). The main cause of high latency in LLMs is the sequential decoding process, which autoregressively generates all labels and mentions for NER, significantly increase the sequence length. To this end, we introduce Parallel Decoding in LLM for NE} (PaDeLLM-NER), a approach that integrates seamlessly into existing generative model frameworks without necessitating additional modules or architectural modifications. PaDeLLM-NER allows for the simultaneous decoding of all mentions, thereby reducing generation latency. Experiments reveal that PaDeLLM-NER significantly increases inference speed that is 1.76 to 10.22 times faster than the autoregressive approach for both English and Chinese. Simultaneously it maintains the quality of predictions as evidenced by the performance that is on par with the state-of-the-art across various datasets.

{{</citation>}}


### (86/135) Long Is More for Alignment: A Simple but Tough-to-Beat Baseline for Instruction Fine-Tuning (Hao Zhao et al., 2024)

{{<citation>}}

Hao Zhao, Maksym Andriushchenko, Francesco Croce, Nicolas Flammarion. (2024)  
**Long Is More for Alignment: A Simple but Tough-to-Beat Baseline for Instruction Fine-Tuning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-3.5, GPT-4, PaLM  
[Paper Link](http://arxiv.org/abs/2402.04833v1)  

---


**ABSTRACT**  
There is a consensus that instruction fine-tuning of LLMs requires high-quality data, but what are they? LIMA (NeurIPS 2023) and AlpaGasus (ICLR 2024) are state-of-the-art methods for selecting such high-quality examples, either via manual curation or using GPT-3.5-Turbo as a quality scorer. We show that the extremely simple baseline of selecting the 1,000 instructions with longest responses from standard datasets can consistently outperform these sophisticated methods according to GPT-4 and PaLM-2 as judges, while remaining competitive on the OpenLLM benchmarks that test factual knowledge. We demonstrate this for several state-of-the-art LLMs (Llama-2-7B, Llama-2-13B, and Mistral-7B) and datasets (Alpaca-52k and Evol-Instruct-70k). In addition, a lightweight refinement of such long instructions can further improve the abilities of the fine-tuned LLMs, and allows us to obtain the 2nd highest-ranked Llama-2-7B-based model on AlpacaEval 2.0 while training on only 1,000 examples and no extra preference data. We also conduct a thorough analysis of our models to ensure that their enhanced performance is not simply due to GPT-4's preference for longer responses, thus ruling out any artificial improvement. In conclusion, our findings suggest that fine-tuning on the longest instructions should be the default baseline for any research on instruction fine-tuning.

{{</citation>}}


### (87/135) Aspect-Based Sentiment Analysis for Open-Ended HR Survey Responses (Lois Rink et al., 2024)

{{<citation>}}

Lois Rink, Job Meijdam, David Graus. (2024)  
**Aspect-Based Sentiment Analysis for Open-Ended HR Survey Responses**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, Sentiment Analysis  
[Paper Link](http://arxiv.org/abs/2402.04812v1)  

---


**ABSTRACT**  
Understanding preferences, opinions, and sentiment of the workforce is paramount for effective employee lifecycle management. Open-ended survey responses serve as a valuable source of information. This paper proposes a machine learning approach for aspect-based sentiment analysis (ABSA) of Dutch open-ended responses in employee satisfaction surveys. Our approach aims to overcome the inherent noise and variability in these responses, enabling a comprehensive analysis of sentiments that can support employee lifecycle management. Through response clustering we identify six key aspects (salary, schedule, contact, communication, personal attention, agreements), which we validate by domain experts. We compile a dataset of 1,458 Dutch survey responses, revealing label imbalance in aspects and sentiments. We propose few-shot approaches for ABSA based on Dutch BERT models, and compare them against bag-of-words and zero-shot baselines. Our work significantly contributes to the field of ABSA by demonstrating the first successful application of Dutch pre-trained language models to aspect-based sentiment analysis in the domain of human resources (HR).

{{</citation>}}


### (88/135) MLLM-as-a-Judge: Assessing Multimodal LLM-as-a-Judge with Vision-Language Benchmark (Dongping Chen et al., 2024)

{{<citation>}}

Dongping Chen, Ruoxi Chen, Shilin Zhang, Yinuo Liu, Yaochen Wang, Huichi Zhou, Qihui Zhang, Pan Zhou, Yao Wan, Lichao Sun. (2024)  
**MLLM-as-a-Judge: Assessing Multimodal LLM-as-a-Judge with Vision-Language Benchmark**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CV, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2402.04788v1)  

---


**ABSTRACT**  
Multimodal Large Language Models (MLLMs) have gained significant attention recently, showing remarkable potential in artificial general intelligence. However, assessing the utility of MLLMs presents considerable challenges, primarily due to the absence multimodal benchmarks that align with human preferences. Inspired by LLM-as-a-Judge in LLMs, this paper introduces a novel benchmark, termed MLLM-as-a-Judge, to assess the ability of MLLMs in assisting judges including three distinct tasks: Scoring Evaluation, Pair Comparison, and Batch Ranking. Our study reveals that, while MLLMs demonstrate remarkable human-like discernment in Pair Comparisons, there is a significant divergence from human preferences in Scoring Evaluation and Batch Ranking tasks. Furthermore, MLLMs still face challenges in judgment, including diverse biases, hallucinatory responses, and inconsistencies, even for advanced models such as GPT-4V. These findings emphasize the pressing need for enhancements and further research efforts regarding MLLMs as fully reliable evaluators. Code and dataset are available at https://github.com/Dongping-Chen/MLLM-as-a-Judge.

{{</citation>}}


### (89/135) A Hypothesis-Driven Framework for the Analysis of Self-Rationalising Models (Marc Braun et al., 2024)

{{<citation>}}

Marc Braun, Jenny Kunz. (2024)  
**A Hypothesis-Driven Framework for the Analysis of Self-Rationalising Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-3.5  
[Paper Link](http://arxiv.org/abs/2402.04787v1)  

---


**ABSTRACT**  
The self-rationalising capabilities of LLMs are appealing because the generated explanations can give insights into the plausibility of the predictions. However, how faithful the explanations are to the predictions is questionable, raising the need to explore the patterns behind them further. To this end, we propose a hypothesis-driven statistical framework. We use a Bayesian network to implement a hypothesis about how a task (in our example, natural language inference) is solved, and its internal states are translated into natural language with templates. Those explanations are then compared to LLM-generated free-text explanations using automatic and human evaluations. This allows us to judge how similar the LLM's and the Bayesian network's decision processes are. We demonstrate the usage of our framework with an example hypothesis and two realisations in Bayesian networks. The resulting models do not exhibit a strong similarity to GPT-3.5. We discuss the implications of this as well as the framework's potential to approximate LLM decisions better in future work.

{{</citation>}}


### (90/135) StableMask: Refining Causal Masking in Decoder-only Transformer (Qingyu Yin et al., 2024)

{{<citation>}}

Qingyu Yin, Xuzheng He, Xiang Zhuang, Yu Zhao, Jianhua Yao, Xiaoyu Shen, Qiang Zhang. (2024)  
**StableMask: Refining Causal Masking in Decoder-only Transformer**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2402.04779v1)  

---


**ABSTRACT**  
The decoder-only Transformer architecture with causal masking and relative position encoding (RPE) has become the de facto choice in language modeling. Despite its exceptional performance across various tasks, we have identified two limitations: First, it requires all attention scores to be non-zero and sum up to 1, even if the current embedding has sufficient self-contained information. This compels the model to assign disproportional excessive attention to specific tokens. Second, RPE-based Transformers are not universal approximators due to their limited capacity at encoding absolute positional information, which limits their application in position-critical tasks. In this work, we propose StableMask: a parameter-free method to address both limitations by refining the causal mask. It introduces pseudo-attention values to balance attention distributions and encodes absolute positional information via a progressively decreasing mask ratio. StableMask's effectiveness is validated both theoretically and empirically, showing significant enhancements in language models with parameter sizes ranging from 71M to 1.4B across diverse datasets and encoding methods. We further show that it naturally supports (1) efficient extrapolation without special tricks such as StreamingLLM and (2) easy integration with existing attention optimization techniques.

{{</citation>}}


### (91/135) Large Language Models As Faithful Explainers (Yu-Neng Chuang et al., 2024)

{{<citation>}}

Yu-Neng Chuang, Guanchu Wang, Chia-Yuan Chang, Ruixiang Tang, Fan Yang, Mengnan Du, Xuanting Cai, Xia Hu. (2024)  
**Large Language Models As Faithful Explainers**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model, NLU  
[Paper Link](http://arxiv.org/abs/2402.04678v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have recently become proficient in addressing complex tasks by utilizing their rich internal knowledge and reasoning ability. Consequently, this complexity hinders traditional input-focused explanation algorithms for explaining the complex decision-making processes of LLMs. Recent advancements have thus emerged for self-explaining their predictions through a single feed-forward inference in a natural language format. However, natural language explanations are often criticized for lack of faithfulness since these explanations may not accurately reflect the decision-making behaviors of the LLMs. In this work, we introduce a generative explanation framework, xLLM, to improve the faithfulness of the explanations provided in natural language formats for LLMs. Specifically, we propose an evaluator to quantify the faithfulness of natural language explanation and enhance the faithfulness by an iterative optimization process of xLLM, with the goal of maximizing the faithfulness scores. Experiments conducted on three NLU datasets demonstrate that xLLM can significantly improve the faithfulness of generated explanations, which are in alignment with the behaviors of LLMs.

{{</citation>}}


### (92/135) Source Identification in Abstractive Summarization (Yoshi Suhara et al., 2024)

{{<citation>}}

Yoshi Suhara, Dimitris Alikaniotis. (2024)  
**Source Identification in Abstractive Summarization**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Summarization  
[Paper Link](http://arxiv.org/abs/2402.04677v1)  

---


**ABSTRACT**  
Neural abstractive summarization models make summaries in an end-to-end manner, and little is known about how the source information is actually converted into summaries. In this paper, we define input sentences that contain essential information in the generated summary as $\textit{source sentences}$ and study how abstractive summaries are made by analyzing the source sentences. To this end, we annotate source sentences for reference summaries and system summaries generated by PEGASUS on document-summary pairs sampled from the CNN/DailyMail and XSum datasets. We also formulate automatic source sentence detection and compare multiple methods to establish a strong baseline for the task. Experimental results show that the perplexity-based method performs well in highly abstractive settings, while similarity-based methods perform robustly in relatively extractive settings. Our code and data are available at https://github.com/suhara/sourcesum.

{{</citation>}}


### (93/135) TransLLaMa: LLM-based Simultaneous Translation System (Roman Koshkin et al., 2024)

{{<citation>}}

Roman Koshkin, Katsuhito Sudoh, Satoshi Nakamura. (2024)  
**TransLLaMa: LLM-based Simultaneous Translation System**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BLEU, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2402.04636v1)  

---


**ABSTRACT**  
Decoder-only large language models (LLMs) have recently demonstrated impressive capabilities in text generation and reasoning. Nonetheless, they have limited applications in simultaneous machine translation (SiMT), currently dominated by encoder-decoder transformers. This study demonstrates that, after fine-tuning on a small dataset comprising causally aligned source and target sentence pairs, a pre-trained open-source LLM can control input segmentation directly by generating a special "wait" token. This obviates the need for a separate policy and enables the LLM to perform English-German and English-Russian SiMT tasks with BLEU scores that are comparable to those of specific state-of-the-art baselines. We also evaluated closed-source models such as GPT-4, which displayed encouraging results in performing the SiMT task without prior training (zero-shot), indicating a promising avenue for enhancing future SiMT systems.

{{</citation>}}


### (94/135) The Future of Cognitive Strategy-enhanced Persuasive Dialogue Agents: New Perspectives and Trends (Mengqi Chen et al., 2024)

{{<citation>}}

Mengqi Chen, Bin Guo, Hao Wang, Haoyu Li, Qian Zhao, Jingqi Liu, Yasan Ding, Yan Pan, Zhiwen Yu. (2024)  
**The Future of Cognitive Strategy-enhanced Persuasive Dialogue Agents: New Perspectives and Trends**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue, Language Model  
[Paper Link](http://arxiv.org/abs/2402.04631v1)  

---


**ABSTRACT**  
Persuasion, as one of the crucial abilities in human communication, has garnered extensive attention from researchers within the field of intelligent dialogue systems. We humans tend to persuade others to change their viewpoints, attitudes or behaviors through conversations in various scenarios (e.g., persuasion for social good, arguing in online platforms). Developing dialogue agents that can persuade others to accept certain standpoints is essential to achieving truly intelligent and anthropomorphic dialogue system. Benefiting from the substantial progress of Large Language Models (LLMs), dialogue agents have acquired an exceptional capability in context understanding and response generation. However, as a typical and complicated cognitive psychological system, persuasive dialogue agents also require knowledge from the domain of cognitive psychology to attain a level of human-like persuasion. Consequently, the cognitive strategy-enhanced persuasive dialogue agent (defined as CogAgent), which incorporates cognitive strategies to achieve persuasive targets through conversation, has become a predominant research paradigm. To depict the research trends of CogAgent, in this paper, we first present several fundamental cognitive psychology theories and give the formalized definition of three typical cognitive strategies, including the persuasion strategy, the topic path planning strategy, and the argument structure prediction strategy. Then we propose a new system architecture by incorporating the formalized definition to lay the foundation of CogAgent. Representative works are detailed and investigated according to the combined cognitive strategy, followed by the summary of authoritative benchmarks and evaluation metrics. Finally, we summarize our insights on open issues and future directions of CogAgent for upcoming researchers.

{{</citation>}}


### (95/135) MEMORYLLM: Towards Self-Updatable Large Language Models (Yu Wang et al., 2024)

{{<citation>}}

Yu Wang, Xiusi Chen, Jingbo Shang, Julian McAuley. (2024)  
**MEMORYLLM: Towards Self-Updatable Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2402.04624v1)  

---


**ABSTRACT**  
Existing Large Language Models (LLMs) usually remain static after deployment, which might make it hard to inject new knowledge into the model. We aim to build models containing a considerable portion of self-updatable parameters, enabling the model to integrate new knowledge effectively and efficiently. To this end, we introduce MEMORYLLM, a model that comprises a transformer and a fixed-size memory pool within the latent space of the transformer. MEMORYLLM can self-update with text knowledge and memorize the knowledge injected earlier. Our evaluations demonstrate the ability of MEMORYLLM to effectively incorporate new knowledge, as evidenced by its performance on model editing benchmarks. Meanwhile, the model exhibits long-term information retention capacity, which is validated through our custom-designed evaluations and long-context benchmarks. MEMORYLLM also shows operational integrity without any sign of performance degradation even after nearly a million memory updates.

{{</citation>}}


### (96/135) TinyLLM: Learning a Small Student from Multiple Large Language Models (Yijun Tian et al., 2024)

{{<citation>}}

Yijun Tian, Yikun Han, Xiusi Chen, Wei Wang, Nitesh V. Chawla. (2024)  
**TinyLLM: Learning a Small Student from Multiple Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2402.04616v1)  

---


**ABSTRACT**  
Transferring the reasoning capability from stronger large language models (LLMs) to smaller ones has been quite appealing, as smaller LLMs are more flexible to deploy with less expense. Among the existing solutions, knowledge distillation stands out due to its outstanding efficiency and generalization. However, existing methods suffer from several drawbacks, including limited knowledge diversity and the lack of rich contextual information. To solve the problems and facilitate the learning of compact language models, we propose TinyLLM, a novel knowledge distillation paradigm to learn a small student LLM from multiple large teacher LLMs. In particular, we encourage the student LLM to not only generate the correct answers but also understand the rationales behind these answers. Given that different LLMs possess diverse reasoning skills, we guide the student model to assimilate knowledge from various teacher LLMs. We further introduce an in-context example generator and a teacher-forcing Chain-of-Thought strategy to ensure that the rationales are accurate and grounded in contextually appropriate scenarios. Extensive experiments on six datasets across two reasoning tasks demonstrate the superiority of our method. Results show that TinyLLM can outperform large teacher LLMs significantly, despite having a considerably smaller model size.

{{</citation>}}


### (97/135) Faithfulness vs. Plausibility: On the (Un)Reliability of Explanations from Large Language Models (Chirag Agarwal et al., 2024)

{{<citation>}}

Chirag Agarwal, Sree Harsha Tanneru, Himabindu Lakkaraju. (2024)  
**Faithfulness vs. Plausibility: On the (Un)Reliability of Explanations from Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2402.04614v2)  

---


**ABSTRACT**  
Large Language Models (LLMs) are deployed as powerful tools for several natural language processing (NLP) applications. Recent works show that modern LLMs can generate self-explanations (SEs), which elicit their intermediate reasoning steps for explaining their behavior. Self-explanations have seen widespread adoption owing to their conversational and plausible nature. However, there is little to no understanding of their faithfulness. In this work, we discuss the dichotomy between faithfulness and plausibility in SEs generated by LLMs. We argue that while LLMs are adept at generating plausible explanations -- seemingly logical and coherent to human users -- these explanations do not necessarily align with the reasoning processes of the LLMs, raising concerns about their faithfulness. We highlight that the current trend towards increasing the plausibility of explanations, primarily driven by the demand for user-friendly interfaces, may come at the cost of diminishing their faithfulness. We assert that the faithfulness of explanations is critical in LLMs employed for high-stakes decision-making. Moreover, we urge the community to identify the faithfulness requirements of real-world applications and ensure explanations meet those needs. Finally, we propose some directions for future work, emphasizing the need for novel methodologies and frameworks that can enhance the faithfulness of self-explanations without compromising their plausibility, essential for the transparent deployment of LLMs in diverse high-stakes domains.

{{</citation>}}


### (98/135) Improving Cross-Domain Low-Resource Text Generation through LLM Post-Editing: A Programmer-Interpreter Approach (Zhuang Li et al., 2024)

{{<citation>}}

Zhuang Li, Levon Haroutunian, Raj Tumuluri, Philip Cohen, Gholamreza Haffari. (2024)  
**Improving Cross-Domain Low-Resource Text Generation through LLM Post-Editing: A Programmer-Interpreter Approach**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-3.5, GPT-4, Low-Resource, Text Generation  
[Paper Link](http://arxiv.org/abs/2402.04609v1)  

---


**ABSTRACT**  
Post-editing has proven effective in improving the quality of text generated by large language models (LLMs) such as GPT-3.5 or GPT-4, particularly when direct updating of their parameters to enhance text quality is infeasible or expensive. However, relying solely on smaller language models for post-editing can limit the LLMs' ability to generalize across domains. Moreover, the editing strategies in these methods are not optimally designed for text-generation tasks. To address these limitations, we propose a neural programmer-interpreter approach that preserves the domain generalization ability of LLMs when editing their output. The editing actions in this framework are specifically devised for text generation. Extensive experiments demonstrate that the programmer-interpreter significantly enhances GPT-3.5's performance in logical form-to-text conversion and low-resource machine translation, surpassing other state-of-the-art (SOTA) LLM post-editing methods in cross-domain settings.

{{</citation>}}


### (99/135) Alirector: Alignment-Enhanced Chinese Grammatical Error Corrector (Haihui Yang et al., 2024)

{{<citation>}}

Haihui Yang, Xiaojun Quan. (2024)  
**Alirector: Alignment-Enhanced Chinese Grammatical Error Corrector**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Seq2Seq  
[Paper Link](http://arxiv.org/abs/2402.04601v1)  

---


**ABSTRACT**  
Chinese grammatical error correction (CGEC) faces serious overcorrection challenges when employing autoregressive generative models such as sequence-to-sequence (Seq2Seq) models and decoder-only large language models (LLMs). While previous methods aim to address overcorrection in Seq2Seq models, they are difficult to adapt to decoder-only LLMs. In this paper, we propose an alignment-enhanced corrector for the overcorrection problem that applies to both Seq2Seq models and decoder-only LLMs. Our method first trains a correction model to generate an initial correction of the source sentence. Then, we combine the source sentence with the initial correction and feed it through an alignment model for another round of correction, aiming to enforce the alignment model to focus on potential overcorrection. Moreover, to enhance the model's ability to identify nuances, we further explore the reverse alignment of the source sentence and the initial correction. Finally, we transfer the alignment knowledge from two alignment models to the correction model, instructing it on how to avoid overcorrection. Experimental results on three CGEC datasets demonstrate the effectiveness of our approach in alleviating overcorrection and improving overall performance.

{{</citation>}}


### (100/135) UltraLink: An Open-Source Knowledge-Enhanced Multilingual Supervised Fine-tuning Dataset (Haoyu Wang et al., 2024)

{{<citation>}}

Haoyu Wang, Shuo Wang, Yukun Yan, Xujia Wang, Zhiyu Yang, Yuzhuang Xu, Zhenghao Liu, Ning Ding, Xu Han, Zhiyuan Liu, Maosong Sun. (2024)  
**UltraLink: An Open-Source Knowledge-Enhanced Multilingual Supervised Fine-tuning Dataset**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Multilingual  
[Paper Link](http://arxiv.org/abs/2402.04588v1)  

---


**ABSTRACT**  
Open-source large language models (LLMs) have gained significant strength across diverse fields. Nevertheless, the majority of studies primarily concentrate on English, with only limited exploration into the realm of multilingual supervised fine-tuning. In this work, we therefore construct an open-source multilingual supervised fine-tuning dataset. Different from previous works that simply translate English instructions, we consider both the language-specific and language-agnostic abilities of LLMs. For language-specific abilities, we introduce a knowledge-grounded data augmentation approach to elicit more culture-specific knowledge of LLMs, improving their ability to serve users from different countries. For language-agnostic abilities, we find through experiments that modern LLMs exhibit strong cross-lingual transfer capabilities, thus repeatedly learning identical content in various languages is not necessary. Consequently, we can substantially prune the language-agnostic SFT data without any performance degradation, making the SFT process more efficient. The resulting UltraLink dataset comprises approximately 1 million samples across five languages, and the proposed data construction method can also be easily extended to other languages. UltraLink-LM, which is trained on UltraLink, outperforms several representative baselines across many tasks.

{{</citation>}}


## eess.IV (3)



### (101/135) Self-calibrated convolution towards glioma segmentation (Felipe C. R. Salvagnini et al., 2024)

{{<citation>}}

Felipe C. R. Salvagnini, Gerson O. Barbosa, Alexandre X. Falcao, Cid A. N. Santos. (2024)  
**Self-calibrated convolution towards glioma segmentation**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2402.05218v1)  

---


**ABSTRACT**  
Accurate brain tumor segmentation in the early stages of the disease is crucial for the treatment's effectiveness, avoiding exhaustive visual inspection of a qualified specialist on 3D MR brain images of multiple protocols (e.g., T1, T2, T2-FLAIR, T1-Gd). Several networks exist for Glioma segmentation, being nnU-Net one of the best. In this work, we evaluate self-calibrated convolutions in different parts of the nnU-Net network to demonstrate that self-calibrated modules in skip connections can significantly improve the enhanced-tumor and tumor-core segmentation accuracy while preserving the wholetumor segmentation accuracy.

{{</citation>}}


### (102/135) Mamba-UNet: UNet-Like Pure Visual Mamba for Medical Image Segmentation (Ziyang Wang et al., 2024)

{{<citation>}}

Ziyang Wang, Jian-Qing Zheng, Yichi Zhang, Ge Cui, Lei Li. (2024)  
**Mamba-UNet: UNet-Like Pure Visual Mamba for Medical Image Segmentation**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2402.05079v1)  

---


**ABSTRACT**  
In recent advancements in medical image analysis, Convolutional Neural Networks (CNN) and Vision Transformers (ViT) have set significant benchmarks. While the former excels in capturing local features through its convolution operations, the latter achieves remarkable global context understanding by leveraging self-attention mechanisms. However, both architectures exhibit limitations in efficiently modeling long-range dependencies within medical images, which is a critical aspect for precise segmentation. Inspired by the Mamba architecture, known for its proficiency in handling long sequences and global contextual information with enhanced computational efficiency as a State Space Model (SSM), we propose Mamba-UNet, a novel architecture that synergizes the U-Net in medical image segmentation with Mamba's capability. Mamba-UNet adopts a pure Visual Mamba (VMamba)-based encoder-decoder structure, infused with skip connections to preserve spatial information across different scales of the network. This design facilitates a comprehensive feature learning process, capturing intricate details and broader semantic contexts within medical images. We introduce a novel integration mechanism within the VMamba blocks to ensure seamless connectivity and information flow between the encoder and decoder paths, enhancing the segmentation performance. We conducted experiments on publicly available MRI cardiac multi-structures segmentation dataset. The results show that Mamba-UNet outperforms UNet, Swin-UNet in medical image segmentation under the same hyper-parameter setting. The source code and baseline implementations are available.

{{</citation>}}


### (103/135) Triplet-constraint Transformer with Multi-scale Refinement for Dose Prediction in Radiotherapy (Lu Wen et al., 2024)

{{<citation>}}

Lu Wen, Qihun Zhang, Zhenghao Feng, Yuanyuan Xu, Xiao Chen, Jiliu Zhou, Yan Wang. (2024)  
**Triplet-constraint Transformer with Multi-scale Refinement for Dose Prediction in Radiotherapy**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2402.04566v1)  

---


**ABSTRACT**  
Radiotherapy is a primary treatment for cancers with the aim of applying sufficient radiation dose to the planning target volume (PTV) while minimizing dose hazards to the organs at risk (OARs). Convolutional neural networks (CNNs) have automated the radiotherapy plan-making by predicting the dose maps. However, current CNN-based methods ignore the remarkable dose difference in the dose map, i.e., high dose value in the interior PTV while low value in the exterior PTV, leading to a suboptimal prediction. In this paper, we propose a triplet-constraint transformer (TCtrans) with multi-scale refinement to predict the high-quality dose distribution. Concretely, a novel PTV-guided triplet constraint is designed to refine dose feature representations in the interior and exterior PTV by utilizing the explicit geometry of PTV. Furthermore, we introduce a multi-scale refinement (MSR) module to effectively fulfill the triplet constraint in different decoding layers with multiple scales. Besides, a transformer encoder is devised to learn the important global dosimetric knowledge. Experiments on a clinical cervical cancer dataset demonstrate the superiority of our method.

{{</citation>}}


## cond-mat.mtrl-sci (1)



### (104/135) Are LLMs Ready for Real-World Materials Discovery? (Santiago Miret et al., 2024)

{{<citation>}}

Santiago Miret, N M Anoop Krishnan. (2024)  
**Are LLMs Ready for Real-World Materials Discovery?**  

---
Primary Category: cond-mat.mtrl-sci  
Categories: cond-mat-mtrl-sci, cond-mat.mtrl-sci, cs-AI, cs-CL, cs-LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2402.05200v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) create exciting possibilities for powerful language processing tools to accelerate research in materials science. While LLMs have great potential to accelerate materials understanding and discovery, they currently fall short in being practical materials science tools. In this position paper, we show relevant failure cases of LLMs in materials science that reveal current limitations of LLMs related to comprehending and reasoning over complex, interconnected materials science knowledge. Given those shortcomings, we outline a framework for developing Materials Science LLMs (MatSci-LLMs) that are grounded in materials science knowledge and hypothesis generation followed by hypothesis testing. The path to attaining performant MatSci-LLMs rests in large part on building high-quality, multi-modal datasets sourced from scientific literature where various information extraction challenges persist. As such, we describe key materials science information extraction challenges which need to be overcome in order to build large-scale, multi-modal datasets that capture valuable materials science knowledge. Finally, we outline a roadmap for applying future MatSci-LLMs for real-world materials discovery via: 1. Automated Knowledge Base Generation; 2. Automated In-Silico Material Design; and 3. MatSci-LLM Integrated Self-Driving Materials Laboratories.

{{</citation>}}


## cs.SE (5)



### (105/135) You Can REST Now: Automated Specification Inference and Black-Box Testing of RESTful APIs with Large Language Models (Alix Decrop et al., 2024)

{{<citation>}}

Alix Decrop, Gilles Perrouin, Mike Papadakis, Xavier Devroey, Pierre-Yves Schobbens. (2024)  
**You Can REST Now: Automated Specification Inference and Black-Box Testing of RESTful APIs with Large Language Models**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2402.05102v1)  

---


**ABSTRACT**  
RESTful APIs are popular web services, requiring documentation to ease their comprehension, reusability and testing practices. The OpenAPI Specification (OAS) is a widely adopted and machine-readable format used to document such APIs. However, manually documenting RESTful APIs is a time-consuming and error-prone task, resulting in unavailable, incomplete, or imprecise documentation. As RESTful API testing tools require an OpenAPI specification as input, insufficient or informal documentation hampers testing quality.   Recently, Large Language Models (LLMs) have demonstrated exceptional abilities to automate tasks based on their colossal training data. Accordingly, such capabilities could be utilized to assist the documentation and testing process of RESTful APIs.   In this paper, we present RESTSpecIT, the first automated RESTful API specification inference and black-box testing approach leveraging LLMs. The approach requires minimal user input compared to state-of-the-art RESTful API inference and testing tools; Given an API name and an LLM key, HTTP requests are generated and mutated with data returned by the LLM. By sending the requests to the API endpoint, HTTP responses can be analyzed for inference and testing purposes. RESTSpecIT utilizes an in-context prompt masking strategy, requiring no model fine-tuning. Our evaluation demonstrates that RESTSpecIT is capable of: (1) inferring specifications with 85.05% of GET routes and 81.05% of query parameters found on average, (2) discovering undocumented and valid routes and parameters, and (3) uncovering server errors in RESTful APIs. Inferred specifications can also be used as testing tool inputs.

{{</citation>}}


### (106/135) What's documented in AI? Systematic Analysis of 32K AI Model Cards (Weixin Liang et al., 2024)

{{<citation>}}

Weixin Liang, Nazneen Rajani, Xinyu Yang, Ezinwanne Ozoani, Eric Wu, Yiqun Chen, Daniel Scott Smith, James Zou. (2024)  
**What's documented in AI? Systematic Analysis of 32K AI Model Cards**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-LG, cs-SE, cs.SE  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2402.05160v1)  

---


**ABSTRACT**  
The rapid proliferation of AI models has underscored the importance of thorough documentation, as it enables users to understand, trust, and effectively utilize these models in various applications. Although developers are encouraged to produce model cards, it's not clear how much information or what information these cards contain. In this study, we conduct a comprehensive analysis of 32,111 AI model documentations on Hugging Face, a leading platform for distributing and deploying AI models. Our investigation sheds light on the prevailing model card documentation practices. Most of the AI models with substantial downloads provide model cards, though the cards have uneven informativeness. We find that sections addressing environmental impact, limitations, and evaluation exhibit the lowest filled-out rates, while the training section is the most consistently filled-out. We analyze the content of each section to characterize practitioners' priorities. Interestingly, there are substantial discussions of data, sometimes with equal or even greater emphasis than the model itself. To evaluate the impact of model cards, we conducted an intervention study by adding detailed model cards to 42 popular models which had no or sparse model cards previously. We find that adding model cards is moderately correlated with an increase weekly download rates. Our study opens up a new perspective for analyzing community norms and practices for model documentation through large-scale data science and linguistics analysis.

{{</citation>}}


### (107/135) Automated Smart Contract Summarization via LLMs (Yingjie Mao et al., 2024)

{{<citation>}}

Yingjie Mao, Xiaoqi Li, Zongwei Li, Wenkai Li. (2024)  
**Automated Smart Contract Summarization via LLMs**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: BLEU, Language Model, Summarization  
[Paper Link](http://arxiv.org/abs/2402.04863v2)  

---


**ABSTRACT**  
Automatic code Summarization generation technology is widely used in the development and maintenance of smart contracts. In recent years, with the advent of Large Language Models (LLMs), Gemini has received a lot of attention as the first Large Multimodal models (LMMs) to support multimodal input. However, it is unclear how LMMs can generate contract code summarization from multimodal inputs. In this paper, we focus on evaluating Gemini on real-world smart contracts, comparing it to the MMTrans, and exploring how to combine multimodal prompts to generate a contract code summarization. We used several widely used metrics (BLEU, METEOR, and ROUGE-L) to measure the quality of the generated summarization. Our experiments show that METEOR and ROUGEL metrics, Gemini-Pro-Vision achieves 21.17% and 21.05% scores for code comments generated by three-shot prompts. These scores are better than those generated by one-shot and five-shot prompts.

{{</citation>}}


### (108/135) Enhancing User Interaction in ChatGPT: Characterizing and Consolidating Multiple Prompts for Issue Resolution (Saikat Mondal et al., 2024)

{{<citation>}}

Saikat Mondal, Suborno Deb Bappon, Chanchal K. Roy. (2024)  
**Enhancing User Interaction in ChatGPT: Characterizing and Consolidating Multiple Prompts for Issue Resolution**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2402.04568v1)  

---


**ABSTRACT**  
Prompt design plays a crucial role in shaping the efficacy of ChatGPT, influencing the model's ability to extract contextually accurate responses. Thus, optimal prompt construction is essential for maximizing the utility and performance of ChatGPT. However, sub-optimal prompt design may necessitate iterative refinement, as imprecise or ambiguous instructions can lead to undesired responses from ChatGPT. Existing studies explore several prompt patterns and strategies to improve the relevance of responses generated by ChatGPT. However, the exploration of constraints that necessitate the submission of multiple prompts is still an unmet attempt. In this study, our contributions are twofold. First, we attempt to uncover gaps in prompt design that demand multiple iterations. In particular, we manually analyze 686 prompts that were submitted to resolve issues related to Java and Python programming languages and identify eleven prompt design gaps (e.g., missing specifications). Such gap exploration can enhance the efficacy of single prompts in ChatGPT. Second, we attempt to reproduce the ChatGPT response by consolidating multiple prompts into a single one. We can completely consolidate prompts with four gaps (e.g., missing context) and partially consolidate prompts with three gaps (e.g., additional functionality). Such an effort provides concrete evidence to users to design more optimal prompts mitigating these gaps. Our study findings and evidence can - (a) save users time, (b) reduce costs, and (c) increase user satisfaction.

{{</citation>}}


### (109/135) The Foundations of Computational Management: A Systematic Approach to Task Automation for the Integration of Artificial Intelligence into Existing Workflows (Tamen Jadad-Garcia et al., 2024)

{{<citation>}}

Tamen Jadad-Garcia, Alejandro R. Jadad. (2024)  
**The Foundations of Computational Management: A Systematic Approach to Task Automation for the Integration of Artificial Intelligence into Existing Workflows**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2402.05142v1)  

---


**ABSTRACT**  
Driven by the rapid ascent of artificial intelligence (AI), organizations are at the epicenter of a seismic shift, facing a crucial question: How can AI be successfully integrated into existing operations? To help answer it, manage expectations and mitigate frustration, this article introduces Computational Management, a systematic approach to task automation for enhancing the ability of organizations to harness AI's potential within existing workflows. Computational Management acts as a bridge between the strategic insights of management science with the analytical rigor of computational thinking. The article offers three easy step-by-step procedures to begin the process of implementing AI within a workflow. Such procedures focus on task (re)formulation, on the assessment of the automation potential of tasks, on the completion of task specification templates for AI selection and adaptation. Included in the article there are manual and automated methods, with prompt suggestions for publicly available LLMs, to complete these three procedures. The first procedure, task (re)formulation, focuses on breaking down work activities into basic units, so they can be completed by one agent, involve a single well-defined action, and produce a distinct outcome. The second, allows the assessment of the granular task and its suitability for automation, using the Task Automation Index to rank tasks based on whether they have standardized input, well-defined rules, repetitiveness, data dependency, and objective outputs. The third, focuses on a task specification template which details information on 16 critical components of tasks, and can be used as a checklist to select or adapt the most suitable AI solution for integration into existing workflows. Computational Management provides a roadmap and a toolkit for humans and AI to thrive together, while enhancing organizational efficiency and innovation.

{{</citation>}}


## cs.RO (7)



### (110/135) Language-Based Augmentation to Address Shortcut Learning in Object Goal Navigation (Dennis Hoftijzer et al., 2024)

{{<citation>}}

Dennis Hoftijzer, Gertjan Burghouts, Luuk Spreeuwers. (2024)  
**Language-Based Augmentation to Address Shortcut Learning in Object Goal Navigation**  

---
Primary Category: cs.RO  
Categories: cs-CV, cs-RO, cs.RO  
Keywords: Augmentation, Language Model, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2402.05090v1)  

---


**ABSTRACT**  
Deep Reinforcement Learning (DRL) has shown great potential in enabling robots to find certain objects (e.g., `find a fridge') in environments like homes or schools. This task is known as Object-Goal Navigation (ObjectNav). DRL methods are predominantly trained and evaluated using environment simulators. Although DRL has shown impressive results, the simulators may be biased or limited. This creates a risk of shortcut learning, i.e., learning a policy tailored to specific visual details of training environments. We aim to deepen our understanding of shortcut learning in ObjectNav, its implications and propose a solution. We design an experiment for inserting a shortcut bias in the appearance of training environments. As a proof-of-concept, we associate room types to specific wall colors (e.g., bedrooms with green walls), and observe poor generalization of a state-of-the-art (SOTA) ObjectNav method to environments where this is not the case (e.g., bedrooms with blue walls). We find that shortcut learning is the root cause: the agent learns to navigate to target objects, by simply searching for the associated wall color of the target object's room. To solve this, we propose Language-Based (L-B) augmentation. Our key insight is that we can leverage the multimodal feature space of a Vision-Language Model (VLM) to augment visual representations directly at the feature-level, requiring no changes to the simulator, and only an addition of one layer to the model. Where the SOTA ObjectNav method's success rate drops 69%, our proposal has only a drop of 23%.

{{</citation>}}


### (111/135) Exploration Without Maps via Zero-Shot Out-of-Distribution Deep Reinforcement Learning (Shathushan Sivashangaran et al., 2024)

{{<citation>}}

Shathushan Sivashangaran, Apoorva Khairnar, Azim Eskandarian. (2024)  
**Exploration Without Maps via Zero-Shot Out-of-Distribution Deep Reinforcement Learning**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2402.05066v1)  

---


**ABSTRACT**  
Operation of Autonomous Mobile Robots (AMRs) of all forms that include wheeled ground vehicles, quadrupeds and humanoids in dynamically changing GPS denied environments without a-priori maps, exclusively using onboard sensors, is an unsolved problem that has potential to transform the economy, and vastly improve humanity's capabilities with improvements to agriculture, manufacturing, disaster response, military and space exploration. Conventional AMR automation approaches are modularized into perception, motion planning and control which is computationally inefficient, and requires explicit feature extraction and engineering, that inhibits generalization, and deployment at scale. Few works have focused on real-world end-to-end approaches that directly map sensor inputs to control outputs due to the large amount of well curated training data required for supervised Deep Learning (DL) which is time consuming and labor intensive to collect and label, and sample inefficiency and challenges to bridging the simulation to reality gap using Deep Reinforcement Learning (DRL). This paper presents a novel method to efficiently train DRL for robust end-to-end AMR exploration, in a constrained environment at physical limits in simulation, transferred zero-shot to the real-world. The representation learned in a compact parameter space with 2 fully connected layers with 64 nodes each is demonstrated to exhibit emergent behavior for out-of-distribution generalization to navigation in new environments that include unstructured terrain without maps, and dynamic obstacle avoidance. The learned policy outperforms conventional navigation algorithms while consuming a fraction of the computation resources, enabling execution on a range of AMR forms with varying embedded computer payloads.

{{</citation>}}


### (112/135) Deep Reinforcement Learning with Dynamic Graphs for Adaptive Informative Path Planning (Apoorva Vashisth et al., 2024)

{{<citation>}}

Apoorva Vashisth, Julius Rückin, Federico Magistri, Cyrill Stachniss, Marija Popović. (2024)  
**Deep Reinforcement Learning with Dynamic Graphs for Adaptive Informative Path Planning**  

---
Primary Category: cs.RO  
Categories: cs-LG, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2402.04894v1)  

---


**ABSTRACT**  
Autonomous robots are often employed for data collection due to their efficiency and low labour costs. A key task in robotic data acquisition is planning paths through an initially unknown environment to collect observations given platform-specific resource constraints, such as limited battery life. Adaptive online path planning in 3D environments is challenging due to the large set of valid actions and the presence of unknown occlusions. To address these issues, we propose a novel deep reinforcement learning approach for adaptively replanning robot paths to map targets of interest in unknown 3D environments. A key aspect of our approach is a dynamically constructed graph that restricts planning actions local to the robot, allowing us to quickly react to newly discovered obstacles and targets of interest. For replanning, we propose a new reward function that balances between exploring the unknown environment and exploiting online-collected data about the targets of interest. Our experiments show that our method enables more efficient target detection compared to state-of-the-art learning and non-learning baselines. We also show the applicability of our approach for orchard monitoring using an unmanned aerial vehicle in a photorealistic simulator.

{{</citation>}}


### (113/135) AINS: Affordable Indoor Navigation Solution via Line Color Identification Using Mono-Camera for Autonomous Vehicles (Nizamuddin Maitlo et al., 2024)

{{<citation>}}

Nizamuddin Maitlo, Nooruddin Noonari, Kaleem Arshid, Naveed Ahmed, Sathishkumar Duraisamy. (2024)  
**AINS: Affordable Indoor Navigation Solution via Line Color Identification Using Mono-Camera for Autonomous Vehicles**  

---
Primary Category: cs.RO  
Categories: cs-CV, cs-RO, cs.RO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2402.04750v1)  

---


**ABSTRACT**  
Recently, researchers have been exploring various ways to improve the effectiveness and efficiency of autonomous vehicles by researching new methods, especially for indoor scenarios. Autonomous Vehicles in indoor navigation systems possess many challenges especially the limited accuracy of GPS in indoor scenarios. Several, robust methods have been explored for autonomous vehicles in indoor scenarios to solve this problem, but the ineffectiveness of the proposed methods is the high deployment cost. To address the above-mentioned problems we have presented A low-cost indoor navigation method for autonomous vehicles called Affordable Indoor Navigation Solution (AINS) which is based on based on Monocular Camera. Our proposed solution is mainly based on a mono camera without relying on various huge or power-inefficient sensors to find the path, such as range finders and other navigation sensors. Our proposed method shows that we can deploy autonomous vehicles indoor navigation systems while taking into consideration the cost. We can observe that the results shown by our solution are better than existing solutions and we can reduce the estimated error and time consumption.

{{</citation>}}


### (114/135) Boosting Reinforcement Learning Algorithms in Continuous Robotic Reaching Tasks using Adaptive Potential Functions (Yifei Chen et al., 2024)

{{<citation>}}

Yifei Chen, Lambert Schomaker, Francisco Cruz. (2024)  
**Boosting Reinforcement Learning Algorithms in Continuous Robotic Reaching Tasks using Adaptive Potential Functions**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2402.04581v1)  

---


**ABSTRACT**  
In reinforcement learning, reward shaping is an efficient way to guide the learning process of an agent, as the reward can indicate the optimal policy of the task. The potential-based reward shaping framework was proposed to guarantee policy invariance after reward shaping, where a potential function is used to calculate the shaping reward. In former work, we proposed a novel adaptive potential function (APF) method to learn the potential function concurrently with training the agent based on information collected by the agent during the training process, and examined the APF method in discrete action space scenarios. This paper investigates the feasibility of using APF in solving continuous-reaching tasks in a real-world robotic scenario with continuous action space. We combine the Deep Deterministic Policy Gradient (DDPG) algorithm and our proposed method to form a new algorithm called APF-DDPG. To compare APF-DDPG with DDPG, we designed a task where the agent learns to control Baxter's right arm to reach a goal position. The experimental results show that the APF-DDPG algorithm outperforms the DDPG algorithm on both learning speed and robustness.

{{</citation>}}


### (115/135) A Comprehensive Survey of Cross-Domain Policy Transfer for Embodied Agents (Haoyi Niu et al., 2024)

{{<citation>}}

Haoyi Niu, Jianming Hu, Guyue Zhou, Xianyuan Zhan. (2024)  
**A Comprehensive Survey of Cross-Domain Policy Transfer for Embodied Agents**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-LG, cs-RO, cs.RO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2402.04580v1)  

---


**ABSTRACT**  
The burgeoning fields of robot learning and embodied AI have triggered an increasing demand for large quantities of data. However, collecting sufficient unbiased data from the target domain remains a challenge due to costly data collection processes and stringent safety requirements. Consequently, researchers often resort to data from easily accessible source domains, such as simulation and laboratory environments, for cost-effective data acquisition and rapid model iteration. Nevertheless, the environments and embodiments of these source domains can be quite different from their target domain counterparts, underscoring the need for effective cross-domain policy transfer approaches. In this paper, we conduct a systematic review of existing cross-domain policy transfer methods. Through a nuanced categorization of domain gaps, we encapsulate the overarching insights and design considerations of each problem setting. We also provide a high-level discussion about the key methodologies used in cross-domain policy transfer problems. Lastly, we summarize the open challenges that lie beyond the capabilities of current paradigms and discuss potential future directions in this field.

{{</citation>}}


### (116/135) FLAGRED -- Fuzzy Logic-based Algorithm Generalizing Risk Estimation for Drones (Samuel Hovington et al., 2024)

{{<citation>}}

Samuel Hovington, Louis Petit, Sophie Stratford, Philippe Hamelin, Alexis Lussier-Desbiens, Francois Ferland. (2024)  
**FLAGRED -- Fuzzy Logic-based Algorithm Generalizing Risk Estimation for Drones**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2402.04518v1)  

---


**ABSTRACT**  
Accurately estimating risk in real-time is essential for ensuring the safety and efficiency of many applications involving autonomous robot systems. This paper presents a novel, generalizable algorithm for the real-time estimation of risks created by external disturbances on multirotors. Unlike conventional approaches, our method requires no additional sensors, accurate drone models, or large datasets. It employs motor command data in a fuzzy logic system, overcoming barriers to real-world implementation. Inherently adaptable, it utilizes fundamental drone characteristics, making it applicable to diverse drone models. The efficiency of the algorithm has been confirmed through comprehensive real-world testing on various platforms. It proficiently discerned between high and low-risk scenarios resulting from diverse wind disturbances and varying thrust-to-weight ratios. The algorithm surpassed the widely-recognized ArduCopter wind estimation algorithm in performance and demonstrated its capability to promptly detect brief gusts.

{{</citation>}}


## cs.SI (3)



### (117/135) Community detection problem based on polarization measures:an application to Twitter: the COVID-19 case in Spain (Inmaculada Gutiérrez et al., 2024)

{{<citation>}}

Inmaculada Gutiérrez, Juan Antonio Guevara, Daniel Gómez, Javier Castro, Rosa Espínola. (2024)  
**Community detection problem based on polarization measures:an application to Twitter: the COVID-19 case in Spain**  

---
Primary Category: cs.SI  
Categories: 62-08,, G-3, cs-SI, cs.SI, math-ST, physics-soc-ph, stat-TH  
Keywords: Social Network, Twitter  
[Paper Link](http://arxiv.org/abs/2402.05028v1)  

---


**ABSTRACT**  
In this paper, we address one of the most important topics in the field of Social Networks Analysis: the community detection problem with additional information. That additional information is modeled by a fuzzy measure that represents the risk of polarization. Particularly, we are interested in dealing with the problem of taking into account the polarization of nodes in the community detection problem. Adding this type of information to the community detection problem makes it more realistic, as a community is more likely to be defined if the corresponding elements are willing to maintain a peaceful dialogue. The polarization capacity is modeled by a fuzzy measure based on the JDJpol measure of polarization related to two poles. We also present an efficient algorithm for finding groups whose elements are no polarized. Hereafter, we work in a real case. It is a network obtained from Twitter, concerning the political position against the Spanish government taken by several influential users. We analyze how the partitions obtained change when some additional information related to how polarized that society is, is added to the problem.

{{</citation>}}


### (118/135) Adaptive Hypergraph Network for Trust Prediction (Rongwei Xu et al., 2024)

{{<citation>}}

Rongwei Xu, Guanfeng Liu, Yan Wang, Xuyun Zhang, Kai Zheng, Xiaofang Zhou. (2024)  
**Adaptive Hypergraph Network for Trust Prediction**  

---
Primary Category: cs.SI  
Categories: cs-AI, cs-SI, cs.SI  
Keywords: Graph Convolutional Network  
[Paper Link](http://arxiv.org/abs/2402.05154v1)  

---


**ABSTRACT**  
Trust plays an essential role in an individual's decision-making. Traditional trust prediction models rely on pairwise correlations to infer potential relationships between users. However, in the real world, interactions between users are usually complicated rather than pairwise only. Hypergraphs offer a flexible approach to modeling these complex high-order correlations (not just pairwise connections), since hypergraphs can leverage hyperedeges to link more than two nodes. However, most hypergraph-based methods are generic and cannot be well applied to the trust prediction task. In this paper, we propose an Adaptive Hypergraph Network for Trust Prediction (AHNTP), a novel approach that improves trust prediction accuracy by using higher-order correlations. AHNTP utilizes Motif-based PageRank to capture high-order social influence information. In addition, it constructs hypergroups from both node-level and structure-level attributes to incorporate complex correlation information. Furthermore, AHNTP leverages adaptive hypergraph Graph Convolutional Network (GCN) layers and multilayer perceptrons (MLPs) to generate comprehensive user embeddings, facilitating trust relationship prediction. To enhance model generalization and robustness, we introduce a novel supervised contrastive learning loss for optimization. Extensive experiments demonstrate the superiority of our model over the state-of-the-art approaches in terms of trust prediction accuracy. The source code of this work can be accessed via https://github.com/Sherry-XU1995/AHNTP.

{{</citation>}}


### (119/135) Comparing Methods for Creating a National Random Sample of Twitter Users (Meysam Alizadeh et al., 2024)

{{<citation>}}

Meysam Alizadeh, Darya Zare, Zeynab Samei, Mohammadamin Alizadeh, Mael Kubli, Mohammadhadi Aliahmadi, Sarvenaz Ebrahimi, Fabrizio Gilardi. (2024)  
**Comparing Methods for Creating a National Random Sample of Twitter Users**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2402.04879v1)  

---


**ABSTRACT**  
Twitter data has been widely used by researchers across various social and computer science disciplines. A common aim when working with Twitter data is the construction of a random sample of users from a given country. However, while several methods have been proposed in the literature, their comparative performance is mostly unexplored. In this paper, we implement four methods to collect a random sample of Twitter users in the US: 1% Stream, Bounding Box, Location Query, and Language Query. Then, we compare the methods according to their tweet- and user-level metrics as well as their accuracy in estimating US population with and without using inclusion probabilities of various demographics. Our results show that the 1% Stream method performs differently than others and best for the construction of a population representative sample, though its statistical significance is questionable due to large confidence intervals. We discuss the conditions under which the 1% Stream method may not be suitable and suggest the Bounding Box method as the second-best method to use.

{{</citation>}}


## cs.MA (1)



### (120/135) Towards Generalizability of Multi-Agent Reinforcement Learning in Graphs with Recurrent Message Passing (Jannis Weil et al., 2024)

{{<citation>}}

Jannis Weil, Zhenghua Bao, Osama Abboud, Tobias Meuser. (2024)  
**Towards Generalizability of Multi-Agent Reinforcement Learning in Graphs with Recurrent Message Passing**  

---
Primary Category: cs.MA  
Categories: cs-AI, cs-MA, cs.MA  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2402.05027v1)  

---


**ABSTRACT**  
Graph-based environments pose unique challenges to multi-agent reinforcement learning. In decentralized approaches, agents operate within a given graph and make decisions based on partial or outdated observations. The size of the observed neighborhood limits the generalizability to different graphs and affects the reactivity of agents, the quality of the selected actions, and the communication overhead. This work focuses on generalizability and resolves the trade-off in observed neighborhood size with a continuous information flow in the whole graph. We propose a recurrent message-passing model that iterates with the environment's steps and allows nodes to create a global representation of the graph by exchanging messages with their neighbors. Agents receive the resulting learned graph observations based on their location in the graph. Our approach can be used in a decentralized manner at runtime and in combination with a reinforcement learning algorithm of choice. We evaluate our method across 1000 diverse graphs in the context of routing in communication networks and find that it enables agents to generalize and adapt to changes in the graph.

{{</citation>}}


## cs.DL (2)



### (121/135) What About the Data? A Mapping Study on Data Engineering for AI Systems (Petra Heck, 2024)

{{<citation>}}

Petra Heck. (2024)  
**What About the Data? A Mapping Study on Data Engineering for AI Systems**  

---
Primary Category: cs.DL  
Categories: cs-AI, cs-DB, cs-DL, cs.DL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2402.05156v1)  

---


**ABSTRACT**  
AI systems cannot exist without data. Now that AI models (data science and AI) have matured and are readily available to apply in practice, most organizations struggle with the data infrastructure to do so. There is a growing need for data engineers that know how to prepare data for AI systems or that can setup enterprise-wide data architectures for analytical projects. But until now, the data engineering part of AI engineering has not been getting much attention, in favor of discussing the modeling part. In this paper we aim to change this by perform a mapping study on data engineering for AI systems, i.e., AI data engineering. We found 25 relevant papers between January 2019 and June 2023, explaining AI data engineering activities. We identify which life cycle phases are covered, which technical solutions or architectures are proposed and which lessons learned are presented. We end by an overall discussion of the papers with implications for practitioners and researchers. This paper creates an overview of the body of knowledge on data engineering for AI. This overview is useful for practitioners to identify solutions and best practices as well as for researchers to identify gaps.

{{</citation>}}


### (122/135) Hierarchical Tree-structured Knowledge Graph For Academic Insight Survey (Jinghong Li et al., 2024)

{{<citation>}}

Jinghong Li, Huy Phan, Wen Gu, Koichi Ota, Shinobu Hasegawa. (2024)  
**Hierarchical Tree-structured Knowledge Graph For Academic Insight Survey**  

---
Primary Category: cs.DL  
Categories: cs-CL, cs-DL, cs-LG, cs.DL  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2402.04854v1)  

---


**ABSTRACT**  
Research surveys have always posed a challenge for beginner researchers who lack of research training. These researchers struggle to understand the directions within their research topic, and the discovery of new research findings within a short time. One way to provide intuitive assistance to beginner researchers is by offering relevant knowledge graphs(KG) and recommending related academic papers. However, existing navigation knowledge graphs primarily rely on keywords in the research field and often fail to present the logical hierarchy among multiple related papers clearly. Moreover, most recommendation systems for academic papers simply rely on high text similarity, which can leave researchers confused as to why a particular article is being recommended. They may lack of grasp important information about the insight connection between "Issue resolved" and "Issue finding" that they hope to obtain. To address these issues, this study aims to support research insight surveys for beginner researchers by establishing a hierarchical tree-structured knowledge graph that reflects the inheritance insight of research topics and the relevance insight among the academic papers.

{{</citation>}}


## cs.HC (3)



### (123/135) ChatScratch: An AI-Augmented System Toward Autonomous Visual Programming Learning for Children Aged 6-12 (Liuqing Chen et al., 2024)

{{<citation>}}

Liuqing Chen, Shuhong Xiao, Yunnong Chen, Ruoyu Wu, Yaxuan Song, Lingyun Sun. (2024)  
**ChatScratch: An AI-Augmented System Toward Autonomous Visual Programming Learning for Children Aged 6-12**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs-PL, cs.HC  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2402.04975v1)  

---


**ABSTRACT**  
As Computational Thinking (CT) continues to permeate younger age groups in K-12 education, established CT platforms such as Scratch face challenges in catering to these younger learners, particularly those in the elementary school (ages 6-12). Through formative investigation with Scratch experts, we uncover three key obstacles to children's autonomous Scratch learning: artist's block in project planning, bounded creativity in asset creation, and inadequate coding guidance during implementation. To address these barriers, we introduce ChatScratch, an AI-augmented system to facilitate autonomous programming learning for young children. ChatScratch employs structured interactive storyboards and visual cues to overcome artist's block, integrates digital drawing and advanced image generation technologies to elevate creativity, and leverages Scratch-specialized Large Language Models (LLMs) for professional coding guidance. Our study shows that, compared to Scratch, ChatScratch efficiently fosters autonomous programming learning, and contributes to the creation of high-quality, personally meaningful Scratch projects for children.

{{</citation>}}


### (124/135) Chatbots in Knowledge-Intensive Contexts: Comparing Intent and LLM-Based Systems (Samuel Kernan Freire et al., 2024)

{{<citation>}}

Samuel Kernan Freire, Chaofan Wang, Evangelos Niforatos. (2024)  
**Chatbots in Knowledge-Intensive Contexts: Comparing Intent and LLM-Based Systems**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: GPT, GPT-4, NLP  
[Paper Link](http://arxiv.org/abs/2402.04955v1)  

---


**ABSTRACT**  
Cognitive assistants (CA) are chatbots that provide context-aware support to human workers in knowledge-intensive tasks. Traditionally, cognitive assistants respond in specific ways to predefined user intents and conversation patterns. However, this rigidness does not handle the diversity of natural language well. Recent advances in natural language processing (NLP), powering large language models (LLM) such as GPT-4, Llama2, and Gemini, could enable CAs to converse in a more flexible, human-like manner. However, the additional degrees of freedom may have unforeseen consequences, especially in knowledge-intensive contexts where accuracy is crucial. As a preliminary step to assessing the potential of using LLMs in these contexts, we conducted a user study comparing an LLM-based CA to an intent-based system regarding interaction efficiency, user experience, workload, and usability. This revealed that LLM-based CAs exhibited better user experience, task completion rate, usability, and perceived performance than intent-based systems, suggesting that switching NLP techniques should be investigated further.

{{</citation>}}


### (125/135) Charting the COVID Long Haul Experience -- A Longitudinal Exploration of Symptoms, Activity, and Clinical Adherence (Jessica Pater et al., 2024)

{{<citation>}}

Jessica Pater, Shaan Chopra, Juliette Zaccour, Jeanne Carroll, Fayika Farhat Nova, Tammy Toscos, Shion Guha, Fen Lei Chang. (2024)  
**Charting the COVID Long Haul Experience -- A Longitudinal Exploration of Symptoms, Activity, and Clinical Adherence**  

---
Primary Category: cs.HC  
Categories: K-4, cs-CY, cs-HC, cs.HC  
Keywords: Clinical  
[Paper Link](http://arxiv.org/abs/2402.04937v1)  

---


**ABSTRACT**  
COVID Long Haul (CLH) is an emerging chronic illness with varied patient experiences. Our understanding of CLH is often limited to data from electronic health records (EHRs), such as diagnoses or problem lists, which do not capture the volatility and severity of symptoms or their impact. To better understand the unique presentation of CLH, we conducted a 3-month long cohort study with 14 CLH patients, collecting objective (EHR, daily Fitbit logs) and subjective (weekly surveys, interviews) data. Our findings reveal a complex presentation of symptoms, associated uncertainty, and the ensuing impact CLH has on patients' personal and professional lives. We identify patient needs, practices, and challenges around adhering to clinical recommendations, engaging with health data, and establishing "new normals" post COVID. We reflect on the potential found at the intersection of these various data streams and the persuasive heuristics possible when designing for this new population and their specific needs.

{{</citation>}}


## cs.CY (1)



### (126/135) What Values Do ImageNet-trained Classifiers Enact? (Will Penman et al., 2024)

{{<citation>}}

Will Penman, Joshua Babu, Abhinaya Raghunathan. (2024)  
**What Values Do ImageNet-trained Classifiers Enact?**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI, ImageNet  
[Paper Link](http://arxiv.org/abs/2402.04911v1)  

---


**ABSTRACT**  
We identify "values" as actions that classifiers take that speak to open questions of significant social concern. Investigating a classifier's values builds on studies of social bias that uncover how classifiers participate in social processes beyond their creators' forethought. In our case, this participation involves what counts as nutritious, what it means to be modest, and more. Unlike AI social bias, however, a classifier's values are not necessarily morally loathsome. Attending to image classifiers' values can facilitate public debate and introspection about the future of society. To substantiate these claims, we report on an extensive examination of both ImageNet training/validation data and ImageNet-trained classifiers with custom testing data. We identify perceptual decision boundaries in 118 categories that address open questions in society, and through quantitative testing of rival datasets we find that ImageNet-trained classifiers enact at least 7 values through their perceptual decisions. To contextualize these results, we develop a conceptual framework that integrates values, social bias, and accuracy, and we describe a rhetorical method for identifying how context affects the values that a classifier enacts. We also discover that classifier performance does not straightforwardly reflect the proportions of subgroups in a training set. Our findings bring a rich sense of the social world to ML researchers that can be applied to other domains beyond computer vision.

{{</citation>}}


## cs.DC (1)



### (127/135) Leveraging knowledge-as-a-service (KaaS) for QoS-aware resource management in multi-user video transcoding (Luis Costero et al., 2024)

{{<citation>}}

Luis Costero, Francisco D. Igual, Katzalin Olcoz, Francisco Tirado. (2024)  
**Leveraging knowledge-as-a-service (KaaS) for QoS-aware resource management in multi-user video transcoding**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2402.04891v1)  

---


**ABSTRACT**  
The coexistence of parallel applications in shared computing nodes, each one featuring different Quality of Service (QoS) requirements, carries out new challenges to improve resource occupation while keeping acceptable rates in terms of QoS. As more application-specific and system-wide metrics are included as QoS dimensions, or under situations in which resource-usage limits are strict, building and serving the most appropriate set of actions (application control knobs and system resource assignment) to concurrent applications in an automatic and optimal fashion becomes mandatory. In this paper, we propose strategies to build and serve this type of knowledge to concurrent applications by leveraging Reinforcement Learning techniques. Taking multi-user video transcoding as a driving example, our experimental results reveal an excellent adaptation of resource and knob management to heterogeneous QoS requests, and increases in the amount of concurrently served users up to 1.24x compared with alternative approaches considering homogeneous QoS requests.

{{</citation>}}


## cs.IT (1)



### (128/135) Detection Schemes with Low-Resolution ADCs and Spatial Oversampling for Transmission with Higher-Order Constellations in the Terahertz Band (Christian Forsch et al., 2024)

{{<citation>}}

Christian Forsch, Peter Zillmann, Osama Alrabadi, Stefan Brueck, Wolfgang Gerstacker. (2024)  
**Detection Schemes with Low-Resolution ADCs and Spatial Oversampling for Transmission with Higher-Order Constellations in the Terahertz Band**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, eess-SP, math-IT  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2402.04728v1)  

---


**ABSTRACT**  
In this work, we consider Terahertz (THz) communications with low-resolution uniform quantization and spatial oversampling at the receiver side. We compare different analog-to-digital converter (ADC) parametrizations in a fair manner by keeping the ADC power consumption constant. Here, 1-, 2-, and 3-bit quantization is investigated with different oversampling factors. We analytically compute the statistics of the detection variable, and we propose the optimal as well as several suboptimal detection schemes for arbitrary quantization resolutions. Then, we evaluate the symbol error rate (SER) of the different detectors for a 16- and a 64-ary quadrature amplitude modulation (QAM) constellation. The results indicate that there is a noticeable performance degradation of the suboptimal detection schemes compared to the optimal scheme when the constellation size is larger than the number of quantization levels. Furthermore, at low signal-to-noise ratios (SNRs), 1-bit quantization outperforms 2- and 3-bit quantization, respectively, even when employing higher-order constellations. We confirm our analytical results by Monte Carlo simulations. Both a pure line-of-sight (LoS) and a more realistically modeled indoor THz channel are considered. Then, we optimize the input signal constellation with respect to SER for 1-bit quantization. The results show that the minimum SER can be lowered significantly for 16-QAM by increasing the distance between the inner and outer points of the input constellation. For larger constellations, however, the achievable reduction of the minimum SER is much smaller compared to 16-QAM.

{{</citation>}}


## cs.GT (1)



### (129/135) Nash Equilibria in Reverse Temporal Voronoi Games (Simeon Pawlowski et al., 2024)

{{<citation>}}

Simeon Pawlowski, Vincent Froese. (2024)  
**Nash Equilibria in Reverse Temporal Voronoi Games**  

---
Primary Category: cs.GT  
Categories: cs-GT, cs.GT  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2402.04696v1)  

---


**ABSTRACT**  
We study Voronoi games on temporal graphs as introduced by Boehmer et al. (IJCAI 2021) where two players each select a vertex in a temporal graph with the goal of reaching the other vertices earlier than the other player. In this work, we consider the reverse temporal Voronoi game, that is, a player wants to maximize the number of vertices reaching her earlier than the other player. Since temporal distances in temporal graphs are not symmetric in general, this yields a different game. We investigate the difference between the two games with respect to the existence of Nash equilibria in various temporal graph classes including temporal trees, cycles, grids, cliques and split graphs. Our extensive results show that the two games indeed behave quite differently depending on the considered temporal graph class.

{{</citation>}}


## q-bio.BM (1)



### (130/135) Structure-Informed Protein Language Model (Zuobai Zhang et al., 2024)

{{<citation>}}

Zuobai Zhang, Jiarui Lu, Vijil Chenthamarakshan, Aurélie Lozano, Payel Das, Jian Tang. (2024)  
**Structure-Informed Protein Language Model**  

---
Primary Category: q-bio.BM  
Categories: cs-LG, q-bio-BM, q-bio.BM  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2402.05856v1)  

---


**ABSTRACT**  
Protein language models are a powerful tool for learning protein representations through pre-training on vast protein sequence datasets. However, traditional protein language models lack explicit structural supervision, despite its relevance to protein function. To address this issue, we introduce the integration of remote homology detection to distill structural information into protein language models without requiring explicit protein structures as input. We evaluate the impact of this structure-informed training on downstream protein function prediction tasks. Experimental results reveal consistent improvements in function annotation accuracy for EC number and GO term prediction. Performance on mutant datasets, however, varies based on the relationship between targeted properties and protein structures. This underscores the importance of considering this relationship when applying structure-aware training to protein function prediction tasks. Code and model weights are available at https://github.com/DeepGraphLearning/esm-s.

{{</citation>}}


## cs.CE (1)



### (131/135) Google Scholar is manipulatable (Hazem Ibrahim et al., 2024)

{{<citation>}}

Hazem Ibrahim, Fengyuan Liu, Yasir Zaki, Talal Rahwan. (2024)  
**Google Scholar is manipulatable**  

---
Primary Category: cs.CE  
Categories: cs-CE, cs-DL, cs-SI, cs.CE, physics-soc-ph  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2402.04607v1)  

---


**ABSTRACT**  
Citations are widely considered in scientists' evaluation. As such, scientists may be incentivized to inflate their citation counts. While previous literature has examined self-citations and citation cartels, it remains unclear whether scientists can purchase citations. Here, we compile a dataset of ~1.6 million profiles on Google Scholar to examine instances of citation fraud on the platform. We survey faculty at highly-ranked universities, and confirm that Google Scholar is widely used when evaluating scientists. Intrigued by a citation-boosting service that we unravelled during our investigation, we contacted the service while undercover as a fictional author, and managed to purchase 50 citations. These findings provide conclusive evidence that citations can be bought in bulk, and highlight the need to look beyond citation counts.

{{</citation>}}


## physics.chem-ph (1)



### (132/135) An Artificial Intelligence (AI) workflow for catalyst design and optimization (Nung Siong Lai et al., 2024)

{{<citation>}}

Nung Siong Lai, Yi Shen Tew, Xialin Zhong, Jun Yin, Jiali Li, Binhang Yan, Xiaonan Wang. (2024)  
**An Artificial Intelligence (AI) workflow for catalyst design and optimization**  

---
Primary Category: physics.chem-ph  
Categories: cs-LG, physics-chem-ph, physics.chem-ph  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2402.04557v1)  

---


**ABSTRACT**  
In the pursuit of novel catalyst development to address pressing environmental concerns and energy demand, conventional design and optimization methods often fall short due to the complexity and vastness of the catalyst parameter space. The advent of Machine Learning (ML) has ushered in a new era in the field of catalyst optimization, offering potential solutions to the shortcomings of traditional techniques. However, existing methods fail to effectively harness the wealth of information contained within the burgeoning body of scientific literature on catalyst synthesis. To address this gap, this study proposes an innovative Artificial Intelligence (AI) workflow that integrates Large Language Models (LLMs), Bayesian optimization, and an active learning loop to expedite and enhance catalyst optimization. Our methodology combines advanced language understanding with robust optimization strategies, effectively translating knowledge extracted from diverse literature into actionable parameters for practical experimentation and optimization. In this article, we demonstrate the application of this AI workflow in the optimization of catalyst synthesis for ammonia production. The results underscore the workflow's ability to streamline the catalyst development process, offering a swift, resource-efficient, and high-precision alternative to conventional methods.

{{</citation>}}


## cs.NI (1)



### (133/135) A Deep Reinforcement Learning Approach for Adaptive Traffic Routing in Next-gen Networks (Akshita Abrol et al., 2024)

{{<citation>}}

Akshita Abrol, Purnima Murali Mohan, Tram Truong-Huu. (2024)  
**A Deep Reinforcement Learning Approach for Adaptive Traffic Routing in Next-gen Networks**  

---
Primary Category: cs.NI  
Categories: cs-AI, cs-NI, cs.NI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2402.04515v1)  

---


**ABSTRACT**  
Next-gen networks require significant evolution of management to enable automation and adaptively adjust network configuration based on traffic dynamics. The advent of software-defined networking (SDN) and programmable switches enables flexibility and programmability. However, traditional techniques that decide traffic policies are usually based on hand-crafted programming optimization and heuristic algorithms. These techniques make non-realistic assumptions, e.g., considering static network load and topology, to obtain tractable solutions, which are inadequate for next-gen networks. In this paper, we design and develop a deep reinforcement learning (DRL) approach for adaptive traffic routing. We design a deep graph convolutional neural network (DGCNN) integrated into the DRL framework to learn the traffic behavior from not only the network topology but also link and node attributes. We adopt the Deep Q-Learning technique to train the DGCNN model in the DRL framework without the need for a labeled training dataset, enabling the framework to quickly adapt to traffic dynamics. The model leverages q-value estimates to select the routing path for every traffic flow request, balancing exploration and exploitation. We perform extensive experiments with various traffic patterns and compare the performance of the proposed approach with the Open Shortest Path First (OSPF) protocol. The experimental results show the effectiveness and adaptiveness of the proposed framework by increasing the network throughput by up to 7.8% and reducing the traffic delay by up to 16.1% compared to OSPF.

{{</citation>}}


## astro-ph.IM (1)



### (134/135) Processing All-Sky Images At Scale On The Amazon Cloud: A HiPS Example (G. Bruce Berriman et al., 2024)

{{<citation>}}

G. Bruce Berriman, John C. Good. (2024)  
**Processing All-Sky Images At Scale On The Amazon Cloud: A HiPS Example**  

---
Primary Category: astro-ph.IM  
Categories: astro-ph-IM, astro-ph.IM, cs-DC  
Keywords: Amazon  
[Paper Link](http://arxiv.org/abs/2402.04506v1)  

---


**ABSTRACT**  
We report here on a project that has developed a practical approach to processing all-sky image collections on cloud platforms, using as an exemplar application the creation of three-color Hierarchical Progressive Survey (HiPS) maps of the 2MASS data set with the Montage Image Mosaic Engine on Amazon Web Services. We will emphasize issues that must be considered by scientists wishing to use cloud platforms to perform such parallel processing, so providing a guide for scientists wishing to exploit cloud platforms for similar large-scale processing. A HiPS map is based on the HEALPix sky-tiling scheme. Progressive zooming of a HiPS map reveals an image sampled at ever smaller or larger spatial scales that are defined by the HEALPix standard. Briefly, the approach used by Montage involves creating a base mosaic at the lowest required HEALPix level, usually chosen to match as closely as possible the spatial sampling of the input images, then cutting out the HiPS cells in PNG format from this mosaic. The process is repeated at successive HEALPix levels to create a nested collection of FITS files, from which PNG files are created that are shown in HiPS viewers. Stretching FITS files to produce PNGs is based on an image histogram. For composite regions (up and including the whole sky), the histograms for each tile can be combined to create a composite histogram for the region. Using this single histogram for each of the individual FITS files means all the PNGs are on the same brightness scale and displaying them side by side in a HiPS viewer produces a continuous uniform map across the entire sky.

{{</citation>}}


## stat.ML (1)



### (135/135) A Primal-Dual Algorithm for Offline Constrained Reinforcement Learning with Low-Rank MDPs (Kihyuk Hong et al., 2024)

{{<citation>}}

Kihyuk Hong, Ambuj Tewari. (2024)  
**A Primal-Dual Algorithm for Offline Constrained Reinforcement Learning with Low-Rank MDPs**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-ML, stat.ML  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2402.04493v1)  

---


**ABSTRACT**  
Offline reinforcement learning (RL) aims to learn a policy that maximizes the expected cumulative reward using a pre-collected dataset. Offline RL with low-rank MDPs or general function approximation has been widely studied recently, but existing algorithms with sample complexity $O(\epsilon^{-2})$ for finding an $\epsilon$-optimal policy either require a uniform data coverage assumptions or are computationally inefficient. In this paper, we propose a primal dual algorithm for offline RL with low-rank MDPs in the discounted infinite-horizon setting. Our algorithm is the first computationally efficient algorithm in this setting that achieves sample complexity of $O(\epsilon^{-2})$ with partial data coverage assumption. This improves upon a recent work that requires $O(\epsilon^{-4})$ samples. Moreover, our algorithm extends the previous work to the offline constrained RL setting by supporting constraints on additional reward signals.

{{</citation>}}
