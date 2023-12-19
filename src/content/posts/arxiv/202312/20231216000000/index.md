---
draft: false
title: "arXiv @ 2023.12.16"
date: 2023-12-16
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.12.16"
    identifier: arxiv_20231216
    parent: 202312_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.LG (33)](#cslg-33)
- [cs.CY (4)](#cscy-4)
- [cs.AR (1)](#csar-1)
- [cs.AI (17)](#csai-17)
- [cs.RO (7)](#csro-7)
- [cs.CL (23)](#cscl-23)
- [cs.CR (4)](#cscr-4)
- [eess.SY (1)](#eesssy-1)
- [cs.CV (49)](#cscv-49)
- [cs.SI (2)](#cssi-2)
- [cs.NI (2)](#csni-2)
- [cs.IR (2)](#csir-2)
- [quant-ph (1)](#quant-ph-1)
- [cs.SD (5)](#cssd-5)
- [cs.SE (2)](#csse-2)
- [cs.DC (1)](#csdc-1)
- [cs.NE (1)](#csne-1)
- [eess.AS (5)](#eessas-5)
- [cs.CG (1)](#cscg-1)
- [eess.IV (3)](#eessiv-3)
- [cs.IT (2)](#csit-2)
- [physics.med-ph (1)](#physicsmed-ph-1)
- [cs.MA (1)](#csma-1)
- [cs.PL (1)](#cspl-1)

## cs.LG (33)



### (1/169) Unbiasing Enhanced Sampling on a High-dimensional Free Energy Surface with Deep Generative Model (Yikai Liu et al., 2023)

{{<citation>}}

Yikai Liu, Tushar K. Ghosh, Guang Lin, Ming Chen. (2023)  
**Unbiasing Enhanced Sampling on a High-dimensional Free Energy Surface with Deep Generative Model**  

---
Primary Category: cs.LG  
Categories: cond-mat-stat-mech, cs-LG, cs.LG, physics-chem-ph  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2312.09404v2)  

---


**ABSTRACT**  
Biased enhanced sampling methods utilizing collective variables (CVs) are powerful tools for sampling conformational ensembles. Due to high intrinsic dimensions, efficiently generating conformational ensembles for complex systems requires enhanced sampling on high-dimensional free energy surfaces. While methods like temperature-accelerated molecular dynamics (TAMD) can adopt many CVs in a simulation, unbiasing the simulation requires accurate modeling of a high-dimensional CV probability distribution, which is challenging for traditional density estimation techniques. Here we propose an unbiasing method based on the score-based diffusion model, a deep generative learning method that excels in density estimation across complex data landscapes. We test the score-based diffusion unbiasing method on TAMD simulations. The results demonstrate that this unbiasing approach significantly outperforms traditional unbiasing methods, and can generate accurate unbiased conformational ensembles for simulations with a number of CVs higher than usual ranges.

{{</citation>}}


### (2/169) Exploiting Symmetric Temporally Sparse BPTT for Efficient RNN Training (Xi Chen et al., 2023)

{{<citation>}}

Xi Chen, Chang Gao, Zuowen Wang, Longbiao Cheng, Sheng Zhou, Shih-Chii Liu, Tobi Delbruck. (2023)  
**Exploiting Symmetric Temporally Sparse BPTT for Efficient RNN Training**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-NE, cs.LG  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2312.09391v1)  

---


**ABSTRACT**  
Recurrent Neural Networks (RNNs) are useful in temporal sequence tasks. However, training RNNs involves dense matrix multiplications which require hardware that can support a large number of arithmetic operations and memory accesses. Implementing online training of RNNs on the edge calls for optimized algorithms for an efficient deployment on hardware. Inspired by the spiking neuron model, the Delta RNN exploits temporal sparsity during inference by skipping over the update of hidden states from those inactivated neurons whose change of activation across two timesteps is below a defined threshold. This work describes a training algorithm for Delta RNNs that exploits temporal sparsity in the backward propagation phase to reduce computational requirements for training on the edge. Due to the symmetric computation graphs of forward and backward propagation during training, the gradient computation of inactivated neurons can be skipped. Results show a reduction of $\sim$80% in matrix operations for training a 56k parameter Delta LSTM on the Fluent Speech Commands dataset with negligible accuracy loss. Logic simulations of a hardware accelerator designed for the training algorithm show 2-10X speedup in matrix computations for an activation sparsity range of 50%-90%. Additionally, we show that the proposed Delta RNN training will be useful for online incremental learning on edge devices with limited computing resources.

{{</citation>}}


### (3/169) Well-calibrated Confidence Measures for Multi-label Text Classification with a Large Number of Labels (Lysimachos Maltoudoglou et al., 2023)

{{<citation>}}

Lysimachos Maltoudoglou, Andreas Paisios, Ladislav Lenc, Jiří Martínek, Pavel Král, Harris Papadopoulos. (2023)  
**Well-calibrated Confidence Measures for Multi-label Text Classification with a Large Number of Labels**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG, stat-ML  
Keywords: Text Classification  
[Paper Link](http://arxiv.org/abs/2312.09304v1)  

---


**ABSTRACT**  
We extend our previous work on Inductive Conformal Prediction (ICP) for multi-label text classification and present a novel approach for addressing the computational inefficiency of the Label Powerset (LP) ICP, arrising when dealing with a high number of unique labels. We present experimental results using the original and the proposed efficient LP-ICP on two English and one Czech language data-sets. Specifically, we apply the LP-ICP on three deep Artificial Neural Network (ANN) classifiers of two types: one based on contextualised (bert) and two on non-contextualised (word2vec) word-embeddings. In the LP-ICP setting we assign nonconformity scores to label-sets from which the corresponding p-values and prediction-sets are determined. Our approach deals with the increased computational burden of LP by eliminating from consideration a significant number of label-sets that will surely have p-values below the specified significance level. This reduces dramatically the computational complexity of the approach while fully respecting the standard CP guarantees. Our experimental results show that the contextualised-based classifier surpasses the non-contextualised-based ones and obtains state-of-the-art performance for all data-sets examined. The good performance of the underlying classifiers is carried on to their ICP counterparts without any significant accuracy loss, but with the added benefits of ICP, i.e. the confidence information encapsulated in the prediction sets. We experimentally demonstrate that the resulting prediction sets can be tight enough to be practically useful even though the set of all possible label-sets contains more than $1e+16$ combinations. Additionally, the empirical error rates of the obtained prediction-sets confirm that our outputs are well-calibrated.

{{</citation>}}


### (4/169) TinyGSM: achieving >80% on GSM8k with small language models (Bingbin Liu et al., 2023)

{{<citation>}}

Bingbin Liu, Sebastien Bubeck, Ronen Eldan, Janardhan Kulkarni, Yuanzhi Li, Anh Nguyen, Rachel Ward, Yi Zhang. (2023)  
**TinyGSM: achieving >80% on GSM8k with small language models**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: GPT, GPT-3.5  
[Paper Link](http://arxiv.org/abs/2312.09241v1)  

---


**ABSTRACT**  
Small-scale models offer various computational advantages, and yet to which extent size is critical for problem-solving abilities remains an open question. Specifically for solving grade school math, the smallest model size so far required to break the 80\% barrier on the GSM8K benchmark remains to be 34B. Our work studies how high-quality datasets may be the key for small language models to acquire mathematical reasoning. We introduce \texttt{TinyGSM}, a synthetic dataset of 12.3M grade school math problems paired with Python solutions, generated fully by GPT-3.5. After finetuning on \texttt{TinyGSM}, we find that a duo of a 1.3B generation model and a 1.3B verifier model can achieve 81.5\% accuracy, outperforming existing models that are orders of magnitude larger. This also rivals the performance of the GPT-3.5 ``teacher'' model (77.4\%), from which our model's training data is generated. Our approach is simple and has two key components: 1) the high-quality dataset \texttt{TinyGSM}, 2) the use of a verifier, which selects the final outputs from multiple candidate generations.

{{</citation>}}


### (5/169) A framework for conditional diffusion modelling with applications in motif scaffolding for protein design (Kieran Didi et al., 2023)

{{<citation>}}

Kieran Didi, Francisco Vargas, Simon V Mathis, Vincent Dutordoir, Emile Mathieu, Urszula J Komorowska, Pietro Lio. (2023)  
**A framework for conditional diffusion modelling with applications in motif scaffolding for protein design**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, q-bio-BM  
Keywords: Computer Vision  
[Paper Link](http://arxiv.org/abs/2312.09236v1)  

---


**ABSTRACT**  
Many protein design applications, such as binder or enzyme design, require scaffolding a structural motif with high precision. Generative modelling paradigms based on denoising diffusion processes emerged as a leading candidate to address this motif scaffolding problem and have shown early experimental success in some cases. In the diffusion paradigm, motif scaffolding is treated as a conditional generation task, and several conditional generation protocols were proposed or imported from the Computer Vision literature. However, most of these protocols are motivated heuristically, e.g. via analogies to Langevin dynamics, and lack a unifying framework, obscuring connections between the different approaches. In this work, we unify conditional training and conditional sampling procedures under one common framework based on the mathematically well-understood Doob's h-transform. This new perspective allows us to draw connections between existing methods and propose a new variation on existing conditional training protocols. We illustrate the effectiveness of this new protocol in both, image outpainting and motif scaffolding and find that it outperforms standard methods.

{{</citation>}}


### (6/169) Successor Heads: Recurring, Interpretable Attention Heads In The Wild (Rhys Gould et al., 2023)

{{<citation>}}

Rhys Gould, Euan Ong, George Ogden, Arthur Conmy. (2023)  
**Successor Heads: Recurring, Interpretable Attention Heads In The Wild**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: Attention, GPT  
[Paper Link](http://arxiv.org/abs/2312.09230v1)  

---


**ABSTRACT**  
In this work we present successor heads: attention heads that increment tokens with a natural ordering, such as numbers, months, and days. For example, successor heads increment 'Monday' into 'Tuesday'. We explain the successor head behavior with an approach rooted in mechanistic interpretability, the field that aims to explain how models complete tasks in human-understandable terms. Existing research in this area has found interpretable language model components in small toy models. However, results in toy models have not yet led to insights that explain the internals of frontier models and little is currently understood about the internal operations of large language models. In this paper, we analyze the behavior of successor heads in large language models (LLMs) and find that they implement abstract representations that are common to different architectures. They form in LLMs with as few as 31 million parameters, and at least as many as 12 billion parameters, such as GPT-2, Pythia, and Llama-2. We find a set of 'mod-10 features' that underlie how successor heads increment in LLMs across different architectures and sizes. We perform vector arithmetic with these features to edit head behavior and provide insights into numeric representations within LLMs. Additionally, we study the behavior of successor heads on natural language data, identifying interpretable polysemanticity in a Pythia successor head.

{{</citation>}}


### (7/169) DIRECT: Deep Active Learning under Imbalance and Label Noise (Shyam Nuggehalli et al., 2023)

{{<citation>}}

Shyam Nuggehalli, Jifan Zhang, Lalit Jain, Robert Nowak. (2023)  
**DIRECT: Deep Active Learning under Imbalance and Label Noise**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2312.09196v1)  

---


**ABSTRACT**  
Class imbalance is a prevalent issue in real world machine learning applications, often leading to poor performance in rare and minority classes. With an abundance of wild unlabeled data, active learning is perhaps the most effective technique in solving the problem at its root -- collecting a more balanced and informative set of labeled examples during annotation. In this work, we propose a novel algorithm that first identifies the class separation threshold and then annotate the most uncertain examples from the minority classes, close to the separation threshold. Through a novel reduction to one-dimensional active learning, our algorithm DIRECT is able to leverage the classic active learning literature to address issues such as batch labeling and tolerance towards label noise. Compared to existing algorithms, our algorithm saves more than 15\% of the annotation budget compared to state-of-art active learning algorithm and more than 90\% of annotation budget compared to random sampling.

{{</citation>}}


### (8/169) Vision-Language Models as a Source of Rewards (Kate Baumli et al., 2023)

{{<citation>}}

Kate Baumli, Satinder Baveja, Feryal Behbahani, Harris Chan, Gheorghe Comanici, Sebastian Flennerhag, Maxime Gazeau, Kristian Holsheimer, Dan Horgan, Michael Laskin, Clare Lyle, Hussain Masoom, Kay McKinney, Volodymyr Mnih, Alexander Neitz, Fabio Pardo, Jack Parker-Holder, John Quan, Tim Rocktäschel, Himanshu Sahni, Tom Schaul, Yannick Schroecker, Stephen Spencer, Richie Steigerwald, Luyu Wang, Lei Zhang. (2023)  
**Vision-Language Models as a Source of Rewards**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.09187v1)  

---


**ABSTRACT**  
Building generalist agents that can accomplish many goals in rich open-ended environments is one of the research frontiers for reinforcement learning. A key limiting factor for building generalist agents with RL has been the need for a large number of reward functions for achieving different goals. We investigate the feasibility of using off-the-shelf vision-language models, or VLMs, as sources of rewards for reinforcement learning agents. We show how rewards for visual achievement of a variety of language goals can be derived from the CLIP family of models, and used to train RL agents that can achieve a variety of language goals. We showcase this approach in two distinct visual domains and present a scaling trend showing how larger VLMs lead to more accurate rewards for visual goal achievement, which in turn produces more capable RL agents.

{{</citation>}}


### (9/169) Split-Ensemble: Efficient OOD-aware Ensemble via Task and Model Splitting (Anthony Chen et al., 2023)

{{<citation>}}

Anthony Chen, Huanrui Yang, Yulu Gan, Denis A Gudovskiy, Zhen Dong, Haofan Wang, Tomoyuki Okuno, Yohei Nakata, Shanghang Zhang, Kurt Keutzer. (2023)  
**Split-Ensemble: Efficient OOD-aware Ensemble via Task and Model Splitting**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2312.09148v1)  

---


**ABSTRACT**  
Uncertainty estimation is crucial for machine learning models to detect out-of-distribution (OOD) inputs. However, the conventional discriminative deep learning classifiers produce uncalibrated closed-set predictions for OOD data. A more robust classifiers with the uncertainty estimation typically require a potentially unavailable OOD dataset for outlier exposure training, or a considerable amount of additional memory and compute to build ensemble models. In this work, we improve on uncertainty estimation without extra OOD data or additional inference costs using an alternative Split-Ensemble method. Specifically, we propose a novel subtask-splitting ensemble training objective, where a common multiclass classification task is split into several complementary subtasks. Then, each subtask's training data can be considered as OOD to the other subtasks. Diverse submodels can therefore be trained on each subtask with OOD-aware objectives. The subtask-splitting objective enables us to share low-level features across submodels to avoid parameter and computational overheads. In particular, we build a tree-like Split-Ensemble architecture by performing iterative splitting and pruning from a shared backbone model, where each branch serves as a submodel corresponding to a subtask. This leads to improved accuracy and uncertainty estimation across submodels under a fixed ensemble computation budget. Empirical study with ResNet-18 backbone shows Split-Ensemble, without additional computation cost, improves accuracy over a single model by 0.8%, 1.8%, and 25.5% on CIFAR-10, CIFAR-100, and Tiny-ImageNet, respectively. OOD detection for the same backbone and in-distribution datasets surpasses a single model baseline by, correspondingly, 2.2%, 8.1%, and 29.6% mean AUROC. Codes will be publicly available at https://antonioo-c.github.io/projects/split-ensemble

{{</citation>}}


### (10/169) Less is more -- the Dispatcher/ Executor principle for multi-task Reinforcement Learning (Martin Riedmiller et al., 2023)

{{<citation>}}

Martin Riedmiller, Tim Hertweck, Roland Hafner. (2023)  
**Less is more -- the Dispatcher/ Executor principle for multi-task Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-RO, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.09120v1)  

---


**ABSTRACT**  
Humans instinctively know how to neglect details when it comes to solve complex decision making problems in environments with unforeseeable variations. This abstraction process seems to be a vital property for most biological systems and helps to 'abstract away' unnecessary details and boost generalisation. In this work we introduce the dispatcher/ executor principle for the design of multi-task Reinforcement Learning controllers. It suggests to partition the controller in two entities, one that understands the task (the dispatcher) and one that computes the controls for the specific device (the executor) - and to connect these two by a strongly regularizing communication channel. The core rationale behind this position paper is that changes in structure and design principles can improve generalisation properties and drastically enforce data-efficiency. It is in some sense a 'yes, and ...' response to the current trend of using large neural networks trained on vast amounts of data and bet on emerging generalisation properties. While we agree on the power of scaling - in the sense of Sutton's 'bitter lesson' - we will give some evidence, that considering structure and adding design principles can be a valuable and critical component in particular when data is not abundant and infinite, but is a precious resource.

{{</citation>}}


### (11/169) COMBHelper: A Neural Approach to Reduce Search Space for Graph Combinatorial Problems (Hao Tian et al., 2023)

{{<citation>}}

Hao Tian, Sourav Medya, Wei Ye. (2023)  
**COMBHelper: A Neural Approach to Reduce Search Space for Graph Combinatorial Problems**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-NE, cs.LG  
Keywords: GNN, Graph Neural Network, Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2312.09086v1)  

---


**ABSTRACT**  
Combinatorial Optimization (CO) problems over graphs appear routinely in many applications such as in optimizing traffic, viral marketing in social networks, and matching for job allocation. Due to their combinatorial nature, these problems are often NP-hard. Existing approximation algorithms and heuristics rely on the search space to find the solutions and become time-consuming when this space is large. In this paper, we design a neural method called COMBHelper to reduce this space and thus improve the efficiency of the traditional CO algorithms based on node selection. Specifically, it employs a Graph Neural Network (GNN) to identify promising nodes for the solution set. This pruned search space is then fed to the traditional CO algorithms. COMBHelper also uses a Knowledge Distillation (KD) module and a problem-specific boosting module to bring further efficiency and efficacy. Our extensive experiments show that the traditional CO algorithms with COMBHelper are at least 2 times faster than their original versions.

{{</citation>}}


### (12/169) ReCoRe: Regularized Contrastive Representation Learning of World Model (Rudra P. K. Poudel et al., 2023)

{{<citation>}}

Rudra P. K. Poudel, Harit Pandya, Stephan Liwicki, Roberto Cipolla. (2023)  
**ReCoRe: Regularized Contrastive Representation Learning of World Model**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs-RO, cs.LG, stat-ML  
Keywords: Reinforcement Learning, Representation Learning  
[Paper Link](http://arxiv.org/abs/2312.09056v1)  

---


**ABSTRACT**  
While recent model-free Reinforcement Learning (RL) methods have demonstrated human-level effectiveness in gaming environments, their success in everyday tasks like visual navigation has been limited, particularly under significant appearance variations. This limitation arises from (i) poor sample efficiency and (ii) over-fitting to training scenarios. To address these challenges, we present a world model that learns invariant features using (i) contrastive unsupervised learning and (ii) an intervention-invariant regularizer. Learning an explicit representation of the world dynamics i.e. a world model, improves sample efficiency while contrastive learning implicitly enforces learning of invariant features, which improves generalization. However, the naive integration of contrastive loss to world models fails due to a lack of supervisory signals to the visual encoder, as world-model-based RL methods independently optimize representation learning and agent policy. To overcome this issue, we propose an intervention-invariant regularizer in the form of an auxiliary task such as depth prediction, image denoising, etc., that explicitly enforces invariance to style-interventions. Our method outperforms current state-of-the-art model-based and model-free RL methods and significantly on out-of-distribution point navigation task evaluated on the iGibson benchmark. We further demonstrate that our approach, with only visual observations, outperforms recent language-guided foundation models for point navigation, which is essential for deployment on robots with limited computation capabilities. Finally, we demonstrate that our proposed model excels at the sim-to-real transfer of its perception module on Gibson benchmark.

{{</citation>}}


### (13/169) Graph Neural Networks with Diverse Spectral Filtering (Jingwei Guo et al., 2023)

{{<citation>}}

Jingwei Guo, Kaizhu Huang, Xinping Yi, Rui Zhang. (2023)  
**Graph Neural Networks with Diverse Spectral Filtering**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SI, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2312.09041v1)  

---


**ABSTRACT**  
Spectral Graph Neural Networks (GNNs) have achieved tremendous success in graph machine learning, with polynomial filters applied for graph convolutions, where all nodes share the identical filter weights to mine their local contexts. Despite the success, existing spectral GNNs usually fail to deal with complex networks (e.g., WWW) due to such homogeneous spectral filtering setting that ignores the regional heterogeneity as typically seen in real-world networks. To tackle this issue, we propose a novel diverse spectral filtering (DSF) framework, which automatically learns node-specific filter weights to exploit the varying local structure properly. Particularly, the diverse filter weights consist of two components -- A global one shared among all nodes, and a local one that varies along network edges to reflect node difference arising from distinct graph parts -- to balance between local and global information. As such, not only can the global graph characteristics be captured, but also the diverse local patterns can be mined with awareness of different node positions. Interestingly, we formulate a novel optimization problem to assist in learning diverse filters, which also enables us to enhance any spectral GNNs with our DSF framework. We showcase the proposed framework on three state-of-the-arts including GPR-GNN, BernNet, and JacobiConv. Extensive experiments over 10 benchmark datasets demonstrate that our framework can consistently boost model performance by up to 4.92% in node classification tasks, producing diverse filters with enhanced interpretability. Code is available at \url{https://github.com/jingweio/DSF}.

{{</citation>}}


### (14/169) Uncertainty in GNN Learning Evaluations: A Comparison Between Measures for Quantifying Randomness in GNN Community Detection (William Leeney et al., 2023)

{{<citation>}}

William Leeney, Ryan McConville. (2023)  
**Uncertainty in GNN Learning Evaluations: A Comparison Between Measures for Quantifying Randomness in GNN Community Detection**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SI, cs.LG, physics-soc-ph  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2312.09015v1)  

---


**ABSTRACT**  
(1) The enhanced capability of Graph Neural Networks (GNNs) in unsupervised community detection of clustered nodes is attributed to their capacity to encode both the connectivity and feature information spaces of graphs. The identification of latent communities holds practical significance in various domains, from social networks to genomics. Current real-world performance benchmarks are perplexing due to the multitude of decisions influencing GNN evaluations for this task. (2) Three metrics are compared to assess the consistency of algorithm rankings in the presence of randomness. The consistency and quality of performance between the results under a hyperparameter optimisation with the default hyperparameters is evaluated. (3) The results compare hyperparameter optimisation with default hyperparameters, revealing a significant performance loss when neglecting hyperparameter investigation. A comparison of metrics indicates that ties in ranks can substantially alter the quantification of randomness. (4) Ensuring adherence to the same evaluation criteria may result in notable differences in the reported performance of methods for this task. The $W$ Randomness coefficient, based on the Wasserstein distance, is identified as providing the most robust assessment of randomness.

{{</citation>}}


### (15/169) FedSSA: Semantic Similarity-based Aggregation for Efficient Model-Heterogeneous Personalized Federated Learning (Liping Yi et al., 2023)

{{<citation>}}

Liping Yi, Han Yu, Zhuan Shi, Gang Wang, Xiaoguang Liu. (2023)  
**FedSSA: Semantic Similarity-based Aggregation for Efficient Model-Heterogeneous Personalized Federated Learning**  

---
Primary Category: cs.LG  
Categories: cs-DC, cs-LG, cs.LG  
Keywords: Semantic Similarity  
[Paper Link](http://arxiv.org/abs/2312.09006v1)  

---


**ABSTRACT**  
Federated learning (FL) is a privacy-preserving collaboratively machine learning paradigm. Traditional FL requires all data owners (a.k.a. FL clients) to train the same local model. This design is not well-suited for scenarios involving data and/or system heterogeneity. Model-Heterogeneous Personalized FL (MHPFL) has emerged to address this challenge. Existing MHPFL approaches often rely on having a public dataset with the same nature of the learning task, or incur high computation and communication costs. To address these limitations, we propose the Federated Semantic Similarity Aggregation (FedSSA) approach, which splits each client's model into a heterogeneous (structure-different) feature extractor and a homogeneous (structure-same) classification header. It performs local-to-global knowledge transfer via semantic similarity-based header parameter aggregation. In addition, global-to-local knowledge transfer is achieved via an adaptive parameter stabilization strategy which fuses the seen-class parameters of historical local headers with that of the latest global header for each client. In this way, FedSSA does not rely on public datasets, while only requiring partial header parameter transmission (thereby saving costs). Theoretical analysis proves the convergence of FedSSA. Extensive experiments demonstrate that FedSSA achieves up to $3.62 \times\%$ higher accuracy, $15.54$ times higher communication efficiency, and $15.52 \times$ higher computational efficiency compared to 7 state-of-the-art MHPFL baselines.

{{</citation>}}


### (16/169) LiFT: Unsupervised Reinforcement Learning with Foundation Models as Teachers (Taewook Nam et al., 2023)

{{<citation>}}

Taewook Nam, Juyong Lee, Jesse Zhang, Sung Ju Hwang, Joseph J. Lim, Karl Pertsch. (2023)  
**LiFT: Unsupervised Reinforcement Learning with Foundation Models as Teachers**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-RO, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.08958v1)  

---


**ABSTRACT**  
We propose a framework that leverages foundation models as teachers, guiding a reinforcement learning agent to acquire semantically meaningful behavior without human feedback. In our framework, the agent receives task instructions grounded in a training environment from large language models. Then, a vision-language model guides the agent in learning the multi-task language-conditioned policy by providing reward feedback. We demonstrate that our method can learn semantically meaningful skills in a challenging open-ended MineDojo environment while prior unsupervised skill discovery methods struggle. Additionally, we discuss observed challenges of using off-the-shelf foundation models as teachers and our efforts to address them.

{{</citation>}}


### (17/169) LSTM Network Analysis of Vehicle-Type Fatalities on Great Britain's Roads (Abiodun Finbarrs Oketunji et al., 2023)

{{<citation>}}

Abiodun Finbarrs Oketunji, James Hanify, Salter Heffron-Smith. (2023)  
**LSTM Network Analysis of Vehicle-Type Fatalities on Great Britain's Roads**  

---
Primary Category: cs.LG  
Categories: I-2-7, cs-AI, cs-LG, cs-SE, cs.LG  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2312.08948v2)  

---


**ABSTRACT**  
This study harnesses the predictive capabilities of Long Short-Term Memory (LSTM) networks to analyse and predict road traffic accidents in Great Britain. It addresses the challenge of traffic accident forecasting, which is paramount for devising effective preventive measures. We utilised an extensive dataset encompassing reported collisions, casualties, and vehicles involvements from 1926 to 2022, provided by the Department for Transport (DfT). The data underwent stringent processing to rectify missing values and normalise features, ensuring robust LSTM network input.

{{</citation>}}


### (18/169) BiPFT: Binary Pre-trained Foundation Transformer with Low-rank Estimation of Binarization Residual Polynomials (Xingrun Xing et al., 2023)

{{<citation>}}

Xingrun Xing, Li Du, Xinyuan Wang, Xianlin Zeng, Yequan Wang, Zheng Zhang, Jiajun Zhang. (2023)  
**BiPFT: Binary Pre-trained Foundation Transformer with Low-rank Estimation of Binarization Residual Polynomials**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GLUE, NLU, Transformer  
[Paper Link](http://arxiv.org/abs/2312.08937v1)  

---


**ABSTRACT**  
Pretrained foundation models offer substantial benefits for a wide range of downstream tasks, which can be one of the most potential techniques to access artificial general intelligence. However, scaling up foundation transformers for maximal task-agnostic knowledge has brought about computational challenges, especially on resource-limited devices such as mobiles. This work proposes the first Binary Pretrained Foundation Transformer (BiPFT) for natural language understanding (NLU) tasks, which remarkably saves 56 times operations and 28 times memory. In contrast to previous task-specific binary transformers, BiPFT exhibits a substantial enhancement in the learning capabilities of binary neural networks (BNNs), promoting BNNs into the era of pre-training. Benefiting from extensive pretraining data, we further propose a data-driven binarization method. Specifically, we first analyze the binarization error in self-attention operations and derive the polynomials of binarization error. To simulate full-precision self-attention, we define binarization error as binarization residual polynomials, and then introduce low-rank estimators to model these polynomials. Extensive experiments validate the effectiveness of BiPFTs, surpassing task-specific baseline by 15.4% average performance on the GLUE benchmark. BiPFT also demonstrates improved robustness to hyperparameter changes, improved optimization efficiency, and reduced reliance on downstream distillation, which consequently generalize on various NLU tasks and simplify the downstream pipeline of BNNs. Our code and pretrained models are publicly available at https://github.com/Xingrun-Xing/BiPFT.

{{</citation>}}


### (19/169) Global Rewards in Multi-Agent Deep Reinforcement Learning for Autonomous Mobility on Demand Systems (Heiko Hoppe et al., 2023)

{{<citation>}}

Heiko Hoppe, Tobias Enders, Quentin Cappart, Maximilian Schiffer. (2023)  
**Global Rewards in Multi-Agent Deep Reinforcement Learning for Autonomous Mobility on Demand Systems**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-MA, cs-SY, cs.LG, eess-SY  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.08884v1)  

---


**ABSTRACT**  
We study vehicle dispatching in autonomous mobility on demand (AMoD) systems, where a central operator assigns vehicles to customer requests or rejects these with the aim of maximizing its total profit. Recent approaches use multi-agent deep reinforcement learning (MADRL) to realize scalable yet performant algorithms, but train agents based on local rewards, which distorts the reward signal with respect to the system-wide profit, leading to lower performance. We therefore propose a novel global-rewards-based MADRL algorithm for vehicle dispatching in AMoD systems, which resolves so far existing goal conflicts between the trained agents and the operator by assigning rewards to agents leveraging a counterfactual baseline. Our algorithm shows statistically significant improvements across various settings on real-world data compared to state-of-the-art MADRL algorithms with local rewards. We further provide a structural analysis which shows that the utilization of global rewards can improve implicit vehicle balancing and demand forecasting abilities. Our code is available at https://github.com/tumBAIS/GR-MADRL-AMoD.

{{</citation>}}


### (20/169) TiMix: Text-aware Image Mixing for Effective Vision-Language Pre-training (Chaoya Jiang et al., 2023)

{{<citation>}}

Chaoya Jiang, Wei ye, Haiyang Xu, Qinghao Ye, Ming Yan, Ji Zhang, Shikun Zhang. (2023)  
**TiMix: Text-aware Image Mixing for Effective Vision-Language Pre-training**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-CV, cs-LG, cs.LG  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2312.08846v1)  

---


**ABSTRACT**  
Self-supervised Multi-modal Contrastive Learning (SMCL) remarkably advances modern Vision-Language Pre-training (VLP) models by aligning visual and linguistic modalities. Due to noises in web-harvested text-image pairs, however, scaling up training data volume in SMCL presents considerable obstacles in terms of computational cost and data inefficiency. To improve data efficiency in VLP, we propose Text-aware Image Mixing (TiMix), which integrates mix-based data augmentation techniques into SMCL, yielding significant performance improvements without significantly increasing computational overhead. We provide a theoretical analysis of TiMixfrom a mutual information (MI) perspective, showing that mixed data samples for cross-modal contrastive learning implicitly serve as a regularizer for the contrastive loss. The experimental results demonstrate that TiMix exhibits a comparable performance on downstream tasks, even with a reduced amount of training data and shorter training time, when benchmarked against existing methods. This work empirically and theoretically demonstrates the potential of data mixing for data-efficient and computationally viable VLP, benefiting broader VLP model adoption in practical scenarios.

{{</citation>}}


### (21/169) A Cyber-Physical Architecture for Microgrids based on Deep learning and LORA Technology (Mojtaba Mohammadi et al., 2023)

{{<citation>}}

Mojtaba Mohammadi, Abdollah KavousiFard, Mortza Dabbaghjamanesh, Mostafa Shaaban, Hatem. H. Zeineldin, Ehab Fahmy El-Saadany. (2023)  
**A Cyber-Physical Architecture for Microgrids based on Deep learning and LORA Technology**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2312.08818v2)  

---


**ABSTRACT**  
This paper proposes a cyber-physical architecture for the secured social operation of isolated hybrid microgrids (HMGs). On the physical side of the proposed architecture, an optimal scheduling scheme considering various renewable energy sources (RESs) and fossil fuel-based distributed generation units (DGs) is proposed. Regarding the cyber layer of MGs, a wireless architecture based on low range wide area (LORA) technology is introduced for advanced metering infrastructure (AMI) in smart electricity grids. In the proposed architecture, the LORA data frame is described in detail and designed for the application of smart meters considering DGs and ac-dc converters. Additionally, since the cyber layer of smart grids is highly vulnerable to cyber-attacks, t1his paper proposes a deep-learning-based cyber-attack detection model (CADM) based on bidirectional long short-term memory (BLSTM) and sequential hypothesis testing (SHT) to detect false data injection attacks (FDIA) on the smart meters within AMI. The performance of the proposed energy management architecture is evaluated using the IEEE 33-bus test system. In order to investigate the effect of FDIA on the isolated HMGs and highlight the interactions between the cyber layer and physical layer, an FDIA is launched against the test system. The results showed that a successful attack can highly damage the system and cause widespread load shedding. Also, the performance of the proposed CADM is examined using a real-world dataset. Results prove the effectiveness of the proposed CADM in detecting the attacks using only two samples.

{{</citation>}}


### (22/169) Deep Learning-Based Cyber-Attack Detection Model for Smart Grids (Mojtaba Mohammadi et al., 2023)

{{<citation>}}

Mojtaba Mohammadi, Arshia Aflaki, Abdollah Kavousifard, Mohsen Gitizadeh. (2023)  
**Deep Learning-Based Cyber-Attack Detection Model for Smart Grids**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2312.08810v1)  

---


**ABSTRACT**  
In this paper, a novel artificial intelligence-based cyber-attack detection model for smart grids is developed to stop data integrity cyber-attacks (DIAs) on the received load data by supervisory control and data acquisition (SCADA). In the proposed model, first the load data is forecasted using a regression model and after processing stage, the processed data is clustered using the unsupervised learning method. In this work, in order to achieve the best performance, three load forecasting methods (i.e. extra tree regression (ETR), long short-term memory (LSTM) and bidirectional long short-term memory (BiLSTM)) are utilized as regression models and their performance is compared. For clustering and outlying detection, the covariance elliptic envelope (EE) is employed as an unsupervised learning method. To examine the proposed model, the hourly load data of the power company of the city of Johor in Malaysia is employed and Two common DIAs, which are DIAs targeting economic loss and DIAs targeting blackouts, are used to evaluate the accuracy of detection methods in several scenarios. The simulation results show that the proposed EE-BiLSTM method can perform more robust and accurate compared to the other two methods.

{{</citation>}}


### (23/169) Random resistive memory-based deep extreme point learning machine for unified visual processing (Shaocong Wang et al., 2023)

{{<citation>}}

Shaocong Wang, Yizhao Gao, Yi Li, Woyu Zhang, Yifei Yu, Bo Wang, Ning Lin, Hegan Chen, Yue Zhang, Yang Jiang, Dingchen Wang, Jia Chen, Peng Dai, Hao Jiang, Peng Lin, Xumeng Zhang, Xiaojuan Qi, Xiaoxin Xu, Hayden So, Zhongrui Wang, Dashan Shang, Qi Liu, Kwang-Ting Cheng, Ming Liu. (2023)  
**Random resistive memory-based deep extreme point learning machine for unified visual processing**  

---
Primary Category: cs.LG  
Categories: cs-AR, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.09262v1)  

---


**ABSTRACT**  
Visual sensors, including 3D LiDAR, neuromorphic DVS sensors, and conventional frame cameras, are increasingly integrated into edge-side intelligent machines. Realizing intensive multi-sensory data analysis directly on edge intelligent machines is crucial for numerous emerging edge applications, such as augmented and virtual reality and unmanned aerial vehicles, which necessitates unified data representation, unprecedented hardware energy efficiency and rapid model training. However, multi-sensory data are intrinsically heterogeneous, causing significant complexity in the system development for edge-side intelligent machines. In addition, the performance of conventional digital hardware is limited by the physically separated processing and memory units, known as the von Neumann bottleneck, and the physical limit of transistor scaling, which contributes to the slowdown of Moore's law. These limitations are further intensified by the tedious training of models with ever-increasing sizes. We propose a novel hardware-software co-design, random resistive memory-based deep extreme point learning machine (DEPLM), that offers efficient unified point set analysis. We show the system's versatility across various data modalities and two different learning tasks. Compared to a conventional digital hardware-based system, our co-design system achieves huge energy efficiency improvements and training cost reduction when compared to conventional systems. Our random resistive memory-based deep extreme point learning machine may pave the way for energy-efficient and training-friendly edge AI across various data modalities and tasks.

{{</citation>}}


### (24/169) Learning from Polar Representation: An Extreme-Adaptive Model for Long-Term Time Series Forecasting (Yanhong Li et al., 2023)

{{<citation>}}

Yanhong Li, Jack Xu, David C. Anastasiu. (2023)  
**Learning from Polar Representation: An Extreme-Adaptive Model for Long-Term Time Series Forecasting**  

---
Primary Category: cs.LG  
Categories: Machine Learning (cs-LG), Artificial Intelligence (cs-AI), cs-AI, cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2312.08763v2)  

---


**ABSTRACT**  
In the hydrology field, time series forecasting is crucial for efficient water resource management, improving flood and drought control and increasing the safety and quality of life for the general population. However, predicting long-term streamflow is a complex task due to the presence of extreme events. It requires the capture of long-range dependencies and the modeling of rare but important extreme values. Existing approaches often struggle to tackle these dual challenges simultaneously. In this paper, we specifically delve into these issues and propose Distance-weighted Auto-regularized Neural network (DAN), a novel extreme-adaptive model for long-range forecasting of stremflow enhanced by polar representation learning. DAN utilizes a distance-weighted multi-loss mechanism and stackable blocks to dynamically refine indicator sequences from exogenous data, while also being able to handle uni-variate time-series by employing Gaussian Mixture probability modeling to improve robustness to severe events. We also introduce Kruskal-Wallis sampling and gate control vectors to handle imbalanced extreme data. On four real-life hydrologic streamflow datasets, we demonstrate that DAN significantly outperforms both state-of-the-art hydrologic time series prediction methods and general methods designed for long-term time series prediction.

{{</citation>}}


### (25/169) Improve Robustness of Reinforcement Learning against Observation Perturbations via $l_\infty$ Lipschitz Policy Networks (Buqing Nie et al., 2023)

{{<citation>}}

Buqing Nie, Jingtian Ji, Yangqing Fu, Yue Gao. (2023)  
**Improve Robustness of Reinforcement Learning against Observation Perturbations via $l_\infty$ Lipschitz Policy Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.08751v1)  

---


**ABSTRACT**  
Deep Reinforcement Learning (DRL) has achieved remarkable advances in sequential decision tasks. However, recent works have revealed that DRL agents are susceptible to slight perturbations in observations. This vulnerability raises concerns regarding the effectiveness and robustness of deploying such agents in real-world applications. In this work, we propose a novel robust reinforcement learning method called SortRL, which improves the robustness of DRL policies against observation perturbations from the perspective of the network architecture. We employ a novel architecture for the policy network that incorporates global $l_\infty$ Lipschitz continuity and provide a convenient method to enhance policy robustness based on the output margin. Besides, a training framework is designed for SortRL, which solves given tasks while maintaining robustness against $l_\infty$ bounded perturbations on the observations. Several experiments are conducted to evaluate the effectiveness of our method, including classic control tasks and video games. The results demonstrate that SortRL achieves state-of-the-art robustness performance against different perturbation strength.

{{</citation>}}


### (26/169) Mitigating Label Bias in Machine Learning: Fairness through Confident Learning (Yixuan Zhang et al., 2023)

{{<citation>}}

Yixuan Zhang, Boyu Li, Zenan Ling, Feng Zhou. (2023)  
**Mitigating Label Bias in Machine Learning: Fairness through Confident Learning**  

---
Primary Category: cs.LG  
Categories: cs-CY, cs-LG, cs.LG  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2312.08749v1)  

---


**ABSTRACT**  
Discrimination can occur when the underlying unbiased labels are overwritten by an agent with potential bias, resulting in biased datasets that unfairly harm specific groups and cause classifiers to inherit these biases. In this paper, we demonstrate that despite only having access to the biased labels, it is possible to eliminate bias by filtering the fairest instances within the framework of confident learning. In the context of confident learning, low self-confidence usually indicates potential label errors; however, this is not always the case. Instances, particularly those from underrepresented groups, might exhibit low confidence scores for reasons other than labeling errors. To address this limitation, our approach employs truncation of the confidence score and extends the confidence interval of the probabilistic threshold. Additionally, we incorporate with co-teaching paradigm for providing a more robust and reliable selection of fair instances and effectively mitigating the adverse effects of biased labels. Through extensive experimentation and evaluation of various datasets, we demonstrate the efficacy of our approach in promoting fairness and reducing the impact of label bias in machine learning models.

{{</citation>}}


### (27/169) Learning a Low-Rank Feature Representation: Achieving Better Trade-Off between Stability and Plasticity in Continual Learning (Zhenrong Liu et al., 2023)

{{<citation>}}

Zhenrong Liu, Yang Li, Yi Gong, Yik-Chung Wu. (2023)  
**Learning a Low-Rank Feature Representation: Achieving Better Trade-Off between Stability and Plasticity in Continual Learning**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2312.08740v1)  

---


**ABSTRACT**  
In continual learning, networks confront a trade-off between stability and plasticity when trained on a sequence of tasks. To bolster plasticity without sacrificing stability, we propose a novel training algorithm called LRFR. This approach optimizes network parameters in the null space of the past tasks' feature representation matrix to guarantee the stability. Concurrently, we judiciously select only a subset of neurons in each layer of the network while training individual tasks to learn the past tasks' feature representation matrix in low-rank. This increases the null space dimension when designing network parameters for subsequent tasks, thereby enhancing the plasticity. Using CIFAR-100 and TinyImageNet as benchmark datasets for continual learning, the proposed approach consistently outperforms state-of-the-art methods.

{{</citation>}}


### (28/169) A Comparative Analysis of Fine-Tuned LLMs and Few-Shot Learning of LLMs for Financial Sentiment Analysis (Sorouralsadat Fatemi et al., 2023)

{{<citation>}}

Sorouralsadat Fatemi, Yuheng Hu. (2023)  
**A Comparative Analysis of Fine-Tuned LLMs and Few-Shot Learning of LLMs for Financial Sentiment Analysis**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Few-Shot, Financial, Language Model, NLP, Natural Language Processing, Sentiment Analysis  
[Paper Link](http://arxiv.org/abs/2312.08725v1)  

---


**ABSTRACT**  
Financial sentiment analysis plays a crucial role in uncovering latent patterns and detecting emerging trends, enabling individuals to make well-informed decisions that may yield substantial advantages within the constantly changing realm of finance. Recently, Large Language Models (LLMs) have demonstrated their effectiveness in diverse domains, showcasing remarkable capabilities even in zero-shot and few-shot in-context learning for various Natural Language Processing (NLP) tasks. Nevertheless, their potential and applicability in the context of financial sentiment analysis have not been thoroughly explored yet. To bridge this gap, we employ two approaches: in-context learning (with a focus on gpt-3.5-turbo model) and fine-tuning LLMs on a finance-domain dataset. Given the computational costs associated with fine-tuning LLMs with large parameter sizes, our focus lies on smaller LLMs, spanning from 250M to 3B parameters for fine-tuning. We then compare the performances with state-of-the-art results to evaluate their effectiveness in the finance-domain. Our results demonstrate that fine-tuned smaller LLMs can achieve comparable performance to state-of-the-art fine-tuned LLMs, even with models having fewer parameters and a smaller training dataset. Additionally, the zero-shot and one-shot performance of LLMs produces comparable results with fine-tuned smaller LLMs and state-of-the-art outcomes. Furthermore, our analysis demonstrates that there is no observed enhancement in performance for finance-domain sentiment analysis when the number of shots for in-context learning is increased.

{{</citation>}}


### (29/169) RdimKD: Generic Distillation Paradigm by Dimensionality Reduction (Yi Guo et al., 2023)

{{<citation>}}

Yi Guo, Yiqian He, Xiaoyang Li, Haotong Qin, Van Tung Pham, Yang Zhang, Shouda Liu. (2023)  
**RdimKD: Generic Distillation Paradigm by Dimensionality Reduction**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2312.08700v1)  

---


**ABSTRACT**  
Knowledge Distillation (KD) emerges as one of the most promising compression technologies to run advanced deep neural networks on resource-limited devices. In order to train a small network (student) under the guidance of a large network (teacher), the intuitive method is regularizing the feature maps or logits of the student using the teacher's information. However, existing methods either over-restrict the student to learn all information from the teacher, which lead to some bad local minimum, or use various fancy and elaborate modules to process and align features, which are complex and lack generality. In this work, we proposed an abstract and general paradigm for the KD task, referred to as DIMensionality Reduction KD (RdimKD), which solely relies on dimensionality reduction, with a very minor modification to naive L2 loss. RdimKD straightforwardly utilizes a projection matrix to project both the teacher's and student's feature maps onto a low-dimensional subspace, which are then optimized during training. RdimKD achieves the goal in the simplest way that not only does the student get valuable information from the teacher, but it also ensures sufficient flexibility to adapt to the student's low-capacity reality. Our extensive empirical findings indicate the effectiveness of RdimKD across various learning tasks and diverse network architectures.

{{</citation>}}


### (30/169) CAT: A Causally Graph Attention Network for Trimming Heterophilic Graph (Silu He et al., 2023)

{{<citation>}}

Silu He, Qinyao Luo, Xinsha Fu, Ling Zhao, Ronghua Du, Haifeng Li. (2023)  
**CAT: A Causally Graph Attention Network for Trimming Heterophilic Graph**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-SI, cs.LG  
Keywords: Attention, Graph Attention Network  
[Paper Link](http://arxiv.org/abs/2312.08672v2)  

---


**ABSTRACT**  
Local Attention-guided Message Passing Mechanism (LAMP) adopted in Graph Attention Networks (GATs) is designed to adaptively learn the importance of neighboring nodes for better local aggregation on the graph, which can bring the representations of similar neighbors closer effectively, thus showing stronger discrimination ability. However, existing GATs suffer from a significant discrimination ability decline in heterophilic graphs because the high proportion of dissimilar neighbors can weaken the self-attention of the central node, jointly resulting in the deviation of the central node from similar nodes in the representation space. This kind of effect generated by neighboring nodes is called the Distraction Effect (DE) in this paper. To estimate and weaken the DE of neighboring nodes, we propose a Causally graph Attention network for Trimming heterophilic graph (CAT). To estimate the DE, since the DE are generated through two paths (grab the attention assigned to neighbors and reduce the self-attention of the central node), we use Total Effect to model DE, which is a kind of causal estimand and can be estimated from intervened data; To weaken the DE, we identify the neighbors with the highest DE (we call them Distraction Neighbors) and remove them. We adopt three representative GATs as the base model within the proposed CAT framework and conduct experiments on seven heterophilic datasets in three different sizes. Comparative experiments show that CAT can improve the node classification accuracy of all base GAT models. Ablation experiments and visualization further validate the enhancement of discrimination ability brought by CAT. The source code is available at https://github.com/GeoX-Lab/CAT.

{{</citation>}}


### (31/169) Uplifting the Expressive Power of Graph Neural Networks through Graph Partitioning (Asela Hevapathige et al., 2023)

{{<citation>}}

Asela Hevapathige, Qing Wang. (2023)  
**Uplifting the Expressive Power of Graph Neural Networks through Graph Partitioning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2312.08671v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) have paved its way for being a cornerstone in graph related learning tasks. From a theoretical perspective, the expressive power of GNNs is primarily characterised according to their ability to distinguish non-isomorphic graphs. It is a well-known fact that most of the conventional GNNs are upper-bounded by Weisfeiler-Lehman graph isomorphism test (1-WL). In this work, we study the expressive power of graph neural networks through the lens of graph partitioning. This follows from our observation that permutation invariant graph partitioning enables a powerful way of exploring structural interactions among vertex sets and subgraphs, and can help uplifting the expressive power of GNNs efficiently. Based on this, we first establish a theoretical connection between graph partitioning and graph isomorphism. Then we introduce a novel GNN architecture, namely Graph Partitioning Neural Networks (GPNNs). We theoretically analyse how a graph partitioning scheme and different kinds of structural interactions relate to the k-WL hierarchy. Empirically, we demonstrate its superior performance over existing GNN models in a variety of graph benchmark tasks.

{{</citation>}}


### (32/169) MaxK-GNN: Towards Theoretical Speed Limits for Accelerating Graph Neural Networks Training (Hongwu Peng et al., 2023)

{{<citation>}}

Hongwu Peng, Xi Xie, Kaustubh Shivdikar, MD Amit Hasan, Jiahui Zhao, Shaoyi Huang, Omer Khan, David Kaeli, Caiwen Ding. (2023)  
**MaxK-GNN: Towards Theoretical Speed Limits for Accelerating Graph Neural Networks Training**  

---
Primary Category: cs.LG  
Categories: I-2; C-5, cs-AI, cs-DC, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2312.08656v2)  

---


**ABSTRACT**  
In the acceleration of deep neural network training, the GPU has become the mainstream platform. GPUs face substantial challenges on GNNs, such as workload imbalance and memory access irregularities, leading to underutilized hardware. Existing solutions such as PyG, DGL with cuSPARSE, and GNNAdvisor frameworks partially address these challenges but memory traffic is still significant.   We argue that drastic performance improvements can only be achieved by the vertical optimization of algorithm and system innovations, rather than treating the speedup optimization as an "after-thought" (i.e., (i) given a GNN algorithm, designing an accelerator, or (ii) given hardware, mainly optimizing the GNN algorithm). In this paper, we present MaxK-GNN, an advanced high-performance GPU training system integrating algorithm and system innovation. (i) We introduce the MaxK nonlinearity and provide a theoretical analysis of MaxK nonlinearity as a universal approximator, and present the Compressed Balanced Sparse Row (CBSR) format, designed to store the data and index of the feature matrix after nonlinearity; (ii) We design a coalescing enhanced forward computation with row-wise product-based SpGEMM Kernel using CBSR for input feature matrix fetching and strategic placement of a sparse output accumulation buffer in shared memory; (iii) We develop an optimized backward computation with outer product-based and SSpMM Kernel.   We conduct extensive evaluations of MaxK-GNN and report the end-to-end system run-time. Experiments show that MaxK-GNN system could approach the theoretical speedup limit according to Amdahl's law. We achieve comparable accuracy to SOTA GNNs, but at a significantly increased speed: 3.22/4.24 times speedup (vs. theoretical limits, 5.52/7.27 times) on Reddit compared to DGL and GNNAdvisor implementations.

{{</citation>}}


### (33/169) Towards Inductive Robustness: Distilling and Fostering Wave-induced Resonance in Transductive GCNs Against Graph Adversarial Attacks (Ao Liu et al., 2023)

{{<citation>}}

Ao Liu, Wenshan Li, Tao Li, Beibei Li, Hanyuan Huang, Pan Zhou. (2023)  
**Towards Inductive Robustness: Distilling and Fostering Wave-induced Resonance in Transductive GCNs Against Graph Adversarial Attacks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Adversarial Attack, GNN  
[Paper Link](http://arxiv.org/abs/2312.08651v1)  

---


**ABSTRACT**  
Graph neural networks (GNNs) have recently been shown to be vulnerable to adversarial attacks, where slight perturbations in the graph structure can lead to erroneous predictions. However, current robust models for defending against such attacks inherit the transductive limitations of graph convolutional networks (GCNs). As a result, they are constrained by fixed structures and do not naturally generalize to unseen nodes. Here, we discover that transductive GCNs inherently possess a distillable robustness, achieved through a wave-induced resonance process. Based on this, we foster this resonance to facilitate inductive and robust learning. Specifically, we first prove that the signal formed by GCN-driven message passing (MP) is equivalent to the edge-based Laplacian wave, where, within a wave system, resonance can naturally emerge between the signal and its transmitting medium. This resonance provides inherent resistance to malicious perturbations inflicted on the signal system. We then prove that merely three MP iterations within GCNs can induce signal resonance between nodes and edges, manifesting as a coupling between nodes and their distillable surrounding local subgraph. Consequently, we present Graph Resonance-fostering Network (GRN) to foster this resonance via learning node representations from their distilled resonating subgraphs. By capturing the edge-transmitted signals within this subgraph and integrating them with the node signal, GRN embeds these combined signals into the central node's representation. This node-wise embedding approach allows for generalization to unseen nodes. We validate our theoretical findings with experiments, and demonstrate that GRN generalizes robustness to unseen nodes, whilst maintaining state-of-the-art classification accuracy on perturbed graphs.

{{</citation>}}


## cs.CY (4)



### (34/169) CERN for AGI: A Theoretical Framework for Autonomous Simulation-Based Artificial Intelligence Testing and Alignment (Ljubisa Bojic et al., 2023)

{{<citation>}}

Ljubisa Bojic, Matteo Cinelli, Dubravko Culibrk, Boris Delibasic. (2023)  
**CERN for AGI: A Theoretical Framework for Autonomous Simulation-Based Artificial Intelligence Testing and Alignment**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CY, cs-GT, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.09402v1)  

---


**ABSTRACT**  
This paper explores the potential of a multidisciplinary approach to testing and aligning artificial general intelligence (AGI) and LLMs. Due to the rapid development and wide application of LLMs, challenges such as ethical alignment, controllability, and predictability of these models have become important research topics. This study investigates an innovative simulation-based multi-agent system within a virtual reality framework that replicates the real-world environment. The framework is populated by automated 'digital citizens,' simulating complex social structures and interactions to examine and optimize AGI. Application of various theories from the fields of sociology, social psychology, computer science, physics, biology, and economics demonstrates the possibility of a more human-aligned and socially responsible AGI. The purpose of such a digital environment is to provide a dynamic platform where advanced AI agents can interact and make independent decisions, thereby mimicking realistic scenarios. The actors in this digital city, operated by the LLMs, serve as the primary agents, exhibiting high degrees of autonomy. While this approach shows immense potential, there are notable challenges and limitations, most significantly the unpredictable nature of real-world social dynamics. This research endeavors to contribute to the development and refinement of AGI, emphasizing the integration of social, ethical, and theoretical dimensions for future research.

{{</citation>}}


### (35/169) Children, Parents, and Misinformation on Social Media (Filipo Sharevski et al., 2023)

{{<citation>}}

Filipo Sharevski, Jennifer Vander Loop. (2023)  
**Children, Parents, and Misinformation on Social Media**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: Google, Social Media  
[Paper Link](http://arxiv.org/abs/2312.09359v1)  

---


**ABSTRACT**  
Children encounter misinformation on social media in a similar capacity as their parents. Unlike their parents, children are an exceptionally vulnerable population because their cognitive abilities and emotional regulation are still maturing, rendering them more susceptible to misinformation and falsehoods online. Yet, little is known about children's experience with misinformation as well as what their parents think of the misinformation's effect on child development. To answer these questions, we combined a qualitative survey of parents (n=87) with semi-structured interviews of both parents and children (n=12). We found that children usually encounter deep fakes, memes with political context, or celebrity/influencer rumors on social media. Children revealed they "ask Siri" whether a social media video or post is true or not before they search on Google or ask their parents about it. Parents expressed discontent that their children are impressionable to misinformation, stating that the burden falls on them to help their children develop critical thinking skills for navigating falsehoods on social media. Here, the majority of parents felt that schools should also teach these skills as well as media literacy to their children. Misinformation, according to both parents and children affects the family relationships especially with grandparents with different political views than theirs.

{{</citation>}}


### (36/169) Older Adults' Experiences with Misinformation on Social Media (Filipo Sharevski et al., 2023)

{{<citation>}}

Filipo Sharevski, Jennifer Vander Loop. (2023)  
**Older Adults' Experiences with Misinformation on Social Media**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: Social Media  
[Paper Link](http://arxiv.org/abs/2312.09354v1)  

---


**ABSTRACT**  
Older adults habitually encounter misinformation on social media, but there is little knowledge about their experiences with it. In this study, we combined a qualitative survey (n=119) with in-depth interviews (n=21) to investigate how older adults in America conceptualize, discern, and contextualize social media misinformation. As misinformation on social media in the past was driven towards influencing voting outcomes, we were particularly interested to approach our study from a voting intention perspective. We found that 62% of the participants intending to vote Democrat saw a manipulative political purpose behind the spread of misinformation while only 5% of those intending to vote Republican believed misinformation has a political dissent purpose. Regardless of the voting intentions, most participants relied on source heuristics combined with fact-checking to discern truth from misinformation on social media. The biggest concern about the misinformation, among all the participants, was that it increasingly leads to biased reasoning influenced by personal values and feelings instead of reasoning based on objective evidence. The participants intending to vote Democrat were in 74% of the cases concerned that misinformation will cause escalation of extremism in the future, while those intending to vote Republican, were undecided, or planned to abstain were concerned that misinformation will further erode the trust in democratic institutions, specifically in the context of public health and free and fair elections. During our interviews, we found that 63% of the participants who intended to vote Republican, were fully aware and acknowledged that Republican or conservative voices often time speak misinformation, even though they are closely aligned to their political ideology.

{{</citation>}}


### (37/169) Casual Social Media Use among the Youth: Effects on Online and Offline Political Participation (Mehdi Barati, 2023)

{{<citation>}}

Mehdi Barati. (2023)  
**Casual Social Media Use among the Youth: Effects on Online and Offline Political Participation**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CY, cs-SI, cs.CY  
Keywords: Social Media  
[Paper Link](http://arxiv.org/abs/2312.10095v1)  

---


**ABSTRACT**  
Background: Previous studies suggest that social media use among the youth is correlated with online and offline political participation. There is also a mixed and inconclusive debate on whether more online political participation in the youth increases their offline political participation. Methods: This study uses three models of OLS, two-way fixed effects, and an instrumental variable approach to make causal inferences about social media use, online, and offline political participation of the youth. Findings: The analyses provide evidence of a large effect of casual social media use on online political participation, and no effect or negligible effect on offline political participation and voting behavior. The results from fixed effects and instrumental variable models provide strong evidence of elasticity between online and offline political participation in young individuals. On average, a one percent increase in online political participation increases the offline political activity index by 0.12 percent.

{{</citation>}}


## cs.AR (1)



### (38/169) Inter-Layer Scheduling Space Exploration for Multi-model Inference on Heterogeneous Chiplets (Mohanad Odema et al., 2023)

{{<citation>}}

Mohanad Odema, Hyoukjun Kwon, Mohammad Abdullah Al Faruque. (2023)  
**Inter-Layer Scheduling Space Exploration for Multi-model Inference on Heterogeneous Chiplets**  

---
Primary Category: cs.AR  
Categories: cs-AI, cs-AR, cs-DC, cs.AR  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2312.09401v1)  

---


**ABSTRACT**  
To address increasing compute demand from recent multi-model workloads with heavy models like large language models, we propose to deploy heterogeneous chiplet-based multi-chip module (MCM)-based accelerators. We develop an advanced scheduling framework for heterogeneous MCM accelerators that comprehensively consider complex heterogeneity and inter-chiplet pipelining. Our experiments using our framework on GPT-2 and ResNet-50 models on a 4-chiplet system have shown upto 2.2x and 1.9x increase in throughput and energy efficiency, compared to a monolithic accelerator with an optimized output-stationary dataflow.

{{</citation>}}


## cs.AI (17)



### (39/169) Large Language Models for Autonomous Driving: Real-World Experiments (Can Cui et al., 2023)

{{<citation>}}

Can Cui, Zichong Yang, Yupeng Zhou, Yunsheng Ma, Juanwu Lu, Ziran Wang. (2023)  
**Large Language Models for Autonomous Driving: Real-World Experiments**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.09397v1)  

---


**ABSTRACT**  
Autonomous driving systems are increasingly popular in today's technological landscape, where vehicles with partial automation have already been widely available on the market, and the full automation era with ``driverless'' capabilities is near the horizon. However, accurately understanding humans' commands, particularly for autonomous vehicles that have only passengers instead of drivers, and achieving a high level of personalization remain challenging tasks in the development of autonomous driving systems. In this paper, we introduce a Large Language Model (LLM)-based framework Talk-to-Drive (Talk2Drive) to process verbal commands from humans and make autonomous driving decisions with contextual information, satisfying their personalized preferences for safety, efficiency, and comfort. First, a speech recognition module is developed for Talk2Drive to interpret verbal inputs from humans to textual instructions, which are then sent to LLMs for reasoning. Then, appropriate commands for the Electrical Control Unit (ECU) are generated, achieving a 100\% success rate in executing codes. Real-world experiments show that our framework can substantially reduce the takeover rate for a diverse range of drivers by up to 90.1\%. To the best of our knowledge, Talk2Drive marks the first instance of employing an LLM-based system in a real-world autonomous driving environment.

{{</citation>}}


### (40/169) ArchiGuesser -- AI Art Architecture Educational Game (Joern Ploennigs et al., 2023)

{{<citation>}}

Joern Ploennigs, Markus Berger, Eva Carnein. (2023)  
**ArchiGuesser -- AI Art Architecture Educational Game**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-MM, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.09334v1)  

---


**ABSTRACT**  
The use of generative AI in education is a controversial topic. Current technology offers the potential to create educational content from text, speech, to images based on simple input prompts. This can enhance productivity by summarizing knowledge and improving communication, quickly adjusting to different types of learners. Moreover, generative AI holds the promise of making the learning itself more fun, by responding to user inputs and dynamically generating high-quality creative material. In this paper we present the multisensory educational game ArchiGuesser that combines various AI technologies from large language models, image generation, to computer vision to serve a single purpose: Teaching students in a playful way the diversity of our architectural history and how generative AI works.

{{</citation>}}


### (41/169) Auto MC-Reward: Automated Dense Reward Design with Large Language Models for Minecraft (Hao Li et al., 2023)

{{<citation>}}

Hao Li, Xue Yang, Zhaokai Wang, Xizhou Zhu, Jie Zhou, Yu Qiao, Xiaogang Wang, Hongsheng Li, Lewei Lu, Jifeng Dai. (2023)  
**Auto MC-Reward: Automated Dense Reward Design with Large Language Models for Minecraft**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.09238v1)  

---


**ABSTRACT**  
Traditional reinforcement-learning-based agents rely on sparse rewards that often only use binary values to indicate task completion or failure. The challenge in exploration efficiency makes it difficult to effectively learn complex tasks in Minecraft. To address this, this paper introduces an advanced learning system, named Auto MC-Reward, that leverages Large Language Models (LLMs) to automatically design dense reward functions, thereby enhancing the learning efficiency. Auto MC-Reward consists of three important components: Reward Designer, Reward Critic, and Trajectory Analyzer. Given the environment information and task descriptions, the Reward Designer first design the reward function by coding an executable Python function with predefined observation inputs. Then, our Reward Critic will be responsible for verifying the code, checking whether the code is self-consistent and free of syntax and semantic errors. Further, the Trajectory Analyzer summarizes possible failure causes and provides refinement suggestions according to collected trajectories. In the next round, Reward Designer will take further refine and iterate the dense reward function based on feedback. Experiments demonstrate a significant improvement in the success rate and learning efficiency of our agents in complex tasks in Minecraft, such as obtaining diamond with the efficient ability to avoid lava, and efficiently explore trees and animals that are sparse on the plains biome.

{{</citation>}}


### (42/169) NestE: Modeling Nested Relational Structures for Knowledge Graph Reasoning (Bo Xiong et al., 2023)

{{<citation>}}

Bo Xiong, Mojtaba Nayyeri, Linhao Luo, Zihao Wang, Shirui Pan, Steffen Staab. (2023)  
**NestE: Modeling Nested Relational Structures for Knowledge Graph Reasoning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Knowledge Graph, Reasoning  
[Paper Link](http://arxiv.org/abs/2312.09219v1)  

---


**ABSTRACT**  
Reasoning with knowledge graphs (KGs) has primarily focused on triple-shaped facts. Recent advancements have been explored to enhance the semantics of these facts by incorporating more potent representations, such as hyper-relational facts. However, these approaches are limited to \emph{atomic facts}, which describe a single piece of information. This paper extends beyond \emph{atomic facts} and delves into \emph{nested facts}, represented by quoted triples where subjects and objects are triples themselves (e.g., ((\emph{BarackObama}, \emph{holds\_position}, \emph{President}), \emph{succeed\_by}, (\emph{DonaldTrump}, \emph{holds\_position}, \emph{President}))). These nested facts enable the expression of complex semantics like \emph{situations} over time and \emph{logical patterns} over entities and relations. In response, we introduce NestE, a novel KG embedding approach that captures the semantics of both atomic and nested factual knowledge. NestE represents each atomic fact as a $1\times3$ matrix, and each nested relation is modeled as a $3\times3$ matrix that rotates the $1\times3$ atomic fact matrix through matrix multiplication. Each element of the matrix is represented as a complex number in the generalized 4D hypercomplex space, including (spherical) quaternions, hyperbolic quaternions, and split-quaternions. Through thorough analysis, we demonstrate the embedding's efficacy in capturing diverse logical patterns over nested facts, surpassing the confines of first-order logic-like expressions. Our experimental results showcase NestE's significant performance gains over current baselines in triple prediction and conditional link prediction. The code and pre-trained models are open available at https://github.com/xiongbo010/NestE.

{{</citation>}}


### (43/169) Weaving Pathways for Justice with GPT: LLM-driven automated drafting of interactive legal applications (Quinten Steenhuis et al., 2023)

{{<citation>}}

Quinten Steenhuis, David Colarusso, Bryce Willey. (2023)  
**Weaving Pathways for Justice with GPT: LLM-driven automated drafting of interactive legal applications**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CV, cs-CY, cs-HC, cs-SI, cs.AI  
Keywords: AI, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2312.09198v1)  

---


**ABSTRACT**  
Can generative AI help us speed up the authoring of tools to help self-represented litigants?   In this paper, we describe 3 approaches to automating the completion of court forms: a generative AI approach that uses GPT-3 to iteratively prompt the user to answer questions, a constrained template-driven approach that uses GPT-4-turbo to generate a draft of questions that are subject to human review, and a hybrid method. We use the open source Docassemble platform in all 3 experiments, together with a tool created at Suffolk University Law School called the Assembly Line Weaver. We conclude that the hybrid model of constrained automated drafting with human review is best suited to the task of authoring guided interviews.

{{</citation>}}


### (44/169) A Sparse Cross Attention-based Graph Convolution Network with Auxiliary Information Awareness for Traffic Flow Prediction (Lingqiang Chen et al., 2023)

{{<citation>}}

Lingqiang Chen, Qinglin Zhao, Guanghui Li, Mengchu Zhou, Chenglong Dai, Yiming Feng. (2023)  
**A Sparse Cross Attention-based Graph Convolution Network with Auxiliary Information Awareness for Traffic Flow Prediction**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, Attention  
[Paper Link](http://arxiv.org/abs/2312.09050v1)  

---


**ABSTRACT**  
Deep graph convolution networks (GCNs) have recently shown excellent performance in traffic prediction tasks. However, they face some challenges. First, few existing models consider the influence of auxiliary information, i.e., weather and holidays, which may result in a poor grasp of spatial-temporal dynamics of traffic data. Second, both the construction of a dynamic adjacent matrix and regular graph convolution operations have quadratic computation complexity, which restricts the scalability of GCN-based models. To address such challenges, this work proposes a deep encoder-decoder model entitled AIMSAN. It contains an auxiliary information-aware module (AIM) and sparse cross attention-based graph convolution network (SAN). The former learns multi-attribute auxiliary information and obtains its embedded presentation of different time-window sizes. The latter uses a cross-attention mechanism to construct dynamic adjacent matrices by fusing traffic data and embedded auxiliary data. Then, SAN applies diffusion GCN on traffic data to mine rich spatial-temporal dynamics. Furthermore, AIMSAN considers and uses the spatial sparseness of traffic nodes to reduce the quadratic computation complexity. Experimental results on three public traffic datasets demonstrate that the proposed method outperforms other counterparts in terms of various performance indices. Specifically, the proposed method has competitive performance with the state-of-the-art algorithms but saves 35.74% of GPU memory usage, 42.25% of training time, and 45.51% of validation time on average.

{{</citation>}}


### (45/169) Proving Conjectures Acquired by Composing Multiple Biases (Jovial Cheukam-Ngouonou et al., 2023)

{{<citation>}}

Jovial Cheukam-Ngouonou, Ramiz Gindullin, Nicolas Beldiceanu, Rémi Douence, Claude-Guy Quimper. (2023)  
**Proving Conjectures Acquired by Composing Multiple Biases**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI, math-CO  
Keywords: AI, Bias  
[Paper Link](http://arxiv.org/abs/2312.08990v1)  

---


**ABSTRACT**  
We present the proofs of the conjectures mentioned in the paper published in the proceedings of the 2024 AAAI conference [1], and discovered by the decomposition methods presented in the same paper.

{{</citation>}}


### (46/169) Math-Shepherd: A Label-Free Step-by-Step Verifier for LLMs in Mathematical Reasoning (Peiyi Wang et al., 2023)

{{<citation>}}

Peiyi Wang, Lei Li, Zhihong Shao, R. X. Xu, Damai Dai, Yifei Li, Deli Chen, Y. Wu, Zhifang Sui. (2023)  
**Math-Shepherd: A Label-Free Step-by-Step Verifier for LLMs in Mathematical Reasoning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-LG, cs.AI  
Keywords: LLaMA, Reasoning  
[Paper Link](http://arxiv.org/abs/2312.08935v1)  

---


**ABSTRACT**  
Large language models (LLMs) have demonstrated remarkable capabilities across a wide range of tasks. However, even the most advanced open-source LLMs, such as the LLaMA family models, still face challenges when it comes to accurately solving complex multi-step mathematical problems. In this paper, we present an innovative process-oriented math verifier called \textbf{Math-Shepherd}, which assigns a reward score to each step of the LLM's outputs on math problems. The training of Math-Shepherd is achieved using automatically constructed process-wise supervision data, breaking the bottleneck of heavy reliance on manual annotation in existing work. With the guidance of Math-Shepherd, a series of open-source LLMs demonstrate exceptional performance. Among them, DeepSeek 67B \citep{DeepSeek-llm} stands out by achieving accuracy rates of 93.3\% on the GSM8K dataset and 48.1\% on the MATH dataset, without external enhancement such as tool usage. Our Math-Shepherd also outperforms the self-consistency method and other existing verification models. We believe that automatic process supervision holds significant potential for the future evolution of LLMs.

{{</citation>}}


### (47/169) Modeling Complex Mathematical Reasoning via Large Language Model based MathAgent (Haoran Liao et al., 2023)

{{<citation>}}

Haoran Liao, Qinyi Du, Shaohua Hu, Hao He, Yanyan Xu, Jidong Tian, Yaohui Jin. (2023)  
**Modeling Complex Mathematical Reasoning via Large Language Model based MathAgent**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: GPT, GPT-4, Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2312.08926v2)  

---


**ABSTRACT**  
Large language models (LLMs) face challenges in solving complex mathematical problems that require comprehensive capacities to parse the statements, associate domain knowledge, perform compound logical reasoning, and integrate the intermediate rationales. Tackling all these problems once could be arduous for LLMs, thus leading to confusion in generation. In this work, we explore the potential of enhancing LLMs with agents by meticulous decomposition and modeling of mathematical reasoning process. Specifically, we propose a formal description of the mathematical solving and extend LLMs with an agent-based zero-shot framework named $\bf{P}$lanner-$\bf{R}$easoner-$\bf{E}$xecutor-$\bf{R}$eflector (PRER). We further provide and implement two MathAgents that define the logical forms and inherent relations via a pool of actions in different grains and orientations: MathAgent-M adapts its actions to LLMs, while MathAgent-H aligns with humankind. Experiments on miniF2F and MATH have demonstrated the effectiveness of PRER and proposed MathAgents, achieving an increase of $12.3\%$($53.9\%\xrightarrow{}66.2\%$) on the MiniF2F, $9.2\%$ ($49.8\%\xrightarrow{}59.0\%$) on MATH, and $13.2\%$($23.2\%\xrightarrow{}35.4\%$) for level-5 problems of MATH against GPT-4. Further analytical results provide more insightful perspectives on exploiting the behaviors of LLMs as agents.

{{</citation>}}


### (48/169) Knowledge-Driven Modulation of Neural Networks with Attention Mechanism for Next Activity Prediction (Ivan Donadello et al., 2023)

{{<citation>}}

Ivan Donadello, Jonghyeon Ko, Fabrizio Maria Maggi, Jan Mendling, Francesco Riva, Matthias Weidlich. (2023)  
**Knowledge-Driven Modulation of Neural Networks with Attention Mechanism for Next Activity Prediction**  

---
Primary Category: cs.AI  
Categories: 68T20 (Primary) 68T01, 68T05, 68T37 (Secondary), I-2-6; I-2-8; I-2-m, cs-AI, cs-LG, cs-NE, cs.AI, stat-ML  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.08847v1)  

---


**ABSTRACT**  
Predictive Process Monitoring (PPM) aims at leveraging historic process execution data to predict how ongoing executions will continue up to their completion. In recent years, PPM techniques for the prediction of the next activities have matured significantly, mainly thanks to the use of Neural Networks (NNs) as a predictor. While their performance is difficult to beat in the general case, there are specific situations where background process knowledge can be helpful. Such knowledge can be leveraged for improving the quality of predictions for exceptional process executions or when the process changes due to a concept drift. In this paper, we present a Symbolic[Neuro] system that leverages background knowledge expressed in terms of a procedural process model to offset the under-sampling in the training data. More specifically, we make predictions using NNs with attention mechanism, an emerging technology in the NN field. The system has been tested on several real-life logs showing an improvement in the performance of the prediction task.

{{</citation>}}


### (49/169) Artificial Intelligence and Human Geography (Song Gao, 2023)

{{<citation>}}

Song Gao. (2023)  
**Artificial Intelligence and Human Geography**  

---
Primary Category: cs.AI  
Categories: I-2, cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.08827v1)  

---


**ABSTRACT**  
This paper examines the recent advances and applications of AI in human geography especially the use of machine (deep) learning, including place representation and modeling, spatial analysis and predictive mapping, and urban planning and design. AI technologies have enabled deeper insights into complex human-environment interactions, contributing to more effective scientific exploration, understanding of social dynamics, and spatial decision-making. Furthermore, human geography offers crucial contributions to AI, particularly in context-aware model development, human-centered design, biases and ethical considerations, and data privacy. The synergy beween AI and human geography is essential for addressing global challenges like disaster resilience, poverty, and equitable resource access. This interdisciplinary collaboration between AI and geography will help advance the development of GeoAI and promise a better and sustainable world for all.

{{</citation>}}


### (50/169) Automated Process Planning Based on a Semantic Capability Model and SMT (Aljosha Köcher et al., 2023)

{{<citation>}}

Aljosha Köcher, Luis Miguel Vieira da Silva, Alexander Fay. (2023)  
**Automated Process Planning Based on a Semantic Capability Model and SMT**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LO, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.08801v1)  

---


**ABSTRACT**  
In research of manufacturing systems and autonomous robots, the term capability is used for a machine-interpretable specification of a system function. Approaches in this research area develop information models that capture all information relevant to interpret the requirements, effects and behavior of functions. These approaches are intended to overcome the heterogeneity resulting from the various types of processes and from the large number of different vendors. However, these models and associated methods do not offer solutions for automated process planning, i.e. finding a sequence of individual capabilities required to manufacture a certain product or to accomplish a mission using autonomous robots. Instead, this is a typical task for AI planning approaches, which unfortunately require a high effort to create the respective planning problem descriptions. In this paper, we present an approach that combines these two topics: Starting from a semantic capability model, an AI planning problem is automatically generated. The planning problem is encoded using Satisfiability Modulo Theories and uses an existing solver to find valid capability sequences including required parameter values. The approach also offers possibilities to integrate existing human expertise and to provide explanations for human operators in order to help understand planning decisions.

{{</citation>}}


### (51/169) Multi-modal Latent Space Learning for Chain-of-Thought Reasoning in Language Models (Liqi He et al., 2023)

{{<citation>}}

Liqi He, Zuchao Li, Xiantao Cai, Ping Wang. (2023)  
**Multi-modal Latent Space Learning for Chain-of-Thought Reasoning in Language Models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Language Model, QA, Reasoning  
[Paper Link](http://arxiv.org/abs/2312.08762v1)  

---


**ABSTRACT**  
Chain-of-thought (CoT) reasoning has exhibited impressive performance in language models for solving complex tasks and answering questions. However, many real-world questions require multi-modal information, such as text and images. Previous research on multi-modal CoT has primarily focused on extracting fixed image features from off-the-shelf vision models and then fusing them with text using attention mechanisms. This approach has limitations because these vision models were not designed for complex reasoning tasks and do not align well with language thoughts. To overcome this limitation, we introduce a novel approach for multi-modal CoT reasoning that utilizes latent space learning via diffusion processes to generate effective image features that align with language thoughts. Our method fuses image features and text representations at a deep level and improves the complex reasoning ability of multi-modal CoT. We demonstrate the efficacy of our proposed method on multi-modal ScienceQA and machine translation benchmarks, achieving state-of-the-art performance on ScienceQA. Overall, our approach offers a more robust and effective solution for multi-modal reasoning in language models, enhancing their ability to tackle complex real-world problems.

{{</citation>}}


### (52/169) Quantifying Divergence for Human-AI Collaboration and Cognitive Trust (Müge Kural et al., 2023)

{{<citation>}}

Müge Kural, Ali Gebeşçe, Tilek Chubakov, Gözde Gül Şahin. (2023)  
**Quantifying Divergence for Human-AI Collaboration and Cognitive Trust**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.08722v1)  

---


**ABSTRACT**  
Predicting the collaboration likelihood and measuring cognitive trust to AI systems is more important than ever. To do that, previous research mostly focus solely on the model features (e.g., accuracy, confidence) and ignore the human factor. To address that, we propose several decision-making similarity measures based on divergence metrics (e.g., KL, JSD) calculated over the labels acquired from humans and a wide range of models. We conduct a user study on a textual entailment task, where the users are provided with soft labels from various models and asked to pick the closest option to them. The users are then shown the similarities/differences to their most similar model and are surveyed for their likelihood of collaboration and cognitive trust to the selected system. Finally, we qualitatively and quantitatively analyze the relation between the proposed decision-making similarity measures and the survey results. We find that people tend to collaborate with their most similar models -- measured via JSD -- yet this collaboration does not necessarily imply a similar level of cognitive trust. We release all resources related to the user study (e.g., design, outputs), models, and metrics at our repo.

{{</citation>}}


### (53/169) Rational Sensibility: LLM Enhanced Empathetic Response Generation Guided by Self-presentation Theory (Linzhuang Sun et al., 2023)

{{<citation>}}

Linzhuang Sun, Nan Xu, Jingxuan Wei, Bihui Yu, Liping Bu, Yin Luo. (2023)  
**Rational Sensibility: LLM Enhanced Empathetic Response Generation Guided by Self-presentation Theory**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: LLaMA  
[Paper Link](http://arxiv.org/abs/2312.08702v2)  

---


**ABSTRACT**  
Having the ability to empathize is crucial for accurately representing human behavior during conversations. Despite numerous research aim to improve the cognitive capability of models by incorporating external knowledge, there has been limited attention on the sensible and rational expression of the conversation itself, which are crucial components of the cognitive empathy. Guided by self-presentation theory in sociology, we have designed an innovative categorical approach that segregates historical dialogues into sensible and rational sentences and subsequently elucidate the context through the designed attention mechanism. However, the rational information within the conversation is restricted and the external knowledge used in previous methods have limitations of semantic contradiction and narrow vision field. Considering the impressive performance of LLM in the domain of intelligent agent. We employ LLaMA2-70b as a rational brain to analyze the profound logical information maintained in conversations, which assists the model assessing the balance of sensibility and rationality to produce quality empathetic responses. Experimental evaluations demonstrate that our method outperforms other comparable methods on both automatic and human evaluations.

{{</citation>}}


### (54/169) Heterogeneous Graph Neural Architecture Search with GPT-4 (Haoyuan Dong et al., 2023)

{{<citation>}}

Haoyuan Dong, Yang Gao, Haishuai Wang, Hong Yang, Peng Zhang. (2023)  
**Heterogeneous Graph Neural Architecture Search with GPT-4**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: GNN, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2312.08680v1)  

---


**ABSTRACT**  
Heterogeneous graph neural architecture search (HGNAS) represents a powerful tool for automatically designing effective heterogeneous graph neural networks. However, existing HGNAS algorithms suffer from inefficient searches and unstable results. In this paper, we present a new GPT-4 based HGNAS model to improve the search efficiency and search accuracy of HGNAS. Specifically, we present a new GPT-4 enhanced Heterogeneous Graph Neural Architecture Search (GHGNAS for short). The basic idea of GHGNAS is to design a set of prompts that can guide GPT-4 toward the task of generating new heterogeneous graph neural architectures. By iteratively asking GPT-4 with the prompts, GHGNAS continually validates the accuracy of the generated HGNNs and uses the feedback to further optimize the prompts. Experimental results show that GHGNAS can design new HGNNs by leveraging the powerful generalization capability of GPT-4. Moreover, GHGNAS runs more effectively and stably than previous HGNAS models based on reinforcement learning and differentiable search algorithms.

{{</citation>}}


### (55/169) ChatSOS: LLM-based knowledge Q&A system for safety engineering (Haiyang Tang et al., 2023)

{{<citation>}}

Haiyang Tang, Zhenyi Liu, Dongping Chen, Qingzhao Chu. (2023)  
**ChatSOS: LLM-based knowledge Q&A system for safety engineering**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2312.08629v1)  

---


**ABSTRACT**  
Recent advancements in large language models (LLMs) have notably propelled natural language processing (NLP) capabilities, demonstrating significant potential in safety engineering applications. Despite these advancements, LLMs face constraints in processing specialized tasks, attributed to factors such as corpus size, input processing limitations, and privacy concerns. Obtaining useful information from reliable sources in a limited time is crucial for LLM. Addressing this, our study introduces an LLM-based Q&A system for safety engineering, enhancing the comprehension and response accuracy of the model. We employed prompt engineering to incorporate external knowledge databases, thus enriching the LLM with up-to-date and reliable information. The system analyzes historical incident reports through statistical methods, utilizes vector embedding to construct a vector database, and offers an efficient similarity-based search functionality. Our findings indicate that the integration of external knowledge significantly augments the capabilities of LLM for in-depth problem analysis and autonomous task assignment. It effectively summarizes accident reports and provides pertinent recommendations. This integration approach not only expands LLM applications in safety engineering but also sets a precedent for future developments towards automation and intelligent systems.

{{</citation>}}


## cs.RO (7)



### (56/169) HiER: Highlight Experience Replay and Easy2Hard Curriculum Learning for Boosting Off-Policy Reinforcement Learning Agents (Dániel Horváth et al., 2023)

{{<citation>}}

Dániel Horváth, Jesús Bujalance Martín, Ferenc Gábor Erdős, Zoltán Istenes, Fabien Moutarde. (2023)  
**HiER: Highlight Experience Replay and Easy2Hard Curriculum Learning for Boosting Off-Policy Reinforcement Learning Agents**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.09394v1)  

---


**ABSTRACT**  
Even though reinforcement-learning-based algorithms achieved superhuman performance in many domains, the field of robotics poses significant challenges as the state and action spaces are continuous, and the reward function is predominantly sparse. In this work, we propose: 1) HiER: highlight experience replay that creates a secondary replay buffer for the most relevant experiences, 2) E2H-ISE: an easy2hard data collection curriculum-learning method based on controlling the entropy of the initial state-goal distribution and with it, indirectly, the task difficulty, and 3) HiER+: the combination of HiER and E2H-ISE. They can be applied with or without the techniques of hindsight experience replay (HER) and prioritized experience replay (PER). While both HiER and E2H-ISE surpass the baselines, HiER+ further improves the results and significantly outperforms the state-of-the-art on the push, slide, and pick-and-place robotic manipulation tasks. Our implementation and further media materials are available on the project site.

{{</citation>}}


### (57/169) LLM-MARS: Large Language Model for Behavior Tree Generation and NLP-enhanced Dialogue in Multi-Agent Robot Systems (Artem Lykov et al., 2023)

{{<citation>}}

Artem Lykov, Maria Dronova, Nikolay Naglov, Mikhail Litvinov, Sergei Satsevich, Artem Bazhenov, Vladimir Berman, Aleksei Shcherbak, Dzmitry Tsetserukou. (2023)  
**LLM-MARS: Large Language Model for Behavior Tree Generation and NLP-enhanced Dialogue in Multi-Agent Robot Systems**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Dialog, Dialogue, Falcon, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2312.09348v1)  

---


**ABSTRACT**  
This paper introduces LLM-MARS, first technology that utilizes a Large Language Model based Artificial Intelligence for Multi-Agent Robot Systems. LLM-MARS enables dynamic dialogues between humans and robots, allowing the latter to generate behavior based on operator commands and provide informative answers to questions about their actions. LLM-MARS is built on a transformer-based Large Language Model, fine-tuned from the Falcon 7B model. We employ a multimodal approach using LoRa adapters for different tasks. The first LoRa adapter was developed by fine-tuning the base model on examples of Behavior Trees and their corresponding commands. The second LoRa adapter was developed by fine-tuning on question-answering examples. Practical trials on a multi-agent system of two robots within the Eurobot 2023 game rules demonstrate promising results. The robots achieve an average task execution accuracy of 79.28% in compound commands. With commands containing up to two tasks accuracy exceeded 90%. Evaluation confirms the system's answers on operators questions exhibit high accuracy, relevance, and informativeness. LLM-MARS and similar multi-agent robotic systems hold significant potential to revolutionize logistics, enabling autonomous exploration missions and advancing Industry 5.0.

{{</citation>}}


### (58/169) A Sim-to-Real Deep Learning-based Framework for Autonomous Nano-drone Racing (Lorenzo Lamberti et al., 2023)

{{<citation>}}

Lorenzo Lamberti, Elia Cereda, Gabriele Abbate, Lorenzo Bellone, Victor Javier Kartsch Morinigo, Michał Barcis, Agata Barcis, Alessandro Giusti, Francesco Conti, Daniele Palossi. (2023)  
**A Sim-to-Real Deep Learning-based Framework for Autonomous Nano-drone Racing**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs-SY, cs.RO, eess-IV, eess-SY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.08991v1)  

---


**ABSTRACT**  
Autonomous drone racing competitions are a proxy to improve unmanned aerial vehicles' perception, planning, and control skills. The recent emergence of autonomous nano-sized drone racing imposes new challenges, as their ~10cm form factor heavily restricts the resources available onboard, including memory, computation, and sensors. This paper describes the methodology and technical implementation of the system winning the first autonomous nano-drone racing international competition: the IMAV 2022 Nanocopter AI Challenge. We developed a fully onboard deep learning approach for visual navigation trained only on simulation images to achieve this goal. Our approach includes a convolutional neural network for obstacle avoidance, a sim-to-real dataset collection procedure, and a navigation policy that we selected, characterized, and adapted through simulation and actual in-field experiments. Our system ranked 1st among seven competing teams at the competition. In our best attempt, we scored 115m of traveled distance in the allotted 5-minute flight, never crashing while dodging static and dynamic obstacles. Sharing our knowledge with the research community, we aim to provide a solid groundwork to foster future development in this field.

{{</citation>}}


### (59/169) How to Raise a Robot -- A Case for Neuro-Symbolic AI in Constrained Task Planning for Humanoid Assistive Robots (Niklas Hemken et al., 2023)

{{<citation>}}

Niklas Hemken, Florian Jacob, Fabian Peller-Konrad, Rainer Kartmann, Tamim Asfour, Hannes Hartenstein. (2023)  
**How to Raise a Robot -- A Case for Neuro-Symbolic AI in Constrained Task Planning for Humanoid Assistive Robots**  

---
Primary Category: cs.RO  
Categories: cs-CR, cs-LG, cs-RO, cs.RO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.08820v2)  

---


**ABSTRACT**  
Humanoid robots will be able to assist humans in their daily life, in particular due to their versatile action capabilities. However, while these robots need a certain degree of autonomy to learn and explore, they also should respect various constraints, for access control and beyond. We explore the novel field of incorporating privacy, security, and access control constraints with robot task planning approaches. We report preliminary results on the classical symbolic approach, deep-learned neural networks, and modern ideas using large language models as knowledge base. From analyzing their trade-offs, we conclude that a hybrid approach is necessary, and thereby present a new use case for the emerging field of neuro-symbolic artificial intelligence.

{{</citation>}}


### (60/169) Toward General-Purpose Robots via Foundation Models: A Survey and Meta-Analysis (Yafei Hu et al., 2023)

{{<citation>}}

Yafei Hu, Quanting Xie, Vidhi Jain, Jonathan Francis, Jay Patrikar, Nikhil Keetha, Seungchan Kim, Yaqi Xie, Tianyi Zhang, Shibo Zhao, Yu Quan Chong, Chen Wang, Katia Sycara, Matthew Johnson-Roberson, Dhruv Batra, Xiaolong Wang, Sebastian Scherer, Zsolt Kira, Fei Xia, Yonatan Bisk. (2023)  
**Toward General-Purpose Robots via Foundation Models: A Survey and Meta-Analysis**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-CV, cs-LG, cs-RO, cs.RO  
Keywords: Computer Vision, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2312.08782v2)  

---


**ABSTRACT**  
Building general-purpose robots that can operate seamlessly, in any environment, with any object, and utilizing various skills to complete diverse tasks has been a long-standing goal in Artificial Intelligence. Unfortunately, however, most existing robotic systems have been constrained - having been designed for specific tasks, trained on specific datasets, and deployed within specific environments. These systems usually require extensively-labeled data, rely on task-specific models, have numerous generalization issues when deployed in real-world scenarios, and struggle to remain robust to distribution shifts. Motivated by the impressive open-set performance and content generation capabilities of web-scale, large-capacity pre-trained models (i.e., foundation models) in research fields such as Natural Language Processing (NLP) and Computer Vision (CV), we devote this survey to exploring (i) how these existing foundation models from NLP and CV can be applied to the field of robotics, and also exploring (ii) what a robotics-specific foundation model would look like. We begin by providing an overview of what constitutes a conventional robotic system and the fundamental barriers to making it universally applicable. Next, we establish a taxonomy to discuss current work exploring ways to leverage existing foundation models for robotics and develop ones catered to robotics. Finally, we discuss key challenges and promising future directions in using foundation models for enabling general-purpose robotic systems. We encourage readers to view our living GitHub repository of resources, including papers reviewed in this survey as well as related projects and repositories for developing foundation models for robotics.

{{</citation>}}


### (61/169) Quadrupedal Locomotion Control On Inclined Surfaces Using Collocation Method (Adarsh Salagame et al., 2023)

{{<citation>}}

Adarsh Salagame, Maria Gianello, Chenghao Wang, Kaushik Venkatesh, Shreyansh Pitroda, Rohit Rajput, Eric Sihite, Miriam Leeser, Alireza Ramezani. (2023)  
**Quadrupedal Locomotion Control On Inclined Surfaces Using Collocation Method**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs-SY, cs.RO, eess-SY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.08621v1)  

---


**ABSTRACT**  
Inspired by Chukars wing-assisted incline running (WAIR), in this work, we employ a high-fidelity model of our Husky Carbon quadrupedal-legged robot to walk over steep slopes of up to 45 degrees. Chukars use the aerodynamic forces generated by their flapping wings to manipulate ground contact forces and traverse steep slopes and even overhangs. By exploiting the thrusters on Husky, we employed a collocation approach to rapidly resolving the joint and thruster actions. Our approach uses a polynomial approximation of the reduced-order dynamics of Husky, called HROM, to quickly and efficiently find optimal control actions that permit high-slope walking without violating friction cone conditions.

{{</citation>}}


### (62/169) UniTeam: Open Vocabulary Mobile Manipulation Challenge (Andrew Melnik et al., 2023)

{{<citation>}}

Andrew Melnik, Michael Büttner, Leon Harz, Lyon Brown, Gora Chand Nandi, Arjun PS, Gaurav Kumar Yadav, Rahul Kala, Robert Haschke. (2023)  
**UniTeam: Open Vocabulary Mobile Manipulation Challenge**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.08611v1)  

---


**ABSTRACT**  
This report introduces our UniTeam agent - an improved baseline for the "HomeRobot: Open Vocabulary Mobile Manipulation" challenge. The challenge poses problems of navigation in unfamiliar environments, manipulation of novel objects, and recognition of open-vocabulary object classes. This challenge aims to facilitate cross-cutting research in embodied AI using recent advances in machine learning, computer vision, natural language, and robotics. In this work, we conducted an exhaustive evaluation of the provided baseline agent; identified deficiencies in perception, navigation, and manipulation skills; and improved the baseline agent's performance. Notably, enhancements were made in perception - minimizing misclassifications; navigation - preventing infinite loop commitments; picking - addressing failures due to changing object visibility; and placing - ensuring accurate positioning for successful object placement.

{{</citation>}}


## cs.CL (23)



### (63/169) Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision (Collin Burns et al., 2023)

{{<citation>}}

Collin Burns, Pavel Izmailov, Jan Hendrik Kirchner, Bowen Baker, Leo Gao, Leopold Aschenbrenner, Yining Chen, Adrien Ecoffet, Manas Joglekar, Jan Leike, Ilya Sutskever, Jeff Wu. (2023)  
**Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-3.5, GPT-4, NLP  
[Paper Link](http://arxiv.org/abs/2312.09390v1)  

---


**ABSTRACT**  
Widely used alignment techniques, such as reinforcement learning from human feedback (RLHF), rely on the ability of humans to supervise model behavior - for example, to evaluate whether a model faithfully followed instructions or generated safe outputs. However, future superhuman models will behave in complex ways too difficult for humans to reliably evaluate; humans will only be able to weakly supervise superhuman models. We study an analogy to this problem: can weak model supervision elicit the full capabilities of a much stronger model? We test this using a range of pretrained language models in the GPT-4 family on natural language processing (NLP), chess, and reward modeling tasks. We find that when we naively finetune strong pretrained models on labels generated by a weak model, they consistently perform better than their weak supervisors, a phenomenon we call weak-to-strong generalization. However, we are still far from recovering the full capabilities of strong models with naive finetuning alone, suggesting that techniques like RLHF may scale poorly to superhuman models without further work. We find that simple methods can often significantly improve weak-to-strong generalization: for example, when finetuning GPT-4 with a GPT-2-level supervisor and an auxiliary confidence loss, we can recover close to GPT-3.5-level performance on NLP tasks. Our results suggest that it is feasible to make empirical progress today on a fundamental challenge of aligning superhuman models.

{{</citation>}}


### (64/169) Arabic Mini-ClimateGPT : A Climate Change and Sustainability Tailored Arabic LLM (Sahal Shaji Mullappilly et al., 2023)

{{<citation>}}

Sahal Shaji Mullappilly, Abdelrahman Shaker, Omkar Thawakar, Hisham Cholakkal, Rao Muhammad Anwer, Salman Khan, Fahad Shahbaz Khan. (2023)  
**Arabic Mini-ClimateGPT : A Climate Change and Sustainability Tailored Arabic LLM**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2312.09366v1)  

---


**ABSTRACT**  
Climate change is one of the most significant challenges we face together as a society. Creating awareness and educating policy makers the wide-ranging impact of climate change is an essential step towards a sustainable future. Recently, Large Language Models (LLMs) like ChatGPT and Bard have shown impressive conversational abilities and excel in a wide variety of NLP tasks. While these models are close-source, recently alternative open-source LLMs such as Stanford Alpaca and Vicuna have shown promising results. However, these open-source models are not specifically tailored for climate related domain specific information and also struggle to generate meaningful responses in other languages such as, Arabic. To this end, we propose a light-weight Arabic Mini-ClimateGPT that is built on an open-source LLM and is specifically fine-tuned on a conversational-style instruction tuning curated Arabic dataset Clima500-Instruct with over 500k instructions about climate change and sustainability. Further, our model also utilizes a vector embedding based retrieval mechanism during inference. We validate our proposed model through quantitative and qualitative evaluations on climate-related queries. Our model surpasses the baseline LLM in 88.3% of cases during ChatGPT-based evaluation. Furthermore, our human expert evaluation reveals an 81.6% preference for our model's responses over multiple popular open-source models. Our open-source demos, code-base and models are available here https://github.com/mbzuai-oryx/ClimateGPT.

{{</citation>}}


### (65/169) Self-Evaluation Improves Selective Generation in Large Language Models (Jie Ren et al., 2023)

{{<citation>}}

Jie Ren, Yao Zhao, Tu Vu, Peter J. Liu, Balaji Lakshminarayanan. (2023)  
**Self-Evaluation Improves Selective Generation in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: GPT, Language Model, PaLM, QA  
[Paper Link](http://arxiv.org/abs/2312.09300v1)  

---


**ABSTRACT**  
Safe deployment of large language models (LLMs) may benefit from a reliable method for assessing their generated content to determine when to abstain or to selectively generate. While likelihood-based metrics such as perplexity are widely employed, recent research has demonstrated the limitations of using sequence-level probability estimates given by LLMs as reliable indicators of generation quality. Conversely, LLMs have demonstrated strong calibration at the token level, particularly when it comes to choosing correct answers in multiple-choice questions or evaluating true/false statements. In this work, we reformulate open-ended generation tasks into token-level prediction tasks, and leverage LLMs' superior calibration at the token level. We instruct an LLM to self-evaluate its answers, employing either a multi-way comparison or a point-wise evaluation approach, with the option to include a ``None of the above'' option to express the model's uncertainty explicitly. We benchmark a range of scoring methods based on self-evaluation and evaluate their performance in selective generation using TruthfulQA and TL;DR. Through experiments with PaLM-2 and GPT-3, we demonstrate that self-evaluation based scores not only improve accuracy, but also correlate better with the overall quality of generated content.

{{</citation>}}


### (66/169) Mitigating Outlier Activations in Low-Precision Fine-Tuning of Language Models (Alireza Ghaffari et al., 2023)

{{<citation>}}

Alireza Ghaffari, Justin Yu, Mahsa Ghazvini Nejad, Masoud Asgharian, Boxing Chen, Vahid Partovi Nia. (2023)  
**Mitigating Outlier Activations in Low-Precision Fine-Tuning of Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.09211v2)  

---


**ABSTRACT**  
Low-precision fine-tuning of language models has gained prominence as a cost-effective and energy-efficient approach to deploying large-scale models in various applications. However, this approach is susceptible to the existence of outlier values in activation. The outlier values in the activation can negatively affect the performance of fine-tuning language models in the low-precision regime since they affect the scaling factor and thus make representing smaller values harder. This paper investigates techniques for mitigating outlier activation in low-precision integer fine-tuning of the language models. Our proposed novel approach enables us to represent the outlier activation values in 8-bit integers instead of floating-point (FP16) values. The benefit of using integers for outlier values is that it enables us to use operator tiling to avoid performing 16-bit integer matrix multiplication to address this problem effectively. We provide theoretical analysis and supporting experiments to demonstrate the effectiveness of our approach in improving the robustness and performance of low-precision fine-tuned language models.

{{</citation>}}


### (67/169) WikiMuTe: A web-sourced dataset of semantic descriptions for music audio (Benno Weck et al., 2023)

{{<citation>}}

Benno Weck, Holger Kirchhoff, Peter Grosche, Xavier Serra. (2023)  
**WikiMuTe: A web-sourced dataset of semantic descriptions for music audio**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-IR, cs-LG, cs-SD, cs.CL, eess-AS  
Keywords: Information Retrieval  
[Paper Link](http://arxiv.org/abs/2312.09207v1)  

---


**ABSTRACT**  
Multi-modal deep learning techniques for matching free-form text with music have shown promising results in the field of Music Information Retrieval (MIR). Prior work is often based on large proprietary data while publicly available datasets are few and small in size. In this study, we present WikiMuTe, a new and open dataset containing rich semantic descriptions of music. The data is sourced from Wikipedia's rich catalogue of articles covering musical works. Using a dedicated text-mining pipeline, we extract both long and short-form descriptions covering a wide range of topics related to music content such as genre, style, mood, instrumentation, and tempo. To show the use of this data, we train a model that jointly learns text and audio representations and performs cross-modal retrieval. The model is evaluated on two tasks: tag-based music retrieval and music auto-tagging. The results show that while our approach has state-of-the-art performance on multiple tasks, but still observe a difference in performance depending on the data used for training.

{{</citation>}}


### (68/169) The Earth is Flat because...: Investigating LLMs' Belief towards Misinformation via Persuasive Conversation (Rongwu Xu et al., 2023)

{{<citation>}}

Rongwu Xu, Brian S. Lin, Shujian Yang, Tianqi Zhang, Weiyan Shi, Tianwei Zhang, Zhixuan Fang, Wei Xu, Han Qiu. (2023)  
**The Earth is Flat because...: Investigating LLMs' Belief towards Misinformation via Persuasive Conversation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CR, cs-CY, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.09085v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) encapsulate vast amounts of knowledge but still remain vulnerable to external misinformation. Existing research mainly studied this susceptibility behavior in a single-turn setting. However, belief can change during a multi-turn conversation, especially a persuasive one. Therefore, in this study, we delve into LLMs' susceptibility to persuasive conversations, particularly on factual questions that they can answer correctly. We first curate the Farm (i.e., Fact to Misinform) dataset, which contains factual questions paired with systematically generated persuasive misinformation. Then, we develop a testing framework to track LLMs' belief changes in a persuasive dialogue. Through extensive experiments, we find that LLMs' correct beliefs on factual knowledge can be easily manipulated by various persuasive strategies.

{{</citation>}}


### (69/169) Towards Verifiable Text Generation with Evolving Memory and Self-Reflection (Hao Sun et al., 2023)

{{<citation>}}

Hao Sun, Hengyi Cai, Bo Wang, Yingyan Hou, Xiaochi Wei, Shuaiqiang Wang, Yan Zhang, Dawei Yin. (2023)  
**Towards Verifiable Text Generation with Evolving Memory and Self-Reflection**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Text Generation  
[Paper Link](http://arxiv.org/abs/2312.09075v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) face several challenges, including the tendency to produce incorrect outputs, known as hallucination. An effective solution is verifiable text generation, which prompts LLMs to generate content with citations for accuracy verification. However, verifiable text generation is non-trivial due to the focus-shifting phenomenon, the dilemma between the precision and scope in document retrieval, and the intricate reasoning required to discern the relationship between the claim and citations. In this paper, we present VTG, an innovative approach for Verifiable Text Generation with evolving memory and self-reflection. VTG maintains evolving long short-term memory to retain both valuable documents and up-to-date documents. Active retrieval and diverse query generation are utilized to enhance both the precision and scope of the retrieved documents. Furthermore, VTG features a two-tier verifier and an evidence finder, enabling rethinking and reflection on the relationship between the claim and citations. We conduct extensive experiments on five datasets across three knowledge-intensive tasks and the results reveal that VTG significantly outperforms existing baselines.

{{</citation>}}


### (70/169) Topic Bias in Emotion Classification (Maximilian Wegge et al., 2023)

{{<citation>}}

Maximilian Wegge, Roman Klinger. (2023)  
**Topic Bias in Emotion Classification**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2312.09043v1)  

---


**ABSTRACT**  
Emotion corpora are typically sampled based on keyword/hashtag search or by asking study participants to generate textual instances. In any case, these corpora are not uniform samples representing the entirety of a domain. We hypothesize that this practice of data acquisition leads to unrealistic correlations between overrepresented topics in these corpora that harm the generalizability of models. Such topic bias could lead to wrong predictions for instances like "I organized the service for my aunt's funeral." when funeral events are over-represented for instances labeled with sadness, despite the emotion of pride being more appropriate here. In this paper, we study this topic bias both from the data and the modeling perspective. We first label a set of emotion corpora automatically via topic modeling and show that emotions in fact correlate with specific topics. Further, we see that emotion classifiers are confounded by such topics. Finally, we show that the established debiasing method of adversarial correction via gradient reversal mitigates the issue. Our work points out issues with existing emotion corpora and that more representative resources are required for fair evaluation of models predicting affective concepts from text.

{{</citation>}}


### (71/169) TAP4LLM: Table Provider on Sampling, Augmenting, and Packing Semi-structured Data for Large Language Model Reasoning (Yuan Sui et al., 2023)

{{<citation>}}

Yuan Sui, Jiaru Zou, Mengyu Zhou, Xinyi He, Lun Du, Shi Han, Dongmei Zhang. (2023)  
**TAP4LLM: Table Provider on Sampling, Augmenting, and Packing Semi-structured Data for Large Language Model Reasoning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2312.09039v1)  

---


**ABSTRACT**  
Table reasoning has shown remarkable progress in a wide range of table-based tasks. These challenging tasks require reasoning over both free-form natural language (NL) questions and semi-structured tabular data. However, previous table reasoning solutions suffer from significant performance degradation on "huge" tables. In addition, most existing methods struggle to reason over complex questions since they lack essential information or they are scattered in different places. To alleviate these challenges, we exploit a table provider, namely TAP4LLM, on versatile sampling, augmentation, and packing methods to achieve effective semi-structured data reasoning using large language models (LLMs), which 1) decompose raw tables into sub-tables with specific rows or columns based on the rules or semantic similarity; 2) augment table information by extracting semantic and statistical metadata from raw tables while retrieving relevant knowledge from trustworthy knowledge sources (e.g., Wolfram Alpha, Wikipedia); 3) pack sampled tables with augmented knowledge into sequence prompts for LLMs reasoning while balancing the token allocation trade-off. We show that TAP4LLM allows for different components as plug-ins, enhancing LLMs' understanding of structured data in diverse tabular tasks.

{{</citation>}}


### (72/169) ComOM at VLSP 2023: A Dual-Stage Framework with BERTology and Unified Multi-Task Instruction Tuning Model for Vietnamese Comparative Opinion Mining (Dang Van Thin et al., 2023)

{{<citation>}}

Dang Van Thin, Duong Ngoc Hao, Ngan Luu-Thuy Nguyen. (2023)  
**ComOM at VLSP 2023: A Dual-Stage Framework with BERTology and Unified Multi-Task Instruction Tuning Model for Vietnamese Comparative Opinion Mining**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, BERTology  
[Paper Link](http://arxiv.org/abs/2312.09000v1)  

---


**ABSTRACT**  
The ComOM shared task aims to extract comparative opinions from product reviews in Vietnamese language. There are two sub-tasks, including (1) Comparative Sentence Identification (CSI) and (2) Comparative Element Extraction (CEE). The first task is to identify whether the input is a comparative review, and the purpose of the second task is to extract the quintuplets mentioned in the comparative review. To address this task, our team proposes a two-stage system based on fine-tuning a BERTology model for the CSI task and unified multi-task instruction tuning for the CEE task. Besides, we apply the simple data augmentation technique to increase the size of the dataset for training our model in the second stage. Experimental results show that our approach outperforms the other competitors and has achieved the top score on the official private test.

{{</citation>}}


### (73/169) Detecting value-expressive text posts in Russian social media (Maria Milkova et al., 2023)

{{<citation>}}

Maria Milkova, Maksim Rudnev, Lidia Okolskaya. (2023)  
**Detecting value-expressive text posts in Russian social media**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2312.08968v1)  

---


**ABSTRACT**  
Basic values are concepts or beliefs which pertain to desirable end-states and transcend specific situations. Studying personal values in social media can illuminate how and why societal values evolve especially when the stimuli-based methods, such as surveys, are inefficient, for instance, in hard-to-reach populations. On the other hand, user-generated content is driven by the massive use of stereotyped, culturally defined speech constructions rather than authentic expressions of personal values. We aimed to find a model that can accurately detect value-expressive posts in Russian social media VKontakte. A training dataset of 5,035 posts was annotated by three experts, 304 crowd-workers and ChatGPT. Crowd-workers and experts showed only moderate agreement in categorizing posts. ChatGPT was more consistent but struggled with spam detection. We applied an ensemble of human- and AI-assisted annotation involving active learning approach, subsequently trained several LLMs and selected a model based on embeddings from pre-trained fine-tuned rubert-tiny2, and reached a high quality of value detection with F1 = 0.75 (F1-macro = 0.80). This model provides a crucial step to a study of values within and between Russian social media users.

{{</citation>}}


### (74/169) Boosting LLM Reasoning: Push the Limits of Few-shot Learning with Reinforced In-Context Pruning (Xijie Huang et al., 2023)

{{<citation>}}

Xijie Huang, Li Lyna Zhang, Kwang-Ting Cheng, Mao Yang. (2023)  
**Boosting LLM Reasoning: Push the Limits of Few-shot Learning with Reinforced In-Context Pruning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-3.5, LLaMA, PaLM, Pruning, Reasoning  
[Paper Link](http://arxiv.org/abs/2312.08901v1)  

---


**ABSTRACT**  
Large language models (LLMs) have shown impressive capabilities in various tasks, yet they still struggle with math reasoning. Despite efforts to optimize Chain-of-Thoughts (CoT) prompts and fine-tune LLMs, the potential of few-shot learning remains unexplored. In this work, we propose CoT-Max, a novel approach pushing the boundaries of few-shot CoT learning to improve LLM math reasoning capabilities. CoT-Max addresses the challenges of the selection of useful examples and limited number of examples due to restricted context window length. Inspired by our observation that natural language inputs contain many redundancy, we propose a coarse-to-fine pruner as a plug-and-play module for LLMs, which first identifies crucial CoT examples from a large batch and then further prunes unimportant tokens. To train the pruner, we collect a math reasoning dataset with diverse difficulty and steps, introduce a reward to measure both the input's effectiveness for math reasoning and token length constraints, and propose a novel training approach with reinforcement learning. As a result, CoT-Max significantly outperforms CoT and few-shot prompting baselines across various LLMs (LLaMA2-7B, 13B, 70B) and 5 mathematical datasets, achieving up to 4.55% absolute improvements. Remarkably, without any fine-tuning, LLaMA2-70B with CoT-Max surpasses GPT-3.5 and a wide range of larger LLMs (PaLM, Minerva, etc.) on the GSM8K.

{{</citation>}}


### (75/169) Evaluating Large Language Models for Health-related Queries with Presuppositions (Navreet Kaur et al., 2023)

{{<citation>}}

Navreet Kaur, Monojit Choudhury, Danish Pruthi. (2023)  
**Evaluating Large Language Models for Health-related Queries with Presuppositions**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-HC, cs-LG, cs.CL  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2312.08800v1)  

---


**ABSTRACT**  
As corporations rush to integrate large language models (LLMs) to their search offerings, it is critical that they provide factually accurate information that is robust to any presuppositions that a user may express. In this work, we introduce UPHILL, a dataset consisting of health-related queries with varying degrees of presuppositions. Using UPHILL, we evaluate the factual accuracy and consistency of InstructGPT, ChatGPT, and BingChat models. We find that while model responses rarely disagree with true health claims (posed as questions), they often fail to challenge false claims: responses from InstructGPT agree with 32% of the false claims, ChatGPT 26% and BingChat 23%. As we increase the extent of presupposition in input queries, the responses from InstructGPT and ChatGPT agree with the claim considerably more often, regardless of its veracity. Responses from BingChat, which rely on retrieved webpages, are not as susceptible. Given the moderate factual accuracy, and the inability of models to consistently correct false assumptions, our work calls for a careful assessment of current LLMs for use in high-stakes scenarios.

{{</citation>}}


### (76/169) PROPRES: Investigating the Projectivity of Presupposition with Various Triggers and Environments (Daiki Asami et al., 2023)

{{<citation>}}

Daiki Asami, Saku Sugawara. (2023)  
**PROPRES: Investigating the Projectivity of Presupposition with Various Triggers and Environments**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2312.08755v1)  

---


**ABSTRACT**  
What makes a presupposition of an utterance -- information taken for granted by its speaker -- different from other pragmatic inferences such as an entailment is projectivity (e.g., the negative sentence the boy did not stop shedding tears presupposes the boy had shed tears before). The projectivity may vary depending on the combination of presupposition triggers and environments. However, prior natural language understanding studies fail to take it into account as they either use no human baseline or include only negation as an entailment-canceling environment to evaluate models' performance. The current study attempts to reconcile these issues. We introduce a new dataset, projectivity of presupposition (PROPRES, which includes 12k premise-hypothesis pairs crossing six triggers involving some lexical variety with five environments. Our human evaluation reveals that humans exhibit variable projectivity in some cases. However, the model evaluation shows that the best-performed model, DeBERTa, does not fully capture it. Our findings suggest that probing studies on pragmatic inferences should take extra care of the human judgment variability and the combination of linguistic items.

{{</citation>}}


### (77/169) Dissecting vocabulary biases datasets through statistical testing and automated data augmentation for artifact mitigation in Natural Language Inference (Dat Thanh Nguyen, 2023)

{{<citation>}}

Dat Thanh Nguyen. (2023)  
**Dissecting vocabulary biases datasets through statistical testing and automated data augmentation for artifact mitigation in Natural Language Inference**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Natural Language Inference  
[Paper Link](http://arxiv.org/abs/2312.08747v1)  

---


**ABSTRACT**  
In recent years, the availability of large-scale annotated datasets, such as the Stanford Natural Language Inference and the Multi-Genre Natural Language Inference, coupled with the advent of pre-trained language models, has significantly contributed to the development of the natural language inference domain. However, these crowdsourced annotated datasets often contain biases or dataset artifacts, leading to overestimated model performance and poor generalization. In this work, we focus on investigating dataset artifacts and developing strategies to address these issues. Through the utilization of a novel statistical testing procedure, we discover a significant association between vocabulary distribution and text entailment classes, emphasizing vocabulary as a notable source of biases. To mitigate these issues, we propose several automatic data augmentation strategies spanning character to word levels. By fine-tuning the ELECTRA pre-trained language model, we compare the performance of boosted models with augmented data against their baseline counterparts. The experiments demonstrate that the proposed approaches effectively enhance model accuracy and reduce biases by up to 0.66% and 1.14%, respectively.

{{</citation>}}


### (78/169) JPIS: A Joint Model for Profile-based Intent Detection and Slot Filling with Slot-to-Intent Attention (Thinh Pham et al., 2023)

{{<citation>}}

Thinh Pham, Dat Quoc Nguyen. (2023)  
**JPIS: A Joint Model for Profile-based Intent Detection and Slot Filling with Slot-to-Intent Attention**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Attention, Intent Detection  
[Paper Link](http://arxiv.org/abs/2312.08737v2)  

---


**ABSTRACT**  
Profile-based intent detection and slot filling are important tasks aimed at reducing the ambiguity in user utterances by leveraging user-specific supporting profile information. However, research in these two tasks has not been extensively explored. To fill this gap, we propose a joint model, namely JPIS, designed to enhance profile-based intent detection and slot filling. JPIS incorporates the supporting profile information into its encoder and introduces a slot-to-intent attention mechanism to transfer slot information representations to intent detection. Experimental results show that our JPIS substantially outperforms previous profile-based models, establishing a new state-of-the-art performance in overall accuracy on the Chinese benchmark dataset ProSLU.

{{</citation>}}


### (79/169) Labels Need Prompts Too: Mask Matching for Natural Language Understanding Tasks (Bo Li et al., 2023)

{{<citation>}}

Bo Li, Wei Ye, Quansen Wang, Wen Zhao, Shikun Zhang. (2023)  
**Labels Need Prompts Too: Mask Matching for Natural Language Understanding Tasks**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: NLU, Natural Language Understanding  
[Paper Link](http://arxiv.org/abs/2312.08726v2)  

---


**ABSTRACT**  
Textual label names (descriptions) are typically semantically rich in many natural language understanding (NLU) tasks. In this paper, we incorporate the prompting methodology, which is widely used to enrich model input, into the label side for the first time. Specifically, we propose a Mask Matching method, which equips an input with a prompt and its label with another, and then makes predictions by matching their mask representations. We evaluate our method extensively on 8 NLU tasks with 14 datasets. The experimental results show that Mask Matching significantly outperforms its counterparts of fine-tuning and conventional prompt-tuning, setting up state-of-the-art performances in several datasets. Mask Matching is particularly good at handling NLU tasks with large label counts and informative label names. As pioneering efforts that investigate the label-side prompt, we also discuss open issues for future study.

{{</citation>}}


### (80/169) TigerBot: An Open Multilingual Multitask LLM (Ye Chen et al., 2023)

{{<citation>}}

Ye Chen, Wei Cai, Liangmin Wu, Xiaowei Li, Zhanxuan Xin, Cong Fu. (2023)  
**TigerBot: An Open Multilingual Multitask LLM**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BLOOM, Multilingual  
[Paper Link](http://arxiv.org/abs/2312.08688v2)  

---


**ABSTRACT**  
We release and introduce the TigerBot family of large language models (LLMs), consisting of base and chat models, sized from 7, 13, 70 and 180 billion parameters. We develop our models embarking from Llama-2 and BLOOM, and push the boundary further in data, training algorithm, infrastructure, and application tools. Our models yield meaningful performance gain over SOTA open-source models, e.g., Llama-2, specifically 6% gain in English and 20% gain in Chinese. TigerBot model family also achieves leading performance in major academic and industrial benchmarks and leaderboards. We believe that TigerBot represents just a snapshot of lightning-fast progression in LLM open-source community. Therefore, we are thrilled to give back by publicly releasing our models and reporting our approach behind, with additional emphases on building SOTA LLMs in a democratized way and making LLMs of use in real-world applications.

{{</citation>}}


### (81/169) Metacognition-Enhanced Few-Shot Prompting With Positive Reinforcement (Yu Ji et al., 2023)

{{<citation>}}

Yu Ji, Wen Wu, Yi Hu, Hong Zheng, Liang He. (2023)  
**Metacognition-Enhanced Few-Shot Prompting With Positive Reinforcement**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2312.08642v1)  

---


**ABSTRACT**  
Few-shot prompting elicits the remarkable abilities of large language models by equipping them with a few demonstration examples in the input. However, the traditional method of providing large language models with all demonstration input-output pairs at once may not effectively guide large language models to learn the specific input-output mapping relationship. In this paper, inspired by the regulatory and supportive role of metacognition in students' learning, we propose a novel metacognition-enhanced few-shot prompting, which guides large language models to reflect on their thought processes to comprehensively learn the given demonstration examples. Furthermore, considering that positive reinforcement can improve students' learning motivation, we introduce positive reinforcement into our metacognition-enhanced few-shot prompting to promote the few-shot learning of large language models by providing response-based positive feedback. The experimental results on two real-world datasets show that our metacognition-enhanced few-shot prompting with positive reinforcement surpasses traditional few-shot prompting in classification accuracy and macro F1.

{{</citation>}}


### (82/169) Zebra: Extending Context Window with Layerwise Grouped Local-Global Attention (Kaiqiang Song et al., 2023)

{{<citation>}}

Kaiqiang Song, Xiaoyang Wang, Sangwoo Cho, Xiaoman Pan, Dong Yu. (2023)  
**Zebra: Extending Context Window with Layerwise Grouped Local-Global Attention**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Attention, Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2312.08618v1)  

---


**ABSTRACT**  
This paper introduces a novel approach to enhance the capabilities of Large Language Models (LLMs) in processing and understanding extensive text sequences, a critical aspect in applications requiring deep comprehension and synthesis of large volumes of information. Recognizing the inherent challenges in extending the context window for LLMs, primarily built on Transformer architecture, we propose a new model architecture, referred to as Zebra. This architecture efficiently manages the quadratic time and memory complexity issues associated with full attention in the Transformer by employing grouped local-global attention layers. Our model, akin to a zebra's alternating stripes, balances local and global attention layers, significantly reducing computational requirements and memory consumption. Comprehensive experiments, including pretraining from scratch, continuation of long context adaptation training, and long instruction tuning, are conducted to evaluate the Zebra's performance. The results show that Zebra achieves comparable or superior performance on both short and long sequence benchmarks, while also enhancing training and inference efficiency.

{{</citation>}}


### (83/169) Unraveling Key Factors of Knowledge Distillation (Jingxuan Wei et al., 2023)

{{<citation>}}

Jingxuan Wei, Linzhuang Sun, Xu Tan, Bihui Yu, Ruifeng Guo. (2023)  
**Unraveling Key Factors of Knowledge Distillation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Knowledge Distillation, Machine Translation  
[Paper Link](http://arxiv.org/abs/2312.08585v1)  

---


**ABSTRACT**  
Knowledge distillation, a technique for model compression and performance enhancement, has gained significant traction in Neural Machine Translation (NMT). However, existing research primarily focuses on empirical applications, and there is a lack of comprehensive understanding of how student model capacity, data complexity, and decoding strategies collectively influence distillation effectiveness. Addressing this gap, our study conducts an in-depth investigation into these factors, particularly focusing on their interplay in word-level and sequence-level distillation within NMT. Through extensive experimentation across datasets like IWSLT13 En$\rightarrow$Fr, IWSLT14 En$\rightarrow$De, and others, we empirically validate hypotheses related to the impact of these factors on knowledge distillation. Our research not only elucidates the significant influence of model capacity, data complexity, and decoding strategies on distillation effectiveness but also introduces a novel, optimized distillation approach. This approach, when applied to the IWSLT14 de$\rightarrow$en translation task, achieves state-of-the-art performance, demonstrating its practical efficacy in advancing the field of NMT.

{{</citation>}}


### (84/169) ZeroQuant(4+2): Redefining LLMs Quantization with a New FP6-Centric Strategy for Diverse Generative Tasks (Xiaoxia Wu et al., 2023)

{{<citation>}}

Xiaoxia Wu, Haojun Xia, Stephen Youn, Zhen Zheng, Shiyang Chen, Arash Bakhtiari, Michael Wyatt, Reza Yazdani Aminabadi, Yuxiong He, Olatunji Ruwase, Leon Song, Zhewei Yao. (2023)  
**ZeroQuant(4+2): Redefining LLMs Quantization with a New FP6-Centric Strategy for Diverse Generative Tasks**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL, stat-ML  
Keywords: AI, GPT, Quantization, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2312.08583v2)  

---


**ABSTRACT**  
This study examines 4-bit quantization methods like GPTQ in large language models (LLMs), highlighting GPTQ's overfitting and limited enhancement in Zero-Shot tasks. While prior works merely focusing on zero-shot measurement, we extend task scope to more generative categories such as code generation and abstractive summarization, in which we found that INT4 quantization can significantly underperform. However, simply shifting to higher precision formats like FP6 has been particularly challenging, thus overlooked, due to poor performance caused by the lack of sophisticated integration and system acceleration strategies on current AI hardware. Our results show that FP6, even with a coarse-grain quantization scheme, performs robustly across various algorithms and tasks, demonstrating its superiority in accuracy and versatility. Notably, with the FP6 quantization, \codestar-15B model performs comparably to its FP16 counterpart in code generation, and for smaller models like the 406M it closely matches their baselines in summarization. Neither can be achieved by INT4. To better accommodate various AI hardware and achieve the best system performance, we propose a novel 4+2 design for FP6 to achieve similar latency to the state-of-the-art INT4 fine-grain quantization. With our design, FP6 can become a promising solution to the current 4-bit quantization methods used in LLMs.

{{</citation>}}


### (85/169) Identifying Planetary Names in Astronomy Papers: A Multi-Step Approach (Golnaz Shapurian et al., 2023)

{{<citation>}}

Golnaz Shapurian, Michael J Kurtz, Alberto Accomazzi. (2023)  
**Identifying Planetary Names in Astronomy Papers: A Multi-Step Approach**  

---
Primary Category: cs.CL  
Categories: astro-ph-IM, cs-CL, cs-LG, cs.CL  
Keywords: NER  
[Paper Link](http://arxiv.org/abs/2312.08579v2)  

---


**ABSTRACT**  
The automatic identification of planetary feature names in astronomy publications presents numerous challenges. These features include craters, defined as roughly circular depressions resulting from impact or volcanic activity; dorsas, which are elongate raised structures or wrinkle ridges; and lacus, small irregular patches of dark, smooth material on the Moon, referred to as "lake" (Planetary Names Working Group, n.d.). Many feature names overlap with places or people's names that they are named after, for example, Syria, Tempe, Einstein, and Sagan, to name a few (U.S. Geological Survey, n.d.). Some feature names have been used in many contexts, for instance, Apollo, which can refer to mission, program, sample, astronaut, seismic, seismometers, core, era, data, collection, instrument, and station, in addition to the crater on the Moon. Some feature names can appear in the text as adjectives, like the lunar craters Black, Green, and White. Some feature names in other contexts serve as directions, like craters West and South on the Moon. Additionally, some features share identical names across different celestial bodies, requiring disambiguation, such as the Adams crater, which exists on both the Moon and Mars. We present a multi-step pipeline combining rule-based filtering, statistical relevance analysis, part-of-speech (POS) tagging, named entity recognition (NER) model, hybrid keyword harvesting, knowledge graph (KG) matching, and inference with a locally installed large language model (LLM) to reliably identify planetary names despite these challenges. When evaluated on a dataset of astronomy papers from the Astrophysics Data System (ADS), this methodology achieves an F1-score over 0.97 in disambiguating planetary feature names.

{{</citation>}}


## cs.CR (4)



### (86/169) Security layers and related services within the Horizon Europe NEUROPULS project (Fabio Pavanello et al., 2023)

{{<citation>}}

Fabio Pavanello, Cedric Marchand, Paul Jimenez, Xavier Letartre, Ricardo Chaves, Niccolò Marastoni, Alberto Lovato, Mariano Ceccato, George Papadimitriou, Vasileios Karakostas, Dimitris Gizopoulos, Roberta Bardini, Tzamn Melendez Carmona, Stefano Di Carlo, Alessandro Savino, Laurence Lerch, Ulrich Ruhrmair, Sergio Vinagrero Gutierrez, Giorgio Di Natale, Elena Ioana Vatajelu. (2023)  
**Security layers and related services within the Horizon Europe NEUROPULS project**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR, eess-SP, physics-optics  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2312.09383v1)  

---


**ABSTRACT**  
In the contemporary security landscape, the incorporation of photonics has emerged as a transformative force, unlocking a spectrum of possibilities to enhance the resilience and effectiveness of security primitives. This integration represents more than a mere technological augmentation; it signifies a paradigm shift towards innovative approaches capable of delivering security primitives with key properties for low-power systems. This not only augments the robustness of security frameworks, but also paves the way for novel strategies that adapt to the evolving challenges of the digital age. This paper discusses the security layers and related services that will be developed, modeled, and evaluated within the Horizon Europe NEUROPULS project. These layers will exploit novel implementations for security primitives based on physical unclonable functions (PUFs) using integrated photonics technology. Their objective is to provide a series of services to support the secure operation of a neuromorphic photonic accelerator for edge computing applications.

{{</citation>}}


### (87/169) DECLASSIFLOW: A Static Analysis for Modeling Non-Speculative Knowledge to Relax Speculative Execution Security Measures (Full Version) (Rutvik Choudhary et al., 2023)

{{<citation>}}

Rutvik Choudhary, Alan Wang, Zirui Neil Zhao, Adam Morrison, Christopher W. Fletcher. (2023)  
**DECLASSIFLOW: A Static Analysis for Modeling Non-Speculative Knowledge to Relax Speculative Execution Security Measures (Full Version)**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2312.09336v1)  

---


**ABSTRACT**  
Speculative execution attacks undermine the security of constant-time programming, the standard technique used to prevent microarchitectural side channels in security-sensitive software such as cryptographic code. Constant-time code must therefore also deploy a defense against speculative execution attacks to prevent leakage of secret data stored in memory or the processor registers. Unfortunately, contemporary defenses, such as speculative load hardening (SLH), can only satisfy this strong security guarantee at a very high performance cost.   This paper proposes DECLASSIFLOW, a static program analysis and protection framework to efficiently protect constant-time code from speculative leakage. DECLASSIFLOW models "attacker knowledge" -- data which is inherently transmitted (or, implicitly declassified) by the code's non-speculative execution -- and statically removes protection on such data from points in the program where it is already guaranteed to leak non-speculatively. Overall, DECLASSIFLOW ensures that data which never leaks during the non-speculative execution does not leak during speculative execution, but with lower overhead than conservative protections like SLH.

{{</citation>}}


### (88/169) On the Difficulty of Defending Contrastive Learning against Backdoor Attacks (Changjiang Li et al., 2023)

{{<citation>}}

Changjiang Li, Ren Pang, Bochuan Cao, Zhaohan Xi, Jinghui Chen, Shouling Ji, Ting Wang. (2023)  
**On the Difficulty of Defending Contrastive Learning against Backdoor Attacks**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs-CV, cs.CR  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2312.09057v1)  

---


**ABSTRACT**  
Recent studies have shown that contrastive learning, like supervised learning, is highly vulnerable to backdoor attacks wherein malicious functions are injected into target models, only to be activated by specific triggers. However, thus far it remains under-explored how contrastive backdoor attacks fundamentally differ from their supervised counterparts, which impedes the development of effective defenses against the emerging threat.   This work represents a solid step toward answering this critical question. Specifically, we define TRL, a unified framework that encompasses both supervised and contrastive backdoor attacks. Through the lens of TRL, we uncover that the two types of attacks operate through distinctive mechanisms: in supervised attacks, the learning of benign and backdoor tasks tends to occur independently, while in contrastive attacks, the two tasks are deeply intertwined both in their representations and throughout their learning processes. This distinction leads to the disparate learning dynamics and feature distributions of supervised and contrastive attacks. More importantly, we reveal that the specificities of contrastive backdoor attacks entail important implications from a defense perspective: existing defenses for supervised attacks are often inadequate and not easily retrofitted to contrastive attacks. We also explore several alternative defenses and discuss their potential challenges. Our findings highlight the need for defenses tailored to the specificities of contrastive backdoor attacks, pointing to promising directions for future research.

{{</citation>}}


### (89/169) Google Tag Manager: Hidden Data Leaks and its Potential Violations under EU Data Protection Law (Gilles Mertens et al., 2023)

{{<citation>}}

Gilles Mertens, Nataliia Bielova, Vincent Roca, Cristiana Santos, Michael Toth. (2023)  
**Google Tag Manager: Hidden Data Leaks and its Potential Violations under EU Data Protection Law**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2312.08806v1)  

---


**ABSTRACT**  
Tag Management Systems were developed in order to support website publishers in installing multiple third-party JavaScript scripts (Tags) on their websites. In 2012, Google developed its own TMS called "Google Tag Manager" (GTM) that is currently present on 28 million live websites. In 2020, a new "Server-side" GTM was introduced, allowing publishers to include Tags directly on the server. However, neither version of GTM has yet been thoroughly evaluated by the academic research community. In this work, we study, for the first time, the two versions of the Google Tag Management (GTM) architectures: Client- and Server-side GTM. By analyzing these systems with 78 Client-side Tags, 8 Server-side Tags and two Consent Management Platforms (CMPs) from the inside, we discover multiple hidden data leaks, Tags bypassing GTM permission system to inject scripts, and consent enabled by default. With a legal expert, we perform an in-depth legal analysis of GTM and its actors to identify potential legal violations and their liabilities. We provide recommendations and propose numerous improvements for GTM to facilitate legal compliance.

{{</citation>}}


## eess.SY (1)



### (90/169) Measurement-based/Model-less Estimation of Voltage Sensitivity Coefficients by Feedforward and LSTM Neural Networks in Power Distribution Grids (Robin Henry et al., 2023)

{{<citation>}}

Robin Henry, Rahul Gupta. (2023)  
**Measurement-based/Model-less Estimation of Voltage Sensitivity Coefficients by Feedforward and LSTM Neural Networks in Power Distribution Grids**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2312.09377v1)  

---


**ABSTRACT**  
The increasing adoption of measurement units in electrical power distribution grids has enabled the deployment of data-driven and measurement-based control schemes. Such schemes rely on measurement-based estimated models, where the models are first estimated using raw measurements and then used in the control problem. This work focuses on measurement-based estimation of the voltage sensitivity coefficients which can be used for voltage control. In the existing literature, these coefficients are estimated using regression-based methods, which do not perform well in the case of high measurement noise. This work proposes tackling this problem by using neural network (NN)-based estimation of the voltage sensitivity coefficients which is robust against measurement noise. In particular, we propose using Feedforward and Long-Short Term Memory (LSTM) neural networks. The trained NNs take measurements of nodal voltage magnitudes and active and reactive powers and output the vector of voltage magnitude sensitivity coefficients. The performance of the proposed scheme is compared against the regression-based method for a CIGRE benchmark network.

{{</citation>}}


## cs.CV (49)



### (91/169) Text-Guided Face Recognition using Multi-Granularity Cross-Modal Contrastive Learning (Md Mahedi Hasan et al., 2023)

{{<citation>}}

Md Mahedi Hasan, Shoaib Meraj Sami, Nasser Nasrabadi. (2023)  
**Text-Guided Face Recognition using Multi-Granularity Cross-Modal Contrastive Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2312.09367v1)  

---


**ABSTRACT**  
State-of-the-art face recognition (FR) models often experience a significant performance drop when dealing with facial images in surveillance scenarios where images are in low quality and often corrupted with noise. Leveraging facial characteristics, such as freckles, scars, gender, and ethnicity, becomes highly beneficial in improving FR performance in such scenarios. In this paper, we introduce text-guided face recognition (TGFR) to analyze the impact of integrating facial attributes in the form of natural language descriptions. We hypothesize that adding semantic information into the loop can significantly improve the image understanding capability of an FR algorithm compared to other soft biometrics. However, learning a discriminative joint embedding within the multimodal space poses a considerable challenge due to the semantic gap in the unaligned image-text representations, along with the complexities arising from ambiguous and incoherent textual descriptions of the face. To address these challenges, we introduce a face-caption alignment module (FCAM), which incorporates cross-modal contrastive losses across multiple granularities to maximize the mutual information between local and global features of the face-caption pair. Within FCAM, we refine both facial and textual features for learning aligned and discriminative features. We also design a face-caption fusion module (FCFM) that applies fine-grained interactions and coarse-grained associations among cross-modal features. Through extensive experiments conducted on three face-caption datasets, proposed TGFR demonstrates remarkable improvements, particularly on low-quality images, over existing FR models and outperforms other related methods and benchmarks.

{{</citation>}}


### (92/169) The Expert Knowledge combined with AI outperforms AI Alone in Seizure Onset Zone Localization using resting state fMRI (Payal Kamboj et al., 2023)

{{<citation>}}

Payal Kamboj, Ayan Banerjee, Varina L. Boerwinkle, Sandeep K. S. Gupta. (2023)  
**The Expert Knowledge combined with AI outperforms AI Alone in Seizure Onset Zone Localization using resting state fMRI**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.09360v1)  

---


**ABSTRACT**  
We evaluated whether integration of expert guidance on seizure onset zone (SOZ) identification from resting state functional MRI (rs-fMRI) connectomics combined with deep learning (DL) techniques enhances the SOZ delineation in patients with refractory epilepsy (RE), compared to utilizing DL alone. Rs-fMRI were collected from 52 children with RE who had subsequently undergone ic-EEG and then, if indicated, surgery for seizure control (n = 25). The resting state functional connectomics data were previously independently classified by two expert epileptologists, as indicative of measurement noise, typical resting state network connectivity, or SOZ. An expert knowledge integrated deep network was trained on functional connectomics data to identify SOZ. Expert knowledge integrated with DL showed a SOZ localization accuracy of 84.8& and F1 score, harmonic mean of positive predictive value and sensitivity, of 91.7%. Conversely, a DL only model yielded an accuracy of less than 50% (F1 score 63%). Activations that initiate in gray matter, extend through white matter and end in vascular regions are seen as the most discriminative expert identified SOZ characteristics. Integration of expert knowledge of functional connectomics can not only enhance the performance of DL in localizing SOZ in RE, but also lead toward potentially useful explanations of prevalent co-activation patterns in SOZ. RE with surgical outcomes and pre-operative rs-fMRI studies can yield expert knowledge most salient for SOZ identification.

{{</citation>}}


### (93/169) Promptable Behaviors: Personalizing Multi-Objective Rewards from Human Preferences (Minyoung Hwang et al., 2023)

{{<citation>}}

Minyoung Hwang, Luca Weihs, Chanwoo Park, Kimin Lee, Aniruddha Kembhavi, Kiana Ehsani. (2023)  
**Promptable Behaviors: Personalizing Multi-Objective Rewards from Human Preferences**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-RO, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.09337v1)  

---


**ABSTRACT**  
Customizing robotic behaviors to be aligned with diverse human preferences is an underexplored challenge in the field of embodied AI. In this paper, we present Promptable Behaviors, a novel framework that facilitates efficient personalization of robotic agents to diverse human preferences in complex environments. We use multi-objective reinforcement learning to train a single policy adaptable to a broad spectrum of preferences. We introduce three distinct methods to infer human preferences by leveraging different types of interactions: (1) human demonstrations, (2) preference feedback on trajectory comparisons, and (3) language instructions. We evaluate the proposed method in personalized object-goal navigation and flee navigation tasks in ProcTHOR and RoboTHOR, demonstrating the ability to prompt agent behaviors to satisfy human preferences in various scenarios. Project page: https://promptable-behaviors.github.io

{{</citation>}}


### (94/169) LIME: Localized Image Editing via Attention Regularization in Diffusion Models (Enis Simsar et al., 2023)

{{<citation>}}

Enis Simsar, Alessio Tonioni, Yongqin Xian, Thomas Hofmann, Federico Tombari. (2023)  
**LIME: Localized Image Editing via Attention Regularization in Diffusion Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.09256v1)  

---


**ABSTRACT**  
Diffusion models (DMs) have gained prominence due to their ability to generate high-quality, varied images, with recent advancements in text-to-image generation. The research focus is now shifting towards the controllability of DMs. A significant challenge within this domain is localized editing, where specific areas of an image are modified without affecting the rest of the content. This paper introduces LIME for localized image editing in diffusion models that do not require user-specified regions of interest (RoI) or additional text input. Our method employs features from pre-trained methods and a simple clustering technique to obtain precise semantic segmentation maps. Then, by leveraging cross-attention maps, it refines these segments for localized edits. Finally, we propose a novel cross-attention regularization technique that penalizes unrelated cross-attention scores in the RoI during the denoising steps, ensuring localized edits. Our approach, without re-training and fine-tuning, consistently improves the performance of existing methods in various editing benchmarks.

{{</citation>}}


### (95/169) VL-GPT: A Generative Pre-trained Transformer for Vision and Language Understanding and Generation (Jinguo Zhu et al., 2023)

{{<citation>}}

Jinguo Zhu, Xiaohan Ding, Yixiao Ge, Yuying Ge, Sijie Zhao, Hengshuang Zhao, Xiaohua Wang, Ying Shan. (2023)  
**VL-GPT: A Generative Pre-trained Transformer for Vision and Language Understanding and Generation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT, Transformer  
[Paper Link](http://arxiv.org/abs/2312.09251v1)  

---


**ABSTRACT**  
In this work, we introduce Vision-Language Generative Pre-trained Transformer (VL-GPT), a transformer model proficient at concurrently perceiving and generating visual and linguistic data. VL-GPT achieves a unified pre-training approach for both image and text modalities by employing a straightforward auto-regressive objective, thereby enabling the model to process image and text as seamlessly as a language model processes text. To accomplish this, we initially propose a novel image tokenizer-detokenizer framework for visual data, specifically designed to transform raw images into a sequence of continuous embeddings and reconstruct them accordingly. In combination with the existing text tokenizer and detokenizer, this framework allows for the encoding of interleaved image-text data into a multimodal sequence, which can subsequently be fed into the transformer model. Consequently, VL-GPT can perform large-scale pre-training on multimodal corpora utilizing a unified auto-regressive objective (i.e., next-token prediction). Upon completion of pre-training, VL-GPT exhibits remarkable zero-shot and few-shot performance across a diverse range of vision and language understanding and generation tasks, including image captioning, visual question answering, text-to-image generation, and more. Additionally, the pre-trained model retrains in-context learning capabilities when provided with multimodal prompts. We further conduct instruction tuning on our VL-GPT, highlighting its exceptional potential for multimodal assistance. The source code and model weights shall be released.

{{</citation>}}


### (96/169) DriveMLM: Aligning Multi-Modal Large Language Models with Behavioral Planning States for Autonomous Driving (Wenhai Wang et al., 2023)

{{<citation>}}

Wenhai Wang, Jiangwei Xie, ChuanYang Hu, Haoming Zou, Jianan Fan, Wenwen Tong, Yang Wen, Silei Wu, Hanming Deng, Zhiqi Li, Hao Tian, Lewei Lu, Xizhou Zhu, Xiaogang Wang, Yu Qiao, Jifeng Dai. (2023)  
**DriveMLM: Aligning Multi-Modal Large Language Models with Behavioral Planning States for Autonomous Driving**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.09245v1)  

---


**ABSTRACT**  
Large language models (LLMs) have opened up new possibilities for intelligent agents, endowing them with human-like thinking and cognitive abilities. In this work, we delve into the potential of large language models (LLMs) in autonomous driving (AD). We introduce DriveMLM, an LLM-based AD framework that can perform close-loop autonomous driving in realistic simulators. To this end, (1) we bridge the gap between the language decisions and the vehicle control commands by standardizing the decision states according to the off-the-shelf motion planning module. (2) We employ a multi-modal LLM (MLLM) to model the behavior planning module of a module AD system, which uses driving rules, user commands, and inputs from various sensors (e.g., camera, lidar) as input and makes driving decisions and provide explanations; This model can plug-and-play in existing AD systems such as Apollo for close-loop driving. (3) We design an effective data engine to collect a dataset that includes decision state and corresponding explanation annotation for model training and evaluation. We conduct extensive experiments and show that our model achieves 76.1 driving score on the CARLA Town05 Long, and surpasses the Apollo baseline by 4.7 points under the same settings, demonstrating the effectiveness of our model. We hope this work can serve as a baseline for autonomous driving with LLMs. Code and models shall be released at https://github.com/OpenGVLab/DriveMLM.

{{</citation>}}


### (97/169) OccNeRF: Self-Supervised Multi-Camera Occupancy Prediction with Neural Radiance Fields (Chubin Zhang et al., 2023)

{{<citation>}}

Chubin Zhang, Juncheng Yan, Yi Wei, Jiaxin Li, Li Liu, Yansong Tang, Yueqi Duan, Jiwen Lu. (2023)  
**OccNeRF: Self-Supervised Multi-Camera Occupancy Prediction with Neural Radiance Fields**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2312.09243v1)  

---


**ABSTRACT**  
As a fundamental task of vision-based perception, 3D occupancy prediction reconstructs 3D structures of surrounding environments. It provides detailed information for autonomous driving planning and navigation. However, most existing methods heavily rely on the LiDAR point clouds to generate occupancy ground truth, which is not available in the vision-based system. In this paper, we propose an OccNeRF method for self-supervised multi-camera occupancy prediction. Different from bounded 3D occupancy labels, we need to consider unbounded scenes with raw image supervision. To solve the issue, we parameterize the reconstructed occupancy fields and reorganize the sampling strategy. The neural rendering is adopted to convert occupancy fields to multi-camera depth maps, supervised by multi-frame photometric consistency. Moreover, for semantic occupancy prediction, we design several strategies to polish the prompts and filter the outputs of a pretrained open-vocabulary 2D segmentation model. Extensive experiments for both self-supervised depth estimation and semantic occupancy prediction tasks on nuScenes dataset demonstrate the effectiveness of our method.

{{</citation>}}


### (98/169) Pixel Aligned Language Models (Jiarui Xu et al., 2023)

{{<citation>}}

Jiarui Xu, Xingyi Zhou, Shen Yan, Xiuye Gu, Anurag Arnab, Chen Sun, Xiaolong Wang, Cordelia Schmid. (2023)  
**Pixel Aligned Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.09237v1)  

---


**ABSTRACT**  
Large language models have achieved great success in recent years, so as their variants in vision. Existing vision-language models can describe images in natural languages, answer visual-related questions, or perform complex reasoning about the image. However, it is yet unclear how localization tasks, such as word grounding or referring localization, can be performed using large language models. In this work, we aim to develop a vision-language model that can take locations, for example, a set of points or boxes, as either inputs or outputs. When taking locations as inputs, the model performs location-conditioned captioning, which generates captions for the indicated object or region. When generating locations as outputs, our model regresses pixel coordinates for each output word generated by the language model, and thus performs dense word grounding. Our model is pre-trained on the Localized Narrative dataset, which contains pixel-word-aligned captioning from human attention. We show our model can be applied to various location-aware vision-language tasks, including referring localization, location-conditioned captioning, and dense object captioning, archiving state-of-the-art performance on RefCOCO and Visual Genome. Project page: https://jerryxu.net/PixelLLM .

{{</citation>}}


### (99/169) DVQI: A Multi-task, Hardware-integrated Artificial Intelligence System for Automated Visual Inspection in Electronics Manufacturing (Audrey Chung et al., 2023)

{{<citation>}}

Audrey Chung, Francis Li, Jeremy Ward, Andrew Hryniowski, Alexander Wong. (2023)  
**DVQI: A Multi-task, Hardware-integrated Artificial Intelligence System for Automated Visual Inspection in Electronics Manufacturing**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.09232v1)  

---


**ABSTRACT**  
As electronics manufacturers continue to face pressure to increase production efficiency amid difficulties with supply chains and labour shortages, many printed circuit board assembly (PCBA) manufacturers have begun to invest in automation and technological innovations to remain competitive. One such method is to leverage artificial intelligence (AI) to greatly augment existing manufacturing processes. In this paper, we present the DarwinAI Visual Quality Inspection (DVQI) system, a hardware-integration artificial intelligence system for the automated inspection of printed circuit board assembly defects in an electronics manufacturing environment. The DVQI system enables multi-task inspection via minimal programming and setup for manufacturing engineers while improving cycle time relative to manual inspection. We also present a case study of the deployed DVQI system's performance and impact for a top electronics manufacturer.

{{</citation>}}


### (100/169) Reliability in Semantic Segmentation: Can We Use Synthetic Data? (Thibaut Loiseau et al., 2023)

{{<citation>}}

Thibaut Loiseau, Tuan-Hung Vu, Mickael Chen, Patrick Pérez, Matthieu Cord. (2023)  
**Reliability in Semantic Segmentation: Can We Use Synthetic Data?**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2312.09231v1)  

---


**ABSTRACT**  
Assessing the reliability of perception models to covariate shifts and out-of-distribution (OOD) detection is crucial for safety-critical applications such as autonomous vehicles. By nature of the task, however, the relevant data is difficult to collect and annotate. In this paper, we challenge cutting-edge generative models to automatically synthesize data for assessing reliability in semantic segmentation. By fine-tuning Stable Diffusion, we perform zero-shot generation of synthetic data in OOD domains or inpainted with OOD objects. Synthetic data is employed to provide an initial assessment of pretrained segmenters, thereby offering insights into their performance when confronted with real edge cases. Through extensive experiments, we demonstrate a high correlation between the performance on synthetic data and the performance on real OOD data, showing the validity approach. Furthermore, we illustrate how synthetic data can be utilized to enhance the calibration and OOD detection capabilities of segmenters.

{{</citation>}}


### (101/169) Mosaic-SDF for 3D Generative Models (Lior Yariv et al., 2023)

{{<citation>}}

Lior Yariv, Omri Puny, Natalia Neverova, Oran Gafni, Yaron Lipman. (2023)  
**Mosaic-SDF for 3D Generative Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-GR, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.09222v1)  

---


**ABSTRACT**  
Current diffusion or flow-based generative models for 3D shapes divide to two: distilling pre-trained 2D image diffusion models, and training directly on 3D shapes. When training a diffusion or flow models on 3D shapes a crucial design choice is the shape representation. An effective shape representation needs to adhere three design principles: it should allow an efficient conversion of large 3D datasets to the representation form; it should provide a good tradeoff of approximation power versus number of parameters; and it should have a simple tensorial form that is compatible with existing powerful neural architectures. While standard 3D shape representations such as volumetric grids and point clouds do not adhere to all these principles simultaneously, we advocate in this paper a new representation that does. We introduce Mosaic-SDF (M-SDF): a simple 3D shape representation that approximates the Signed Distance Function (SDF) of a given shape by using a set of local grids spread near the shape's boundary. The M-SDF representation is fast to compute for each shape individually making it readily parallelizable; it is parameter efficient as it only covers the space around the shape's boundary; and it has a simple matrix form, compatible with Transformer-based architectures. We demonstrate the efficacy of the M-SDF representation by using it to train a 3D generative flow model including class-conditioned generation with the 3D Warehouse dataset, and text-to-3D generation using a dataset of about 600k caption-shape pairs.

{{</citation>}}


### (102/169) General Object Foundation Model for Images and Videos at Scale (Junfeng Wu et al., 2023)

{{<citation>}}

Junfeng Wu, Yi Jiang, Qihao Liu, Zehuan Yuan, Xiang Bai, Song Bai. (2023)  
**General Object Foundation Model for Images and Videos at Scale**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.09158v1)  

---


**ABSTRACT**  
We present GLEE in this work, an object-level foundation model for locating and identifying objects in images and videos. Through a unified framework, GLEE accomplishes detection, segmentation, tracking, grounding, and identification of arbitrary objects in the open world scenario for various object perception tasks. Adopting a cohesive learning strategy, GLEE acquires knowledge from diverse data sources with varying supervision levels to formulate general object representations, excelling in zero-shot transfer to new data and tasks. Specifically, we employ an image encoder, text encoder, and visual prompter to handle multi-modal inputs, enabling to simultaneously solve various object-centric downstream tasks while maintaining state-of-the-art performance. Demonstrated through extensive training on over five million images from diverse benchmarks, GLEE exhibits remarkable versatility and improved generalization performance, efficiently tackling downstream tasks without the need for task-specific adaptation. By integrating large volumes of automatically labeled data, we further enhance its zero-shot generalization capabilities. Additionally, GLEE is capable of being integrated into Large Language Models, serving as a foundational model to provide universal object-level information for multi-modal tasks. We hope that the versatility and universality of our method will mark a significant step in the development of efficient visual foundation models for AGI systems. The model and code will be released at https://glee-vision.github.io .

{{</citation>}}


### (103/169) Triplane Meets Gaussian Splatting: Fast and Generalizable Single-View 3D Reconstruction with Transformers (Zi-Xin Zou et al., 2023)

{{<citation>}}

Zi-Xin Zou, Zhipeng Yu, Yuan-Chen Guo, Yangguang Li, Ding Liang, Yan-Pei Cao, Song-Hai Zhang. (2023)  
**Triplane Meets Gaussian Splatting: Fast and Generalizable Single-View 3D Reconstruction with Transformers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.09147v1)  

---


**ABSTRACT**  
Recent advancements in 3D reconstruction from single images have been driven by the evolution of generative models. Prominent among these are methods based on Score Distillation Sampling (SDS) and the adaptation of diffusion models in the 3D domain. Despite their progress, these techniques often face limitations due to slow optimization or rendering processes, leading to extensive training and optimization times. In this paper, we introduce a novel approach for single-view reconstruction that efficiently generates a 3D model from a single image via feed-forward inference. Our method utilizes two transformer-based networks, namely a point decoder and a triplane decoder, to reconstruct 3D objects using a hybrid Triplane-Gaussian intermediate representation. This hybrid representation strikes a balance, achieving a faster rendering speed compared to implicit representations while simultaneously delivering superior rendering quality than explicit representations. The point decoder is designed for generating point clouds from single images, offering an explicit representation which is then utilized by the triplane decoder to query Gaussian features for each point. This design choice addresses the challenges associated with directly regressing explicit 3D Gaussian attributes characterized by their non-structural nature. Subsequently, the 3D Gaussians are decoded by an MLP to enable rapid rendering through splatting. Both decoders are built upon a scalable, transformer-based architecture and have been efficiently trained on large-scale 3D datasets. The evaluations conducted on both synthetic datasets and real-world images demonstrate that our method not only achieves higher quality but also ensures a faster runtime in comparison to previous state-of-the-art techniques. Please see our project page at https://zouzx.github.io/TriplaneGaussian/.

{{</citation>}}


### (104/169) Class-Wise Buffer Management for Incremental Object Detection: An Effective Buffer Training Strategy (Junsu Kim et al., 2023)

{{<citation>}}

Junsu Kim, Sumin Hong, Chanwoo Kim, Jihyeon Kim, Yihalem Yimolal Tiruneh, Jeongwan On, Jihyun Song, Sunhwa Choi, Seungryul Baek. (2023)  
**Class-Wise Buffer Management for Incremental Object Detection: An Effective Buffer Training Strategy**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2312.09139v1)  

---


**ABSTRACT**  
Class incremental learning aims to solve a problem that arises when continuously adding unseen class instances to an existing model This approach has been extensively studied in the context of image classification; however its applicability to object detection is not well established yet. Existing frameworks using replay methods mainly collect replay data without considering the model being trained and tend to rely on randomness or the number of labels of each sample. Also, despite the effectiveness of the replay, it was not yet optimized for the object detection task. In this paper, we introduce an effective buffer training strategy (eBTS) that creates the optimized replay buffer on object detection. Our approach incorporates guarantee minimum and hierarchical sampling to establish the buffer customized to the trained model. %These methods can facilitate effective retrieval of prior knowledge. Furthermore, we use the circular experience replay training to optimally utilize the accumulated buffer data. Experiments on the MS COCO dataset demonstrate that our eBTS achieves state-of-the-art performance compared to the existing replay schemes.

{{</citation>}}


### (105/169) Agent Attention: On the Integration of Softmax and Linear Attention (Dongchen Han et al., 2023)

{{<citation>}}

Dongchen Han, Tianzhu Ye, Yizeng Han, Zhuofan Xia, Shiji Song, Gao Huang. (2023)  
**Agent Attention: On the Integration of Softmax and Linear Attention**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.08874v1)  

---


**ABSTRACT**  
The attention module is the key component in Transformers. While the global attention mechanism offers high expressiveness, its excessive computational cost restricts its applicability in various scenarios. In this paper, we propose a novel attention paradigm, Agent Attention, to strike a favorable balance between computational efficiency and representation power. Specifically, the Agent Attention, denoted as a quadruple $(Q, A, K, V)$, introduces an additional set of agent tokens $A$ into the conventional attention module. The agent tokens first act as the agent for the query tokens $Q$ to aggregate information from $K$ and $V$, and then broadcast the information back to $Q$. Given the number of agent tokens can be designed to be much smaller than the number of query tokens, the agent attention is significantly more efficient than the widely adopted Softmax attention, while preserving global context modelling capability. Interestingly, we show that the proposed agent attention is equivalent to a generalized form of linear attention. Therefore, agent attention seamlessly integrates the powerful Softmax attention and the highly efficient linear attention. Extensive experiments demonstrate the effectiveness of agent attention with various vision Transformers and across diverse vision tasks, including image classification, object detection, semantic segmentation and image generation. Notably, agent attention has shown remarkable performance in high-resolution scenarios, owning to its linear attention nature. For instance, when applied to Stable Diffusion, our agent attention accelerates generation and substantially enhances image generation quality without any additional training. Code is available at https://github.com/LeapLabTHU/Agent-Attention.

{{</citation>}}


### (106/169) Learned Fusion: 3D Object Detection using Calibration-Free Transformer Feature Fusion (Michael Fürst et al., 2023)

{{<citation>}}

Michael Fürst, Rahul Jakkamsetty, René Schuster, Didier Stricker. (2023)  
**Learned Fusion: 3D Object Detection using Calibration-Free Transformer Feature Fusion**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection, Transformer  
[Paper Link](http://arxiv.org/abs/2312.09082v1)  

---


**ABSTRACT**  
The state of the art in 3D object detection using sensor fusion heavily relies on calibration quality, which is difficult to maintain in large scale deployment outside a lab environment. We present the first calibration-free approach for 3D object detection. Thus, eliminating the need for complex and costly calibration procedures. Our approach uses transformers to map the features between multiple views of different sensors at multiple abstraction levels. In an extensive evaluation for object detection, we not only show that our approach outperforms single modal setups by 14.1% in BEV mAP, but also that the transformer indeed learns mapping. By showing calibration is not necessary for sensor fusion, we hope to motivate other researchers following the direction of calibration-free fusion. Additionally, resulting approaches have a substantial resilience against rotation and translation changes.

{{</citation>}}


### (107/169) Holodeck: Language Guided Generation of 3D Embodied AI Environments (Yue Yang et al., 2023)

{{<citation>}}

Yue Yang, Fan-Yun Sun, Luca Weihs, Eli VanderBilt, Alvaro Herrasti, Winson Han, Jiajun Wu, Nick Haber, Ranjay Krishna, Lingjie Liu, Chris Callison-Burch, Mark Yatskar, Aniruddha Kembhavi, Christopher Clark. (2023)  
**Holodeck: Language Guided Generation of 3D Embodied AI Environments**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs-RO, cs.CV  
Keywords: AI, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2312.09067v1)  

---


**ABSTRACT**  
3D simulated environments play a critical role in Embodied AI, but their creation requires expertise and extensive manual effort, restricting their diversity and scope. To mitigate this limitation, we present Holodeck, a system that generates 3D environments to match a user-supplied prompt fully automatedly. Holodeck can generate diverse scenes, e.g., arcades, spas, and museums, adjust the designs for styles, and can capture the semantics of complex queries such as "apartment for a researcher with a cat" and "office of a professor who is a fan of Star Wars". Holodeck leverages a large language model (GPT-4) for common sense knowledge about what the scene might look like and uses a large collection of 3D assets from Objaverse to populate the scene with diverse objects. To address the challenge of positioning objects correctly, we prompt GPT-4 to generate spatial relational constraints between objects and then optimize the layout to satisfy those constraints. Our large-scale human evaluation shows that annotators prefer Holodeck over manually designed procedural baselines in residential scenes and that Holodeck can produce high-quality outputs for diverse scene types. We also demonstrate an exciting application of Holodeck in Embodied AI, training agents to navigate in novel scenes like music rooms and daycares without human-constructed data, which is a significant step forward in developing general-purpose embodied agents.

{{</citation>}}


### (108/169) Auto-Prox: Training-Free Vision Transformer Architecture Search via Automatic Proxy Discovery (Zimian Wei et al., 2023)

{{<citation>}}

Zimian Wei, Lujun Li, Peijie Dong, Zheng Hui, Anggeng Li, Menglong Lu, Hengyue Pan, Zhiliang Tian, Dongsheng Li. (2023)  
**Auto-Prox: Training-Free Vision Transformer Architecture Search via Automatic Proxy Discovery**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Transformer  
[Paper Link](http://arxiv.org/abs/2312.09059v1)  

---


**ABSTRACT**  
The substantial success of Vision Transformer (ViT) in computer vision tasks is largely attributed to the architecture design. This underscores the necessity of efficient architecture search for designing better ViTs automatically. As training-based architecture search methods are computationally intensive, there is a growing interest in training-free methods that use zero-cost proxies to score ViTs. However, existing training-free approaches require expert knowledge to manually design specific zero-cost proxies. Moreover, these zero-cost proxies exhibit limitations to generalize across diverse domains. In this paper, we introduce Auto-Prox, an automatic proxy discovery framework, to address the problem. First, we build the ViT-Bench-101, which involves different ViT candidates and their actual performance on multiple datasets. Utilizing ViT-Bench-101, we can evaluate zero-cost proxies based on their score-accuracy correlation. Then, we represent zero-cost proxies with computation graphs and organize the zero-cost proxy search space with ViT statistics and primitive operations. To discover generic zero-cost proxies, we propose a joint correlation metric to evolve and mutate different zero-cost proxy candidates. We introduce an elitism-preserve strategy for search efficiency to achieve a better trade-off between exploitation and exploration. Based on the discovered zero-cost proxy, we conduct a ViT architecture search in a training-free manner. Extensive experiments demonstrate that our method generalizes well to different datasets and achieves state-of-the-art results both in ranking correlation and final accuracy. Codes can be found at https://github.com/lilujunai/Auto-Prox-AAAI24.

{{</citation>}}


### (109/169) Scene 3-D Reconstruction System in Scattering Medium (Zhuoyifan Zhang et al., 2023)

{{<citation>}}

Zhuoyifan Zhang, Lu Zhang, Liang Wang, Haoming Wu. (2023)  
**Scene 3-D Reconstruction System in Scattering Medium**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: NER  
[Paper Link](http://arxiv.org/abs/2312.09005v1)  

---


**ABSTRACT**  
The research on neural radiance fields for new view synthesis has experienced explosive growth with the development of new models and extensions. The NERF algorithm, suitable for underwater scenes or scattering media, is also evolving. Existing underwater 3D reconstruction systems still face challenges such as extensive training time and low rendering efficiency. This paper proposes an improved underwater 3D reconstruction system to address these issues and achieve rapid, high-quality 3D reconstruction.To begin with, we enhance underwater videos captured by a monocular camera to correct the poor image quality caused by the physical properties of the water medium while ensuring consistency in enhancement across adjacent frames. Subsequently, we perform keyframe selection on the video frames to optimize resource utilization and eliminate the impact of dynamic objects on the reconstruction results. The selected keyframes, after pose estimation using COLMAP, undergo a three-dimensional reconstruction improvement process using neural radiance fields based on multi-resolution hash coding for model construction and rendering.

{{</citation>}}


### (110/169) CL2CM: Improving Cross-Lingual Cross-Modal Retrieval via Cross-Lingual Knowledge Transfer (Yabing Wang et al., 2023)

{{<citation>}}

Yabing Wang, Fan Wang, Jianfeng Dong, Hao Luo. (2023)  
**CL2CM: Improving Cross-Lingual Cross-Modal Retrieval via Cross-Lingual Knowledge Transfer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2312.08984v1)  

---


**ABSTRACT**  
Cross-lingual cross-modal retrieval has garnered increasing attention recently, which aims to achieve the alignment between vision and target language (V-T) without using any annotated V-T data pairs. Current methods employ machine translation (MT) to construct pseudo-parallel data pairs, which are then used to learn a multi-lingual and multi-modal embedding space that aligns visual and target-language representations. However, the large heterogeneous gap between vision and text, along with the noise present in target language translations, poses significant challenges in effectively aligning their representations. To address these challenges, we propose a general framework, Cross-Lingual to Cross-Modal (CL2CM), which improves the alignment between vision and target language using cross-lingual transfer. This approach allows us to fully leverage the merits of multi-lingual pre-trained models (e.g., mBERT) and the benefits of the same modality structure, i.e., smaller gap, to provide reliable and comprehensive semantic correspondence (knowledge) for the cross-modal network. We evaluate our proposed approach on two multilingual image-text datasets, Multi30K and MSCOCO, and one video-text dataset, VATEX. The results clearly demonstrate the effectiveness of our proposed method and its high potential for large-scale retrieval.

{{</citation>}}


### (111/169) LEMON: Learning 3D Human-Object Interaction Relation from 2D Images (Yuhang Yang et al., 2023)

{{<citation>}}

Yuhang Yang, Wei Zhai, Hongchen Luo, Yang Cao, Zheng-Jun Zha. (2023)  
**LEMON: Learning 3D Human-Object Interaction Relation from 2D Images**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.08963v1)  

---


**ABSTRACT**  
Learning 3D human-object interaction relation is pivotal to embodied AI and interaction modeling. Most existing methods approach the goal by learning to predict isolated interaction elements, e.g., human contact, object affordance, and human-object spatial relation, primarily from the perspective of either the human or the object. Which underexploit certain correlations between the interaction counterparts (human and object), and struggle to address the uncertainty in interactions. Actually, objects' functionalities potentially affect humans' interaction intentions, which reveals what the interaction is. Meanwhile, the interacting humans and objects exhibit matching geometric structures, which presents how to interact. In light of this, we propose harnessing these inherent correlations between interaction counterparts to mitigate the uncertainty and jointly anticipate the above interaction elements in 3D space. To achieve this, we present LEMON (LEarning 3D huMan-Object iNteraction relation), a unified model that mines interaction intentions of the counterparts and employs curvatures to guide the extraction of geometric correlations, combining them to anticipate the interaction elements. Besides, the 3D Interaction Relation dataset (3DIR) is collected to serve as the test bed for training and evaluation. Extensive experiments demonstrate the superiority of LEMON over methods estimating each element in isolation.

{{</citation>}}


### (112/169) Depicting Beyond Scores: Advancing Image Quality Assessment through Multi-modal Language Models (Zhiyuan You et al., 2023)

{{<citation>}}

Zhiyuan You, Zheyuan Li, Jinjin Gu, Zhenfei Yin, Tianfan Xue, Chao Dong. (2023)  
**Depicting Beyond Scores: Advancing Image Quality Assessment through Multi-modal Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model, QA  
[Paper Link](http://arxiv.org/abs/2312.08962v1)  

---


**ABSTRACT**  
We introduce a Depicted image Quality Assessment method (DepictQA), overcoming the constraints of traditional score-based approaches. DepictQA leverages Multi-modal Large Language Models (MLLMs), allowing for detailed, language-based, human-like evaluation of image quality. Unlike conventional Image Quality Assessment (IQA) methods relying on scores, DepictQA interprets image content and distortions descriptively and comparatively, aligning closely with humans' reasoning process. To build the DepictQA model, we establish a hierarchical task framework, and collect a multi-modal IQA training dataset, named M-BAPPS. To navigate the challenges in limited training data and processing multiple images, we propose to use multi-source training data and specialized image tags. Our DepictQA demonstrates a better performance than score-based methods on the BAPPS benchmark. Moreover, compared with general MLLMs, our DepictQA can generate more accurate reasoning descriptive languages. Our research indicates that language-based IQA methods have the potential to be customized for individual preferences. Datasets and codes will be released publicly.

{{</citation>}}


### (113/169) An Incremental Unified Framework for Small Defect Inspection (Jiaqi Tang et al., 2023)

{{<citation>}}

Jiaqi Tang, Hao Lu, Xiaogang Xu, Ruizheng Wu, Sixing Hu, Tong Zhang, Tsz Wa Cheng, Ming Ge, Ying-Cong Chen, Fugee Tsung. (2023)  
**An Incremental Unified Framework for Small Defect Inspection**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI, Attention, Self-Attention  
[Paper Link](http://arxiv.org/abs/2312.08917v1)  

---


**ABSTRACT**  
Artificial Intelligence (AI)-driven defect inspection is pivotal in industrial manufacturing. Yet, many methods, tailored to specific pipelines, grapple with diverse product portfolios and evolving processes. Addressing this, we present the Incremental Unified Framework (IUF) that can reduce the feature conflict problem when continuously integrating new objects in the pipeline, making it advantageous in object-incremental learning scenarios. Employing a state-of-the-art transformer, we introduce Object-Aware Self-Attention (OASA) to delineate distinct semantic boundaries. Semantic Compression Loss (SCL) is integrated to optimize non-primary semantic space, enhancing network adaptability for novel objects. Additionally, we prioritize retaining the features of established objects during weight updates. Demonstrating prowess in both image and pixel-level defect inspection, our approach achieves state-of-the-art performance, proving indispensable for dynamic and scalable industrial inspections. Our code will be released at https://github.com/jqtangust/IUF.

{{</citation>}}


### (114/169) Progressive Feature Self-reinforcement for Weakly Supervised Semantic Segmentation (Jingxuan He et al., 2023)

{{<citation>}}

Jingxuan He, Lechao Cheng, Chaowei Fang, Zunlei Feng, Tingting Mu, Mingli Song. (2023)  
**Progressive Feature Self-reinforcement for Weakly Supervised Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2312.08916v2)  

---


**ABSTRACT**  
Compared to conventional semantic segmentation with pixel-level supervision, Weakly Supervised Semantic Segmentation (WSSS) with image-level labels poses the challenge that it always focuses on the most discriminative regions, resulting in a disparity between fully supervised conditions. A typical manifestation is the diminished precision on the object boundaries, leading to a deteriorated accuracy of WSSS. To alleviate this issue, we propose to adaptively partition the image content into deterministic regions (e.g., confident foreground and background) and uncertain regions (e.g., object boundaries and misclassified categories) for separate processing. For uncertain cues, we employ an activation-based masking strategy and seek to recover the local information with self-distilled knowledge. We further assume that the unmasked confident regions should be robust enough to preserve the global semantics. Building upon this, we introduce a complementary self-enhancement method that constrains the semantic consistency between these confident regions and an augmented image with the same class labels. Extensive experiments conducted on PASCAL VOC 2012 and MS COCO 2014 demonstrate that our proposed single-stage approach for WSSS not only outperforms state-of-the-art benchmarks remarkably but also surpasses multi-stage methodologies that trade complexity for accuracy. The code can be found at \url{https://github.com/Jessie459/feature-self-reinforcement}.

{{</citation>}}


### (115/169) CogAgent: A Visual Language Model for GUI Agents (Wenyi Hong et al., 2023)

{{<citation>}}

Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxiao Dong, Ming Ding, Jie Tang. (2023)  
**CogAgent: A Visual Language Model for GUI Agents**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, ChatGPT, GPT, Language Model, QA  
[Paper Link](http://arxiv.org/abs/2312.08914v1)  

---


**ABSTRACT**  
People are spending an enormous amount of time on digital devices through graphical user interfaces (GUIs), e.g., computer or smartphone screens. Large language models (LLMs) such as ChatGPT can assist people in tasks like writing emails, but struggle to understand and interact with GUIs, thus limiting their potential to increase automation levels. In this paper, we introduce CogAgent, an 18-billion-parameter visual language model (VLM) specializing in GUI understanding and navigation. By utilizing both low-resolution and high-resolution image encoders, CogAgent supports input at a resolution of 1120*1120, enabling it to recognize tiny page elements and text. As a generalist visual language model, CogAgent achieves the state of the art on five text-rich and four general VQA benchmarks, including VQAv2, OK-VQA, Text-VQA, ST-VQA, ChartQA, infoVQA, DocVQA, MM-Vet, and POPE. CogAgent, using only screenshots as input, outperforms LLM-based methods that consume extracted HTML text on both PC and Android GUI navigation tasks -- Mind2Web and AITW, advancing the state of the art. The model and codes are available at \url{https://github.com/THUDM/CogVLM}.

{{</citation>}}


### (116/169) Dataset Distillation via Adversarial Prediction Matching (Mingyang Chen et al., 2023)

{{<citation>}}

Mingyang Chen, Bo Huang, Junda Lu, Bing Li, Yi Wang, Minhao Cheng, Wei Wang. (2023)  
**Dataset Distillation via Adversarial Prediction Matching**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2312.08912v1)  

---


**ABSTRACT**  
Dataset distillation is the technique of synthesizing smaller condensed datasets from large original datasets while retaining necessary information to persist the effect. In this paper, we approach the dataset distillation problem from a novel perspective: we regard minimizing the prediction discrepancy on the real data distribution between models, which are respectively trained on the large original dataset and on the small distilled dataset, as a conduit for condensing information from the raw data into the distilled version. An adversarial framework is proposed to solve the problem efficiently. In contrast to existing distillation methods involving nested optimization or long-range gradient unrolling, our approach hinges on single-level optimization. This ensures the memory efficiency of our method and provides a flexible tradeoff between time and memory budgets, allowing us to distil ImageNet-1K using a minimum of only 6.5GB of GPU memory. Under the optimal tradeoff strategy, it requires only 2.5$\times$ less memory and 5$\times$ less runtime compared to the state-of-the-art. Empirically, our method can produce synthetic datasets just 10% the size of the original, yet achieve, on average, 94% of the test accuracy of models trained on the full original datasets including ImageNet-1K, significantly surpassing state-of-the-art. Additionally, extensive tests reveal that our distilled datasets excel in cross-architecture generalization capabilities.

{{</citation>}}


### (117/169) Motion Flow Matching for Human Motion Synthesis and Editing (Vincent Tao Hu et al., 2023)

{{<citation>}}

Vincent Tao Hu, Wenzhe Yin, Pingchuan Ma, Yunlu Chen, Basura Fernando, Yuki M Asano, Efstratios Gavves, Pascal Mettes, Bjorn Ommer, Cees G. M. Snoek. (2023)  
**Motion Flow Matching for Human Motion Synthesis and Editing**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2312.08895v1)  

---


**ABSTRACT**  
Human motion synthesis is a fundamental task in computer animation. Recent methods based on diffusion models or GPT structure demonstrate commendable performance but exhibit drawbacks in terms of slow sampling speeds and error accumulation. In this paper, we propose \emph{Motion Flow Matching}, a novel generative model designed for human motion generation featuring efficient sampling and effectiveness in motion editing applications. Our method reduces the sampling complexity from thousand steps in previous diffusion models to just ten steps, while achieving comparable performance in text-to-motion and action-to-motion generation benchmarks. Noticeably, our approach establishes a new state-of-the-art Fr\'echet Inception Distance on the KIT-ML dataset. What is more, we tailor a straightforward motion editing paradigm named \emph{sampling trajectory rewriting} leveraging the ODE-style generative models and apply it to various editing scenarios including motion prediction, motion in-between prediction, motion interpolation, and upper-body editing. Our code will be released.

{{</citation>}}


### (118/169) Improving Cross-modal Alignment with Synthetic Pairs for Text-only Image Captioning (Zhiyue Liu et al., 2023)

{{<citation>}}

Zhiyue Liu, Jinyuan Liu, Fanrong Ma. (2023)  
**Improving Cross-modal Alignment with Synthetic Pairs for Text-only Image Captioning**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: Image Captioning  
[Paper Link](http://arxiv.org/abs/2312.08865v1)  

---


**ABSTRACT**  
Although image captioning models have made significant advancements in recent years, the majority of them heavily depend on high-quality datasets containing paired images and texts which are costly to acquire. Previous works leverage the CLIP's cross-modal association ability for image captioning, relying solely on textual information under unsupervised settings. However, not only does a modality gap exist between CLIP text and image features, but a discrepancy also arises between training and inference due to the unavailability of real-world images, which hinders the cross-modal alignment in text-only captioning. This paper proposes a novel method to address these issues by incorporating synthetic image-text pairs. A pre-trained text-to-image model is deployed to obtain images that correspond to textual data, and the pseudo features of generated images are optimized toward the real ones in the CLIP embedding space. Furthermore, textual information is gathered to represent image features, resulting in the image features with various semantics and the bridged modality gap. To unify training and inference, synthetic image features would serve as the training prefix for the language decoder, while real images are used for inference. Additionally, salient objects in images are detected as assistance to enhance the learning of modality alignment. Experimental results demonstrate that our method obtains the state-of-the-art performance on benchmark datasets.

{{</citation>}}


### (119/169) Achelous++: Power-Oriented Water-Surface Panoptic Perception Framework on Edge Devices based on Vision-Radar Fusion and Pruning of Heterogeneous Modalities (Runwei Guan et al., 2023)

{{<citation>}}

Runwei Guan, Haocheng Zhao, Shanliang Yao, Ka Lok Man, Xiaohui Zhu, Limin Yu, Yong Yue, Jeremy Smith, Eng Gee Lim, Weiping Ding, Yutao Yue. (2023)  
**Achelous++: Power-Oriented Water-Surface Panoptic Perception Framework on Edge Devices based on Vision-Radar Fusion and Pruning of Heterogeneous Modalities**  

---
Primary Category: cs.CV  
Categories: cs-CE, cs-CV, cs-RO, cs.CV  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2312.08851v1)  

---


**ABSTRACT**  
Urban water-surface robust perception serves as the foundation for intelligent monitoring of aquatic environments and the autonomous navigation and operation of unmanned vessels, especially in the context of waterway safety. It is worth noting that current multi-sensor fusion and multi-task learning models consume substantial power and heavily rely on high-power GPUs for inference. This contributes to increased carbon emissions, a concern that runs counter to the prevailing emphasis on environmental preservation and the pursuit of sustainable, low-carbon urban environments. In light of these concerns, this paper concentrates on low-power, lightweight, multi-task panoptic perception through the fusion of visual and 4D radar data, which is seen as a promising low-cost perception method. We propose a framework named Achelous++ that facilitates the development and comprehensive evaluation of multi-task water-surface panoptic perception models. Achelous++ can simultaneously execute five perception tasks with high speed and low power consumption, including object detection, object semantic segmentation, drivable-area segmentation, waterline segmentation, and radar point cloud semantic segmentation. Furthermore, to meet the demand for developers to customize models for real-time inference on low-performance devices, a novel multi-modal pruning strategy known as Heterogeneous-Aware SynFlow (HA-SynFlow) is proposed. Besides, Achelous++ also supports random pruning at initialization with different layer-wise sparsity, such as Uniform and Erdos-Renyi-Kernel (ERK). Overall, our Achelous++ framework achieves state-of-the-art performance on the WaterScenes benchmark, excelling in both accuracy and power efficiency compared to other single-task and multi-task models. We release and maintain the code at https://github.com/GuanRunwei/Achelous.

{{</citation>}}


### (120/169) Guided Diffusion from Self-Supervised Diffusion Features (Vincent Tao Hu et al., 2023)

{{<citation>}}

Vincent Tao Hu, Yunlu Chen, Mathilde Caron, Yuki M. Asano, Cees G. M. Snoek, Bjorn Ommer. (2023)  
**Guided Diffusion from Self-Supervised Diffusion Features**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2312.08825v1)  

---


**ABSTRACT**  
Guidance serves as a key concept in diffusion models, yet its effectiveness is often limited by the need for extra data annotation or classifier pretraining. That is why guidance was harnessed from self-supervised learning backbones, like DINO. However, recent studies have revealed that the feature representation derived from diffusion model itself is discriminative for numerous downstream tasks as well, which prompts us to propose a framework to extract guidance from, and specifically for, diffusion models. Our research has yielded several significant contributions. Firstly, the guidance signals from diffusion models are on par with those from class-conditioned diffusion models. Secondly, feature regularization, when based on the Sinkhorn-Knopp algorithm, can further enhance feature discriminability in comparison to unconditional diffusion models. Thirdly, we have constructed an online training approach that can concurrently derive guidance from diffusion models for diffusion models. Lastly, we have extended the application of diffusion models along the constant velocity path of ODE to achieve a more favorable balance between sampling steps and fidelity. The performance of our methods has been outstanding, outperforming related baseline comparisons in large-resolution datasets, such as ImageNet256, ImageNet256-100 and LSUN-Churches. Our code will be released.

{{</citation>}}


### (121/169) VSFormer: Visual-Spatial Fusion Transformer for Correspondence Pruning (Tangfei Liao et al., 2023)

{{<citation>}}

Tangfei Liao, Xiaoqin Zhang, Li Zhao, Tao Wang, Guobao Xiao. (2023)  
**VSFormer: Visual-Spatial Fusion Transformer for Correspondence Pruning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Pruning, Transformer  
[Paper Link](http://arxiv.org/abs/2312.08774v2)  

---


**ABSTRACT**  
Correspondence pruning aims to find correct matches (inliers) from an initial set of putative correspondences, which is a fundamental task for many applications. The process of finding is challenging, given the varying inlier ratios between scenes/image pairs due to significant visual differences. However, the performance of the existing methods is usually limited by the problem of lacking visual cues (\eg texture, illumination, structure) of scenes. In this paper, we propose a Visual-Spatial Fusion Transformer (VSFormer) to identify inliers and recover camera poses accurately. Firstly, we obtain highly abstract visual cues of a scene with the cross attention between local features of two-view images. Then, we model these visual cues and correspondences by a joint visual-spatial fusion module, simultaneously embedding visual cues into correspondences for pruning. Additionally, to mine the consistency of correspondences, we also design a novel module that combines the KNN-based graph and the transformer, effectively capturing both local and global contexts. Extensive experiments have demonstrated that the proposed VSFormer outperforms state-of-the-art methods on outdoor and indoor benchmarks.

{{</citation>}}


### (122/169) Offshore Wind Plant Instance Segmentation Using Sentinel-1 Time Series, GIS, and Semantic Segmentation Models (Osmar Luiz Ferreira de Carvalho et al., 2023)

{{<citation>}}

Osmar Luiz Ferreira de Carvalho, Osmar Abilio de Carvalho Junior, Anesmar Olino de Albuquerque, Daniel Guerreiro e Silva. (2023)  
**Offshore Wind Plant Instance Segmentation Using Sentinel-1 Time Series, GIS, and Semantic Segmentation Models**  

---
Primary Category: cs.CV  
Categories: 68T45, I-4-6, cs-AI, cs-CV, cs-LG, cs.CV, eess-IV  
Keywords: Semantic Segmentation, Time Series  
[Paper Link](http://arxiv.org/abs/2312.08773v1)  

---


**ABSTRACT**  
Offshore wind farms represent a renewable energy source with a significant global growth trend, and their monitoring is strategic for territorial and environmental planning. This study's primary objective is to detect offshore wind plants at an instance level using semantic segmentation models and Sentinel-1 time series. The secondary objectives are: (a) to develop a database consisting of labeled data and S-1 time series; (b) to compare the performance of five deep semantic segmentation architectures (U-Net, U-Net++, Feature Pyramid Network - FPN, DeepLabv3+, and LinkNet); (c) develop a novel augmentation strategy that shuffles the positions of the images within the time series; (d) investigate different dimensions of time series intervals (1, 5, 10, and 15 images); and (e) evaluate the semantic-to-instance conversion procedure. LinkNet was the top-performing model, followed by U-Net++ and U-Net, while FPN and DeepLabv3+ presented the worst results. The evaluation of semantic segmentation models reveals enhanced Intersection over Union (IoU) (25%) and F-score metrics (18%) with the augmentation of time series images. The study showcases the augmentation strategy's capability to mitigate biases and precisely detect invariant targets. Furthermore, the conversion from semantic to instance segmentation demonstrates its efficacy in accurately isolating individual instances within classified regions - simplifying training data and reducing annotation effort and complexity.

{{</citation>}}


### (123/169) DreamDrone (Hanyang Kong et al., 2023)

{{<citation>}}

Hanyang Kong, Dongze Lian, Michael Bi Mi, Xinchao Wang. (2023)  
**DreamDrone**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2312.08746v2)  

---


**ABSTRACT**  
We introduce DreamDrone, an innovative method for generating unbounded flythrough scenes from textual prompts. Central to our method is a novel feature-correspondence-guidance diffusion process, which utilizes the strong correspondence of intermediate features in the diffusion model. Leveraging this guidance strategy, we further propose an advanced technique for editing the intermediate latent code, enabling the generation of subsequent novel views with geometric consistency. Extensive experiments reveal that DreamDrone significantly surpasses existing methods, delivering highly authentic scene generation with exceptional visual quality. This approach marks a significant step in zero-shot perpetual view generation from textual prompts, enabling the creation of diverse scenes, including natural landscapes like oases and caves, as well as complex urban settings such as Lego-style street views. Our code is publicly available.

{{</citation>}}


### (124/169) Polyper: Boundary Sensitive Polyp Segmentation (Hao Shao et al., 2023)

{{<citation>}}

Hao Shao, Yang Zhang, Qibin Hou. (2023)  
**Polyper: Boundary Sensitive Polyp Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.08735v1)  

---


**ABSTRACT**  
We present a new boundary sensitive framework for polyp segmentation, called Polyper. Our method is motivated by a clinical approach that seasoned medical practitioners often leverage the inherent features of interior polyp regions to tackle blurred boundaries.Inspired by this, we propose explicitly leveraging polyp regions to bolster the model's boundary discrimination capability while minimizing computation. Our approach first extracts boundary and polyp regions from the initial segmentation map through morphological operators. Then, we design the boundary sensitive attention that concentrates on augmenting the features near the boundary regions using the interior polyp regions's characteristics to generate good segmentation results. Our proposed method can be seamlessly integrated with classical encoder networks, like ResNet-50, MiT-B1, and Swin Transformer. To evaluate the effectiveness of Polyper, we conduct experiments on five publicly available challenging datasets, and receive state-of-the-art performance on all of them. Code is available at https://github.com/haoshao-nku/medical_seg.git.

{{</citation>}}


### (125/169) VMT-Adapter: Parameter-Efficient Transfer Learning for Multi-Task Dense Scene Understanding (Yi Xin et al., 2023)

{{<citation>}}

Yi Xin, Junlong Du, Qiang Wang, Zhiwen Lin, Ke Yan. (2023)  
**VMT-Adapter: Parameter-Efficient Transfer Learning for Multi-Task Dense Scene Understanding**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2312.08733v2)  

---


**ABSTRACT**  
Large-scale pre-trained models have achieved remarkable success in various computer vision tasks. A standard approach to leverage these models is to fine-tune all model parameters for downstream tasks, which poses challenges in terms of computational and storage costs. Recently, inspired by Natural Language Processing (NLP), parameter-efficient transfer learning has been successfully applied to vision tasks. However, most existing techniques primarily focus on single-task adaptation, and despite limited research on multi-task adaptation, these methods often exhibit suboptimal training and inference efficiency. In this paper, we first propose an once-for-all Vision Multi-Task Adapter (VMT-Adapter), which strikes approximately O(1) training and inference efficiency w.r.t task number. Concretely, VMT-Adapter shares the knowledge from multiple tasks to enhance cross-task interaction while preserves task-specific knowledge via independent knowledge extraction modules. Notably, since task-specific modules require few parameters, VMT-Adapter can handle an arbitrary number of tasks with a negligible increase of trainable parameters. We also propose VMT-Adapter-Lite, which further reduces the trainable parameters by learning shared parameters between down- and up-projections. Extensive experiments on four dense scene understanding tasks demonstrate the superiority of VMT-Adapter(-Lite), achieving a 3.96%(1.34%) relative improvement compared to single-task full fine-tuning, while utilizing merely ~1% (0.36%) trainable parameters of the pre-trained model.

{{</citation>}}


### (126/169) CPST: Comprehension-Preserving Style Transfer for Multi-Modal Narratives (Yi-Chun Chen et al., 2023)

{{<citation>}}

Yi-Chun Chen, Arnav Jhala. (2023)  
**CPST: Comprehension-Preserving Style Transfer for Multi-Modal Narratives**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-MM, cs.CV  
Keywords: Style Transfer  
[Paper Link](http://arxiv.org/abs/2312.08695v1)  

---


**ABSTRACT**  
We investigate the challenges of style transfer in multi-modal visual narratives. Among static visual narratives such as comics and manga, there are distinct visual styles in terms of presentation. They include style features across multiple dimensions, such as panel layout, size, shape, and color. They include both visual and text media elements. The layout of both text and media elements is also significant in terms of narrative communication. The sequential transitions between panels are where readers make inferences about the narrative world. These feature differences provide an interesting challenge for style transfer in which there are distinctions between the processing of features for each modality. We introduce the notion of comprehension-preserving style transfer (CPST) in such multi-modal domains. CPST requires not only traditional metrics of style transfer but also metrics of narrative comprehension. To spur further research in this area, we present an annotated dataset of comics and manga and an initial set of algorithms that utilize separate style transfer modules for the visual, textual, and layout parameters. To test whether the style transfer preserves narrative semantics, we evaluate this algorithm through visual story cloze tests inspired by work in computational cognition of narrative systems. Understanding the connection between style and narrative semantics provides insight for applications ranging from informational brochure designs to data storytelling.

{{</citation>}}


### (127/169) SpectralNeRF: Physically Based Spectral Rendering with Neural Radiance Field (Ru Li et al., 2023)

{{<citation>}}

Ru Li, Jia Liu, Guanghui Liu, Shengping Zhang, Bing Zeng, Shuaicheng Liu. (2023)  
**SpectralNeRF: Physically Based Spectral Rendering with Neural Radiance Field**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.08692v1)  

---


**ABSTRACT**  
In this paper, we propose SpectralNeRF, an end-to-end Neural Radiance Field (NeRF)-based architecture for high-quality physically based rendering from a novel spectral perspective. We modify the classical spectral rendering into two main steps, 1) the generation of a series of spectrum maps spanning different wavelengths, 2) the combination of these spectrum maps for the RGB output. Our SpectralNeRF follows these two steps through the proposed multi-layer perceptron (MLP)-based architecture (SpectralMLP) and Spectrum Attention UNet (SAUNet). Given the ray origin and the ray direction, the SpectralMLP constructs the spectral radiance field to obtain spectrum maps of novel views, which are then sent to the SAUNet to produce RGB images of white-light illumination. Applying NeRF to build up the spectral rendering is a more physically-based way from the perspective of ray-tracing. Further, the spectral radiance fields decompose difficult scenes and improve the performance of NeRF-based methods. Comprehensive experimental results demonstrate the proposed SpectralNeRF is superior to recent NeRF-based methods when synthesizing new views on synthetic and real datasets. The codes and datasets are available at https://github.com/liru0126/SpectralNeRF.

{{</citation>}}


### (128/169) AVA: Inconspicuous Attribute Variation-based Adversarial Attack bypassing DeepFake Detection (Xiangtao Meng et al., 2023)

{{<citation>}}

Xiangtao Meng, Li Wang, Shanqing Guo, Lei Ju, Qingchuan Zhao. (2023)  
**AVA: Inconspicuous Attribute Variation-based Adversarial Attack bypassing DeepFake Detection**  

---
Primary Category: cs.CV  
Categories: cs-CR, cs-CV, cs.CV  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2312.08675v1)  

---


**ABSTRACT**  
While DeepFake applications are becoming popular in recent years, their abuses pose a serious privacy threat. Unfortunately, most related detection algorithms to mitigate the abuse issues are inherently vulnerable to adversarial attacks because they are built atop DNN-based classification models, and the literature has demonstrated that they could be bypassed by introducing pixel-level perturbations. Though corresponding mitigation has been proposed, we have identified a new attribute-variation-based adversarial attack (AVA) that perturbs the latent space via a combination of Gaussian prior and semantic discriminator to bypass such mitigation. It perturbs the semantics in the attribute space of DeepFake images, which are inconspicuous to human beings (e.g., mouth open) but can result in substantial differences in DeepFake detection. We evaluate our proposed AVA attack on nine state-of-the-art DeepFake detection algorithms and applications. The empirical results demonstrate that AVA attack defeats the state-of-the-art black box attacks against DeepFake detectors and achieves more than a 95% success rate on two commercial DeepFake detectors. Moreover, our human study indicates that AVA-generated DeepFake images are often imperceptible to humans, which presents huge security and privacy concerns.

{{</citation>}}


### (129/169) Segment Beyond View: Handling Partially Missing Modality for Audio-Visual Semantic Segmentation (Renjie Wu et al., 2023)

{{<citation>}}

Renjie Wu, Hu Wang, Feras Dayoub, Hsiang-Ting Chen. (2023)  
**Segment Beyond View: Handling Partially Missing Modality for Audio-Visual Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-SD, cs.CV, eess-AS  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2312.08673v1)  

---


**ABSTRACT**  
Augmented Reality (AR) devices, emerging as prominent mobile interaction platforms, face challenges in user safety, particularly concerning oncoming vehicles. While some solutions leverage onboard camera arrays, these cameras often have limited field-of-view (FoV) with front or downward perspectives. Addressing this, we propose a new out-of-view semantic segmentation task and Segment Beyond View (SBV), a novel audio-visual semantic segmentation method. SBV supplements the visual modality, which miss the information beyond FoV, with the auditory information using a teacher-student distillation model (Omni2Ego). The model consists of a vision teacher utilising panoramic information, an auditory teacher with 8-channel audio, and an audio-visual student that takes views with limited FoV and binaural audio as input and produce semantic segmentation for objects outside FoV. SBV outperforms existing models in comparative evaluations and shows a consistent performance across varying FoV ranges and in monaural audio settings.

{{</citation>}}


### (130/169) SPEAL: Skeletal Prior Embedded Attention Learning for Cross-Source Point Cloud Registration (Kezheng Xiong et al., 2023)

{{<citation>}}

Kezheng Xiong, Maoji Zheng, Qingshan Xu, Chenglu Wen, Siqi Shen, Cheng Wang. (2023)  
**SPEAL: Skeletal Prior Embedded Attention Learning for Cross-Source Point Cloud Registration**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2312.08664v1)  

---


**ABSTRACT**  
Point cloud registration, a fundamental task in 3D computer vision, has remained largely unexplored in cross-source point clouds and unstructured scenes. The primary challenges arise from noise, outliers, and variations in scale and density. However, neglected geometric natures of point clouds restricts the performance of current methods. In this paper, we propose a novel method termed SPEAL to leverage skeletal representations for effective learning of intrinsic topologies of point clouds, facilitating robust capture of geometric intricacy. Specifically, we design the Skeleton Extraction Module to extract skeleton points and skeletal features in an unsupervised manner, which is inherently robust to noise and density variances. Then, we propose the Skeleton-Aware GeoTransformer to encode high-level skeleton-aware features. It explicitly captures the topological natures and inter-point-cloud skeletal correlations with the noise-robust and density-invariant skeletal representations. Next, we introduce the Correspondence Dual-Sampler to facilitate correspondences by augmenting the correspondence set with skeletal correspondences. Furthermore, we construct a challenging novel large-scale cross-source point cloud dataset named KITTI CrossSource for benchmarking cross-source point cloud registration methods. Extensive quantitative and qualitative experiments are conducted to demonstrate our approach's superiority and robustness on both cross-source and same-source datasets. To the best of our knowledge, our approach is the first to facilitate point cloud registration with skeletal geometric priors.

{{</citation>}}


### (131/169) A Simple Knowledge Distillation Framework for Open-world Object Detection (Shuailei Ma et al., 2023)

{{<citation>}}

Shuailei Ma, Yuefeng Wang, Ying Wei, Jiaqi Fan, Xinyu Sun, Peihao Chen, Enming Zhang. (2023)  
**A Simple Knowledge Distillation Framework for Open-world Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Knowledge Distillation, Object Detection  
[Paper Link](http://arxiv.org/abs/2312.08653v1)  

---


**ABSTRACT**  
Open World Object Detection (OWOD) is a novel computer vision task with a considerable challenge, bridging the gap between classic object detection (OD) benchmarks and real-world object detection. In addition to detecting and classifying seen/known objects, OWOD algorithms are expected to localize all potential unseen/unknown objects and incrementally learn them. The large pre-trained vision-language grounding models (VLM,eg, GLIP) have rich knowledge about the open world, but are limited by text prompts and cannot localize indescribable objects. However, there are many detection scenarios which pre-defined language descriptions are unavailable during inference. In this paper, we attempt to specialize the VLM model for OWOD task by distilling its open-world knowledge into a language-agnostic detector. Surprisingly, we observe that the combination of a simple knowledge distillation approach and the automatic pseudo-labeling mechanism in OWOD can achieve better performance for unknown object detection, even with a small amount of data. Unfortunately, knowledge distillation for unknown objects severely affects the learning of detectors with conventional structures for known objects, leading to catastrophic forgetting. To alleviate these problems, we propose the down-weight loss function for knowledge distillation from vision-language to single vision modality. Meanwhile, we decouple the learning of localization and recognition to reduce the impact of category interactions of known and unknown objects on the localization learning process. Comprehensive experiments performed on MS-COCO and PASCAL VOC demonstrate the effectiveness of our methods.

{{</citation>}}


### (132/169) Generative Model-based Feature Knowledge Distillation for Action Recognition (Guiqin Wang et al., 2023)

{{<citation>}}

Guiqin Wang, Peng Zhao, Yanjiang Shi, Cong Zhao, Shusen Yang. (2023)  
**Generative Model-based Feature Knowledge Distillation for Action Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2312.08644v1)  

---


**ABSTRACT**  
Knowledge distillation (KD), a technique widely employed in computer vision, has emerged as a de facto standard for improving the performance of small neural networks. However, prevailing KD-based approaches in video tasks primarily focus on designing loss functions and fusing cross-modal information. This overlooks the spatial-temporal feature semantics, resulting in limited advancements in model compression. Addressing this gap, our paper introduces an innovative knowledge distillation framework, with the generative model for training a lightweight student model. In particular, the framework is organized into two steps: the initial phase is Feature Representation, wherein a generative model-based attention module is trained to represent feature semantics; Subsequently, the Generative-based Feature Distillation phase encompasses both Generative Distillation and Attention Distillation, with the objective of transferring attention-based feature semantics with the generative model. The efficacy of our approach is demonstrated through comprehensive experiments on diverse popular datasets, proving considerable enhancements in video action recognition task. Moreover, the effectiveness of our proposed framework is validated in the context of more intricate video action detection task. Our code is available at https://github.com/aaai-24/Generative-based-KD.

{{</citation>}}


### (133/169) Semi-supervised Semantic Segmentation Meets Masked Modeling:Fine-grained Locality Learning Matters in Consistency Regularization (Wentao Pan et al., 2023)

{{<citation>}}

Wentao Pan, Zhe Xu, Jiangpeng Yan, Zihan Wu, Raymond Kai-yu Tong, Xiu Li, Jianhua Yao. (2023)  
**Semi-supervised Semantic Segmentation Meets Masked Modeling:Fine-grained Locality Learning Matters in Consistency Regularization**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2312.08631v1)  

---


**ABSTRACT**  
Semi-supervised semantic segmentation aims to utilize limited labeled images and abundant unlabeled images to achieve label-efficient learning, wherein the weak-to-strong consistency regularization framework, popularized by FixMatch, is widely used as a benchmark scheme. Despite its effectiveness, we observe that such scheme struggles with satisfactory segmentation for the local regions. This can be because it originally stems from the image classification task and lacks specialized mechanisms to capture fine-grained local semantics that prioritizes in dense prediction. To address this issue, we propose a novel framework called \texttt{MaskMatch}, which enables fine-grained locality learning to achieve better dense segmentation. On top of the original teacher-student framework, we design a masked modeling proxy task that encourages the student model to predict the segmentation given the unmasked image patches (even with 30\% only) and enforces the predictions to be consistent with pseudo-labels generated by the teacher model using the complete image. Such design is motivated by the intuition that if the predictions are more consistent given insufficient neighboring information, stronger fine-grained locality perception is achieved. Besides, recognizing the importance of reliable pseudo-labels in the above locality learning and the original consistency learning scheme, we design a multi-scale ensembling strategy that considers context at different levels of abstraction for pseudo-label generation. Extensive experiments on benchmark datasets demonstrate the superiority of our method against previous approaches and its plug-and-play flexibility.

{{</citation>}}


### (134/169) Factorization Vision Transformer: Modeling Long Range Dependency with Local Window Cost (Haolin Qin et al., 2023)

{{<citation>}}

Haolin Qin, Daquan Zhou, Tingfa Xu, Ziyang Bian, Jianan Li. (2023)  
**Factorization Vision Transformer: Modeling Long Range Dependency with Local Window Cost**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.08614v1)  

---


**ABSTRACT**  
Transformers have astounding representational power but typically consume considerable computation which is quadratic with image resolution. The prevailing Swin transformer reduces computational costs through a local window strategy. However, this strategy inevitably causes two drawbacks: (1) the local window-based self-attention hinders global dependency modeling capability; (2) recent studies point out that local windows impair robustness. To overcome these challenges, we pursue a preferable trade-off between computational cost and performance. Accordingly, we propose a novel factorization self-attention mechanism (FaSA) that enjoys both the advantages of local window cost and long-range dependency modeling capability. By factorizing the conventional attention matrix into sparse sub-attention matrices, FaSA captures long-range dependencies while aggregating mixed-grained information at a computational cost equivalent to the local window-based self-attention. Leveraging FaSA, we present the factorization vision transformer (FaViT) with a hierarchical structure. FaViT achieves high performance and robustness, with linear computational complexity concerning input image spatial resolution. Extensive experiments have shown FaViT's advanced performance in classification and downstream tasks. Furthermore, it also exhibits strong model robustness to corrupted and biased data and hence demonstrates benefits in favor of practical applications. In comparison to the baseline model Swin-T, our FaViT-B2 significantly improves classification accuracy by 1% and robustness by 7%, while reducing model parameters by 14%. Our code will soon be publicly available at https://github.com/q2479036243/FaViT.

{{</citation>}}


### (135/169) VQCNIR: Clearer Night Image Restoration with Vector-Quantized Codebook (Wenbin Zou et al., 2023)

{{<citation>}}

Wenbin Zou, Hongxia Gao, Tian Ye, Liang Chen, Weipeng Yang, Shasha Huang, Hongsheng Chen, Sixiang Chen. (2023)  
**VQCNIR: Clearer Night Image Restoration with Vector-Quantized Codebook**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Attention  
[Paper Link](http://arxiv.org/abs/2312.08606v2)  

---


**ABSTRACT**  
Night photography often struggles with challenges like low light and blurring, stemming from dark environments and prolonged exposures. Current methods either disregard priors and directly fitting end-to-end networks, leading to inconsistent illumination, or rely on unreliable handcrafted priors to constrain the network, thereby bringing the greater error to the final result. We believe in the strength of data-driven high-quality priors and strive to offer a reliable and consistent prior, circumventing the restrictions of manual priors. In this paper, we propose Clearer Night Image Restoration with Vector-Quantized Codebook (VQCNIR) to achieve remarkable and consistent restoration outcomes on real-world and synthetic benchmarks. To ensure the faithful restoration of details and illumination, we propose the incorporation of two essential modules: the Adaptive Illumination Enhancement Module (AIEM) and the Deformable Bi-directional Cross-Attention (DBCA) module. The AIEM leverages the inter-channel correlation of features to dynamically maintain illumination consistency between degraded features and high-quality codebook features. Meanwhile, the DBCA module effectively integrates texture and structural information through bi-directional cross-attention and deformable convolution, resulting in enhanced fine-grained detail and structural fidelity across parallel decoders. Extensive experiments validate the remarkable benefits of VQCNIR in enhancing image quality under low-light conditions, showcasing its state-of-the-art performance on both synthetic and real-world datasets. The code is available at https://github.com/AlexZou14/VQCNIR.

{{</citation>}}


### (136/169) CartoMark: a benchmark dataset for map pattern recognition and 1 map content retrieval with machine intelligence (Xiran Zhou et al., 2023)

{{<citation>}}

Xiran Zhou, Yi Wen, Honghao Li, Kaiyuan Li, Zhenfeng Shao, Zhigang Yan, Xiao Xie. (2023)  
**CartoMark: a benchmark dataset for map pattern recognition and 1 map content retrieval with machine intelligence**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.08600v1)  

---


**ABSTRACT**  
Maps are fundamental medium to visualize and represent the real word in a simple and 16 philosophical way. The emergence of the 3rd wave information has made a proportion of maps are available to be generated ubiquitously, which would significantly enrich the dimensions and perspectives to understand the characteristics of the real world. However, a majority of map dataset have never been discovered, acquired and effectively used, and the map data used in many applications might not be completely fitted for the authentic demands of these applications. This challenge is emerged due to the lack of numerous well-labelled benchmark datasets for implementing the deep learning approaches into identifying complicated map content. Thus, we develop a large-scale benchmark dataset that includes well-labelled dataset for map text annotation recognition, map scene classification, map super-resolution reconstruction, and map style transferring. Furthermore, these well-labelled datasets would facilitate the state-of-the-art machine intelligence technologies to conduct map feature detection, map pattern recognition and map content retrieval. We hope our efforts would be useful for AI-enhanced cartographical applications.

{{</citation>}}


### (137/169) CT-MVSNet: Efficient Multi-View Stereo with Cross-scale Transformer (Sicheng Wang et al., 2023)

{{<citation>}}

Sicheng Wang, Hao Jiang, Lei Xiang. (2023)  
**CT-MVSNet: Efficient Multi-View Stereo with Cross-scale Transformer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.08594v1)  

---


**ABSTRACT**  
Recent deep multi-view stereo (MVS) methods have widely incorporated transformers into cascade network for high-resolution depth estimation, achieving impressive results. However, existing transformer-based methods are constrained by their computational costs, preventing their extension to finer stages. In this paper, we propose a novel cross-scale transformer (CT) that processes feature representations at different stages without additional computation. Specifically, we introduce an adaptive matching-aware transformer (AMT) that employs different interactive attention combinations at multiple scales. This combined strategy enables our network to capture intra-image context information and enhance inter-image feature relationships. Besides, we present a dual-feature guided aggregation (DFGA) that embeds the coarse global semantic information into the finer cost volume construction to further strengthen global and local feature awareness. Meanwhile, we design a feature metric loss (FM Loss) that evaluates the feature bias before and after transformation to reduce the impact of feature mismatch on depth estimation. Extensive experiments on DTU dataset and Tanks and Temples (T\&T) benchmark demonstrate that our method achieves state-of-the-art results. Code is available at https://github.com/wscstrive/CT-MVSNet.

{{</citation>}}


### (138/169) Dietary Assessment with Multimodal ChatGPT: A Systematic Analysis (Frank P. -W. Lo et al., 2023)

{{<citation>}}

Frank P. -W. Lo, Jianing Qiu, Zeyu Wang, Junhong Chen, Bo Xiao, Wu Yuan, Stamatia Giannarou, Gary Frost, Benny Lo. (2023)  
**Dietary Assessment with Multimodal ChatGPT: A Systematic Analysis**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, ChatGPT, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2312.08592v1)  

---


**ABSTRACT**  
Conventional approaches to dietary assessment are primarily grounded in self-reporting methods or structured interviews conducted under the supervision of dietitians. These methods, however, are often subjective, potentially inaccurate, and time-intensive. Although artificial intelligence (AI)-based solutions have been devised to automate the dietary assessment process, these prior AI methodologies encounter challenges in their ability to generalize across a diverse range of food types, dietary behaviors, and cultural contexts. This results in AI applications in the dietary field that possess a narrow specialization and limited accuracy. Recently, the emergence of multimodal foundation models such as GPT-4V powering the latest ChatGPT has exhibited transformative potential across a wide range of tasks (e.g., Scene understanding and image captioning) in numerous research domains. These models have demonstrated remarkable generalist intelligence and accuracy, capable of processing various data modalities. In this study, we explore the application of multimodal ChatGPT within the realm of dietary assessment. Our findings reveal that GPT-4V excels in food detection under challenging conditions with accuracy up to 87.5% without any fine-tuning or adaptation using food-specific datasets. By guiding the model with specific language prompts (e.g., African cuisine), it shifts from recognizing common staples like rice and bread to accurately identifying regional dishes like banku and ugali. Another GPT-4V's standout feature is its contextual awareness. GPT-4V can leverage surrounding objects as scale references to deduce the portion sizes of food items, further enhancing its accuracy in translating food weight into nutritional content. This alignment with the USDA National Nutrient Database underscores GPT-4V's potential to advance nutritional science and dietary assessment techniques.

{{</citation>}}


### (139/169) Joint2Human: High-quality 3D Human Generation via Compact Spherical Embedding of 3D Joints (Muxin Zhang et al., 2023)

{{<citation>}}

Muxin Zhang, Qiao Feng, Zhuo Su, Chao Wen, Zhou Xue, Kun Li. (2023)  
**Joint2Human: High-quality 3D Human Generation via Compact Spherical Embedding of 3D Joints**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2312.08591v1)  

---


**ABSTRACT**  
3D human generation is increasingly significant in various applications. However, the direct use of 2D generative methods in 3D generation often results in significant loss of local details, while methods that reconstruct geometry from generated images struggle with global view consistency. In this work, we introduce Joint2Human, a novel method that leverages 2D diffusion models to generate detailed 3D human geometry directly, ensuring both global structure and local details. To achieve this, we employ the Fourier occupancy field (FOF) representation, enabling the direct production of 3D shapes as preliminary results using 2D generative models. With the proposed high-frequency enhancer and the multi-view recarving strategy, our method can seamlessly integrate the details from different views into a uniform global shape.To better utilize the 3D human prior and enhance control over the generated geometry, we introduce a compact spherical embedding of 3D joints. This allows for effective application of pose guidance during the generation process. Additionally, our method is capable of generating 3D humans guided by textual inputs. Our experimental results demonstrate the capability of our method to ensure global structure, local details, high resolution, and low computational cost, simultaneously. More results and code can be found on our project page at http://cic.tju.edu.cn/faculty/likun/projects/Joint2Human.

{{</citation>}}


## cs.SI (2)



### (140/169) Echo chamber formation sharpened by priority users (Henrique F. de Arruda et al., 2023)

{{<citation>}}

Henrique F. de Arruda, Kleber A. Oliveira, Yamir Moreno. (2023)  
**Echo chamber formation sharpened by priority users**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI, physics-soc-ph  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2312.09358v1)  

---


**ABSTRACT**  
Priority users (e.g., verified profiles on Twitter) are social media users whose content is promoted by recommendation algorithms. However, the impact of this heterogeneous user influence on opinion dynamics, such as polarization phenomena, is unknown. We conduct a computational mechanistic investigation of such consequences in a stylized setting. First, we allow priority users, whose content has greater reach (similar to algorithmic boosting), into an opinion model on adaptive networks. Then, to exploit this gain in influence, we incorporate stubborn user behavior, i.e., zealot users who remain committed to opinions throughout the dynamics. Using a novel measure of echo chamber formation, we find that prioritizing users can inadvertently reduce polarization if they post according to the same rule but sharpen echo chamber formation if they behave heterogeneously. Moreover, we show that a minority of extremist ideologues (i.e., users who are both stubborn and priority) can push the system into a transition from consensus to polarization with echo chambers. Our findings imply that the implementation of the platform's prioritization policy should be carefully monitored in order to ensure there is no abuse of users with extra influence.

{{</citation>}}


### (141/169) A Generalized Neural Diffusion Framework on Graphs (Yibo Li et al., 2023)

{{<citation>}}

Yibo Li, Xiao Wang, Hongrui Liu, Chuan Shi. (2023)  
**A Generalized Neural Diffusion Framework on Graphs**  

---
Primary Category: cs.SI  
Categories: cs-AI, cs-SI, cs.SI  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2312.08616v1)  

---


**ABSTRACT**  
Recent studies reveal the connection between GNNs and the diffusion process, which motivates many diffusion-based GNNs to be proposed. However, since these two mechanisms are closely related, one fundamental question naturally arises: Is there a general diffusion framework that can formally unify these GNNs? The answer to this question can not only deepen our understanding of the learning process of GNNs, but also may open a new door to design a broad new class of GNNs. In this paper, we propose a general diffusion equation framework with the fidelity term, which formally establishes the relationship between the diffusion process with more GNNs. Meanwhile, with this framework, we identify one characteristic of graph diffusion networks, i.e., the current neural diffusion process only corresponds to the first-order diffusion equation. However, by an experimental investigation, we show that the labels of high-order neighbors actually exhibit monophily property, which induces the similarity based on labels among high-order neighbors without requiring the similarity among first-order neighbors. This discovery motives to design a new high-order neighbor-aware diffusion equation, and derive a new type of graph diffusion network (HiD-Net) based on the framework. With the high-order diffusion equation, HiD-Net is more robust against attacks and works on both homophily and heterophily graphs. We not only theoretically analyze the relation between HiD-Net with high-order random walk, but also provide a theoretical convergence guarantee. Extensive experimental results well demonstrate the effectiveness of HiD-Net over state-of-the-art graph diffusion networks.

{{</citation>}}


## cs.NI (2)



### (142/169) iOn-Profiler: intelligent Online multi-objective VNF Profiling with Reinforcement Learning (Xenofon Vasilakos et al., 2023)

{{<citation>}}

Xenofon Vasilakos, Shadi Moazzeni, Anderson Bravalheri, Pratchaya Jaisudthi, Reza Nejabati, Dimitra Simeonidou. (2023)  
**iOn-Profiler: intelligent Online multi-objective VNF Profiling with Reinforcement Learning**  

---
Primary Category: cs.NI  
Categories: cs-LG, cs-NI, cs-PF, cs.NI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.09355v1)  

---


**ABSTRACT**  
Leveraging the potential of Virtualised Network Functions (VNFs) requires a clear understanding of the link between resource consumption and performance. The current state of the art tries to do that by utilising Machine Learning (ML) and specifically Supervised Learning (SL) models for given network environments and VNF types assuming single-objective optimisation targets. Taking a different approach poses a novel VNF profiler optimising multi-resource type allocation and performance objectives using adapted Reinforcement Learning (RL). Our approach can meet Key Performance Indicator (KPI) targets while minimising multi-resource type consumption and optimising the VNF output rate compared to existing single-objective solutions. Our experimental evaluation with three real-world VNF types over a total of 39 study scenarios (13 per VNF), for three resource types (virtual CPU, memory, and network link capacity), verifies the accuracy of resource allocation predictions and corresponding successful profiling decisions via a benchmark comparison between our RL model and SL models. We also conduct a complementary exhaustive search-space study revealing that different resources impact performance in varying ways per VNF type, implying the necessity of multi-objective optimisation, individualised examination per VNF type, and adaptable online profile learning, such as with the autonomous online learning approach of iOn-Profiler.

{{</citation>}}


### (143/169) Networking for the Metaverse: The Standardization Landscape (Cedric Westphal et al., 2023)

{{<citation>}}

Cedric Westphal, Jungha Hong, Shin-Gak Kang, Leonardo Chiariglione, Tianji Jiang. (2023)  
**Networking for the Metaverse: The Standardization Landscape**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs-SI, cs.NI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.09295v1)  

---


**ABSTRACT**  
New applications are being supported by current and future networks. In particular, it is expected that Metaverse applications will be deployed in the near future, as 5G and 6G network provide sufficient bandwidth and sufficiently low latency to provide a satisfying end-user experience. However, networks still need to evolve to better support this type of application. We present here a basic taxonomy of the metaverse, which allows to identify some of the networking requirements for such an application; we also provide an overview of the current state of balthe standardization efforts in different standardization organizations, including ITU-T, 3GPP, IETF and MPAI.

{{</citation>}}


## cs.IR (2)



### (144/169) O Contract, Where Art Thou? Contract Management as a SharePoint Oddity (Sasha Vtyurina et al., 2023)

{{<citation>}}

Sasha Vtyurina, Adam Roegiest. (2023)  
**O Contract, Where Art Thou? Contract Management as a SharePoint Oddity**  

---
Primary Category: cs.IR  
Categories: cs-CY, cs-IR, cs.IR  
Keywords: Microsoft  
[Paper Link](http://arxiv.org/abs/2312.09312v1)  

---


**ABSTRACT**  
For many legal operations teams, the management of the contracts and agreements that their organization are negotiating or have been executed is an encompassing and time-consuming task. This has resulted in specialized tools for Contract Lifecycle Management (CLM) have grown steadily in demand over the last decade. Transitioning to such tools can itself be an arduous and costly process and so a logical step would be to augment existing storage solutions. In this paper, we present the analysis of 26 semi-structured interviews with legal operations professionals about their trials and tribulations with using Microsoft SharePoint for contract management. We find that while there is promise, too much of what is needed to be successful requires more technical prowess than might be easily available to those empowered to put it in place.

{{</citation>}}


### (145/169) Evaluative Item-Contrastive Explanations in Rankings (Alessandro Castelnovo et al., 2023)

{{<citation>}}

Alessandro Castelnovo, Riccardo Crupi, Nicolò Mombelli, Gabriele Nanino, Daniele Regoli. (2023)  
**Evaluative Item-Contrastive Explanations in Rankings**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-CY, cs-HC, cs-IR, cs.IR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.10094v1)  

---


**ABSTRACT**  
The remarkable success of Artificial Intelligence in advancing automated decision-making is evident both in academia and industry. Within the plethora of applications, ranking systems hold significant importance in various domains. This paper advocates for the application of a specific form of Explainable AI -- namely, contrastive explanations -- as particularly well-suited for addressing ranking problems. This approach is especially potent when combined with an Evaluative AI methodology, which conscientiously evaluates both positive and negative aspects influencing a potential ranking. Therefore, the present work introduces Evaluative Item-Contrastive Explanations tailored for ranking systems and illustrates its application and characteristics through an experiment conducted on publicly available data.

{{</citation>}}


## quant-ph (1)



### (146/169) Towards Efficient Quantum Anomaly Detection: One-Class SVMs using Variable Subsampling and Randomized Measurements (Michael Kölle et al., 2023)

{{<citation>}}

Michael Kölle, Afrae Ahouzi, Pascal Debus, Robert Müller, Danielle Schuman, Claudia Linnhoff-Popien. (2023)  
**Towards Efficient Quantum Anomaly Detection: One-Class SVMs using Variable Subsampling and Randomized Measurements**  

---
Primary Category: quant-ph  
Categories: cs-LG, quant-ph, quant-ph  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2312.09174v1)  

---


**ABSTRACT**  
Quantum computing, with its potential to enhance various machine learning tasks, allows significant advancements in kernel calculation and model precision. Utilizing the one-class Support Vector Machine alongside a quantum kernel, known for its classically challenging representational capacity, notable improvements in average precision compared to classical counterparts were observed in previous studies. Conventional calculations of these kernels, however, present a quadratic time complexity concerning data size, posing challenges in practical applications. To mitigate this, we explore two distinct approaches: utilizing randomized measurements to evaluate the quantum kernel and implementing the variable subsampling ensemble method, both targeting linear time complexity. Experimental results demonstrate a substantial reduction in training and inference times by up to 95\% and 25\% respectively, employing these methods. Although unstable, the average precision of randomized measurements discernibly surpasses that of the classical Radial Basis Function kernel, suggesting a promising direction for further research in scalable, efficient quantum computing applications in machine learning.

{{</citation>}}


## cs.SD (5)



### (147/169) F1-EV Score: Measuring the Likelihood of Estimating a Good Decision Threshold for Semi-Supervised Anomaly Detection (Kevin Wilkinghoff et al., 2023)

{{<citation>}}

Kevin Wilkinghoff, Keisuke Imoto. (2023)  
**F1-EV Score: Measuring the Likelihood of Estimating a Good Decision Threshold for Semi-Supervised Anomaly Detection**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Anomaly Detection, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2312.09143v1)  

---


**ABSTRACT**  
Anomalous sound detection (ASD) systems are usually compared by using threshold-independent performance measures such as AUC-ROC. However, for practical applications a decision threshold is needed to decide whether a given test sample is normal or anomalous. Estimating such a threshold is highly non-trivial in a semi-supervised setting where only normal training samples are available. In this work, F1-EV a novel threshold-independent performance measure for ASD systems that also includes the likelihood of estimating a good decision threshold is proposed and motivated using specific toy examples. In experimental evaluations, multiple performance measures are evaluated for all systems submitted to the ASD task of the DCASE Challenge 2023. It is shown that F1-EV is strongly correlated with AUC-ROC while having a significantly stronger correlation with the F1-score obtained with estimated and optimal decision thresholds than AUC-ROC.

{{</citation>}}


### (148/169) STaR: Distilling Speech Temporal Relation for Lightweight Speech Self-Supervised Learning Models (Kangwook Jang et al., 2023)

{{<citation>}}

Kangwook Jang, Sungnyun Kim, Hoirin Kim. (2023)  
**STaR: Distilling Speech Temporal Relation for Lightweight Speech Self-Supervised Learning Models**  

---
Primary Category: cs.SD  
Categories: cs-CL, cs-SD, cs.SD, eess-AS  
Keywords: BERT, Self-Supervised, Transformer  
[Paper Link](http://arxiv.org/abs/2312.09040v1)  

---


**ABSTRACT**  
Albeit great performance of Transformer-based speech selfsupervised learning (SSL) models, their large parameter size and computational cost make them unfavorable to utilize. In this study, we propose to compress the speech SSL models by distilling speech temporal relation (STaR). Unlike previous works that directly match the representation for each speech frame, STaR distillation transfers temporal relation between speech frames, which is more suitable for lightweight student with limited capacity. We explore three STaR distillation objectives and select the best combination as the final STaR loss. Our model distilled from HuBERT BASE achieves an overall score of 79.8 on SUPERB benchmark, the best performance among models with up to 27 million parameters. We show that our method is applicable across different speech SSL models and maintains robust performance with further reduced parameters.

{{</citation>}}


### (149/169) Acoustic models of Brazilian Portuguese Speech based on Neural Transformers (Marcelo Matheus Gauy et al., 2023)

{{<citation>}}

Marcelo Matheus Gauy, Marcelo Finger. (2023)  
**Acoustic models of Brazilian Portuguese Speech based on Neural Transformers**  

---
Primary Category: cs.SD  
Categories: cs-CL, cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.09265v1)  

---


**ABSTRACT**  
An acoustic model, trained on a significant amount of unlabeled data, consists of a self-supervised learned speech representation useful for solving downstream tasks, perhaps after a fine-tuning of the model in the respective downstream task. In this work, we build an acoustic model of Brazilian Portuguese Speech through a Transformer neural network. This model was pretrained on more than $800$ hours of Brazilian Portuguese Speech, using a combination of pretraining techniques. Using a labeled dataset collected for the detection of respiratory insufficiency in Brazilian Portuguese speakers, we fine-tune the pretrained Transformer neural network on the following tasks: respiratory insufficiency detection, gender recognition and age group classification. We compare the performance of pretrained Transformers on these tasks with that of Transformers without previous pretraining, noting a significant improvement. In particular, the performance of respiratory insufficiency detection obtains the best reported results so far, indicating this kind of acoustic model as a promising tool for speech-as-biomarker approach. Moreover, the performance of gender recognition is comparable to the state of the art models in English.

{{</citation>}}


### (150/169) Hourglass-AVSR: Down-Up Sampling-based Computational Efficiency Model for Audio-Visual Speech Recognition (Fan Yu et al., 2023)

{{<citation>}}

Fan Yu, Haoxu Wang, Ziyang Ma, Shiliang Zhang. (2023)  
**Hourglass-AVSR: Down-Up Sampling-based Computational Efficiency Model for Audio-Visual Speech Recognition**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2312.08850v1)  

---


**ABSTRACT**  
Recently audio-visual speech recognition (AVSR), which better leverages video modality as additional information to extend automatic speech recognition (ASR), has shown promising results in complex acoustic environments. However, there is still substantial space to improve as complex computation of visual modules and ineffective fusion of audio-visual modalities. To eliminate these drawbacks, we propose a down-up sampling-based AVSR model (Hourglass-AVSR) to enjoy high efficiency and performance, whose time length is scaled during the intermediate processing, resembling an hourglass. Firstly, we propose a context and residual aware video upsampling approach to improve the recognition performance, which utilizes contextual information from visual representations and captures residual information between adjacent video frames. Secondly, we introduce a visual-audio alignment approach during the upsampling by explicitly incorporating boundary constraint loss. Besides, we propose a cross-layer attention fusion to capture the modality dependencies within each visual encoder layer. Experiments conducted on the MISP-AVSR dataset reveal that our proposed Hourglass-AVSR model outperforms ASR model by 12.9% and 20.8% relative concatenated minimum permutation character error rate (cpCER) reduction on far-field and middle-field test sets, respectively. Moreover, compared to other state-of-the-art AVSR models, our model exhibits the highest improvement in cpCER for the visual module. Furthermore, on the benefit of our down-up sampling approach, Hourglass-AVSR model reduces 54.2% overall computation costs with minor performance degradation.

{{</citation>}}


### (151/169) SEF-VC: Speaker Embedding Free Zero-Shot Voice Conversion with Cross Attention (Junjie Li et al., 2023)

{{<citation>}}

Junjie Li, Yiwei Guo, Xie Chen, Kai Yu. (2023)  
**SEF-VC: Speaker Embedding Free Zero-Shot Voice Conversion with Cross Attention**  

---
Primary Category: cs.SD  
Categories: cs-CL, cs-SD, cs.SD, eess-AS  
Keywords: Attention, BERT, Embedding, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2312.08676v1)  

---


**ABSTRACT**  
Zero-shot voice conversion (VC) aims to transfer the source speaker timbre to arbitrary unseen target speaker timbre, while keeping the linguistic content unchanged. Although the voice of generated speech can be controlled by providing the speaker embedding of the target speaker, the speaker similarity still lags behind the ground truth recordings. In this paper, we propose SEF-VC, a speaker embedding free voice conversion model, which is designed to learn and incorporate speaker timbre from reference speech via a powerful position-agnostic cross-attention mechanism, and then reconstruct waveform from HuBERT semantic tokens in a non-autoregressive manner. The concise design of SEF-VC enhances its training stability and voice conversion performance. Objective and subjective evaluations demonstrate the superiority of SEF-VC to generate high-quality speech with better similarity to target reference than strong zero-shot VC baselines, even for very short reference speeches.

{{</citation>}}


## cs.SE (2)



### (152/169) Towards Trustworthy AI Software Development Assistance (Daniel Maninger et al., 2023)

{{<citation>}}

Daniel Maninger, Krishna Narasimhan, Mira Mezini. (2023)  
**Towards Trustworthy AI Software Development Assistance**  

---
Primary Category: cs.SE  
Categories: cs-LG, cs-SE, cs.SE  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.09126v1)  

---


**ABSTRACT**  
It is expected that in the near future, AI software development assistants will play an important role in the software industry. However, current software development assistants tend to be unreliable, often producing incorrect, unsafe, or low-quality code. We seek to resolve these issues by introducing a holistic architecture for constructing, training, and using trustworthy AI software development assistants. In the center of the architecture, there is a foundational LLM trained on datasets representative of real-world coding scenarios and complex software architectures, and fine-tuned on code quality criteria beyond correctness. The LLM will make use of graph-based code representations for advanced semantic comprehension. We envision a knowledge graph integrated into the system to provide up-to-date background knowledge and to enable the assistant to provide appropriate explanations. Finally, a modular framework for constrained decoding will ensure that certain guarantees (e.g., for correctness and security) hold for the generated code.

{{</citation>}}


### (153/169) Entity-Augmented Code Generation (Anton Shapkin et al., 2023)

{{<citation>}}

Anton Shapkin, Denis Litvinov, Timofey Bryksin. (2023)  
**Entity-Augmented Code Generation**  

---
Primary Category: cs.SE  
Categories: cs-LG, cs-SE, cs.SE  
Keywords: Question Answering  
[Paper Link](http://arxiv.org/abs/2312.08976v1)  

---


**ABSTRACT**  
The current state-of-the-art large language models (LLMs) are effective in generating high-quality text and encapsulating a broad spectrum of world knowledge. However, these models often hallucinate during generation and are not designed to utilize external information sources. To enable requests to the external knowledge bases, also called knowledge grounding, retrieval-augmented LLMs were introduced. For now, their applications have largely involved Open Domain Question Answering, Abstractive Question Answering, and such. In this paper, we broaden the scope of retrieval-augmented LLMs by venturing into a new task - code generation using external entities. For this task, we collect and publish a new dataset for project-level code generation, where the model should reuse functions defined in the project during generation. As we show, existing retrieval-augmented LLMs fail to assign relevance scores between similar entity names, and to mitigate it, they expand entity names with description context and append it to the input. In practice, due to the limited context size they can not accommodate the indefinitely large context of the whole project. To solve this issue, we propose a novel end-to-end trainable architecture with an scalable entity retriever injected directly into the LLM decoder. We demonstrate that our model can outperform common baselines in several scenarios, including project-level code generation, as well as Bash and SQL scripting.

{{</citation>}}


## cs.DC (1)



### (154/169) MRL-PoS: A Multi-agent Reinforcement Learning based Proof of Stake Consensus Algorithm for Blockchain (Tariqul Islam et al., 2023)

{{<citation>}}

Tariqul Islam, Faisal Haque Bappy, Tarannum Shaila Zaman, Md Sajidul Islam Sajid, Mir Mehedi Ahsan Pritom. (2023)  
**MRL-PoS: A Multi-agent Reinforcement Learning based Proof of Stake Consensus Algorithm for Blockchain**  

---
Primary Category: cs.DC  
Categories: cs-CR, cs-DC, cs.DC  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.09123v1)  

---


**ABSTRACT**  
The core of a blockchain network is its consensus algorithm. Starting with the Proof-of-Work, there have been various versions of consensus algorithms, such as Proof-of-Stake (PoS), Proof-of-Authority (PoA), and Practical Byzantine Fault Tolerance (PBFT). Each of these algorithms focuses on different aspects to ensure efficient and reliable processing of transactions. Blockchain operates in a decentralized manner where there is no central authority and the network is composed of diverse users. This openness creates the potential for malicious nodes to disrupt the network in various ways. Therefore, it is crucial to embed a mechanism within the blockchain network to constantly monitor, identify, and eliminate these malicious nodes. However, there is no one-size-fits-all mechanism to identify all malicious nodes. Hence, the dynamic adaptability of the blockchain network is important to maintain security and reliability at all times. This paper introduces MRL-PoS, a Proof-of-Stake consensus algorithm based on multi-agent reinforcement learning. MRL-PoS employs reinforcement learning for dynamically adjusting to the behavior of all users. It incorporates a system of rewards and penalties to eliminate malicious nodes and incentivize honest ones. Additionally, MRL-PoS has the capability to learn and respond to new malicious tactics by continually training its agents.

{{</citation>}}


## cs.NE (1)



### (155/169) Language Modeling on a SpiNNaker 2 Neuromorphic Chip (Khaleelulla Khan Nazeer et al., 2023)

{{<citation>}}

Khaleelulla Khan Nazeer, Mark Schöne, Rishav Mukherji, Christian Mayr, David Kappel, Anand Subramoney. (2023)  
**Language Modeling on a SpiNNaker 2 Neuromorphic Chip**  

---
Primary Category: cs.NE  
Categories: cs-CL, cs-ET, cs-LG, cs-NE, cs.NE  
Keywords: LSTM, Language Model  
[Paper Link](http://arxiv.org/abs/2312.09084v1)  

---


**ABSTRACT**  
As large language models continue to scale in size rapidly, so too does the computational power required to run them. Event-based networks on neuromorphic devices offer a potential way to reduce energy consumption for inference significantly. However, to date, most event-based networks that can run on neuromorphic hardware, including spiking neural networks (SNNs), have not achieved task performance even on par with LSTM models for language modeling. As a result, language modeling on neuromorphic devices has seemed a distant prospect. In this work, we demonstrate the first-ever implementation of a language model on a neuromorphic device - specifically the SpiNNaker 2 chip - based on a recently published event-based architecture called the EGRU. SpiNNaker 2 is a many-core neuromorphic chip designed for large-scale asynchronous processing, while the EGRU is architected to leverage such hardware efficiently while maintaining competitive task performance. This implementation marks the first time a neuromorphic language model matches LSTMs, setting the stage for taking task performance to the level of large language models. We also demonstrate results on a gesture recognition task based on inputs from a DVS camera. Overall, our results showcase the feasibility of this neuro-inspired neural network in hardware, highlighting significant gains versus conventional hardware in energy efficiency for the common use case of single batch inference.

{{</citation>}}


## eess.AS (5)



### (156/169) Fusion of Audio and Visual Embeddings for Sound Event Localization and Detection (Davide Berghi et al., 2023)

{{<citation>}}

Davide Berghi, Peipei Wu, Jinzheng Zhao, Wenwu Wang, Philip J. B. Jackson. (2023)  
**Fusion of Audio and Visual Embeddings for Sound Event Localization and Detection**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess-IV, eess.AS  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2312.09034v1)  

---


**ABSTRACT**  
Sound event localization and detection (SELD) combines two subtasks: sound event detection (SED) and direction of arrival (DOA) estimation. SELD is usually tackled as an audio-only problem, but visual information has been recently included. Few audio-visual (AV)-SELD works have been published and most employ vision via face/object bounding boxes, or human pose keypoints. In contrast, we explore the integration of audio and visual feature embeddings extracted with pre-trained deep networks. For the visual modality, we tested ResNet50 and Inflated 3D ConvNet (I3D). Our comparison of AV fusion methods includes the AV-Conformer and Cross-Modal Attentive Fusion (CMAF) model. Our best models outperform the DCASE 2023 Task3 audio-only and AV baselines by a wide margin on the development set of the STARSS23 dataset, making them competitive amongst state-of-the-art results of the AV challenge, without model ensembling, heavy data augmentation, or prediction post-processing. Such techniques and further pre-training could be applied as next steps to improve performance.

{{</citation>}}


### (157/169) Attention-Guided Adaptation for Code-Switching Speech Recognition (Bobbi Aditya et al., 2023)

{{<citation>}}

Bobbi Aditya, Mahdin Rohmatillah, Liang-Hsuan Tai, Jen-Tzung Chien. (2023)  
**Attention-Guided Adaptation for Code-Switching Speech Recognition**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Attention, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2312.08856v1)  

---


**ABSTRACT**  
The prevalence of the powerful multilingual models, such as Whisper, has significantly advanced the researches on speech recognition. However, these models often struggle with handling the code-switching setting, which is essential in multilingual speech recognition. Recent studies have attempted to address this setting by separating the modules for different languages to ensure distinct latent representations for languages. Some other methods considered the switching mechanism based on language identification. In this study, a new attention-guided adaptation is proposed to conduct parameter-efficient learning for bilingual ASR. This method selects those attention heads in a model which closely express language identities and then guided those heads to be correctly attended with their corresponding languages. The experiments on the Mandarin-English code-switching speech corpus show that the proposed approach achieves a 14.2% mixed error rate, surpassing state-of-the-art method, where only 5.6% additional parameters over Whisper are trained.

{{</citation>}}


### (158/169) Towards Automatic Data Augmentation for Disordered Speech Recognition (Zengrui Jin et al., 2023)

{{<citation>}}

Zengrui Jin, Xurong Xie, Tianzi Wang, Mengzhe Geng, Jiajun Deng, Guinan Li, Shujie Hu, Xunying Liu. (2023)  
**Towards Automatic Data Augmentation for Disordered Speech Recognition**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Augmentation, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2312.08641v1)  

---


**ABSTRACT**  
Automatic recognition of disordered speech remains a highly challenging task to date due to data scarcity. This paper presents a reinforcement learning (RL) based on-the-fly data augmentation approach for training state-of-the-art PyChain TDNN and end-to-end Conformer ASR systems on such data. The handcrafted temporal and spectral mask operations in the standard SpecAugment method that are task and system dependent, together with additionally introduced minimum and maximum cut-offs of these time-frequency masks, are now automatically learned using an RNN-based policy controller and tightly integrated with ASR system training. Experiments on the UASpeech corpus suggest the proposed RL-based data augmentation approach consistently produced performance superior or comparable that obtained using expert or handcrafted SpecAugment policies. Our RL auto-augmented PyChain TDNN system produced an overall WER of 28.79% on the UASpeech test set of 16 dysarthric speakers.

{{</citation>}}


### (159/169) Scalable Ensemble-based Detection Method against Adversarial Attacks for speaker verification (Haibin Wu et al., 2023)

{{<citation>}}

Haibin Wu, Heng-Cheng Kuo, Yu Tsao, Hung-yi Lee. (2023)  
**Scalable Ensemble-based Detection Method against Adversarial Attacks for speaker verification**  

---
Primary Category: eess.AS  
Categories: cs-LG, cs-SD, eess-AS, eess.AS  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2312.08622v1)  

---


**ABSTRACT**  
Automatic speaker verification (ASV) is highly susceptible to adversarial attacks. Purification modules are usually adopted as a pre-processing to mitigate adversarial noise. However, they are commonly implemented across diverse experimental settings, rendering direct comparisons challenging. This paper comprehensively compares mainstream purification techniques in a unified framework. We find these methods often face a trade-off between user experience and security, as they struggle to simultaneously maintain genuine sample performance and reduce adversarial perturbations. To address this challenge, some efforts have extended purification modules to encompass detection capabilities, aiming to alleviate the trade-off. However, advanced purification modules will always come into the stage to surpass previous detection method. As a result, we further propose an easy-to-follow ensemble approach that integrates advanced purification modules for detection, achieving state-of-the-art (SOTA) performance in countering adversarial noise. Our ensemble method has great potential due to its compatibility with future advanced purification techniques.

{{</citation>}}


### (160/169) NeXt-TDNN: Modernizing Multi-Scale Temporal Convolution Backbone for Speaker Verification (Hyun-Jun Heo et al., 2023)

{{<citation>}}

Hyun-Jun Heo, Ui-Hyeop Shin, Ran Lee, YoungJu Cheon, Hyung-Min Park. (2023)  
**NeXt-TDNN: Modernizing Multi-Scale Temporal Convolution Backbone for Speaker Verification**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Speaker Verification, Transformer  
[Paper Link](http://arxiv.org/abs/2312.08603v2)  

---


**ABSTRACT**  
In speaker verification, ECAPA-TDNN has shown remarkable improvement by utilizing one-dimensional(1D) Res2Net block and squeeze-and-excitation(SE) module, along with multi-layer feature aggregation (MFA). Meanwhile, in vision tasks, ConvNet structures have been modernized by referring to Transformer, resulting in improved performance. In this paper, we present an improved block design for TDNN in speaker verification. Inspired by recent ConvNet structures, we replace the SE-Res2Net block in ECAPA-TDNN with a novel 1D two-step multi-scale ConvNeXt block, which we call TS-ConvNeXt. The TS-ConvNeXt block is constructed using two separated sub-modules: a temporal multi-scale convolution (MSC) and a frame-wise feed-forward network (FFN). This two-step design allows for flexible capturing of inter-frame and intra-frame contexts. Additionally, we introduce global response normalization (GRN) for the FFN modules to enable more selective feature propagation, similar to the SE module in ECAPA-TDNN. Experimental results demonstrate that NeXt-TDNN, with a modernized backbone block, significantly improved performance in speaker verification tasks while reducing parameter size and inference time. We have released our code for future studies.

{{</citation>}}


## cs.CG (1)



### (161/169) On the Complexity of Simultaneous Geometric Embedding for Edge-Disjoint Graphs (Benedikt Künzel et al., 2023)

{{<citation>}}

Benedikt Künzel, Jonathan Rollin. (2023)  
**On the Complexity of Simultaneous Geometric Embedding for Edge-Disjoint Graphs**  

---
Primary Category: cs.CG  
Categories: cs-CG, cs-DM, cs.CG, math-CO  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2312.09025v1)  

---


**ABSTRACT**  
Simultaneous Geometric Embedding (SGE) asks whether, for a given collection of graphs on the same vertex set V, there is an embedding of V in the plane that admits a crossing-free drawing with straightline edges for each of the given graphs. It is known that SGE is $\exists\mathbb{R}$-complete, that is, the problem is polynomially equivalent to deciding whether a system of polynomial equations and inequalities with integer coefficients has a real solution. We prove that SGE remains $\exists\mathbb{R}$-complete for edge-disjoint input graphs, that is, for collections of graphs without so-called public edges.   As an intermediate result, we prove that it is $\exists\mathbb{R}$-complete to decide whether a directional walk without repeating edges is realizable. Here, a directional walk consists of a sequence of not-necessarily distinct vertices (a walk) and a function prescribing for each inner position whether the walk shall turn left or shall turn right. A directional walk is realizable, if there is an embedding of its vertices in the plane such that the embedded walk turns according to the given directions. Previously it was known that realization is $\exists\mathbb{R}$-complete to decide for directional walks repeating each edge at most 336 times.   This answers two questions posed by Schaefer ["On the Complexity of Some Geometric Problems With Fixed Parameters", JGAA 2021].

{{</citation>}}


## eess.IV (3)



### (162/169) Brain Diffuser with Hierarchical Transformer for MCI Causality Analysis (Qiankun Zuo et al., 2023)

{{<citation>}}

Qiankun Zuo, Ling Chen, Shuqiang Wang. (2023)  
**Brain Diffuser with Hierarchical Transformer for MCI Causality Analysis**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV, q-bio-NC  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.09022v1)  

---


**ABSTRACT**  
Effective connectivity estimation plays a crucial role in understanding the interactions and information flow between different brain regions. However, the functional time series used for estimating effective connentivity is derived from certain software, which may lead to large computing errors because of different parameter settings and degrade the ability to model complex causal relationships between brain regions. In this paper, a brain diffuser with hierarchical transformer (BDHT) is proposed to estimate effective connectivity for mild cognitive impairment (MCI) analysis. To our best knowledge, the proposed brain diffuer is the first generative model to apply diffusion models in the application of generating and analyzing multimodal brain networks. Specifically, the BDHT leverages the structural connectivity to guide the reverse processes in an efficient way. It makes the denoising process more reliable and guarantees effective connectivity estimation accuracy. To improve denoising quality, the hierarchical denoising transformer is designed to learn multi-scale features in topological space. Furthermore, the GraphConFormer block can concentrate on both global and adjacent connectivity information. By stacking the multi-head attention and graph convolutional network, the proposed model enhances structure-function complementarity and improves the ability in noise estimation. Experimental evaluations of the denoising diffusion model demonstrate its effectiveness in estimating effective connectivity. The method achieves superior performance in terms of accuracy and robustness compared to existing approaches. It can captures both unidirectal and bidirectional interactions between brain regions, providing a comprehensive understanding of the brain's information processing mechanisms.

{{</citation>}}


### (163/169) MCANet: Medical Image Segmentation with Multi-Scale Cross-Axis Attention (Hao Shao et al., 2023)

{{<citation>}}

Hao Shao, Quansheng Zeng, Qibin Hou, Jufeng Yang. (2023)  
**MCANet: Medical Image Segmentation with Multi-Scale Cross-Axis Attention**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2312.08866v1)  

---


**ABSTRACT**  
Efficiently capturing multi-scale information and building long-range dependencies among pixels are essential for medical image segmentation because of the various sizes and shapes of the lesion regions or organs. In this paper, we present Multi-scale Cross-axis Attention (MCA) to solve the above challenging issues based on the efficient axial attention. Instead of simply connecting axial attention along the horizontal and vertical directions sequentially, we propose to calculate dual cross attentions between two parallel axial attentions to capture global information better. To process the significant variations of lesion regions or organs in individual sizes and shapes, we also use multiple convolutions of strip-shape kernels with different kernel sizes in each axial attention path to improve the efficiency of the proposed MCA in encoding spatial information. We build the proposed MCA upon the MSCAN backbone, yielding our network, termed MCANet. Our MCANet with only 4M+ parameters performs even better than most previous works with heavy backbones (e.g., Swin Transformer) on four challenging tasks, including skin lesion segmentation, nuclei segmentation, abdominal multi-organ segmentation, and polyp segmentation. Code is available at https:// github.com/ haoshao-nku/ medical seg.git.

{{</citation>}}


### (164/169) RankDVQA-mini: Knowledge Distillation-Driven Deep Video Quality Assessment (Chen Feng et al., 2023)

{{<citation>}}

Chen Feng, Duolikun Danier, Haoran Wang, Fan Zhang, David Bull. (2023)  
**RankDVQA-mini: Knowledge Distillation-Driven Deep Video Quality Assessment**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Knowledge Distillation, QA  
[Paper Link](http://arxiv.org/abs/2312.08864v1)  

---


**ABSTRACT**  
Deep learning-based video quality assessment (deep VQA) has demonstrated significant potential in surpassing conventional metrics, with promising improvements in terms of correlation with human perception. However, the practical deployment of such deep VQA models is often limited due to their high computational complexity and large memory requirements. To address this issue, we aim to significantly reduce the model size and runtime of one of the state-of-the-art deep VQA methods, RankDVQA, by employing a two-phase workflow that integrates pruning-driven model compression with multi-level knowledge distillation. The resulting lightweight quality metric, RankDVQA-mini, requires less than 10% of the model parameters compared to its full version (14% in terms of FLOPs), while still retaining a quality prediction performance that is superior to most existing deep VQA methods. The source code of the RankDVQA-mini has been released at https://chenfeng-bristol.github.io/RankDVQA-mini/ for public evaluation.

{{</citation>}}


## cs.IT (2)



### (165/169) LLMind: Orchestrating AI and IoT with LLMs for Complex Task Execution (Hongwei Cui et al., 2023)

{{<citation>}}

Hongwei Cui, Yuyang Du, Qun Yang, Yulin Shao, Soung Chang Liew. (2023)  
**LLMind: Orchestrating AI and IoT with LLMs for Complex Task Execution**  

---
Primary Category: cs.IT  
Categories: cs-AI, cs-IT, cs.IT, math-IT  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.09007v1)  

---


**ABSTRACT**  
In this article, we introduce LLMind, an innovative AI framework that utilizes large language models (LLMs) as a central orchestrator. The framework integrates LLMs with domain-specific AI modules, enabling IoT devices to collaborate effectively in executing complex tasks. The LLM performs planning and generates control scripts using a reliable and precise language-code transformation approach based on finite state machines (FSMs). The LLM engages in natural conversations with users, employing role-playing techniques to generate contextually appropriate responses. Additionally, users can interact easily with the AI agent via a user-friendly social media platform. The framework also incorporates semantic analysis and response optimization techniques to enhance speed and effectiveness. Ultimately, this framework is designed not only to innovate IoT device control and enrich user experiences but also to foster an intelligent and integrated IoT device ecosystem that evolves and becomes more sophisticated through continuing user and machine interactions.

{{</citation>}}


### (166/169) Localization with Reconfigurable Intelligent Surface: An Active Sensing Approach (Zhongze Zhang et al., 2023)

{{<citation>}}

Zhongze Zhang, Tao Jiang, Wei Yu. (2023)  
**Localization with Reconfigurable Intelligent Surface: An Active Sensing Approach**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs-LG, cs.IT, eess-SP, math-IT  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2312.09002v2)  

---


**ABSTRACT**  
This paper addresses an uplink localization problem in which a base station (BS) aims to locate a remote user with the help of reconfigurable intelligent surfaces (RISs). We propose a strategy in which the user transmits pilots sequentially and the BS adaptively adjusts the sensing vectors, including the BS beamforming vector and multiple RIS reflection coefficients based on the observations already made, to eventually produce an estimated user position. This is a challenging active sensing problem for which finding an optimal solution involves searching through a complicated functional space whose dimension increases with the number of measurements. We show that the long short-term memory (LSTM) network can be used to exploit the latent temporal correlation between measurements to automatically construct scalable state vectors. Subsequently, the state vector is mapped to the sensing vectors for the next time frame via a deep neural network (DNN). A final DNN is used to map the state vector to the estimated user position. Numerical result illustrates the advantage of the active sensing design as compared to non-active sensing methods. The proposed solution produces interpretable results and is generalizable in the number of sensing stages. Remarkably, we show that a network with one BS and multiple RISs can outperform a comparable setting with multiple BSs.

{{</citation>}}


## physics.med-ph (1)



### (167/169) Speeding up Photoacoustic Imaging using Diffusion Models (Irem Loc et al., 2023)

{{<citation>}}

Irem Loc, Mehmet Burcin Unlu. (2023)  
**Speeding up Photoacoustic Imaging using Diffusion Models**  

---
Primary Category: physics.med-ph  
Categories: cs-AI, physics-med-ph, physics.med-ph  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.08834v1)  

---


**ABSTRACT**  
Background: Photoacoustic Microscopy (PAM) integrates optical and acoustic imaging, offering enhanced penetration depth for detecting optical-absorbing components in tissues. Nonetheless, challenges arise in scanning large areas with high spatial resolution. With speed limitations imposed by laser pulse repetition rates, the potential role of computational methods is highlighted in accelerating PAM imaging. Purpose: We are proposing a novel and highly adaptable DiffPam algorithm that utilizes diffusion models for speeding up the photoacoustic imaging process. Method: We leveraged a diffusion model trained exclusively on natural images, comparing its performance with an in-domain trained U-Net model using a dataset focused on PAM images of mice brain microvasculature. Results: Our findings indicate that DiffPam achieves comparable performance to a dedicated U-Net model, without the need for a large dataset or training a deep learning model. The study also introduces the efficacy of shortened diffusion processes for reducing computing time without compromising accuracy. Conclusion: This study underscores the significance of DiffPam as a practical algorithm for reconstructing undersampled PAM images, particularly for researchers with limited AI expertise and computational resources.

{{</citation>}}


## cs.MA (1)



### (168/169) From Centralized to Self-Supervised: Pursuing Realistic Multi-Agent Reinforcement Learning (Violet Xiang et al., 2023)

{{<citation>}}

Violet Xiang, Logan Cross, Jan-Philipp Fränken, Nick Haber. (2023)  
**From Centralized to Self-Supervised: Pursuing Realistic Multi-Agent Reinforcement Learning**  

---
Primary Category: cs.MA  
Categories: cs-MA, cs.MA  
Keywords: Reinforcement Learning, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2312.08662v1)  

---


**ABSTRACT**  
In real-world environments, autonomous agents rely on their egocentric observations. They must learn adaptive strategies to interact with others who possess mixed motivations, discernible only through visible cues. Several Multi-Agent Reinforcement Learning (MARL) methods adopt centralized approaches that involve either centralized training or reward-sharing, often violating the realistic ways in which living organisms, like animals or humans, process information and interact. MARL strategies deploying decentralized training with intrinsic motivation offer a self-supervised approach, enable agents to develop flexible social strategies through the interaction of autonomous agents. However, by contrasting the self-supervised and centralized methods, we reveal that populations trained with reward-sharing methods surpass those using self-supervised methods in a mixed-motive environment. We link this superiority to specialized role emergence and an agent's expertise in its role. Interestingly, this gap shrinks in pure-motive settings, emphasizing the need for evaluations in more complex, realistic environments (mixed-motive). Our preliminary results suggest a gap in population performance that can be closed by improving self-supervised methods and thereby pushing MARL closer to real-world readiness.

{{</citation>}}


## cs.PL (1)



### (169/169) RTLCoder: Outperforming GPT-3.5 in Design RTL Generation with Our Open-Source Dataset and Lightweight Solution (Shang Liu et al., 2023)

{{<citation>}}

Shang Liu, Wenji Fang, Yao Lu, Qijun Zhang, Hongce Zhang, Zhiyao Xie. (2023)  
**RTLCoder: Outperforming GPT-3.5 in Design RTL Generation with Our Open-Source Dataset and Lightweight Solution**  

---
Primary Category: cs.PL  
Categories: cs-AR, cs-PL, cs.PL  
Keywords: ChatGPT, GPT, GPT-3.5  
[Paper Link](http://arxiv.org/abs/2312.08617v1)  

---


**ABSTRACT**  
The automatic generation of RTL code (e.g., Verilog) using natural language instructions and large language models (LLMs) has attracted significant research interest recently. However, most existing approaches heavily rely on commercial LLMs such as ChatGPT, while open-source LLMs tailored for this specific design generation task exhibit notably inferior performance. The absence of high-quality open-source solutions restricts the flexibility and data privacy of this emerging technique. In this study, we present a new customized LLM solution with a modest parameter count of only 7B, achieving better performance than GPT-3.5 on two representative benchmarks for RTL code generation. This remarkable balance between accuracy and efficiency is made possible by leveraging our new RTL code dataset and a customized LLM algorithm, both of which will be made fully open-source. Furthermore, we have successfully quantized our LLM to 4-bit with a total size of 4GB, enabling it to function on a single laptop with only slight performance degradation. This efficiency allows the RTL generator to serve as a local assistant for engineers, ensuring all design privacy concerns are addressed.

{{</citation>}}
