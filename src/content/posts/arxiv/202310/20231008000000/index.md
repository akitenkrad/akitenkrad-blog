---
draft: false
title: "arXiv @ 2023.10.08"
date: 2023-10-08
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.10.08"
    identifier: arxiv_20231008
    parent: 202310_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.LG (16)](#cslg-16)
- [cs.CV (14)](#cscv-14)
- [cs.CL (11)](#cscl-11)
- [cs.AI (3)](#csai-3)
- [cs.CR (2)](#cscr-2)
- [stat.ML (2)](#statml-2)
- [cs.NI (2)](#csni-2)
- [math.OC (1)](#mathoc-1)
- [q-fin.CP (1)](#q-fincp-1)
- [cs.SE (1)](#csse-1)
- [eess.SY (2)](#eesssy-2)
- [cs.RO (1)](#csro-1)
- [cs.IR (2)](#csir-2)
- [eess.IV (1)](#eessiv-1)
- [cs.SI (1)](#cssi-1)
- [cs.CY (2)](#cscy-2)
- [cs.DS (2)](#csds-2)
- [cs.GR (1)](#csgr-1)
- [cs.SD (4)](#cssd-4)
- [cs.HC (1)](#cshc-1)

## cs.LG (16)



### (1/70) Functional Interpolation for Relative Positions Improves Long Context Transformers (Shanda Li et al., 2023)

{{<citation>}}

Shanda Li, Chong You, Guru Guruganesh, Joshua Ainslie, Santiago Ontanon, Manzil Zaheer, Sumit Sanghai, Yiming Yang, Sanjiv Kumar, Srinadh Bhojanapalli. (2023)  
**Functional Interpolation for Relative Positions Improves Long Context Transformers**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: T5, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.04418v1)  

---


**ABSTRACT**  
Preventing the performance decay of Transformers on inputs longer than those used for training has been an important challenge in extending the context length of these models. Though the Transformer architecture has fundamentally no limits on the input sequence lengths it can process, the choice of position encoding used during training can limit the performance of these models on longer inputs. We propose a novel functional relative position encoding with progressive interpolation, FIRE, to improve Transformer generalization to longer contexts. We theoretically prove that this can represent some of the popular relative position encodings, such as T5's RPE, Alibi, and Kerple. We next empirically show that FIRE models have better generalization to longer contexts on both zero-shot language modeling and long text benchmarks.

{{</citation>}}


### (2/70) Beyond Uniform Sampling: Offline Reinforcement Learning with Imbalanced Datasets (Zhang-Wei Hong et al., 2023)

{{<citation>}}

Zhang-Wei Hong, Aviral Kumar, Sathwik Karnik, Abhishek Bhandwaldar, Akash Srivastava, Joni Pajarinen, Romain Laroche, Abhishek Gupta, Pulkit Agrawal. (2023)  
**Beyond Uniform Sampling: Offline Reinforcement Learning with Imbalanced Datasets**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.04413v1)  

---


**ABSTRACT**  
Offline policy learning is aimed at learning decision-making policies using existing datasets of trajectories without collecting additional data. The primary motivation for using reinforcement learning (RL) instead of supervised learning techniques such as behavior cloning is to find a policy that achieves a higher average return than the trajectories constituting the dataset. However, we empirically find that when a dataset is dominated by suboptimal trajectories, state-of-the-art offline RL algorithms do not substantially improve over the average return of trajectories in the dataset. We argue this is due to an assumption made by current offline RL algorithms of staying close to the trajectories in the dataset. If the dataset primarily consists of sub-optimal trajectories, this assumption forces the policy to mimic the suboptimal actions. We overcome this issue by proposing a sampling strategy that enables the policy to only be constrained to ``good data" rather than all actions in the dataset (i.e., uniform sampling). We present a realization of the sampling strategy and an algorithm that can be used as a plug-and-play module in standard offline RL algorithms. Our evaluation demonstrates significant performance gains in 72 imbalanced datasets, D4RL dataset, and across three different offline RL algorithms. Code is available at https://github.com/Improbable-AI/dw-offline-rl.

{{</citation>}}


### (3/70) On the Embedding Collapse when Scaling up Recommendation Models (Xingzhuo Guo et al., 2023)

{{<citation>}}

Xingzhuo Guo, Junwei Pan, Ximei Wang, Baixu Chen, Jie Jiang, Mingsheng Long. (2023)  
**On the Embedding Collapse when Scaling up Recommendation Models**  

---
Primary Category: cs.LG  
Categories: cs-IR, cs-LG, cs.LG  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2310.04400v1)  

---


**ABSTRACT**  
Recent advances in deep foundation models have led to a promising trend of developing large recommendation models to leverage vast amounts of available data. However, we experiment to scale up existing recommendation models and observe that the enlarged models do not improve satisfactorily. In this context, we investigate the embedding layers of enlarged models and identify a phenomenon of embedding collapse, which ultimately hinders scalability, wherein the embedding matrix tends to reside in a low-dimensional subspace. Through empirical and theoretical analysis, we demonstrate that the feature interaction module specific to recommendation models has a two-sided effect. On the one hand, the interaction restricts embedding learning when interacting with collapsed embeddings, exacerbating the collapse issue. On the other hand, feature interaction is crucial in mitigating the fitting of spurious features, thereby improving scalability. Based on this analysis, we propose a simple yet effective multi-embedding design incorporating embedding-set-specific interaction modules to capture diverse patterns and reduce collapse. Extensive experiments demonstrate that this proposed design provides consistent scalability for various recommendation models.

{{</citation>}}


### (4/70) Exploiting Transformer Activation Sparsity with Dynamic Inference (Mikołaj Piórczyński et al., 2023)

{{<citation>}}

Mikołaj Piórczyński, Filip Szatkowski, Klaudia Bałazy, Bartosz Wójcik. (2023)  
**Exploiting Transformer Activation Sparsity with Dynamic Inference**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: BERT, Transformer  
[Paper Link](http://arxiv.org/abs/2310.04361v1)  

---


**ABSTRACT**  
Transformer models, despite their impressive performance, often face practical limitations due to their high computational requirements. At the same time, previous studies have revealed significant activation sparsity in these models, indicating the presence of redundant computations. In this paper, we propose Dynamic Sparsified Transformer Inference (DSTI), a method that radically reduces the inference cost of Transformer models by enforcing activation sparsity and subsequently transforming a dense model into its sparse Mixture of Experts (MoE) version. We demonstrate that it is possible to train small gating networks that successfully predict the relative contribution of each expert during inference. Furthermore, we introduce a mechanism that dynamically determines the number of executed experts individually for each token. DSTI can be applied to any Transformer-based architecture and has negligible impact on the accuracy. For the BERT-base classification model, we reduce inference cost by almost 60%.

{{</citation>}}


### (5/70) A Language-Agent Approach to Formal Theorem-Proving (Amitayush Thakur et al., 2023)

{{<citation>}}

Amitayush Thakur, Yeming Wen, Swarat Chaudhuri. (2023)  
**A Language-Agent Approach to Formal Theorem-Proving**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-LO, cs-PL, cs.LG  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2310.04353v1)  

---


**ABSTRACT**  
Language agents, which use a large language model (LLM) capable of in-context learning to interact with an external environment, have recently emerged as a promising approach to control tasks. We present the first language-agent approach to formal theorem-proving. Our method, COPRA, uses a high-capacity, black-box LLM (GPT-4) as part of a policy for a stateful backtracking search. During the search, the policy can select proof tactics and retrieve lemmas and definitions from an external database. Each selected tactic is executed in the underlying proof framework, and the execution feedback is used to build the prompt for the next policy invocation. The search also tracks selected information from its history and uses it to reduce hallucinations and unnecessary LLM queries.   We evaluate COPRA on the miniF2F benchmark for Lean and a set of Coq tasks from the Compcert project. On these benchmarks, COPRA is significantly better than one-shot invocations of GPT-4, as well as state-of-the-art models fine-tuned on proof data, at finding correct proofs quickly.

{{</citation>}}


### (6/70) Saliency-Guided Hidden Associative Replay for Continual Learning (Guangji Bai et al., 2023)

{{<citation>}}

Guangji Bai, Qilong Zhao, Xiaoyang Jiang, Yifei Zhang, Liang Zhao. (2023)  
**Saliency-Guided Hidden Associative Replay for Continual Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.04334v1)  

---


**ABSTRACT**  
Continual Learning is a burgeoning domain in next-generation AI, focusing on training neural networks over a sequence of tasks akin to human learning. While CL provides an edge over traditional supervised learning, its central challenge remains to counteract catastrophic forgetting and ensure the retention of prior tasks during subsequent learning. Amongst various strategies to tackle this, replay based methods have emerged as preeminent, echoing biological memory mechanisms. However, these methods are memory intensive, often preserving entire data samples, an approach inconsistent with humans selective memory retention of salient experiences. While some recent works have explored the storage of only significant portions of data in episodic memory, the inherent nature of partial data necessitates innovative retrieval mechanisms. Current solutions, like inpainting, approximate full data reconstruction from partial cues, a method that diverges from genuine human memory processes. Addressing these nuances, this paper presents the Saliency Guided Hidden Associative Replay for Continual Learning. This novel framework synergizes associative memory with replay-based strategies. SHARC primarily archives salient data segments via sparse memory encoding. Importantly, by harnessing associative memory paradigms, it introduces a content focused memory retrieval mechanism, promising swift and near-perfect recall, bringing CL a step closer to authentic human memory processes. Extensive experimental results demonstrate the effectiveness of our proposed method for various continual learning tasks.

{{</citation>}}


### (7/70) Adjustable Robust Reinforcement Learning for Online 3D Bin Packing (Yuxin Pan et al., 2023)

{{<citation>}}

Yuxin Pan, Yize Chen, Fangzhen Lin. (2023)  
**Adjustable Robust Reinforcement Learning for Online 3D Bin Packing**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.04323v1)  

---


**ABSTRACT**  
Designing effective policies for the online 3D bin packing problem (3D-BPP) has been a long-standing challenge, primarily due to the unpredictable nature of incoming box sequences and stringent physical constraints. While current deep reinforcement learning (DRL) methods for online 3D-BPP have shown promising results in optimizing average performance over an underlying box sequence distribution, they often fail in real-world settings where some worst-case scenarios can materialize. Standard robust DRL algorithms tend to overly prioritize optimizing the worst-case performance at the expense of performance under normal problem instance distribution. To address these issues, we first introduce a permutation-based attacker to investigate the practical robustness of both DRL-based and heuristic methods proposed for solving online 3D-BPP. Then, we propose an adjustable robust reinforcement learning (AR2L) framework that allows efficient adjustment of robustness weights to achieve the desired balance of the policy's performance in average and worst-case environments. Specifically, we formulate the objective function as a weighted sum of expected and worst-case returns, and derive the lower performance bound by relating to the return under a mixture dynamics. To realize this lower bound, we adopt an iterative procedure that searches for the associated mixture dynamics and improves the corresponding policy. We integrate this procedure into two popular robust adversarial algorithms to develop the exact and approximate AR2L algorithms. Experiments demonstrate that AR2L is versatile in the sense that it improves policy robustness while maintaining an acceptable level of performance for the nominal case.

{{</citation>}}


### (8/70) Comparing Auxiliary Tasks for Learning Representations for Reinforcement Learning (Moritz Lange et al., 2023)

{{<citation>}}

Moritz Lange, Noah Krystiniak, Raphael C. Engelhardt, Wolfgang Konen, Laurenz Wiskott. (2023)  
**Comparing Auxiliary Tasks for Learning Representations for Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.04241v1)  

---


**ABSTRACT**  
Learning state representations has gained steady popularity in reinforcement learning (RL) due to its potential to improve both sample efficiency and returns on many environments. A straightforward and efficient method is to generate representations with a distinct neural network trained on an auxiliary task, i.e. a task that differs from the actual RL task. While a whole range of such auxiliary tasks has been proposed in the literature, a comparison on typical continuous control benchmark environments is computationally expensive and has, to the best of our knowledge, not been performed before. This paper presents such a comparison of common auxiliary tasks, based on hundreds of agents trained with state-of-the-art off-policy RL algorithms. We compare possible improvements in both sample efficiency and returns for environments ranging from simple pendulum to a complex simulated robotics task. Our findings show that representation learning with auxiliary tasks is beneficial for environments of higher dimension and complexity, and that learning environment dynamics is preferable to predicting rewards. We believe these insights will enable other researchers to make more informed decisions on how to utilize representation learning for their specific problem.

{{</citation>}}


### (9/70) Non-Redundant Graph Neural Networks with Improved Expressiveness (Franka Bause et al., 2023)

{{<citation>}}

Franka Bause, Samir Moustafa, Johannes Langguth, Wilfried N. Gansterer, Nils M. Kriege. (2023)  
**Non-Redundant Graph Neural Networks with Improved Expressiveness**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.04190v1)  

---


**ABSTRACT**  
Message passing graph neural networks iteratively compute node embeddings by aggregating messages from all neighbors. This procedure can be viewed as a neural variant of the Weisfeiler-Leman method, which limits their expressive power. Moreover, oversmoothing and oversquashing restrict the number of layers these networks can effectively utilize. The repeated exchange and encoding of identical information in message passing amplifies oversquashing. We propose a novel aggregation scheme based on neighborhood trees, which allows for controlling the redundancy by pruning branches of the unfolding trees underlying standard message passing. We prove that reducing redundancy improves expressivity and experimentally show that it alleviates oversquashing. We investigate the interaction between redundancy in message passing and redundancy in computation and propose a compact representation of neighborhood trees, from which we compute node and graph embeddings via a neural tree canonization technique. Our method is provably more expressive than the Weisfeiler-Leman method, less susceptible to oversquashing than message passing neural networks, and provides high classification accuracy on widely-used benchmark datasets.

{{</citation>}}


### (10/70) Introducing the Attribution Stability Indicator: a Measure for Time Series XAI Attributions (Udo Schlegel et al., 2023)

{{<citation>}}

Udo Schlegel, Daniel A. Keim. (2023)  
**Introducing the Attribution Stability Indicator: a Measure for Time Series XAI Attributions**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI, Time Series  
[Paper Link](http://arxiv.org/abs/2310.04178v1)  

---


**ABSTRACT**  
Given the increasing amount and general complexity of time series data in domains such as finance, weather forecasting, and healthcare, there is a growing need for state-of-the-art performance models that can provide interpretable insights into underlying patterns and relationships. Attribution techniques enable the extraction of explanations from time series models to gain insights but are hard to evaluate for their robustness and trustworthiness. We propose the Attribution Stability Indicator (ASI), a measure to incorporate robustness and trustworthiness as properties of attribution techniques for time series into account. We extend a perturbation analysis with correlations of the original time series to the perturbed instance and the attributions to include wanted properties in the measure. We demonstrate the wanted properties based on an analysis of the attributions in a dimension-reduced space and the ASI scores distribution over three whole time series classification datasets.

{{</citation>}}


### (11/70) Dynamic Relation-Attentive Graph Neural Networks for Fraud Detection (Heehyeon Kim et al., 2023)

{{<citation>}}

Heehyeon Kim, Jinhyeok Choi, Joyce Jiyoung Whang. (2023)  
**Dynamic Relation-Attentive Graph Neural Networks for Fraud Detection**  

---
Primary Category: cs.LG  
Categories: I-2, cs-AI, cs-CR, cs-LG, cs.LG  
Keywords: Fraud Detection, GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.04171v1)  

---


**ABSTRACT**  
Fraud detection aims to discover fraudsters deceiving other users by, for example, leaving fake reviews or making abnormal transactions. Graph-based fraud detection methods consider this task as a classification problem with two classes: frauds or normal. We address this problem using Graph Neural Networks (GNNs) by proposing a dynamic relation-attentive aggregation mechanism. Based on the observation that many real-world graphs include different types of relations, we propose to learn a node representation per relation and aggregate the node representations using a learnable attention function that assigns a different attention coefficient to each relation. Furthermore, we combine the node representations from different layers to consider both the local and global structures of a target node, which is beneficial to improving the performance of fraud detection on graphs with heterophily. By employing dynamic graph attention in all the aggregation processes, our method adaptively computes the attention coefficients for each node. Experimental results show that our method, DRAG, outperforms state-of-the-art fraud detection methods on real-world benchmark datasets.

{{</citation>}}


### (12/70) Reinforcement Learning with Fast and Forgetful Memory (Steven Morad et al., 2023)

{{<citation>}}

Steven Morad, Ryan Kortvelesy, Stephan Liwicki, Amanda Prorok. (2023)  
**Reinforcement Learning with Fast and Forgetful Memory**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.04128v1)  

---


**ABSTRACT**  
Nearly all real world tasks are inherently partially observable, necessitating the use of memory in Reinforcement Learning (RL). Most model-free approaches summarize the trajectory into a latent Markov state using memory models borrowed from Supervised Learning (SL), even though RL tends to exhibit different training and efficiency characteristics. Addressing this discrepancy, we introduce Fast and Forgetful Memory, an algorithm-agnostic memory model designed specifically for RL. Our approach constrains the model search space via strong structural priors inspired by computational psychology. It is a drop-in replacement for recurrent neural networks (RNNs) in recurrent RL algorithms, achieving greater reward than RNNs across various recurrent benchmarks and algorithms without changing any hyperparameters. Moreover, Fast and Forgetful Memory exhibits training speeds two orders of magnitude faster than RNNs, attributed to its logarithmic time and linear space complexity. Our implementation is available at https://github.com/proroklab/ffm.

{{</citation>}}


### (13/70) AUTOPARLLM: GNN-Guided Automatic Code Parallelization using Large Language Models (Quazi Ishtiaque Mahmud et al., 2023)

{{<citation>}}

Quazi Ishtiaque Mahmud, Ali TehraniJamsaz, Hung D Phan, Nesreen K. Ahmed, Ali Jannesari. (2023)  
**AUTOPARLLM: GNN-Guided Automatic Code Parallelization using Large Language Models**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Language Model  
[Paper Link](http://arxiv.org/abs/2310.04047v1)  

---


**ABSTRACT**  
Parallelizing sequentially written programs is a challenging task. Even experienced developers need to spend considerable time finding parallelism opportunities and then actually writing parallel versions of sequentially written programs. To address this issue, we present AUTOPARLLM, a framework for automatically discovering parallelism and generating the parallel version of the sequentially written program. Our framework consists of two major components: i) a heterogeneous Graph Neural Network (GNN) based parallelism discovery and parallel pattern detection module, and ii) an LLM-based code generator to generate the parallel counterpart of the sequential programs. We use the GNN to learn the flow-aware characteristics of the programs to identify parallel regions in sequential programs and then construct an enhanced prompt using the GNN's results for the LLM-based generator to finally produce the parallel counterparts of the sequential programs. We evaluate AUTOPARLLM on 11 applications of 2 well-known benchmark suites: NAS Parallel Benchmark and Rodinia Benchmark. Our results show that AUTOPARLLM is indeed effective in improving the state-of-the-art LLM-based models for the task of parallel code generation in terms of multiple code generation metrics. AUTOPARLLM also improves the average runtime of the parallel code generated by the state-of-the-art LLMs by as high as 3.4% and 2.9% for the NAS Parallel Benchmark and Rodinia Benchmark respectively. Additionally, to overcome the issue that well-known metrics for translation evaluation have not been optimized to evaluate the quality of the generated parallel code, we propose OMPScore for evaluating the quality of the generated code. We show that OMPScore exhibits a better correlation with human judgment than existing metrics, measured by up to 75% improvement of Spearman correlation.

{{</citation>}}


### (14/70) PGraphDTA: Improving Drug Target Interaction Prediction using Protein Language Models and Contact Maps (Rakesh Bal et al., 2023)

{{<citation>}}

Rakesh Bal, Yijia Xiao, Wei Wang. (2023)  
**PGraphDTA: Improving Drug Target Interaction Prediction using Protein Language Models and Contact Maps**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, q-bio-QM  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.04017v1)  

---


**ABSTRACT**  
Developing and discovering new drugs is a complex and resource-intensive endeavor that often involves substantial costs, time investment, and safety concerns. A key aspect of drug discovery involves identifying novel drug-target (DT) interactions. Existing computational methods for predicting DT interactions have primarily focused on binary classification tasks, aiming to determine whether a DT pair interacts or not. However, protein-ligand interactions exhibit a continuum of binding strengths, known as binding affinity, presenting a persistent challenge for accurate prediction. In this study, we investigate various techniques employed in Drug Target Interaction (DTI) prediction and propose novel enhancements to enhance their performance. Our approaches include the integration of Protein Language Models (PLMs) and the incorporation of Contact Map information as an inductive bias within current models. Through extensive experimentation, we demonstrate that our proposed approaches outperform the baseline models considered in this study, presenting a compelling case for further development in this direction. We anticipate that the insights gained from this work will significantly narrow the search space for potential drugs targeting specific proteins, thereby accelerating drug discovery. Code and data for PGraphDTA are available at https://anonymous.4open.science/r/PGraphDTA.

{{</citation>}}


### (15/70) Perfect Alignment May be Poisonous to Graph Contrastive Learning (Jingyu Liu et al., 2023)

{{<citation>}}

Jingyu Liu, Huayi Tang, Yong Liu. (2023)  
**Perfect Alignment May be Poisonous to Graph Contrastive Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2310.03977v1)  

---


**ABSTRACT**  
Graph Contrastive Learning (GCL) aims to learn node representations by aligning positive pairs and separating negative ones. However, limited research has been conducted on the inner law behind specific augmentations used in graph-based learning. What kind of augmentation will help downstream performance, how does contrastive learning actually influence downstream tasks, and why the magnitude of augmentation matters? This paper seeks to address these questions by establishing a connection between augmentation and downstream performance, as well as by investigating the generalization of contrastive learning. Our findings reveal that GCL contributes to downstream tasks mainly by separating different classes rather than gathering nodes of the same class. So perfect alignment and augmentation overlap which draw all intra-class samples the same can not explain the success of contrastive learning. Then in order to comprehend how augmentation aids the contrastive learning process, we conduct further investigations into its generalization, finding that perfect alignment that draw positive pair the same could help contrastive loss but is poisonous to generalization, on the contrary, imperfect alignment enhances the model's generalization ability. We analyse the result by information theory and graph spectrum theory respectively, and propose two simple but effective methods to verify the theories. The two methods could be easily applied to various GCL algorithms and extensive experiments are conducted to prove its effectiveness.

{{</citation>}}


### (16/70) Understanding prompt engineering may not require rethinking generalization (Victor Akinwande et al., 2023)

{{<citation>}}

Victor Akinwande, Yiding Jiang, Dylan Sam, J. Zico Kolter. (2023)  
**Understanding prompt engineering may not require rethinking generalization**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2310.03957v1)  

---


**ABSTRACT**  
Zero-shot learning in prompted vision-language models, the practice of crafting prompts to build classifiers without an explicit training process, has achieved impressive performance in many settings. This success presents a seemingly surprising observation: these methods suffer relatively little from overfitting, i.e., when a prompt is manually engineered to achieve low error on a given training set (thus rendering the method no longer actually zero-shot), the approach still performs well on held-out test data. In this paper, we show that we can explain such performance well via recourse to classical PAC-Bayes bounds. Specifically, we show that the discrete nature of prompts, combined with a PAC-Bayes prior given by a language model, results in generalization bounds that are remarkably tight by the standards of the literature: for instance, the generalization bound of an ImageNet classifier is often within a few percentage points of the true test error. We demonstrate empirically that this holds for existing handcrafted prompts and prompts generated through simple greedy search. Furthermore, the resulting bound is well-suited for model selection: the models with the best bound typically also have the best test performance. This work thus provides a possible justification for the widespread practice of prompt engineering, even if it seems that such methods could potentially overfit the training data.

{{</citation>}}


## cs.CV (14)



### (17/70) FedConv: Enhancing Convolutional Neural Networks for Handling Data Heterogeneity in Federated Learning (Peiran Xu et al., 2023)

{{<citation>}}

Peiran Xu, Zeyu Wang, Jieru Mei, Liangqiong Qu, Alan Yuille, Cihang Xie, Yuyin Zhou. (2023)  
**FedConv: Enhancing Convolutional Neural Networks for Handling Data Heterogeneity in Federated Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.04412v1)  

---


**ABSTRACT**  
Federated learning (FL) is an emerging paradigm in machine learning, where a shared model is collaboratively learned using data from multiple devices to mitigate the risk of data leakage. While recent studies posit that Vision Transformer (ViT) outperforms Convolutional Neural Networks (CNNs) in addressing data heterogeneity in FL, the specific architectural components that underpin this advantage have yet to be elucidated. In this paper, we systematically investigate the impact of different architectural elements, such as activation functions and normalization layers, on the performance within heterogeneous FL. Through rigorous empirical analyses, we are able to offer the first-of-its-kind general guidance on micro-architecture design principles for heterogeneous FL.   Intriguingly, our findings indicate that with strategic architectural modifications, pure CNNs can achieve a level of robustness that either matches or even exceeds that of ViTs when handling heterogeneous data clients in FL. Additionally, our approach is compatible with existing FL techniques and delivers state-of-the-art solutions across a broad spectrum of FL benchmarks. The code is publicly available at https://github.com/UCSC-VLAA/FedConv

{{</citation>}}


### (18/70) Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference (Simian Luo et al., 2023)

{{<citation>}}

Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, Hang Zhao. (2023)  
**Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.04378v1)  

---


**ABSTRACT**  
Latent Diffusion models (LDMs) have achieved remarkable results in synthesizing high-resolution images. However, the iterative sampling process is computationally intensive and leads to slow generation. Inspired by Consistency Models (song et al.), we propose Latent Consistency Models (LCMs), enabling swift inference with minimal steps on any pre-trained LDMs, including Stable Diffusion (rombach et al). Viewing the guided reverse diffusion process as solving an augmented probability flow ODE (PF-ODE), LCMs are designed to directly predict the solution of such ODE in latent space, mitigating the need for numerous iterations and allowing rapid, high-fidelity sampling. Efficiently distilled from pre-trained classifier-free guided diffusion models, a high-quality 768 x 768 2~4-step LCM takes only 32 A100 GPU hours for training. Furthermore, we introduce Latent Consistency Fine-tuning (LCF), a novel method that is tailored for fine-tuning LCMs on customized image datasets. Evaluation on the LAION-5B-Aesthetics dataset demonstrates that LCMs achieve state-of-the-art text-to-image generation performance with few-step inference. Project Page: https://latent-consistency-models.github.io/

{{</citation>}}


### (19/70) Towards A Robust Group-level Emotion Recognition via Uncertainty-Aware Learning (Qing Zhu et al., 2023)

{{<citation>}}

Qing Zhu, Qirong Mao, Jialin Zhang, Xiaohua Huang, Wenming Zheng. (2023)  
**Towards A Robust Group-level Emotion Recognition via Uncertainty-Aware Learning**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2310.04306v1)  

---


**ABSTRACT**  
Group-level emotion recognition (GER) is an inseparable part of human behavior analysis, aiming to recognize an overall emotion in a multi-person scene. However, the existing methods are devoted to combing diverse emotion cues while ignoring the inherent uncertainties under unconstrained environments, such as congestion and occlusion occurring within a group. Additionally, since only group-level labels are available, inconsistent emotion predictions among individuals in one group can confuse the network. In this paper, we propose an uncertainty-aware learning (UAL) method to extract more robust representations for GER. By explicitly modeling the uncertainty of each individual, we utilize stochastic embedding drawn from a Gaussian distribution instead of deterministic point embedding. This representation captures the probabilities of different emotions and generates diverse predictions through this stochasticity during the inference stage. Furthermore, uncertainty-sensitive scores are adaptively assigned as the fusion weights of individuals' face within each group. Moreover, we develop an image enhancement module to enhance the model's robustness against severe noise. The overall three-branch model, encompassing face, object, and scene component, is guided by a proportional-weighted fusion strategy and integrates the proposed uncertainty-aware method to produce the final group-level output. Experimental results demonstrate the effectiveness and generalization ability of our method across three widely used databases.

{{</citation>}}


### (20/70) Collaborative Camouflaged Object Detection: A Large-Scale Dataset and Benchmark (Cong Zhang et al., 2023)

{{<citation>}}

Cong Zhang, Hongbo Bi, Tian-Zhu Xiang, Ranwan Wu, Jinghui Tong, Xiufang Wang. (2023)  
**Collaborative Camouflaged Object Detection: A Large-Scale Dataset and Benchmark**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2310.04253v1)  

---


**ABSTRACT**  
In this paper, we provide a comprehensive study on a new task called collaborative camouflaged object detection (CoCOD), which aims to simultaneously detect camouflaged objects with the same properties from a group of relevant images. To this end, we meticulously construct the first large-scale dataset, termed CoCOD8K, which consists of 8,528 high-quality and elaborately selected images with object mask annotations, covering 5 superclasses and 70 subclasses. The dataset spans a wide range of natural and artificial camouflage scenes with diverse object appearances and backgrounds, making it a very challenging dataset for CoCOD. Besides, we propose the first baseline model for CoCOD, named bilateral-branch network (BBNet), which explores and aggregates co-camouflaged cues within a single image and between images within a group, respectively, for accurate camouflaged object detection in given images. This is implemented by an inter-image collaborative feature exploration (CFE) module, an intra-image object feature search (OFS) module, and a local-global refinement (LGR) module. We benchmark 18 state-of-the-art models, including 12 COD algorithms and 6 CoSOD algorithms, on the proposed CoCOD8K dataset under 5 widely used evaluation metrics. Extensive experiments demonstrate the effectiveness of the proposed method and the significantly superior performance compared to other competitors. We hope that our proposed dataset and model will boost growth in the COD community. The dataset, model, and results will be available at: https://github.com/zc199823/BBNet--CoCOD.

{{</citation>}}


### (21/70) Degradation-Aware Self-Attention Based Transformer for Blind Image Super-Resolution (Qingguo Liu et al., 2023)

{{<citation>}}

Qingguo Liu, Pan Gao, Kang Han, Ningzhong Liu, Wei Xiang. (2023)  
**Degradation-Aware Self-Attention Based Transformer for Blind Image Super-Resolution**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Self-Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2310.04180v1)  

---


**ABSTRACT**  
Compared to CNN-based methods, Transformer-based methods achieve impressive image restoration outcomes due to their abilities to model remote dependencies. However, how to apply Transformer-based methods to the field of blind super-resolution (SR) and further make an SR network adaptive to degradation information is still an open problem. In this paper, we propose a new degradation-aware self-attention-based Transformer model, where we incorporate contrastive learning into the Transformer network for learning the degradation representations of input images with unknown noise. In particular, we integrate both CNN and Transformer components into the SR network, where we first use the CNN modulated by the degradation information to extract local features, and then employ the degradation-aware Transformer to extract global semantic features. We apply our proposed model to several popular large-scale benchmark datasets for testing, and achieve the state-of-the-art performance compared to existing methods. In particular, our method yields a PSNR of 32.43 dB on the Urban100 dataset at $\times$2 scale, 0.94 dB higher than DASR, and 26.62 dB on the Urban100 dataset at $\times$4 scale, 0.26 dB improvement over KDSR, setting a new benchmark in this area. Source code is available at: https://github.com/I2-Multimedia-Lab/DSAT/tree/main.

{{</citation>}}


### (22/70) Entropic Score metric: Decoupling Topology and Size in Training-free NAS (Niccolò Cavagnero et al., 2023)

{{<citation>}}

Niccolò Cavagnero, Luca Robbiano, Francesca Pistilli, Barbara Caputo, Giuseppe Averta. (2023)  
**Entropic Score metric: Decoupling Topology and Size in Training-free NAS**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: ImageNet, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.04179v1)  

---


**ABSTRACT**  
Neural Networks design is a complex and often daunting task, particularly for resource-constrained scenarios typical of mobile-sized models. Neural Architecture Search is a promising approach to automate this process, but existing competitive methods require large training time and computational resources to generate accurate models. To overcome these limits, this paper contributes with: i) a novel training-free metric, named Entropic Score, to estimate model expressivity through the aggregated element-wise entropy of its activations; ii) a cyclic search algorithm to separately yet synergistically search model size and topology. Entropic Score shows remarkable ability in searching for the topology of the network, and a proper combination with LogSynflow, to search for model size, yields superior capability to completely design high-performance Hybrid Transformers for edge applications in less than 1 GPU hour, resulting in the fastest and most accurate NAS method for ImageNet classification.

{{</citation>}}


### (23/70) Self-Supervised Neuron Segmentation with Multi-Agent Reinforcement Learning (Yinda Chen et al., 2023)

{{<citation>}}

Yinda Chen, Wei Huang, Shenglong Zhou, Qi Chen, Zhiwei Xiong. (2023)  
**Self-Supervised Neuron Segmentation with Multi-Agent Reinforcement Learning**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Reinforcement Learning, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2310.04148v1)  

---


**ABSTRACT**  
The performance of existing supervised neuron segmentation methods is highly dependent on the number of accurate annotations, especially when applied to large scale electron microscopy (EM) data. By extracting semantic information from unlabeled data, self-supervised methods can improve the performance of downstream tasks, among which the mask image model (MIM) has been widely used due to its simplicity and effectiveness in recovering original information from masked images. However, due to the high degree of structural locality in EM images, as well as the existence of considerable noise, many voxels contain little discriminative information, making MIM pretraining inefficient on the neuron segmentation task. To overcome this challenge, we propose a decision-based MIM that utilizes reinforcement learning (RL) to automatically search for optimal image masking ratio and masking strategy. Due to the vast exploration space, using single-agent RL for voxel prediction is impractical. Therefore, we treat each input patch as an agent with a shared behavior policy, allowing for multi-agent collaboration. Furthermore, this multi-agent model can capture dependencies between voxels, which is beneficial for the downstream segmentation task. Experiments conducted on representative EM datasets demonstrate that our approach has a significant advantage over alternative self-supervised methods on the task of neuron segmentation. Code is available at \url{https://github.com/ydchen0806/dbMiM}.

{{</citation>}}


### (24/70) TiC: Exploring Vision Transformer in Convolution (Song Zhang et al., 2023)

{{<citation>}}

Song Zhang, Qingzhong Wang, Jiang Bian, Haoyi Xiong. (2023)  
**TiC: Exploring Vision Transformer in Convolution**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, ImageNet, Self-Attention, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.04134v1)  

---


**ABSTRACT**  
While models derived from Vision Transformers (ViTs) have been phonemically surging, pre-trained models cannot seamlessly adapt to arbitrary resolution images without altering the architecture and configuration, such as sampling the positional encoding, limiting their flexibility for various vision tasks. For instance, the Segment Anything Model (SAM) based on ViT-Huge requires all input images to be resized to 1024$\times$1024. To overcome this limitation, we propose the Multi-Head Self-Attention Convolution (MSA-Conv) that incorporates Self-Attention within generalized convolutions, including standard, dilated, and depthwise ones. Enabling transformers to handle images of varying sizes without retraining or rescaling, the use of MSA-Conv further reduces computational costs compared to global attention in ViT, which grows costly as image size increases. Later, we present the Vision Transformer in Convolution (TiC) as a proof of concept for image classification with MSA-Conv, where two capacity enhancing strategies, namely Multi-Directional Cyclic Shifted Mechanism and Inter-Pooling Mechanism, have been proposed, through establishing long-distance connections between tokens and enlarging the effective receptive field. Extensive experiments have been carried out to validate the overall effectiveness of TiC. Additionally, ablation studies confirm the performance improvement made by MSA-Conv and the two capacity enhancing strategies separately. Note that our proposal aims at studying an alternative to the global attention used in ViT, while MSA-Conv meets our goal by making TiC comparable to state-of-the-art on ImageNet-1K. Code will be released at https://github.com/zs670980918/MSA-Conv.

{{</citation>}}


### (25/70) Automated 3D Segmentation of Kidneys and Tumors in MICCAI KiTS 2023 Challenge (Andriy Myronenko et al., 2023)

{{<citation>}}

Andriy Myronenko, Dong Yang, Yufan He, Daguang Xu. (2023)  
**Automated 3D Segmentation of Kidneys and Tumors in MICCAI KiTS 2023 Challenge**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.04110v1)  

---


**ABSTRACT**  
Kidney and Kidney Tumor Segmentation Challenge (KiTS) 2023 offers a platform for researchers to compare their solutions to segmentation from 3D CT. In this work, we describe our submission to the challenge using automated segmentation of Auto3DSeg available in MONAI. Our solution achieves the average dice of 0.835 and surface dice of 0.723, which ranks first and wins the KiTS 2023 challenge.

{{</citation>}}


### (26/70) ClusVPR: Efficient Visual Place Recognition with Clustering-based Weighted Transformer (Yifan Xu et al., 2023)

{{<citation>}}

Yifan Xu, Pourya Shamsolmoali, Jie Yang. (2023)  
**ClusVPR: Efficient Visual Place Recognition with Clustering-based Weighted Transformer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.04099v1)  

---


**ABSTRACT**  
Visual place recognition (VPR) is a highly challenging task that has a wide range of applications, including robot navigation and self-driving vehicles. VPR is particularly difficult due to the presence of duplicate regions and the lack of attention to small objects in complex scenes, resulting in recognition deviations. In this paper, we present ClusVPR, a novel approach that tackles the specific issues of redundant information in duplicate regions and representations of small objects. Different from existing methods that rely on Convolutional Neural Networks (CNNs) for feature map generation, ClusVPR introduces a unique paradigm called Clustering-based Weighted Transformer Network (CWTNet). CWTNet leverages the power of clustering-based weighted feature maps and integrates global dependencies to effectively address visual deviations encountered in large-scale VPR problems. We also introduce the optimized-VLAD (OptLAD) layer that significantly reduces the number of parameters and enhances model efficiency. This layer is specifically designed to aggregate the information obtained from scale-wise image patches. Additionally, our pyramid self-supervised strategy focuses on extracting representative and diverse information from scale-wise image patches instead of entire images, which is crucial for capturing representative and diverse information in VPR. Extensive experiments on four VPR datasets show our model's superior performance compared to existing models while being less complex.

{{</citation>}}


### (27/70) A Deeply Supervised Semantic Segmentation Method Based on GAN (Wei Zhao et al., 2023)

{{<citation>}}

Wei Zhao, Qiyu Wei, Zeng Zeng. (2023)  
**A Deeply Supervised Semantic Segmentation Method Based on GAN**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CE, cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2310.04081v1)  

---


**ABSTRACT**  
In recent years, the field of intelligent transportation has witnessed rapid advancements, driven by the increasing demand for automation and efficiency in transportation systems. Traffic safety, one of the tasks integral to intelligent transport systems, requires accurately identifying and locating various road elements, such as road cracks, lanes, and traffic signs. Semantic segmentation plays a pivotal role in achieving this task, as it enables the partition of images into meaningful regions with accurate boundaries. In this study, we propose an improved semantic segmentation model that combines the strengths of adversarial learning with state-of-the-art semantic segmentation techniques. The proposed model integrates a generative adversarial network (GAN) framework into the traditional semantic segmentation model, enhancing the model's performance in capturing complex and subtle features in transportation images. The effectiveness of our approach is demonstrated by a significant boost in performance on the road crack dataset compared to the existing methods, \textit{i.e.,} SEGAN. This improvement can be attributed to the synergistic effect of adversarial learning and semantic segmentation, which leads to a more refined and accurate representation of road structures and conditions. The enhanced model not only contributes to better detection of road cracks but also to a wide range of applications in intelligent transportation, such as traffic sign recognition, vehicle detection, and lane segmentation.

{{</citation>}}


### (28/70) Excision and Recovery: Enhancing Surface Anomaly Detection with Attention-based Single Deterministic Masking (YeongHyeon Park et al., 2023)

{{<citation>}}

YeongHyeon Park, Sungho Kang, Myung Jin Kim, Yeonho Lee, Juneho Yi. (2023)  
**Excision and Recovery: Enhancing Surface Anomaly Detection with Attention-based Single Deterministic Masking**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV, eess-IV  
Keywords: Anomaly Detection, Attention  
[Paper Link](http://arxiv.org/abs/2310.04010v1)  

---


**ABSTRACT**  
Anomaly detection (AD) in surface inspection is an essential yet challenging task in manufacturing due to the quantity imbalance problem of scarce abnormal data. To overcome the above, a reconstruction encoder-decoder (ED) such as autoencoder or U-Net which is trained with only anomaly-free samples is widely adopted, in the hope that unseen abnormals should yield a larger reconstruction error than normal. Over the past years, researches on self-supervised reconstruction-by-inpainting have been reported. They mask out suspected defective regions for inpainting in order to make them invisible to the reconstruction ED to deliberately cause inaccurate reconstruction for abnormals. However, their limitation is multiple random masking to cover the whole input image due to defective regions not being known in advance. We propose a novel reconstruction-by-inpainting method dubbed Excision and Recovery (EAR) that features single deterministic masking. For this, we exploit a pre-trained spatial attention model to predict potential suspected defective regions that should be masked out. We also employ a variant of U-Net as our ED to further limit the reconstruction ability of the U-Net model for abnormals, in which skip connections of different layers can be selectively disabled. In the training phase, all the skip connections are switched on to fully take the benefits from the U-Net architecture. In contrast, for inferencing, we only keep deeper skip connections with shallower connections off. We validate the effectiveness of EAR using an MNIST pre-trained attention for a commonly used surface AD dataset, KolektorSDD2. The experimental results show that EAR achieves both better AD performance and higher throughput than state-of-the-art methods. We expect that the proposed EAR model can be widely adopted as training and inference strategies for AD purposes.

{{</citation>}}


### (29/70) CUPre: Cross-domain Unsupervised Pre-training for Few-Shot Cell Segmentation (Weibin Liao et al., 2023)

{{<citation>}}

Weibin Liao, Xuhong Li, Qingzhong Wang, Yanwu Xu, Zhaozheng Yin, Haoyi Xiong. (2023)  
**CUPre: Cross-domain Unsupervised Pre-training for Few-Shot Cell Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2310.03981v1)  

---


**ABSTRACT**  
While pre-training on object detection tasks, such as Common Objects in Contexts (COCO) [1], could significantly boost the performance of cell segmentation, it still consumes on massive fine-annotated cell images [2] with bounding boxes, masks, and cell types for every cell in every image, to fine-tune the pre-trained model. To lower the cost of annotation, this work considers the problem of pre-training DNN models for few-shot cell segmentation, where massive unlabeled cell images are available but only a small proportion is annotated. Hereby, we propose Cross-domain Unsupervised Pre-training, namely CUPre, transferring the capability of object detection and instance segmentation for common visual objects (learned from COCO) to the visual domain of cells using unlabeled images. Given a standard COCO pre-trained network with backbone, neck, and head modules, CUPre adopts an alternate multi-task pre-training (AMT2) procedure with two sub-tasks -- in every iteration of pre-training, AMT2 first trains the backbone with cell images from multiple cell datasets via unsupervised momentum contrastive learning (MoCo) [3], and then trains the whole model with vanilla COCO datasets via instance segmentation. After pre-training, CUPre fine-tunes the whole model on the cell segmentation task using a few annotated images. We carry out extensive experiments to evaluate CUPre using LIVECell [2] and BBBC038 [4] datasets in few-shot instance segmentation settings. The experiment shows that CUPre can outperform existing pre-training methods, achieving the highest average precision (AP) for few-shot cell segmentation and detection.

{{</citation>}}


### (30/70) Sub-token ViT Embedding via Stochastic Resonance Transformers (Dong Lao et al., 2023)

{{<citation>}}

Dong Lao, Yangchao Wu, Tian Yu Liu, Alex Wong, Stefano Soatto. (2023)  
**Sub-token ViT Embedding via Stochastic Resonance Transformers**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Embedding, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.03967v1)  

---


**ABSTRACT**  
We discover the presence of quantization artifacts in Vision Transformers (ViTs), which arise due to the image tokenization step inherent in these architectures. These artifacts result in coarsely quantized features, which negatively impact performance, especially on downstream dense prediction tasks. We present a zero-shot method to improve how pre-trained ViTs handle spatial quantization. In particular, we propose to ensemble the features obtained from perturbing input images via sub-token spatial translations, inspired by Stochastic Resonance, a method traditionally applied to climate dynamics and signal processing. We term our method ``Stochastic Resonance Transformer" (SRT), which we show can effectively super-resolve features of pre-trained ViTs, capturing more of the local fine-grained structures that might otherwise be neglected as a result of tokenization. SRT can be applied at any layer, on any task, and does not require any fine-tuning. The advantage of the former is evident when applied to monocular depth prediction, where we show that ensembling model outputs are detrimental while applying SRT on intermediate ViT features outperforms the baseline models by an average of 4.7% and 14.9% on the RMSE and RMSE-log metrics across three different architectures. When applied to semi-supervised video object segmentation, SRT also improves over the baseline models uniformly across all metrics, and by an average of 2.4% in F&J score. We further show that these quantization artifacts can be attenuated to some extent via self-distillation. On the unsupervised salient region segmentation, SRT improves upon the base model by an average of 2.1% on the maxF metric. Finally, despite operating purely on pixel-level features, SRT generalizes to non-dense prediction tasks such as image retrieval and object discovery, yielding consistent improvements of up to 2.6% and 1.0% respectively.

{{</citation>}}


## cs.CL (11)



### (31/70) RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation (Fangyuan Xu et al., 2023)

{{<citation>}}

Fangyuan Xu, Weijia Shi, Eunsol Choi. (2023)  
**RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2310.04408v1)  

---


**ABSTRACT**  
Retrieving documents and prepending them in-context at inference time improves performance of language model (LMs) on a wide range of tasks. However, these documents, often spanning hundreds of words, make inference substantially more expensive. We propose compressing the retrieved documents into textual summaries prior to in-context integration. This not only reduces the computational costs but also relieves the burden of LMs to identify relevant information in long retrieved documents. We present two compressors -- an extractive compressor which selects useful sentences from retrieved documents and an abstractive compressor which generates summaries by synthesizing information from multiple documents. Both compressors are trained to improve LMs' performance on end tasks when the generated summaries are prepended to the LMs' input, while keeping the summary concise.If the retrieved documents are irrelevant to the input or offer no additional information to LM, our compressor can return an empty string, implementing selective augmentation.We evaluate our approach on language modeling task and open domain question answering task. We achieve a compression rate of as low as 6% with minimal loss in performance for both tasks, significantly outperforming the off-the-shelf summarization models. We show that our compressors trained for one LM can transfer to other LMs on the language modeling task and provide summaries largely faithful to the retrieved documents.

{{</citation>}}


### (32/70) Policy-Gradient Training of Language Models for Ranking (Ge Gao et al., 2023)

{{<citation>}}

Ge Gao, Jonathan D. Chang, Claire Cardie, Kianté Brantley, Thorsten Joachim. (2023)  
**Policy-Gradient Training of Language Models for Ranking**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-IR, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.04407v1)  

---


**ABSTRACT**  
Text retrieval plays a crucial role in incorporating factual knowledge for decision making into language processing pipelines, ranging from chat-based web search to question answering systems. Current state-of-the-art text retrieval models leverage pre-trained large language models (LLMs) to achieve competitive performance, but training LLM-based retrievers via typical contrastive losses requires intricate heuristics, including selecting hard negatives and using additional supervision as learning signals. This reliance on heuristics stems from the fact that the contrastive loss itself is heuristic and does not directly optimize the downstream metrics of decision quality at the end of the processing pipeline. To address this issue, we introduce Neural PG-RANK, a novel training algorithm that learns to rank by instantiating a LLM as a Plackett-Luce ranking policy. Neural PG-RANK provides a principled method for end-to-end training of retrieval models as part of larger decision systems via policy gradient, with little reliance on complex heuristics, and it effectively unifies the training objective with downstream decision-making quality. We conduct extensive experiments on various text retrieval benchmarks. The results demonstrate that when the training objective aligns with the evaluation setup, Neural PG-RANK yields remarkable in-domain performance improvement, with substantial out-of-domain generalization to some critical datasets employed in downstream question answering tasks.

{{</citation>}}


### (33/70) Large-Scale Korean Text Dataset for Classifying Biased Speech in Real-World Online Services (Dasol Choi et al., 2023)

{{<citation>}}

Dasol Choi, Jooyoung Song, Eunsun Lee, Jinwoo Seo, Heejune Park, Dongbin Na. (2023)  
**Large-Scale Korean Text Dataset for Classifying Biased Speech in Real-World Online Services**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, Bias  
[Paper Link](http://arxiv.org/abs/2310.04313v1)  

---


**ABSTRACT**  
With the growth of online services, the need for advanced text classification algorithms, such as sentiment analysis and biased text detection, has become increasingly evident. The anonymous nature of online services often leads to the presence of biased and harmful language, posing challenges to maintaining the health of online communities. This phenomenon is especially relevant in South Korea, where large-scale hate speech detection algorithms have not yet been broadly explored. In this paper, we introduce a new comprehensive, large-scale dataset collected from a well-known South Korean SNS platform. Our proposed dataset provides annotations including (1) Preferences, (2) Profanities, and (3) Nine types of Bias for the text samples, enabling multi-task learning for simultaneous classification of user-generated texts. Leveraging state-of-the-art BERT-based language models, our approach surpasses human-level accuracy across diverse classification tasks, as measured by various metrics. Beyond academic contributions, our work can provide practical solutions for real-world hate speech and bias mitigation, contributing directly to the improvement of online community health. Our work provides a robust foundation for future research aiming to improve the quality of online discourse and foster societal well-being. All source codes and datasets are publicly accessible at https://github.com/Dasol-Choi/KoMultiText.

{{</citation>}}


### (34/70) A Comprehensive Evaluation of Large Language Models on Benchmark Biomedical Text Processing Tasks (Israt Jahan et al., 2023)

{{<citation>}}

Israt Jahan, Md Tahmid Rahman Laskar, Chun Peng, Jimmy Huang. (2023)  
**A Comprehensive Evaluation of Large Language Models on Benchmark Biomedical Text Processing Tasks**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.04270v1)  

---


**ABSTRACT**  
Recently, Large Language Models (LLM) have demonstrated impressive capability to solve a wide range of tasks. However, despite their success across various tasks, no prior work has investigated their capability in the biomedical domain yet. To this end, this paper aims to evaluate the performance of LLMs on benchmark biomedical tasks. For this purpose, we conduct a comprehensive evaluation of 4 popular LLMs in 6 diverse biomedical tasks across 26 datasets. To the best of our knowledge, this is the first work that conducts an extensive evaluation and comparison of various LLMs in the biomedical domain. Interestingly, we find based on our evaluation that in biomedical datasets that have smaller training sets, zero-shot LLMs even outperform the current state-of-the-art fine-tuned biomedical models. This suggests that pretraining on large text corpora makes LLMs quite specialized even in the biomedical domain. We also find that not a single LLM can outperform other LLMs in all tasks, with the performance of different LLMs may vary depending on the task. While their performance is still quite poor in comparison to the biomedical models that were fine-tuned on large training sets, our findings demonstrate that LLMs have the potential to be a valuable tool for various biomedical tasks that lack large annotated data.

{{</citation>}}


### (35/70) Automatic Aspect Extraction from Scientific Texts (Anna Marshalova et al., 2023)

{{<citation>}}

Anna Marshalova, Elena Bruches, Tatiana Batura. (2023)  
**Automatic Aspect Extraction from Scientific Texts**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2310.04074v1)  

---


**ABSTRACT**  
Being able to extract from scientific papers their main points, key insights, and other important information, referred to here as aspects, might facilitate the process of conducting a scientific literature review. Therefore, the aim of our research is to create a tool for automatic aspect extraction from Russian-language scientific texts of any domain. In this paper, we present a cross-domain dataset of scientific texts in Russian, annotated with such aspects as Task, Contribution, Method, and Conclusion, as well as a baseline algorithm for aspect extraction, based on the multilingual BERT model fine-tuned on our data. We show that there are some differences in aspect representation in different domains, but even though our model was trained on a limited number of scientific domains, it is still able to generalize to new domains, as was proved by cross-domain experiments. The code and the dataset are available at \url{https://github.com/anna-marshalova/automatic-aspect-extraction-from-scientific-texts}.

{{</citation>}}


### (36/70) Analysis of the Reasoning with Redundant Information Provided Ability of Large Language Models (Wenbei Xie, 2023)

{{<citation>}}

Wenbei Xie. (2023)  
**Analysis of the Reasoning with Redundant Information Provided Ability of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-3.5, Language Model, QA, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.04039v1)  

---


**ABSTRACT**  
Recent advancements in Large Language Models (LLMs) have demonstrated impressive capabilities across a range of natural language processing tasks, especially in reasoning, a cornerstone for achieving Artificial General Intelligence (AGI). However, commonly used benchmarks may not fully encapsulate the inferential abilities of these models in real-world scenarios. To address this gap, a new form of Question-Answering (QA) task, termed Reasoning with Redundant Information Provided (RRIP), is introduced. The study designed a modified version of the grade school math 8K (GSM-8K) dataset which has several variants focusing on different attributes of redundant information. This investigation evaluates two popular LLMs, LlaMA2-13B-chat and generative pre-trained transformer 3.5 (GPT-3.5), contrasting their performance on traditional QA tasks against the RRIP tasks. Findings indicate that while these models achieved moderate success on standard QA benchmarks, their performance notably declines when assessed on RRIP tasks. The study not only highlights the limitations of current LLMs in handling redundant information but also suggests that future training of these models should focus on incorporating redundant information into the training data to increase the performance on RRIP tasks.

{{</citation>}}


### (37/70) Enhancing Financial Sentiment Analysis via Retrieval Augmented Large Language Models (Boyu Zhang et al., 2023)

{{<citation>}}

Boyu Zhang, Hongyang Yang, Tianyu Zhou, Ali Babar, Xiao-Yang Liu. (2023)  
**Enhancing Financial Sentiment Analysis via Retrieval Augmented Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL, q-fin-ST, q-fin-TR  
Keywords: ChatGPT, Financial, GPT, LLaMA, Language Model, NLP, Sentiment Analysis  
[Paper Link](http://arxiv.org/abs/2310.04027v1)  

---


**ABSTRACT**  
Financial sentiment analysis is critical for valuation and investment decision-making. Traditional NLP models, however, are limited by their parameter size and the scope of their training datasets, which hampers their generalization capabilities and effectiveness in this field. Recently, Large Language Models (LLMs) pre-trained on extensive corpora have demonstrated superior performance across various NLP tasks due to their commendable zero-shot abilities. Yet, directly applying LLMs to financial sentiment analysis presents challenges: The discrepancy between the pre-training objective of LLMs and predicting the sentiment label can compromise their predictive performance. Furthermore, the succinct nature of financial news, often devoid of sufficient context, can significantly diminish the reliability of LLMs' sentiment analysis. To address these challenges, we introduce a retrieval-augmented LLMs framework for financial sentiment analysis. This framework includes an instruction-tuned LLMs module, which ensures LLMs behave as predictors of sentiment labels, and a retrieval-augmentation module which retrieves additional context from reliable external sources. Benchmarked against traditional models and LLMs like ChatGPT and LLaMA, our approach achieves 15\% to 48\% performance gain in accuracy and F1 score.

{{</citation>}}


### (38/70) SemStamp: A Semantic Watermark with Paraphrastic Robustness for Text Generation (Abe Bohan Hou et al., 2023)

{{<citation>}}

Abe Bohan Hou, Jingyu Zhang, Tianxing He, Yichen Wang, Yung-Sung Chuang, Hongwei Wang, Lingfeng Shen, Benjamin Van Durme, Daniel Khashabi, Yulia Tsvetkov. (2023)  
**SemStamp: A Semantic Watermark with Paraphrastic Robustness for Text Generation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Text Generation  
[Paper Link](http://arxiv.org/abs/2310.03991v1)  

---


**ABSTRACT**  
Existing watermarking algorithms are vulnerable to paraphrase attacks because of their token-level design. To address this issue, we propose SemStamp, a robust sentence-level semantic watermarking algorithm based on locality-sensitive hashing (LSH), which partitions the semantic space of sentences. The algorithm encodes and LSH-hashes a candidate sentence generated by an LLM, and conducts sentence-level rejection sampling until the sampled sentence falls in watermarked partitions in the semantic embedding space. A margin-based constraint is used to enhance its robustness. To show the advantages of our algorithm, we propose a "bigram" paraphrase attack using the paraphrase that has the fewest bigram overlaps with the original sentence. This attack is shown to be effective against the existing token-level watermarking method. Experimental results show that our novel semantic watermark algorithm is not only more robust than the previous state-of-the-art method on both common and bigram paraphrase attacks, but also is better at preserving the quality of generation.

{{</citation>}}


### (39/70) Dementia Assessment Using Mandarin Speech with an Attention-based Speech Recognition Encoder (Zih-Jyun Lin et al., 2023)

{{<citation>}}

Zih-Jyun Lin, Yi-Ju Chen, Po-Chih Kuo, Likai Huang, Chaur-Jong Hu, Cheng-Yu Chen. (2023)  
**Dementia Assessment Using Mandarin Speech with an Attention-based Speech Recognition Encoder**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs-SD, cs.CL, eess-AS  
Keywords: Attention, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2310.03985v1)  

---


**ABSTRACT**  
Dementia diagnosis requires a series of different testing methods, which is complex and time-consuming. Early detection of dementia is crucial as it can prevent further deterioration of the condition. This paper utilizes a speech recognition model to construct a dementia assessment system tailored for Mandarin speakers during the picture description task. By training an attention-based speech recognition model on voice data closely resembling real-world scenarios, we have significantly enhanced the model's recognition capabilities. Subsequently, we extracted the encoder from the speech recognition model and added a linear layer for dementia assessment. We collected Mandarin speech data from 99 subjects and acquired their clinical assessments from a local hospital. We achieved an accuracy of 92.04% in Alzheimer's disease detection and a mean absolute error of 9% in clinical dementia rating score prediction.

{{</citation>}}


### (40/70) Quantized Transformer Language Model Implementations on Edge Devices (Mohammad Wali Ur Rahman et al., 2023)

{{<citation>}}

Mohammad Wali Ur Rahman, Murad Mehrab Abrar, Hunter Gibbons Copening, Salim Hariri, Sicong Shao, Pratik Satam, Soheil Salehi. (2023)  
**Quantized Transformer Language Model Implementations on Edge Devices**  

---
Primary Category: cs.CL  
Categories: cs-AR, cs-CL, cs.CL  
Keywords: BERT, Language Model, NLP, Natural Language Processing, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.03971v1)  

---


**ABSTRACT**  
Large-scale transformer-based models like the Bidirectional Encoder Representations from Transformers (BERT) are widely used for Natural Language Processing (NLP) applications, wherein these models are initially pre-trained with a large corpus with millions of parameters and then fine-tuned for a downstream NLP task. One of the major limitations of these large-scale models is that they cannot be deployed on resource-constrained devices due to their large model size and increased inference latency. In order to overcome these limitations, such large-scale models can be converted to an optimized FlatBuffer format, tailored for deployment on resource-constrained edge devices. Herein, we evaluate the performance of such FlatBuffer transformed MobileBERT models on three different edge devices, fine-tuned for Reputation analysis of English language tweets in the RepLab 2013 dataset. In addition, this study encompassed an evaluation of the deployed models, wherein their latency, performance, and resource efficiency were meticulously assessed. Our experiment results show that, compared to the original BERT large model, the converted and quantized MobileBERT models have 160$\times$ smaller footprints for a 4.1% drop in accuracy while analyzing at least one tweet per second on edge devices. Furthermore, our study highlights the privacy-preserving aspect of TinyML systems as all data is processed locally within a serverless environment.

{{</citation>}}


### (41/70) Chain of Natural Language Inference for Reducing Large Language Model Ungrounded Hallucinations (Deren Lei et al., 2023)

{{<citation>}}

Deren Lei, Yaxi Li, Mengya, Hu, Mingyu Wang, Vincent Yun, Emily Ching, Eslam Kamal. (2023)  
**Chain of Natural Language Inference for Reducing Large Language Model Ungrounded Hallucinations**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, NLI, Natural Language Inference  
[Paper Link](http://arxiv.org/abs/2310.03951v1)  

---


**ABSTRACT**  
Large language models (LLMs) can generate fluent natural language texts when given relevant documents as background context. This ability has attracted considerable interest in developing industry applications of LLMs. However, LLMs are prone to generate hallucinations that are not supported by the provided sources. In this paper, we propose a hierarchical framework to detect and mitigate such ungrounded hallucination. Our framework uses Chain of Natural Language Inference (CoNLI) for hallucination detection and hallucination reduction via post-editing. Our approach achieves state-of-the-art performance on hallucination detection and enhances text quality through rewrite, using LLMs without any fine-tuning or domain-specific prompt engineering. We show that this simple plug-and-play framework can serve as an effective choice for hallucination detection and reduction, achieving competitive performance across various contexts.

{{</citation>}}


## cs.AI (3)



### (42/70) Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models (Andy Zhou et al., 2023)

{{<citation>}}

Andy Zhou, Kai Yan, Michal Shlapentokh-Rothman, Haohan Wang, Yu-Xiong Wang. (2023)  
**Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs.AI  
Keywords: GPT, GPT-3.5, GPT-4, Language Model, QA, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.04406v1)  

---


**ABSTRACT**  
While large language models (LLMs) have demonstrated impressive performance on a range of decision-making tasks, they rely on simple acting processes and fall short of broad deployment as autonomous agents. We introduce LATS (Language Agent Tree Search), a general framework that synergizes the capabilities of LLMs in planning, acting, and reasoning. Drawing inspiration from Monte Carlo tree search in model-based reinforcement learning, LATS employs LLMs as agents, value functions, and optimizers, repurposing their latent strengths for enhanced decision-making. What is crucial in this method is the use of an environment for external feedback, which offers a more deliberate and adaptive problem-solving mechanism that moves beyond the limitations of existing techniques. Our experimental evaluation across diverse domains, such as programming, HotPotQA, and WebShop, illustrates the applicability of LATS for both reasoning and acting. In particular, LATS achieves 94.4\% for programming on HumanEval with GPT-4 and an average score of 75.9 for web browsing on WebShop with GPT-3.5, demonstrating the effectiveness and generality of our method.

{{</citation>}}


### (43/70) From task structures to world models: What do LLMs know? (Ilker Yildirim et al., 2023)

{{<citation>}}

Ilker Yildirim, L. A. Paul. (2023)  
**From task structures to world models: What do LLMs know?**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI, q-bio-NC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.04276v1)  

---


**ABSTRACT**  
In what sense does a large language model have knowledge? The answer to this question extends beyond the capabilities of a particular AI system, and challenges our assumptions about the nature of knowledge and intelligence. We answer by granting LLMs "instrumental knowledge"; knowledge defined by a certain set of abilities. We then ask how such knowledge is related to the more ordinary, "worldly" knowledge exhibited by human agents, and explore this in terms of the degree to which instrumental knowledge can be said to incorporate the structured world models of cognitive science. We discuss ways LLMs could recover degrees of worldly knowledge, and suggest such recovery will be governed by an implicit, resource-rational tradeoff between world models and task demands.

{{</citation>}}


### (44/70) Thought Propagation: An Analogical Approach to Complex Reasoning with Large Language Models (Junchi Yu et al., 2023)

{{<citation>}}

Junchi Yu, Ran He, Rex Ying. (2023)  
**Thought Propagation: An Analogical Approach to Complex Reasoning with Large Language Models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.03965v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have achieved remarkable success in reasoning tasks with the development of prompting methods. However, existing prompting approaches cannot reuse insights of solving similar problems and suffer from accumulated errors in multi-step reasoning, since they prompt LLMs to reason \textit{from scratch}. To address these issues, we propose \textbf{\textit{Thought Propagation} (TP)}, which explores the analogous problems and leverages their solutions to enhance the complex reasoning ability of LLMs. These analogous problems are related to the input one, with reusable solutions and problem-solving strategies. Thus, it is promising to propagate insights of solving previous analogous problems to inspire new problem-solving. To achieve this, TP first prompts LLMs to propose and solve a set of analogous problems that are related to the input one. Then, TP reuses the results of analogous problems to directly yield a new solution or derive a knowledge-intensive plan for execution to amend the initial solution obtained from scratch. TP is compatible with existing prompting approaches, allowing plug-and-play generalization and enhancement in a wide range of tasks without much labor in task-specific prompt engineering. Experiments across three challenging tasks demonstrate TP enjoys a substantial improvement over the baselines by an average of 12\% absolute increase in finding the optimal solutions in Shortest-path Reasoning, 13\% improvement of human preference in Creative Writing, and 15\% enhancement in the task completion rate of LLM-Agent Planning.

{{</citation>}}


## cs.CR (2)



### (45/70) Hermes: Unlocking Security Analysis of Cellular Network Protocols by Synthesizing Finite State Machines from Natural Language Specifications (Abdullah Al Ishtiaq et al., 2023)

{{<citation>}}

Abdullah Al Ishtiaq, Sarkar Snigdha Sarathi Das, Syed Md Mukit Rashid, Ali Ranjbar, Kai Tu, Tianwei Wu, Zhezheng Song, Weixuan Wang, Mujtahid Akon, Rui Zhang, Syed Rafiul Hussain. (2023)  
**Hermes: Unlocking Security Analysis of Cellular Network Protocols by Synthesizing Finite State Machines from Natural Language Specifications**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CL, cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2310.04381v1)  

---


**ABSTRACT**  
In this paper, we present Hermes, an end-to-end framework to automatically generate formal representations from natural language cellular specifications. We first develop a neural constituency parser, NEUTREX, to process transition-relevant texts and extract transition components (i.e., states, conditions, and actions). We also design a domain-specific language to translate these transition components to logical formulas by leveraging dependency parse trees. Finally, we compile these logical formulas to generate transitions and create the formal model as finite state machines. To demonstrate the effectiveness of Hermes, we evaluate it on 4G NAS, 5G NAS, and 5G RRC specifications and obtain an overall accuracy of 81-87%, which is a substantial improvement over the state-of-the-art. Our security analysis of the extracted models uncovers 3 new vulnerabilities and identifies 19 previous attacks in 4G and 5G specifications, and 7 deviations in commercial 4G basebands.

{{</citation>}}


### (46/70) Kick Bad Guys Out! Zero-Knowledge-Proof-Based Anomaly Detection in Federated Learning (Shanshan Han et al., 2023)

{{<citation>}}

Shanshan Han, Wenxuan Wu, Baturalp Buyukates, Weizhao Jin, Yuhang Yao, Qifan Zhang, Salman Avestimehr, Chaoyang He. (2023)  
**Kick Bad Guys Out! Zero-Knowledge-Proof-Based Anomaly Detection in Federated Learning**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2310.04055v1)  

---


**ABSTRACT**  
Federated learning (FL) systems are vulnerable to malicious clients that submit poisoned local models to achieve their adversarial goals, such as preventing the convergence of the global model or inducing the global model to misclassify some data. Many existing defense mechanisms are impractical in real-world FL systems, as they require prior knowledge of the number of malicious clients or rely on re-weighting or modifying submissions. This is because adversaries typically do not announce their intentions before attacking, and re-weighting might change aggregation results even in the absence of attacks. To address these challenges in real FL systems, this paper introduces a cutting-edge anomaly detection approach with the following features: i) Detecting the occurrence of attacks and performing defense operations only when attacks happen; ii) Upon the occurrence of an attack, further detecting the malicious client models and eliminating them without harming the benign ones; iii) Ensuring honest execution of defense mechanisms at the server by leveraging a zero-knowledge proof mechanism. We validate the superior performance of the proposed approach with extensive experiments.

{{</citation>}}


## stat.ML (2)



### (47/70) A Marketplace Price Anomaly Detection System at Scale (Akshit Sarpal et al., 2023)

{{<citation>}}

Akshit Sarpal, Qiwen Kang, Fangping Huang, Yang Song, Lijie Wan. (2023)  
**A Marketplace Price Anomaly Detection System at Scale**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-ML, stat.ML  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2310.04367v1)  

---


**ABSTRACT**  
Online marketplaces execute large volume of price updates that are initiated by individual marketplace sellers each day on the platform. This price democratization comes with increasing challenges with data quality. Lack of centralized guardrails that are available for a traditional online retailer causes a higher likelihood for inaccurate prices to get published on the website, leading to poor customer experience and potential for revenue loss. We present MoatPlus (Masked Optimal Anchors using Trees, Proximity-based Labeling and Unsupervised Statistical-features), a scalable price anomaly detection framework for a growing marketplace platform. The goal is to leverage proximity and historical price trends from unsupervised statistical features to generate an upper price bound. We build an ensemble of models to detect irregularities in price-based features, exclude irregular features and use optimized weighting scheme to build a reliable price bound in real-time pricing pipeline. We observed that our approach improves precise anchor coverage by up to 46.6% in high-vulnerability item subsets

{{</citation>}}


### (48/70) Fair Feature Importance Scores for Interpreting Tree-Based Methods and Surrogates (Camille Olivia Little et al., 2023)

{{<citation>}}

Camille Olivia Little, Debolina Halder Lina, Genevera I. Allen. (2023)  
**Fair Feature Importance Scores for Interpreting Tree-Based Methods and Surrogates**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-ML, stat.ML  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.04352v1)  

---


**ABSTRACT**  
Across various sectors such as healthcare, criminal justice, national security, finance, and technology, large-scale machine learning (ML) and artificial intelligence (AI) systems are being deployed to make critical data-driven decisions. Many have asked if we can and should trust these ML systems to be making these decisions. Two critical components are prerequisites for trust in ML systems: interpretability, or the ability to understand why the ML system makes the decisions it does, and fairness, which ensures that ML systems do not exhibit bias against certain individuals or groups. Both interpretability and fairness are important and have separately received abundant attention in the ML literature, but so far, there have been very few methods developed to directly interpret models with regard to their fairness. In this paper, we focus on arguably the most popular type of ML interpretation: feature importance scores. Inspired by the use of decision trees in knowledge distillation, we propose to leverage trees as interpretable surrogates for complex black-box ML models. Specifically, we develop a novel fair feature importance score for trees that can be used to interpret how each feature contributes to fairness or bias in trees, tree-based ensembles, or tree-based surrogates of any complex ML system. Like the popular mean decrease in impurity for trees, our Fair Feature Importance Score is defined based on the mean decrease (or increase) in group bias. Through simulations as well as real examples on benchmark fairness datasets, we demonstrate that our Fair Feature Importance Score offers valid interpretations for both tree-based ensembles and tree-based surrogates of other ML systems.

{{</citation>}}


## cs.NI (2)



### (49/70) Enhanced Backpressure Routing with Wireless Link Features (Zhongyuan Zhao et al., 2023)

{{<citation>}}

Zhongyuan Zhao, Gunjan Verma, Ananthram Swami, Santiago Segarra. (2023)  
**Enhanced Backpressure Routing with Wireless Link Features**  

---
Primary Category: cs.NI  
Categories: 05C90, C-2-1; C-2-2, cs-NI, cs.NI, eess-SP  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2310.04364v1)  

---


**ABSTRACT**  
Backpressure (BP) routing is a well-established framework for distributed routing and scheduling in wireless multi-hop networks. However, the basic BP scheme suffers from poor end-to-end delay due to the drawbacks of slow startup, random walk, and the last packet problem. Biased BP with shortest path awareness can address the first two drawbacks, and sojourn time-based backlog metrics have been proposed for the last packet problem. Furthermore, these BP variations require no additional signaling overhead in each time step compared to the basic BP. In this work, we further address three long-standing challenges associated with the aforementioned low-cost BP variations, including optimal scaling of the biases, bias maintenance under mobility, and incorporating sojourn time awareness into biased BP. Our analysis and experimental results show that proper scaling of biases can be achieved with the help of common link features, which can effectively reduce end-to-end delay of BP by mitigating the random walk of packets under low-to-medium traffic, including the last packet scenario. In addition, our low-overhead bias maintenance scheme is shown to be effective under mobility, and our bio-inspired sojourn time-aware backlog metric is demonstrated to be more efficient and effective for the last packet problem than existing approaches when incorporated into biased BP.

{{</citation>}}


### (50/70) The Role of Federated Learning in a Wireless World with Foundation Models (Zihan Chen et al., 2023)

{{<citation>}}

Zihan Chen, Howard H. Yang, Y. C. Tay, Kai Fong Ernest Chong, Tony Q. S. Quek. (2023)  
**The Role of Federated Learning in a Wireless World with Foundation Models**  

---
Primary Category: cs.NI  
Categories: cs-DC, cs-LG, cs-NI, cs-SY, cs.NI, eess-SY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.04003v1)  

---


**ABSTRACT**  
Foundation models (FMs) are general-purpose artificial intelligence (AI) models that have recently enabled multiple brand-new generative AI applications. The rapid advances in FMs serve as an important contextual backdrop for the vision of next-generation wireless networks, where federated learning (FL) is a key enabler of distributed network intelligence. Currently, the exploration of the interplay between FMs and FL is still in its nascent stage. Naturally, FMs are capable of boosting the performance of FL, and FL could also leverage decentralized data and computing resources to assist in the training of FMs. However, the exceptionally high requirements that FMs have for computing resources, storage, and communication overhead would pose critical challenges to FL-enabled wireless networks. In this article, we explore the extent to which FMs are suitable for FL over wireless networks, including a broad overview of research challenges and opportunities. In particular, we discuss multiple new paradigms for realizing future intelligent networks that integrate FMs and FL. We also consolidate several broad research directions associated with these paradigms.

{{</citation>}}


## math.OC (1)



### (51/70) Neur2RO: Neural Two-Stage Robust Optimization (Justin Dumouchelle et al., 2023)

{{<citation>}}

Justin Dumouchelle, Esther Julien, Jannis Kurtz, Elias B. Khalil. (2023)  
**Neur2RO: Neural Two-Stage Robust Optimization**  

---
Primary Category: math.OC  
Categories: cs-AI, cs-LG, math-OC, math.OC  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2310.04345v1)  

---


**ABSTRACT**  
Robust optimization provides a mathematical framework for modeling and solving decision-making problems under worst-case uncertainty. This work addresses two-stage robust optimization (2RO) problems (also called adjustable robust optimization), wherein first-stage and second-stage decisions are made before and after uncertainty is realized, respectively. This results in a nested min-max-min optimization problem which is extremely challenging computationally, especially when the decisions are discrete. We propose Neur2RO, an efficient machine learning-driven instantiation of column-and-constraint generation (CCG), a classical iterative algorithm for 2RO. Specifically, we learn to estimate the value function of the second-stage problem via a novel neural network architecture that is easy to optimize over by design. Embedding our neural network into CCG yields high-quality solutions quickly as evidenced by experiments on two 2RO benchmarks, knapsack and capital budgeting. For knapsack, Neur2RO finds solutions that are within roughly $2\%$ of the best-known values in a few seconds compared to the three hours of the state-of-the-art exact branch-and-price algorithm; for larger and more complex instances, Neur2RO finds even better solutions. For capital budgeting, Neur2RO outperforms three variants of the $k$-adaptability algorithm, particularly on the largest instances, with a 5 to 10-fold reduction in solution time. Our code and data are available at https://github.com/khalil-research/Neur2RO.

{{</citation>}}


## q-fin.CP (1)



### (52/70) Applying Reinforcement Learning to Option Pricing and Hedging (Zoran Stoiljkovic, 2023)

{{<citation>}}

Zoran Stoiljkovic. (2023)  
**Applying Reinforcement Learning to Option Pricing and Hedging**  

---
Primary Category: q-fin.CP  
Categories: cs-CE, cs-LG, q-fin-CP, q-fin.CP  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.04336v1)  

---


**ABSTRACT**  
This thesis provides an overview of the recent advances in reinforcement learning in pricing and hedging financial instruments, with a primary focus on a detailed explanation of the Q-Learning Black Scholes approach, introduced by Halperin (2017). This reinforcement learning approach bridges the traditional Black and Scholes (1973) model with novel artificial intelligence algorithms, enabling option pricing and hedging in a completely model-free and data-driven way. This paper also explores the algorithm's performance under different state variables and scenarios for a European put option. The results reveal that the model is an accurate estimator under different levels of volatility and hedging frequency. Moreover, this method exhibits robust performance across various levels of option's moneyness. Lastly, the algorithm incorporates proportional transaction costs, indicating diverse impacts on profit and loss, affected by different statistical properties of the state variables.

{{</citation>}}


## cs.SE (1)



### (53/70) Coding by Design: GPT-4 empowers Agile Model Driven Development (Ahmed R. Sadik et al., 2023)

{{<citation>}}

Ahmed R. Sadik, Sebastian Brulin, Markus Olhofer. (2023)  
**Coding by Design: GPT-4 empowers Agile Model Driven Development**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-FL, cs-MA, cs-PL, cs-SE, cs.SE  
Keywords: AI, ChatGPT, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2310.04304v1)  

---


**ABSTRACT**  
Generating code from a natural language using Large Language Models (LLMs) such as ChatGPT, seems groundbreaking. Yet, with more extensive use, it's evident that this approach has its own limitations. The inherent ambiguity of natural language presents challenges for complex software designs. Accordingly, our research offers an Agile Model-Driven Development (MDD) approach that enhances code auto-generation using OpenAI's GPT-4. Our work emphasizes "Agility" as a significant contribution to the current MDD method, particularly when the model undergoes changes or needs deployment in a different programming language. Thus, we present a case-study showcasing a multi-agent simulation system of an Unmanned Vehicle Fleet. In the first and second layer of our approach, we constructed a textual representation of the case-study using Unified Model Language (UML) diagrams. In the next layer, we introduced two sets of constraints that minimize model ambiguity. Object Constraints Language (OCL) is applied to fine-tune the code constructions details, while FIPA ontology is used to shape communication semantics and protocols. Ultimately, leveraging GPT-4, our last layer auto-generates code in both Java and Python. The Java code is deployed within the JADE framework, while the Python code is deployed in PADE framework. Concluding our research, we engaged in a comprehensive evaluation of the generated code. From a behavioural standpoint, the auto-generated code aligned perfectly with the expected UML sequence diagram. Structurally, we compared the complexity of code derived from UML diagrams constrained solely by OCL to that influenced by both OCL and FIPA-ontology. Results indicate that ontology-constrained model produce inherently more intricate code, but it remains manageable and low-risk for further testing and maintenance.

{{</citation>}}


## eess.SY (2)



### (54/70) Searching for Optimal Runtime Assurance via Reachability and Reinforcement Learning (Kristina Miller et al., 2023)

{{<citation>}}

Kristina Miller, Christopher K. Zeitler, William Shen, Kerianne Hobbs, Sayan Mitra, John Schierman, Mahesh Viswanathan. (2023)  
**Searching for Optimal Runtime Assurance via Reachability and Reinforcement Learning**  

---
Primary Category: eess.SY  
Categories: cs-AI, cs-FL, cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.04288v1)  

---


**ABSTRACT**  
A runtime assurance system (RTA) for a given plant enables the exercise of an untrusted or experimental controller while assuring safety with a backup (or safety) controller. The relevant computational design problem is to create a logic that assures safety by switching to the safety controller as needed, while maximizing some performance criteria, such as the utilization of the untrusted controller. Existing RTA design strategies are well-known to be overly conservative and, in principle, can lead to safety violations. In this paper, we formulate the optimal RTA design problem and present a new approach for solving it. Our approach relies on reward shaping and reinforcement learning. It can guarantee safety and leverage machine learning technologies for scalability. We have implemented this algorithm and present experimental results comparing our approach with state-of-the-art reachability and simulation-based RTA approaches in a number of scenarios using aircraft models in 3D space with complex safety requirements. Our approach can guarantee safety while increasing utilization of the experimental controller over existing approaches.

{{</citation>}}


### (55/70) Topology-Aware Neural Networks for Fast Contingency Analysis of Power Systems (Agnes M. Nakiganda et al., 2023)

{{<citation>}}

Agnes M. Nakiganda, Catherine Cheylan, Spyros Chatzivasileiadis. (2023)  
**Topology-Aware Neural Networks for Fast Contingency Analysis of Power Systems**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2310.04213v1)  

---


**ABSTRACT**  
Training Neural Networks able to capture the topology changes of the power grid is one of the significant challenges towards the adoption of machine learning techniques for N-k security computations and a wide range of other operations that involve grid reconfiguration. As the number of N-k scenarios increases exponentially with increasing system size, such problems are extremely time-consuming to solve with traditional solvers. In this paper, we combine Physics-Informed Neural Networks with both a Guided-Dropout (GD) (which associates dedicated neurons with specific line connections/disconnections) and an edge-varying Graph Neural Neural Network (GNN) architecture to learn the setpoints for a grid that considers all probable single-line reconfigurations (all critical N-1 scenarios) and subsequently apply the trained models to N-k scenarios. We demonstrate how incorporating the underlying physical equations for the network equations along with the GD and the GNN methods, performs with N-1, N-2, and N-3 case studies. Using the AC Power Flow as a guiding application, we test our methods on the 14-bus, 30-bus, 57-bus, and 118-bus systems, and we compare the models in terms of the accuracy and computational performance that each one achieves for each study and provide recommendations on their adoption for contingency analysis of power systems.

{{</citation>}}


## cs.RO (1)



### (56/70) DRIFT: Deep Reinforcement Learning for Intelligent Floating Platforms Trajectories (Matteo El-Hariry et al., 2023)

{{<citation>}}

Matteo El-Hariry, Antoine Richard, Vivek Muralidharan, Baris Can Yalcin, Matthieu Geist, Miguel Olivares-Mendez. (2023)  
**DRIFT: Deep Reinforcement Learning for Intelligent Floating Platforms Trajectories**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.04266v1)  

---


**ABSTRACT**  
This investigation introduces a novel deep reinforcement learning-based suite to control floating platforms in both simulated and real-world environments. Floating platforms serve as versatile test-beds to emulate microgravity environments on Earth. Our approach addresses the system and environmental uncertainties in controlling such platforms by training policies capable of precise maneuvers amid dynamic and unpredictable conditions. Leveraging state-of-the-art deep reinforcement learning techniques, our suite achieves robustness, adaptability, and good transferability from simulation to reality. Our Deep Reinforcement Learning (DRL) framework provides advantages such as fast training times, large-scale testing capabilities, rich visualization options, and ROS bindings for integration with real-world robotic systems. Beyond policy development, our suite provides a comprehensive platform for researchers, offering open-access at https://github.com/elharirymatteo/RANS/tree/ICRA24.

{{</citation>}}


## cs.IR (2)



### (57/70) Keyword Augmented Retrieval: Novel framework for Information Retrieval integrated with speech interface (Anupam Purwar et al., 2023)

{{<citation>}}

Anupam Purwar, Rahul Sundar. (2023)  
**Keyword Augmented Retrieval: Novel framework for Information Retrieval integrated with speech interface**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-CL, cs-HC, cs-IR, cs.IR  
Keywords: GPT, Information Retrieval  
[Paper Link](http://arxiv.org/abs/2310.04205v1)  

---


**ABSTRACT**  
Retrieving answers in a quick and low cost manner without hallucinations from a combination of structured and unstructured data using Language models is a major hurdle which prevents employment of Language models in knowledge retrieval automation. This becomes accentuated when one wants to integrate a speech interface. Besides, for commercial search and chatbot applications, complete reliance on commercial large language models (LLMs) like GPT 3.5 etc. can be very costly. In this work, authors have addressed this problem by first developing a keyword based search framework which augments discovery of the context to be provided to the large language model. The keywords in turn are generated by LLM and cached for comparison with keywords generated by LLM against the query raised. This significantly reduces time and cost to find the context within documents. Once the context is set, LLM uses that to provide answers based on a prompt tailored for Q&A. This research work demonstrates that use of keywords in context identification reduces the overall inference time and cost of information retrieval. Given this reduction in inference time and cost with the keyword augmented retrieval framework, a speech based interface for user input and response readout was integrated. This allowed a seamless interaction with the language model.

{{</citation>}}


### (58/70) AdaRec: Adaptive Sequential Recommendation for Reinforcing Long-term User Engagement (Zhenghai Xue et al., 2023)

{{<citation>}}

Zhenghai Xue, Qingpeng Cai, Tianyou Zuo, Bin Yang, Lantao Hu, Peng Jiang, Kun Gai, Bo An. (2023)  
**AdaRec: Adaptive Sequential Recommendation for Reinforcing Long-term User Engagement**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs-LG, cs.IR  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.03984v1)  

---


**ABSTRACT**  
Growing attention has been paid to Reinforcement Learning (RL) algorithms when optimizing long-term user engagement in sequential recommendation tasks. One challenge in large-scale online recommendation systems is the constant and complicated changes in users' behavior patterns, such as interaction rates and retention tendencies. When formulated as a Markov Decision Process (MDP), the dynamics and reward functions of the recommendation system are continuously affected by these changes. Existing RL algorithms for recommendation systems will suffer from distribution shift and struggle to adapt in such an MDP. In this paper, we introduce a novel paradigm called Adaptive Sequential Recommendation (AdaRec) to address this issue. AdaRec proposes a new distance-based representation loss to extract latent information from users' interaction trajectories. Such information reflects how RL policy fits to current user behavior patterns, and helps the policy to identify subtle changes in the recommendation system. To make rapid adaptation to these changes, AdaRec encourages exploration with the idea of optimism under uncertainty. The exploration is further guarded by zero-order action optimization to ensure stable recommendation quality in complicated environments. We conduct extensive empirical analyses in both simulator-based and live sequential recommendation tasks, where AdaRec exhibits superior long-term performance compared to all baseline algorithms.

{{</citation>}}


## eess.IV (1)



### (59/70) Aorta Segmentation from 3D CT in MICCAI SEG.A. 2023 Challenge (Andriy Myronenko et al., 2023)

{{<citation>}}

Andriy Myronenko, Dong Yang, Yufan He, Daguang Xu. (2023)  
**Aorta Segmentation from 3D CT in MICCAI SEG.A. 2023 Challenge**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.04114v1)  

---


**ABSTRACT**  
Aorta provides the main blood supply of the body. Screening of aorta with imaging helps for early aortic disease detection and monitoring. In this work, we describe our solution to the Segmentation of the Aorta (SEG.A.231) from 3D CT challenge. We use automated segmentation method Auto3DSeg available in MONAI. Our solution achieves an average Dice score of 0.920 and 95th percentile of the Hausdorff Distance (HD95) of 6.013, which ranks first and wins the SEG.A. 2023 challenge.

{{</citation>}}


## cs.SI (1)



### (60/70) Marketing to Children Through Online Targeted Advertising: Targeting Mechanisms and Legal Aspects (Tinhinane Medjkoune et al., 2023)

{{<citation>}}

Tinhinane Medjkoune, Oana Goga, Juliette Senechal. (2023)  
**Marketing to Children Through Online Targeted Advertising: Targeting Mechanisms and Legal Aspects**  

---
Primary Category: cs.SI  
Categories: cs-CY, cs-SI, cs.SI  
Keywords: Google, Legal  
[Paper Link](http://arxiv.org/abs/2310.04104v1)  

---


**ABSTRACT**  
Many researchers and organizations, such as WHO and UNICEF, have raised awareness of the dangers of advertisements targeted at children. While most existing laws only regulate ads on television that may reach children, lawmakers have been working on extending regulations to online advertising and, for example, forbid (e.g., the DSA) or restrict (e.g., the COPPA) advertising based on profiling to children. At first sight, ad platforms such as Google seem to protect children by not allowing advertisers to target their ads to users who are less than 18 years old. However, this paper shows that other targeting features can be exploited to reach children. For example, on YouTube, advertisers can target their ads to users watching a particular video through placement-based targeting, a form of contextual targeting. Hence, advertisers can target children by placing their ads in children-focused videos. Through a series of ad experiments, we show that placement-based targeting is possible on children-focused videos and enables marketing to children. In addition, our ad experiments show that advertisers can use targeting based on profiling (e.g., interest, location, behavior) in combination with placement-based advertising on children-focused videos. We discuss the lawfulness of these two practices concerning DSA and COPPA. Finally, we investigate to which extent real-world advertisers are employing placement-based targeting to reach children with ads on YouTube. We propose a measurement methodology consisting of building a Chrome extension to capture ads and instrument six browser profiles to watch children-focused videos. Our results show that 7% of ads that appear in the children-focused videos we test use placement-based targeting. Hence, targeting children with ads on YouTube is not only hypothetically possible but also occurs in practice...

{{</citation>}}


## cs.CY (2)



### (61/70) Impact of Gender on the Evaluation of Security Decisions (Winnie Mbaka et al., 2023)

{{<citation>}}

Winnie Mbaka, Katja Tuma. (2023)  
**Impact of Gender on the Evaluation of Security Decisions**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2310.04097v1)  

---


**ABSTRACT**  
Security decisions are made by human analysts under uncertain conditions which leaves room for bias judgement. However, little is known about how demographics like gender and education impact these judgments. We conducted an empirical study to investigate their influence on security decision evaluations, addressing this knowledge gap.

{{</citation>}}


### (62/70) AI Regulation in Europe: From the AI Act to Future Regulatory Challenges (Philipp Hacker, 2023)

{{<citation>}}

Philipp Hacker. (2023)  
**AI Regulation in Europe: From the AI Act to Future Regulatory Challenges**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.04072v1)  

---


**ABSTRACT**  
This chapter provides a comprehensive discussion on AI regulation in the European Union, contrasting it with the more sectoral and self-regulatory approach in the UK. It argues for a hybrid regulatory strategy that combines elements from both philosophies, emphasizing the need for agility and safe harbors to ease compliance. The paper examines the AI Act as a pioneering legislative effort to address the multifaceted challenges posed by AI, asserting that, while the Act is a step in the right direction, it has shortcomings that could hinder the advancement of AI technologies. The paper also anticipates upcoming regulatory challenges, such as the management of toxic content, environmental concerns, and hybrid threats. It advocates for immediate action to create protocols for regulated access to high-performance, potentially open-source AI systems. Although the AI Act is a significant legislative milestone, it needs additional refinement and global collaboration for the effective governance of rapidly evolving AI technologies.

{{</citation>}}


## cs.DS (2)



### (63/70) Deterministic Clustering in High Dimensional Spaces: Sketches and Approximation (Vincent Cohen-Addad et al., 2023)

{{<citation>}}

Vincent Cohen-Addad, David Saulpic, Chris Schwiegelshohn. (2023)  
**Deterministic Clustering in High Dimensional Spaces: Sketches and Approximation**  

---
Primary Category: cs.DS  
Categories: cs-DS, cs.DS  
Keywords: Sketch  
[Paper Link](http://arxiv.org/abs/2310.04076v1)  

---


**ABSTRACT**  
In all state-of-the-art sketching and coreset techniques for clustering, as well as in the best known fixed-parameter tractable approximation algorithms, randomness plays a key role. For the classic $k$-median and $k$-means problems, there are no known deterministic dimensionality reduction procedure or coreset construction that avoid an exponential dependency on the input dimension $d$, the precision parameter $\varepsilon^{-1}$ or $k$. Furthermore, there is no coreset construction that succeeds with probability $1-1/n$ and whose size does not depend on the number of input points, $n$. This has led researchers in the area to ask what is the power of randomness for clustering sketches [Feldman, WIREs Data Mining Knowl. Discov'20]. Similarly, the best approximation ratio achievable deterministically without a complexity exponential in the dimension are $\Omega(1)$ for both $k$-median and $k$-means, even when allowing a complexity FPT in the number of clusters $k$. This stands in sharp contrast with the $(1+\varepsilon)$-approximation achievable in that case, when allowing randomization.   In this paper, we provide deterministic sketches constructions for clustering, whose size bounds are close to the best-known randomized ones. We also construct a deterministic algorithm for computing $(1+\varepsilon)$-approximation to $k$-median and $k$-means in high dimensional Euclidean spaces in time $2^{k^2/\varepsilon^{O(1)}} poly(nd)$, close to the best randomized complexity.   Furthermore, our new insights on sketches also yield a randomized coreset construction that uses uniform sampling, that immediately improves over the recent results of [Braverman et al. FOCS '22] by a factor $k$.

{{</citation>}}


### (64/70) How to Capture Higher-order Correlations? Generalizing Matrix Softmax Attention to Kronecker Computation (Josh Alman et al., 2023)

{{<citation>}}

Josh Alman, Zhao Song. (2023)  
**How to Capture Higher-order Correlations? Generalizing Matrix Softmax Attention to Kronecker Computation**  

---
Primary Category: cs.DS  
Categories: cs-CC, cs-CL, cs-DS, cs-LG, cs.DS, stat-ML  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2310.04064v1)  

---


**ABSTRACT**  
In the classical transformer attention scheme, we are given three $n \times d$ size matrices $Q, K, V$ (the query, key, and value tokens), and the goal is to compute a new $n \times d$ size matrix $D^{-1} \exp(QK^\top) V$ where $D = \mathrm{diag}( \exp(QK^\top) {\bf 1}_n )$. In this work, we study a generalization of attention which captures triple-wise correlations. This generalization is able to solve problems about detecting triple-wise connections that were shown to be impossible for transformers. The potential downside of this generalization is that it appears as though computations are even more difficult, since the straightforward algorithm requires cubic time in $n$. However, we show that in the bounded-entry setting (which arises in practice, and which is well-studied in both theory and practice), there is actually a near-linear time algorithm. More precisely, we show that bounded entries are both necessary and sufficient for quickly performing generalized computations:   $\bullet$ On the positive side, if all entries of the input matrices are bounded above by $o(\sqrt[3]{\log n})$ then we show how to approximate the ``tensor-type'' attention matrix in $n^{1+o(1)}$ time.   $\bullet$ On the negative side, we show that if the entries of the input matrices may be as large as $\Omega(\sqrt[3]{\log n})$, then there is no algorithm that runs faster than $n^{3-o(1)}$ (assuming the Strong Exponential Time Hypothesis from fine-grained complexity theory).   We also show that our construction, algorithms, and lower bounds naturally generalize to higher-order tensors and correlations. Interestingly, the higher the order of the tensors, the lower the bound on the entries needs to be for an efficient algorithm. Our results thus yield a natural tradeoff between the boundedness of the entries, and order of the tensor one may use for more expressive, efficient attention computation.

{{</citation>}}


## cs.GR (1)



### (65/70) In the Blink of an Eye: Event-based Emotion Recognition (Haiwei Zhang et al., 2023)

{{<citation>}}

Haiwei Zhang, Jiqing Zhang, Bo Dong, Pieter Peers, Wenwei Wu, Xiaopeng Wei, Felix Heide, Xin Yang. (2023)  
**In the Blink of an Eye: Event-based Emotion Recognition**  

---
Primary Category: cs.GR  
Categories: cs-CV, cs-GR, cs.GR  
Keywords: Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2310.04043v1)  

---


**ABSTRACT**  
We introduce a wearable single-eye emotion recognition device and a real-time approach to recognizing emotions from partial observations of an emotion that is robust to changes in lighting conditions. At the heart of our method is a bio-inspired event-based camera setup and a newly designed lightweight Spiking Eye Emotion Network (SEEN). Compared to conventional cameras, event-based cameras offer a higher dynamic range (up to 140 dB vs. 80 dB) and a higher temporal resolution. Thus, the captured events can encode rich temporal cues under challenging lighting conditions. However, these events lack texture information, posing problems in decoding temporal information effectively. SEEN tackles this issue from two different perspectives. First, we adopt convolutional spiking layers to take advantage of the spiking neural network's ability to decode pertinent temporal information. Second, SEEN learns to extract essential spatial cues from corresponding intensity frames and leverages a novel weight-copy scheme to convey spatial attention to the convolutional spiking layers during training and inference. We extensively validate and demonstrate the effectiveness of our approach on a specially collected Single-eye Event-based Emotion (SEE) dataset. To the best of our knowledge, our method is the first eye-based emotion recognition method that leverages event-based cameras and spiking neural network.

{{</citation>}}


## cs.SD (4)



### (66/70) U-Style: Cascading U-nets with Multi-level Speaker and Style Modeling for Zero-Shot Voice Cloning (Tao Li et al., 2023)

{{<citation>}}

Tao Li, Zhichao Wang, Xinfa Zhu, Jian Cong, Qiao Tian, Yuping Wang, Lei Xie. (2023)  
**U-Style: Cascading U-nets with Multi-level Speaker and Style Modeling for Zero-Shot Voice Cloning**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2310.04004v1)  

---


**ABSTRACT**  
Zero-shot speaker cloning aims to synthesize speech for any target speaker unseen during TTS system building, given only a single speech reference of the speaker at hand. Although more practical in real applications, the current zero-shot methods still produce speech with undesirable naturalness and speaker similarity. Moreover, endowing the target speaker with arbitrary speaking styles in the zero-shot setup has not been considered. This is because the unique challenge of zero-shot speaker and style cloning is to learn the disentangled speaker and style representations from only short references representing an arbitrary speaker and an arbitrary style. To address this challenge, we propose U-Style, which employs Grad-TTS as the backbone, particularly cascading a speaker-specific encoder and a style-specific encoder between the text encoder and the diffusion decoder. Thus, leveraging signal perturbation, U-Style is explicitly decomposed into speaker- and style-specific modeling parts, achieving better speaker and style disentanglement. To improve unseen speaker and style modeling ability, these two encoders conduct multi-level speaker and style modeling by skip-connected U-nets, incorporating the representation extraction and information reconstruction process. Besides, to improve the naturalness of synthetic speech, we adopt mean-based instance normalization and style adaptive layer normalization in these encoders to perform representation extraction and condition adaptation, respectively. Experiments show that U-Style significantly surpasses the state-of-the-art methods in unseen speaker cloning regarding naturalness and speaker similarity. Notably, U-Style can transfer the style from an unseen source speaker to another unseen target speaker, achieving flexible combinations of desired speaker timbre and style in zero-shot voice cloning.

{{</citation>}}


### (67/70) Layer-Adapted Implicit Distribution Alignment Networks for Cross-Corpus Speech Emotion Recognition (Yan Zhao et al., 2023)

{{<citation>}}

Yan Zhao, Yuan Zong, Jincen Wang, Hailun Lian, Cheng Lu, Li Zhao, Wenming Zheng. (2023)  
**Layer-Adapted Implicit Distribution Alignment Networks for Cross-Corpus Speech Emotion Recognition**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2310.03992v1)  

---


**ABSTRACT**  
In this paper, we propose a new unsupervised domain adaptation (DA) method called layer-adapted implicit distribution alignment networks (LIDAN) to address the challenge of cross-corpus speech emotion recognition (SER). LIDAN extends our previous ICASSP work, deep implicit distribution alignment networks (DIDAN), whose key contribution lies in the introduction of a novel regularization term called implicit distribution alignment (IDA). This term allows DIDAN trained on source (training) speech samples to remain applicable to predicting emotion labels for target (testing) speech samples, regardless of corpus variance in cross-corpus SER. To further enhance this method, we extend IDA to layer-adapted IDA (LIDA), resulting in LIDAN. This layer-adpated extention consists of three modified IDA terms that consider emotion labels at different levels of granularity. These terms are strategically arranged within different fully connected layers in LIDAN, aligning with the increasing emotion-discriminative abilities with respect to the layer depth. This arrangement enables LIDAN to more effectively learn emotion-discriminative and corpus-invariant features for SER across various corpora compared to DIDAN. It is also worthy to mention that unlike most existing methods that rely on estimating statistical moments to describe pre-assumed explicit distributions, both IDA and LIDA take a different approach. They utilize an idea of target sample reconstruction to directly bridge the feature distribution gap without making assumptions about their distribution type. As a result, DIDAN and LIDAN can be viewed as implicit cross-corpus SER methods. To evaluate LIDAN, we conducted extensive cross-corpus SER experiments on EmoDB, eNTERFACE, and CASIA corpora. The experimental results demonstrate that LIDAN surpasses recent state-of-the-art explicit unsupervised DA methods in tackling cross-corpus SER tasks.

{{</citation>}}


### (68/70) HuBERTopic: Enhancing Semantic Representation of HuBERT through Self-supervision Utilizing Topic Model (Takashi Maekaku et al., 2023)

{{<citation>}}

Takashi Maekaku, Jiatong Shi, Xuankai Chang, Yuya Fujita, Shinji Watanabe. (2023)  
**HuBERTopic: Enhancing Semantic Representation of HuBERT through Self-supervision Utilizing Topic Model**  

---
Primary Category: cs.SD  
Categories: cs-CL, cs-SD, cs.SD  
Keywords: BERT, Topic Model  
[Paper Link](http://arxiv.org/abs/2310.03975v1)  

---


**ABSTRACT**  
Recently, the usefulness of self-supervised representation learning (SSRL) methods has been confirmed in various downstream tasks. Many of these models, as exemplified by HuBERT and WavLM, use pseudo-labels generated from spectral features or the model's own representation features. From previous studies, it is known that the pseudo-labels contain semantic information. However, the masked prediction task, the learning criterion of HuBERT, focuses on local contextual information and may not make effective use of global semantic information such as speaker, theme of speech, and so on. In this paper, we propose a new approach to enrich the semantic representation of HuBERT. We apply topic model to pseudo-labels to generate a topic label for each utterance. An auxiliary topic classification task is added to HuBERT by using topic labels as teachers. This allows additional global semantic information to be incorporated in an unsupervised manner. Experimental results demonstrate that our method achieves comparable or better performance than the baseline in most tasks, including automatic speech recognition and five out of the eight SUPERB tasks. Moreover, we find that topic labels include various information about utterance, such as gender, speaker, and its theme. This highlights the effectiveness of our approach in capturing multifaceted semantic nuances.

{{</citation>}}


### (69/70) Zero-Shot Emotion Transfer For Cross-Lingual Speech Synthesis (Yuke Li et al., 2023)

{{<citation>}}

Yuke Li, Xinfa Zhu, Yi Lei, Hai Li, Junhui Liu, Danming Xie, Lei Xie. (2023)  
**Zero-Shot Emotion Transfer For Cross-Lingual Speech Synthesis**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: BERT, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2310.03963v1)  

---


**ABSTRACT**  
Zero-shot emotion transfer in cross-lingual speech synthesis aims to transfer emotion from an arbitrary speech reference in the source language to the synthetic speech in the target language. Building such a system faces challenges of unnatural foreign accents and difficulty in modeling the shared emotional expressions of different languages. Building on the DelightfulTTS neural architecture, this paper addresses these challenges by introducing specifically-designed modules to model the language-specific prosody features and language-shared emotional expressions separately. Specifically, the language-specific speech prosody is learned by a non-autoregressive predictive coding (NPC) module to improve the naturalness of the synthetic cross-lingual speech. The shared emotional expression between different languages is extracted from a pre-trained self-supervised model HuBERT with strong generalization capabilities. We further use hierarchical emotion modeling to capture more comprehensive emotions across different languages. Experimental results demonstrate the proposed framework's effectiveness in synthesizing bi-lingual emotional speech for the monolingual target speaker without emotional training data.

{{</citation>}}


## cs.HC (1)



### (70/70) From Text to Self: Users' Perceptions of Potential of AI on Interpersonal Communication and Self (Yue Fu et al., 2023)

{{<citation>}}

Yue Fu, Sami Foell, Xuhai Xu, Alexis Hiniker. (2023)  
**From Text to Self: Users' Perceptions of Potential of AI on Interpersonal Communication and Self**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2310.03976v1)  

---


**ABSTRACT**  
In the rapidly evolving landscape of AI-mediated communication (AIMC), tools powered by Large Language Models (LLMs) are becoming integral to interpersonal communication. Employing a mixed-methods approach, we conducted a one-week diary and interview study to explore users' perceptions of these tools' ability to: 1) support interpersonal communication in the short-term, and 2) lead to potential long-term effects. Our findings indicate that participants view AIMC support favorably, citing benefits such as increased communication confidence, and finding precise language to express their thoughts, navigating linguistic and cultural barriers. However, the study also uncovers current limitations of AIMC tools, including verbosity, unnatural responses, and excessive emotional intensity. These shortcomings are further exacerbated by user concerns about inauthenticity and potential overreliance on the technology. Furthermore, we identified four key communication spaces delineated by communication stakes (high or low) and relationship dynamics (formal or informal) that differentially predict users' attitudes toward AIMC tools. Specifically, participants found the tool is more suitable for communicating in formal relationships than informal ones and more beneficial in high-stakes than low-stakes communication.

{{</citation>}}
