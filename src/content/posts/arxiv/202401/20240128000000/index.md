---
draft: false
title: "arXiv @ 2024.01.28"
date: 2024-01-28
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2024"]
menu:
  sidebar:
    name: "arXiv @ 2024.01.28"
    identifier: arxiv_20240128
    parent: 202401_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.LG (15)](#cslg-15)
- [cs.CV (11)](#cscv-11)
- [cs.NE (3)](#csne-3)
- [cs.CL (16)](#cscl-16)
- [cs.CR (2)](#cscr-2)
- [cs.RO (3)](#csro-3)
- [eess.AS (1)](#eessas-1)
- [cs.NI (2)](#csni-2)
- [physics.soc-ph (1)](#physicssoc-ph-1)
- [cs.SE (6)](#csse-6)
- [econ.TH (1)](#econth-1)
- [cs.IR (4)](#csir-4)
- [cs.HC (2)](#cshc-2)
- [cs.SI (2)](#cssi-2)
- [cs.AI (3)](#csai-3)
- [cs.CY (1)](#cscy-1)
- [cs.SD (2)](#cssd-2)
- [cs.GT (1)](#csgt-1)
- [q-bio.QM (1)](#q-bioqm-1)
- [cs.IT (2)](#csit-2)
- [eess.IV (1)](#eessiv-1)

## cs.LG (15)



### (1/80) EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty (Yuhui Li et al., 2024)

{{<citation>}}

Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang. (2024)  
**EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2401.15077v1)  

---


**ABSTRACT**  
Auto-regressive decoding makes the inference of Large Language Models (LLMs) time-consuming. We propose a simple framework, EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency), for lossless acceleration. Unlike traditional speculative sampling methods, EAGLE operates the drafting process auto-regressively at the more regular (second-top-layer) feature level and addresses the sampling uncertainty issues in the next-feature prediction problems by integrating tokens from one time step ahead. The acceleration provided by EAGLE is lossless: it involves no fine-tuning of the target LLM, and the generated text maintains the same distribution as that of vanilla auto-regressive decoding. As of the submission of this paper, EAGLE is the fastest known framework within the speculative sampling family. On MT-bench, EAGLE is 3x faster than vanilla decoding, 2x faster than Lookahead, and 1.6x faster than Medusa. Using gpt-fast, EAGLE attains on average 160 tokens/s with LLaMA2-Chat 13B on a single RTX 3090 GPU, compared to 24 tokens/s of Huggingface's implementations.

{{</citation>}}


### (2/80) Fully Independent Communication in Multi-Agent Reinforcement Learning (Rafael Pina et al., 2024)

{{<citation>}}

Rafael Pina, Varuna De Silva, Corentin Artaud, Xiaolan Liu. (2024)  
**Fully Independent Communication in Multi-Agent Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-MA, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.15059v1)  

---


**ABSTRACT**  
Multi-Agent Reinforcement Learning (MARL) comprises a broad area of research within the field of multi-agent systems. Several recent works have focused specifically on the study of communication approaches in MARL. While multiple communication methods have been proposed, these might still be too complex and not easily transferable to more practical contexts. One of the reasons for that is due to the use of the famous parameter sharing trick. In this paper, we investigate how independent learners in MARL that do not share parameters can communicate. We demonstrate that this setting might incur into some problems, to which we propose a new learning scheme as a solution. Our results show that, despite the challenges, independent agents can still learn communication strategies following our method. Additionally, we use this method to investigate how communication in MARL is affected by different network capacities, both for sharing and not sharing parameters. We observe that communication may not always be needed and that the chosen agent network sizes need to be considered when used together with communication in order to achieve efficient learning.

{{</citation>}}


### (3/80) On the generalization capacity of neural networks during generic multimodal reasoning (Takuya Ito et al., 2024)

{{<citation>}}

Takuya Ito, Soham Dan, Mattia Rigotti, James Kozloski, Murray Campbell. (2024)  
**On the generalization capacity of neural networks during generic multimodal reasoning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.15030v1)  

---


**ABSTRACT**  
The advent of the Transformer has led to the development of large language models (LLM), which appear to demonstrate human-like capabilities. To assess the generality of this class of models and a variety of other base neural network architectures to multimodal domains, we evaluated and compared their capacity for multimodal generalization. We introduce a multimodal question-answer benchmark to evaluate three specific types of out-of-distribution (OOD) generalization performance: distractor generalization (generalization in the presence of distractors), systematic compositional generalization (generalization to new task permutations), and productive compositional generalization (generalization to more complex tasks structures). We found that across model architectures (e.g., RNNs, Transformers, Perceivers, etc.), models with multiple attention layers, or models that leveraged cross-attention mechanisms between input domains, fared better. Our positive results demonstrate that for multimodal distractor and systematic generalization, either cross-modal attention or models with deeper attention layers are key architectural features required to integrate multimodal inputs. On the other hand, neither of these architectural features led to productive generalization, suggesting fundamental limitations of existing architectures for specific types of multimodal generalization. These results demonstrate the strengths and limitations of specific architectural components underlying modern neural models for multimodal reasoning. Finally, we provide Generic COG (gCOG), a configurable benchmark with several multimodal generalization splits, for future studies to explore.

{{</citation>}}


### (4/80) SliceGPT: Compress Large Language Models by Deleting Rows and Columns (Saleh Ashkboos et al., 2024)

{{<citation>}}

Saleh Ashkboos, Maximilian L. Croci, Marcelo Gennari do Nascimento, Torsten Hoefler, James Hensman. (2024)  
**SliceGPT: Compress Large Language Models by Deleting Rows and Columns**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: GPT, Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2401.15024v1)  

---


**ABSTRACT**  
Large language models have become the cornerstone of natural language processing, but their use comes with substantial costs in terms of compute and memory resources. Sparsification provides a solution to alleviate these resource constraints, and recent works have shown that trained models can be sparsified post-hoc. Existing sparsification techniques face challenges as they need additional data structures and offer constrained speedup with current hardware. In this paper we present SliceGPT, a new post-training sparsification scheme which replaces each weight matrix with a smaller (dense) matrix, reducing the embedding dimension of the network. Through extensive experimentation, we show that SliceGPT can remove up to 25% of the model parameters (including embeddings) for LLAMA2-70B, OPT 66B and Phi-2 models while maintaining 99%, 99% and 90% zero-shot task performance of the dense model respectively. Our sliced models run on fewer GPUs and run faster without any additional code optimization: on 24GB consumer GPUs we reduce the total compute for inference on LLAMA2-70B to 64% of that of the dense model; on 40GB A100 GPUs we reduce it to 66%. We offer a new insight, computational invariance in transformer networks, which enables SliceGPT and we hope it will inspire and enable future avenues to reduce memory and computation demands for pre-trained models. Code is available at: https://github.com/microsoft/TransformerCompression

{{</citation>}}


### (5/80) Graph-based Active Learning for Entity Cluster Repair (Victor Christen et al., 2024)

{{<citation>}}

Victor Christen, Daniel Obraczka, Marvin Hofer, Martin Franke, Erhard Rahm. (2024)  
**Graph-based Active Learning for Entity Cluster Repair**  

---
Primary Category: cs.LG  
Categories: cs-DB, cs-LG, cs.LG  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2401.14992v1)  

---


**ABSTRACT**  
Cluster repair methods aim to determine errors in clusters and modify them so that each cluster consists of records representing the same entity. Current cluster repair methodologies primarily assume duplicate-free data sources, where each record from one source corresponds to a unique record from another. However, real-world data often deviates from this assumption due to quality issues. Recent approaches apply clustering methods in combination with link categorization methods so they can be applied to data sources with duplicates. Nevertheless, the results do not show a clear picture since the quality highly varies depending on the configuration and dataset. In this study, we introduce a novel approach for cluster repair that utilizes graph metrics derived from the underlying similarity graphs. These metrics are pivotal in constructing a classification model to distinguish between correct and incorrect edges. To address the challenge of limited training data, we integrate an active learning mechanism tailored to cluster-specific attributes. The evaluation shows that the method outperforms existing cluster repair methods without distinguishing between duplicate-free or dirty data sources. Notably, our modified active learning strategy exhibits enhanced performance when dealing with datasets containing duplicates, showcasing its effectiveness in such scenarios.

{{</citation>}}


### (6/80) Learning Universal Predictors (Jordi Grau-Moya et al., 2024)

{{<citation>}}

Jordi Grau-Moya, Tim Genewein, Marcus Hutter, Laurent Orseau, Grégoire Delétang, Elliot Catt, Anian Ruoss, Li Kevin Wenliang, Christopher Mattern, Matthew Aitchison, Joel Veness. (2024)  
**Learning Universal Predictors**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: LSTM, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.14953v1)  

---


**ABSTRACT**  
Meta-learning has emerged as a powerful approach to train neural networks to learn new tasks quickly from limited data. Broad exposure to different tasks leads to versatile representations enabling general problem solving. But, what are the limits of meta-learning? In this work, we explore the potential of amortizing the most powerful universal predictor, namely Solomonoff Induction (SI), into neural networks via leveraging meta-learning to its limits. We use Universal Turing Machines (UTMs) to generate training data used to expose networks to a broad range of patterns. We provide theoretical analysis of the UTM data generation processes and meta-training protocols. We conduct comprehensive experiments with neural architectures (e.g. LSTMs, Transformers) and algorithmic data generators of varying complexity and universality. Our results suggest that UTM data is a valuable resource for meta-learning, and that it can be used to train neural networks capable of learning universal prediction strategies.

{{</citation>}}


### (7/80) Conserve-Update-Revise to Cure Generalization and Robustness Trade-off in Adversarial Training (Shruthi Gowda et al., 2024)

{{<citation>}}

Shruthi Gowda, Bahram Zonooz, Elahe Arani. (2024)  
**Conserve-Update-Revise to Cure Generalization and Robustness Trade-off in Adversarial Training**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: Adversarial Training  
[Paper Link](http://arxiv.org/abs/2401.14948v1)  

---


**ABSTRACT**  
Adversarial training improves the robustness of neural networks against adversarial attacks, albeit at the expense of the trade-off between standard and robust generalization. To unveil the underlying factors driving this phenomenon, we examine the layer-wise learning capabilities of neural networks during the transition from a standard to an adversarial setting. Our empirical findings demonstrate that selectively updating specific layers while preserving others can substantially enhance the network's learning capacity. We therefore propose CURE, a novel training framework that leverages a gradient prominence criterion to perform selective conservation, updating, and revision of weights. Importantly, CURE is designed to be dataset- and architecture-agnostic, ensuring its applicability across various scenarios. It effectively tackles both memorization and overfitting issues, thus enhancing the trade-off between robustness and generalization and additionally, this training approach also aids in mitigating "robust overfitting". Furthermore, our study provides valuable insights into the mechanisms of selective adversarial training and offers a promising avenue for future research.

{{</citation>}}


### (8/80) A structured regression approach for evaluating model performance across intersectional subgroups (Christine Herlihy et al., 2024)

{{<citation>}}

Christine Herlihy, Kimberly Truong, Alexandra Chouldechova, Miroslav Dudik. (2024)  
**A structured regression approach for evaluating model performance across intersectional subgroups**  

---
Primary Category: cs.LG  
Categories: cs-CY, cs-LG, cs.LG, stat-AP, stat-ML  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.14893v1)  

---


**ABSTRACT**  
Disaggregated evaluation is a central task in AI fairness assessment, with the goal to measure an AI system's performance across different subgroups defined by combinations of demographic or other sensitive attributes. The standard approach is to stratify the evaluation data across subgroups and compute performance metrics separately for each group. However, even for moderately-sized evaluation datasets, sample sizes quickly get small once considering intersectional subgroups, which greatly limits the extent to which intersectional groups are considered in many disaggregated evaluations. In this work, we introduce a structured regression approach to disaggregated evaluation that we demonstrate can yield reliable system performance estimates even for very small subgroups. We also provide corresponding inference strategies for constructing confidence intervals and explore how goodness-of-fit testing can yield insight into the structure of fairness-related harms experienced by intersectional groups. We evaluate our approach on two publicly available datasets, and several variants of semi-synthetic data. The results show that our method is considerably more accurate than the standard approach, especially for small subgroups, and goodness-of-fit testing helps identify the key factors that drive differences in performance.

{{</citation>}}


### (9/80) Cross-Space Adaptive Filter: Integrating Graph Topology and Node Attributes for Alleviating the Over-smoothing Problem (Chen Huang et al., 2024)

{{<citation>}}

Chen Huang, Haoyang Li, Yifan Zhang, Wenqiang Lei, Jiancheng Lv. (2024)  
**Cross-Space Adaptive Filter: Integrating Graph Topology and Node Attributes for Alleviating the Over-smoothing Problem**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Graph Convolutional Network  
[Paper Link](http://arxiv.org/abs/2401.14876v1)  

---


**ABSTRACT**  
The vanilla Graph Convolutional Network (GCN) uses a low-pass filter to extract low-frequency signals from graph topology, which may lead to the over-smoothing problem when GCN goes deep. To this end, various methods have been proposed to create an adaptive filter by incorporating an extra filter (e.g., a high-pass filter) extracted from the graph topology. However, these methods heavily rely on topological information and ignore the node attribute space, which severely sacrifices the expressive power of the deep GCNs, especially when dealing with disassortative graphs. In this paper, we propose a cross-space adaptive filter, called CSF, to produce the adaptive-frequency information extracted from both the topology and attribute spaces. Specifically, we first derive a tailored attribute-based high-pass filter that can be interpreted theoretically as a minimizer for semi-supervised kernel ridge regression. Then, we cast the topology-based low-pass filter as a Mercer's kernel within the context of GCNs. This serves as a foundation for combining it with the attribute-based filter to capture the adaptive-frequency information. Finally, we derive the cross-space filter via an effective multiple-kernel learning strategy, which unifies the attribute-based high-pass filter and the topology-based low-pass filter. This helps to address the over-smoothing problem while maintaining effectiveness. Extensive experiments demonstrate that CSF not only successfully alleviates the over-smoothing problem but also promotes the effectiveness of the node classification task.

{{</citation>}}


### (10/80) Off-Policy Primal-Dual Safe Reinforcement Learning (Zifan Wu et al., 2024)

{{<citation>}}

Zifan Wu, Bo Tang, Qian Lin, Chao Yu, Shangqin Mao, Qianlong Xie, Xingxing Wang, Dong Wang. (2024)  
**Off-Policy Primal-Dual Safe Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.14758v1)  

---


**ABSTRACT**  
Primal-dual safe RL methods commonly perform iterations between the primal update of the policy and the dual update of the Lagrange Multiplier. Such a training paradigm is highly susceptible to the error in cumulative cost estimation since this estimation serves as the key bond connecting the primal and dual update processes. We show that this problem causes significant underestimation of cost when using off-policy methods, leading to the failure to satisfy the safety constraint. To address this issue, we propose \textit{conservative policy optimization}, which learns a policy in a constraint-satisfying area by considering the uncertainty in cost estimation. This improves constraint satisfaction but also potentially hinders reward maximization. We then introduce \textit{local policy convexification} to help eliminate such suboptimality by gradually reducing the estimation uncertainty. We provide theoretical interpretations of the joint coupling effect of these two ingredients and further verify them by extensive experiments. Results on benchmark tasks show that our method not only achieves an asymptotic performance comparable to state-of-the-art on-policy methods while using much fewer samples, but also significantly reduces constraint violation during training. Our code is available at https://github.com/ZifanWu/CAL.

{{</citation>}}


### (11/80) Residual Quantization with Implicit Neural Codebooks (Iris Huijben et al., 2024)

{{<citation>}}

Iris Huijben, Matthijs Douze, Matthew Muckley, Ruud van Sloun, Jakob Verbeek. (2024)  
**Residual Quantization with Implicit Neural Codebooks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2401.14732v1)  

---


**ABSTRACT**  
Vector quantization is a fundamental operation for data compression and vector search. To obtain high accuracy, multi-codebook methods increase the rate by representing each vector using codewords across multiple codebooks. Residual quantization (RQ) is one such method, which increases accuracy by iteratively quantizing the error of the previous step. The error distribution is dependent on previously selected codewords. This dependency is, however, not accounted for in conventional RQ as it uses a generic codebook per quantization step. In this paper, we propose QINCo, a neural RQ variant which predicts specialized codebooks per vector using a neural network that is conditioned on the approximation of the vector from previous steps. Experiments show that QINCo outperforms state-of-the-art methods by a large margin on several datasets and code sizes. For example, QINCo achieves better nearest-neighbor search accuracy using 12 bytes codes than other methods using 16 bytes on the BigANN and Deep1B dataset.

{{</citation>}}


### (12/80) TA-RNN: an Attention-based Time-aware Recurrent Neural Network Architecture for Electronic Health Records (Mohammad Al Olaimat et al., 2024)

{{<citation>}}

Mohammad Al Olaimat, Serdar Bozdag, the Alzheimer's Disease Neuroimaging Initiative. (2024)  
**TA-RNN: an Attention-based Time-aware Recurrent Neural Network Architecture for Electronic Health Records**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2401.14694v1)  

---


**ABSTRACT**  
Motivation: Electronic Health Records (EHR) represent a comprehensive resource of a patient's medical history. EHR are essential for utilizing advanced technologies such as deep learning (DL), enabling healthcare providers to analyze extensive data, extract valuable insights, and make precise and data-driven clinical decisions. DL methods such as Recurrent Neural Networks (RNN) have been utilized to analyze EHR to model disease progression and predict diagnosis. However, these methods do not address some inherent irregularities in EHR data such as irregular time intervals between clinical visits. Furthermore, most DL models are not interpretable. In this study, we propose two interpretable DL architectures based on RNN, namely Time-Aware RNN (TA-RNN) and TA-RNN-Autoencoder (TA-RNN-AE) to predict patient's clinical outcome in EHR at next visit and multiple visits ahead, respectively. To mitigate the impact of irregular time intervals, we propose incorporating time embedding of the elapsed times between visits. For interpretability, we propose employing a dual-level attention mechanism that operates between visits and features within each visit.   Results: The results of the experiments conducted on Alzheimer's Disease Neuroimaging Initiative (ADNI) and National Alzheimer's Coordinating Center (NACC) datasets indicated superior performance of proposed models for predicting Alzheimer's Disease (AD) compared to state-of-the-art and baseline approaches based on F2 and sensitivity. Additionally, TA-RNN showed superior performance on Medical Information Mart for Intensive Care (MIMIC-III) dataset for mortality prediction. In our ablation study, we observed enhanced predictive performance by incorporating time embedding and attention mechanisms. Finally, investigating attention weights helped identify influential visits and features in predictions.   Availability: https://github.com/bozdaglab/TA-RNN

{{</citation>}}


### (13/80) Omnipredictors for Regression and the Approximate Rank of Convex Functions (Parikshit Gopalan et al., 2024)

{{<citation>}}

Parikshit Gopalan, Princewill Okoroafor, Prasad Raghavendra, Abhishek Shetty, Mihir Singhal. (2024)  
**Omnipredictors for Regression and the Approximate Rank of Convex Functions**  

---
Primary Category: cs.LG  
Categories: cs-CC, cs-DS, cs-LG, cs.LG  
Keywords: GLM  
[Paper Link](http://arxiv.org/abs/2401.14645v1)  

---


**ABSTRACT**  
Consider the supervised learning setting where the goal is to learn to predict labels $\mathbf y$ given points $\mathbf x$ from a distribution. An \textit{omnipredictor} for a class $\mathcal L$ of loss functions and a class $\mathcal C$ of hypotheses is a predictor whose predictions incur less expected loss than the best hypothesis in $\mathcal C$ for every loss in $\mathcal L$. Since the work of [GKR+21] that introduced the notion, there has been a large body of work in the setting of binary labels where $\mathbf y \in \{0, 1\}$, but much less is known about the regression setting where $\mathbf y \in [0,1]$ can be continuous. Our main conceptual contribution is the notion of \textit{sufficient statistics} for loss minimization over a family of loss functions: these are a set of statistics about a distribution such that knowing them allows one to take actions that minimize the expected loss for any loss in the family. The notion of sufficient statistics relates directly to the approximate rank of the family of loss functions.   Our key technical contribution is a bound of $O(1/\varepsilon^{2/3})$ on the $\epsilon$-approximate rank of convex, Lipschitz functions on the interval $[0,1]$, which we show is tight up to a factor of $\mathrm{polylog} (1/\epsilon)$. This yields improved runtimes for learning omnipredictors for the class of all convex, Lipschitz loss functions under weak learnability assumptions about the class $\mathcal C$. We also give efficient omnipredictors when the loss families have low-degree polynomial approximations, or arise from generalized linear models (GLMs). This translation from sufficient statistics to faster omnipredictors is made possible by lifting the technique of loss outcome indistinguishability introduced by [GKH+23] for Boolean labels to the regression setting.

{{</citation>}}


### (14/80) Design Your Own Universe: A Physics-Informed Agnostic Method for Enhancing Graph Neural Networks (Dai Shi et al., 2024)

{{<citation>}}

Dai Shi, Andi Han, Lequan Lin, Yi Guo, Zhiyong Wang, Junbin Gao. (2024)  
**Design Your Own Universe: A Physics-Informed Agnostic Method for Enhancing Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2401.14580v1)  

---


**ABSTRACT**  
Physics-informed Graph Neural Networks have achieved remarkable performance in learning through graph-structured data by mitigating common GNN challenges such as over-smoothing, over-squashing, and heterophily adaption. Despite these advancements, the development of a simple yet effective paradigm that appropriately integrates previous methods for handling all these challenges is still underway. In this paper, we draw an analogy between the propagation of GNNs and particle systems in physics, proposing a model-agnostic enhancement framework. This framework enriches the graph structure by introducing additional nodes and rewiring connections with both positive and negative weights, guided by node labeling information. We theoretically verify that GNNs enhanced through our approach can effectively circumvent the over-smoothing issue and exhibit robustness against over-squashing. Moreover, we conduct a spectral analysis on the rewired graph to demonstrate that the corresponding GNNs can fit both homophilic and heterophilic graphs. Empirical validations on benchmarks for homophilic, heterophilic graphs, and long-term graph datasets show that GNNs enhanced by our method significantly outperform their original counterparts.

{{</citation>}}


### (15/80) GOAt: Explaining Graph Neural Networks via Graph Output Attribution (Shengyao Lu et al., 2024)

{{<citation>}}

Shengyao Lu, Keith G. Mills, Jiao He, Bang Liu, Di Niu. (2024)  
**GOAt: Explaining Graph Neural Networks via Graph Output Attribution**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2401.14578v1)  

---


**ABSTRACT**  
Understanding the decision-making process of Graph Neural Networks (GNNs) is crucial to their interpretability. Most existing methods for explaining GNNs typically rely on training auxiliary models, resulting in the explanations remain black-boxed. This paper introduces Graph Output Attribution (GOAt), a novel method to attribute graph outputs to input graph features, creating GNN explanations that are faithful, discriminative, as well as stable across similar samples. By expanding the GNN as a sum of scalar products involving node features, edge features and activation patterns, we propose an efficient analytical method to compute contribution of each node or edge feature to each scalar product and aggregate the contributions from all scalar products in the expansion form to derive the importance of each node and edge. Through extensive experiments on synthetic and real-world data, we show that our method not only outperforms various state-ofthe-art GNN explainers in terms of the commonly used fidelity metric, but also exhibits stronger discriminability, and stability by a remarkable margin.

{{</citation>}}


## cs.CV (11)



### (16/80) From GPT-4 to Gemini and Beyond: Assessing the Landscape of MLLMs on Generalizability, Trustworthiness and Causality through Four Modalities (Chaochao Lu et al., 2024)

{{<citation>}}

Chaochao Lu, Chen Qian, Guodong Zheng, Hongxing Fan, Hongzhi Gao, Jie Zhang, Jing Shao, Jingyi Deng, Jinlan Fu, Kexin Huang, Kunchang Li, Lijun Li, Limin Wang, Lu Sheng, Meiqi Chen, Ming Zhang, Qibing Ren, Sirui Chen, Tao Gui, Wanli Ouyang, Yali Wang, Yan Teng, Yaru Wang, Yi Wang, Yinan He, Yingchun Wang, Yixu Wang, Yongting Zhang, Yu Qiao, Yujiong Shen, Yurong Mou, Yuxi Chen, Zaibin Zhang, Zhelun Shi, Zhenfei Yin, Zhipin Wang. (2024)  
**From GPT-4 to Gemini and Beyond: Assessing the Landscape of MLLMs on Generalizability, Trustworthiness and Causality through Four Modalities**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, GPT, GPT-4, Google, Language Model  
[Paper Link](http://arxiv.org/abs/2401.15071v1)  

---


**ABSTRACT**  
Multi-modal Large Language Models (MLLMs) have shown impressive abilities in generating reasonable responses with respect to multi-modal contents. However, there is still a wide gap between the performance of recent MLLM-based applications and the expectation of the broad public, even though the most powerful OpenAI's GPT-4 and Google's Gemini have been deployed. This paper strives to enhance understanding of the gap through the lens of a qualitative study on the generalizability, trustworthiness, and causal reasoning capabilities of recent proprietary and open-source MLLMs across four modalities: ie, text, code, image, and video, ultimately aiming to improve the transparency of MLLMs. We believe these properties are several representative factors that define the reliability of MLLMs, in supporting various downstream applications. To be specific, we evaluate the closed-source GPT-4 and Gemini and 6 open-source LLMs and MLLMs. Overall we evaluate 230 manually designed cases, where the qualitative results are then summarized into 12 scores (ie, 4 modalities times 3 properties). In total, we uncover 14 empirical findings that are useful to understand the capabilities and limitations of both proprietary and open-source MLLMs, towards more reliable downstream multi-modal applications.

{{</citation>}}


### (17/80) Unrecognizable Yet Identifiable: Image Distortion with Preserved Embeddings (Dmytro Zakharov et al., 2024)

{{<citation>}}

Dmytro Zakharov, Oleksandr Kuznetsov, Emanuele Frontoni. (2024)  
**Unrecognizable Yet Identifiable: Image Distortion with Preserved Embeddings**  

---
Primary Category: cs.CV  
Categories: cs-CR, cs-CV, cs.CV  
Keywords: AI, Embedding  
[Paper Link](http://arxiv.org/abs/2401.15048v1)  

---


**ABSTRACT**  
In the realm of security applications, biometric authentication systems play a crucial role, yet one often encounters challenges concerning privacy and security while developing one. One of the most fundamental challenges lies in avoiding storing biometrics directly in the storage but still achieving decently high accuracy. Addressing this issue, we contribute to both artificial intelligence and engineering fields. We introduce an innovative image distortion technique that effectively renders facial images unrecognizable to the eye while maintaining their identifiability by neural network models. From the theoretical perspective, we explore how reliable state-of-the-art biometrics recognition neural networks are by checking the maximal degree of image distortion, which leaves the predicted identity unchanged. On the other hand, applying this technique demonstrates a practical solution to the engineering challenge of balancing security, precision, and performance in biometric authentication systems. Through experimenting on the widely used datasets, we assess the effectiveness of our method in preserving AI feature representation and distorting relative to conventional metrics. We also compare our method with previously used approaches.

{{</citation>}}


### (18/80) DAM: Diffusion Activation Maximization for 3D Global Explanations (Hanxiao Tan, 2024)

{{<citation>}}

Hanxiao Tan. (2024)  
**DAM: Diffusion Activation Maximization for 3D Global Explanations**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.14938v1)  

---


**ABSTRACT**  
In recent years, the performance of point cloud models has been rapidly improved. However, due to the limited amount of relevant explainability studies, the unreliability and opacity of these black-box models may lead to potential risks in applications where human lives are at stake, e.g. autonomous driving or healthcare. This work proposes a DDPM-based point cloud global explainability method (DAM) that leverages Point Diffusion Transformer (PDT), a novel point-wise symmetric model, with dual-classifier guidance to generate high-quality global explanations. In addition, an adapted path gradient integration method for DAM is proposed, which not only provides a global overview of the saliency maps for point cloud categories, but also sheds light on how the attributions of the explanations vary during the generation process. Extensive experiments indicate that our method outperforms existing ones in terms of perceptibility, representativeness, and diversity, with a significant reduction in generation time. Our code is available at: https://github.com/Explain3D/DAM

{{</citation>}}


### (19/80) MPTQ-ViT:Mixed-PrecisionPost-TrainingQuantizationforVisionTransformer (Yu-Shan Tai et al., 2024)

{{<citation>}}

Yu-Shan Tai, An-Yeu, Wu. (2024)  
**MPTQ-ViT:Mixed-PrecisionPost-TrainingQuantizationforVisionTransformer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Quantization, Transformer  
[Paper Link](http://arxiv.org/abs/2401.14895v1)  

---


**ABSTRACT**  
While vision transformers (ViTs) have shown great potential in computer vision tasks, their intense computation and memory requirements pose challenges for practical applications. Existing post-training quantization methods leverage value redistribution or specialized quantizers to address the non-normal distribution in ViTs. However, without considering the asymmetry in activations and relying on hand-crafted settings, these methods often struggle to maintain performance under low-bit quantization. To overcome these challenges, we introduce SmoothQuant with bias term (SQ-b) to alleviate the asymmetry issue and reduce the clamping loss. We also introduce optimal scaling factor ratio search (OPT-m) to determine quantization parameters by a data-dependent mechanism automatically. To further enhance the compressibility, we incorporate the above-mentioned techniques and propose a mixed-precision post-training quantization framework for vision transformers (MPTQ-ViT). We develop greedy mixed-precision quantization (Greedy MP) to allocate layer-wise bit-width considering both model performance and compressibility. Our experiments on ViT, DeiT, and Swin demonstrate significant accuracy improvements compared with SOTA on the ImageNet dataset. Specifically, our proposed methods achieve accuracy improvements ranging from 0.90% to 23.35% on 4-bit ViTs with single-precision and from 3.82% to 78.14% on 5-bit fully quantized ViTs with mixed-precision.

{{</citation>}}


### (20/80) Memory-Inspired Temporal Prompt Interaction for Text-Image Classification (Xinyao Yu et al., 2024)

{{<citation>}}

Xinyao Yu, Hao Sun, Ziwei Niu, Rui Qin, Zhenjia Bai, Yen-Wei Chen, Lanfen Lin. (2024)  
**Memory-Inspired Temporal Prompt Interaction for Text-Image Classification**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2401.14856v1)  

---


**ABSTRACT**  
In recent years, large-scale pre-trained multimodal models (LMM) generally emerge to integrate the vision and language modalities, achieving considerable success in various natural language processing and computer vision tasks. The growing size of LMMs, however, results in a significant computational cost for fine-tuning these models for downstream tasks. Hence, prompt-based interaction strategy is studied to align modalities more efficiently. In this contex, we propose a novel prompt-based multimodal interaction strategy inspired by human memory strategy, namely Memory-Inspired Temporal Prompt Interaction (MITP). Our proposed method involves in two stages as in human memory strategy: the acquiring stage, and the consolidation and activation stage. We utilize temporal prompts on intermediate layers to imitate the acquiring stage, leverage similarity-based prompt interaction to imitate memory consolidation, and employ prompt generation strategy to imitate memory activation. The main strength of our paper is that we interact the prompt vectors on intermediate layers to leverage sufficient information exchange between modalities, with compressed trainable parameters and memory usage. We achieve competitive results on several datasets with relatively small memory usage and 2.0M of trainable parameters (about 1% of the pre-trained foundation model).

{{</citation>}}


### (21/80) Adaptive Point Transformer (Alessandro Baiocchi et al., 2024)

{{<citation>}}

Alessandro Baiocchi, Indro Spinelli, Alessandro Nicolosi, Simone Scardapane. (2024)  
**Adaptive Point Transformer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.14845v1)  

---


**ABSTRACT**  
The recent surge in 3D data acquisition has spurred the development of geometric deep learning models for point cloud processing, boosted by the remarkable success of transformers in natural language processing. While point cloud transformers (PTs) have achieved impressive results recently, their quadratic scaling with respect to the point cloud size poses a significant scalability challenge for real-world applications. To address this issue, we propose the Adaptive Point Cloud Transformer (AdaPT), a standard PT model augmented by an adaptive token selection mechanism. AdaPT dynamically reduces the number of tokens during inference, enabling efficient processing of large point clouds. Furthermore, we introduce a budget mechanism to flexibly adjust the computational cost of the model at inference time without the need for retraining or fine-tuning separate models. Our extensive experimental evaluation on point cloud classification tasks demonstrates that AdaPT significantly reduces computational complexity while maintaining competitive accuracy compared to standard PTs. The code for AdaPT is made publicly available.

{{</citation>}}


### (22/80) PL-FSCIL: Harnessing the Power of Prompts for Few-Shot Class-Incremental Learning (Songsong Tian et al., 2024)

{{<citation>}}

Songsong Tian, Lusi Li, Weijun Li, Hang Ran, Li Li, Xin Ning. (2024)  
**PL-FSCIL: Harnessing the Power of Prompts for Few-Shot Class-Incremental Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Few-Shot, Transformer  
[Paper Link](http://arxiv.org/abs/2401.14807v1)  

---


**ABSTRACT**  
Few-Shot Class-Incremental Learning (FSCIL) aims to enable deep neural networks to learn new tasks incrementally from a small number of labeled samples without forgetting previously learned tasks, closely mimicking human learning patterns. In this paper, we propose a novel approach called Prompt Learning for FSCIL (PL-FSCIL), which harnesses the power of prompts in conjunction with a pre-trained Vision Transformer (ViT) model to address the challenges of FSCIL effectively. Our work pioneers the use of visual prompts in FSCIL, which is characterized by its notable simplicity. PL-FSCIL consists of two distinct prompts: the Domain Prompt and the FSCIL Prompt. Both are vectors that augment the model by embedding themselves into the attention layer of the ViT model. Specifically, the Domain Prompt assists the ViT model in adapting to new data domains. The task-specific FSCIL Prompt, coupled with a prototype classifier, amplifies the model's ability to effectively handle FSCIL tasks. We validate the efficacy of PL-FSCIL on widely used benchmark datasets such as CIFAR-100 and CUB-200. The results showcase competitive performance, underscoring its promising potential for real-world applications where high-quality data is often scarce. The source code is available at: https://github.com/TianSongS/PL-FSCIL.

{{</citation>}}


### (23/80) VJT: A Video Transformer on Joint Tasks of Deblurring, Low-light Enhancement and Denoising (Yuxiang Hui et al., 2024)

{{<citation>}}

Yuxiang Hui, Yang Liu, Yaofang Liu, Fan Jia, Jinshan Pan, Raymond Chan, Tieyong Zeng. (2024)  
**VJT: A Video Transformer on Joint Tasks of Deblurring, Low-light Enhancement and Denoising**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.14754v1)  

---


**ABSTRACT**  
Video restoration task aims to recover high-quality videos from low-quality observations. This contains various important sub-tasks, such as video denoising, deblurring and low-light enhancement, since video often faces different types of degradation, such as blur, low light, and noise. Even worse, these kinds of degradation could happen simultaneously when taking videos in extreme environments. This poses significant challenges if one wants to remove these artifacts at the same time. In this paper, to the best of our knowledge, we are the first to propose an efficient end-to-end video transformer approach for the joint task of video deblurring, low-light enhancement, and denoising. This work builds a novel multi-tier transformer where each tier uses a different level of degraded video as a target to learn the features of video effectively. Moreover, we carefully design a new tier-to-tier feature fusion scheme to learn video features incrementally and accelerate the training process with a suitable adaptive weighting scheme. We also provide a new Multiscene-Lowlight-Blur-Noise (MLBN) dataset, which is generated according to the characteristics of the joint task based on the RealBlur dataset and YouTube videos to simulate realistic scenes as far as possible. We have conducted extensive experiments, compared with many previous state-of-the-art methods, to show the effectiveness of our approach clearly.

{{</citation>}}


### (24/80) Sketch and Refine: Towards Fast and Accurate Lane Detection (Chao Chen et al., 2024)

{{<citation>}}

Chao Chen, Jie Liu, Chang Zhou, Jie Tang, Gangshan Wu. (2024)  
**Sketch and Refine: Towards Fast and Accurate Lane Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Sketch  
[Paper Link](http://arxiv.org/abs/2401.14729v1)  

---


**ABSTRACT**  
Lane detection is to determine the precise location and shape of lanes on the road. Despite efforts made by current methods, it remains a challenging task due to the complexity of real-world scenarios. Existing approaches, whether proposal-based or keypoint-based, suffer from depicting lanes effectively and efficiently. Proposal-based methods detect lanes by distinguishing and regressing a collection of proposals in a streamlined top-down way, yet lack sufficient flexibility in lane representation. Keypoint-based methods, on the other hand, construct lanes flexibly from local descriptors, which typically entail complicated post-processing. In this paper, we present a "Sketch-and-Refine" paradigm that utilizes the merits of both keypoint-based and proposal-based methods. The motivation is that local directions of lanes are semantically simple and clear. At the "Sketch" stage, local directions of keypoints can be easily estimated by fast convolutional layers. Then we can build a set of lane proposals accordingly with moderate accuracy. At the "Refine" stage, we further optimize these proposals via a novel Lane Segment Association Module (LSAM), which allows adaptive lane segment adjustment. Last but not least, we propose multi-level feature integration to enrich lane feature representations more efficiently. Based on the proposed "Sketch and Refine" paradigm, we propose a fast yet effective lane detector dubbed "SRLane". Experiments show that our SRLane can run at a fast speed (i.e., 278 FPS) while yielding an F1 score of 78.9\%. The source code is available at: https://github.com/passerer/SRLane.

{{</citation>}}


### (25/80) SSR: SAM is a Strong Regularizer for domain adaptive semantic segmentation (Yanqi Ge et al., 2024)

{{<citation>}}

Yanqi Ge, Ye Huang, Wen Li, Lixin Duan. (2024)  
**SSR: SAM is a Strong Regularizer for domain adaptive semantic segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2401.14686v1)  

---


**ABSTRACT**  
We introduced SSR, which utilizes SAM (segment-anything) as a strong regularizer during training, to greatly enhance the robustness of the image encoder for handling various domains. Specifically, given the fact that SAM is pre-trained with a large number of images over the internet, which cover a diverse variety of domains, the feature encoding extracted by the SAM is obviously less dependent on specific domains when compared to the traditional ImageNet pre-trained image encoder. Meanwhile, the ImageNet pre-trained image encoder is still a mature choice of backbone for the semantic segmentation task, especially when the SAM is category-irrelevant. As a result, our SSR provides a simple yet highly effective design. It uses the ImageNet pre-trained image encoder as the backbone, and the intermediate feature of each stage (ie there are 4 stages in MiT-B5) is regularized by SAM during training. After extensive experimentation on GTA5$\rightarrow$Cityscapes, our SSR significantly improved performance over the baseline without introducing any extra inference overhead.

{{</citation>}}


### (26/80) From Blurry to Brilliant Detection: YOLOv5-Based Aerial Object Detection with Super Resolution (Ragib Amin Nihal et al., 2024)

{{<citation>}}

Ragib Amin Nihal, Benjamin Yen, Katsutoshi Itoyama, Kazuhiro Nakadai. (2024)  
**From Blurry to Brilliant Detection: YOLOv5-Based Aerial Object Detection with Super Resolution**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: AI, Drone, Object Detection, Transformer  
[Paper Link](http://arxiv.org/abs/2401.14661v1)  

---


**ABSTRACT**  
The demand for accurate object detection in aerial imagery has surged with the widespread use of drones and satellite technology. Traditional object detection models, trained on datasets biased towards large objects, struggle to perform optimally in aerial scenarios where small, densely clustered objects are prevalent. To address this challenge, we present an innovative approach that combines super-resolution and an adapted lightweight YOLOv5 architecture. We employ a range of datasets, including VisDrone-2023, SeaDroneSee, VEDAI, and NWPU VHR-10, to evaluate our model's performance. Our Super Resolved YOLOv5 architecture features Transformer encoder blocks, allowing the model to capture global context and context information, leading to improved detection results, especially in high-density, occluded conditions. This lightweight model not only delivers improved accuracy but also ensures efficient resource utilization, making it well-suited for real-time applications. Our experimental results demonstrate the model's superior performance in detecting small and densely clustered objects, underlining the significance of dataset choice and architectural adaptation for this specific task. In particular, the method achieves 52.5% mAP on VisDrone, exceeding top prior works. This approach promises to significantly advance object detection in aerial imagery, contributing to more accurate and reliable results in a variety of real-world applications.

{{</citation>}}


## cs.NE (3)



### (27/80) Digital-analog hybrid matrix multiplication processor for optical neural networks (Xiansong Meng et al., 2024)

{{<citation>}}

Xiansong Meng, Deming Kong, Kwangwoong Kim, Qiuchi Li, Po Dong, Ingemar J. Cox, Christina Lioma, Hao Hu. (2024)  
**Digital-analog hybrid matrix multiplication processor for optical neural networks**  

---
Primary Category: cs.NE  
Categories: cs-ET, cs-NE, cs.NE, physics-optics  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.15061v1)  

---


**ABSTRACT**  
The computational demands of modern AI have spurred interest in optical neural networks (ONNs) which offer the potential benefits of increased speed and lower power consumption. However, current ONNs face various challenges,most significantly a limited calculation precision (typically around 4 bits) and the requirement for high-resolution signal format converters (digital-to-analogue conversions (DACs) and analogue-to-digital conversions (ADCs)). These challenges are inherent to their analog computing nature and pose significant obstacles in practical implementation. Here, we propose a digital-analog hybrid optical computing architecture for ONNs, which utilizes digital optical inputs in the form of binary words. By introducing the logic levels and decisions based on thresholding, the calculation precision can be significantly enhanced. The DACs for input data can be removed and the resolution of the ADCs can be greatly reduced. This can increase the operating speed at a high calculation precision and facilitate the compatibility with microelectronics. To validate our approach, we have fabricated a proof-of-concept photonic chip and built up a hybrid optical processor (HOP) system for neural network applications. We have demonstrated an unprecedented 16-bit calculation precision for high-definition image processing, with a pixel error rate (PER) as low as $1.8\times10^{-3}$ at an signal-to-noise ratio (SNR) of 18.2 dB. We have also implemented a convolutional neural network for handwritten digit recognition that shows the same accuracy as the one achieved by a desktop computer. The concept of the digital-analog hybrid optical computing architecture offers a methodology that could potentially be applied to various ONN implementations and may intrigue new research into efficient and accurate domain-specific optical computing architectures for neural networks.

{{</citation>}}


### (28/80) Emulating Complex Synapses Using Interlinked Proton Conductors (Lifu Zhang et al., 2024)

{{<citation>}}

Lifu Zhang, Ji-An Li, Yang Hu, Jie Jiang, Rongjie Lai, Marcus K. Benna, Jian Shi. (2024)  
**Emulating Complex Synapses Using Interlinked Proton Conductors**  

---
Primary Category: cs.NE  
Categories: cs-NE, cs.NE  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.15045v1)  

---


**ABSTRACT**  
In terms of energy efficiency and computational speed, neuromorphic electronics based on non-volatile memory devices is expected to be one of most promising hardware candidates for future artificial intelligence (AI). However, catastrophic forgetting, networks rapidly overwriting previously learned weights when learning new tasks, remains as a pivotal hurdle in either digital or analog AI chips for unleashing the true power of brain-like computing. To address catastrophic forgetting in the context of online memory storage, a complex synapse model (the Benna-Fusi model) has been proposed recently[1], whose synaptic weight and internal variables evolve following a diffusion dynamics. In this work, by designing a proton transistor with a series of charge-diffusion-controlled storage components, we have experimentally realized the Benna-Fusi artificial complex synapse. The memory consolidation from coupled storage components is revealed by both numerical simulations and experimental observations. Different memory timescales for the complex synapse are engineered by the diffusion length of charge carriers, the capacity and number of coupled storage components. The advantage of the demonstrated complex synapse in both memory capacity and memory consolidation is revealed by neural network simulations of face familiarity detection. Our experimental realization of the complex synapse suggests a promising approach to enhance memory capacity and to enable continual learning.

{{</citation>}}


### (29/80) LitE-SNN: Designing Lightweight and Efficient Spiking Neural Network through Spatial-Temporal Compressive Network Search and Joint Optimization (Qianhui Liu et al., 2024)

{{<citation>}}

Qianhui Liu, Jiaqi Yan, Malu Zhang, Gang Pan, Haizhou Li. (2024)  
**LitE-SNN: Designing Lightweight and Efficient Spiking Neural Network through Spatial-Temporal Compressive Network Search and Joint Optimization**  

---
Primary Category: cs.NE  
Categories: cs-NE, cs.NE  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2401.14652v1)  

---


**ABSTRACT**  
Spiking Neural Networks (SNNs) mimic the information-processing mechanisms of the human brain and are highly energy-efficient, making them well-suited for low-power edge devices. However, the pursuit of accuracy in current studies leads to large, long-timestep SNNs, conflicting with the resource constraints of these devices. In order to design lightweight and efficient SNNs, we propose a new approach named LitESNN that incorporates both spatial and temporal compression into the automated network design process. Spatially, we present a novel Compressive Convolution block (CompConv) to expand the search space to support pruning and mixed-precision quantization while utilizing the shared weights and pruning mask to reduce the computation. Temporally, we are the first to propose a compressive timestep search to identify the optimal number of timesteps under specific computation cost constraints. Finally, we formulate a joint optimization to simultaneously learn the architecture parameters and spatial-temporal compression strategies to achieve high performance while minimizing memory and computation costs. Experimental results on CIFAR10, CIFAR100, and Google Speech Command datasets demonstrate our proposed LitESNNs can achieve competitive or even higher accuracy with remarkably smaller model sizes and fewer computation costs. Furthermore, we validate the effectiveness of our LitESNN on the trade-off between accuracy and resource cost and show the superiority of our joint optimization. Additionally, we conduct energy analysis to further confirm the energy efficiency of LitESNN

{{</citation>}}


## cs.CL (16)



### (30/80) LongFin: A Multimodal Document Understanding Model for Long Financial Domain Documents (Ahmed Masry et al., 2024)

{{<citation>}}

Ahmed Masry, Amir Hajian. (2024)  
**LongFin: A Multimodal Document Understanding Model for Long Financial Domain Documents**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Financial  
[Paper Link](http://arxiv.org/abs/2401.15050v1)  

---


**ABSTRACT**  
Document AI is a growing research field that focuses on the comprehension and extraction of information from scanned and digital documents to make everyday business operations more efficient. Numerous downstream tasks and datasets have been introduced to facilitate the training of AI models capable of parsing and extracting information from various document types such as receipts and scanned forms. Despite these advancements, both existing datasets and models fail to address critical challenges that arise in industrial contexts. Existing datasets primarily comprise short documents consisting of a single page, while existing models are constrained by a limited maximum length, often set at 512 tokens. Consequently, the practical application of these methods in financial services, where documents can span multiple pages, is severely impeded. To overcome these challenges, we introduce LongFin, a multimodal document AI model capable of encoding up to 4K tokens. We also propose the LongForms dataset, a comprehensive financial dataset that encapsulates several industrial challenges in financial documents. Through an extensive evaluation, we demonstrate the effectiveness of the LongFin model on the LongForms dataset, surpassing the performance of existing public models while maintaining comparable results on existing single-page benchmarks.

{{</citation>}}


### (31/80) Health Text Simplification: An Annotated Corpus for Digestive Cancer Education and Novel Strategies for Reinforcement Learning (Md Mushfiqur Rahman et al., 2024)

{{<citation>}}

Md Mushfiqur Rahman, Mohammad Sabik Irbaz, Kai North, Michelle S. Williams, Marcos Zampieri, Kevin Lybarger. (2024)  
**Health Text Simplification: An Annotated Corpus for Digestive Cancer Education and Novel Strategies for Reinforcement Learning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: GPT, GPT-4, Language Model, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.15043v1)  

---


**ABSTRACT**  
Objective: The reading level of health educational materials significantly influences information understandability and accessibility, particularly for minoritized populations. Many patient educational resources surpass the reading level and complexity of widely accepted standards. There is a critical need for high-performing text simplification models in health information to enhance dissemination and literacy. This need is particularly acute in cancer education, where effective prevention and screening education can substantially reduce morbidity and mortality.   Methods: We introduce Simplified Digestive Cancer (SimpleDC), a parallel corpus of cancer education materials tailored for health text simplification research. Utilizing SimpleDC alongside the existing Med-EASi corpus, we explore Large Language Model (LLM)-based simplification methods, including fine-tuning, reinforcement learning (RL), reinforcement learning with human feedback (RLHF), domain adaptation, and prompt-based approaches. Our experimentation encompasses Llama 2 and GPT-4. A novel RLHF reward function is introduced, featuring a lightweight model adept at distinguishing between original and simplified texts, thereby enhancing the model's effectiveness with unlabeled data.   Results: Fine-tuned Llama 2 models demonstrated high performance across various metrics. Our innovative RLHF reward function surpassed existing RL text simplification reward functions in effectiveness. The results underscore that RL/RLHF can augment fine-tuning, facilitating model training on unlabeled text and improving performance. Additionally, these methods effectively adapt out-of-domain text simplification models to targeted domains.

{{</citation>}}


### (32/80) PROXYQA: An Alternative Framework for Evaluating Long-Form Text Generation with Large Language Models (Haochen Tan et al., 2024)

{{<citation>}}

Haochen Tan, Zhijiang Guo, Zhan Shi, Lu Xu, Zhili Liu, Xiaoguang Li, Yasheng Wang, Lifeng Shang, Qun Liu, Linqi Song. (2024)  
**PROXYQA: An Alternative Framework for Evaluating Long-Form Text Generation with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, QA, Text Generation  
[Paper Link](http://arxiv.org/abs/2401.15042v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have exhibited remarkable success in long-form context comprehension tasks. However, their capacity to generate long contents, such as reports and articles, remains insufficiently explored. Current benchmarks do not adequately assess LLMs' ability to produce informative and comprehensive content, necessitating a more rigorous evaluation approach. In this study, we introduce \textsc{ProxyQA}, a framework for evaluating long-form text generation, comprising in-depth human-curated \textit{meta-questions} spanning various domains. Each meta-question contains corresponding \textit{proxy-questions} with annotated answers. LLMs are prompted to generate extensive content in response to these meta-questions. Utilizing an evaluator and incorporating generated content as background context, \textsc{ProxyQA} evaluates the quality of generated content based on the evaluator's performance in answering the \textit{proxy-questions}. We examine multiple LLMs, emphasizing \textsc{ProxyQA}'s demanding nature as a high-quality assessment tool. Human evaluation demonstrates that evaluating through \textit{proxy-questions} is a highly self-consistent and human-criteria-correlated validation method. The dataset and leaderboard will be available at \url{https://github.com/Namco0816/ProxyQA}.

{{</citation>}}


### (33/80) ChemDFM: Dialogue Foundation Model for Chemistry (Zihan Zhao et al., 2024)

{{<citation>}}

Zihan Zhao, Da Ma, Lu Chen, Liangtai Sun, Zihao Li, Hongshen Xu, Zichen Zhu, Su Zhu, Shuai Fan, Guodong Shen, Xin Chen, Kai Yu. (2024)  
**ChemDFM: Dialogue Foundation Model for Chemistry**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-DL, cs.CL  
Keywords: Dialog, Dialogue, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2401.14818v1)  

---


**ABSTRACT**  
Large language models (LLMs) have established great success in the general domain of natural language processing. Their emerging task generalization and free-form dialogue capabilities can greatly help to design Chemical General Intelligence (CGI) to assist real-world research in chemistry. However, the existence of specialized language and knowledge in the field of chemistry, such as the highly informative SMILES notation, hinders the performance of general-domain LLMs in chemistry. To this end, we develop ChemDFM, the first LLM towards CGI. ChemDFM-13B is trained on 34B tokens from chemical literature, textbooks, and instructions as well as various data from the general domain. Therefore, it can store, understand, and reason over chemical knowledge and languages while still possessing advanced free-form language comprehension capabilities. Extensive quantitative evaluation shows that ChemDFM can significantly outperform the representative open-sourced LLMs. Moreover, ChemDFM can also surpass GPT-4 on a great portion of chemical tasks, despite the significant size difference. Further qualitative evaluations demonstrate the efficiency and effectiveness of ChemDFM in real-world research scenarios. We will open-source the ChemDFM model soon.

{{</citation>}}


### (34/80) Large Language Model Adaptation for Financial Sentiment Analysis (Pau Rodriguez Inserte et al., 2024)

{{<citation>}}

Pau Rodriguez Inserte, Mariam Nakhlé, Raheel Qader, Gaetan Caillaut, Jingshu Liu. (2024)  
**Large Language Model Adaptation for Financial Sentiment Analysis**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Financial, Language Model, NLP, Sentiment Analysis  
[Paper Link](http://arxiv.org/abs/2401.14777v1)  

---


**ABSTRACT**  
Natural language processing (NLP) has recently gained relevance within financial institutions by providing highly valuable insights into companies and markets' financial documents. However, the landscape of the financial domain presents extra challenges for NLP, due to the complexity of the texts and the use of specific terminology. Generalist language models tend to fall short in tasks specifically tailored for finance, even when using large language models (LLMs) with great natural language understanding and generative capabilities. This paper presents a study on LLM adaptation methods targeted at the financial domain and with high emphasis on financial sentiment analysis. To this purpose, two foundation models with less than 1.5B parameters have been adapted using a wide range of strategies. We show that through careful fine-tuning on both financial documents and instructions, these foundation models can be adapted to the target domain. Moreover, we observe that small LLMs have comparable performance to larger scale models, while being more efficient in terms of parameters and data. In addition to the models, we show how to generate artificial instructions through LLMs to augment the number of samples of the instruction dataset.

{{</citation>}}


### (35/80) Turn-taking and Backchannel Prediction with Acoustic and Large Language Model Fusion (Jinhan Wang et al., 2024)

{{<citation>}}

Jinhan Wang, Long Chen, Aparna Khare, Anirudh Raju, Pranav Dheram, Di He, Minhua Wu, Andreas Stolcke, Venkatesh Ravichandran. (2024)  
**Turn-taking and Backchannel Prediction with Acoustic and Large Language Model Fusion**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs-SD, cs.CL, eess-AS  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2401.14717v1)  

---


**ABSTRACT**  
We propose an approach for continuous prediction of turn-taking and backchanneling locations in spoken dialogue by fusing a neural acoustic model with a large language model (LLM). Experiments on the Switchboard human-human conversation dataset demonstrate that our approach consistently outperforms the baseline models with single modality. We also develop a novel multi-task instruction fine-tuning strategy to further benefit from LLM-encoded knowledge for understanding the tasks and conversational contexts, leading to additional improvements. Our approach demonstrates the potential of combined LLMs and acoustic models for a more natural and conversational interaction between humans and speech-enabled AI agents.

{{</citation>}}


### (36/80) Taiyi-Diffusion-XL: Advancing Bilingual Text-to-Image Generation with Large Vision-Language Model Support (Xiaojun Wu et al., 2024)

{{<citation>}}

Xiaojun Wu, Dixiang Zhang, Ruyi Gan, Junyu Lu, Ziwei Wu, Renliang Sun, Jiaxing Zhang, Pingjian Zhang, Yan Song. (2024)  
**Taiyi-Diffusion-XL: Advancing Bilingual Text-to-Image Generation with Large Vision-Language Model Support**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.14688v1)  

---


**ABSTRACT**  
Recent advancements in text-to-image models have significantly enhanced image generation capabilities, yet a notable gap of open-source models persists in bilingual or Chinese language support. To address this need, we present Taiyi-Diffusion-XL, a new Chinese and English bilingual text-to-image model which is developed by extending the capabilities of CLIP and Stable-Diffusion-XL through a process of bilingual continuous pre-training. This approach includes the efficient expansion of vocabulary by integrating the most frequently used Chinese characters into CLIP's tokenizer and embedding layers, coupled with an absolute position encoding expansion. Additionally, we enrich text prompts by large vision-language model, leading to better images captions and possess higher visual quality. These enhancements are subsequently applied to downstream text-to-image models. Our empirical results indicate that the developed CLIP model excels in bilingual image-text retrieval.Furthermore, the bilingual image generation capabilities of Taiyi-Diffusion-XL surpass previous models. This research leads to the development and open-sourcing of the Taiyi-Diffusion-XL model, representing a notable advancement in the field of image generation, particularly for Chinese language applications. This contribution is a step forward in addressing the need for more diverse language support in multimodal research. The model and demonstration are made publicly available at \href{https://huggingface.co/IDEA-CCNL/Taiyi-Stable-Diffusion-XL-3.5B/}{this https URL}, fostering further research and collaboration in this domain.

{{</citation>}}


### (37/80) MasonTigers@LT-EDI-2024: An Ensemble Approach towards Detecting Homophobia and Transphobia in Social Media Comments (Dhiman Goswami et al., 2024)

{{<citation>}}

Dhiman Goswami, Sadiya Sayara Chowdhury Puspo, Md Nishat Raihan, Al Nahian Bin Emran. (2024)  
**MasonTigers@LT-EDI-2024: An Ensemble Approach towards Detecting Homophobia and Transphobia in Social Media Comments**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Social Media  
[Paper Link](http://arxiv.org/abs/2401.14681v1)  

---


**ABSTRACT**  
In this paper, we describe our approaches and results for Task 2 of the LT-EDI 2024 Workshop, aimed at detecting homophobia and/or transphobia across ten languages. Our methodologies include monolingual transformers and ensemble methods, capitalizing on the strengths of each to enhance the performance of the models. The ensemble models worked well, placing our team, MasonTigers, in the top five for eight of the ten languages, as measured by the macro F1 score. Our work emphasizes the efficacy of ensemble methods in multilingual scenarios, addressing the complexities of language-specific tasks.

{{</citation>}}


### (38/80) MaLLaM -- Malaysia Large Language Model (Husein Zolkepli et al., 2024)

{{<citation>}}

Husein Zolkepli, Aisyah Razak, Kamarul Adha, Ariff Nazhan. (2024)  
**MaLLaM -- Malaysia Large Language Model**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2401.14680v1)  

---


**ABSTRACT**  
Addressing the gap in Large Language Model pretrained from scratch with Malaysian context, We trained models with 1.1 billion, 3 billion, and 5 billion parameters on a substantial 349GB dataset, equivalent to 90 billion tokens based on our pretrained Byte Pair Encoding (BPE) tokenizer for a single epoch. MaLLaM contributes to enhanced natural language understanding and generation tasks in the Malay language. Although trained on a smaller dataset of 90 billion tokens, our instruction-tuned MaLLaM models perform competitively. When compared to ChatGPT3.5 and Malaysian Mistral, MaLLaM's instruction-tuned models demonstrate notable proficiency, underscoring the effectiveness of our approach in capturing and understanding the nuances of the Malaysian language. MaLLaM models mark a significant contribution to the field, providing comprehensive language representations grounded in Malaysian context. This endeavor aims to pave the way for enhanced natural language understanding and generation tasks specific to the linguistic nuances present in Malaysia. We discuss the training methodology, dataset composition, and the potential impact of MaLLaM in advancing the capabilities of large language models within the context of the Malay language.   All models released at https://huggingface.co/collections/mesolitica/mallam-6577b59d1e0b436ae75f930f

{{</citation>}}


### (39/80) Scientific Large Language Models: A Survey on Biological & Chemical Domains (Qiang Zhang et al., 2024)

{{<citation>}}

Qiang Zhang, Keyang Ding, Tianwen Lyv, Xinda Wang, Qingyu Yin, Yiwen Zhang, Jing Yu, Yuhao Wang, Xiaotong Li, Zhuoyi Xiang, Xiang Zhuang, Zeyuan Wang, Ming Qin, Mengyao Zhang, Jinlu Zhang, Jiyu Cui, Renjun Xu, Hongyang Chen, Xiaohui Fan, Huabin Xing, Huajun Chen. (2024)  
**Scientific Large Language Models: A Survey on Biological & Chemical Domains**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2401.14656v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have emerged as a transformative power in enhancing natural language comprehension, representing a significant stride toward artificial general intelligence. The application of LLMs extends beyond conventional linguistic boundaries, encompassing specialized linguistic systems developed within various scientific disciplines. This growing interest has led to the advent of scientific LLMs, a novel subclass specifically engineered for facilitating scientific discovery. As a burgeoning area in the community of AI for Science, scientific LLMs warrant comprehensive exploration. However, a systematic and up-to-date survey introducing them is currently lacking. In this paper, we endeavor to methodically delineate the concept of "scientific language", whilst providing a thorough review of the latest advancements in scientific LLMs. Given the expansive realm of scientific disciplines, our analysis adopts a focused lens, concentrating on the biological and chemical domains. This includes an in-depth examination of LLMs for textual knowledge, small molecules, macromolecular proteins, genomic sequences, and their combinations, analyzing them in terms of model architectures, capabilities, datasets, and evaluation. Finally, we critically examine the prevailing challenges and point out promising research directions along with the advances of LLMs. By offering a comprehensive overview of technical developments in this field, this survey aspires to be an invaluable resource for researchers navigating the intricate landscape of scientific LLMs.

{{</citation>}}


### (40/80) A Korean Legal Judgment Prediction Dataset for Insurance Disputes (Alice Saebom Kwak et al., 2024)

{{<citation>}}

Alice Saebom Kwak, Cheonkam Jeong, Ji Weon Lim, Byeongcheol Min. (2024)  
**A Korean Legal Judgment Prediction Dataset for Insurance Disputes**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Legal, Transformer  
[Paper Link](http://arxiv.org/abs/2401.14654v1)  

---


**ABSTRACT**  
This paper introduces a Korean legal judgment prediction (LJP) dataset for insurance disputes. Successful LJP models on insurance disputes can benefit insurance companies and their customers. It can save both sides' time and money by allowing them to predict how the result would come out if they proceed to the dispute mediation process. As is often the case with low-resource languages, there is a limitation on the amount of data available for this specific task. To mitigate this issue, we investigate how one can achieve a good performance despite the limitation in data. In our experiment, we demonstrate that Sentence Transformer Fine-tuning (SetFit, Tunstall et al., 2022) is a good alternative to standard fine-tuning when training data are limited. The models fine-tuned with the SetFit approach on our data show similar performance to the Korean LJP benchmark models (Hwang et al., 2022) despite the much smaller data size.

{{</citation>}}


### (41/80) Benchmarking Large Language Models in Complex Question Answering Attribution using Knowledge Graphs (Nan Hu et al., 2024)

{{<citation>}}

Nan Hu, Jiaoyan Chen, Yike Wu, Guilin Qi, Sheng Bi, Tongtong Wu, Jeff Z. Pan. (2024)  
**Benchmarking Large Language Models in Complex Question Answering Attribution using Knowledge Graphs**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Knowledge Graph, Language Model, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2401.14640v1)  

---


**ABSTRACT**  
The attribution of question answering is to provide citations for supporting generated statements, and has attracted wide research attention. The current methods for automatically evaluating the attribution, which are often based on Large Language Models (LLMs), are still inadequate, particularly in recognizing subtle differences between attributions, and complex relationships between citations and statements. To compare these attribution evaluation methods and develop new ones, we introduce a set of fine-grained categories (i.e., supportive, insufficient, contradictory and irrelevant) for measuring the attribution, and develop a Complex Attributed Question Answering (CAQA) benchmark by leveraging knowledge graphs (KGs) for automatically generating attributions of different categories to question-answer pairs. Our analysis reveals that existing evaluators perform poorly under fine-grained attribution settings and exhibit weaknesses in complex citation-statement reasoning. Our CAQA benchmark, validated with human annotations, emerges as a promising tool for selecting and developing LLM attribution evaluators.

{{</citation>}}


### (42/80) T-Rex: Text-assisted Retrosynthesis Prediction (Yifeng Liu et al., 2024)

{{<citation>}}

Yifeng Liu, Hanwen Xu, Tangqi Fang, Haocheng Xi, Zixuan Liu, Sheng Zhang, Hoifung Poon, Sheng Wang. (2024)  
**T-Rex: Text-assisted Retrosynthesis Prediction**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2401.14637v1)  

---


**ABSTRACT**  
As a fundamental task in computational chemistry, retrosynthesis prediction aims to identify a set of reactants to synthesize a target molecule. Existing template-free approaches only consider the graph structures of the target molecule, which often cannot generalize well to rare reaction types and large molecules. Here, we propose T-Rex, a text-assisted retrosynthesis prediction approach that exploits pre-trained text language models, such as ChatGPT, to assist the generation of reactants. T-Rex first exploits ChatGPT to generate a description for the target molecule and rank candidate reaction centers based both the description and the molecular graph. It then re-ranks these candidates by querying the descriptions for each reactants and examines which group of reactants can best synthesize the target molecule. We observed that T-Rex substantially outperformed graph-based state-of-the-art approaches on two datasets, indicating the effectiveness of considering text information. We further found that T-Rex outperformed the variant that only use ChatGPT-based description without the re-ranking step, demonstrate how our framework outperformed a straightforward integration of ChatGPT and graph information. Collectively, we show that text generated by pre-trained language models can substantially improve retrosynthesis prediction, opening up new avenues for exploiting ChatGPT to advance computational chemistry. And the codes can be found at https://github.com/lauyikfung/T-Rex.

{{</citation>}}


### (43/80) An Empirical Investigation of Domain Adaptation Ability for Chinese Spelling Check Models (Xi Wang et al., 2024)

{{<citation>}}

Xi Wang, Ruoqing Zhao, Hongliang Dai, Piji Li. (2024)  
**An Empirical Investigation of Domain Adaptation Ability for Chinese Spelling Check Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2401.14630v1)  

---


**ABSTRACT**  
Chinese Spelling Check (CSC) is a meaningful task in the area of Natural Language Processing (NLP) which aims at detecting spelling errors in Chinese texts and then correcting these errors. However, CSC models are based on pretrained language models, which are trained on a general corpus. Consequently, their performance may drop when confronted with downstream tasks involving domain-specific terms. In this paper, we conduct a thorough evaluation about the domain adaption ability of various typical CSC models by building three new datasets encompassing rich domain-specific terms from the financial, medical, and legal domains. Then we conduct empirical investigations in the corresponding domain-specific test datasets to ascertain the cross-domain adaptation ability of several typical CSC models. We also test the performance of the popular large language model ChatGPT. As shown in our experiments, the performances of the CSC models drop significantly in the new domains.

{{</citation>}}


### (44/80) Toward Practical Automatic Speech Recognition and Post-Processing: a Call for Explainable Error Benchmark Guideline (Seonmin Koo et al., 2024)

{{<citation>}}

Seonmin Koo, Chanjun Park, Jinsung Kim, Jaehyung Seo, Sugyeong Eo, Hyeonseok Moon, Heuiseok Lim. (2024)  
**Toward Practical Automatic Speech Recognition and Post-Processing: a Call for Explainable Error Benchmark Guideline**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2401.14625v1)  

---


**ABSTRACT**  
Automatic speech recognition (ASR) outcomes serve as input for downstream tasks, substantially impacting the satisfaction level of end-users. Hence, the diagnosis and enhancement of the vulnerabilities present in the ASR model bear significant importance. However, traditional evaluation methodologies of ASR systems generate a singular, composite quantitative metric, which fails to provide comprehensive insight into specific vulnerabilities. This lack of detail extends to the post-processing stage, resulting in further obfuscation of potential weaknesses. Despite an ASR model's ability to recognize utterances accurately, subpar readability can negatively affect user satisfaction, giving rise to a trade-off between recognition accuracy and user-friendliness. To effectively address this, it is imperative to consider both the speech-level, crucial for recognition accuracy, and the text-level, critical for user-friendliness. Consequently, we propose the development of an Error Explainable Benchmark (EEB) dataset. This dataset, while considering both speech- and text-level, enables a granular understanding of the model's shortcomings. Our proposition provides a structured pathway for a more `real-world-centric' evaluation, a marked shift away from abstracted, traditional methods, allowing for the detection and rectification of nuanced system weaknesses, ultimately aiming for an improved user experience.

{{</citation>}}


### (45/80) Enhancing Diagnostic Accuracy through Multi-Agent Conversations: Using Large Language Models to Mitigate Cognitive Bias (Yu He Ke et al., 2024)

{{<citation>}}

Yu He Ke, Rui Yang, Sui An Lie, Taylor Xin Yi Lim, Hairil Rizal Abdullah, Daniel Shu Wei Ting, Nan Liu. (2024)  
**Enhancing Diagnostic Accuracy through Multi-Agent Conversations: Using Large Language Models to Mitigate Cognitive Bias**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Bias, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2401.14589v1)  

---


**ABSTRACT**  
Background: Cognitive biases in clinical decision-making significantly contribute to errors in diagnosis and suboptimal patient outcomes. Addressing these biases presents a formidable challenge in the medical field. This study explores the role of large language models (LLMs) in mitigating these biases through the utilization of a multi-agent framework. We simulate the clinical decision-making processes through multi-agent conversation and evaluate its efficacy in improving diagnostic accuracy. Methods: A total of 16 published and unpublished case reports where cognitive biases have resulted in misdiagnoses were identified from the literature. In the multi-agent system, we leveraged GPT-4 Turbo to facilitate interactions among four simulated agents to replicate clinical team dynamics. Each agent has a distinct role: 1) To make the initial and final diagnosis after considering the discussions, 2) The devil's advocate and correct confirmation and anchoring bias, 3) The tutor and facilitator of the discussion to reduce premature closure bias, and 4) To record and summarize the findings. A total of 80 simulations were evaluated for the accuracy of initial diagnosis, top differential diagnosis and final two differential diagnoses. Findings: In a total of 80 responses evaluating both initial and final diagnoses, the initial diagnosis had an accuracy of 0% (0/80), but following multi-agent discussions, the accuracy for the top differential diagnosis increased to 71.3% (57/80), and for the final two differential diagnoses, to 80.0% (64/80). The system demonstrated an ability to reevaluate and correct misconceptions, even in scenarios with misleading initial investigations. Interpretation: The LLM-driven multi-agent conversation system shows promise in enhancing diagnostic accuracy in diagnostically challenging medical scenarios.

{{</citation>}}


## cs.CR (2)



### (46/80) Computationally Bounded Robust Compilation and Universally Composable Security (Robert Künnemann et al., 2024)

{{<citation>}}

Robert Künnemann, Marco Patrignani, Ethan Cecchetti. (2024)  
**Computationally Bounded Robust Compilation and Universally Composable Security**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-PL, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2401.15041v1)  

---


**ABSTRACT**  
Universal Composability (UC) is the gold standard for cryptographic security, but mechanizing proofs of UC is notoriously difficult. A recently-discovered connection between UC and Robust Compilation (RC)$\unicode{x2014}$a novel theory of secure compilation$\unicode{x2014}$provides a means to verify UC proofs using tools that mechanize equality results. Unfortunately, the existing methods apply only to perfect UC security, and real-world protocols relying on cryptography are only computationally secure.   This paper addresses this gap by lifting the connection between UC and RC to the computational setting, extending techniques from the RC setting to apply to computational UC security. Moreover, it further generalizes the UC$\unicode{x2013}$RC connection beyond computational security to arbitrary equalities, providing a framework to subsume the existing perfect case, and to instantiate future theories with more complex notions of security. This connection allows the use of tools for proofs of computational indistinguishability to properly mechanize proofs of computational UC security. We demonstrate this power by using CryptoVerif to mechanize a proof that parts of the Wireguard protocol are computationally UC secure. Finally, all proofs of the framework itself are verified in Isabelle/HOL.

{{</citation>}}


### (47/80) Coca: Improving and Explaining Graph Neural Network-Based Vulnerability Detection Systems (Sicong Cao et al., 2024)

{{<citation>}}

Sicong Cao, Xiaobing Sun, Xiaoxue Wu, David Lo, Lili Bo, Bin Li, Wei Liu. (2024)  
**Coca: Improving and Explaining Graph Neural Network-Based Vulnerability Detection Systems**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-SE, cs.CR  
Keywords: GNN, Graph Neural Network, Vulnerability Detection  
[Paper Link](http://arxiv.org/abs/2401.14886v1)  

---


**ABSTRACT**  
Recently, Graph Neural Network (GNN)-based vulnerability detection systems have achieved remarkable success. However, the lack of explainability poses a critical challenge to deploy black-box models in security-related domains. For this reason, several approaches have been proposed to explain the decision logic of the detection model by providing a set of crucial statements positively contributing to its predictions. Unfortunately, due to the weakly-robust detection models and suboptimal explanation strategy, they have the danger of revealing spurious correlations and redundancy issue.   In this paper, we propose Coca, a general framework aiming to 1) enhance the robustness of existing GNN-based vulnerability detection models to avoid spurious explanations; and 2) provide both concise and effective explanations to reason about the detected vulnerabilities. \sysname consists of two core parts referred to as Trainer and Explainer. The former aims to train a detection model which is robust to random perturbation based on combinatorial contrastive learning, while the latter builds an explainer to derive crucial code statements that are most decisive to the detected vulnerability via dual-view causal inference as explanations. We apply Coca over three typical GNN-based vulnerability detectors. Experimental results show that Coca can effectively mitigate the spurious correlation issue, and provide more useful high-quality explanations.

{{</citation>}}


## cs.RO (3)



### (48/80) Multi-Agent Coordination for a Partially Observable and Dynamic Robot Soccer Environment with Limited Communication (Daniele Affinita et al., 2024)

{{<citation>}}

Daniele Affinita, Flavio Volpi, Valerio Spagnoli, Vincenzo Suriani, Daniele Nardi, Domenico D. Bloisi. (2024)  
**Multi-Agent Coordination for a Partially Observable and Dynamic Robot Soccer Environment with Limited Communication**  

---
Primary Category: cs.RO  
Categories: I-2-9; I-2-11, cs-MA, cs-RO, cs.RO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.15026v1)  

---


**ABSTRACT**  
RoboCup represents an International testbed for advancing research in AI and robotics, focusing on a definite goal: developing a robot team that can win against the human world soccer champion team by the year 2050. To achieve this goal, autonomous humanoid robots' coordination is crucial. This paper explores novel solutions within the RoboCup Standard Platform League (SPL), where a reduction in WiFi communication is imperative, leading to the development of new coordination paradigms. The SPL has experienced a substantial decrease in network packet rate, compelling the need for advanced coordination architectures to maintain optimal team functionality in dynamic environments. Inspired by market-based task assignment, we introduce a novel distributed coordination system to orchestrate autonomous robots' actions efficiently in low communication scenarios. This approach has been tested with NAO robots during official RoboCup competitions and in the SimRobot simulator, demonstrating a notable reduction in task overlaps in limited communication settings.

{{</citation>}}


### (49/80) RESPRECT: Speeding-up Multi-fingered Grasping with Residual Reinforcement Learning (Federico Ceola et al., 2024)

{{<citation>}}

Federico Ceola, Lorenzo Rosasco, Lorenzo Natale. (2024)  
**RESPRECT: Speeding-up Multi-fingered Grasping with Residual Reinforcement Learning**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.14858v1)  

---


**ABSTRACT**  
Deep Reinforcement Learning (DRL) has proven effective in learning control policies using robotic grippers, but much less practical for solving the problem of grasping with dexterous hands -- especially on real robotic platforms -- due to the high dimensionality of the problem. In this work, we focus on the multi-fingered grasping task with the anthropomorphic hand of the iCub humanoid. We propose the RESidual learning with PREtrained CriTics (RESPRECT) method that, starting from a policy pre-trained on a large set of objects, can learn a residual policy to grasp a novel object in a fraction ($\sim 5 \times$ faster) of the timesteps required to train a policy from scratch, without requiring any task demonstration. To our knowledge, this is the first Residual Reinforcement Learning (RRL) approach that learns a residual policy on top of another policy pre-trained with DRL. We exploit some components of the pre-trained policy during residual learning that further speed-up the training. We benchmark our results in the iCub simulated environment, and we show that RESPRECT can be effectively used to learn a multi-fingered grasping policy on the real iCub robot. The code to reproduce the experiments is released together with the paper with an open source license.

{{</citation>}}


### (50/80) Generative Expressive Robot Behaviors using Large Language Models (Karthik Mahadevan et al., 2024)

{{<citation>}}

Karthik Mahadevan, Jonathan Chien, Noah Brown, Zhuo Xu, Carolina Parada, Fei Xia, Andy Zeng, Leila Takayama, Dorsa Sadigh. (2024)  
**Generative Expressive Robot Behaviors using Large Language Models**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.14673v1)  

---


**ABSTRACT**  
People employ expressive behaviors to effectively communicate and coordinate their actions with others, such as nodding to acknowledge a person glancing at them or saying "excuse me" to pass people in a busy corridor. We would like robots to also demonstrate expressive behaviors in human-robot interaction. Prior work proposes rule-based methods that struggle to scale to new communication modalities or social situations, while data-driven methods require specialized datasets for each social situation the robot is used in. We propose to leverage the rich social context available from large language models (LLMs) and their ability to generate motion based on instructions or user preferences, to generate expressive robot motion that is adaptable and composable, building upon each other. Our approach utilizes few-shot chain-of-thought prompting to translate human language instructions into parametrized control code using the robot's available and learned skills. Through user studies and simulation experiments, we demonstrate that our approach produces behaviors that users found to be competent and easy to understand. Supplementary material can be found at https://generative-expressive-motion.github.io/.

{{</citation>}}


## eess.AS (1)



### (51/80) Enhancement of a Text-Independent Speaker Verification System by using Feature Combination and Parallel-Structure Classifiers (Kerlos Atia Abdalmalak et al., 2024)

{{<citation>}}

Kerlos Atia Abdalmalak, Ascensión Gallardo-Antol'in. (2024)  
**Enhancement of a Text-Independent Speaker Verification System by using Feature Combination and Parallel-Structure Classifiers**  

---
Primary Category: eess.AS  
Categories: cs-LG, cs-SD, eess-AS, eess.AS  
Keywords: Speaker Verification  
[Paper Link](http://arxiv.org/abs/2401.15018v1)  

---


**ABSTRACT**  
Speaker Verification (SV) systems involve mainly two individual stages: feature extraction and classification. In this paper, we explore these two modules with the aim of improving the performance of a speaker verification system under noisy conditions. On the one hand, the choice of the most appropriate acoustic features is a crucial factor for performing robust speaker verification. The acoustic parameters used in the proposed system are: Mel Frequency Cepstral Coefficients (MFCC), their first and second derivatives (Deltas and Delta- Deltas), Bark Frequency Cepstral Coefficients (BFCC), Perceptual Linear Predictive (PLP), and Relative Spectral Transform - Perceptual Linear Predictive (RASTA-PLP). In this paper, a complete comparison of different combinations of the previous features is discussed. On the other hand, the major weakness of a conventional Support Vector Machine (SVM) classifier is the use of generic traditional kernel functions to compute the distances among data points. However, the kernel function of an SVM has great influence on its performance. In this work, we propose the combination of two SVM-based classifiers with different kernel functions: Linear kernel and Gaussian Radial Basis Function (RBF) kernel with a Logistic Regression (LR) classifier. The combination is carried out by means of a parallel structure approach, in which different voting rules to take the final decision are considered. Results show that significant improvement in the performance of the SV system is achieved by using the combined features with the combined classifiers either with clean speech or in the presence of noise. Finally, to enhance the system more in noisy environments, the inclusion of the multiband noise removal technique as a preprocessing stage is proposed.

{{</citation>}}


## cs.NI (2)



### (52/80) Reinforcement Learning-based Relay Selection for Cooperative WSNs in the Presence of Bursty Impulsive Noise (Hazem Barka et al., 2024)

{{<citation>}}

Hazem Barka, Md Sahabul Alam, Georges Kaddoum, Minh Au, Basile L. Agba. (2024)  
**Reinforcement Learning-based Relay Selection for Cooperative WSNs in the Presence of Bursty Impulsive Noise**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.15008v1)  

---


**ABSTRACT**  
The problem of relay selection is pivotal in the realm of cooperative communication. However, this issue has not been thoroughly examined, particularly when the background noise is assumed to possess an impulsive characteristic with consistent memory as observed in smart grid communications and some other wireless communication scenarios. In this paper, we investigate the impact of this specific type of noise on the performance of cooperative Wireless Sensor Networks (WSNs) with the Decode and Forward (DF) relaying scheme, considering Symbol-Error-Rate (SER) and battery power consumption fairness across all nodes as the performance metrics. We introduce two innovative relay selection methods that depend on noise state detection and the residual battery power of each relay. The first method encompasses the adaptation of the Max-Min criterion to this specific context, whereas the second employs Reinforcement Learning (RL) to surmount this challenge. Our empirical outcomes demonstrate that the impacts of bursty impulsive noise on the SER performance can be effectively mitigated and that a balance in battery power consumption among all nodes can be established using the proposed methods.

{{</citation>}}


### (53/80) A Deep Reinforcement Learning-based Approach for Adaptive Handover Protocols in Mobile Networks (Peter J. Gu et al., 2024)

{{<citation>}}

Peter J. Gu, Johannes Voigt, Peter M. Rost. (2024)  
**A Deep Reinforcement Learning-based Approach for Adaptive Handover Protocols in Mobile Networks**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.14823v1)  

---


**ABSTRACT**  
Due to an ever-increasing number of participants and new areas of application, the demands on mobile communications systems are continually increasing. In order to deliver higher data rates, enable mobility and guarantee QoS requirements of subscribers, these systems and the protocols used are becoming more complex. By using higher frequency spectrums, cells become smaller and more base stations have to be deployed. This leads to an increased number of handovers of user equipments between base stations in order to enable mobility, resulting in potentially more frequent radio link failures and rate reduction. The persistent switching between the same base stations, commonly referred to as "ping-pong", leads to a consistent reduction of data rates. In this work, we propose a method for handover optimization by using proximal policy optimization in mobile communications to learn an adaptive handover protocol. The resulting agent is highly flexible regarding different travelling speeds of user equipments, while outperforming the standard 5G NR handover protocol by 3GPP in terms of average data rate and number of radio link failures. Furthermore, the design of the proposed environment demonstrates remarkable accuracy, ensuring a fair comparison with the standard 3GPP protocol.

{{</citation>}}


## physics.soc-ph (1)



### (54/80) The causal role of the Reddit collective action on the GameStop short squeeze (Antonio Desiderio et al., 2024)

{{<citation>}}

Antonio Desiderio, Luca Maria Aiello, Giulio Cimini, Laura Alessandretti. (2024)  
**The causal role of the Reddit collective action on the GameStop short squeeze**  

---
Primary Category: physics.soc-ph  
Categories: cs-CY, cs-SI, physics-soc-ph, physics.soc-ph  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2401.14999v1)  

---


**ABSTRACT**  
In early 2021, the stock prices of GameStop, AMC, Nokia, and BlackBerry experienced dramatic increases, triggered by short squeeze operations that have been largely attributed to Reddit's retail investors. These events showcased, for the first time, the potential of online social networks to catalyze financial collective action. How, when and to what extent Reddit users played a causal role in driving up these prices, however, remains unclear. To address these questions, we employ causal inference techniques, leveraging data capturing activity on Reddit and Twitter, and trading volume with a high temporal resolution. We find that Reddit discussions foreshadowed trading volume before the GameStop short squeeze, with their predictive power being particularly strong on hourly time scales. This effect emerged abruptly and became prominent a few weeks before the event, but waned once the community of investors gained widespread visibility through Twitter. As the causal link unfolded, the collective investment of the Reddit community, quantified through each user's financial position on GameStop, closely mirrored the market capitalization of the stock. The evidence from our study suggests that Reddit users fueled the GameStop short squeeze, and thereby Reddit served as a coordination hub for a shared financial strategy. Towards the end of January, users talking about GameStop contributed to raise the popularity of BlackBerry, AMC and Nokia, which emerged as the most popular stocks as the community gained global recognition. Overall, our results shed light on the dynamics behind the first large-scale financial collective action driven by social media users.

{{</citation>}}


## cs.SE (6)



### (55/80) Embedding-based search in JetBrains IDEs (Evgeny Abramov et al., 2024)

{{<citation>}}

Evgeny Abramov, Nikolai Palchikov. (2024)  
**Embedding-based search in JetBrains IDEs**  

---
Primary Category: cs.SE  
Categories: cs-CL, cs-IR, cs-LG, cs-SE, cs.SE  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2401.14975v1)  

---


**ABSTRACT**  
Most modern Integrated Development Environments (IDEs) and code editors have a feature to search across available functionality and items in an open project. In JetBrains IDEs, this feature is called Search Everywhere: it allows users to search for files, actions, classes, symbols, settings, and anything from VCS history from a single entry point. However, it works with the candidates obtained by algorithms that don't account for semantics, e.g., synonyms, complex word permutations, part of the speech modifications, and typos. In this work, we describe the machine learning approach we implemented to improve the discoverability of search items. We also share the obstacles encountered during this process and how we overcame them.

{{</citation>}}


### (56/80) Reassessing Java Code Readability Models with a Human-Centered Approach (Agnia Sergeyuk et al., 2024)

{{<citation>}}

Agnia Sergeyuk, Olga Lvova, Sergey Titov, Anastasiia Serova, Farid Bagirov, Evgeniia Kirillova, Timofey Bryksin. (2024)  
**Reassessing Java Code Readability Models with a Human-Centered Approach**  

---
Primary Category: cs.SE  
Categories: cs-HC, cs-SE, cs.SE  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2401.14936v1)  

---


**ABSTRACT**  
To ensure that Large Language Models (LLMs) effectively support user productivity, they need to be adjusted. Existing Code Readability (CR) models can guide this alignment. However, there are concerns about their relevance in modern software engineering since they often miss the developers' notion of readability and rely on outdated code. This research assesses existing Java CR models for LLM adjustments, measuring the correlation between their and developers' evaluations of AI-generated Java code. Using the Repertory Grid Technique with 15 developers, we identified 12 key code aspects influencing CR that were consequently assessed by 390 programmers when labeling 120 AI-generated snippets. Our findings indicate that when AI generates concise and executable code, it is often considered readable by CR models and developers. However, a limited correlation between these evaluations underscores the importance of future research on learning objectives for adjusting LLMs and on the aspects influencing CR evaluations included in predictive models.

{{</citation>}}


### (57/80) On Repairing Quantum Programs Using ChatGPT (Xiaoyu Guo et al., 2024)

{{<citation>}}

Xiaoyu Guo, Jianjun Zhao, Pengzhan Zhao. (2024)  
**On Repairing Quantum Programs Using ChatGPT**  

---
Primary Category: cs.SE  
Categories: cs-PL, cs-SE, cs.SE  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2401.14913v1)  

---


**ABSTRACT**  
Automated Program Repair (APR) is a vital area in software engineering aimed at generating automatic patches for vulnerable programs. While numerous techniques have been proposed for repairing classical programs, the realm of quantum programming lacks a comparable automated repair technique. In this initial exploration, we investigate the use of ChatGPT for quantum program repair and evaluate its performance on Bugs4Q, a benchmark suite of quantum program bugs. Our findings demonstrate the feasibility of employing ChatGPT for quantum program repair. Specifically, we assess ChatGPT's ability to address bugs within the Bugs4Q benchmark, revealing its success in repairing 29 out of 38 bugs. This research represents a promising step towards automating the repair process for quantum programs.

{{</citation>}}


### (58/80) SparseCoder: Identifier-Aware Sparse Transformer for File-Level Code Summarization (Yanlin Wang et al., 2024)

{{<citation>}}

Yanlin Wang, Yanxian Huang, Daya Guo, Hongyu Zhang, Zibin Zheng. (2024)  
**SparseCoder: Identifier-Aware Sparse Transformer for File-Level Code Summarization**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Summarization, Transformer  
[Paper Link](http://arxiv.org/abs/2401.14727v1)  

---


**ABSTRACT**  
Code summarization aims to generate natural language descriptions of source code, facilitating programmers to understand and maintain it rapidly. While previous code summarization efforts have predominantly focused on method-level, this paper studies file-level code summarization, which can assist programmers in understanding and maintaining large source code projects. Unlike method-level code summarization,file-level code summarization typically involves long source code within a single file, which makes it challenging for Transformer-based models to understand the code semantics for the maximum input length of these models is difficult to set to a large number that can handle long code input well, due to the quadratic scaling of computational complexity with the input sequence length. To address this challenge, we propose SparseCoder, an identifier-aware sparse transformer for effectively handling long code sequences. Specifically, the SparseCoder employs a sliding window mechanism for self-attention to model short-term dependencies and leverages the structure message of code to capture long-term dependencies among source code identifiers by introducing two types of sparse attention patterns named global and identifier attention. To evaluate the performance of SparseCoder, we construct a new dataset FILE-CS for file-level code summarization in Python. Experimental results show that our SparseCoder model achieves state-of-the-art performance compared with other pre-trained models, including full self-attention and sparse models. Additionally, our model has low memory overhead and achieves comparable performance with models using full self-attention mechanism.

{{</citation>}}


### (59/80) Inferring Data Preconditions from Deep Learning Models for Trustworthy Prediction in Deployment (Shibbir Ahmed et al., 2024)

{{<citation>}}

Shibbir Ahmed, Hongyang Gao, Hridesh Rajan. (2024)  
**Inferring Data Preconditions from Deep Learning Models for Trustworthy Prediction in Deployment**  

---
Primary Category: cs.SE  
Categories: cs-LG, cs-SE, cs.SE  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.14628v1)  

---


**ABSTRACT**  
Deep learning models are trained with certain assumptions about the data during the development stage and then used for prediction in the deployment stage. It is important to reason about the trustworthiness of the model's predictions with unseen data during deployment. Existing methods for specifying and verifying traditional software are insufficient for this task, as they cannot handle the complexity of DNN model architecture and expected outcomes. In this work, we propose a novel technique that uses rules derived from neural network computations to infer data preconditions for a DNN model to determine the trustworthiness of its predictions. Our approach, DeepInfer involves introducing a novel abstraction for a trained DNN model that enables weakest precondition reasoning using Dijkstra's Predicate Transformer Semantics. By deriving rules over the inductive type of neural network abstract representation, we can overcome the matrix dimensionality issues that arise from the backward non-linear computation from the output layer to the input layer. We utilize the weakest precondition computation using rules of each kind of activation function to compute layer-wise precondition from the given postcondition on the final output of a deep neural network. We extensively evaluated DeepInfer on 29 real-world DNN models using four different datasets collected from five different sources and demonstrated the utility, effectiveness, and performance improvement over closely related work. DeepInfer efficiently detects correct and incorrect predictions of high-accuracy models with high recall (0.98) and high F-1 score (0.84) and has significantly improved over prior technique, SelfChecker. The average runtime overhead of DeepInfer is low, 0.22 sec for all unseen datasets. We also compared runtime overhead using the same hardware settings and found that DeepInfer is 3.27 times faster than SelfChecker.

{{</citation>}}


### (60/80) A Systematic Literature Review on Explainability for Machine/Deep Learning-based Software Engineering Research (Sicong Cao et al., 2024)

{{<citation>}}

Sicong Cao, Xiaobing Sun, Ratnadira Widyasari, David Lo, Xiaoxue Wu, Lili Bo, Jiale Zhang, Bin Li, Wei Liu, Di Wu, Yixin Chen. (2024)  
**A Systematic Literature Review on Explainability for Machine/Deep Learning-based Software Engineering Research**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.14617v1)  

---


**ABSTRACT**  
The remarkable achievements of Artificial Intelligence (AI) algorithms, particularly in Machine Learning (ML) and Deep Learning (DL), have fueled their extensive deployment across multiple sectors, including Software Engineering (SE). However, due to their black-box nature, these promising AI-driven SE models are still far from being deployed in practice. This lack of explainability poses unwanted risks for their applications in critical tasks, such as vulnerability detection, where decision-making transparency is of paramount importance. This paper endeavors to elucidate this interdisciplinary domain by presenting a systematic literature review of approaches that aim to improve the explainability of AI models within the context of SE. The review canvasses work appearing in the most prominent SE & AI conferences and journals, and spans 63 papers across 21 unique SE tasks. Based on three key Research Questions (RQs), we aim to (1) summarize the SE tasks where XAI techniques have shown success to date; (2) classify and analyze different XAI techniques; and (3) investigate existing evaluation approaches. Based on our findings, we identified a set of challenges remaining to be addressed in existing studies, together with a roadmap highlighting potential opportunities we deemed appropriate and important for future work.

{{</citation>}}


## econ.TH (1)



### (61/80) Could AI change the scientific publishing market once and for all? (Wadim Strielkowski, 2024)

{{<citation>}}

Wadim Strielkowski. (2024)  
**Could AI change the scientific publishing market once and for all?**  

---
Primary Category: econ.TH  
Categories: cs-DL, econ-TH, econ.TH  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2401.14952v1)  

---


**ABSTRACT**  
Artificial-intelligence tools in research like ChatGPT are playing an increasingly transformative role in revolutionizing scientific publishing and re-shaping its economic background. They can help academics to tackle such issues as limited space in academic journals, accessibility of knowledge, delayed dissemination, or the exponential growth of academic output. Moreover, AI tools could potentially change scientific communication and academic publishing market as we know them. They can help to promote Open Access (OA) in the form of preprints, dethrone the entrenched journals and publishers, as well as introduce novel approaches to the assessment of research output. It is also imperative that they should do just that, once and for all.

{{</citation>}}


## cs.IR (4)



### (62/80) Macro Graph Neural Networks for Online Billion-Scale Recommender Systems (Hao Chen et al., 2024)

{{<citation>}}

Hao Chen, Yuanchen Bei, Qijie Shen, Yue Xu, Sheng Zhou, Wenbing Huang, Feiran Huang, Senzhang Wang, Xiao Huang. (2024)  
**Macro Graph Neural Networks for Online Billion-Scale Recommender Systems**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2401.14939v1)  

---


**ABSTRACT**  
Predicting Click-Through Rate (CTR) in billion-scale recommender systems poses a long-standing challenge for Graph Neural Networks (GNNs) due to the overwhelming computational complexity involved in aggregating billions of neighbors. To tackle this, GNN-based CTR models usually sample hundreds of neighbors out of the billions to facilitate efficient online recommendations. However, sampling only a small portion of neighbors results in a severe sampling bias and the failure to encompass the full spectrum of user or item behavioral patterns. To address this challenge, we name the conventional user-item recommendation graph as "micro recommendation graph" and introduce a more suitable MAcro Recommendation Graph (MAG) for billion-scale recommendations. MAG resolves the computational complexity problems in the infrastructure by reducing the node count from billions to hundreds. Specifically, MAG groups micro nodes (users and items) with similar behavior patterns to form macro nodes. Subsequently, we introduce tailored Macro Graph Neural Networks (MacGNN) to aggregate information on a macro level and revise the embeddings of macro nodes. MacGNN has already served Taobao's homepage feed for two months, providing recommendations for over one billion users. Extensive offline experiments on three public benchmark datasets and an industrial dataset present that MacGNN significantly outperforms twelve CTR baselines while remaining computationally efficient. Besides, online A/B tests confirm MacGNN's superiority in billion-scale recommender systems.

{{</citation>}}


### (63/80) The Power of Noise: Redefining Retrieval for RAG Systems (Florin Cuconasu et al., 2024)

{{<citation>}}

Florin Cuconasu, Giovanni Trappolini, Federico Siciliano, Simone Filice, Cesare Campagnano, Yoelle Maarek, Nicola Tonellotto, Fabrizio Silvestri. (2024)  
**The Power of Noise: Redefining Retrieval for RAG Systems**  

---
Primary Category: cs.IR  
Categories: cs-CL, cs-IR, cs.IR  
Keywords: Information Retrieval, Language Model  
[Paper Link](http://arxiv.org/abs/2401.14887v1)  

---


**ABSTRACT**  
Retrieval-Augmented Generation (RAG) systems represent a significant advancement over traditional Large Language Models (LLMs). RAG systems enhance their generation ability by incorporating external data retrieved through an Information Retrieval (IR) phase, overcoming the limitations of standard LLMs, which are restricted to their pre-trained knowledge and limited context window. Most research in this area has predominantly concentrated on the generative aspect of LLMs within RAG systems. Our study fills this gap by thoroughly and critically analyzing the influence of IR components on RAG systems. This paper analyzes which characteristics a retriever should possess for an effective RAG's prompt formulation, focusing on the type of documents that should be retrieved. We evaluate various elements, such as the relevance of the documents to the prompt, their position, and the number included in the context. Our findings reveal, among other insights, that including irrelevant documents can unexpectedly enhance performance by more than 30% in accuracy, contradicting our initial assumption of diminished quality. These findings call for developing specialized approaches tailored to the specific demands of integrating retrieval with language generation models and pave the way for future research. These results underscore the need for developing specialized strategies to integrate retrieval with language generation models, thereby laying the groundwork for future research in this field.

{{</citation>}}


### (64/80) Prompt-enhanced Federated Content Representation Learning for Cross-domain Recommendation (Lei Guo et al., 2024)

{{<citation>}}

Lei Guo, Ziang Lu, Junliang Yu, Nguyen Quoc Viet Hung, Hongzhi Yin. (2024)  
**Prompt-enhanced Federated Content Representation Learning for Cross-domain Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2401.14678v1)  

---


**ABSTRACT**  
Cross-domain Recommendation (CDR) as one of the effective techniques in alleviating the data sparsity issues has been widely studied in recent years. However, previous works may cause domain privacy leakage since they necessitate the aggregation of diverse domain data into a centralized server during the training process. Though several studies have conducted privacy preserving CDR via Federated Learning (FL), they still have the following limitations: 1) They need to upload users' personal information to the central server, posing the risk of leaking user privacy. 2) Existing federated methods mainly rely on atomic item IDs to represent items, which prevents them from modeling items in a unified feature space, increasing the challenge of knowledge transfer among domains. 3) They are all based on the premise of knowing overlapped users between domains, which proves impractical in real-world applications. To address the above limitations, we focus on Privacy-preserving Cross-domain Recommendation (PCDR) and propose PFCR as our solution. For Limitation 1, we develop a FL schema by exclusively utilizing users' interactions with local clients and devising an encryption method for gradient encryption. For Limitation 2, we model items in a universal feature space by their description texts. For Limitation 3, we initially learn federated content representations, harnessing the generality of natural language to establish bridges between domains. Subsequently, we craft two prompt fine-tuning strategies to tailor the pre-trained model to the target domain. Extensive experiments on two real-world datasets demonstrate the superiority of our PFCR method compared to the SOTA approaches.

{{</citation>}}


### (65/80) Physical Trajectory Inference Attack and Defense in Decentralized POI Recommendation (Jing Long et al., 2024)

{{<citation>}}

Jing Long, Tong Chen, Guanhua Ye, Kai Zheng, Nguyen Quoc Viet Hung, Hongzhi Yin. (2024)  
**Physical Trajectory Inference Attack and Defense in Decentralized POI Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Social Network  
[Paper Link](http://arxiv.org/abs/2401.14583v1)  

---


**ABSTRACT**  
As an indispensable personalized service within Location-Based Social Networks (LBSNs), the Point-of-Interest (POI) recommendation aims to assist individuals in discovering attractive and engaging places. However, the accurate recommendation capability relies on the powerful server collecting a vast amount of users' historical check-in data, posing significant risks of privacy breaches. Although several collaborative learning (CL) frameworks for POI recommendation enhance recommendation resilience and allow users to keep personal data on-device, they still share personal knowledge to improve recommendation performance, thus leaving vulnerabilities for potential attackers. Given this, we design a new Physical Trajectory Inference Attack (PTIA) to expose users' historical trajectories. Specifically, for each user, we identify the set of interacted POIs by analyzing the aggregated information from the target POIs and their correlated POIs. We evaluate the effectiveness of PTIA on two real-world datasets across two types of decentralized CL frameworks for POI recommendation. Empirical results demonstrate that PTIA poses a significant threat to users' historical trajectories. Furthermore, Local Differential Privacy (LDP), the traditional privacy-preserving method for CL frameworks, has also been proven ineffective against PTIA. In light of this, we propose a novel defense mechanism (AGD) against PTIA based on an adversarial game to eliminate sensitive POIs and their information in correlated POIs. After conducting intensive experiments, AGD has been proven precise and practical, with minimal impact on recommendation performance.

{{</citation>}}


## cs.HC (2)



### (66/80) Appropriateness of LLM-equipped Robotic Well-being Coach Language in the Workplace: A Qualitative Evaluation (Micol Spitale et al., 2024)

{{<citation>}}

Micol Spitale, Minja Axelsson, Hatice Gunes. (2024)  
**Appropriateness of LLM-equipped Robotic Well-being Coach Language in the Workplace: A Qualitative Evaluation**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs-RO, cs.HC  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.14935v1)  

---


**ABSTRACT**  
Robotic coaches have been recently investigated to promote mental well-being in various contexts such as workplaces and homes. With the widespread use of Large Language Models (LLMs), HRI researchers are called to consider language appropriateness when using such generated language for robotic mental well-being coaches in the real world. Therefore, this paper presents the first work that investigated the language appropriateness of robot mental well-being coach in the workplace. To this end, we conducted an empirical study that involved 17 employees who interacted over 4 weeks with a robotic mental well-being coach equipped with LLM-based capabilities. After the study, we individually interviewed them and we conducted a focus group of 1.5 hours with 11 of them. The focus group consisted of: i) an ice-breaking activity, ii) evaluation of robotic coach language appropriateness in various scenarios, and iii) listing shoulds and shouldn'ts for designing appropriate robotic coach language for mental well-being. From our qualitative evaluation, we found that a language-appropriate robotic coach should (1) ask deep questions which explore feelings of the coachees, rather than superficial questions, (2) express and show emotional and empathic understanding of the context, and (3) not make any assumptions without clarifying with follow-up questions to avoid bias and stereotyping. These results can inform the design of language-appropriate robotic coach to promote mental well-being in real-world contexts.

{{</citation>}}


### (67/80) Charting the Future of AI in Project-Based Learning: A Co-Design Exploration with Students (Chengbo Zheng et al., 2024)

{{<citation>}}

Chengbo Zheng, Kangyu Yuan, Bingcan Guo, Reza Hadi Mogavi, Zhenhui Peng, Shuai Ma, Xiaojuan Ma. (2024)  
**Charting the Future of AI in Project-Based Learning: A Co-Design Exploration with Students**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.14915v1)  

---


**ABSTRACT**  
The increasing use of Artificial Intelligence (AI) by students in learning presents new challenges for assessing their learning outcomes in project-based learning (PBL). This paper introduces a co-design study to explore the potential of students' AI usage data as a novel material for PBL assessment. We conducted workshops with 18 college students, encouraging them to speculate an alternative world where they could freely employ AI in PBL while needing to report this process to assess their skills and contributions. Our workshops yielded various scenarios of students' use of AI in PBL and ways of analyzing these uses grounded by students' vision of education goal transformation. We also found students with different attitudes toward AI exhibited distinct preferences in how to analyze and understand the use of AI. Based on these findings, we discuss future research opportunities on student-AI interactions and understanding AI-enhanced learning.

{{</citation>}}


## cs.SI (2)



### (68/80) More inclusive and on wider sources: A Comparative Analysis of Data and Political Journalists on Twitter in Germany (Benedict Witzenberger et al., 2024)

{{<citation>}}

Benedict Witzenberger, Jürgen Pfeffer. (2024)  
**More inclusive and on wider sources: A Comparative Analysis of Data and Political Journalists on Twitter in Germany**  

---
Primary Category: cs.SI  
Categories: J-5; K-4-2; H-4-3, cs-SI, cs.SI  
Keywords: Social Media, Twitter  
[Paper Link](http://arxiv.org/abs/2401.14925v1)  

---


**ABSTRACT**  
Women are underrepresented in many areas of journalistic newsrooms. In this paper, we examine if this established effect continues in the new forms of journalistic communication, Social Media Networks. We used mentions, retweets, and hashtags as journalistic amplification and legitimation measures. Furthermore, we compared two groups of journalists in different stages of development: political and data journalists in Germany in 2021. Our results show that journalists regarded as women tend to favor their other women in mentions and retweets on Twitter, compared to men. While both professions are dominated by many men and a high share of men-authored tweets, women are mentioning and retweeting other women to a more extensive degree than their male colleagues. Women data journalists also leveraged different sources than men. In addition, we have found data journalists to be more inclusive towards non-member sources in their network compared to political journalists.

{{</citation>}}


### (69/80) Assembling a Multi-Platform Ensemble Social Bot Detector with Applications to US 2020 Elections (Lynnette Hui Xian Ng et al., 2024)

{{<citation>}}

Lynnette Hui Xian Ng, Kathleen M. Carley. (2024)  
**Assembling a Multi-Platform Ensemble Social Bot Detector with Applications to US 2020 Elections**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2401.14607v1)  

---


**ABSTRACT**  
Bots have been in the spotlight for many social media studies, for they have been observed to be participating in the manipulation of information and opinions on social media. These studies analyzed the activity and influence of bots in a variety of contexts: elections, protests, health communication and so forth. Prior to this analyses is the identification of bot accounts to segregate the class of social media users. In this work, we propose an ensemble method for bot detection, designing a multi-platform bot detection architecture to handle several problems along the bot detection pipeline: incomplete data input, minimal feature engineering, optimized classifiers for each data field, and also eliminate the need for a threshold value for classification determination. With these design decisions, we generalize our bot detection framework across Twitter, Reddit and Instagram. We also perform feature importance analysis, observing that the entropy of names and number of interactions (retweets/shares) are important factors in bot determination. Finally, we apply our multi-platform bot detector to the US 2020 presidential elections to identify and analyze bot activity across multiple social media platforms, showcasing the difference in online discourse of bots from different platforms.

{{</citation>}}


## cs.AI (3)



### (70/80) Reinforcement Learning Interventions on Boundedly Rational Human Agents in Frictionful Tasks (Eura Nofshin et al., 2024)

{{<citation>}}

Eura Nofshin, Siddharth Swaroop, Weiwei Pan, Susan Murphy, Finale Doshi-Velez. (2024)  
**Reinforcement Learning Interventions on Boundedly Rational Human Agents in Frictionful Tasks**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.14923v1)  

---


**ABSTRACT**  
Many important behavior changes are frictionful; they require individuals to expend effort over a long period with little immediate gratification. Here, an artificial intelligence (AI) agent can provide personalized interventions to help individuals stick to their goals. In these settings, the AI agent must personalize rapidly (before the individual disengages) and interpretably, to help us understand the behavioral interventions. In this paper, we introduce Behavior Model Reinforcement Learning (BMRL), a framework in which an AI agent intervenes on the parameters of a Markov Decision Process (MDP) belonging to a boundedly rational human agent. Our formulation of the human decision-maker as a planning agent allows us to attribute undesirable human policies (ones that do not lead to the goal) to their maladapted MDP parameters, such as an extremely low discount factor. Furthermore, we propose a class of tractable human models that captures fundamental behaviors in frictionful tasks. Introducing a notion of MDP equivalence specific to BMRL, we theoretically and empirically show that AI planning with our human models can lead to helpful policies on a wide range of more complex, ground-truth humans.

{{</citation>}}


### (71/80) On the Limitations of Markovian Rewards to Express Multi-Objective, Risk-Sensitive, and Modal Tasks (Joar Skalse et al., 2024)

{{<citation>}}

Joar Skalse, Alessandro Abate. (2024)  
**On the Limitations of Markovian Rewards to Express Multi-Objective, Risk-Sensitive, and Modal Tasks**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.14811v1)  

---


**ABSTRACT**  
In this paper, we study the expressivity of scalar, Markovian reward functions in Reinforcement Learning (RL), and identify several limitations to what they can express. Specifically, we look at three classes of RL tasks; multi-objective RL, risk-sensitive RL, and modal RL. For each class, we derive necessary and sufficient conditions that describe when a problem in this class can be expressed using a scalar, Markovian reward. Moreover, we find that scalar, Markovian rewards are unable to express most of the instances in each of these three classes. We thereby contribute to a more complete understanding of what standard reward functions can and cannot express. In addition to this, we also call attention to modal problems as a new class of problems, since they have so far not been given any systematic treatment in the RL literature. We also briefly outline some approaches for solving some of the problems we discuss, by means of bespoke RL algorithms.

{{</citation>}}


### (72/80) Synthetic Multimodal Dataset for Empowering Safety and Well-being in Home Environments (Takanori Ugai et al., 2024)

{{<citation>}}

Takanori Ugai, Shusaku Egami, Swe Nwe Nwe Htun, Kouji Kozaki, Takahiro Kawamura, Ken Fukuda. (2024)  
**Synthetic Multimodal Dataset for Empowering Safety and Well-being in Home Environments**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Knowledge Graph, Reasoning  
[Paper Link](http://arxiv.org/abs/2401.14743v1)  

---


**ABSTRACT**  
This paper presents a synthetic multimodal dataset of daily activities that fuses video data from a 3D virtual space simulator with knowledge graphs depicting the spatiotemporal context of the activities. The dataset is developed for the Knowledge Graph Reasoning Challenge for Social Issues (KGRC4SI), which focuses on identifying and addressing hazardous situations in the home environment. The dataset is available to the public as a valuable resource for researchers and practitioners developing innovative solutions recognizing human behaviors to enhance safety and well-being in

{{</citation>}}


## cs.CY (1)



### (73/80) A Framework for Assurance Audits of Algorithmic Systems (Khoa Lam et al., 2024)

{{<citation>}}

Khoa Lam, Benjamin Lange, Borhane Blili-Hamelin, Jovana Davidovic, Shea Brown, Ali Hasan. (2024)  
**A Framework for Assurance Audits of Algorithmic Systems**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.14908v1)  

---


**ABSTRACT**  
An increasing number of regulations propose the notion of AI audits as an enforcement mechanism for achieving transparency and accountability for AI systems. Despite some converging norms around various forms of AI auditing, auditing for the purpose of compliance and assurance currently have little to no agreed upon practices, procedures, taxonomies, and standards. We propose the criterion audit as an operationalizable compliance and assurance external audit framework. We model elements of this approach after financial auditing practices, and argue that AI audits should similarly provide assurance to their stakeholders about AI organizations' ability to govern their algorithms in ways that mitigate harms and uphold human values. We discuss the necessary conditions for the criterion audit, and provide a procedural blueprint for performing an audit engagement in practice. We illustrate how this framework can be adapted to current regulations by deriving the criteria on which bias audits for hiring algorithms can be performed, as required by the recently effective New York City Local Law 144 of 2021. We conclude by offering critical discussion on the benefits, inherent limitations, and implementation challenges of applying practices of the more mature financial auditing industry to AI auditing where robust guardrails against quality assurance issues are only starting to emerge. Our discussion as informed by experiences in performing these audits in practice highlights the critical role that an audit ecosystem plays in ensuring the effectiveness of such methodology.

{{</citation>}}


## cs.SD (2)



### (74/80) Expressivity-aware Music Performance Retrieval using Mid-level Perceptual Features and Emotion Word Embeddings (Shreyan Chowdhury et al., 2024)

{{<citation>}}

Shreyan Chowdhury, Gerhard Widmer. (2024)  
**Expressivity-aware Music Performance Retrieval using Mid-level Perceptual Features and Emotion Word Embeddings**  

---
Primary Category: cs.SD  
Categories: cs-IR, cs-SD, cs.SD, eess-AS  
Keywords: Embedding, Word Embedding  
[Paper Link](http://arxiv.org/abs/2401.14826v1)  

---


**ABSTRACT**  
This paper explores a specific sub-task of cross-modal music retrieval. We consider the delicate task of retrieving a performance or rendition of a musical piece based on a description of its style, expressive character, or emotion from a set of different performances of the same piece. We observe that a general purpose cross-modal system trained to learn a common text-audio embedding space does not yield optimal results for this task. By introducing two changes -- one each to the text encoder and the audio encoder -- we demonstrate improved performance on a dataset of piano performances and associated free-text descriptions. On the text side, we use emotion-enriched word embeddings (EWE) and on the audio side, we extract mid-level perceptual features instead of generic audio embeddings. Our results highlight the effectiveness of mid-level perceptual features learnt from music and emotion enriched word embeddings learnt from emotion-labelled text in capturing musical expression in a cross-modal setting. Additionally, our interpretable mid-level features provide a route for introducing explainability in the retrieval and downstream recommendation processes.

{{</citation>}}


### (75/80) UNIT-DSR: Dysarthric Speech Reconstruction System Using Speech Unit Normalization (Yuejiao Wang et al., 2024)

{{<citation>}}

Yuejiao Wang, Xixin Wu, Disong Wang, Lingwei Meng, Helen Meng. (2024)  
**UNIT-DSR: Dysarthric Speech Reconstruction System Using Speech Unit Normalization**  

---
Primary Category: cs.SD  
Categories: cs-CL, cs-SD, cs.SD, eess-AS  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2401.14664v1)  

---


**ABSTRACT**  
Dysarthric speech reconstruction (DSR) systems aim to automatically convert dysarthric speech into normal-sounding speech. The technology eases communication with speakers affected by the neuromotor disorder and enhances their social inclusion. NED-based (Neural Encoder-Decoder) systems have significantly improved the intelligibility of the reconstructed speech as compared with GAN-based (Generative Adversarial Network) approaches, but the approach is still limited by training inefficiency caused by the cascaded pipeline and auxiliary tasks of the content encoder, which may in turn affect the quality of reconstruction. Inspired by self-supervised speech representation learning and discrete speech units, we propose a Unit-DSR system, which harnesses the powerful domain-adaptation capacity of HuBERT for training efficiency improvement and utilizes speech units to constrain the dysarthric content restoration in a discrete linguistic space. Compared with NED approaches, the Unit-DSR system only consists of a speech unit normalizer and a Unit HiFi-GAN vocoder, which is considerably simpler without cascaded sub-modules or auxiliary tasks. Results on the UASpeech corpus indicate that Unit-DSR outperforms competitive baselines in terms of content restoration, reaching a 28.2% relative average word error rate reduction when compared to original dysarthric speech, and shows robustness against speed perturbation and noise.

{{</citation>}}


## cs.GT (1)



### (76/80) Keeping the Harmony Between Neighbors: Local Fairness in Graph Fair Division (Halvard Hummel et al., 2024)

{{<citation>}}

Halvard Hummel, Ayumi Igarashi. (2024)  
**Keeping the Harmony Between Neighbors: Local Fairness in Graph Fair Division**  

---
Primary Category: cs.GT  
Categories: 91B32, cs-DS, cs-GT, cs-MA, cs.GT  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.14825v1)  

---


**ABSTRACT**  
We study the problem of allocating indivisible resources under the connectivity constraints of a graph $G$. This model, initially introduced by Bouveret et al. (published in IJCAI, 2017), effectively encompasses a diverse array of scenarios characterized by spatial or temporal limitations, including the division of land plots and the allocation of time plots. In this paper, we introduce a novel fairness concept that integrates local comparisons within the social network formed by a connected allocation of the item graph. Our particular focus is to achieve pairwise-maximin fair share (PMMS) among the "neighbors" within this network. For any underlying graph structure, we show that a connected allocation that maximizes Nash welfare guarantees a $(1/2)$-PMMS fairness. Moreover, for two agents, we establish that a $(3/4)$-PMMS allocation can be efficiently computed. Additionally, we demonstrate that for three agents and the items aligned on a path, a PMMS allocation is always attainable and can be computed in polynomial time. Lastly, when agents have identical additive utilities, we present a pseudo-polynomial-time algorithm for a $(3/4)$-PMMS allocation, irrespective of the underlying graph $G$. Furthermore, we provide a polynomial-time algorithm for obtaining a PMMS allocation when $G$ is a tree.

{{</citation>}}


## q-bio.QM (1)



### (77/80) Endowing Protein Language Models with Structural Knowledge (Dexiong Chen et al., 2024)

{{<citation>}}

Dexiong Chen, Philip Hartout, Paolo Pellizzoni, Carlos Oliver, Karsten Borgwardt. (2024)  
**Endowing Protein Language Models with Structural Knowledge**  

---
Primary Category: q-bio.QM  
Categories: cs-LG, q-bio-BM, q-bio-QM, q-bio.QM  
Keywords: Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2401.14819v1)  

---


**ABSTRACT**  
Understanding the relationships between protein sequence, structure and function is a long-standing biological challenge with manifold implications from drug design to our understanding of evolution. Recently, protein language models have emerged as the preferred method for this challenge, thanks to their ability to harness large sequence databases. Yet, their reliance on expansive sequence data and parameter sets limits their flexibility and practicality in real-world scenarios. Concurrently, the recent surge in computationally predicted protein structures unlocks new opportunities in protein representation learning. While promising, the computational burden carried by such complex data still hinders widely-adopted practical applications. To address these limitations, we introduce a novel framework that enhances protein language models by integrating protein structural data. Drawing from recent advances in graph transformers, our approach refines the self-attention mechanisms of pretrained language transformers by integrating structural information with structure extractor modules. This refined model, termed Protein Structure Transformer (PST), is further pretrained on a small protein structure database, using the same masked language modeling objective as traditional protein language models. Empirical evaluations of PST demonstrate its superior parameter efficiency relative to protein language models, despite being pretrained on a dataset comprising only 542K structures. Notably, PST consistently outperforms the state-of-the-art foundation model for protein sequences, ESM-2, setting a new benchmark in protein function prediction. Our findings underscore the potential of integrating structural information into protein language models, paving the way for more effective and efficient protein modeling Code and pretrained models are available at https://github.com/BorgwardtLab/PST.

{{</citation>}}


## cs.IT (2)



### (78/80) Adversarial Attacks and Defenses in 6G Network-Assisted IoT Systems (Bui Duc Son et al., 2024)

{{<citation>}}

Bui Duc Son, Trinh Van Chien, Waqas Khalid, Mohamed Amine Ferrag, Wan Choi, Merouane Debbah. (2024)  
**Adversarial Attacks and Defenses in 6G Network-Assisted IoT Systems**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, math-IT  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2401.14780v1)  

---


**ABSTRACT**  
The Internet of Things (IoT) and massive IoT systems are key to sixth-generation (6G) networks due to dense connectivity, ultra-reliability, low latency, and high throughput. Artificial intelligence, including deep learning and machine learning, offers solutions for optimizing and deploying cutting-edge technologies for future radio communications. However, these techniques are vulnerable to adversarial attacks, leading to degraded performance and erroneous predictions, outcomes unacceptable for ubiquitous networks. This survey extensively addresses adversarial attacks and defense methods in 6G network-assisted IoT systems. The theoretical background and up-to-date research on adversarial attacks and defenses are discussed. Furthermore, we provide Monte Carlo simulations to validate the effectiveness of adversarial attacks compared to jamming attacks. Additionally, we examine the vulnerability of 6G IoT systems by demonstrating attack strategies applicable to key technologies, including reconfigurable intelligent surfaces, massive multiple-input multiple-output (MIMO)/cell-free massive MIMO, satellites, the metaverse, and semantic communications. Finally, we outline the challenges and future developments associated with adversarial attacks and defenses in 6G IoT systems.

{{</citation>}}


### (79/80) Topology-Aware Exploration of Energy-Based Models Equilibrium: Toric QC-LDPC Codes and Hyperbolic MET QC-LDPC Codes (Vasiliy Usatyuk et al., 2024)

{{<citation>}}

Vasiliy Usatyuk, Denis Sapozhnikov, Sergey Egorov. (2024)  
**Topology-Aware Exploration of Energy-Based Models Equilibrium: Toric QC-LDPC Codes and Hyperbolic MET QC-LDPC Codes**  

---
Primary Category: cs.IT  
Categories: cs-AI, cs-CV, cs-IT, cs-LG, cs-SY, cs.IT, eess-SY, math-IT  
Keywords: Natural Language Processing, Transformer  
[Paper Link](http://arxiv.org/abs/2401.14749v1)  

---


**ABSTRACT**  
This paper presents a method for achieving equilibrium in the ISING Hamiltonian when confronted with unevenly distributed charges on an irregular grid. Employing (Multi-Edge) QC-LDPC codes and the Boltzmann machine, our approach involves dimensionally expanding the system, substituting charges with circulants, and representing distances through circulant shifts. This results in a systematic mapping of the charge system onto a space, transforming the irregular grid into a uniform configuration, applicable to Torical and Circular Hyperboloid Topologies. The paper covers fundamental definitions and notations related to QC-LDPC Codes, Multi-Edge QC-LDPC codes, and the Boltzmann machine. It explores the marginalization problem in code on the graph probabilistic models for evaluating the partition function, encompassing exact and approximate estimation techniques. Rigorous proof is provided for the attainability of equilibrium states for the Boltzmann machine under Torical and Circular Hyperboloid, paving the way for the application of our methodology. Practical applications of our approach are investigated in Finite Geometry QC-LDPC Codes, specifically in Material Science. The paper further explores its effectiveness in the realm of Natural Language Processing Transformer Deep Neural Networks, examining Generalized Repeat Accumulate Codes, Spatially-Coupled and Cage-Graph QC-LDPC Codes. The versatile and impactful nature of our topology-aware hardware-efficient quasi-cycle codes equilibrium method is showcased across diverse scientific domains without the use of specific section delineations.

{{</citation>}}


## eess.IV (1)



### (80/80) Additional Look into GAN-based Augmentation for Deep Learning COVID-19 Image Classification (Oleksandr Fedoruk et al., 2024)

{{<citation>}}

Oleksandr Fedoruk, Konrad Klimaszewski, Aleksander Ogonowski, Michał Kruk. (2024)  
**Additional Look into GAN-based Augmentation for Deep Learning COVID-19 Image Classification**  

---
Primary Category: eess.IV  
Categories: I-5, cs-CV, eess-IV, eess.IV  
Keywords: Augmentation, Image Classification  
[Paper Link](http://arxiv.org/abs/2401.14705v1)  

---


**ABSTRACT**  
The availability of training data is one of the main limitations in deep learning applications for medical imaging. Data augmentation is a popular approach to overcome this problem. A new approach is a Machine Learning based augmentation, in particular usage of Generative Adversarial Networks (GAN). In this case, GANs generate images similar to the original dataset so that the overall training data amount is bigger, which leads to better performance of trained networks. A GAN model consists of two networks, a generator and a discriminator interconnected in a feedback loop which creates a competitive environment. This work is a continuation of the previous research where we trained StyleGAN2-ADA by Nvidia on the limited COVID-19 chest X-ray image dataset. In this paper, we study the dependence of the GAN-based augmentation performance on dataset size with a focus on small samples. Two datasets are considered, one with 1000 images per class (4000 images in total) and the second with 500 images per class (2000 images in total). We train StyleGAN2-ADA with both sets and then, after validating the quality of generated images, we use trained GANs as one of the augmentations approaches in multi-class classification problems. We compare the quality of the GAN-based augmentation approach to two different approaches (classical augmentation and no augmentation at all) by employing transfer learning-based classification of COVID-19 chest X-ray images. The results are quantified using different classification quality metrics and compared to the results from the literature. The GAN-based augmentation approach is found to be comparable with classical augmentation in the case of medium and large datasets but underperforms in the case of smaller datasets. The correlation between the size of the original dataset and the quality of classification is visible independently from the augmentation approach.

{{</citation>}}
