---
draft: false
title: "arXiv @ 2023.09.03"
date: 2023-09-03
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.09.03"
    identifier: arxiv_20230903
    parent: 202309_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.LG (18)](#cslg-18)
- [cs.CV (17)](#cscv-17)
- [cs.CL (14)](#cscl-14)
- [cs.AI (4)](#csai-4)
- [cs.SI (2)](#cssi-2)
- [cs.DC (1)](#csdc-1)
- [cs.SE (1)](#csse-1)
- [stat.ML (1)](#statml-1)
- [cs.CE (1)](#csce-1)
- [cs.IR (2)](#csir-2)
- [cs.CY (1)](#cscy-1)
- [cs.RO (3)](#csro-3)
- [cs.MS (1)](#csms-1)
- [cs.SD (2)](#cssd-2)
- [physics.chem-ph (1)](#physicschem-ph-1)
- [eess.IV (1)](#eessiv-1)
- [cs.AR (1)](#csar-1)

## cs.LG (18)



### (1/71) Efficient RLHF: Reducing the Memory Usage of PPO (Michael Santacroce et al., 2023)

{{<citation>}}

Michael Santacroce, Yadong Lu, Han Yu, Yuanzhi Li, Yelong Shen. (2023)  
**Efficient RLHF: Reducing the Memory Usage of PPO**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.00754v1)  

---


**ABSTRACT**  
Reinforcement Learning with Human Feedback (RLHF) has revolutionized language modeling by aligning models with human preferences. However, the RL stage, Proximal Policy Optimization (PPO), requires over 3x the memory of Supervised Fine-Tuning (SFT), making it infeasible to use for most practitioners. To address this issue, we present a comprehensive analysis the memory usage, performance, and training time of memory-savings techniques for PPO. We introduce Hydra-RLHF by first integrating the SFT and Reward models and then dynamically turning LoRA "off" during training. Our experiments show: 1. Using LoRA during PPO reduces its memory usage to be smaller than SFT while improving alignment across four public benchmarks, and 2. Hydra-PPO reduces the latency per sample of LoRA-PPO by up to 65% while maintaining its performance. Our results demonstrate that Hydra-PPO is a simple and promising solution for enabling more widespread usage of RLHF.

{{</citation>}}


### (2/71) Universal Normalization Enhanced Graph Representation Learning for Gene Network Prediction (Zehao Dong et al., 2023)

{{<citation>}}

Zehao Dong, Muhan Zhang, Qihang Zhao, Philip R. O. Payne, Michael Province, Carlos Cruchaga, Tianyu Zhao, Yixin Chen, Fuhai Li. (2023)  
**Universal Normalization Enhanced Graph Representation Learning for Gene Network Prediction**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Representation Learning  
[Paper Link](http://arxiv.org/abs/2309.00738v1)  

---


**ABSTRACT**  
Effective gene network representation learning is of great importance in bioinformatics to predict/understand the relation of gene profiles and disease phenotypes. Though graph neural networks (GNNs) have been the dominant architecture for analyzing various graph-structured data like social networks, their predicting on gene networks often exhibits subpar performance. In this paper, we formally investigate the gene network representation learning problem and characterize a notion of \textit{universal graph normalization}, where graph normalization can be applied in an universal manner to maximize the expressive power of GNNs while maintaining the stability. We propose a novel UNGNN (Universal Normalized GNN) framework, which leverages universal graph normalization in both the message passing phase and readout layer to enhance the performance of a base GNN. UNGNN has a plug-and-play property and can be combined with any GNN backbone in practice. A comprehensive set of experiments on gene-network-based bioinformatical tasks demonstrates that our UNGNN model significantly outperforms popular GNN benchmarks and provides an overall performance improvement of 16 $\%$ on average compared to previous state-of-the-art (SOTA) baselines. Furthermore, we also evaluate our theoretical findings on other graph datasets where the universal graph normalization is solvable, and we observe that UNGNN consistently achieves the superior performance.

{{</citation>}}


### (3/71) Geometric Deep Learning: a Temperature Based Analysis of Graph Neural Networks (M. Lapenna et al., 2023)

{{<citation>}}

M. Lapenna, F. Faglioni, F. Zanchetta, R. Fioresi. (2023)  
**Geometric Deep Learning: a Temperature Based Analysis of Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2309.00699v1)  

---


**ABSTRACT**  
We examine a Geometric Deep Learning model as a thermodynamic system treating the weights as non-quantum and non-relativistic particles. We employ the notion of temperature previously defined in [7] and study it in the various layers for GCN and GAT models. Potential future applications of our findings are discussed.

{{</citation>}}


### (4/71) Jointly Exploring Client Drift and Catastrophic Forgetting in Dynamic Learning (Niklas Babendererde et al., 2023)

{{<citation>}}

Niklas Babendererde, Moritz Fuchs, Camila Gonzalez, Yuri Tolkach, Anirban Mukhopadhyay. (2023)  
**Jointly Exploring Client Drift and Catastrophic Forgetting in Dynamic Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Computer Vision  
[Paper Link](http://arxiv.org/abs/2309.00688v1)  

---


**ABSTRACT**  
Federated and Continual Learning have emerged as potential paradigms for the robust and privacy-aware use of Deep Learning in dynamic environments. However, Client Drift and Catastrophic Forgetting are fundamental obstacles to guaranteeing consistent performance. Existing work only addresses these problems separately, which neglects the fact that the root cause behind both forms of performance deterioration is connected. We propose a unified analysis framework for building a controlled test environment for Client Drift -- by perturbing a defined ratio of clients -- and Catastrophic Forgetting -- by shifting all clients with a particular strength. Our framework further leverages this new combined analysis by generating a 3D landscape of the combined performance impact from both. We demonstrate that the performance drop through Client Drift, caused by a certain share of shifted clients, is correlated to the drop from Catastrophic Forgetting resulting from a corresponding shift strength. Correlation tests between both problems for Computer Vision (CelebA) and Medical Imaging (PESO) support this new perspective, with an average Pearson rank correlation coefficient of over 0.94. Our framework's novel ability of combined spatio-temporal shift analysis allows us to investigate how both forms of distribution shift behave in mixed scenarios, opening a new pathway for better generalization. We show that a combination of moderate Client Drift and Catastrophic Forgetting can even improve the performance of the resulting model (causing a "Generalization Bump") compared to when only one of the shifts occurs individually. We apply a simple and commonly used method from Continual Learning in the federated setting and observe this phenomenon to be reoccurring, leveraging the ability of our framework to analyze existing and novel methods for Federated and Continual Learning.

{{</citation>}}


### (5/71) Baseline Defenses for Adversarial Attacks Against Aligned Language Models (Neel Jain et al., 2023)

{{<citation>}}

Neel Jain, Avi Schwarzschild, Yuxin Wen, Gowthami Somepalli, John Kirchenbauer, Ping-yeh Chiang, Micah Goldblum, Aniruddha Saha, Jonas Geiping, Tom Goldstein. (2023)  
**Baseline Defenses for Adversarial Attacks Against Aligned Language Models**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-CR, cs-LG, cs.LG  
Keywords: Adversarial Attack, Language Model  
[Paper Link](http://arxiv.org/abs/2309.00614v2)  

---


**ABSTRACT**  
As Large Language Models quickly become ubiquitous, it becomes critical to understand their security vulnerabilities. Recent work shows that text optimizers can produce jailbreaking prompts that bypass moderation and alignment. Drawing from the rich body of work on adversarial machine learning, we approach these attacks with three questions: What threat models are practically useful in this domain? How do baseline defense techniques perform in this new domain? How does LLM security differ from computer vision?   We evaluate several baseline defense strategies against leading adversarial attacks on LLMs, discussing the various settings in which each is feasible and effective. Particularly, we look at three types of defenses: detection (perplexity based), input preprocessing (paraphrase and retokenization), and adversarial training. We discuss white-box and gray-box settings and discuss the robustness-performance trade-off for each of the defenses considered. We find that the weakness of existing discrete optimizers for text, combined with the relatively high costs of optimization, makes standard adaptive attacks more challenging for LLMs. Future research will be needed to uncover whether more powerful optimizers can be developed, or whether the strength of filtering and preprocessing defenses is greater in the LLMs domain than it has been in computer vision.

{{</citation>}}


### (6/71) Fast and Regret Optimal Best Arm Identification: Fundamental Limits and Low-Complexity Algorithms (Qining Zhang et al., 2023)

{{<citation>}}

Qining Zhang, Lei Ying. (2023)  
**Fast and Regret Optimal Best Arm Identification: Fundamental Limits and Low-Complexity Algorithms**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.00591v1)  

---


**ABSTRACT**  
This paper considers a stochastic multi-armed bandit (MAB) problem with dual objectives: (i) quick identification and commitment to the optimal arm, and (ii) reward maximization throughout a sequence of $T$ consecutive rounds. Though each objective has been individually well-studied, i.e., best arm identification for (i) and regret minimization for (ii), the simultaneous realization of both objectives remains an open problem, despite its practical importance. This paper introduces \emph{Regret Optimal Best Arm Identification} (ROBAI) which aims to achieve these dual objectives. To solve ROBAI with both pre-determined stopping time and adaptive stopping time requirements, we present the $\mathsf{EOCP}$ algorithm and its variants respectively, which not only achieve asymptotic optimal regret in both Gaussian and general bandits, but also commit to the optimal arm in $\mathcal{O}(\log T)$ rounds with pre-determined stopping time and $\mathcal{O}(\log^2 T)$ rounds with adaptive stopping time. We further characterize lower bounds on the commitment time (equivalent to sample complexity) of ROBAI, showing that $\mathsf{EOCP}$ and its variants are sample optimal with pre-determined stopping time, and almost sample optimal with adaptive stopping time. Numerical results confirm our theoretical analysis and reveal an interesting ``over-exploration'' phenomenon carried by classic $\mathsf{UCB}$ algorithms, such that $\mathsf{EOCP}$ has smaller regret even though it stops exploration much earlier than $\mathsf{UCB}$ ($\mathcal{O}(\log T)$ versus $\mathcal{O}(T)$), which suggests over-exploration is unnecessary and potentially harmful to system performance.

{{</citation>}}


### (7/71) PolyGET: Accelerating Polymer Simulations by Accurate and Generalizable Forcefield with Equivariant Transformer (Rui Feng et al., 2023)

{{<citation>}}

Rui Feng, Huan Tran, Aubrey Toland, Binghong Chen, Qi Zhu, Rampi Ramprasad, Chao Zhang. (2023)  
**PolyGET: Accelerating Polymer Simulations by Accurate and Generalizable Forcefield with Equivariant Transformer**  

---
Primary Category: cs.LG  
Categories: cond-mat-soft, cs-LG, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.00585v1)  

---


**ABSTRACT**  
Polymer simulation with both accuracy and efficiency is a challenging task. Machine learning (ML) forcefields have been developed to achieve both the accuracy of ab initio methods and the efficiency of empirical force fields. However, existing ML force fields are usually limited to single-molecule settings, and their simulations are not robust enough. In this paper, we present PolyGET, a new framework for Polymer Forcefields with Generalizable Equivariant Transformers. PolyGET is designed to capture complex quantum interactions between atoms and generalize across various polymer families, using a deep learning model called Equivariant Transformers. We propose a new training paradigm that focuses exclusively on optimizing forces, which is different from existing methods that jointly optimize forces and energy. This simple force-centric objective function avoids competing objectives between energy and forces, thereby allowing for learning a unified forcefield ML model over different polymer families. We evaluated PolyGET on a large-scale dataset of 24 distinct polymer types and demonstrated state-of-the-art performance in force accuracy and robust MD simulations. Furthermore, PolyGET can simulate large polymers with high fidelity to the reference ab initio DFT method while being able to generalize to unseen polymers.

{{</citation>}}


### (8/71) Curating Naturally Adversarial Datasets for Trustworthy AI in Healthcare (Sydney Pugh et al., 2023)

{{<citation>}}

Sydney Pugh, Ivan Ruchkin, Insup Lee, James Weimer. (2023)  
**Curating Naturally Adversarial Datasets for Trustworthy AI in Healthcare**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.00543v1)  

---


**ABSTRACT**  
Deep learning models have shown promising predictive accuracy for time-series healthcare applications. However, ensuring the robustness of these models is vital for building trustworthy AI systems. Existing research predominantly focuses on robustness to synthetic adversarial examples, crafted by adding imperceptible perturbations to clean input data. However, these synthetic adversarial examples do not accurately reflect the most challenging real-world scenarios, especially in the context of healthcare data. Consequently, robustness to synthetic adversarial examples may not necessarily translate to robustness against naturally occurring adversarial examples, which is highly desirable for trustworthy AI. We propose a method to curate datasets comprised of natural adversarial examples to evaluate model robustness. The method relies on probabilistic labels obtained from automated weakly-supervised labeling that combines noisy and cheap-to-obtain labeling heuristics. Based on these labels, our method adversarially orders the input data and uses this ordering to construct a sequence of increasingly adversarial datasets. Our evaluation on six medical case studies and three non-medical case studies demonstrates the efficacy and statistical validity of our approach to generating naturally adversarial datasets

{{</citation>}}


### (9/71) Geometry-aware Line Graph Transformer Pre-training for Molecular Property Prediction (Peizhen Bai et al., 2023)

{{<citation>}}

Peizhen Bai, Xianyuan Liu, Haiping Lu. (2023)  
**Geometry-aware Line Graph Transformer Pre-training for Molecular Property Prediction**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, q-bio-BM  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.00483v1)  

---


**ABSTRACT**  
Molecular property prediction with deep learning has gained much attention over the past years. Owing to the scarcity of labeled molecules, there has been growing interest in self-supervised learning methods that learn generalizable molecular representations from unlabeled data. Molecules are typically treated as 2D topological graphs in modeling, but it has been discovered that their 3D geometry is of great importance in determining molecular functionalities. In this paper, we propose the Geometry-aware line graph transformer (Galformer) pre-training, a novel self-supervised learning framework that aims to enhance molecular representation learning with 2D and 3D modalities. Specifically, we first design a dual-modality line graph transformer backbone to encode the topological and geometric information of a molecule. The designed backbone incorporates effective structural encodings to capture graph structures from both modalities. Then we devise two complementary pre-training tasks at the inter and intra-modality levels. These tasks provide properly supervised information and extract discriminative 2D and 3D knowledge from unlabeled molecules. Finally, we evaluate Galformer against six state-of-the-art baselines on twelve property prediction benchmarks via downstream fine-tuning. Experimental results show that Galformer consistently outperforms all baselines on both classification and regression tasks, demonstrating its effectiveness.

{{</citation>}}


### (10/71) Where Did the Gap Go? Reassessing the Long-Range Graph Benchmark (Jan Tönshoff et al., 2023)

{{<citation>}}

Jan Tönshoff, Martin Ritzert, Eran Rosenbluth, Martin Grohe. (2023)  
**Where Did the Gap Go? Reassessing the Long-Range Graph Benchmark**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.00367v2)  

---


**ABSTRACT**  
The recent Long-Range Graph Benchmark (LRGB, Dwivedi et al. 2022) introduced a set of graph learning tasks strongly dependent on long-range interaction between vertices. Empirical evidence suggests that on these tasks Graph Transformers significantly outperform Message Passing GNNs (MPGNNs). In this paper, we carefully reevaluate multiple MPGNN baselines as well as the Graph Transformer GPS (Ramp\'a\v{s}ek et al. 2022) on LRGB. Through a rigorous empirical analysis, we demonstrate that the reported performance gap is overestimated due to suboptimal hyperparameter choices. It is noteworthy that across multiple datasets the performance gap completely vanishes after basic hyperparameter optimization. In addition, we discuss the impact of lacking feature normalization for LRGB's vision datasets and highlight a spurious implementation of LRGB's link prediction metric. The principal aim of our paper is to establish a higher standard of empirical rigor within the graph machine learning community.

{{</citation>}}


### (11/71) FederatedScope-LLM: A Comprehensive Package for Fine-tuning Large Language Models in Federated Learning (Weirui Kuang et al., 2023)

{{<citation>}}

Weirui Kuang, Bingchen Qian, Zitao Li, Daoyuan Chen, Dawei Gao, Xuchen Pan, Yuexiang Xie, Yaliang Li, Bolin Ding, Jingren Zhou. (2023)  
**FederatedScope-LLM: A Comprehensive Package for Fine-tuning Large Language Models in Federated Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2309.00363v1)  

---


**ABSTRACT**  
LLMs have demonstrated great capabilities in various NLP tasks. Different entities can further improve the performance of those LLMs on their specific downstream tasks by fine-tuning LLMs. When several entities have similar interested tasks, but their data cannot be shared because of privacy concerns regulations, federated learning (FL) is a mainstream solution to leverage the data of different entities. However, fine-tuning LLMs in federated learning settings still lacks adequate support from existing FL frameworks because it has to deal with optimizing the consumption of significant communication and computational resources, data preparation for different tasks, and distinct information protection demands. This paper first discusses these challenges of federated fine-tuning LLMs, and introduces our package FS-LLM as a main contribution, which consists of the following components: (1) we build an end-to-end benchmarking pipeline, automizing the processes of dataset preprocessing, federated fine-tuning execution, and performance evaluation on federated LLM fine-tuning; (2) we provide comprehensive federated parameter-efficient fine-tuning algorithm implementations and versatile programming interfaces for future extension in FL scenarios with low communication and computation costs, even without accessing the full model; (3) we adopt several accelerating and resource-efficient operators for fine-tuning LLMs with limited resources and the flexible pluggable sub-routines for interdisciplinary study. We conduct extensive experiments to validate the effectiveness of FS-LLM and benchmark advanced LLMs with state-of-the-art parameter-efficient fine-tuning algorithms in FL settings, which also yields valuable insights into federated fine-tuning LLMs for the research community. To facilitate further research and adoption, we release FS-LLM at https://github.com/alibaba/FederatedScope/tree/llm.

{{</citation>}}


### (12/71) Explainable Active Learning for Preference Elicitation (Furkan Cantürk et al., 2023)

{{<citation>}}

Furkan Cantürk, Reyhan Aydoğan. (2023)  
**Explainable Active Learning for Preference Elicitation**  

---
Primary Category: cs.LG  
Categories: I-2-0, cs-AI, cs-IR, cs-LG, cs.LG  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2309.00356v1)  

---


**ABSTRACT**  
Gaining insights into the preferences of new users and subsequently personalizing recommendations necessitate managing user interactions intelligently, namely, posing pertinent questions to elicit valuable information effectively. In this study, our focus is on a specific scenario of the cold-start problem, where the recommendation system lacks adequate user presence or access to other users' data is restricted, obstructing employing user profiling methods utilizing existing data in the system. We employ Active Learning (AL) to solve the addressed problem with the objective of maximizing information acquisition with minimal user effort. AL operates for selecting informative data from a large unlabeled set to inquire an oracle to label them and eventually updating a machine learning (ML) model. We operate AL in an integrated process of unsupervised, semi-supervised, and supervised ML within an explanatory preference elicitation process. It harvests user feedback (given for the system's explanations on the presented items) over informative samples to update an underlying ML model estimating user preferences. The designed user interaction facilitates personalizing the system by incorporating user feedback into the ML model and also enhances user trust by refining the system's explanations on recommendations. We implement the proposed preference elicitation methodology for food recommendation. We conducted human experiments to assess its efficacy in the short term and also experimented with several AL strategies over synthetic user profiles that we created for two food datasets, aiming for long-term performance analysis. The experimental results demonstrate the efficiency of the proposed preference elicitation with limited user-labeled data while also enhancing user trust through accurate explanations.

{{</citation>}}


### (13/71) Multi-fidelity reduced-order surrogate modeling (Paolo Conti et al., 2023)

{{<citation>}}

Paolo Conti, Mengwu Guo, Andrea Manzoni, Attilio Frangi, Steven L. Brunton, J. Nathan Kutz. (2023)  
**Multi-fidelity reduced-order surrogate modeling**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-NA, cs.LG, math-NA  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2309.00325v1)  

---


**ABSTRACT**  
High-fidelity numerical simulations of partial differential equations (PDEs) given a restricted computational budget can significantly limit the number of parameter configurations considered and/or time window evaluated for modeling a given system. Multi-fidelity surrogate modeling aims to leverage less accurate, lower-fidelity models that are computationally inexpensive in order to enhance predictive accuracy when high-fidelity data are limited or scarce. However, low-fidelity models, while often displaying important qualitative spatio-temporal features, fail to accurately capture the onset of instability and critical transients observed in the high-fidelity models, making them impractical as surrogate models. To address this shortcoming, we present a new data-driven strategy that combines dimensionality reduction with multi-fidelity neural network surrogates. The key idea is to generate a spatial basis by applying the classical proper orthogonal decomposition (POD) to high-fidelity solution snapshots, and approximate the dynamics of the reduced states - time-parameter-dependent expansion coefficients of the POD basis - using a multi-fidelity long-short term memory (LSTM) network. By mapping low-fidelity reduced states to their high-fidelity counterpart, the proposed reduced-order surrogate model enables the efficient recovery of full solution fields over time and parameter variations in a non-intrusive manner. The generality and robustness of this method is demonstrated by a collection of parametrized, time-dependent PDE problems where the low-fidelity model can be defined by coarser meshes and/or time stepping, as well as by misspecified physical features. Importantly, the onset of instabilities and transients are well captured by this surrogate modeling technique.

{{</citation>}}


### (14/71) Leveraging Learning Metrics for Improved Federated Learning (Andre Fu, 2023)

{{<citation>}}

Andre Fu. (2023)  
**Leveraging Learning Metrics for Improved Federated Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.00257v1)  

---


**ABSTRACT**  
Currently in the federated setting, no learning schemes leverage the emerging research of explainable artificial intelligence (XAI) in particular the novel learning metrics that help determine how well a model is learning. One of these novel learning metrics is termed `Effective Rank' (ER) which measures the Shannon Entropy of the singular values of a matrix, thus enabling a metric determining how well a layer is mapping. By joining federated learning and the learning metric, effective rank, this work will \textbf{(1)} give the first federated learning metric aggregation method \textbf{(2)} show that effective rank is well-suited to federated problems by out-performing baseline Federated Averaging \cite{konevcny2016federated} and \textbf{(3)} develop a novel weight-aggregation scheme relying on effective rank.

{{</citation>}}


### (15/71) Why do universal adversarial attacks work on large language models?: Geometry might be the answer (Varshini Subhash et al., 2023)

{{<citation>}}

Varshini Subhash, Anna Bialas, Weiwei Pan, Finale Doshi-Velez. (2023)  
**Why do universal adversarial attacks work on large language models?: Geometry might be the answer**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-CR, cs-LG, cs.LG  
Keywords: GPT, Transformer  
[Paper Link](http://arxiv.org/abs/2309.00254v1)  

---


**ABSTRACT**  
Transformer based large language models with emergent capabilities are becoming increasingly ubiquitous in society. However, the task of understanding and interpreting their internal workings, in the context of adversarial attacks, remains largely unsolved. Gradient-based universal adversarial attacks have been shown to be highly effective on large language models and potentially dangerous due to their input-agnostic nature. This work presents a novel geometric perspective explaining universal adversarial attacks on large language models. By attacking the 117M parameter GPT-2 model, we find evidence indicating that universal adversarial triggers could be embedding vectors which merely approximate the semantic information in their adversarial training region. This hypothesis is supported by white-box model analysis comprising dimensionality reduction and similarity measurement of hidden representations. We believe this new geometric perspective on the underlying mechanism driving universal attacks could help us gain deeper insight into the internal workings and failure modes of LLMs, thus enabling their mitigation.

{{</citation>}}


### (16/71) NeuroSurgeon: A Toolkit for Subnetwork Analysis (Michael A. Lepori et al., 2023)

{{<citation>}}

Michael A. Lepori, Ellie Pavlick, Thomas Serre. (2023)  
**NeuroSurgeon: A Toolkit for Subnetwork Analysis**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.00244v1)  

---


**ABSTRACT**  
Despite recent advances in the field of explainability, much remains unknown about the algorithms that neural networks learn to represent. Recent work has attempted to understand trained models by decomposing them into functional circuits (Csord\'as et al., 2020; Lepori et al., 2023). To advance this research, we developed NeuroSurgeon, a python library that can be used to discover and manipulate subnetworks within models in the Huggingface Transformers library (Wolf et al., 2019). NeuroSurgeon is freely available at https://github.com/mlepori1/NeuroSurgeon.

{{</citation>}}


### (17/71) Image Hijacking: Adversarial Images can Control Generative Models at Runtime (Luke Bailey et al., 2023)

{{<citation>}}

Luke Bailey, Euan Ong, Stuart Russell, Scott Emmons. (2023)  
**Image Hijacking: Adversarial Images can Control Generative Models at Runtime**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-CR, cs-LG, cs.LG  
Keywords: LLaMA  
[Paper Link](http://arxiv.org/abs/2309.00236v1)  

---


**ABSTRACT**  
Are foundation models secure from malicious actors? In this work, we focus on the image input to a vision-language model (VLM). We discover image hijacks, adversarial images that control generative models at runtime. We introduce Behavior Matching, a general method for creating image hijacks, and we use it to explore three types of attacks. Specific string attacks generate arbitrary output of the adversary's choosing. Leak context attacks leak information from the context window into the output. Jailbreak attacks circumvent a model's safety training. We study these attacks against LLaVA-2, a state-of-the-art VLM based on CLIP and LLaMA-2, and find that all our attack types have above a 90\% success rate. Moreover, our attacks are automated and require only small image perturbations. These findings raise serious concerns about the security of foundation models. If image hijacks are as difficult to defend against as adversarial examples in CIFAR-10, then it might be many years before a solution is found -- if it even exists.

{{</citation>}}


### (18/71) Subjectivity in Unsupervised Machine Learning Model Selection (Wanyi Chen et al., 2023)

{{<citation>}}

Wanyi Chen, Mary L. Cummings. (2023)  
**Subjectivity in Unsupervised Machine Learning Model Selection**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.00201v1)  

---


**ABSTRACT**  
Model selection is a necessary step in unsupervised machine learning. Despite numerous criteria and metrics, model selection remains subjective. A high degree of subjectivity may lead to questions about repeatability and reproducibility of various machine learning studies and doubts about the robustness of models deployed in the real world. Yet, the impact of modelers' preferences on model selection outcomes remains largely unexplored. This study uses the Hidden Markov Model as an example to investigate the subjectivity involved in model selection. We asked 33 participants and three Large Language Models (LLMs) to make model selections in three scenarios. Results revealed variability and inconsistencies in both the participants' and the LLMs' choices, especially when different criteria and metrics disagree. Sources of subjectivity include varying opinions on the importance of different criteria and metrics, differing views on how parsimonious a model should be, and how the size of a dataset should influence model selection. The results underscore the importance of developing a more standardized way to document subjective choices made in model selection processes.

{{</citation>}}


## cs.CV (17)



### (19/71) Affine-Transformation-Invariant Image Classification by Differentiable Arithmetic Distribution Module (Zijie Tan et al., 2023)

{{<citation>}}

Zijie Tan, Guanfang Dong, Chenqiu Zhao, Anup Basu. (2023)  
**Affine-Transformation-Invariant Image Classification by Differentiable Arithmetic Distribution Module**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2309.00752v1)  

---


**ABSTRACT**  
Although Convolutional Neural Networks (CNNs) have achieved promising results in image classification, they still are vulnerable to affine transformations including rotation, translation, flip and shuffle. The drawback motivates us to design a module which can alleviate the impact from different affine transformations. Thus, in this work, we introduce a more robust substitute by incorporating distribution learning techniques, focusing particularly on learning the spatial distribution information of pixels in images. To rectify the issue of non-differentiability of prior distribution learning methods that rely on traditional histograms, we adopt the Kernel Density Estimation (KDE) to formulate differentiable histograms. On this foundation, we present a novel Differentiable Arithmetic Distribution Module (DADM), which is designed to extract the intrinsic probability distributions from images. The proposed approach is able to enhance the model's robustness to affine transformations without sacrificing its feature extraction capabilities, thus bridging the gap between traditional CNNs and distribution-based learning. We validate the effectiveness of the proposed approach through ablation study and comparative experiments with LeNet.

{{</citation>}}


### (20/71) PathLDM: Text conditioned Latent Diffusion Model for Histopathology (Srikar Yellapragada et al., 2023)

{{<citation>}}

Srikar Yellapragada, Alexandros Graikos, Prateek Prasanna, Tahsin Kurc, Joel Saltz, Dimitris Samaras. (2023)  
**PathLDM: Text conditioned Latent Diffusion Model for Histopathology**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2309.00748v1)  

---


**ABSTRACT**  
To achieve high-quality results, diffusion models must be trained on large datasets. This can be notably prohibitive for models in specialized domains, such as computational pathology. Conditioning on labeled data is known to help in data-efficient model training. Therefore, histopathology reports, which are rich in valuable clinical information, are an ideal choice as guidance for a histopathology generative model. In this paper, we introduce PathLDM, the first text-conditioned Latent Diffusion Model tailored for generating high-quality histopathology images. Leveraging the rich contextual information provided by pathology text reports, our approach fuses image and textual data to enhance the generation process. By utilizing GPT's capabilities to distill and summarize complex text reports, we establish an effective conditioning mechanism. Through strategic conditioning and necessary architectural enhancements, we achieved a SoTA FID score of 7.64 for text-to-image generation on the TCGA-BRCA dataset, significantly outperforming the closest text-conditioned competitor with FID 30.1.

{{</citation>}}


### (21/71) Learned Visual Features to Textual Explanations (Saeid Asgari Taghanaki et al., 2023)

{{<citation>}}

Saeid Asgari Taghanaki, Aliasghar Khani, Amir Khasahmadi, Aditya Sanghi, Karl D. D. Willis, Ali Mahdavi-Amiri. (2023)  
**Learned Visual Features to Textual Explanations**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2309.00733v1)  

---


**ABSTRACT**  
Interpreting the learned features of vision models has posed a longstanding challenge in the field of machine learning. To address this issue, we propose a novel method that leverages the capabilities of large language models (LLMs) to interpret the learned features of pre-trained image classifiers. Our method, called TExplain, tackles this task by training a neural network to establish a connection between the feature space of image classifiers and LLMs. Then, during inference, our approach generates a vast number of sentences to explain the features learned by the classifier for a given image. These sentences are then used to extract the most frequent words, providing a comprehensive understanding of the learned features and patterns within the classifier. Our method, for the first time, utilizes these frequent words corresponding to a visual representation to provide insights into the decision-making process of the independently trained classifier, enabling the detection of spurious correlations, biases, and a deeper comprehension of its behavior. To validate the effectiveness of our approach, we conduct experiments on diverse datasets, including ImageNet-9L and Waterbirds. The results demonstrate the potential of our method to enhance the interpretability and robustness of image classifiers.

{{</citation>}}


### (22/71) AAN: Attributes-Aware Network for Temporal Action Detection (Rui Dai et al., 2023)

{{<citation>}}

Rui Dai, Srijan Das, Michael S. Ryoo, Francois Bremond. (2023)  
**AAN: Attributes-Aware Network for Temporal Action Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2309.00696v1)  

---


**ABSTRACT**  
The challenge of long-term video understanding remains constrained by the efficient extraction of object semantics and the modelling of their relationships for downstream tasks. Although the CLIP visual features exhibit discriminative properties for various vision tasks, particularly in object encoding, they are suboptimal for long-term video understanding. To address this issue, we present the Attributes-Aware Network (AAN), which consists of two key components: the Attributes Extractor and a Graph Reasoning block. These components facilitate the extraction of object-centric attributes and the modelling of their relationships within the video. By leveraging CLIP features, AAN outperforms state-of-the-art approaches on two popular action detection datasets: Charades and Toyota Smarthome Untrimmed datasets.

{{</citation>}}


### (23/71) Point-Bind & Point-LLM: Aligning Point Cloud with Multi-modality for 3D Understanding, Generation, and Instruction Following (Ziyu Guo et al., 2023)

{{<citation>}}

Ziyu Guo, Renrui Zhang, Xiangyang Zhu, Yiwen Tang, Xianzheng Ma, Jiaming Han, Kexin Chen, Peng Gao, Xianzhi Li, Hongsheng Li, Pheng-Ann Heng. (2023)  
**Point-Bind & Point-LLM: Aligning Point Cloud with Multi-modality for 3D Understanding, Generation, and Instruction Following**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs-MM, cs.CV  
Keywords: LLaMA  
[Paper Link](http://arxiv.org/abs/2309.00615v1)  

---


**ABSTRACT**  
We introduce Point-Bind, a 3D multi-modality model aligning point clouds with 2D image, language, audio, and video. Guided by ImageBind, we construct a joint embedding space between 3D and multi-modalities, enabling many promising applications, e.g., any-to-3D generation, 3D embedding arithmetic, and 3D open-world understanding. On top of this, we further present Point-LLM, the first 3D large language model (LLM) following 3D multi-modal instructions. By parameter-efficient fine-tuning techniques, Point-LLM injects the semantics of Point-Bind into pre-trained LLMs, e.g., LLaMA, which requires no 3D instruction data, but exhibits superior 3D and multi-modal question-answering capacity. We hope our work may cast a light on the community for extending 3D point clouds to multi-modality applications. Code is available at https://github.com/ZiyuGuo99/Point-Bind_Point-LLM.

{{</citation>}}


### (24/71) CityDreamer: Compositional Generative Model of Unbounded 3D Cities (Haozhe Xie et al., 2023)

{{<citation>}}

Haozhe Xie, Zhaoxi Chen, Fangzhou Hong, Ziwei Liu. (2023)  
**CityDreamer: Compositional Generative Model of Unbounded 3D Cities**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2309.00610v1)  

---


**ABSTRACT**  
In recent years, extensive research has focused on 3D natural scene generation, but the domain of 3D city generation has not received as much exploration. This is due to the greater challenges posed by 3D city generation, mainly because humans are more sensitive to structural distortions in urban environments. Additionally, generating 3D cities is more complex than 3D natural scenes since buildings, as objects of the same class, exhibit a wider range of appearances compared to the relatively consistent appearance of objects like trees in natural scenes. To address these challenges, we propose CityDreamer, a compositional generative model designed specifically for unbounded 3D cities, which separates the generation of building instances from other background objects, such as roads, green lands, and water areas, into distinct modules. Furthermore, we construct two datasets, OSM and GoogleEarth, containing a vast amount of real-world city imagery to enhance the realism of the generated 3D cities both in their layouts and appearances. Through extensive experiments, CityDreamer has proven its superiority over state-of-the-art methods in generating a wide range of lifelike 3D cities.

{{</citation>}}


### (25/71) Time Series Analysis of Urban Liveability (Alex Levering et al., 2023)

{{<citation>}}

Alex Levering, Diego Marcos, Devis Tuia. (2023)  
**Time Series Analysis of Urban Liveability**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2309.00594v1)  

---


**ABSTRACT**  
In this paper we explore deep learning models to monitor longitudinal liveability changes in Dutch cities at the neighbourhood level. Our liveability reference data is defined by a country-wise yearly survey based on a set of indicators combined into a liveability score, the Leefbaarometer. We pair this reference data with yearly-available high-resolution aerial images, which creates yearly timesteps at which liveability can be monitored. We deploy a convolutional neural network trained on an aerial image from 2016 and the Leefbaarometer score to predict liveability at new timesteps 2012 and 2020. The results in a city used for training (Amsterdam) and one never seen during training (Eindhoven) show some trends which are difficult to interpret, especially in light of the differences in image acquisitions at the different time steps. This demonstrates the complexity of liveability monitoring across time periods and the necessity for more sophisticated methods compensating for changes unrelated to liveability dynamics.

{{</citation>}}


### (26/71) SQLdepth: Generalizable Self-Supervised Fine-Structured Monocular Depth Estimation (Youhong Wang et al., 2023)

{{<citation>}}

Youhong Wang, Yunji Liang, Hao Xu, Shaohui Jiao, Hongkai Yu. (2023)  
**SQLdepth: Generalizable Self-Supervised Fine-Structured Monocular Depth Estimation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2309.00526v1)  

---


**ABSTRACT**  
Recently, self-supervised monocular depth estimation has gained popularity with numerous applications in autonomous driving and robotics. However, existing solutions primarily seek to estimate depth from immediate visual features, and struggle to recover fine-grained scene details with limited generalization. In this paper, we introduce SQLdepth, a novel approach that can effectively learn fine-grained scene structures from motion. In SQLdepth, we propose a novel Self Query Layer (SQL) to build a self-cost volume and infer depth from it, rather than inferring depth from feature maps. The self-cost volume implicitly captures the intrinsic geometry of the scene within a single frame. Each individual slice of the volume signifies the relative distances between points and objects within a latent space. Ultimately, this volume is compressed to the depth map via a novel decoding approach. Experimental results on KITTI and Cityscapes show that our method attains remarkable state-of-the-art performance (AbsRel = $0.082$ on KITTI, $0.052$ on KITTI with improved ground-truth and $0.106$ on Cityscapes), achieves $9.9\%$, $5.5\%$ and $4.5\%$ error reduction from the previous best. In addition, our approach showcases reduced training complexity, computational efficiency, improved generalization, and the ability to recover fine-grained scene details. Moreover, the self-supervised pre-trained and metric fine-tuned SQLdepth can surpass existing supervised methods by significant margins (AbsRel = $0.043$, $14\%$ error reduction). self-matching-oriented relative distance querying in SQL improves the robustness and zero-shot generalization capability of SQLdepth. Code and the pre-trained weights will be publicly available. Code is available at \href{https://github.com/hisfog/SQLdepth-Impl}{https://github.com/hisfog/SQLdepth-Impl}.

{{</citation>}}


### (27/71) A Theoretical and Practical Framework for Evaluating Uncertainty Calibration in Object Detection (Pedro Conde et al., 2023)

{{<citation>}}

Pedro Conde, Rui L. Lopes, Cristiano Premebida. (2023)  
**A Theoretical and Practical Framework for Evaluating Uncertainty Calibration in Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2309.00464v1)  

---


**ABSTRACT**  
The proliferation of Deep Neural Networks has resulted in machine learning systems becoming increasingly more present in various real-world applications. Consequently, there is a growing demand for highly reliable models in these domains, making the problem of uncertainty calibration pivotal, when considering the future of deep learning. This is especially true when considering object detection systems, that are commonly present in safety-critical application such as autonomous driving and robotics. For this reason, this work presents a novel theoretical and practical framework to evaluate object detection systems in the context of uncertainty calibration. The robustness of the proposed uncertainty calibration metrics is shown through a series of representative experiments. Code for the proposed uncertainty calibration metrics at: https://github.com/pedrormconde/Uncertainty_Calibration_Object_Detection.

{{</citation>}}


### (28/71) Zero-Shot Video Moment Retrieval from Frozen Vision-Language Models (Dezhao Luo et al., 2023)

{{<citation>}}

Dezhao Luo, Jiabo Huang, Shaogang Gong, Hailin Jin, Yang Liu. (2023)  
**Zero-Shot Video Moment Retrieval from Frozen Vision-Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2309.00661v1)  

---


**ABSTRACT**  
Accurate video moment retrieval (VMR) requires universal visual-textual correlations that can handle unknown vocabulary and unseen scenes. However, the learned correlations are likely either biased when derived from a limited amount of moment-text data which is hard to scale up because of the prohibitive annotation cost (fully-supervised), or unreliable when only the video-text pairwise relationships are available without fine-grained temporal annotations (weakly-supervised). Recently, the vision-language models (VLM) demonstrate a new transfer learning paradigm to benefit different vision tasks through the universal visual-textual correlations derived from large-scale vision-language pairwise web data, which has also shown benefits to VMR by fine-tuning in the target domains. In this work, we propose a zero-shot method for adapting generalisable visual-textual priors from arbitrary VLM to facilitate moment-text alignment, without the need for accessing the VMR data. To this end, we devise a conditional feature refinement module to generate boundary-aware visual features conditioned on text queries to enable better moment boundary understanding. Additionally, we design a bottom-up proposal generation strategy that mitigates the impact of domain discrepancies and breaks down complex-query retrieval tasks into individual action retrievals, thereby maximizing the benefits of VLM. Extensive experiments conducted on three VMR benchmark datasets demonstrate the notable performance advantages of our zero-shot algorithm, especially in the novel-word and novel-location out-of-distribution setups.

{{</citation>}}


### (29/71) Fine-grained Recognition with Learnable Semantic Data Augmentation (Yifan Pu et al., 2023)

{{<citation>}}

Yifan Pu, Yizeng Han, Yulin Wang, Junlan Feng, Chao Deng, Gao Huang. (2023)  
**Fine-grained Recognition with Learnable Semantic Data Augmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2309.00399v1)  

---


**ABSTRACT**  
Fine-grained image recognition is a longstanding computer vision challenge that focuses on differentiating objects belonging to multiple subordinate categories within the same meta-category. Since images belonging to the same meta-category usually share similar visual appearances, mining discriminative visual cues is the key to distinguishing fine-grained categories. Although commonly used image-level data augmentation techniques have achieved great success in generic image classification problems, they are rarely applied in fine-grained scenarios, because their random editing-region behavior is prone to destroy the discriminative visual cues residing in the subtle regions. In this paper, we propose diversifying the training data at the feature-level to alleviate the discriminative region loss problem. Specifically, we produce diversified augmented samples by translating image features along semantically meaningful directions. The semantic directions are estimated with a covariance prediction network, which predicts a sample-wise covariance matrix to adapt to the large intra-class variation inherent in fine-grained images. Furthermore, the covariance prediction network is jointly optimized with the classification network in a meta-learning manner to alleviate the degenerate solution problem. Experiments on four competitive fine-grained recognition benchmarks (CUB-200-2011, Stanford Cars, FGVC Aircrafts, NABirds) demonstrate that our method significantly improves the generalization performance on several popular classification networks (e.g., ResNets, DenseNets, EfficientNets, RegNets and ViT). Combined with a recently proposed method, our semantic data augmentation approach achieves state-of-the-art performance on the CUB-200-2011 dataset. The source code will be released.

{{</citation>}}


### (30/71) MuraNet: Multi-task Floor Plan Recognition with Relation Attention (Lingxiao Huang et al., 2023)

{{<citation>}}

Lingxiao Huang, Jung-Hsuan Wu, Chiching Wei, Wilson Li. (2023)  
**MuraNet: Multi-task Floor Plan Recognition with Relation Attention**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2309.00348v1)  

---


**ABSTRACT**  
The recognition of information in floor plan data requires the use of detection and segmentation models. However, relying on several single-task models can result in ineffective utilization of relevant information when there are multiple tasks present simultaneously. To address this challenge, we introduce MuraNet, an attention-based multi-task model for segmentation and detection tasks in floor plan data. In MuraNet, we adopt a unified encoder called MURA as the backbone with two separated branches: an enhanced segmentation decoder branch and a decoupled detection head branch based on YOLOX, for segmentation and detection tasks respectively. The architecture of MuraNet is designed to leverage the fact that walls, doors, and windows usually constitute the primary structure of a floor plan's architecture. By jointly training the model on both detection and segmentation tasks, we believe MuraNet can effectively extract and utilize relevant features for both tasks. Our experiments on the CubiCasa5k public dataset show that MuraNet improves convergence speed during training compared to single-task models like U-Net and YOLOv3. Moreover, we observe improvements in the average AP and IoU in detection and segmentation tasks, respectively.Our ablation experiments demonstrate that the attention-based unified backbone of MuraNet achieves better feature extraction in floor plan recognition tasks, and the use of decoupled multi-head branches for different tasks further improves model performance. We believe that our proposed MuraNet model can address the disadvantages of single-task models and improve the accuracy and efficiency of floor plan data recognition.

{{</citation>}}


### (31/71) Robust Point Cloud Processing through Positional Embedding (Jianqiao Zheng et al., 2023)

{{<citation>}}

Jianqiao Zheng, Xueqian Li, Sameera Ramasinghe, Simon Lucey. (2023)  
**Robust Point Cloud Processing through Positional Embedding**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2309.00339v1)  

---


**ABSTRACT**  
End-to-end trained per-point embeddings are an essential ingredient of any state-of-the-art 3D point cloud processing such as detection or alignment. Methods like PointNet, or the more recent point cloud transformer -- and its variants -- all employ learned per-point embeddings. Despite impressive performance, such approaches are sensitive to out-of-distribution (OOD) noise and outliers. In this paper, we explore the role of an analytical per-point embedding based on the criterion of bandwidth. The concept of bandwidth enables us to draw connections with an alternate per-point embedding -- positional embedding, particularly random Fourier features. We present compelling robust results across downstream tasks such as point cloud classification and registration with several categories of OOD noise.

{{</citation>}}


### (32/71) Human trajectory prediction using LSTM with Attention mechanism (Amin Manafi Soltan Ahmadi et al., 2023)

{{<citation>}}

Amin Manafi Soltan Ahmadi, Samaneh Hoseini Semnani. (2023)  
**Human trajectory prediction using LSTM with Attention mechanism**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV  
Keywords: Attention, LSTM  
[Paper Link](http://arxiv.org/abs/2309.00331v1)  

---


**ABSTRACT**  
In this paper, we propose a human trajectory prediction model that combines a Long Short-Term Memory (LSTM) network with an attention mechanism. To do that, we use attention scores to determine which parts of the input data the model should focus on when making predictions. Attention scores are calculated for each input feature, with a higher score indicating the greater significance of that feature in predicting the output. Initially, these scores are determined for the target human position, velocity, and their neighboring individual's positions and velocities. By using attention scores, our model can prioritize the most relevant information in the input data and make more accurate predictions. We extract attention scores from our attention mechanism and integrate them into the trajectory prediction module to predict human future trajectories. To achieve this, we introduce a new neural layer that processes attention scores after extracting them and concatenates them with positional information. We evaluate our approach on the publicly available ETH and UCY datasets and measure its performance using the final displacement error (FDE) and average displacement error (ADE) metrics. We show that our modified algorithm performs better than the Social LSTM in predicting the future trajectory of pedestrians in crowded spaces. Specifically, our model achieves an improvement of 6.2% in ADE and 6.3% in FDE compared to the Social LSTM results in the literature.

{{</citation>}}


### (33/71) Fine-Grained Spatiotemporal Motion Alignment for Contrastive Video Representation Learning (Minghao Zhu et al., 2023)

{{<citation>}}

Minghao Zhu, Xiao Lin, Ronghao Dang, Chengju Liu, Qijun Chen. (2023)  
**Fine-Grained Spatiotemporal Motion Alignment for Contrastive Video Representation Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2309.00297v1)  

---


**ABSTRACT**  
As the most essential property in a video, motion information is critical to a robust and generalized video representation. To inject motion dynamics, recent works have adopted frame difference as the source of motion information in video contrastive learning, considering the trade-off between quality and cost. However, existing works align motion features at the instance level, which suffers from spatial and temporal weak alignment across modalities. In this paper, we present a \textbf{Fi}ne-grained \textbf{M}otion \textbf{A}lignment (FIMA) framework, capable of introducing well-aligned and significant motion information. Specifically, we first develop a dense contrastive learning framework in the spatiotemporal domain to generate pixel-level motion supervision. Then, we design a motion decoder and a foreground sampling strategy to eliminate the weak alignments in terms of time and space. Moreover, a frame-level motion contrastive loss is presented to improve the temporal diversity of the motion features. Extensive experiments demonstrate that the representations learned by FIMA possess great motion-awareness capabilities and achieve state-of-the-art or competitive results on downstream tasks across UCF101, HMDB51, and Diving48 datasets. Code is available at \url{https://github.com/ZMHH-H/FIMA}.

{{</citation>}}


### (34/71) Interpretable Medical Imagery Diagnosis with Self-Attentive Transformers: A Review of Explainable AI for Health Care (Tin Lai, 2023)

{{<citation>}}

Tin Lai. (2023)  
**Interpretable Medical Imagery Diagnosis with Self-Attentive Transformers: A Review of Explainable AI for Health Care**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: AI, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.00252v1)  

---


**ABSTRACT**  
Recent advancements in artificial intelligence (AI) have facilitated its widespread adoption in primary medical services, addressing the demand-supply imbalance in healthcare. Vision Transformers (ViT) have emerged as state-of-the-art computer vision models, benefiting from self-attention modules. However, compared to traditional machine-learning approaches, deep-learning models are complex and are often treated as a "black box" that can cause uncertainty regarding how they operate. Explainable Artificial Intelligence (XAI) refers to methods that explain and interpret machine learning models' inner workings and how they come to decisions, which is especially important in the medical domain to guide the healthcare decision-making process. This review summarises recent ViT advancements and interpretative approaches to understanding the decision-making process of ViT, enabling transparency in medical diagnosis applications.

{{</citation>}}


### (35/71) Human-Inspired Facial Sketch Synthesis with Dynamic Adaptation (Fei Gao et al., 2023)

{{<citation>}}

Fei Gao, Yifan Zhu, Chang Jiang, Nannan Wang. (2023)  
**Human-Inspired Facial Sketch Synthesis with Dynamic Adaptation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV  
Keywords: Sketch  
[Paper Link](http://arxiv.org/abs/2309.00216v1)  

---


**ABSTRACT**  
Facial sketch synthesis (FSS) aims to generate a vivid sketch portrait from a given facial photo. Existing FSS methods merely rely on 2D representations of facial semantic or appearance. However, professional human artists usually use outlines or shadings to covey 3D geometry. Thus facial 3D geometry (e.g. depth map) is extremely important for FSS. Besides, different artists may use diverse drawing techniques and create multiple styles of sketches; but the style is globally consistent in a sketch. Inspired by such observations, in this paper, we propose a novel Human-Inspired Dynamic Adaptation (HIDA) method. Specially, we propose to dynamically modulate neuron activations based on a joint consideration of both facial 3D geometry and 2D appearance, as well as globally consistent style control. Besides, we use deformable convolutions at coarse-scales to align deep features, for generating abstract and distinct outlines. Experiments show that HIDA can generate high-quality sketches in multiple styles, and significantly outperforms previous methods, over a large range of challenging faces. Besides, HIDA allows precise style control of the synthesized sketch, and generalizes well to natural scenes and other artistic styles. Our code and results have been released online at: https://github.com/AiArt-HDU/HIDA.

{{</citation>}}


## cs.CL (14)



### (36/71) Let the Models Respond: Interpreting Language Model Detoxification Through the Lens of Prompt Dependence (Daniel Scalena et al., 2023)

{{<citation>}}

Daniel Scalena, Gabriele Sarti, Malvina Nissim, Elisabetta Fersini. (2023)  
**Let the Models Respond: Interpreting Language Model Detoxification Through the Lens of Prompt Dependence**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.00751v1)  

---


**ABSTRACT**  
Due to language models' propensity to generate toxic or hateful responses, several techniques were developed to align model generations with users' preferences. Despite the effectiveness of such methods in improving the safety of model interactions, their impact on models' internal processes is still poorly understood. In this work, we apply popular detoxification approaches to several language models and quantify their impact on the resulting models' prompt dependence using feature attribution methods. We evaluate the effectiveness of counter-narrative fine-tuning and compare it with reinforcement learning-driven detoxification, observing differences in prompt reliance between the two methods despite their similar detoxification performances.

{{</citation>}}


### (37/71) Contextual Biasing of Named-Entities with Large Language Models (Chuanneng Sun et al., 2023)

{{<citation>}}

Chuanneng Sun, Zeeshan Ahmed, Yingyi Ma, Zhe Liu, Yutong Pang, Ozlem Kalinli. (2023)  
**Contextual Biasing of Named-Entities with Large Language Models**  

---
Primary Category: cs.CL  
Categories: 68T10, I-2-7, cs-AI, cs-CL, cs-LG, cs-SD, cs.CL, eess-AS  
Keywords: Bias, Language Model, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2309.00723v1)  

---


**ABSTRACT**  
This paper studies contextual biasing with Large Language Models (LLMs), where during second-pass rescoring additional contextual information is provided to a LLM to boost Automatic Speech Recognition (ASR) performance. We propose to leverage prompts for a LLM without fine tuning during rescoring which incorporate a biasing list and few-shot examples to serve as additional information when calculating the score for the hypothesis. In addition to few-shot prompt learning, we propose multi-task training of the LLM to predict both the entity class and the next token. To improve the efficiency for contextual biasing and to avoid exceeding LLMs' maximum sequence lengths, we propose dynamic prompting, where we select the most likely class using the class tag prediction, and only use entities in this class as contexts for next token prediction. Word Error Rate (WER) evaluation is performed on i) an internal calling, messaging, and dictation dataset, and ii) the SLUE-Voxpopuli dataset. Results indicate that biasing lists and few-shot examples can achieve 17.8% and 9.6% relative improvement compared to first pass ASR, and that multi-task training and dynamic prompting can achieve 20.0% and 11.3% relative WER improvement, respectively.

{{</citation>}}


### (38/71) Taken out of context: On measuring situational awareness in LLMs (Lukas Berglund et al., 2023)

{{<citation>}}

Lukas Berglund, Asa Cooper Stickland, Mikita Balesni, Max Kaufmann, Meg Tong, Tomasz Korbak, Daniel Kokotajlo, Owain Evans. (2023)  
**Taken out of context: On measuring situational awareness in LLMs**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: GPT, LLaMA  
[Paper Link](http://arxiv.org/abs/2309.00667v1)  

---


**ABSTRACT**  
We aim to better understand the emergence of `situational awareness' in large language models (LLMs). A model is situationally aware if it's aware that it's a model and can recognize whether it's currently in testing or deployment. Today's LLMs are tested for safety and alignment before they are deployed. An LLM could exploit situational awareness to achieve a high score on safety tests, while taking harmful actions after deployment. Situational awareness may emerge unexpectedly as a byproduct of model scaling. One way to better foresee this emergence is to run scaling experiments on abilities necessary for situational awareness. As such an ability, we propose `out-of-context reasoning' (in contrast to in-context learning). We study out-of-context reasoning experimentally. First, we finetune an LLM on a description of a test while providing no examples or demonstrations. At test time, we assess whether the model can pass the test. To our surprise, we find that LLMs succeed on this out-of-context reasoning task. Their success is sensitive to the training setup and only works when we apply data augmentation. For both GPT-3 and LLaMA-1, performance improves with model size. These findings offer a foundation for further empirical study, towards predicting and potentially controlling the emergence of situational awareness in LLMs. Code is available at: https://github.com/AsaCooperStickland/situational-awareness-evals.

{{</citation>}}


### (39/71) BatchPrompt: Accomplish more with less (Jianzhe Lin et al., 2023)

{{<citation>}}

Jianzhe Lin, Maurice Diesendruck, Liang Du, Robin Abraham. (2023)  
**BatchPrompt: Accomplish more with less**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, NLP  
[Paper Link](http://arxiv.org/abs/2309.00384v1)  

---


**ABSTRACT**  
Many LLMs are trained to perform zero-shot or few-shot inference using instruction-based prompts. Crafting prompts for these LLMs typically requires the user to provide a detailed task description, examples of context and completion, and single example of context for inference. This regular prompt baseline is referred to as SinglePrompt in this paper. However, for NLP tasks where each data point for inference is not necessarily lengthy, the token count for instructions and few-shot examples in the prompt may be considerably larger than that of the data point, resulting in lower token-resource utilization compared with encoder-based models like fine-tuned BERT. This cost-efficiency issue, affecting inference speed and compute budget, counteracts the many benefits LLMs have to offer. This paper aims to alleviate the preceding problem by batching multiple data points into a single prompt, a prompting strategy we refer to as BatchPrompt. This strategy increases the density of data points, which in turn leads to improved token utilization. Applying BatchPrompt naively, however, is very challenging due to significant performance degradation, as observed in our experiments. We also noticed varying inference outcomes for the same data point appearing in different positions within a prompt. To address the quality issue while remain high token-resource utilization, we introduce Batch Permutation and Ensembling for BatchPrompt, a simple way that recovers labeling quality through majority votes from data points placed in varying positions in a batch at the price of more token usage. To counterbalance the additional token usage caused by the voting process, we further propose Self-reflection-guided EArly Stopping, which can terminate the voting process early for data points the LLM confidently handles.

{{</citation>}}


### (40/71) When Do Discourse Markers Affect Computational Sentence Understanding? (Ruiqi Li et al., 2023)

{{<citation>}}

Ruiqi Li, Liesbeth Allein, Damien Sileo, Marie-Francine Moens. (2023)  
**When Do Discourse Markers Affect Computational Sentence Understanding?**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2309.00368v1)  

---


**ABSTRACT**  
The capabilities and use cases of automatic natural language processing (NLP) have grown significantly over the last few years. While much work has been devoted to understanding how humans deal with discourse connectives, this phenomenon is understudied in computational systems. Therefore, it is important to put NLP models under the microscope and examine whether they can adequately comprehend, process, and reason within the complexity of natural language. In this chapter, we introduce the main mechanisms behind automatic sentence processing systems step by step and then focus on evaluating discourse connective processing. We assess nine popular systems in their ability to understand English discourse connectives and analyze how context and language understanding tasks affect their connective comprehension. The results show that NLP systems do not process all discourse connectives equally well and that the computational processing complexity of different connective kinds is not always consistently in line with the presumed complexity order found in human processing. In addition, while humans are more inclined to be influenced during the reading procedure but not necessarily in the final comprehension performance, discourse connectives have a significant impact on the final accuracy of NLP systems. The richer knowledge of connectives a system learns, the more negative effect inappropriate connectives have on it. This suggests that the correct explicitation of discourse connectives is important for computational natural language processing.

{{</citation>}}


### (41/71) Large Content And Behavior Models To Understand, Simulate, And Optimize Content And Behavior (Ashmit Khandelwal et al., 2023)

{{<citation>}}

Ashmit Khandelwal, Aditya Agrawal, Aanisha Bhattacharyya, Yaman K Singla, Somesh Singh, Uttaran Bhattacharya, Ishita Dasgupta, Stefano Petrangeli, Rajiv Ratn Shah, Changyou Chen, Balaji Krishnamurthy. (2023)  
**Large Content And Behavior Models To Understand, Simulate, And Optimize Content And Behavior**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CV, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.00359v1)  

---


**ABSTRACT**  
Shannon, in his seminal paper introducing information theory, divided the communication into three levels: technical, semantic, and effectivenss. While the technical level is concerned with accurate reconstruction of transmitted symbols, the semantic and effectiveness levels deal with the inferred meaning and its effect on the receiver. Thanks to telecommunications, the first level problem has produced great advances like the internet. Large Language Models (LLMs) make some progress towards the second goal, but the third level still remains largely untouched. The third problem deals with predicting and optimizing communication for desired receiver behavior. LLMs, while showing wide generalization capabilities across a wide range of tasks, are unable to solve for this. One reason for the underperformance could be a lack of "behavior tokens" in LLMs' training corpora. Behavior tokens define receiver behavior over a communication, such as shares, likes, clicks, purchases, retweets, etc. While preprocessing data for LLM training, behavior tokens are often removed from the corpora as noise. Therefore, in this paper, we make some initial progress towards reintroducing behavior tokens in LLM training. The trained models, other than showing similar performance to LLMs on content understanding tasks, show generalization capabilities on behavior simulation, content simulation, behavior understanding, and behavior domain adaptation. Using a wide range of tasks on two corpora, we show results on all these capabilities. We call these models Large Content and Behavior Models (LCBMs). Further, to spur more research on LCBMs, we release our new Content Behavior Corpus (CBC), a repository containing communicator, message, and corresponding receiver behavior.

{{</citation>}}


### (42/71) Comparative Topic Modeling for Determinants of Divergent Report Results Applied to Macular Degeneration Studies (Lucas Cassiel Jacaruso, 2023)

{{<citation>}}

Lucas Cassiel Jacaruso. (2023)  
**Comparative Topic Modeling for Determinants of Divergent Report Results Applied to Macular Degeneration Studies**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP, Natural Language Processing, Topic Model, Topic Modeling  
[Paper Link](http://arxiv.org/abs/2309.00312v1)  

---


**ABSTRACT**  
Topic modeling and text mining are subsets of Natural Language Processing with relevance for conducting meta-analysis (MA) and systematic review (SR). For evidence synthesis, the above NLP methods are conventionally used for topic-specific literature searches or extracting values from reports to automate essential phases of SR and MA. Instead, this work proposes a comparative topic modeling approach to analyze reports of contradictory results on the same general research question. Specifically, the objective is to find topics exhibiting distinct associations with significant results for an outcome of interest by ranking them according to their proportional occurrence and consistency of distribution across reports of significant results. The proposed method was tested on broad-scope studies addressing whether supplemental nutritional compounds significantly benefit macular degeneration (MD). Eight compounds were identified as having a particular association with reports of significant results for benefitting MD. Six of these were further supported in terms of effectiveness upon conducting a follow-up literature search for validation (omega-3 fatty acids, copper, zeaxanthin, lutein, zinc, and nitrates). The two not supported by the follow-up literature search (niacin and molybdenum) also had the lowest scores under the proposed methods ranking system, suggesting that the proposed method's score for a given topic is a viable proxy for its degree of association with the outcome of interest. These results underpin the proposed methods potential to add specificity in understanding effects from broad-scope reports, elucidate topics of interest for future research, and guide evidence synthesis in a systematic and scalable way.

{{</citation>}}


### (43/71) RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback (Harrison Lee et al., 2023)

{{<citation>}}

Harrison Lee, Samrat Phatale, Hassan Mansoor, Kellie Lu, Thomas Mesnard, Colton Bishop, Victor Carbune, Abhinav Rastogi. (2023)  
**RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.00267v1)  

---


**ABSTRACT**  
Reinforcement learning from human feedback (RLHF) is effective at aligning large language models (LLMs) to human preferences, but gathering high quality human preference labels is a key bottleneck. We conduct a head-to-head comparison of RLHF vs. RL from AI Feedback (RLAIF) - a technique where preferences are labeled by an off-the-shelf LLM in lieu of humans, and we find that they result in similar improvements. On the task of summarization, human evaluators prefer generations from both RLAIF and RLHF over a baseline supervised fine-tuned model in ~70% of cases. Furthermore, when asked to rate RLAIF vs. RLHF summaries, humans prefer both at equal rates. These results suggest that RLAIF can yield human-level performance, offering a potential solution to the scalability limitations of RLHF.

{{</citation>}}


### (44/71) Detecting Suicidality in Arabic Tweets Using Machine Learning and Deep Learning Techniques (Asma Abdulsalam et al., 2023)

{{<citation>}}

Asma Abdulsalam, Areej Alhothali, Saleh Al-Ghamdi. (2023)  
**Detecting Suicidality in Arabic Tweets Using Machine Learning and Deep Learning Techniques**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, Twitter  
[Paper Link](http://arxiv.org/abs/2309.00246v1)  

---


**ABSTRACT**  
Social media platforms have revolutionized traditional communication techniques by enabling people globally to connect instantaneously, openly, and frequently. People use social media to share personal stories and express their opinion. Negative emotions such as thoughts of death, self-harm, and hardship are commonly expressed on social media, particularly among younger generations. As a result, using social media to detect suicidal thoughts will help provide proper intervention that will ultimately deter others from self-harm and committing suicide and stop the spread of suicidal ideation on social media. To investigate the ability to detect suicidal thoughts in Arabic tweets automatically, we developed a novel Arabic suicidal tweets dataset, examined several machine learning models, including Na\"ive Bayes, Support Vector Machine, K-Nearest Neighbor, Random Forest, and XGBoost, trained on word frequency and word embedding features, and investigated the ability of pre-trained deep learning models, AraBert, AraELECTRA, and AraGPT2, to identify suicidal thoughts in Arabic tweets. The results indicate that SVM and RF models trained on character n-gram features provided the best performance in the machine learning models, with 86% accuracy and an F1 score of 79%. The results of the deep learning models show that AraBert model outperforms other machine and deep learning models, achieving an accuracy of 91\% and an F1-score of 88%, which significantly improves the detection of suicidal ideation in the Arabic tweets dataset. To the best of our knowledge, this is the first study to develop an Arabic suicidality detection dataset from Twitter and to use deep-learning approaches in detecting suicidality in Arabic posts.

{{</citation>}}


### (45/71) FactLLaMA: Optimizing Instruction-Following Language Models with External Knowledge for Automated Fact-Checking (Tsun-Hin Cheung et al., 2023)

{{<citation>}}

Tsun-Hin Cheung, Kin-Man Lam. (2023)  
**FactLLaMA: Optimizing Instruction-Following Language Models with External Knowledge for Automated Fact-Checking**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Fact-Checking, GPT, LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2309.00240v1)  

---


**ABSTRACT**  
Automatic fact-checking plays a crucial role in combating the spread of misinformation. Large Language Models (LLMs) and Instruction-Following variants, such as InstructGPT and Alpaca, have shown remarkable performance in various natural language processing tasks. However, their knowledge may not always be up-to-date or sufficient, potentially leading to inaccuracies in fact-checking. To address this limitation, we propose combining the power of instruction-following language models with external evidence retrieval to enhance fact-checking performance. Our approach involves leveraging search engines to retrieve relevant evidence for a given input claim. This external evidence serves as valuable supplementary information to augment the knowledge of the pretrained language model. Then, we instruct-tune an open-sourced language model, called LLaMA, using this evidence, enabling it to predict the veracity of the input claim more accurately. To evaluate our method, we conducted experiments on two widely used fact-checking datasets: RAWFC and LIAR. The results demonstrate that our approach achieves state-of-the-art performance in fact-checking tasks. By integrating external evidence, we bridge the gap between the model's knowledge and the most up-to-date and sufficient context available, leading to improved fact-checking outcomes. Our findings have implications for combating misinformation and promoting the dissemination of accurate information on online platforms. Our released materials are accessible at: https://thcheung.github.io/factllama.

{{</citation>}}


### (46/71) Publicly Shareable Clinical Large Language Model Built on Synthetic Clinical Notes (Sunjun Kweon et al., 2023)

{{<citation>}}

Sunjun Kweon, Junu Kim, Jiyoun Kim, Sujeong Im, Eunbyeol Cho, Seongsu Bae, Jungwoo Oh, Gyubok Lee, Jong Hak Moon, Seng Chan You, Seungjin Baek, Chang Hoon Han, Yoon Bin Jung, Yohan Jo, Edward Choi. (2023)  
**Publicly Shareable Clinical Large Language Model Built on Synthetic Clinical Notes**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Clinical, GPT, GPT-3.5, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2309.00237v1)  

---


**ABSTRACT**  
The development of large language models tailored for handling patients' clinical notes is often hindered by the limited accessibility and usability of these notes due to strict privacy regulations. To address these challenges, we first create synthetic large-scale clinical notes using publicly available case reports extracted from biomedical literature. We then use these synthetic notes to train our specialized clinical large language model, Asclepius. While Asclepius is trained on synthetic data, we assess its potential performance in real-world applications by evaluating it using real clinical notes. We benchmark Asclepius against several other large language models, including GPT-3.5-turbo and other open-source alternatives. To further validate our approach using synthetic notes, we also compare Asclepius with its variants trained on real clinical notes. Our findings convincingly demonstrate that synthetic clinical notes can serve as viable substitutes for real ones when constructing high-performing clinical language models. This conclusion is supported by detailed evaluations conducted by both GPT-4 and medical professionals. All resources including weights, codes, and data used in the development of Asclepius are made publicly accessible for future research.

{{</citation>}}


### (47/71) JoTR: A Joint Transformer and Reinforcement Learning Framework for Dialog Policy Learning (Wai-Chung Kwan et al., 2023)

{{<citation>}}

Wai-Chung Kwan, Huimin Wang, Hongru Wang, Zezhong Wang, Xian Wu, Yefeng Zheng, Kam-Fai Wong. (2023)  
**JoTR: A Joint Transformer and Reinforcement Learning Framework for Dialog Policy Learning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue, Reinforcement Learning, Transformer  
[Paper Link](http://arxiv.org/abs/2309.00230v1)  

---


**ABSTRACT**  
Dialogue policy learning (DPL) is a crucial component of dialogue modelling. Its primary role is to determine the appropriate abstract response, commonly referred to as the "dialogue action". Traditional DPL methodologies have treated this as a sequential decision problem, using pre-defined action candidates extracted from a corpus. However, these incomplete candidates can significantly limit the diversity of responses and pose challenges when dealing with edge cases, which are scenarios that occur only at extreme operating parameters. To address these limitations, we introduce a novel framework, JoTR. This framework is unique as it leverages a text-to-text Transformer-based model to generate flexible dialogue actions. Unlike traditional methods, JoTR formulates a word-level policy that allows for a more dynamic and adaptable dialogue action generation, without the need for any action templates. This setting enhances the diversity of responses and improves the system's ability to handle edge cases effectively. In addition, JoTR employs reinforcement learning with a reward-shaping mechanism to efficiently finetune the word-level dialogue policy, which allows the model to learn from its interactions, improving its performance over time. We conducted an extensive evaluation of JoTR to assess its effectiveness. Our extensive evaluation shows that JoTR achieves state-of-the-art performance on two benchmark dialogue modelling tasks, as assessed by both user simulators and human evaluators.

{{</citation>}}


### (48/71) Large Language Models for Semantic Monitoring of Corporate Disclosures: A Case Study on Korea's Top 50 KOSPI Companies (Junwon Sung et al., 2023)

{{<citation>}}

Junwon Sung, Woojin Heo, Yunkyung Byun, Youngsam Kim. (2023)  
**Large Language Models for Semantic Monitoring of Corporate Disclosures: A Case Study on Korea's Top 50 KOSPI Companies**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, GPT, GPT-3.5, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2309.00208v1)  

---


**ABSTRACT**  
In the rapidly advancing domain of artificial intelligence, state-of-the-art language models such as OpenAI's GPT-3.5-turbo and GPT-4 offer unprecedented opportunities for automating complex tasks. This research paper delves into the capabilities of these models for semantically analyzing corporate disclosures in the Korean context, specifically for timely disclosure. The study focuses on the top 50 publicly traded companies listed on the Korean KOSPI, based on market capitalization, and scrutinizes their monthly disclosure summaries over a period of 17 months. Each summary was assigned a sentiment rating on a scale ranging from 1(very negative) to 5(very positive). To gauge the effectiveness of the language models, their sentiment ratings were compared with those generated by human experts. Our findings reveal a notable performance disparity between GPT-3.5-turbo and GPT-4, with the latter demonstrating significant accuracy in human evaluation tests. The Spearman correlation coefficient was registered at 0.61, while the simple concordance rate was recorded at 0.82. This research contributes valuable insights into the evaluative characteristics of GPT models, thereby laying the groundwork for future innovations in the field of automated semantic monitoring.

{{</citation>}}


### (49/71) Will Sentiment Analysis Need Subculture? A New Data Augmentation Approach (Zhenhua Wang et al., 2023)

{{<citation>}}

Zhenhua Wang, Simin He, Guang Xu, Ming Ren. (2023)  
**Will Sentiment Analysis Need Subculture? A New Data Augmentation Approach**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Augmentation, Sentiment Analysis  
[Paper Link](http://arxiv.org/abs/2309.00178v1)  

---


**ABSTRACT**  
The renowned proverb that "The pen is mightier than the sword" underscores the formidable influence wielded by text expressions in shaping sentiments. Indeed, well-crafted written can deeply resonate within cultures, conveying profound sentiments. Nowadays, the omnipresence of the Internet has fostered a subculture that congregates around the contemporary milieu. The subculture artfully articulates the intricacies of human feelings by ardently pursuing the allure of novelty, a fact that cannot be disregarded in the sentiment analysis. This paper strives to enrich data through the lens of subculture, to address the insufficient training data faced by sentiment analysis. To this end, a new approach of subculture-based data augmentation (SCDA) is proposed, which engenders six enhanced texts for each training text by leveraging the creation of six diverse subculture expression generators. The extensive experiments attest to the effectiveness and potential of SCDA. The results also shed light on the phenomenon that disparate subculture expressions elicit varying degrees of sentiment stimulation. Moreover, an intriguing conjecture arises, suggesting the linear reversibility of certain subculture expressions. It is our fervent aspiration that this study serves as a catalyst in fostering heightened perceptiveness towards the tapestry of information, sentiment and culture, thereby enriching our collective understanding.

{{</citation>}}


## cs.AI (4)



### (50/71) Reinforcement Learning with Human Feedback for Realistic Traffic Simulation (Yulong Cao et al., 2023)

{{<citation>}}

Yulong Cao, Boris Ivanovic, Chaowei Xiao, Marco Pavone. (2023)  
**Reinforcement Learning with Human Feedback for Realistic Traffic Simulation**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs-RO, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.00709v1)  

---


**ABSTRACT**  
In light of the challenges and costs of real-world testing, autonomous vehicle developers often rely on testing in simulation for the creation of reliable systems. A key element of effective simulation is the incorporation of realistic traffic models that align with human knowledge, an aspect that has proven challenging due to the need to balance realism and diversity. This works aims to address this by developing a framework that employs reinforcement learning with human preference (RLHF) to enhance the realism of existing traffic models. This study also identifies two main challenges: capturing the nuances of human preferences on realism and the unification of diverse traffic simulation models. To tackle these issues, we propose using human feedback for alignment and employ RLHF due to its sample efficiency. We also introduce the first dataset for realism alignment in traffic modeling to support such research. Our framework, named TrafficRLHF, demonstrates its proficiency in generating realistic traffic scenarios that are well-aligned with human preferences, as corroborated by comprehensive evaluations on the nuScenes dataset.

{{</citation>}}


### (51/71) Declarative Reasoning on Explanations Using Constraint Logic Programming (Laura State et al., 2023)

{{<citation>}}

Laura State, Salvatore Ruggieri, Franco Turini. (2023)  
**Declarative Reasoning on Explanations Using Constraint Logic Programming**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CY, cs-LG, cs-SC, cs.AI  
Keywords: AI, Reasoning  
[Paper Link](http://arxiv.org/abs/2309.00422v1)  

---


**ABSTRACT**  
Explaining opaque Machine Learning (ML) models is an increasingly relevant problem. Current explanation in AI (XAI) methods suffer several shortcomings, among others an insufficient incorporation of background knowledge, and a lack of abstraction and interactivity with the user. We propose REASONX, an explanation method based on Constraint Logic Programming (CLP). REASONX can provide declarative, interactive explanations for decision trees, which can be the ML models under analysis or global/local surrogate models of any black-box model. Users can express background or common sense knowledge using linear constraints and MILP optimization over features of factual and contrastive instances, and interact with the answer constraints at different levels of abstraction through constraint projection. We present here the architecture of REASONX, which consists of a Python layer, closer to the user, and a CLP layer. REASONX's core execution engine is a Prolog meta-program with declarative semantics in terms of logic theories.

{{</citation>}}


### (52/71) On the Aggregation of Rules for Knowledge Graph Completion (Patrick Betz et al., 2023)

{{<citation>}}

Patrick Betz, Stefan Lüdtke, Christian Meilicke, Heiner Stuckenschmidt. (2023)  
**On the Aggregation of Rules for Knowledge Graph Completion**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2309.00306v1)  

---


**ABSTRACT**  
Rule learning approaches for knowledge graph completion are efficient, interpretable and competitive to purely neural models. The rule aggregation problem is concerned with finding one plausibility score for a candidate fact which was simultaneously predicted by multiple rules. Although the problem is ubiquitous, as data-driven rule learning can result in noisy and large rulesets, it is underrepresented in the literature and its theoretical foundations have not been studied before in this context. In this work, we demonstrate that existing aggregation approaches can be expressed as marginal inference operations over the predicting rules. In particular, we show that the common Max-aggregation strategy, which scores candidates based on the rule with the highest confidence, has a probabilistic interpretation. Finally, we propose an efficient and overlooked baseline which combines the previous strategies and is competitive to computationally more expensive approaches.

{{</citation>}}


### (53/71) ALJP: An Arabic Legal Judgment Prediction in Personal Status Cases Using Machine Learning Models (Salwa Abbara et al., 2023)

{{<citation>}}

Salwa Abbara, Mona Hafez, Aya Kazzaz, Areej Alhothali, Alhanouf Alsolami. (2023)  
**ALJP: An Arabic Legal Judgment Prediction in Personal Status Cases Using Machine Learning Models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: LSTM, Legal, NLP  
[Paper Link](http://arxiv.org/abs/2309.00238v1)  

---


**ABSTRACT**  
Legal Judgment Prediction (LJP) aims to predict judgment outcomes based on case description. Several researchers have developed techniques to assist potential clients by predicting the outcome in the legal profession. However, none of the proposed techniques were implemented in Arabic, and only a few attempts were implemented in English, Chinese, and Hindi. In this paper, we develop a system that utilizes deep learning (DL) and natural language processing (NLP) techniques to predict the judgment outcome from Arabic case scripts, especially in cases of custody and annulment of marriage. This system will assist judges and attorneys in improving their work and time efficiency while reducing sentencing disparity. In addition, it will help litigants, lawyers, and law students analyze the probable outcomes of any given case before trial. We use a different machine and deep learning models such as Support Vector Machine (SVM), Logistic regression (LR), Long Short Term Memory (LSTM), and Bidirectional Long Short-Term Memory (BiLSTM) using representation techniques such as TF-IDF and word2vec on the developed dataset. Experimental results demonstrate that compared with the five baseline methods, the SVM model with word2vec and LR with TF-IDF achieve the highest accuracy of 88% and 78% in predicting the judgment on custody cases and annulment of marriage, respectively. Furthermore, the LR and SVM with word2vec and BiLSTM model with TF-IDF achieved the highest accuracy of 88% and 69% in predicting the probability of outcomes on custody cases and annulment of marriage, respectively.

{{</citation>}}


## cs.SI (2)



### (54/71) The Power of Patents: Leveraging Text Mining and Social Network Analysis to Forecast IoT Trends (Mehrdad Maghsoudi et al., 2023)

{{<citation>}}

Mehrdad Maghsoudi, Reza Nourbakhsh, Mehrdad Agha Mohammadali Kermani, Rahim Khanizad. (2023)  
**The Power of Patents: Leveraging Text Mining and Social Network Analysis to Forecast IoT Trends**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Security, Social Network  
[Paper Link](http://arxiv.org/abs/2309.00707v1)  

---


**ABSTRACT**  
Technology has become an indispensable competitive tool as science and technology have progressed throughout history. Organizations can compete on an equal footing by implementing technology appropriately. Technology trends or technology lifecycles begin during the initiation phase. Finally, it reaches saturation after entering the maturity phase. As technology reaches saturation, it will be removed or replaced by another. This makes investing in technologies during this phase unjustifiable. Technology forecasting is a critical tool for research and development to determine the future direction of technology. Based on registered patents, this study examined the trends of IOT technologies. A total of 3697 patents related to the Internet of Things from the last six years of patenting have been gathered using lens.org for this purpose. The main people and companies were identified through the creation of the IOT patent registration cooperation network, and the main groups active in patent registration were identified by the community detection technique. The patents were then divided into six technology categories: Safety and Security, Information Services, Public Safety and Environment Monitoring, Collaborative Aware Systems, Smart Homes/Buildings, and Smart Grid. And their technical maturity was identified and examined using the Sigma Plot program. Based on the findings, information services technologies are in the saturation stage, while both smart homes/buildings, and smart grid technologies are in the saturation stage. Three technologies, Safety and Security, Public Safety and Environment Monitoring, and Collaborative Aware Systems are in the maturity stage.

{{</citation>}}


### (55/71) A normative approach to radicalization in social networks (Vincent Bouttier et al., 2023)

{{<citation>}}

Vincent Bouttier, Salomé Leclercq, Renaud Jardri, Sophie Deneve. (2023)  
**A normative approach to radicalization in social networks**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2309.00513v1)  

---


**ABSTRACT**  
In recent decades, the massification of online social connections has made information globally accessible in a matter of seconds. Unfortunately, this has been accompanied by a dramatic surge in extreme opinions, without a clear solution in sight. Using a model performing probabilistic inference in large-scale loopy graphs through exchange of messages between nodes, we show how circularity in the social graph directly leads to radicalization and the polarization of opinions. We demonstrate that these detrimental effects could be avoided by actively decorrelating the messages in social media feeds. This approach is based on an extension of Belief Propagation (BP) named Circular Belief Propagation (CBP) that can be trained to drastically improve inference within a cyclic graph. CBP was benchmarked using data from Facebook and Twitter. This approach could inspire new methods for preventing the viral spreading and amplification of misinformation online, improving the capacity of social networks to share knowledge globally without resorting to censorship.

{{</citation>}}


## cs.DC (1)



### (56/71) Randomized Polar Codes for Anytime Distributed Machine Learning (Burak Bartan et al., 2023)

{{<citation>}}

Burak Bartan, Mert Pilanci. (2023)  
**Randomized Polar Codes for Anytime Distributed Machine Learning**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs-IT, cs-LG, cs.DC, math-IT  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2309.00682v1)  

---


**ABSTRACT**  
We present a novel distributed computing framework that is robust to slow compute nodes, and is capable of both approximate and exact computation of linear operations. The proposed mechanism integrates the concepts of randomized sketching and polar codes in the context of coded computation. We propose a sequential decoding algorithm designed to handle real valued data while maintaining low computational complexity for recovery. Additionally, we provide an anytime estimator that can generate provably accurate estimates even when the set of available node outputs is not decodable. We demonstrate the potential applications of this framework in various contexts, such as large-scale matrix multiplication and black-box optimization. We present the implementation of these methods on a serverless cloud computing system and provide numerical results to demonstrate their scalability in practice, including ImageNet scale computations.

{{</citation>}}


## cs.SE (1)



### (57/71) Copiloting the Copilots: Fusing Large Language Models with Completion Engines for Automated Program Repair (Yuxiang Wei et al., 2023)

{{<citation>}}

Yuxiang Wei, Chunqiu Steven Xia, Lingming Zhang. (2023)  
**Copiloting the Copilots: Fusing Large Language Models with Completion Engines for Automated Program Repair**  

---
Primary Category: cs.SE  
Categories: cs-LG, cs-PL, cs-SE, cs.SE  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2309.00608v1)  

---


**ABSTRACT**  
During Automated Program Repair (APR), it can be challenging to synthesize correct patches for real-world systems in general-purpose programming languages. Recent Large Language Models (LLMs) have been shown to be helpful "copilots" in assisting developers with various coding tasks, and have also been directly applied for patch synthesis. However, most LLMs treat programs as sequences of tokens, meaning that they are ignorant of the underlying semantics constraints of the target programming language. This results in plenty of statically invalid generated patches, impeding the practicality of the technique. Therefore, we propose Repilot, a framework to further copilot the AI "copilots" (i.e., LLMs) by synthesizing more valid patches during the repair process. Our key insight is that many LLMs produce outputs autoregressively (i.e., token by token), resembling human writing programs, which can be significantly boosted and guided through a Completion Engine. Repilot synergistically synthesizes a candidate patch through the interaction between an LLM and a Completion Engine, which 1) prunes away infeasible tokens suggested by the LLM and 2) proactively completes the token based on the suggestions provided by the Completion Engine. Our evaluation on a subset of the widely-used Defects4j 1.2 and 2.0 datasets shows that Repilot fixes 66 and 50 bugs, respectively, surpassing the best-performing baseline by 14 and 16 bugs fixed. More importantly, Repilot is capable of producing more valid and correct patches than the base LLM when given the same generation budget.

{{</citation>}}


## stat.ML (1)



### (58/71) Mechanism of feature learning in convolutional neural networks (Daniel Beaglehole et al., 2023)

{{<citation>}}

Daniel Beaglehole, Adityanarayanan Radhakrishnan, Parthe Pandit, Mikhail Belkin. (2023)  
**Mechanism of feature learning in convolutional neural networks**  

---
Primary Category: stat.ML  
Categories: cs-CV, cs-LG, stat-ML, stat.ML  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2309.00570v1)  

---


**ABSTRACT**  
Understanding the mechanism of how convolutional neural networks learn features from image data is a fundamental problem in machine learning and computer vision. In this work, we identify such a mechanism. We posit the Convolutional Neural Feature Ansatz, which states that covariances of filters in any convolutional layer are proportional to the average gradient outer product (AGOP) taken with respect to patches of the input to that layer. We present extensive empirical evidence for our ansatz, including identifying high correlation between covariances of filters and patch-based AGOPs for convolutional layers in standard neural architectures, such as AlexNet, VGG, and ResNets pre-trained on ImageNet. We also provide supporting theoretical evidence. We then demonstrate the generality of our result by using the patch-based AGOP to enable deep feature learning in convolutional kernel machines. We refer to the resulting algorithm as (Deep) ConvRFM and show that our algorithm recovers similar features to deep convolutional networks including the notable emergence of edge detectors. Moreover, we find that Deep ConvRFM overcomes previously identified limitations of convolutional kernels, such as their inability to adapt to local signals in images and, as a result, leads to sizable performance improvement over fixed convolutional kernels.

{{</citation>}}


## cs.CE (1)



### (59/71) Catalyst Property Prediction with CatBERTa: Unveiling Feature Exploration Strategies through Large Language Models (Janghoon Ock et al., 2023)

{{<citation>}}

Janghoon Ock, Chakradhar Guntuboina, Amir Barati Farimani. (2023)  
**Catalyst Property Prediction with CatBERTa: Unveiling Feature Exploration Strategies through Large Language Models**  

---
Primary Category: cs.CE  
Categories: cs-CE, cs.CE, physics-chem-ph  
Keywords: Attention, BERT, GNN, Graph Neural Network, Graph Neural Networks, Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2309.00563v1)  

---


**ABSTRACT**  
Efficient catalyst screening necessitates predictive models for adsorption energy, a key property of reactivity. However, prevailing methods, notably graph neural networks (GNNs), demand precise atomic coordinates for constructing graph representations, while integrating observable attributes remains challenging. This research introduces CatBERTa, an energy prediction Transformer model using textual inputs. Built on a pretrained Transformer encoder, CatBERTa processes human-interpretable text, incorporating target features. Attention score analysis reveals CatBERTa's focus on tokens related to adsorbates, bulk composition, and their interacting atoms. Moreover, interacting atoms emerge as effective descriptors for adsorption configurations, while factors such as bond length and atomic properties of these atoms offer limited predictive contributions. By predicting adsorption energy from the textual representation of initial structures, CatBERTa achieves a mean absolute error (MAE) of 0.75 eV-comparable to vanilla Graph Neural Networks (GNNs). Furthermore, the subtraction of the CatBERTa-predicted energies effectively cancels out their systematic errors by as much as 19.3% for chemically similar systems, surpassing the error reduction observed in GNNs. This outcome highlights its potential to enhance the accuracy of energy difference predictions. This research establishes a fundamental framework for text-based catalyst property prediction, without relying on graph representations, while also unveiling intricate feature-property relationships.

{{</citation>}}


## cs.IR (2)



### (60/71) NeMig -- A Bilingual News Collection and Knowledge Graph about Migration (Andreea Iana et al., 2023)

{{<citation>}}

Andreea Iana, Mehwish Alam, Alexander Grote, Nevena Nikolajevic, Katharina Ludwig, Philipp Müller, Christof Weinhardt, Heiko Paulheim. (2023)  
**NeMig -- A Bilingual News Collection and Knowledge Graph about Migration**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2309.00550v1)  

---


**ABSTRACT**  
News recommendation plays a critical role in shaping the public's worldviews through the way in which it filters and disseminates information about different topics. Given the crucial impact that media plays in opinion formation, especially for sensitive topics, understanding the effects of personalized recommendation beyond accuracy has become essential in today's digital society. In this work, we present NeMig, a bilingual news collection on the topic of migration, and corresponding rich user data. In comparison to existing news recommendation datasets, which comprise a large variety of monolingual news, NeMig covers articles on a single controversial topic, published in both Germany and the US. We annotate the sentiment polarization of the articles and the political leanings of the media outlets, in addition to extracting subtopics and named entities disambiguated through Wikidata. These features can be used to analyze the effects of algorithmic news curation beyond accuracy-based performance, such as recommender biases and the creation of filter bubbles. We construct domain-specific knowledge graphs from the news text and metadata, thus encoding knowledge-level connections between articles. Importantly, while existing datasets include only click behavior, we collect user socio-demographic and political information in addition to explicit click feedback. We demonstrate the utility of NeMig through experiments on the tasks of news recommenders benchmarking, analysis of biases in recommenders, and news trends analysis. NeMig aims to provide a useful resource for the news recommendation community and to foster interdisciplinary research into the multidimensional effects of algorithmic news curation.

{{</citation>}}


### (61/71) Towards Contrastive Learning in Music Video Domain (Karel Veldkamp et al., 2023)

{{<citation>}}

Karel Veldkamp, Mariya Hendriksen, Zoltán Szlávik, Alexander Keijser. (2023)  
**Towards Contrastive Learning in Music Video Domain**  

---
Primary Category: cs.IR  
Categories: cs-CV, cs-IR, cs-MM, cs-SD, cs.IR, eess-AS  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2309.00347v1)  

---


**ABSTRACT**  
Contrastive learning is a powerful way of learning multimodal representations across various domains such as image-caption retrieval and audio-visual representation learning. In this work, we investigate if these findings generalize to the domain of music videos. Specifically, we create a dual en-coder for the audio and video modalities and train it using a bidirectional contrastive loss. For the experiments, we use an industry dataset containing 550 000 music videos as well as the public Million Song Dataset, and evaluate the quality of learned representations on the downstream tasks of music tagging and genre classification. Our results indicate that pre-trained networks without contrastive fine-tuning outperform our contrastive learning approach when evaluated on both tasks. To gain a better understanding of the reasons contrastive learning was not successful for music videos, we perform a qualitative analysis of the learned representations, revealing why contrastive learning might have difficulties uniting embeddings from two modalities. Based on these findings, we outline possible directions for future work. To facilitate the reproducibility of our results, we share our code and the pre-trained model.

{{</citation>}}


## cs.CY (1)



### (62/71) Rural Access Index: A global study (Quan Sun et al., 2023)

{{<citation>}}

Quan Sun, Wanjing Li, Qi Zhou. (2023)  
**Rural Access Index: A global study**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.00505v1)  

---


**ABSTRACT**  
The Rural Access Index (RAI), one of the UN Sustainable Development Goal indicators (SDG 9.1.1), represents the proportion of the rural population residing within 2 km of all-season roads. It reflects the accessibility of rural residents to transportation services and could provide guidance for the improvement of road infrastructure. The primary deficiencies in assessing the RAI include the limited studying area, its incomplete meaning and the absence of correlation analysis with other influencing factors. To address these issues, this study proposes the "Not-served Rural Population (NSRP)" as a complementary indicator to RAI. Utilizing multi-source open data, we analysed the spatial patterns of RAI and NSRP indicators for 203 countries and then explored the correlation between these 2 indicators and other 10 relevant factors. The main findings are as follows: 1) North America, Europe, and Oceania exhibit relatively high RAI values (>80%) and low NSRP values (<1 million). In contrast, African regions have relatively low RAI values (<40%) and high NSRP values (>5 million). There is a negative correlation between RAI and NSRP. 2) There is spatial autocorrelation and significant imbalances in the distribution of these two indicators. 3) The RAI exhibit a positive correlation with the factors showing levels of the development of countries such as GDP, education, indicating that improving the road infrastructure could reduce the poverty rates and enhance access to education. And in contrast with RAI, NSRP exhibit the completely negative correlations with these factors.

{{</citation>}}


## cs.RO (3)



### (63/71) Learning-based NLOS Detection and Uncertainty Prediction of GNSS Observations with Transformer-Enhanced LSTM Network (Haoming Zhang et al., 2023)

{{<citation>}}

Haoming Zhang, Zhanxin Wang, Heike Vallery. (2023)  
**Learning-based NLOS Detection and Uncertainty Prediction of GNSS Observations with Transformer-Enhanced LSTM Network**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: LSTM, Transformer  
[Paper Link](http://arxiv.org/abs/2309.00480v1)  

---


**ABSTRACT**  
The global navigation satellite systems (GNSS) play a vital role in transport systems for accurate and consistent vehicle localization. However, GNSS observations can be distorted due to multipath effects and non-line-of-sight (NLOS) receptions in challenging environments such as urban canyons. In such cases, traditional methods to classify and exclude faulty GNSS observations may fail, leading to unreliable state estimation and unsafe system operations. This work proposes a Deep-Learning-based method to detect NLOS receptions and predict GNSS pseudorange errors by analyzing GNSS observations as a spatio-temporal modeling problem. Compared to previous works, we construct a transformer-like attention mechanism to enhance the long short-term memory (LSTM) networks, improving model performance and generalization. For the training and evaluation of the proposed network, we used labeled datasets from the cities of Hong Kong and Aachen. We also introduce a dataset generation process to label the GNSS observations using lidar maps. In experimental studies, we compare the proposed network with a deep-learning-based model and classical machine-learning models. Furthermore, we conduct ablation studies of our network components and integrate the NLOS detection with data out-of-distribution in a state estimator. As a result, our network presents improved precision and recall ratios compared to other models. Additionally, we show that the proposed method avoids trajectory divergence in real-world vehicle localization by classifying and excluding NLOS observations.

{{</citation>}}


### (64/71) End-to-end Lidar-Driven Reinforcement Learning for Autonomous Racing (Meraj Mammadov, 2023)

{{<citation>}}

Meraj Mammadov. (2023)  
**End-to-end Lidar-Driven Reinforcement Learning for Autonomous Racing**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-LG, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.00296v1)  

---


**ABSTRACT**  
Reinforcement Learning (RL) has emerged as a transformative approach in the domains of automation and robotics, offering powerful solutions to complex problems that conventional methods struggle to address. In scenarios where the problem definitions are elusive and challenging to quantify, learning-based solutions such as RL become particularly valuable. One instance of such complexity can be found in the realm of car racing, a dynamic and unpredictable environment that demands sophisticated decision-making algorithms. This study focuses on developing and training an RL agent to navigate a racing environment solely using feedforward raw lidar and velocity data in a simulated context. The agent's performance, trained in the simulation environment, is then experimentally evaluated in a real-world racing scenario. This exploration underlines the feasibility and potential benefits of RL algorithm enhancing autonomous racing performance, especially in the environments where prior map information is not available.

{{</citation>}}


### (65/71) Parallel Distributional Prioritized Deep Reinforcement Learning for Unmanned Aerial Vehicles (Alisson Henrique Kolling et al., 2023)

{{<citation>}}

Alisson Henrique Kolling, Victor Augusto Kich, Junior Costa de Jesus, Andressa Cavalcante da Silva, Ricardo Bedin Grando, Paulo Lilles Jorge Drews-Jr, Daniel F. T. Gamarra. (2023)  
**Parallel Distributional Prioritized Deep Reinforcement Learning for Unmanned Aerial Vehicles**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.00176v1)  

---


**ABSTRACT**  
This work presents a study on parallel and distributional deep reinforcement learning applied to the mapless navigation of UAVs. For this, we developed an approach based on the Soft Actor-Critic method, producing a distributed and distributional variant named PDSAC, and compared it with a second one based on the traditional SAC algorithm. In addition, we also embodied a prioritized memory system into them. The UAV used in the study is based on the Hydrone vehicle, a hybrid quadrotor operating solely in the air. The inputs for the system are 23 range findings from a Lidar sensor and the distance and angles towards a desired goal, while the outputs consist of the linear, angular, and, altitude velocities. The methods were trained in environments of varying complexity, from obstacle-free environments to environments with multiple obstacles in three dimensions. The results obtained, demonstrate a concise improvement in the navigation capabilities by the proposed approach when compared to the agent based on the SAC for the same amount of training steps. In summary, this work presented a study on deep reinforcement learning applied to mapless navigation of drones in three dimensions, with promising results and potential applications in various contexts related to robotics and autonomous air navigation with distributed and distributional variants.

{{</citation>}}


## cs.MS (1)



### (66/71) A FAIR File Format for Mathematical Software (Antony Della Vecchia et al., 2023)

{{<citation>}}

Antony Della Vecchia, Michael Joswig, Benjamin Lorenz. (2023)  
**A FAIR File Format for Mathematical Software**  

---
Primary Category: cs.MS  
Categories: cs-MS, cs.MS  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.00465v1)  

---


**ABSTRACT**  
We describe a generic JSON based file format which is suitable for computations in computer algebra. This is implemented in the computer algebra system OSCAR, but we also indicate how it can be used in a different context.

{{</citation>}}


## cs.SD (2)



### (67/71) CoNeTTE: An efficient Audio Captioning system leveraging multiple datasets with Task Embedding (Étienne Labbé et al., 2023)

{{<citation>}}

Étienne Labbé, Thomas Pellegrini, Julien Pinquier. (2023)  
**CoNeTTE: An efficient Audio Captioning system leveraging multiple datasets with Task Embedding**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Embedding, Transformer  
[Paper Link](http://arxiv.org/abs/2309.00454v1)  

---


**ABSTRACT**  
Automated Audio Captioning (AAC) involves generating natural language descriptions of audio content, using encoder-decoder architectures. An audio encoder produces audio embeddings fed to a decoder, usually a Transformer decoder, for caption generation. In this work, we describe our model, which novelty, compared to existing models, lies in the use of a ConvNeXt architecture as audio encoder, adapted from the vision domain to audio classification. This model, called CNext-trans, achieved state-of-the-art scores on the AudioCaps (AC) dataset and performed competitively on Clotho (CL), while using four to forty times fewer parameters than existing models. We examine potential biases in the AC dataset due to its origin from AudioSet by investigating unbiased encoder's impact on performance. Using the well-known PANN's CNN14, for instance, as an unbiased encoder, we observed a 1.7% absolute reduction in SPIDEr score (where higher scores indicate better performance). To improve cross-dataset performance, we conducted experiments by combining multiple AAC datasets (AC, CL, MACS, WavCaps) for training. Although this strategy enhanced overall model performance across datasets, it still fell short compared to models trained specifically on a single target dataset, indicating the absence of a one-size-fits-all model. To mitigate performance gaps between datasets, we introduced a Task Embedding (TE) token, allowing the model to identify the source dataset for each input sample. We provide insights into the impact of these TEs on both the form (words) and content (sound event types) of the generated captions. The resulting model, named CoNeTTE, an unbiased CNext-trans model enriched with dataset-specific Task Embeddings, achieved SPIDEr scores of 44.1% and 30.5% on AC and CL, respectively. Code available: https://github.com/Labbeti/conette-audio-captioning.

{{</citation>}}


### (68/71) Mi-Go: Test Framework which uses YouTube as Data Source for Evaluating Speech Recognition Models like OpenAI's Whisper (Tomasz Wojnar et al., 2023)

{{<citation>}}

Tomasz Wojnar, Jaroslaw Hryszko, Adam Roman. (2023)  
**Mi-Go: Test Framework which uses YouTube as Data Source for Evaluating Speech Recognition Models like OpenAI's Whisper**  

---
Primary Category: cs.SD  
Categories: cs-LG, cs-SD, cs-SE, cs.SD, eess-AS  
Keywords: AI, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2309.00329v1)  

---


**ABSTRACT**  
This article introduces Mi-Go, a novel testing framework aimed at evaluating the performance and adaptability of general-purpose speech recognition machine learning models across diverse real-world scenarios. The framework leverages YouTube as a rich and continuously updated data source, accounting for multiple languages, accents, dialects, speaking styles, and audio quality levels. To demonstrate the effectiveness of the framework, the Whisper model, developed by OpenAI, was employed as a test object. The tests involve using a total of 124 YouTube videos to test all Whisper model versions. The results underscore the utility of YouTube as a valuable testing platform for speech recognition models, ensuring their robustness, accuracy, and adaptability to diverse languages and acoustic conditions. Additionally, by contrasting the machine-generated transcriptions against human-made subtitles, the Mi-Go framework can help pinpoint potential misuse of YouTube subtitles, like Search Engine Optimization.

{{</citation>}}


## physics.chem-ph (1)



### (69/71) Bespoke Nanoparticle Synthesis and Chemical Knowledge Discovery Via Autonomous Experimentations (Hyuk Jun Yoo et al., 2023)

{{<citation>}}

Hyuk Jun Yoo, Nayeon Kim, Heeseung Lee, Daeho Kim, Leslie Tiong Ching Ow, Hyobin Nam, Chansoo Kim, Seung Yong Lee, Kwan-Young Lee, Donghun Kim, Sang Soo Han. (2023)  
**Bespoke Nanoparticle Synthesis and Chemical Knowledge Discovery Via Autonomous Experimentations**  

---
Primary Category: physics.chem-ph  
Categories: cs-LG, physics-chem-ph, physics.chem-ph  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.00349v1)  

---


**ABSTRACT**  
The optimization of nanomaterial synthesis using numerous synthetic variables is considered to be extremely laborious task because the conventional combinatorial explorations are prohibitively expensive. In this work, we report an autonomous experimentation platform developed for the bespoke design of nanoparticles (NPs) with targeted optical properties. This platform operates in a closed-loop manner between a batch synthesis module of NPs and a UV- Vis spectroscopy module, based on the feedback of the AI optimization modeling. With silver (Ag) NPs as a representative example, we demonstrate that the Bayesian optimizer implemented with the early stopping criterion can efficiently produce Ag NPs precisely possessing the desired absorption spectra within only 200 iterations (when optimizing among five synthetic reagents). In addition to the outstanding material developmental efficiency, the analysis of synthetic variables further reveals a novel chemistry involving the effects of citrate in Ag NP synthesis. The amount of citrate is a key to controlling the competitions between spherical and plate-shaped NPs and, as a result, affects the shapes of the absorption spectra as well. Our study highlights both capabilities of the platform to enhance search efficiencies and to provide a novel chemical knowledge by analyzing datasets accumulated from the autonomous experimentations.

{{</citation>}}


## eess.IV (1)



### (70/71) Application of Machine Learning in Melanoma Detection and the Identification of 'Ugly Duckling' and Suspicious Naevi: A Review (Fatima Al Zegair et al., 2023)

{{<citation>}}

Fatima Al Zegair, Nathasha Naranpanawa, Brigid Betz-Stablein, Monika Janda, H. Peter Soyer, Shekhar S. Chandra. (2023)  
**Application of Machine Learning in Melanoma Detection and the Identification of 'Ugly Duckling' and Suspicious Naevi: A Review**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.00265v2)  

---


**ABSTRACT**  
Skin lesions known as naevi exhibit diverse characteristics such as size, shape, and colouration. The concept of an "Ugly Duckling Naevus" comes into play when monitoring for melanoma, referring to a lesion with distinctive features that sets it apart from other lesions in the vicinity. As lesions within the same individual typically share similarities and follow a predictable pattern, an ugly duckling naevus stands out as unusual and may indicate the presence of a cancerous melanoma. Computer-aided diagnosis (CAD) has become a significant player in the research and development field, as it combines machine learning techniques with a variety of patient analysis methods. Its aim is to increase accuracy and simplify decision-making, all while responding to the shortage of specialized professionals. These automated systems are especially important in skin cancer diagnosis where specialist availability is limited. As a result, their use could lead to life-saving benefits and cost reductions within healthcare. Given the drastic change in survival when comparing early stage to late-stage melanoma, early detection is vital for effective treatment and patient outcomes. Machine learning (ML) and deep learning (DL) techniques have gained popularity in skin cancer classification, effectively addressing challenges, and providing results equivalent to that of specialists. This article extensively covers modern Machine Learning and Deep Learning algorithms for detecting melanoma and suspicious naevi. It begins with general information on skin cancer and different types of naevi, then introduces AI, ML, DL, and CAD. The article then discusses the successful applications of various ML techniques like convolutional neural networks (CNN) for melanoma detection compared to dermatologists' performance. Lastly, it examines ML methods for UD naevus detection and identifying suspicious naevi.

{{</citation>}}


## cs.AR (1)



### (71/71) Security Verification of Low-Trust Architectures (Qinhan Tan et al., 2023)

{{<citation>}}

Qinhan Tan, Yonathan Fisseha, Shibo Chen, Lauren Biernacki, Jean-Baptiste Jeannin, Sharad Malik, Todd Austin. (2023)  
**Security Verification of Low-Trust Architectures**  

---
Primary Category: cs.AR  
Categories: cs-AR, cs-CR, cs.AR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2309.00181v1)  

---


**ABSTRACT**  
Low-trust architectures work on, from the viewpoint of software, always-encrypted data, and significantly reduce the amount of hardware trust to a small software-free enclave component. In this paper, we perform a complete formal verification of a specific low-trust architecture, the Sequestered Encryption (SE) architecture, to show that the design is secure against direct data disclosures and digital side channels for all possible programs. We first define the security requirements of the ISA of SE low-trust architecture. Looking upwards, this ISA serves as an abstraction of the hardware for the software, and is used to show how any program comprising these instructions cannot leak information, including through digital side channels. Looking downwards this ISA is a specification for the hardware, and is used to define the proof obligations for any RTL implementation arising from the ISA-level security requirements. These cover both functional and digital side-channel leakage. Next, we show how these proof obligations can be successfully discharged using commercial formal verification tools. We demonstrate the efficacy of our RTL security verification technique for seven different correct and buggy implementations of the SE architecture.

{{</citation>}}
