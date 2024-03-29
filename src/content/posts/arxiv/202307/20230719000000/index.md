---
draft: false
title: "arXiv @ 2023.07.19"
date: 2023-07-19
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.07.19"
    identifier: arxiv_20230719
    parent: 202307_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.LG (24)](#cslg-24)
- [cs.SI (2)](#cssi-2)
- [cs.AI (12)](#csai-12)
- [cs.DS (3)](#csds-3)
- [cs.CV (40)](#cscv-40)
- [cs.CY (1)](#cscy-1)
- [cs.CL (20)](#cscl-20)
- [cs.IR (3)](#csir-3)
- [cs.GL (1)](#csgl-1)
- [eess.SY (1)](#eesssy-1)
- [eess.IV (4)](#eessiv-4)
- [cs.GR (1)](#csgr-1)
- [math.ST (1)](#mathst-1)
- [physics.acc-ph (1)](#physicsacc-ph-1)
- [cs.HC (1)](#cshc-1)
- [q-bio.NC (1)](#q-bionc-1)
- [cs.CR (3)](#cscr-3)
- [cs.SE (3)](#csse-3)
- [cs.RO (2)](#csro-2)
- [cs.SD (2)](#cssd-2)
- [cs.DB (2)](#csdb-2)
- [eess.AS (2)](#eessas-2)
- [quant-ph (1)](#quant-ph-1)
- [q-bio.BM (1)](#q-biobm-1)

## cs.LG (24)



### (1/132) Basal-Bolus Advisor for Type 1 Diabetes (T1D) Patients Using Multi-Agent Reinforcement Learning (RL) Methodology (Mehrad Jalolia et al., 2023)

{{<citation>}}

Mehrad Jalolia, Marzia Cescon. (2023)  
**Basal-Bolus Advisor for Type 1 Diabetes (T1D) Patients Using Multi-Agent Reinforcement Learning (RL) Methodology**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CE, cs-LG, cs-SY, cs.LG, eess-SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.08897v1)  

---


**ABSTRACT**  
This paper presents a novel multi-agent reinforcement learning (RL) approach for personalized glucose control in individuals with type 1 diabetes (T1D). The method employs a closed-loop system consisting of a blood glucose (BG) metabolic model and a multi-agent soft actor-critic RL model acting as the basal-bolus advisor. Performance evaluation is conducted in three scenarios, comparing the RL agents to conventional therapy. Evaluation metrics include glucose levels (minimum, maximum, and mean), time spent in different BG ranges, and average daily bolus and basal insulin dosages. Results demonstrate that the RL-based basal-bolus advisor significantly improves glucose control, reducing glycemic variability and increasing time spent within the target range (70-180 mg/dL). Hypoglycemia events are effectively prevented, and severe hyperglycemia events are reduced. The RL approach also leads to a statistically significant reduction in average daily basal insulin dosage compared to conventional therapy. These findings highlight the effectiveness of the multi-agent RL approach in achieving better glucose control and mitigating the risk of severe hyperglycemia in individuals with T1D.

{{</citation>}}


### (2/132) Disentangling Node Attributes from Graph Topology for Improved Generalizability in Link Prediction (Ayan Chatterjee et al., 2023)

{{<citation>}}

Ayan Chatterjee, Robin Walters, Giulia Menichetti, Tina Eliassi-Rad. (2023)  
**Disentangling Node Attributes from Graph Topology for Improved Generalizability in Link Prediction**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SI, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2307.08877v1)  

---


**ABSTRACT**  
Link prediction is a crucial task in graph machine learning with diverse applications. We explore the interplay between node attributes and graph topology and demonstrate that incorporating pre-trained node attributes improves the generalization power of link prediction models. Our proposed method, UPNA (Unsupervised Pre-training of Node Attributes), solves the inductive link prediction problem by learning a function that takes a pair of node attributes and predicts the probability of an edge, as opposed to Graph Neural Networks (GNN), which can be prone to topological shortcuts in graphs with power-law degree distribution. In this manner, UPNA learns a significant part of the latent graph generation mechanism since the learned function can be used to add incoming nodes to a growing graph. By leveraging pre-trained node attributes, we overcome observational bias and make meaningful predictions about unobserved nodes, surpassing state-of-the-art performance (3X to 34X improvement on benchmark datasets). UPNA can be applied to various pairwise learning tasks and integrated with existing link prediction models to enhance their generalizability and bolster graph generative models.

{{</citation>}}


### (3/132) Natural Actor-Critic for Robust Reinforcement Learning with Function Approximation (Ruida Zhou et al., 2023)

{{<citation>}}

Ruida Zhou, Tao Liu, Min Cheng, Dileep Kalathil, P. R. Kumar, Chao Tian. (2023)  
**Natural Actor-Critic for Robust Reinforcement Learning with Function Approximation**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-RO, cs.LG, math-OC  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.08875v1)  

---


**ABSTRACT**  
We study robust reinforcement learning (RL) with the goal of determining a well-performing policy that is robust against model mismatch between the training simulator and the testing environment. Previous policy-based robust RL algorithms mainly focus on the tabular setting under uncertainty sets that facilitate robust policy evaluation, but are no longer tractable when the number of states scales up. To this end, we propose two novel uncertainty set formulations, one based on double sampling and the other on an integral probability metric. Both make large-scale robust RL tractable even when one only has access to a simulator. We propose a robust natural actor-critic (RNAC) approach that incorporates the new uncertainty sets and employs function approximation. We provide finite-time convergence guarantees for the proposed RNAC algorithm to the optimal robust policy within the function approximation error. Finally, we demonstrate the robust performance of the policy learned by our proposed RNAC approach in multiple MuJoCo environments and a real-world TurtleBot navigation task.

{{</citation>}}


### (4/132) Latent Space Representations of Neural Algorithmic Reasoners (Vladimir V. Mirjanić et al., 2023)

{{<citation>}}

Vladimir V. Mirjanić, Razvan Pascanu, Petar Veličković. (2023)  
**Latent Space Representations of Neural Algorithmic Reasoners**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: GNN, Graph Neural Network, Reasoning  
[Paper Link](http://arxiv.org/abs/2307.08874v1)  

---


**ABSTRACT**  
Neural Algorithmic Reasoning (NAR) is a research area focused on designing neural architectures that can reliably capture classical computation, usually by learning to execute algorithms. A typical approach is to rely on Graph Neural Network (GNN) architectures, which encode inputs in high-dimensional latent spaces that are repeatedly transformed during the execution of the algorithm. In this work we perform a detailed analysis of the structure of the latent space induced by the GNN when executing algorithms. We identify two possible failure modes: (i) loss of resolution, making it hard to distinguish similar values; (ii) inability to deal with values outside the range observed during training. We propose to solve the first issue by relying on a softmax aggregator, and propose to decay the latent space in order to deal with out-of-range values. We show that these changes lead to improvements on the majority of algorithms in the standard CLRS-30 benchmark when using the state-of-the-art Triplet-GMPNN processor. Our code is available at \href{https://github.com/mirjanic/nar-latent-spaces}{https://github.com/mirjanic/nar-latent-spaces}.

{{</citation>}}


### (5/132) An Alternative to Variance: Gini Deviation for Risk-averse Policy Gradient (Yudong Luo et al., 2023)

{{<citation>}}

Yudong Luo, Guiliang Liu, Pascal Poupart, Yangchen Pan. (2023)  
**An Alternative to Variance: Gini Deviation for Risk-averse Policy Gradient**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.08873v1)  

---


**ABSTRACT**  
Restricting the variance of a policy's return is a popular choice in risk-averse Reinforcement Learning (RL) due to its clear mathematical definition and easy interpretability. Traditional methods directly restrict the total return variance. Recent methods restrict the per-step reward variance as a proxy. We thoroughly examine the limitations of these variance-based methods, such as sensitivity to numerical scale and hindering of policy learning, and propose to use an alternative risk measure, Gini deviation, as a substitute. We study various properties of this new risk measure and derive a policy gradient algorithm to minimize it. Empirical evaluation in domains where risk-aversion can be clearly defined, shows that our algorithm can mitigate the limitations of variance-based risk measures and achieves high return with low risk in terms of variance and Gini deviation when others fail to learn a reasonable policy.

{{</citation>}}


### (6/132) Curriculum Learning for Graph Neural Networks: A Multiview Competence-based Approach (Nidhi Vakil et al., 2023)

{{<citation>}}

Nidhi Vakil, Hadi Amiri. (2023)  
**Curriculum Learning for Graph Neural Networks: A Multiview Competence-based Approach**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2307.08859v1)  

---


**ABSTRACT**  
A curriculum is a planned sequence of learning materials and an effective one can make learning efficient and effective for both humans and machines. Recent studies developed effective data-driven curriculum learning approaches for training graph neural networks in language applications. However, existing curriculum learning approaches often employ a single criterion of difficulty in their training paradigms. In this paper, we propose a new perspective on curriculum learning by introducing a novel approach that builds on graph complexity formalisms (as difficulty criteria) and model competence during training. The model consists of a scheduling scheme which derives effective curricula by accounting for different views of sample difficulty and model competence during training. The proposed solution advances existing research in curriculum learning for graph neural networks with the ability to incorporate a fine-grained spectrum of graph difficulty criteria in their training paradigms. Experimental results on real-world link prediction and node classification tasks illustrate the effectiveness of the proposed approach.

{{</citation>}}


### (7/132) Bayesian Safe Policy Learning with Chance Constrained Optimization: Application to Military Security Assessment during the Vietnam War (Zeyang Jia et al., 2023)

{{<citation>}}

Zeyang Jia, Eli Ben-Michael, Kosuke Imai. (2023)  
**Bayesian Safe Policy Learning with Chance Constrained Optimization: Application to Military Security Assessment during the Vietnam War**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-AP  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2307.08840v1)  

---


**ABSTRACT**  
Algorithmic and data-driven decisions and recommendations are commonly used in high-stakes decision-making settings such as criminal justice, medicine, and public policy. We investigate whether it would have been possible to improve a security assessment algorithm employed during the Vietnam War, using outcomes measured immediately after its introduction in late 1969. This empirical application raises several methodological challenges that frequently arise in high-stakes algorithmic decision-making. First, before implementing a new algorithm, it is essential to characterize and control the risk of yielding worse outcomes than the existing algorithm. Second, the existing algorithm is deterministic, and learning a new algorithm requires transparent extrapolation. Third, the existing algorithm involves discrete decision tables that are common but difficult to optimize over.   To address these challenges, we introduce the Average Conditional Risk (ACRisk), which first quantifies the risk that a new algorithmic policy leads to worse outcomes for subgroups of individual units and then averages this over the distribution of subgroups. We also propose a Bayesian policy learning framework that maximizes the posterior expected value while controlling the posterior expected ACRisk. This framework separates the estimation of heterogeneous treatment effects from policy optimization, enabling flexible estimation of effects and optimization over complex policy classes. We characterize the resulting chance-constrained optimization problem as a constrained linear programming problem. Our analysis shows that compared to the actual algorithm used during the Vietnam War, the learned algorithm assesses most regions as more secure and emphasizes economic and political factors over military factors.

{{</citation>}}


### (8/132) Towards Accelerating Benders Decomposition via Reinforcement Learning Surrogate Models (Stephen Mak et al., 2023)

{{<citation>}}

Stephen Mak, Kyle Mana, Parisa Zehtabi, Michael Cashmore, Daniele Magazzeni, Manuela Veloso. (2023)  
**Towards Accelerating Benders Decomposition via Reinforcement Learning Surrogate Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, math-OC  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.08816v1)  

---


**ABSTRACT**  
Stochastic optimization (SO) attempts to offer optimal decisions in the presence of uncertainty. Often, the classical formulation of these problems becomes intractable due to (a) the number of scenarios required to capture the uncertainty and (b) the discrete nature of real-world planning problems. To overcome these tractability issues, practitioners turn to decomposition methods that divide the problem into smaller, more tractable sub-problems. The focal decomposition method of this paper is Benders decomposition (BD), which decomposes stochastic optimization problems on the basis of scenario independence. In this paper we propose a method of accelerating BD with the aid of a surrogate model in place of an NP-hard integer master problem. Through the acceleration method we observe 30% faster average convergence when compared to other accelerated BD implementations. We introduce a reinforcement learning agent as a surrogate and demonstrate how it can be used to solve a stochastic inventory management problem.

{{</citation>}}


### (9/132) Anomaly Detection with Selective Dictionary Learning (Denis C. Ilie-Ablachim et al., 2023)

{{<citation>}}

Denis C. Ilie-Ablachim, Bogdan Dumitrescu. (2023)  
**Anomaly Detection with Selective Dictionary Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2307.08807v1)  

---


**ABSTRACT**  
In this paper we present new methods of anomaly detection based on Dictionary Learning (DL) and Kernel Dictionary Learning (KDL). The main contribution consists in the adaption of known DL and KDL algorithms in the form of unsupervised methods, used for outlier detection. We propose a reduced kernel version (RKDL), which is useful for problems with large data sets, due to the large kernel matrix. We also improve the DL and RKDL methods by the use of a random selection of signals, which aims to eliminate the outliers from the training procedure. All our algorithms are introduced in an anomaly detection toolbox and are compared to standard benchmark results.

{{</citation>}}


### (10/132) Non-Stationary Policy Learning for Multi-Timescale Multi-Agent Reinforcement Learning (Patrick Emami et al., 2023)

{{<citation>}}

Patrick Emami, Xiangyu Zhang, David Biagioni, Ahmed S. Zamzam. (2023)  
**Non-Stationary Policy Learning for Multi-Timescale Multi-Agent Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-MA, cs-SY, cs.LG, eess-SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.08794v1)  

---


**ABSTRACT**  
In multi-timescale multi-agent reinforcement learning (MARL), agents interact across different timescales. In general, policies for time-dependent behaviors, such as those induced by multiple timescales, are non-stationary. Learning non-stationary policies is challenging and typically requires sophisticated or inefficient algorithms. Motivated by the prevalence of this control problem in real-world complex systems, we introduce a simple framework for learning non-stationary policies for multi-timescale MARL. Our approach uses available information about agent timescales to define a periodic time encoding. In detail, we theoretically demonstrate that the effects of non-stationarity introduced by multiple timescales can be learned by a periodic multi-agent policy. To learn such policies, we propose a policy gradient algorithm that parameterizes the actor and critic with phase-functioned neural networks, which provide an inductive bias for periodicity. The framework's ability to effectively learn multi-timescale policies is validated on a gridworld and building energy management environment.

{{</citation>}}


### (11/132) FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning (Tri Dao, 2023)

{{<citation>}}

Tri Dao. (2023)  
**FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Attention, GPT, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.08691v1)  

---


**ABSTRACT**  
Scaling Transformers to longer sequence lengths has been a major problem in the last several years, promising to improve performance in language modeling and high-resolution image understanding, as well as to unlock new applications in code, audio, and video generation. The attention layer is the main bottleneck in scaling to longer sequences, as its runtime and memory increase quadratically in the sequence length. FlashAttention exploits the asymmetric GPU memory hierarchy to bring significant memory saving (linear instead of quadratic) and runtime speedup (2-4$\times$ compared to optimized baselines), with no approximation. However, FlashAttention is still not nearly as fast as optimized matrix-multiply (GEMM) operations, reaching only 25-40\% of the theoretical maximum FLOPs/s. We observe that the inefficiency is due to suboptimal work partitioning between different thread blocks and warps on the GPU, causing either low-occupancy or unnecessary shared memory reads/writes. We propose FlashAttention-2, with better work partitioning to address these issues. In particular, we (1) tweak the algorithm to reduce the number of non-matmul FLOPs (2) parallelize the attention computation, even for a single head, across different thread blocks to increase occupancy, and (3) within each thread block, distribute the work between warps to reduce communication through shared memory. These yield around 2$\times$ speedup compared to FlashAttention, reaching 50-73\% of the theoretical maximum FLOPs/s on A100 and getting close to the efficiency of GEMM operations. We empirically validate that when used end-to-end to train GPT-style models, FlashAttention-2 reaches training speed of up to 225 TFLOPs/s per A100 GPU (72\% model FLOPs utilization).

{{</citation>}}


### (12/132) LuckyMera: a Modular AI Framework for Building Hybrid NetHack Agents (Luigi Quarantiello et al., 2023)

{{<citation>}}

Luigi Quarantiello, Simone Marzeddu, Antonio Guzzi, Vincenzo Lomonaco. (2023)  
**LuckyMera: a Modular AI Framework for Building Hybrid NetHack Agents**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.08532v1)  

---


**ABSTRACT**  
In the last few decades we have witnessed a significant development in Artificial Intelligence (AI) thanks to the availability of a variety of testbeds, mostly based on simulated environments and video games. Among those, roguelike games offer a very good trade-off in terms of complexity of the environment and computational costs, which makes them perfectly suited to test AI agents generalization capabilities. In this work, we present LuckyMera, a flexible, modular, extensible and configurable AI framework built around NetHack, a popular terminal-based, single-player roguelike video game. This library is aimed at simplifying and speeding up the development of AI agents capable of successfully playing the game and offering a high-level interface for designing game strategies. LuckyMera comes with a set of off-the-shelf symbolic and neural modules (called "skills"): these modules can be either hard-coded behaviors, or neural Reinforcement Learning approaches, with the possibility of creating compositional hybrid solutions. Additionally, LuckyMera comes with a set of utility features to save its experiences in the form of trajectories for further analysis and to use them as datasets to train neural modules, with a direct interface to the NetHack Learning Environment and MiniHack. Through an empirical evaluation we validate our skills implementation and propose a strong baseline agent that can reach state-of-the-art performances in the complete NetHack game. LuckyMera is open-source and available at https://github.com/Pervasive-AI-Lab/LuckyMera.

{{</citation>}}


### (13/132) Can We Trust Race Prediction? (Cangyuan Li, 2023)

{{<citation>}}

Cangyuan Li. (2023)  
**Can We Trust Race Prediction?**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CY, cs-LG, cs.LG, stat-ML  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2307.08496v1)  

---


**ABSTRACT**  
In the absence of sensitive race and ethnicity data, researchers, regulators, and firms alike turn to proxies. In this paper, I train a Bidirectional Long Short-Term Memory (BiLSTM) model on a novel dataset of voter registration data from all 50 US states and create an ensemble that achieves up to 36.8% higher out of sample (OOS) F1 scores than the best performing machine learning models in the literature. Additionally, I construct the most comprehensive database of first and surname distributions in the US in order to improve the coverage and accuracy of Bayesian Improved Surname Geocoding (BISG) and Bayesian Improved Firstname Surname Geocoding (BIFSG). Finally, I provide the first high-quality benchmark dataset in order to fairly compare existing models and aid future model developers.

{{</citation>}}


### (14/132) Fairness in KI-Systemen (Janine Strotherm et al., 2023)

{{<citation>}}

Janine Strotherm, Alissa Müller, Barbara Hammer, Benjamin Paaßen. (2023)  
**Fairness in KI-Systemen**  

---
Primary Category: cs.LG  
Categories: cs-CY, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.08486v1)  

---


**ABSTRACT**  
The more AI-assisted decisions affect people's lives, the more important the fairness of such decisions becomes. In this chapter, we provide an introduction to research on fairness in machine learning. We explain the main fairness definitions and strategies for achieving fairness using concrete examples and place fairness research in the European context. Our contribution is aimed at an interdisciplinary audience and therefore avoids mathematical formulation but emphasizes visualizations and examples.   --   Je mehr KI-gest\"utzte Entscheidungen das Leben von Menschen betreffen, desto wichtiger ist die Fairness solcher Entscheidungen. In diesem Kapitel geben wir eine Einf\"uhrung in die Forschung zu Fairness im maschinellen Lernen. Wir erkl\"aren die wesentlichen Fairness-Definitionen und Strategien zur Erreichung von Fairness anhand konkreter Beispiele und ordnen die Fairness-Forschung in den europ\"aischen Kontext ein. Unser Beitrag richtet sich dabei an ein interdisziplin\"ares Publikum und verzichtet daher auf die mathematische Formulierung sondern betont Visualisierungen und Beispiele.

{{</citation>}}


### (15/132) Artificial Intelligence for Science in Quantum, Atomistic, and Continuum Systems (Xuan Zhang et al., 2023)

{{<citation>}}

Xuan Zhang, Limei Wang, Jacob Helwig, Youzhi Luo, Cong Fu, Yaochen Xie, Meng Liu, Yuchao Lin, Zhao Xu, Keqiang Yan, Keir Adams, Maurice Weiler, Xiner Li, Tianfan Fu, Yucheng Wang, Haiyang Yu, YuQing Xie, Xiang Fu, Alex Strasser, Shenglong Xu, Yi Liu, Yuanqi Du, Alexandra Saxton, Hongyi Ling, Hannah Lawrence, Hannes Stärk, Shurui Gui, Carl Edwards, Nicholas Gao, Adriana Ladera, Tailin Wu, Elyssa F. Hofgard, Aria Mansouri Tehrani, Rui Wang, Ameya Daigavane, Montgomery Bohde, Jerry Kurtin, Qian Huang, Tuong Phung, Minkai Xu, Chaitanya K. Joshi, Simon V. Mathis, Kamyar Azizzadenesheli, Ada Fang, Alán Aspuru-Guzik, Erik Bekkers, Michael Bronstein, Marinka Zitnik, Anima Anandkumar, Stefano Ermon, Pietro Liò, Rose Yu, Stephan Günnemann, Jure Leskovec, Heng Ji, Jimeng Sun, Regina Barzilay, Tommi Jaakkola, Connor W. Coley, Xiaoning Qian, Xiaofeng Qian, Tess Smidt, Shuiwang Ji. (2023)  
**Artificial Intelligence for Science in Quantum, Atomistic, and Continuum Systems**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, physics-comp-ph  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.08423v1)  

---


**ABSTRACT**  
Advances in artificial intelligence (AI) are fueling a new paradigm of discoveries in natural sciences. Today, AI has started to advance natural sciences by improving, accelerating, and enabling our understanding of natural phenomena at a wide range of spatial and temporal scales, giving rise to a new area of research known as AI for science (AI4Science). Being an emerging research paradigm, AI4Science is unique in that it is an enormous and highly interdisciplinary area. Thus, a unified and technical treatment of this field is needed yet challenging. This paper aims to provide a technically thorough account of a subarea of AI4Science; namely, AI for quantum, atomistic, and continuum systems. These areas aim at understanding the physical world from the subatomic (wavefunctions and electron density), atomic (molecules, proteins, materials, and interactions), to macro (fluids, climate, and subsurface) scales and form an important subarea of AI4Science. A unique advantage of focusing on these areas is that they largely share a common set of challenges, thereby allowing a unified and foundational treatment. A key common challenge is how to capture physics first principles, especially symmetries, in natural systems by deep learning methods. We provide an in-depth yet intuitive account of techniques to achieve equivariance to symmetry transformations. We also discuss other common technical challenges, including explainability, out-of-distribution generalization, knowledge transfer with foundation and large language models, and uncertainty quantification. To facilitate learning and education, we provide categorized lists of resources that we found to be useful. We strive to be thorough and unified and hope this initial effort may trigger more community interests and efforts to further advance AI4Science.

{{</citation>}}


### (16/132) Correlation-aware Spatial-Temporal Graph Learning for Multivariate Time-series Anomaly Detection (Yu Zheng et al., 2023)

{{<citation>}}

Yu Zheng, Huan Yee Koh, Ming Jin, Lianhua Chi, Khoa T. Phan, Shirui Pan, Yi-Ping Phoebe Chen, Wei Xiang. (2023)  
**Correlation-aware Spatial-Temporal Graph Learning for Multivariate Time-series Anomaly Detection**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Anomaly Detection, GNN, LSTM  
[Paper Link](http://arxiv.org/abs/2307.08390v1)  

---


**ABSTRACT**  
Multivariate time-series anomaly detection is critically important in many applications, including retail, transportation, power grid, and water treatment plants. Existing approaches for this problem mostly employ either statistical models which cannot capture the non-linear relations well or conventional deep learning models (e.g., CNN and LSTM) that do not explicitly learn the pairwise correlations among variables. To overcome these limitations, we propose a novel method, correlation-aware spatial-temporal graph learning (termed CST-GL), for time series anomaly detection. CST-GL explicitly captures the pairwise correlations via a multivariate time series correlation learning module based on which a spatial-temporal graph neural network (STGNN) can be developed. Then, by employing a graph convolution network that exploits one- and multi-hop neighbor information, our STGNN component can encode rich spatial information from complex pairwise dependencies between variables. With a temporal module that consists of dilated convolutional functions, the STGNN can further capture long-range dependence over time. A novel anomaly scoring component is further integrated into CST-GL to estimate the degree of an anomaly in a purely unsupervised manner. Experimental results demonstrate that CST-GL can detect anomalies effectively in general settings as well as enable early detection across different time delays.

{{</citation>}}


### (17/132) Tabular Machine Learning Methods for Predicting Gas Turbine Emissions (Rebecca Potts et al., 2023)

{{<citation>}}

Rebecca Potts, Rick Hackney, Georgios Leontidis. (2023)  
**Tabular Machine Learning Methods for Predicting Gas Turbine Emissions**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.08386v1)  

---


**ABSTRACT**  
Predicting emissions for gas turbines is critical for monitoring harmful pollutants being released into the atmosphere. In this study, we evaluate the performance of machine learning models for predicting emissions for gas turbines. We compare an existing predictive emissions model, a first principles-based Chemical Kinetics model, against two machine learning models we developed based on SAINT and XGBoost, to demonstrate improved predictive performance of nitrogen oxides (NOx) and carbon monoxide (CO) using machine learning techniques. Our analysis utilises a Siemens Energy gas turbine test bed tabular dataset to train and validate the machine learning models. Additionally, we explore the trade-off between incorporating more features to enhance the model complexity, and the resulting presence of increased missing values in the dataset.

{{</citation>}}


### (18/132) Zero-th Order Algorithm for Softmax Attention Optimization (Yichuan Deng et al., 2023)

{{<citation>}}

Yichuan Deng, Zhihang Li, Sridhar Mahadevan, Zhao Song. (2023)  
**Zero-th Order Algorithm for Softmax Attention Optimization**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2307.08352v1)  

---


**ABSTRACT**  
Large language models (LLMs) have brought about significant transformations in human society. Among the crucial computations in LLMs, the softmax unit holds great importance. Its helps the model generating a probability distribution on potential subsequent words or phrases, considering a series of input words. By utilizing this distribution, the model selects the most probable next word or phrase, based on the assigned probabilities. The softmax unit assumes a vital function in LLM training as it facilitates learning from data through the adjustment of neural network weights and biases.   With the development of the size of LLMs, computing the gradient becomes expensive. However, Zero-th Order method can approximately compute the gradient with only forward passes. In this paper, we present a Zero-th Order algorithm specifically tailored for Softmax optimization. We demonstrate the convergence of our algorithm, highlighting its effectiveness in efficiently computing gradients for large-scale LLMs. By leveraging the Zeroth-Order method, our work contributes to the advancement of optimization techniques in the context of complex language models.

{{</citation>}}


### (19/132) Efficient selective attention LSTM for well log curve synthesis (Yuankai Zhou et al., 2023)

{{<citation>}}

Yuankai Zhou, Huanyu Li. (2023)  
**Efficient selective attention LSTM for well log curve synthesis**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2307.10253v1)  

---


**ABSTRACT**  
Non-core drilling has gradually become the primary exploration method in geological engineering, and well logging curves have increasingly gained importance as the main carriers of geological information. However, factors such as geological environment, logging equipment, borehole quality, and unexpected events can all impact the quality of well logging curves. Previous methods of re-logging or manual corrections have been associated with high costs and low efficiency. This paper proposes a machine learning method that utilizes existing data to predict missing well logging curves, and its effectiveness and feasibility have been validated through experiments. The proposed method builds upon the traditional Long Short-Term Memory (LSTM) neural network by incorporating a self-attention mechanism to analyze the spatial dependencies of the data. It selectively includes the dominant computational results in the LSTM, reducing the computational complexity from O(n^2) to O(nlogn) and improving model efficiency. Experimental results demonstrate that the proposed method achieves higher accuracy compared to traditional curve synthesis methods based on Fully Connected Neural Networks (FCNN) and LSTM. This accurate, efficient, and cost-effective prediction method holds practical value in engineering applications.

{{</citation>}}


### (20/132) GBT: Two-stage transformer framework for non-stationary time series forecasting (Li Shen et al., 2023)

{{<citation>}}

Li Shen, Yuning Wei, Yangzhu Wang. (2023)  
**GBT: Two-stage transformer framework for non-stationary time series forecasting**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.08302v1)  

---


**ABSTRACT**  
This paper shows that time series forecasting Transformer (TSFT) suffers from severe over-fitting problem caused by improper initialization method of unknown decoder inputs, esp. when handling non-stationary time series. Based on this observation, we propose GBT, a novel two-stage Transformer framework with Good Beginning. It decouples the prediction process of TSFT into two stages, including Auto-Regression stage and Self-Regression stage to tackle the problem of different statistical properties between input and prediction sequences.Prediction results of Auto-Regression stage serve as a Good Beginning, i.e., a better initialization for inputs of Self-Regression stage. We also propose Error Score Modification module to further enhance the forecasting capability of the Self-Regression stage in GBT. Extensive experiments on seven benchmark datasets demonstrate that GBT outperforms SOTA TSFTs (FEDformer, Pyraformer, ETSformer, etc.) and many other forecasting models (SCINet, N-HiTS, etc.) with only canonical attention and convolution while owning less time and space complexity. It is also general enough to couple with these models to strengthen their forecasting capability. The source code is available at: https://github.com/OrigamiSL/GBT

{{</citation>}}


### (21/132) Complexity Matters: Rethinking the Latent Space for Generative Modeling (Tianyang Hu et al., 2023)

{{<citation>}}

Tianyang Hu, Fei Chen, Haonan Wang, Jiawei Li, Wenjia Wang, Jiacheng Sun, Zhenguo Li. (2023)  
**Complexity Matters: Rethinking the Latent Space for Generative Modeling**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.08283v1)  

---


**ABSTRACT**  
In generative modeling, numerous successful approaches leverage a low-dimensional latent space, e.g., Stable Diffusion models the latent space induced by an encoder and generates images through a paired decoder. Although the selection of the latent space is empirically pivotal, determining the optimal choice and the process of identifying it remain unclear. In this study, we aim to shed light on this under-explored topic by rethinking the latent space from the perspective of model complexity. Our investigation starts with the classic generative adversarial networks (GANs). Inspired by the GAN training objective, we propose a novel "distance" between the latent and data distributions, whose minimization coincides with that of the generator complexity. The minimizer of this distance is characterized as the optimal data-dependent latent that most effectively capitalizes on the generator's capacity. Then, we consider parameterizing such a latent distribution by an encoder network and propose a two-stage training strategy called Decoupled Autoencoder (DAE), where the encoder is only updated in the first stage with an auxiliary decoder and then frozen in the second stage while the actual decoder is being trained. DAE can improve the latent distribution and as a result, improve the generative performance. Our theoretical analyses are corroborated by comprehensive experiments on various models such as VQGAN and Diffusion Transformer, where our modifications yield significant improvements in sample quality with decreased model complexity.

{{</citation>}}


### (22/132) Certifying the Fairness of KNN in the Presence of Dataset Bias (Yannan Li et al., 2023)

{{<citation>}}

Yannan Li, Jingbo Wang, Chao Wang. (2023)  
**Certifying the Fairness of KNN in the Presence of Dataset Bias**  

---
Primary Category: cs.LG  
Categories: cs-CY, cs-LG, cs-SE, cs.LG  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2307.08722v1)  

---


**ABSTRACT**  
We propose a method for certifying the fairness of the classification result of a widely used supervised learning algorithm, the k-nearest neighbors (KNN), under the assumption that the training data may have historical bias caused by systematic mislabeling of samples from a protected minority group. To the best of our knowledge, this is the first certification method for KNN based on three variants of the fairness definition: individual fairness, $\epsilon$-fairness, and label-flipping fairness. We first define the fairness certification problem for KNN and then propose sound approximations of the complex arithmetic computations used in the state-of-the-art KNN algorithm. This is meant to lift the computation results from the concrete domain to an abstract domain, to reduce the computational cost. We show effectiveness of this abstract interpretation based technique through experimental evaluation on six datasets widely used in the fairness research literature. We also show that the method is accurate enough to obtain fairness certifications for a large number of test inputs, despite the presence of historical bias in the datasets.

{{</citation>}}


### (23/132) Learning for Counterfactual Fairness from Observational Data (Jing Ma et al., 2023)

{{<citation>}}

Jing Ma, Ruocheng Guo, Aidong Zhang, Jundong Li. (2023)  
**Learning for Counterfactual Fairness from Observational Data**  

---
Primary Category: cs.LG  
Categories: cs-CY, cs-LG, cs.LG, stat-ML  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.08232v1)  

---


**ABSTRACT**  
Fairness-aware machine learning has attracted a surge of attention in many domains, such as online advertising, personalized recommendation, and social media analysis in web applications. Fairness-aware machine learning aims to eliminate biases of learning models against certain subgroups described by certain protected (sensitive) attributes such as race, gender, and age. Among many existing fairness notions, counterfactual fairness is a popular notion defined from a causal perspective. It measures the fairness of a predictor by comparing the prediction of each individual in the original world and that in the counterfactual worlds in which the value of the sensitive attribute is modified. A prerequisite for existing methods to achieve counterfactual fairness is the prior human knowledge of the causal model for the data. However, in real-world scenarios, the underlying causal model is often unknown, and acquiring such human knowledge could be very difficult. In these scenarios, it is risky to directly trust the causal models obtained from information sources with unknown reliability and even causal discovery methods, as incorrect causal models can consequently bring biases to the predictor and lead to unfair predictions. In this work, we address the problem of counterfactually fair prediction from observational data without given causal models by proposing a novel framework CLAIRE. Specifically, under certain general assumptions, CLAIRE effectively mitigates the biases from the sensitive attribute with a representation learning framework based on counterfactual data augmentation and an invariant penalty. Experiments conducted on both synthetic and real-world datasets validate the superiority of CLAIRE in both counterfactual fairness and prediction performance.

{{</citation>}}


### (24/132) Can Euclidean Symmetry be Leveraged in Reinforcement Learning and Planning? (Linfeng Zhao et al., 2023)

{{<citation>}}

Linfeng Zhao, Owen Howell, Jung Yeon Park, Xupeng Zhu, Robin Walters, Lawson L. S. Wong. (2023)  
**Can Euclidean Symmetry be Leveraged in Reinforcement Learning and Planning?**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-RO, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.08226v1)  

---


**ABSTRACT**  
In robotic tasks, changes in reference frames typically do not influence the underlying physical properties of the system, which has been known as invariance of physical laws.These changes, which preserve distance, encompass isometric transformations such as translations, rotations, and reflections, collectively known as the Euclidean group. In this work, we delve into the design of improved learning algorithms for reinforcement learning and planning tasks that possess Euclidean group symmetry. We put forth a theory on that unify prior work on discrete and continuous symmetry in reinforcement learning, planning, and optimal control. Algorithm side, we further extend the 2D path planning with value-based planning to continuous MDPs and propose a pipeline for constructing equivariant sampling-based planning algorithms. Our work is substantiated with empirical evidence and illustrated through examples that explain the benefits of equivariance to Euclidean symmetry in tackling natural control problems.

{{</citation>}}


## cs.SI (2)



### (25/132) Examining the Effects of Degree Distribution and Homophily in Graph Learning Models (Mustafa Yasir et al., 2023)

{{<citation>}}

Mustafa Yasir, John Palowitch, Anton Tsitsulin, Long Tran-Thanh, Bryan Perozzi. (2023)  
**Examining the Effects of Degree Distribution and Homophily in Graph Learning Models**  

---
Primary Category: cs.SI  
Categories: cs-LG, cs-SI, cs.SI  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2307.08881v1)  

---


**ABSTRACT**  
Despite a surge in interest in GNN development, homogeneity in benchmarking datasets still presents a fundamental issue to GNN research. GraphWorld is a recent solution which uses the Stochastic Block Model (SBM) to generate diverse populations of synthetic graphs for benchmarking any GNN task. Despite its success, the SBM imposed fundamental limitations on the kinds of graph structure GraphWorld could create.   In this work we examine how two additional synthetic graph generators can improve GraphWorld's evaluation; LFR, a well-established model in the graph clustering literature and CABAM, a recent adaptation of the Barabasi-Albert model tailored for GNN benchmarking. By integrating these generators, we significantly expand the coverage of graph space within the GraphWorld framework while preserving key graph properties observed in real-world networks. To demonstrate their effectiveness, we generate 300,000 graphs to benchmark 11 GNN models on a node classification task. We find GNN performance variations in response to homophily, degree distribution and feature signal. Based on these findings, we classify models by their sensitivity to the new generators under these properties. Additionally, we release the extensions made to GraphWorld on the GitHub repository, offering further evaluation of GNN performance on new graphs.

{{</citation>}}


### (26/132) Temporally Stable Multilayer Network Embeddings: A Longitudinal Study of Russian Propaganda (Daniel Matter et al., 2023)

{{<citation>}}

Daniel Matter, Elizaveta Kuznetsova, Victoria Vziatysheva, Ilaria Vitulano, Juergen Pfeffer. (2023)  
**Temporally Stable Multilayer Network Embeddings: A Longitudinal Study of Russian Propaganda**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2307.10264v1)  

---


**ABSTRACT**  
Russian propaganda outlet RT (formerly, Russia Today) produces content in seven languages. There is ample evidence that RT's communication techniques differ for different language audiences. In this article, we offer the first comprehensive analysis of RT's multi-lingual article collection, analyzing all 2.4 million articles available on the online platform from 2006 until 06/2023. Annual semantic networks are created from the co-occurrence of the articles' tags. Within one language, we use AlignedUMAP to get stable inter-temporal embeddings. Between languages, we propose a new method to align multiple, sparsely connected networks in an intermediate representation before projecting them into the final embedding space. With respect to RT's communication strategy, our findings hint at a lack of a coherent strategy in RT's targeting of audiences in different languages, evident through differences in tag usage, clustering patterns, and uneven shifts in the prioritization of themes within language versions. Although identified clusters of tags align with the key themes in Russian propaganda, such as Ukraine, foreign affairs, Western countries, and the Middle East, we have observed significant differences in the attention given to specific issues across languages that are rather reactive to the information environment than representing a cohesive approach.

{{</citation>}}


## cs.AI (12)



### (27/132) AI for the Generation and Testing of Ideas Towards an AI Supported Knowledge Development Environment (Ted Selker, 2023)

{{<citation>}}

Ted Selker. (2023)  
**AI for the Generation and Testing of Ideas Towards an AI Supported Knowledge Development Environment**  

---
Primary Category: cs.AI  
Categories: H-5-0; I-2-0, cs-AI, cs.AI  
Keywords: AI, ChatGPT, GPT, Generative AI, Language Model, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.08876v1)  

---


**ABSTRACT**  
New systems employ Machine Learning to sift through large knowledge sources, creating flexible Large Language Models. These models discern context and predict sequential information in various communication forms. Generative AI, leveraging Transformers, generates textual or visual outputs mimicking human responses. It proposes one or multiple contextually feasible solutions for a user to contemplate. However, generative AI does not currently support traceability of ideas, a useful feature provided by search engines indicating origin of information. The narrative style of generative AI has gained positive reception. People learn from stories. Yet, early ChatGPT efforts had difficulty with truth, reference, calculations, and aspects like accurate maps. Current capabilities of referencing locations and linking to apps seem to be better catered by the link-centric search methods we've used for two decades. Deploying truly believable solutions extends beyond simulating contextual relevance as done by generative AI. Combining the creativity of generative AI with the provenance of internet sources in hybrid scenarios could enhance internet usage. Generative AI, viewed as drafts, stimulates thinking, offering alternative ideas for final versions or actions. Scenarios for information requests are considered. We discuss how generative AI can boost idea generation by eliminating human bias. We also describe how search can verify facts, logic, and context. The user evaluates these generated ideas for selection and usage. This paper introduces a system for knowledge workers, Generate And Search Test, enabling individuals to efficiently create solutions previously requiring top collaborations of experts.

{{</citation>}}


### (28/132) Operator Guidance Informed by AI-Augmented Simulations (Samuel J. Edwards et al., 2023)

{{<citation>}}

Samuel J. Edwards, Michael Levine. (2023)  
**Operator Guidance Informed by AI-Augmented Simulations**  

---
Primary Category: cs.AI  
Categories: 68T07, J-2, cs-AI, cs-LG, cs.AI, physics-ao-ph, stat-AP  
Keywords: AI, LSTM  
[Paper Link](http://arxiv.org/abs/2307.08810v1)  

---


**ABSTRACT**  
This paper will present a multi-fidelity, data-adaptive approach with a Long Short-Term Memory (LSTM) neural network to estimate ship response statistics in bimodal, bidirectional seas. The study will employ a fast low-fidelity, volume-based tool SimpleCode and a higher-fidelity tool known as the Large Amplitude Motion Program (LAMP). SimpleCode and LAMP data were generated by common bi-modal, bi-directional sea conditions in the North Atlantic as training data. After training an LSTM network with LAMP ship motion response data, a sample route was traversed and randomly sampled historical weather was input into SimpleCode and the LSTM network, and compared against the higher fidelity results.

{{</citation>}}


### (29/132) GEAR: Augmenting Language Models with Generalizable and Efficient Tool Resolution (Yining Lu et al., 2023)

{{<citation>}}

Yining Lu, Haoping Yu, Daniel Khashabi. (2023)  
**GEAR: Augmenting Language Models with Generalizable and Efficient Tool Resolution**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2307.08775v1)  

---


**ABSTRACT**  
Augmenting large language models (LLM) to use external tools enhances their performance across a variety of tasks. However, prior works over-rely on task-specific demonstration of tool use that limits their generalizability and computational cost due to making many calls to large-scale LLMs. We introduce GEAR, a computationally efficient query-tool grounding algorithm that is generalizable to various tasks that require tool use while not relying on task-specific demonstrations. GEAR achieves better efficiency by delegating tool grounding and execution to small language models (SLM) and LLM, respectively; while leveraging semantic and pattern-based evaluation at both question and answer levels for generalizable tool grounding. We evaluate GEAR on 14 datasets across 6 downstream tasks, demonstrating its strong generalizability to novel tasks, tools and different SLMs. Despite offering more efficiency, GEAR achieves higher precision in tool grounding compared to prior strategies using LLM prompting, thus improving downstream accuracy at a reduced computational cost. For example, we demonstrate that GEAR-augmented GPT-J and GPT-3 outperform counterpart tool-augmented baselines because of better tool use.

{{</citation>}}


### (30/132) Reflections from the Workshop on AI-Assisted Decision Making for Conservation (Lily Xu et al., 2023)

{{<citation>}}

Lily Xu, Esther Rolf, Sara Beery, Joseph R. Bennett, Tanya Berger-Wolf, Tanya Birch, Elizabeth Bondi-Kelly, Justin Brashares, Melissa Chapman, Anthony Corso, Andrew Davies, Nikhil Garg, Angela Gaylard, Robert Heilmayr, Hannah Kerner, Konstantin Klemmer, Vipin Kumar, Lester Mackey, Claire Monteleoni, Paul Moorcroft, Jonathan Palmer, Andrew Perrault, David Thau, Milind Tambe. (2023)  
**Reflections from the Workshop on AI-Assisted Decision Making for Conservation**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.08774v1)  

---


**ABSTRACT**  
In this white paper, we synthesize key points made during presentations and discussions from the AI-Assisted Decision Making for Conservation workshop, hosted by the Center for Research on Computation and Society at Harvard University on October 20-21, 2022. We identify key open research questions in resource allocation, planning, and interventions for biodiversity conservation, highlighting conservation challenges that not only require AI solutions, but also require novel methodological advances. In addition to providing a summary of the workshop talks and discussions, we hope this document serves as a call-to-action to orient the expansion of algorithmic decision-making approaches to prioritize real-world conservation challenges, through collaborative efforts of ecologists, conservation decision-makers, and AI researchers.

{{</citation>}}


### (31/132) TableGPT: Towards Unifying Tables, Nature Language and Commands into One GPT (Liangyu Zha et al., 2023)

{{<citation>}}

Liangyu Zha, Junlin Zhou, Liyao Li, Rui Wang, Qingyi Huang, Saisai Yang, Jing Yuan, Changbao Su, Xiang Li, Aofeng Su, Tao Zhang, Chen Zhou, Kaizhe Shou, Miao Wang, Wufang Zhu, Guoshan Lu, Chao Ye, Yali Ye, Wentao Ye, Yiming Zhang, Xinglong Deng, Jie Xu, Haobo Wang, Gang Chen, Junbo Zhao. (2023)  
**TableGPT: Towards Unifying Tables, Nature Language and Commands into One GPT**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2307.08674v2)  

---


**ABSTRACT**  
Tables are prevalent in real-world databases, requiring significant time and effort for humans to analyze and manipulate. The advancements in large language models (LLMs) have made it possible to interact with tables using natural language input, bringing this capability closer to reality. In this paper, we present TableGPT, a unified fine-tuned framework that enables LLMs to understand and operate on tables using external functional commands. It introduces the capability to seamlessly interact with tables, enabling a wide range of functionalities such as question answering, data manipulation (e.g., insert, delete, query, and modify operations), data visualization, analysis report generation, and automated prediction. TableGPT aims to provide convenience and accessibility to users by empowering them to effortlessly leverage tabular data. At the core of TableGPT lies the novel concept of global tabular representations, which empowers LLMs to gain a comprehensive understanding of the entire table beyond meta-information. By jointly training LLMs on both table and text modalities, TableGPT achieves a deep understanding of tabular data and the ability to perform complex operations on tables through chain-of-command instructions. Importantly, TableGPT offers the advantage of being a self-contained system rather than relying on external API interfaces. Moreover, it supports efficient data process flow, query rejection (when appropriate) and private deployment, enabling faster domain data fine-tuning and ensuring data privacy, which enhances the framework's adaptability to specific use cases.

{{</citation>}}


### (32/132) Navigating Fairness Measures and Trade-Offs (Stefan Buijsman, 2023)

{{<citation>}}

Stefan Buijsman. (2023)  
**Navigating Fairness Measures and Trade-Offs**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.08484v1)  

---


**ABSTRACT**  
In order to monitor and prevent bias in AI systems we can use a wide range of (statistical) fairness measures. However, it is mathematically impossible to optimize for all of these measures at the same time. In addition, optimizing a fairness measure often greatly reduces the accuracy of the system (Kozodoi et al, 2022). As a result, we need a substantive theory that informs us how to make these decisions and for what reasons. I show that by using Rawls' notion of justice as fairness, we can create a basis for navigating fairness measures and the accuracy trade-off. In particular, this leads to a principled choice focusing on both the most vulnerable groups and the type of fairness measure that has the biggest impact on that group. This also helps to close part of the gap between philosophical accounts of distributive justice and the fairness literature that has been observed (Kuppler et al, 2021) and to operationalise the value of fairness.

{{</citation>}}


### (33/132) Towards eXplainable AI for Mobility Data Science (Anahid Jalali et al., 2023)

{{<citation>}}

Anahid Jalali, Anita Graser, Clemens Heistracher. (2023)  
**Towards eXplainable AI for Mobility Data Science**  

---
Primary Category: cs.AI  
Categories: F-2-2, cs-AI, cs.AI  
Keywords: AI, GNN  
[Paper Link](http://arxiv.org/abs/2307.08461v1)  

---


**ABSTRACT**  
This paper presents our ongoing work towards XAI for Mobility Data Science applications, focusing on explainable models that can learn from dense trajectory data, such as GPS tracks of vehicles and vessels using temporal graph neural networks (GNNs) and counterfactuals. We review the existing GeoXAI studies, argue the need for comprehensible explanations with human-centered approaches, and outline a research path toward XAI for Mobility Data Science.

{{</citation>}}


### (34/132) Long-range Dependency based Multi-Layer Perceptron for Heterogeneous Information Networks (Chao Li et al., 2023)

{{<citation>}}

Chao Li, Zijie Guo, Qiuting He, Hao Xu, Kun He. (2023)  
**Long-range Dependency based Multi-Layer Perceptron for Heterogeneous Information Networks**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2307.08430v1)  

---


**ABSTRACT**  
Existing heterogeneous graph neural networks (HGNNs) have achieved great success in utilizing the rich semantic information in heterogeneous information networks (HINs). However, few works have delved into the utilization of long-range dependencies in HINs, which is extremely valuable as many real-world HINs are sparse, and each node has only a few directly connected neighbors. Although some HGNNs can utilize distant neighbors by stacking multiple layers or leveraging long meta-paths, the exponentially increased number of nodes in the receptive field or the number of meta-paths incurs high computation and memory costs. To address these issues, we investigate the importance of different meta-paths and propose Long-range Dependency based Multi-Layer Perceptron (LDMLP). Specifically, to solve the high-cost problem of leveraging long-range dependencies, LDMLP adopts a search stage to discover effective meta-paths automatically, reducing the exponentially increased number of meta-paths to a constant. To avoid the influence of specific modules on search results, LDMLP utilizes a simple architecture with only multi-layer perceptions in the search stage, improving the generalization of searched meta-paths. As a result, the searched meta-paths not only perform well in LDMLP but also enable other HGNNs like HAN and SeHGNN to perform better. Extensive experiments on eight heterogeneous datasets demonstrate that LDMLP achieves state-of-the-art performance while enjoying high efficiency and generalization, especially on sparse HINs.

{{</citation>}}


### (35/132) Neurosymbolic AI for Reasoning on Biomedical Knowledge Graphs (Lauren Nicole DeLong et al., 2023)

{{<citation>}}

Lauren Nicole DeLong, Ramon Fernández Mir, Zonglin Ji, Fiona Niamh Coulter Smith, Jacques D. Fleuriot. (2023)  
**Neurosymbolic AI for Reasoning on Biomedical Knowledge Graphs**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs-LO, cs.AI  
Keywords: AI, Knowledge Graph, Reasoning  
[Paper Link](http://arxiv.org/abs/2307.08411v1)  

---


**ABSTRACT**  
Biomedical datasets are often modeled as knowledge graphs (KGs) because they capture the multi-relational, heterogeneous, and dynamic natures of biomedical systems. KG completion (KGC), can, therefore, help researchers make predictions to inform tasks like drug repositioning. While previous approaches for KGC were either rule-based or embedding-based, hybrid approaches based on neurosymbolic artificial intelligence are becoming more popular. Many of these methods possess unique characteristics which make them even better suited toward biomedical challenges. Here, we survey such approaches with an emphasis on their utilities and prospective benefits for biomedicine.

{{</citation>}}


### (36/132) Gender mobility in the labor market with skills-based matching models (Ajaya Adhikari et al., 2023)

{{<citation>}}

Ajaya Adhikari, Steven Vethman, Daan Vos, Marc Lenz, Ioana Cocu, Ioannis Tolios, Cor J. Veenman. (2023)  
**Gender mobility in the labor market with skills-based matching models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2307.08368v1)  

---


**ABSTRACT**  
Skills-based matching promises mobility of workers between different sectors and occupations in the labor market. In this case, job seekers can look for jobs they do not yet have experience in, but for which they do have relevant skills. Currently, there are multiple occupations with a skewed gender distribution. For skills-based matching, it is unclear if and how a shift in the gender distribution, which we call gender mobility, between occupations will be effected. It is expected that the skills-based matching approach will likely be data-driven, including computational language models and supervised learning methods.   This work, first, shows the presence of gender segregation in language model-based skills representation of occupations. Second, we assess the use of these representations in a potential application based on simulated data, and show that the gender segregation is propagated by various data-driven skills-based matching models.These models are based on different language representations (bag of words, word2vec, and BERT), and distance metrics (static and machine learning-based). Accordingly, we show how skills-based matching approaches can be evaluated and compared on matching performance as well as on the risk of gender segregation. Making the gender segregation bias of models more explicit can help in generating healthy trust in the use of these models in practice.

{{</citation>}}


### (37/132) Abductive Reasoning with the GPT-4 Language Model: Case studies from criminal investigation, medical practice, scientific research (Remo Pareschi, 2023)

{{<citation>}}

Remo Pareschi. (2023)  
**Abductive Reasoning with the GPT-4 Language Model: Case studies from criminal investigation, medical practice, scientific research**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, GPT, GPT-4, Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2307.10250v1)  

---


**ABSTRACT**  
This study evaluates the GPT-4 Large Language Model's abductive reasoning in complex fields like medical diagnostics, criminology, and cosmology. Using an interactive interview format, the AI assistant demonstrated reliability in generating and selecting hypotheses. It inferred plausible medical diagnoses based on patient data and provided potential causes and explanations in criminology and cosmology. The results highlight the potential of LLMs in complex problem-solving and the need for further research to maximize their practical applications.

{{</citation>}}


### (38/132) Team Badminseok at IJCAI CoachAI Badminton Challenge 2023: Multi-Layer Multi-Input Transformer Network (MuLMINet) with Weighted Loss (Minwoo Seong et al., 2023)

{{<citation>}}

Minwoo Seong, Jeongseok Oh, SeungJun Kim. (2023)  
**Team Badminseok at IJCAI CoachAI Badminton Challenge 2023: Multi-Layer Multi-Input Transformer Network (MuLMINet) with Weighted Loss**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, Transformer  
[Paper Link](http://arxiv.org/abs/2307.08262v1)  

---


**ABSTRACT**  
The increasing use of artificial intelligence (AI) technology in turn-based sports, such as badminton, has sparked significant interest in evaluating strategies through the analysis of match video data. Predicting future shots based on past ones plays a vital role in coaching and strategic planning. In this study, we present a Multi-Layer Multi-Input Transformer Network (MuLMINet) that leverages professional badminton player match data to accurately predict future shot types and area coordinates. Our approach resulted in achieving the runner-up (2nd place) in the IJCAI CoachAI Badminton Challenge 2023, Track 2. To facilitate further research, we have made our code publicly accessible online, contributing to the broader research community's knowledge and advancements in the field of AI-assisted sports analysis.

{{</citation>}}


## cs.DS (3)



### (39/132) Quantum Tutte Embeddings (Shion Fukuzawa et al., 2023)

{{<citation>}}

Shion Fukuzawa, Michael T. Goodrich, Sandy Irani. (2023)  
**Quantum Tutte Embeddings**  

---
Primary Category: cs.DS  
Categories: cs-DS, cs.DS, quant-ph  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2307.08851v1)  

---


**ABSTRACT**  
Using the framework of Tutte embeddings, we begin an exploration of \emph{quantum graph drawing}, which uses quantum computers to visualize graphs. The main contributions of this paper include formulating a model for quantum graph drawing, describing how to create a graph-drawing quantum circuit from a given graph, and showing how a Tutte embedding can be calculated as a quantum state in this circuit that can then be sampled to extract the embedding. To evaluate the complexity of our quantum Tutte embedding circuits, we compare them to theoretical bounds established in the classical computing setting derived from a well-known classical algorithm for solving the types of linear systems that arise from Tutte embeddings. We also present empirical results obtained from experimental quantum simulations.

{{</citation>}}


### (40/132) Resource Augmentation Analysis of the Greedy Algorithm for the Online Transportation Problem (Stephen Arndt et al., 2023)

{{<citation>}}

Stephen Arndt, Josh Ascher, Kirk Pruhs. (2023)  
**Resource Augmentation Analysis of the Greedy Algorithm for the Online Transportation Problem**  

---
Primary Category: cs.DS  
Categories: cs-DS, cs.DS  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2307.08832v1)  

---


**ABSTRACT**  
We consider the online transportation problem set in a metric space containing parking garages of various capacities. Cars arrive over time, and must be assigned to an unfull parking garage upon their arrival. The objective is to minimize the aggregate distance that cars have to travel to their assigned parking garage. We show that the natural greedy algorithm, augmented with garages of $k\ge3$ times the capacity, is $\left(1 + \frac{2}{k-2}\right)$-competitive.

{{</citation>}}


### (41/132) Dynamic Planar Embedding is in DynFO (Samir Datta et al., 2023)

{{<citation>}}

Samir Datta, Asif Khan, Anish Mukherjee. (2023)  
**Dynamic Planar Embedding is in DynFO**  

---
Primary Category: cs.DS  
Categories: cs-CC, cs-DS, cs-LO, cs.DS  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2307.09473v1)  

---


**ABSTRACT**  
Planar Embedding is a drawing of a graph on the plane such that the edges do not intersect each other except at the vertices. We know that testing the planarity of a graph and computing its embedding (if it exists), can efficiently be computed, both sequentially [HT] and in parallel [RR94], when the entire graph is presented as input.   In the dynamic setting, the input graph changes one edge at a time through insertion and deletions and planarity testing/embedding has to be updated after every change. By storing auxilliary information we can improve the complexity of dynamic planarity testing/embedding over the obvious recomputation from scratch. In the sequential dynamic setting, there has been a series of works [EGIS, IPR, HIKLR, HR1], culminating in the breakthrough result of polylog(n) sequential time (amortized) planarity testing algorithm of Holm and Rotenberg [HR2].   In this paper, we study planar embedding through the lens of DynFO, a parallel dynamic complexity class introduced by Patnaik et al. [PI] (also [DST95]). We show that it is possible to dynamically maintain whether an edge can be inserted to a planar graph without causing non-planarity in DynFO. We extend this to show how to maintain an embedding of a planar graph under both edge insertions and deletions, while rejecting edge insertions that violate planarity.   Our main idea is to maintain embeddings of only the triconnected components and a special two-colouring of separating pairs that enables us to side-step cascading flips when embedding of a biconnected planar graph changes, a major issue for sequential dynamic algorithms [HR1, HR2].

{{</citation>}}


## cs.CV (40)



### (42/132) DARTS: Double Attention Reference-based Transformer for Super-resolution (Masoomeh Aslahishahri et al., 2023)

{{<citation>}}

Masoomeh Aslahishahri, Jordan Ubbens, Ian Stavness. (2023)  
**DARTS: Double Attention Reference-based Transformer for Super-resolution**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2307.08837v1)  

---


**ABSTRACT**  
We present DARTS, a transformer model for reference-based image super-resolution. DARTS learns joint representations of two image distributions to enhance the content of low-resolution input images through matching correspondences learned from high-resolution reference images. Current state-of-the-art techniques in reference-based image super-resolution are based on a multi-network, multi-stage architecture. In this work, we adapt the double attention block from the GAN literature, processing the two visual streams separately and combining self-attention and cross-attention blocks through a gating attention strategy. Our work demonstrates how the attention mechanism can be adapted for the particular requirements of reference-based image super-resolution, significantly simplifying the architecture and training pipeline. We show that our transformer-based model performs competitively with state-of-the-art models, while maintaining a simpler overall architecture and training process. In particular, we obtain state-of-the-art on the SUN80 dataset, with a PSNR/SSIM of 29.83 / .809. These results show that attention alone is sufficient for the RSR task, without multiple purpose-built subnetworks, knowledge distillation, or multi-stage training.

{{</citation>}}


### (43/132) Harnessing the Power of AI based Image Generation Model DALLE 2 in Agricultural Settings (Ranjan Sapkota, 2023)

{{<citation>}}

Ranjan Sapkota. (2023)  
**Harnessing the Power of AI based Image Generation Model DALLE 2 in Agricultural Settings**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, GPT  
[Paper Link](http://arxiv.org/abs/2307.08789v1)  

---


**ABSTRACT**  
This study investigates the potential impact of artificial intelligence (AI) on the enhancement of visualization processes in the agricultural sector, using the advanced AI image generator, DALLE 2, developed by OpenAI. By synergistically utilizing the natural language processing proficiency of chatGPT and the generative prowess of the DALLE 2 model, which employs a Generative Adversarial Networks (GANs) framework, our research offers an innovative method to transform textual descriptors into realistic visual content. Our rigorously assembled datasets include a broad spectrum of agricultural elements such as fruits, plants, and scenarios differentiating crops from weeds, maintained for AI-generated versus original images. The quality and accuracy of the AI-generated images were evaluated via established metrics including mean squared error (MSE), peak signal-to-noise ratio (PSNR), and feature similarity index (FSIM). The results underline the significant role of the DALLE 2 model in enhancing visualization processes in agriculture, aiding in more informed decision-making, and improving resource distribution. The outcomes of this research highlight the imminent rise of an AI-led transformation in the realm of precision agriculture.

{{</citation>}}


### (44/132) On the Real-Time Semantic Segmentation of Aphid Clusters in the Wild (Raiyan Rahman et al., 2023)

{{<citation>}}

Raiyan Rahman, Christopher Indris, Tianxiao Zhang, Kaidong Li, Brian McCornack, Daniel Flippo, Ajay Sharda, Guanghui Wang. (2023)  
**On the Real-Time Semantic Segmentation of Aphid Clusters in the Wild**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2307.10267v1)  

---


**ABSTRACT**  
Aphid infestations can cause extensive damage to wheat and sorghum fields and spread plant viruses, resulting in significant yield losses in agriculture. To address this issue, farmers often rely on chemical pesticides, which are inefficiently applied over large areas of fields. As a result, a considerable amount of pesticide is wasted on areas without pests, while inadequate amounts are applied to areas with severe infestations. The paper focuses on the urgent need for an intelligent autonomous system that can locate and spray infestations within complex crop canopies, reducing pesticide use and environmental impact. We have collected and labeled a large aphid image dataset in the field, and propose the use of real-time semantic segmentation models to segment clusters of aphids. A multiscale dataset is generated to allow for learning the clusters at different scales. We compare the segmentation speeds and accuracy of four state-of-the-art real-time semantic segmentation models on the aphid cluster dataset, benchmarking them against nonreal-time models. The study results show the effectiveness of a real-time solution, which can reduce inefficient pesticide use and increase crop yields, paving the way towards an autonomous pest detection system.

{{</citation>}}


### (45/132) Similarity Min-Max: Zero-Shot Day-Night Domain Adaptation (Rundong Luo et al., 2023)

{{<citation>}}

Rundong Luo, Wenjing Wang, Wenhan Yang, Jiaying Liu. (2023)  
**Similarity Min-Max: Zero-Shot Day-Night Domain Adaptation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2307.08779v2)  

---


**ABSTRACT**  
Low-light conditions not only hamper human visual experience but also degrade the model's performance on downstream vision tasks. While existing works make remarkable progress on day-night domain adaptation, they rely heavily on domain knowledge derived from the task-specific nighttime dataset. This paper challenges a more complicated scenario with border applicability, i.e., zero-shot day-night domain adaptation, which eliminates reliance on any nighttime data. Unlike prior zero-shot adaptation approaches emphasizing either image-level translation or model-level adaptation, we propose a similarity min-max paradigm that considers them under a unified framework. On the image level, we darken images towards minimum feature similarity to enlarge the domain gap. Then on the model level, we maximize the feature similarity between the darkened images and their normal-light counterparts for better model adaptation. To the best of our knowledge, this work represents the pioneering effort in jointly optimizing both aspects, resulting in a significant improvement of model generalizability. Extensive experiments demonstrate our method's effectiveness and broad applicability on various nighttime vision tasks, including classification, semantic segmentation, visual place recognition, and video action recognition. Code and pre-trained models are available at https://red-fairy.github.io/ZeroShotDayNightDA-Webpage/.

{{</citation>}}


### (46/132) UPSCALE: Unconstrained Channel Pruning (Alvin Wan et al., 2023)

{{<citation>}}

Alvin Wan, Hanxiang Hao, Kaushik Patnaik, Yueyang Xu, Omer Hadad, David Güera, Zhile Ren, Qi Shan. (2023)  
**UPSCALE: Unconstrained Channel Pruning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Pruning  
[Paper Link](http://arxiv.org/abs/2307.08771v1)  

---


**ABSTRACT**  
As neural networks grow in size and complexity, inference speeds decline. To combat this, one of the most effective compression techniques -- channel pruning -- removes channels from weights. However, for multi-branch segments of a model, channel removal can introduce inference-time memory copies. In turn, these copies increase inference latency -- so much so that the pruned model can be slower than the unpruned model. As a workaround, pruners conventionally constrain certain channels to be pruned together. This fully eliminates memory copies but, as we show, significantly impairs accuracy. We now have a dilemma: Remove constraints but increase latency, or add constraints and impair accuracy. In response, our insight is to reorder channels at export time, (1) reducing latency by reducing memory copies and (2) improving accuracy by removing constraints. Using this insight, we design a generic algorithm UPSCALE to prune models with any pruning pattern. By removing constraints from existing pruners, we improve ImageNet accuracy for post-training pruned models by 2.1 points on average -- benefiting DenseNet (+16.9), EfficientNetV2 (+7.9), and ResNet (+6.2). Furthermore, by reordering channels, UPSCALE improves inference speeds by up to 2x over a baseline export.

{{</citation>}}


### (47/132) Diffusion Models Beat GANs on Image Classification (Soumik Mukhopadhyay et al., 2023)

{{<citation>}}

Soumik Mukhopadhyay, Matthew Gwilliam, Vatsal Agarwal, Namitha Padmanabhan, Archana Swaminathan, Srinidhi Hegde, Tianyi Zhou, Abhinav Shrivastava. (2023)  
**Diffusion Models Beat GANs on Image Classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Image Classification, ImageNet  
[Paper Link](http://arxiv.org/abs/2307.08702v1)  

---


**ABSTRACT**  
While many unsupervised learning models focus on one family of tasks, either generative or discriminative, we explore the possibility of a unified representation learner: a model which uses a single pre-training stage to address both families of tasks simultaneously. We identify diffusion models as a prime candidate. Diffusion models have risen to prominence as a state-of-the-art method for image generation, denoising, inpainting, super-resolution, manipulation, etc. Such models involve training a U-Net to iteratively predict and remove noise, and the resulting model can synthesize high fidelity, diverse, novel images. The U-Net architecture, as a convolution-based architecture, generates a diverse set of feature representations in the form of intermediate feature maps. We present our findings that these embeddings are useful beyond the noise prediction task, as they contain discriminative information and can also be leveraged for classification. We explore optimal methods for extracting and using these embeddings for classification tasks, demonstrating promising results on the ImageNet classification task. We find that with careful feature selection and pooling, diffusion models outperform comparable generative-discriminative methods such as BigBiGAN for classification tasks. We investigate diffusion models in the transfer learning regime, examining their performance on several fine-grained visual classification datasets. We compare these embeddings to those generated by competing architectures and pre-trainings for classification tasks.

{{</citation>}}


### (48/132) Flow Matching in Latent Space (Quan Dao et al., 2023)

{{<citation>}}

Quan Dao, Hao Phung, Binh Nguyen, Anh Tran. (2023)  
**Flow Matching in Latent Space**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: AI, ImageNet  
[Paper Link](http://arxiv.org/abs/2307.08698v1)  

---


**ABSTRACT**  
Flow matching is a recent framework to train generative models that exhibits impressive empirical performance while being relatively easier to train compared with diffusion-based models. Despite its advantageous properties, prior methods still face the challenges of expensive computing and a large number of function evaluations of off-the-shelf solvers in the pixel space. Furthermore, although latent-based generative methods have shown great success in recent years, this particular model type remains underexplored in this area. In this work, we propose to apply flow matching in the latent spaces of pretrained autoencoders, which offers improved computational efficiency and scalability for high-resolution image synthesis. This enables flow-matching training on constrained computational resources while maintaining their quality and flexibility. Additionally, our work stands as a pioneering contribution in the integration of various conditions into flow matching for conditional generation tasks, including label-conditioned image generation, image inpainting, and semantic-to-image generation. Through extensive experiments, our approach demonstrates its effectiveness in both quantitative and qualitative results on various datasets, such as CelebA-HQ, FFHQ, LSUN Church & Bedroom, and ImageNet. We also provide a theoretical control of the Wasserstein-2 distance between the reconstructed latent flow distribution and true data distribution, showing it is upper-bounded by the latent flow matching objective. Our code will be available at https://github.com/VinAIResearch/LFM.git.

{{</citation>}}


### (49/132) Implementation of a perception system for autonomous vehicles using a detection-segmentation network in SoC FPGA (Maciej Baczmanski et al., 2023)

{{<citation>}}

Maciej Baczmanski, Mateusz Wasala, Tomasz Kryjak. (2023)  
**Implementation of a perception system for autonomous vehicles using a detection-segmentation network in SoC FPGA**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.08682v1)  

---


**ABSTRACT**  
Perception and control systems for autonomous vehicles are an active area of scientific and industrial research. These solutions should be characterised by high efficiency in recognising obstacles and other environmental elements in different road conditions, real-time capability, and energy efficiency. Achieving such functionality requires an appropriate algorithm and a suitable computing platform. In this paper, we have used the MultiTaskV3 detection-segmentation network as the basis for a perception system that can perform both functionalities within a single architecture. It was appropriately trained, quantised, and implemented on the AMD Xilinx Kria KV260 Vision AI embedded platform. By using this device, it was possible to parallelise and accelerate the computations. Furthermore, the whole system consumes relatively little power compared to a CPU-based implementation (an average of 5 watts, compared to the minimum of 55 watts for weaker CPUs, and the small size (119mm x 140mm x 36mm) of the platform allows it to be used in devices where the amount of space available is limited. It also achieves an accuracy higher than 97% of the mAP (mean average precision) for object detection and above 90% of the mIoU (mean intersection over union) for image segmentation. The article also details the design of the Mecanum wheel vehicle, which was used to test the proposed solution in a mock-up city.

{{</citation>}}


### (50/132) PolyGNN: Polyhedron-based Graph Neural Network for 3D Building Reconstruction from Point Clouds (Zhaiyu Chen et al., 2023)

{{<citation>}}

Zhaiyu Chen, Yilei Shi, Liangliang Nan, Zhitong Xiong, Xiao Xiang Zhu. (2023)  
**PolyGNN: Polyhedron-based Graph Neural Network for 3D Building Reconstruction from Point Clouds**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2307.08636v1)  

---


**ABSTRACT**  
We present PolyGNN, a polyhedron-based graph neural network for 3D building reconstruction from point clouds. PolyGNN learns to assemble primitives obtained by polyhedral decomposition via graph node classification, achieving a watertight, compact, and weakly semantic reconstruction. To effectively represent arbitrary-shaped polyhedra in the neural network, we propose three different sampling strategies to select representative points as polyhedron-wise queries, enabling efficient occupancy inference. Furthermore, we incorporate the inter-polyhedron adjacency to enhance the classification of the graph nodes. We also observe that existing city-building models are abstractions of the underlying instances. To address this abstraction gap and provide a fair evaluation of the proposed method, we develop our method on a large-scale synthetic dataset covering 500k+ buildings with well-defined ground truths of polyhedral class labels. We further conduct a transferability analysis across cities and on real-world point clouds. Both qualitative and quantitative results demonstrate the effectiveness of our method, particularly its efficiency for large-scale reconstructions. The source code and data of our work are available at https://github.com/chenzhaiyu/polygnn.

{{</citation>}}


### (51/132) Deficiency-Aware Masked Transformer for Video Inpainting (Yongsheng Yu et al., 2023)

{{<citation>}}

Yongsheng Yu, Heng Fan, Libo Zhang. (2023)  
**Deficiency-Aware Masked Transformer for Video Inpainting**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.08629v1)  

---


**ABSTRACT**  
Recent video inpainting methods have made remarkable progress by utilizing explicit guidance, such as optical flow, to propagate cross-frame pixels. However, there are cases where cross-frame recurrence of the masked video is not available, resulting in a deficiency. In such situation, instead of borrowing pixels from other frames, the focus of the model shifts towards addressing the inverse problem. In this paper, we introduce a dual-modality-compatible inpainting framework called Deficiency-aware Masked Transformer (DMT), which offers three key advantages. Firstly, we pretrain a image inpainting model DMT_img serve as a prior for distilling the video model DMT_vid, thereby benefiting the hallucination of deficiency cases. Secondly, the self-attention module selectively incorporates spatiotemporal tokens to accelerate inference and remove noise signals. Thirdly, a simple yet effective Receptive Field Contextualizer is integrated into DMT, further improving performance. Extensive experiments conducted on YouTube-VOS and DAVIS datasets demonstrate that DMT_vid significantly outperforms previous solutions. The code and video demonstrations can be found at github.com/yeates/DMT.

{{</citation>}}


### (52/132) Benchmarking fixed-length Fingerprint Representations across different Embedding Sizes and Sensor Types (Tim Rohwedder et al., 2023)

{{<citation>}}

Tim Rohwedder, Daile Osorio-Roig, Christian Rathgeb, Christoph Busch. (2023)  
**Benchmarking fixed-length Fingerprint Representations across different Embedding Sizes and Sensor Types**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2307.08615v1)  

---


**ABSTRACT**  
Traditional minutiae-based fingerprint representations consist of a variable-length set of minutiae. This necessitates a more complex comparison causing the drawback of high computational cost in one-to-many comparison. Recently, deep neural networks have been proposed to extract fixed-length embeddings from fingerprints. In this paper, we explore to what extent fingerprint texture information contained in such embeddings can be reduced in terms of dimension while preserving high biometric performance. This is of particular interest since it would allow to reduce the number of operations incurred at comparisons. We also study the impact in terms of recognition performance of the fingerprint textural information for two sensor types, i.e. optical and capacitive. Furthermore, the impact of rotation and translation of fingerprint images on the extraction of fingerprint embeddings is analysed. Experimental results conducted on a publicly available database reveal an optimal embedding size of 512 feature elements for the texture-based embedding part of fixed-length fingerprint representations. In addition, differences in performance between sensor types can be perceived.

{{</citation>}}


### (53/132) BuboGPT: Enabling Visual Grounding in Multi-Modal LLMs (Yang Zhao et al., 2023)

{{<citation>}}

Yang Zhao, Zhijie Lin, Daquan Zhou, Zilong Huang, Jiashi Feng, Bingyi Kang. (2023)  
**BuboGPT: Enabling Visual Grounding in Multi-Modal LLMs**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2307.08581v1)  

---


**ABSTRACT**  
LLMs have demonstrated remarkable abilities at interacting with humans through language, especially with the usage of instruction-following data. Recent advancements in LLMs, such as MiniGPT-4, LLaVA, and X-LLM, further enlarge their abilities by incorporating multi-modal inputs, including image, video, and speech. Despite their effectiveness at generating precise and detailed language understanding of the given modality signal, these LLMs give up the ability to ground specific parts of inputs, thus only constructing a coarse-grained mapping. However, explicit and informative correspondence between text and other modalities will not only improve the user experience but also help to expand the application scenario of multi-modal LLMs. Therefore, we propose BuboGPT, a multi-modal LLM with visual grounding that can perform cross-modal interaction between vision, audio and language, providing fine-grained understanding of visual objects and other given modalities. As a result, BuboGPT is able to point out the specific location of an object in the image, when it is generating response or description for that object. Our contributions are two-fold: 1) An off-the-shelf visual grounding module based on SAM that extracts entities in a sentence and find corresponding masks in the image. 2) A two-stage training scheme and instruction dataset to endow joint text-image-audio understanding. Our experiments show that BuboGPT achieves impressive multi-modality understanding and visual grounding abilities during the interaction with human. It performs consistently well when provided by arbitrary modality combinations (either aligned or unaligned). Our code, model and dataset are available at https://bubo-gpt.github.io .

{{</citation>}}


### (54/132) Scale-Aware Modulation Meet Transformer (Weifeng Lin et al., 2023)

{{<citation>}}

Weifeng Lin, Ziheng Wu, Jiayu Chen, Jun Huang, Lianwen Jin. (2023)  
**Scale-Aware Modulation Meet Transformer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Transformer  
[Paper Link](http://arxiv.org/abs/2307.08579v1)  

---


**ABSTRACT**  
This paper presents a new vision Transformer, Scale-Aware Modulation Transformer (SMT), that can handle various downstream tasks efficiently by combining the convolutional network and vision Transformer. The proposed Scale-Aware Modulation (SAM) in the SMT includes two primary novel designs. Firstly, we introduce the Multi-Head Mixed Convolution (MHMC) module, which can capture multi-scale features and expand the receptive field. Secondly, we propose the Scale-Aware Aggregation (SAA) module, which is lightweight but effective, enabling information fusion across different heads. By leveraging these two modules, convolutional modulation is further enhanced. Furthermore, in contrast to prior works that utilized modulations throughout all stages to build an attention-free network, we propose an Evolutionary Hybrid Network (EHN), which can effectively simulate the shift from capturing local to global dependencies as the network becomes deeper, resulting in superior performance. Extensive experiments demonstrate that SMT significantly outperforms existing state-of-the-art models across a wide range of visual tasks. Specifically, SMT with 11.5M / 2.4GFLOPs and 32M / 7.7GFLOPs can achieve 82.2% and 84.3% top-1 accuracy on ImageNet-1K, respectively. After pretrained on ImageNet-22K in 224^2 resolution, it attains 87.1% and 88.1% top-1 accuracy when finetuned with resolution 224^2 and 384^2, respectively. For object detection with Mask R-CNN, the SMT base trained with 1x and 3x schedule outperforms the Swin Transformer counterpart by 4.2 and 1.3 mAP on COCO, respectively. For semantic segmentation with UPerNet, the SMT base test at single- and multi-scale surpasses Swin by 2.0 and 1.1 mIoU respectively on the ADE20K.

{{</citation>}}


### (55/132) Variational Probabilistic Fusion Network for RGB-T Semantic Segmentation (Baihong Lin et al., 2023)

{{<citation>}}

Baihong Lin, Zengrong Lin, Yulan Guo, Yulan Zhang, Jianxiao Zou, Shicai Fan. (2023)  
**Variational Probabilistic Fusion Network for RGB-T Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2307.08536v1)  

---


**ABSTRACT**  
RGB-T semantic segmentation has been widely adopted to handle hard scenes with poor lighting conditions by fusing different modality features of RGB and thermal images. Existing methods try to find an optimal fusion feature for segmentation, resulting in sensitivity to modality noise, class-imbalance, and modality bias. To overcome the problems, this paper proposes a novel Variational Probabilistic Fusion Network (VPFNet), which regards fusion features as random variables and obtains robust segmentation by averaging segmentation results under multiple samples of fusion features. The random samples generation of fusion features in VPFNet is realized by a novel Variational Feature Fusion Module (VFFM) designed based on variation attention. To further avoid class-imbalance and modality bias, we employ the weighted cross-entropy loss and introduce prior information of illumination and category to control the proposed VFFM. Experimental results on MFNet and PST900 datasets demonstrate that the proposed VPFNet can achieve state-of-the-art segmentation performance.

{{</citation>}}


### (56/132) Multi-Domain Learning with Modulation Adapters (Ekaterina Iakovleva et al., 2023)

{{<citation>}}

Ekaterina Iakovleva, Karteek Alahari, Jakob Verbeek. (2023)  
**Multi-Domain Learning with Modulation Adapters**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: ImageNet, Sketch  
[Paper Link](http://arxiv.org/abs/2307.08528v1)  

---


**ABSTRACT**  
Deep convolutional networks are ubiquitous in computer vision, due to their excellent performance across different tasks for various domains. Models are, however, often trained in isolation for each task, failing to exploit relatedness between tasks and domains to learn more compact models that generalise better in low-data regimes. Multi-domain learning aims to handle related tasks, such as image classification across multiple domains, simultaneously. Previous work on this problem explored the use of a pre-trained and fixed domain-agnostic base network, in combination with smaller learnable domain-specific adaptation modules. In this paper, we introduce Modulation Adapters, which update the convolutional filter weights of the model in a multiplicative manner for each task. Parameterising these adaptation weights in a factored manner allows us to scale the number of per-task parameters in a flexible manner, and to strike different parameter-accuracy trade-offs. We evaluate our approach on the Visual Decathlon challenge, composed of ten image classification tasks across different domains, and on the ImageNet-to-Sketch benchmark, which consists of six image classification tasks. Our approach yields excellent results, with accuracies that are comparable to or better than those of existing state-of-the-art approaches.

{{</citation>}}


### (57/132) Image Captions are Natural Prompts for Text-to-Image Models (Shiye Lei et al., 2023)

{{<citation>}}

Shiye Lei, Hao Chen, Sen Zhang, Bo Zhao, Dacheng Tao. (2023)  
**Image Captions are Natural Prompts for Text-to-Image Models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: AI, ImageNet  
[Paper Link](http://arxiv.org/abs/2307.08526v1)  

---


**ABSTRACT**  
With the rapid development of Artificial Intelligence Generated Content (AIGC), it has become common practice in many learning tasks to train or fine-tune large models on synthetic data due to the data-scarcity and privacy leakage problems. Albeit promising with unlimited data generation, owing to massive and diverse information conveyed in real images, it is challenging for text-to-image generative models to synthesize informative training data with hand-crafted prompts, which usually leads to inferior generalization performance when training downstream models. In this paper, we theoretically analyze the relationship between the training effect of synthetic data and the synthetic data distribution induced by prompts. Then we correspondingly propose a simple yet effective method that prompts text-to-image generative models to synthesize more informative and diverse training data. Specifically, we caption each real image with the advanced captioning model to obtain informative and faithful prompts that extract class-relevant information and clarify the polysemy of class names. The image captions and class names are concatenated to prompt generative models for training image synthesis. Extensive experiments on ImageNette, ImageNet-100, and ImageNet-1K verify that our method significantly improves the performance of models trained on synthetic training data, i.e., 10% classification accuracy improvements on average.

{{</citation>}}


### (58/132) Does Visual Pretraining Help End-to-End Reasoning? (Chen Sun et al., 2023)

{{<citation>}}

Chen Sun, Calvin Luo, Xingyi Zhou, Anurag Arnab, Cordelia Schmid. (2023)  
**Does Visual Pretraining Help End-to-End Reasoning?**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2307.08506v1)  

---


**ABSTRACT**  
We aim to investigate whether end-to-end learning of visual reasoning can be achieved with general-purpose neural networks, with the help of visual pretraining. A positive result would refute the common belief that explicit visual abstraction (e.g. object detection) is essential for compositional generalization on visual reasoning, and confirm the feasibility of a neural network "generalist" to solve visual recognition and reasoning tasks. We propose a simple and general self-supervised framework which "compresses" each video frame into a small set of tokens with a transformer network, and reconstructs the remaining frames based on the compressed temporal context. To minimize the reconstruction loss, the network must learn a compact representation for each image, as well as capture temporal dynamics and object permanence from temporal context. We perform evaluation on two visual reasoning benchmarks, CATER and ACRE. We observe that pretraining is essential to achieve compositional generalization for end-to-end visual reasoning. Our proposed framework outperforms traditional supervised pretraining, including image classification and explicit object detection, by large margins.

{{</citation>}}


### (59/132) BUS:Efficient and Effective Vision-language Pre-training with Bottom-Up Patch Summarization (Chaoya Jiang et al., 2023)

{{<citation>}}

Chaoya Jiang, Haiyang Xu, Wei Ye, Qinghao Ye, Chenliang Li, Ming Yan, Bin Bi, Shikun Zhang, Fei Huang, Songfang Huang. (2023)  
**BUS:Efficient and Effective Vision-language Pre-training with Bottom-Up Patch Summarization**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Summarization, Transformer  
[Paper Link](http://arxiv.org/abs/2307.08504v1)  

---


**ABSTRACT**  
Vision Transformer (ViT) based Vision-Language Pre-training (VLP) models have demonstrated impressive performance in various tasks. However, the lengthy visual token sequences fed into ViT can lead to training inefficiency and ineffectiveness. Existing efforts address the challenge by either bottom-level patch extraction in the ViT backbone or top-level patch abstraction outside, not balancing training efficiency and effectiveness well. Inspired by text summarization in natural language processing, we propose a Bottom-Up Patch Summarization approach named BUS, coordinating bottom-level extraction and top-level abstraction to learn a concise summary of lengthy visual token sequences efficiently. Specifically, We incorporate a Text-Semantics-Aware Patch Selector (TSPS) into the ViT backbone to perform a coarse-grained visual token extraction and then attach a flexible Transformer-based Patch Abstraction Decoder (PAD) upon the backbone for top-level visual abstraction. This bottom-up collaboration enables our BUS to yield high training efficiency while maintaining or even improving effectiveness. We evaluate our approach on various visual-language understanding and generation tasks and show competitive downstream task performance while boosting the training efficiency by 50\%. Additionally, our model achieves state-of-the-art performance on many downstream tasks by increasing input image resolution without increasing computational costs over baselines.

{{</citation>}}


### (60/132) Cumulative Spatial Knowledge Distillation for Vision Transformers (Borui Zhao et al., 2023)

{{<citation>}}

Borui Zhao, Renjie Song, Jiajun Liang. (2023)  
**Cumulative Spatial Knowledge Distillation for Vision Transformers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Knowledge Distillation, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.08500v1)  

---


**ABSTRACT**  
Distilling knowledge from convolutional neural networks (CNNs) is a double-edged sword for vision transformers (ViTs). It boosts the performance since the image-friendly local-inductive bias of CNN helps ViT learn faster and better, but leading to two problems: (1) Network designs of CNN and ViT are completely different, which leads to different semantic levels of intermediate features, making spatial-wise knowledge transfer methods (e.g., feature mimicking) inefficient. (2) Distilling knowledge from CNN limits the network convergence in the later training period since ViT's capability of integrating global information is suppressed by CNN's local-inductive-bias supervision. To this end, we present Cumulative Spatial Knowledge Distillation (CSKD). CSKD distills spatial-wise knowledge to all patch tokens of ViT from the corresponding spatial responses of CNN, without introducing intermediate features. Furthermore, CSKD exploits a Cumulative Knowledge Fusion (CKF) module, which introduces the global response of CNN and increasingly emphasizes its importance during the training. Applying CKF leverages CNN's local inductive bias in the early training period and gives full play to ViT's global capability in the later one. Extensive experiments and analysis on ImageNet-1k and downstream datasets demonstrate the superiority of our CSKD. Code will be publicly available.

{{</citation>}}


### (61/132) SVDFormer: Complementing Point Cloud via Self-view Augmentation and Self-structure Dual-generator (Zhe Zhu et al., 2023)

{{<citation>}}

Zhe Zhu, Honghua Chen, Xing He, Weiming Wang, Jing Qin, Mingqiang Wei. (2023)  
**SVDFormer: Complementing Point Cloud via Self-view Augmentation and Self-structure Dual-generator**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2307.08492v1)  

---


**ABSTRACT**  
In this paper, we propose a novel network, SVDFormer, to tackle two specific challenges in point cloud completion: understanding faithful global shapes from incomplete point clouds and generating high-accuracy local structures. Current methods either perceive shape patterns using only 3D coordinates or import extra images with well-calibrated intrinsic parameters to guide the geometry estimation of the missing parts. However, these approaches do not always fully leverage the cross-modal self-structures available for accurate and high-quality point cloud completion. To this end, we first design a Self-view Fusion Network that leverages multiple-view depth image information to observe incomplete self-shape and generate a compact global shape. To reveal highly detailed structures, we then introduce a refinement module, called Self-structure Dual-generator, in which we incorporate learned shape priors and geometric self-similarities for producing new points. By perceiving the incompleteness of each point, the dual-path design disentangles refinement strategies conditioned on the structural type of each point. SVDFormer absorbs the wisdom of self-structures, avoiding any additional paired information such as color images with precisely calibrated camera intrinsic parameters. Comprehensive experiments indicate that our method achieves state-of-the-art performance on widely-used benchmarks. Code will be available at https://github.com/czvvd/SVDFormer.

{{</citation>}}


### (62/132) Differentiable Transportation Pruning (Yunqiang Li et al., 2023)

{{<citation>}}

Yunqiang Li, Jan C. van Gemert, Torsten Hoefler, Bert Moons, Evangelos Eleftheriou, Bram-Ernst Verhoef. (2023)  
**Differentiable Transportation Pruning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2307.08483v1)  

---


**ABSTRACT**  
Deep learning algorithms are increasingly employed at the edge. However, edge devices are resource constrained and thus require efficient deployment of deep neural networks. Pruning methods are a key tool for edge deployment as they can improve storage, compute, memory bandwidth, and energy usage. In this paper we propose a novel accurate pruning technique that allows precise control over the output network size. Our method uses an efficient optimal transportation scheme which we make end-to-end differentiable and which automatically tunes the exploration-exploitation behavior of the algorithm to find accurate sparse sub-networks. We show that our method achieves state-of-the-art performance compared to previous pruning methods on 3 different datasets, using 5 different models, across a wide range of pruning ratios, and with two types of sparsity budgets and pruning granularities.

{{</citation>}}


### (63/132) SkeletonMAE: Graph-based Masked Autoencoder for Skeleton Sequence Pre-training (Hong Yan et al., 2023)

{{<citation>}}

Hong Yan, Yang Liu, Yushen Wei, Zhen Li, Guanbin Li, Liang Lin. (2023)  
**SkeletonMAE: Graph-based Masked Autoencoder for Skeleton Sequence Pre-training**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Graph Convolutional Network, Representation Learning  
[Paper Link](http://arxiv.org/abs/2307.08476v1)  

---


**ABSTRACT**  
Skeleton sequence representation learning has shown great advantages for action recognition due to its promising ability to model human joints and topology. However, the current methods usually require sufficient labeled data for training computationally expensive models, which is labor-intensive and time-consuming. Moreover, these methods ignore how to utilize the fine-grained dependencies among different skeleton joints to pre-train an efficient skeleton sequence learning model that can generalize well across different datasets. In this paper, we propose an efficient skeleton sequence learning framework, named Skeleton Sequence Learning (SSL). To comprehensively capture the human pose and obtain discriminative skeleton sequence representation, we build an asymmetric graph-based encoder-decoder pre-training architecture named SkeletonMAE, which embeds skeleton joint sequence into Graph Convolutional Network (GCN) and reconstructs the masked skeleton joints and edges based on the prior human topology knowledge. Then, the pre-trained SkeletonMAE encoder is integrated with the Spatial-Temporal Representation Learning (STRL) module to build the SSL framework. Extensive experimental results show that our SSL generalizes well across different datasets and outperforms the state-of-the-art self-supervised skeleton-based action recognition methods on FineGym, Diving48, NTU 60 and NTU 120 datasets. Additionally, we obtain comparable performance to some fully supervised methods. The code is avaliable at https://github.com/HongYan1123/SkeletonMAE.

{{</citation>}}


### (64/132) DOT: A Distillation-Oriented Trainer (Borui Zhao et al., 2023)

{{<citation>}}

Borui Zhao, Quan Cui, Renjie Song, Jiajun Liang. (2023)  
**DOT: A Distillation-Oriented Trainer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2307.08436v1)  

---


**ABSTRACT**  
Knowledge distillation transfers knowledge from a large model to a small one via task and distillation losses. In this paper, we observe a trade-off between task and distillation losses, i.e., introducing distillation loss limits the convergence of task loss. We believe that the trade-off results from the insufficient optimization of distillation loss. The reason is: The teacher has a lower task loss than the student, and a lower distillation loss drives the student more similar to the teacher, then a better-converged task loss could be obtained. To break the trade-off, we propose the Distillation-Oriented Trainer (DOT). DOT separately considers gradients of task and distillation losses, then applies a larger momentum to distillation loss to accelerate its optimization. We empirically prove that DOT breaks the trade-off, i.e., both losses are sufficiently optimized. Extensive experiments validate the superiority of DOT. Notably, DOT achieves a +2.59% accuracy improvement on ImageNet-1k for the ResNet50-MobileNetV1 pair. Conclusively, DOT greatly benefits the student's optimization properties in terms of loss convergence and model generalization. Code will be made publicly available.

{{</citation>}}


### (65/132) Dense Affinity Matching for Few-Shot Segmentation (Hao Chen et al., 2023)

{{<citation>}}

Hao Chen, Yonghan Dong, Zheming Lu, Yunlong Yu, Yingming Li, Jungong Han, Zhongfei Zhang. (2023)  
**Dense Affinity Matching for Few-Shot Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2307.08434v1)  

---


**ABSTRACT**  
Few-Shot Segmentation (FSS) aims to segment the novel class images with a few annotated samples. In this paper, we propose a dense affinity matching (DAM) framework to exploit the support-query interaction by densely capturing both the pixel-to-pixel and pixel-to-patch relations in each support-query pair with the bidirectional 3D convolutions. Different from the existing methods that remove the support background, we design a hysteretic spatial filtering module (HSFM) to filter the background-related query features and retain the foreground-related query features with the assistance of the support background, which is beneficial for eliminating interference objects in the query background. We comprehensively evaluate our DAM on ten benchmarks under cross-category, cross-dataset, and cross-domain FSS tasks. Experimental results demonstrate that DAM performs very competitively under different settings with only 0.68M parameters, especially under cross-domain FSS tasks, showing its effectiveness and efficiency.

{{</citation>}}


### (66/132) Monocular 3D Object Detection with LiDAR Guided Semi Supervised Active Learning (Aral Hekimoglu et al., 2023)

{{<citation>}}

Aral Hekimoglu, Michael Schmidt, Alvaro Marcos-Ramiro. (2023)  
**Monocular 3D Object Detection with LiDAR Guided Semi Supervised Active Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Active Learning, Object Detection  
[Paper Link](http://arxiv.org/abs/2307.08415v1)  

---


**ABSTRACT**  
We propose a novel semi-supervised active learning (SSAL) framework for monocular 3D object detection with LiDAR guidance (MonoLiG), which leverages all modalities of collected data during model development. We utilize LiDAR to guide the data selection and training of monocular 3D detectors without introducing any overhead in the inference phase. During training, we leverage the LiDAR teacher, monocular student cross-modal framework from semi-supervised learning to distill information from unlabeled data as pseudo-labels. To handle the differences in sensor characteristics, we propose a data noise-based weighting mechanism to reduce the effect of propagating noise from LiDAR modality to monocular. For selecting which samples to label to improve the model performance, we propose a sensor consistency-based selection score that is also coherent with the training objective. Extensive experimental results on KITTI and Waymo datasets verify the effectiveness of our proposed framework. In particular, our selection strategy consistently outperforms state-of-the-art active learning baselines, yielding up to 17% better saving rate in labeling costs. Our training strategy attains the top place in KITTI 3D and birds-eye-view (BEV) monocular object detection official benchmarks by improving the BEV Average Precision (AP) by 2.02.

{{</citation>}}


### (67/132) Active Learning for Object Detection with Non-Redundant Informative Sampling (Aral Hekimoglu et al., 2023)

{{<citation>}}

Aral Hekimoglu, Adrian Brucker, Alper Kagan Kayali, Michael Schmidt, Alvaro Marcos-Ramiro. (2023)  
**Active Learning for Object Detection with Non-Redundant Informative Sampling**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Active Learning, Object Detection  
[Paper Link](http://arxiv.org/abs/2307.08414v1)  

---


**ABSTRACT**  
Curating an informative and representative dataset is essential for enhancing the performance of 2D object detectors. We present a novel active learning sampling strategy that addresses both the informativeness and diversity of the selections. Our strategy integrates uncertainty and diversity-based selection principles into a joint selection objective by measuring the collective information score of the selected samples. Specifically, our proposed NORIS algorithm quantifies the impact of training with a sample on the informativeness of other similar samples. By exclusively selecting samples that are simultaneously informative and distant from other highly informative samples, we effectively avoid redundancy while maintaining a high level of informativeness. Moreover, instead of utilizing whole image features to calculate distances between samples, we leverage features extracted from detected object regions within images to define object features. This allows us to construct a dataset encompassing diverse object types, shapes, and angles. Extensive experiments on object detection and image classification tasks demonstrate the effectiveness of our strategy over the state-of-the-art baselines. Specifically, our selection strategy achieves a 20% and 30% reduction in labeling costs compared to random selection for PASCAL-VOC and KITTI, respectively.

{{</citation>}}


### (68/132) M-FLAG: Medical Vision-Language Pre-training with Frozen Language Models and Latent Space Geometry Optimization (Che Liu et al., 2023)

{{<citation>}}

Che Liu, Sibo Cheng, Chen Chen, Mengyun Qiao, Weitong Zhang, Anand Shah, Wenjia Bai, Rossella Arcucci. (2023)  
**M-FLAG: Medical Vision-Language Pre-training with Frozen Language Models and Latent Space Geometry Optimization**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: ImageNet, Language Model  
[Paper Link](http://arxiv.org/abs/2307.08347v2)  

---


**ABSTRACT**  
Medical vision-language models enable co-learning and integrating features from medical imaging and clinical text. However, these models are not easy to train and the latent representation space can be complex. Here we propose a novel way for pre-training and regularising medical vision-language models. The proposed method, named Medical vision-language pre-training with Frozen language models and Latent spAce Geometry optimization (M-FLAG), leverages a frozen language model for training stability and efficiency and introduces a novel orthogonality loss to harmonize the latent space geometry. We demonstrate the potential of the pre-trained model on three downstream tasks: medical image classification, segmentation, and object detection. Extensive experiments across five public datasets demonstrate that M-FLAG significantly outperforms existing medical vision-language pre-training approaches and reduces the number of parameters by 78\%. Notably, M-FLAG achieves outstanding performance on the segmentation task while using only 1\% of the RSNA dataset, even outperforming ImageNet pre-trained models that have been fine-tuned using 100\% of the data.

{{</citation>}}


### (69/132) Multi-Task Cross-Modality Attention-Fusion for 2D Object Detection (Huawei Sun et al., 2023)

{{<citation>}}

Huawei Sun, Hao Feng, Georg Stettinger, Lorenzo Servadei, Robert Wille. (2023)  
**Multi-Task Cross-Modality Attention-Fusion for 2D Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-MM, cs.CV  
Keywords: Attention, Object Detection  
[Paper Link](http://arxiv.org/abs/2307.08339v1)  

---


**ABSTRACT**  
Accurate and robust object detection is critical for autonomous driving. Image-based detectors face difficulties caused by low visibility in adverse weather conditions. Thus, radar-camera fusion is of particular interest but presents challenges in optimally fusing heterogeneous data sources. To approach this issue, we propose two new radar preprocessing techniques to better align radar and camera data. In addition, we introduce a Multi-Task Cross-Modality Attention-Fusion Network (MCAF-Net) for object detection, which includes two new fusion blocks. These allow for exploiting information from the feature maps more comprehensively. The proposed algorithm jointly detects objects and segments free space, which guides the model to focus on the more relevant part of the scene, namely, the occupied space. Our approach outperforms current state-of-the-art radar-camera fusion-based object detectors in the nuScenes dataset and achieves more robust results in adverse weather conditions and nighttime scenarios.

{{</citation>}}


### (70/132) Bridging the Gap: Multi-Level Cross-Modality Joint Alignment for Visible-Infrared Person Re-Identification (Tengfei Liang et al., 2023)

{{<citation>}}

Tengfei Liang, Yi Jin, Wu Liu, Tao Wang, Songhe Feng, Yidong Li. (2023)  
**Bridging the Gap: Multi-Level Cross-Modality Joint Alignment for Visible-Infrared Person Re-Identification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2307.08316v1)  

---


**ABSTRACT**  
Visible-Infrared person Re-IDentification (VI-ReID) is a challenging cross-modality image retrieval task that aims to match pedestrians' images across visible and infrared cameras. To solve the modality gap, existing mainstream methods adopt a learning paradigm converting the image retrieval task into an image classification task with cross-entropy loss and auxiliary metric learning losses. These losses follow the strategy of adjusting the distribution of extracted embeddings to reduce the intra-class distance and increase the inter-class distance. However, such objectives do not precisely correspond to the final test setting of the retrieval task, resulting in a new gap at the optimization level. By rethinking these keys of VI-ReID, we propose a simple and effective method, the Multi-level Cross-modality Joint Alignment (MCJA), bridging both modality and objective-level gap. For the former, we design the Modality Alignment Augmentation, which consists of three novel strategies, the weighted grayscale, cross-channel cutmix, and spectrum jitter augmentation, effectively reducing modality discrepancy in the image space. For the latter, we introduce a new Cross-Modality Retrieval loss. It is the first work to constrain from the perspective of the ranking list, aligning with the goal of the testing stage. Moreover, based on the global feature only, our method exhibits good performance and can serve as a strong baseline method for the VI-ReID community.

{{</citation>}}


### (71/132) A Novel Multi-Task Model Imitating Dermatologists for Accurate Differential Diagnosis of Skin Diseases in Clinical Images (Yan-Jie Zhou et al., 2023)

{{<citation>}}

Yan-Jie Zhou, Wei Liu, Yuan Gao, Jing Xu, Le Lu, Yuping Duan, Hao Cheng, Na Jin, Xiaoyong Man, Shuang Zhao, Yu Wang. (2023)  
**A Novel Multi-Task Model Imitating Dermatologists for Accurate Differential Diagnosis of Skin Diseases in Clinical Images**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Clinical  
[Paper Link](http://arxiv.org/abs/2307.08308v1)  

---


**ABSTRACT**  
Skin diseases are among the most prevalent health issues, and accurate computer-aided diagnosis methods are of importance for both dermatologists and patients. However, most of the existing methods overlook the essential domain knowledge required for skin disease diagnosis. A novel multi-task model, namely DermImitFormer, is proposed to fill this gap by imitating dermatologists' diagnostic procedures and strategies. Through multi-task learning, the model simultaneously predicts body parts and lesion attributes in addition to the disease itself, enhancing diagnosis accuracy and improving diagnosis interpretability. The designed lesion selection module mimics dermatologists' zoom-in action, effectively highlighting the local lesion features from noisy backgrounds. Additionally, the presented cross-interaction module explicitly models the complicated diagnostic reasoning between body parts, lesion attributes, and diseases. To provide a more robust evaluation of the proposed method, a large-scale clinical image dataset of skin diseases with significantly more cases than existing datasets has been established. Extensive experiments on three different datasets consistently demonstrate the state-of-the-art recognition performance of the proposed approach.

{{</citation>}}


### (72/132) ShiftNAS: Improving One-shot NAS via Probability Shift (Mingyang Zhang et al., 2023)

{{<citation>}}

Mingyang Zhang, Xinyi Yu, Haodong Zhao, Linlin Ou. (2023)  
**ShiftNAS: Improving One-shot NAS via Probability Shift**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2307.08300v1)  

---


**ABSTRACT**  
One-shot Neural architecture search (One-shot NAS) has been proposed as a time-efficient approach to obtain optimal subnet architectures and weights under different complexity cases by training only once. However, the subnet performance obtained by weight sharing is often inferior to the performance achieved by retraining. In this paper, we investigate the performance gap and attribute it to the use of uniform sampling, which is a common approach in supernet training. Uniform sampling concentrates training resources on subnets with intermediate computational resources, which are sampled with high probability. However, subnets with different complexity regions require different optimal training strategies for optimal performance. To address the problem of uniform sampling, we propose ShiftNAS, a method that can adjust the sampling probability based on the complexity of subnets. We achieve this by evaluating the performance variation of subnets with different complexity and designing an architecture generator that can accurately and efficiently provide subnets with the desired complexity. Both the sampling probability and the architecture generator can be trained end-to-end in a gradient-based manner. With ShiftNAS, we can directly obtain the optimal model architecture and parameters for a given computational complexity. We evaluate our approach on multiple visual network models, including convolutional neural networks (CNNs) and vision transformers (ViTs), and demonstrate that ShiftNAS is model-agnostic. Experimental results on ImageNet show that ShiftNAS can improve the performance of one-shot NAS without additional consumption. Source codes are available at https://github.com/bestfleer/ShiftNAS.

{{</citation>}}


### (73/132) Rethinking Intersection Over Union for Small Object Detection in Few-Shot Regime (Pierre Le Jeune et al., 2023)

{{<citation>}}

Pierre Le Jeune, Anissa Mokraoui. (2023)  
**Rethinking Intersection Over Union for Small Object Detection in Few-Shot Regime**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Few-Shot, Object Detection  
[Paper Link](http://arxiv.org/abs/2307.09562v1)  

---


**ABSTRACT**  
In Few-Shot Object Detection (FSOD), detecting small objects is extremely difficult. The limited supervision cripples the localization capabilities of the models and a few pixels shift can dramatically reduce the Intersection over Union (IoU) between the ground truth and predicted boxes for small objects. To this end, we propose Scale-adaptive Intersection over Union (SIoU), a novel box similarity measure. SIoU changes with the objects' size, it is more lenient with small object shifts. We conducted a user study and SIoU better aligns than IoU with human judgment. Employing SIoU as an evaluation criterion helps to build more user-oriented models. SIoU can also be used as a loss function to prioritize small objects during training, outperforming existing loss functions. SIoU improves small object detection in the non-few-shot regime, but this setting is unrealistic in the industry as annotated detection datasets are often too expensive to acquire. Hence, our experiments mainly focus on the few-shot regime to demonstrate the superiority and versatility of SIoU loss. SIoU improves significantly FSOD performance on small objects in both natural (Pascal VOC and COCO datasets) and aerial images (DOTA and DIOR). In aerial imagery, small objects are critical and SIoU loss achieves new state-of-the-art FSOD on DOTA and DIOR.

{{</citation>}}


### (74/132) RCM-Fusion: Radar-Camera Multi-Level Fusion for 3D Object Detection (Jisong Kim et al., 2023)

{{<citation>}}

Jisong Kim, Minjae Seong, Geonho Bang, Dongsuk Kum, Jun Won Choi. (2023)  
**RCM-Fusion: Radar-Camera Multi-Level Fusion for 3D Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2307.10249v1)  

---


**ABSTRACT**  
While LiDAR sensors have been succesfully applied to 3D object detection, the affordability of radar and camera sensors has led to a growing interest in fusiong radars and cameras for 3D object detection. However, previous radar-camera fusion models have not been able to fully utilize radar information in that initial 3D proposals were generated based on the camera features only and the instance-level fusion is subsequently conducted. In this paper, we propose radar-camera multi-level fusion (RCM-Fusion), which fuses radar and camera modalities at both the feature-level and instance-level to fully utilize radar information. At the feature-level, we propose a Radar Guided BEV Encoder which utilizes radar Bird's-Eye-View (BEV) features to transform image features into precise BEV representations and then adaptively combines the radar and camera BEV features. At the instance-level, we propose a Radar Grid Point Refinement module that reduces localization error by considering the characteristics of the radar point clouds. The experiments conducted on the public nuScenes dataset demonstrate that our proposed RCM-Fusion offers 11.8% performance gain in nuScenes detection score (NDS) over the camera-only baseline model and achieves state-of-the-art performaces among radar-camera fusion methods in the nuScenes 3D object detection benchmark. Code will be made publicly available.

{{</citation>}}


### (75/132) Adversarial Attacks on Traffic Sign Recognition: A Survey (Svetlana Pavlitska et al., 2023)

{{<citation>}}

Svetlana Pavlitska, Nico Lambing, J. Marius Zöllner. (2023)  
**Adversarial Attacks on Traffic Sign Recognition: A Survey**  

---
Primary Category: cs.CV  
Categories: cs-CR, cs-CV, cs-LG, cs.CV  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2307.08278v1)  

---


**ABSTRACT**  
Traffic sign recognition is an essential component of perception in autonomous vehicles, which is currently performed almost exclusively with deep neural networks (DNNs). However, DNNs are known to be vulnerable to adversarial attacks. Several previous works have demonstrated the feasibility of adversarial attacks on traffic sign recognition models. Traffic signs are particularly promising for adversarial attack research due to the ease of performing real-world attacks using printed signs or stickers. In this work, we survey existing works performing either digital or real-world attacks on traffic sign detection and classification models. We provide an overview of the latest advancements and highlight the existing research areas that require further investigation.

{{</citation>}}


### (76/132) Hierarchical Spatiotemporal Transformers for Video Object Segmentation (Jun-Sang Yoo et al., 2023)

{{<citation>}}

Jun-Sang Yoo, Hongjae Lee, Seung-Won Jung. (2023)  
**Hierarchical Spatiotemporal Transformers for Video Object Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.08263v1)  

---


**ABSTRACT**  
This paper presents a novel framework called HST for semi-supervised video object segmentation (VOS). HST extracts image and video features using the latest Swin Transformer and Video Swin Transformer to inherit their inductive bias for the spatiotemporal locality, which is essential for temporally coherent VOS. To take full advantage of the image and video features, HST casts image and video features as a query and memory, respectively. By applying efficient memory read operations at multiple scales, HST produces hierarchical features for the precise reconstruction of object masks. HST shows effectiveness and robustness in handling challenging scenarios with occluded and fast-moving objects under cluttered backgrounds. In particular, HST-B outperforms the state-of-the-art competitors on multiple popular benchmarks, i.e., YouTube-VOS (85.0%), DAVIS 2017 (85.9%), and DAVIS 2016 (94.0%).

{{</citation>}}


### (77/132) Random Boxes Are Open-world Object Detectors (Yanghao Wang et al., 2023)

{{<citation>}}

Yanghao Wang, Zhongqi Yue, Xian-Sheng Hua, Hanwang Zhang. (2023)  
**Random Boxes Are Open-world Object Detectors**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection, Transformer  
[Paper Link](http://arxiv.org/abs/2307.08249v1)  

---


**ABSTRACT**  
We show that classifiers trained with random region proposals achieve state-of-the-art Open-world Object Detection (OWOD): they can not only maintain the accuracy of the known objects (w/ training labels), but also considerably improve the recall of unknown ones (w/o training labels). Specifically, we propose RandBox, a Fast R-CNN based architecture trained on random proposals at each training iteration, surpassing existing Faster R-CNN and Transformer based OWOD. Its effectiveness stems from the following two benefits introduced by randomness. First, as the randomization is independent of the distribution of the limited known objects, the random proposals become the instrumental variable that prevents the training from being confounded by the known objects. Second, the unbiased training encourages more proposal explorations by using our proposed matching score that does not penalize the random proposals whose prediction scores do not match the known objects. On two benchmarks: Pascal-VOC/MS-COCO and LVIS, RandBox significantly outperforms the previous state-of-the-art in all metrics. We also detail the ablations on randomization and loss designs. Codes are available at https://github.com/scuwyh2000/RandBox.

{{</citation>}}


### (78/132) Uncertainty-aware State Space Transformer for Egocentric 3D Hand Trajectory Forecasting (Wentao Bao et al., 2023)

{{<citation>}}

Wentao Bao, Lele Chen, Libing Zeng, Zhong Li, Yi Xu, Junsong Yuan, Yu Kong. (2023)  
**Uncertainty-aware State Space Transformer for Egocentric 3D Hand Trajectory Forecasting**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.08243v1)  

---


**ABSTRACT**  
Hand trajectory forecasting from egocentric views is crucial for enabling a prompt understanding of human intentions when interacting with AR/VR systems. However, existing methods handle this problem in a 2D image space which is inadequate for 3D real-world applications. In this paper, we set up an egocentric 3D hand trajectory forecasting task that aims to predict hand trajectories in a 3D space from early observed RGB videos in a first-person view. To fulfill this goal, we propose an uncertainty-aware state space Transformer (USST) that takes the merits of the attention mechanism and aleatoric uncertainty within the framework of the classical state-space model. The model can be further enhanced by the velocity constraint and visual prompt tuning (VPT) on large vision transformers. Moreover, we develop an annotation workflow to collect 3D hand trajectories with high quality. Experimental results on H2O and EgoPAT3D datasets demonstrate the superiority of USST for both 2D and 3D trajectory forecasting. The code and datasets are publicly released: https://github.com/Cogito2012/USST.

{{</citation>}}


### (79/132) ROFusion: Efficient Object Detection using Hybrid Point-wise Radar-Optical Fusion (Liu Liu et al., 2023)

{{<citation>}}

Liu Liu, Shuaifeng Zhi, Zhenhua Du, Li Liu, Xinyu Zhang, Kai Huo, Weidong Jiang. (2023)  
**ROFusion: Efficient Object Detection using Hybrid Point-wise Radar-Optical Fusion**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2307.08233v1)  

---


**ABSTRACT**  
Radars, due to their robustness to adverse weather conditions and ability to measure object motions, have served in autonomous driving and intelligent agents for years. However, Radar-based perception suffers from its unintuitive sensing data, which lack of semantic and structural information of scenes. To tackle this problem, camera and Radar sensor fusion has been investigated as a trending strategy with low cost, high reliability and strong maintenance. While most recent works explore how to explore Radar point clouds and images, rich contextual information within Radar observation are discarded. In this paper, we propose a hybrid point-wise Radar-Optical fusion approach for object detection in autonomous driving scenarios. The framework benefits from dense contextual information from both the range-doppler spectrum and images which are integrated to learn a multi-modal feature representation. Furthermore, we propose a novel local coordinate formulation, tackling the object detection task in an object-centric coordinate. Extensive results show that with the information gained from optical images, we could achieve leading performance in object detection (97.69\% recall) compared to recent state-of-the-art methods FFT-RadNet (82.86\% recall). Ablation studies verify the key design choices and practicability of our approach given machine generated imperfect detections. The code will be available at https://github.com/LiuLiu-55/ROFusion.

{{</citation>}}


### (80/132) Ada3D : Exploiting the Spatial Redundancy with Adaptive Inference for Efficient 3D Object Detection (Tianchen Zhao et al., 2023)

{{<citation>}}

Tianchen Zhao, Xuefei Ning, Ke Hong, Zhongyuan Qiu, Pu Lu, Yali Zhao, Linfeng Zhang, Lipu Zhou, Guohao Dai, Huazhong Yang, Yu Wang. (2023)  
**Ada3D : Exploiting the Spatial Redundancy with Adaptive Inference for Efficient 3D Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2307.08209v1)  

---


**ABSTRACT**  
Voxel-based methods have achieved state-of-the-art performance for 3D object detection in autonomous driving. However, their significant computational and memory costs pose a challenge for their application to resource-constrained vehicles. One reason for this high resource consumption is the presence of a large number of redundant background points in Lidar point clouds, resulting in spatial redundancy in both 3D voxel and dense BEV map representations. To address this issue, we propose an adaptive inference framework called Ada3D, which focuses on exploiting the input-level spatial redundancy. Ada3D adaptively filters the redundant input, guided by a lightweight importance predictor and the unique properties of the Lidar point cloud. Additionally, we utilize the BEV features' intrinsic sparsity by introducing the Sparsity Preserving Batch Normalization. With Ada3D, we achieve 40% reduction for 3D voxels and decrease the density of 2D BEV feature maps from 100% to 20% without sacrificing accuracy. Ada3D reduces the model computational and memory cost by 5x, and achieves 1.52x/1.45x end-to-end GPU latency and 1.5x/4.5x GPU peak memory optimization for the 3D and 2D backbone respectively.

{{</citation>}}


### (81/132) Zero-Shot Image Harmonization with Generative Model Prior (Jianqi Chen et al., 2023)

{{<citation>}}

Jianqi Chen, Zhengxia Zou, Yilan Zhang, Keyan Chen, Zhenwei Shi. (2023)  
**Zero-Shot Image Harmonization with Generative Model Prior**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2307.08182v1)  

---


**ABSTRACT**  
Recent image harmonization methods have demonstrated promising results. However, due to their heavy reliance on a large number of composite images, these works are expensive in the training phase and often fail to generalize to unseen images. In this paper, we draw lessons from human behavior and come up with a zero-shot image harmonization method. Specifically, in the harmonization process, a human mainly utilizes his long-term prior on harmonious images and makes a composite image close to that prior. To imitate that, we resort to pretrained generative models for the prior of natural images. For the guidance of the harmonization direction, we propose an Attention-Constraint Text which is optimized to well illustrate the image environments. Some further designs are introduced for preserving the foreground content structure. The resulting framework, highly consistent with human behavior, can achieve harmonious results without burdensome training. Extensive experiments have demonstrated the effectiveness of our approach, and we have also explored some interesting applications.

{{</citation>}}


## cs.CY (1)



### (82/132) Risk assessment at AGI companies: A review of popular risk assessment techniques from other safety-critical industries (Leonie Koessler et al., 2023)

{{<citation>}}

Leonie Koessler, Jonas Schuett. (2023)  
**Risk assessment at AGI companies: A review of popular risk assessment techniques from other safety-critical industries**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI, Google  
[Paper Link](http://arxiv.org/abs/2307.08823v1)  

---


**ABSTRACT**  
Companies like OpenAI, Google DeepMind, and Anthropic have the stated goal of building artificial general intelligence (AGI) - AI systems that perform as well as or better than humans on a wide variety of cognitive tasks. However, there are increasing concerns that AGI would pose catastrophic risks. In light of this, AGI companies need to drastically improve their risk management practices. To support such efforts, this paper reviews popular risk assessment techniques from other safety-critical industries and suggests ways in which AGI companies could use them to assess catastrophic risks from AI. The paper discusses three risk identification techniques (scenario analysis, fishbone method, and risk typologies and taxonomies), five risk analysis techniques (causal mapping, Delphi technique, cross-impact analysis, bow tie analysis, and system-theoretic process analysis), and two risk evaluation techniques (checklists and risk matrices). For each of them, the paper explains how they work, suggests ways in which AGI companies could use them, discusses their benefits and limitations, and makes recommendations. Finally, the paper discusses when to conduct risk assessments, when to use which technique, and how to use any of them. The reviewed techniques will be obvious to risk management professionals in other industries. And they will not be sufficient to assess catastrophic risks from AI. However, AGI companies should not skip the straightforward step of reviewing best practices from other industries.

{{</citation>}}


## cs.CL (20)



### (83/132) Comparative Performance Evaluation of Large Language Models for Extracting Molecular Interactions and Pathway Knowledge (Gilchan Park et al., 2023)

{{<citation>}}

Gilchan Park, Byung-Jun Yoon, Xihaier Luo, Vanessa López-Marrero, Patrick Johnstone, Shinjae Yoo, Francis J. Alexander. (2023)  
**Comparative Performance Evaluation of Large Language Models for Extracting Molecular Interactions and Pathway Knowledge**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.08813v1)  

---


**ABSTRACT**  
Understanding protein interactions and pathway knowledge is crucial for unraveling the complexities of living systems and investigating the underlying mechanisms of biological functions and complex diseases. While existing databases provide curated biological data from literature and other sources, they are often incomplete and their maintenance is labor-intensive, necessitating alternative approaches. In this study, we propose to harness the capabilities of large language models to address these issues by automatically extracting such knowledge from the relevant scientific literature. Toward this goal, in this work, we investigate the effectiveness of different large language models in tasks that involve recognizing protein interactions, pathways, and gene regulatory relations. We thoroughly evaluate the performance of various models, highlight the significant findings, and discuss both the future opportunities and the remaining challenges associated with this approach. The code and data are available at: https://github.com/boxorange/BioIE-LLM

{{</citation>}}


### (84/132) A mixed policy to improve performance of language models on math problems (Gang Chen, 2023)

{{<citation>}}

Gang Chen. (2023)  
**A mixed policy to improve performance of language models on math problems**  

---
Primary Category: cs.CL  
Categories: 68T10, I-2-6, cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2307.08767v1)  

---


**ABSTRACT**  
When to solve math problems, most language models take a sampling strategy to predict next word according conditional probabilities. In the math reasoning step, it may generate wrong answer. Considering math problems are deterministic, we propose a mixed policy exploration approach to solve math problems with reinforcement learning. In peculiar, we propose a two level token exploration policy: the abstract level explores next token with probability and the second level is deterministic. Specifically, the abstract level policy will decide whether the token is operator or operand with probability sampling, while the second level is deterministic to select next token with the highest score in a greedy way. We test our method on GSM8K dataset with GPT-2 model, and demonstrate more than $2\%$ performance gain. Our implementation is available at https://github.com/vividitytech/math_lm_rl.

{{</citation>}}


### (85/132) AlpaGasus: Training A Better Alpaca with Fewer Data (Lichang Chen et al., 2023)

{{<citation>}}

Lichang Chen, Shiyang Li, Jun Yan, Hai Wang, Kalpa Gunaratna, Vikas Yadav, Zheng Tang, Vijay Srinivasan, Tianyi Zhou, Heng Huang, Hongxia Jin. (2023)  
**AlpaGasus: Training A Better Alpaca with Fewer Data**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2307.08701v1)  

---


**ABSTRACT**  
Large language models~(LLMs) obtain instruction-following capability through instruction-finetuning (IFT) on supervised instruction/response data. However, widely used IFT datasets (e.g., Alpaca's 52k data) surprisingly contain many low-quality instances with incorrect or irrelevant responses, which are misleading and detrimental to IFT. In this paper, we propose a simple and effective data selection strategy that automatically identifies and removes low-quality data using a strong LLM (e.g., ChatGPT). To this end, we introduce AlpaGasus, which is finetuned on only 9k high-quality data filtered from the 52k Alpaca data. AlpaGasus significantly outperforms the original Alpaca as evaluated by GPT-4 on multiple test sets and its 13B variant matches $>90\%$ performance of its teacher LLM (i.e., Text-Davinci-003) on test tasks. It also provides 5.7x faster training, reducing the training time for a 7B variant from 80 minutes (for Alpaca) to 14 minutes \footnote{We apply IFT for the same number of epochs as Alpaca(7B) but on fewer data, using 4$\times$NVIDIA A100 (80GB) GPUs and following the original Alpaca setting and hyperparameters.}. Overall, AlpaGasus demonstrates a novel data-centric IFT paradigm that can be generally applied to instruction-tuning data, leading to faster training and better instruction-following models. Our project page is available at: \url{https://lichang-chen.github.io/AlpaGasus/}.

{{</citation>}}


### (86/132) COLLIE: Systematic Construction of Constrained Text Generation Tasks (Shunyu Yao et al., 2023)

{{<citation>}}

Shunyu Yao, Howard Chen, Austin W. Hanjie, Runzhe Yang, Karthik Narasimhan. (2023)  
**COLLIE: Systematic Construction of Constrained Text Generation Tasks**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: GPT, GPT-4, Text Generation  
[Paper Link](http://arxiv.org/abs/2307.08689v1)  

---


**ABSTRACT**  
Text generation under constraints have seen increasing interests in natural language processing, especially with the rapidly improving capabilities of large language models. However, existing benchmarks for constrained generation usually focus on fixed constraint types (e.g.,generate a sentence containing certain words) that have proved to be easy for state-of-the-art models like GPT-4. We present COLLIE, a grammar-based framework that allows the specification of rich, compositional constraints with diverse generation levels (word, sentence, paragraph, passage) and modeling challenges (e.g.,language understanding, logical reasoning, counting, semantic planning). We also develop tools for automatic extraction of task instances given a constraint structure and a raw text corpus. Using COLLIE, we compile the COLLIE-v1 dataset with 2080 instances comprising 13 constraint structures. We perform systematic experiments across five state-of-the-art instruction-tuned language models and analyze their performances to reveal shortcomings. COLLIE is designed to be extensible and lightweight, and we hope the community finds it useful to develop more complex constraints and evaluations in the future.

{{</citation>}}


### (87/132) Do Models Explain Themselves? Counterfactual Simulatability of Natural Language Explanations (Yanda Chen et al., 2023)

{{<citation>}}

Yanda Chen, Ruiqi Zhong, Narutatsu Ri, Chen Zhao, He He, Jacob Steinhardt, Zhou Yu, Kathleen McKeown. (2023)  
**Do Models Explain Themselves? Counterfactual Simulatability of Natural Language Explanations**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2307.08678v1)  

---


**ABSTRACT**  
Large language models (LLMs) are trained to imitate humans to explain human decisions. However, do LLMs explain themselves? Can they help humans build mental models of how LLMs process different inputs? To answer these questions, we propose to evaluate $\textbf{counterfactual simulatability}$ of natural language explanations: whether an explanation can enable humans to precisely infer the model's outputs on diverse counterfactuals of the explained input. For example, if a model answers "yes" to the input question "Can eagles fly?" with the explanation "all birds can fly", then humans would infer from the explanation that it would also answer "yes" to the counterfactual input "Can penguins fly?". If the explanation is precise, then the model's answer should match humans' expectations.   We implemented two metrics based on counterfactual simulatability: precision and generality. We generated diverse counterfactuals automatically using LLMs. We then used these metrics to evaluate state-of-the-art LLMs (e.g., GPT-4) on two tasks: multi-hop factual reasoning and reward modeling. We found that LLM's explanations have low precision and that precision does not correlate with plausibility. Therefore, naively optimizing human approvals (e.g., RLHF) may not be a sufficient solution.

{{</citation>}}


### (88/132) Multilingual Speech-to-Speech Translation into Multiple Target Languages (Hongyu Gong et al., 2023)

{{<citation>}}

Hongyu Gong, Ning Dong, Sravya Popuri, Vedanuj Goswami, Ann Lee, Juan Pino. (2023)  
**Multilingual Speech-to-Speech Translation into Multiple Target Languages**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: Multilingual  
[Paper Link](http://arxiv.org/abs/2307.08655v1)  

---


**ABSTRACT**  
Speech-to-speech translation (S2ST) enables spoken communication between people talking in different languages. Despite a few studies on multilingual S2ST, their focus is the multilinguality on the source side, i.e., the translation from multiple source languages to one target language. We present the first work on multilingual S2ST supporting multiple target languages. Leveraging recent advance in direct S2ST with speech-to-unit and vocoder, we equip these key components with multilingual capability. Speech-to-masked-unit (S2MU) is the multilingual extension of S2U, which applies masking to units which don't belong to the given target language to reduce the language interference. We also propose multilingual vocoder which is trained with language embedding and the auxiliary loss of language identification. On benchmark translation testsets, our proposed multilingual model shows superior performance than bilingual models in the translation from English into $16$ target languages.

{{</citation>}}


### (89/132) Retentive Network: A Successor to Transformer for Large Language Models (Yutao Sun et al., 2023)

{{<citation>}}

Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, Furu Wei. (2023)  
**Retentive Network: A Successor to Transformer for Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2307.08621v2)  

---


**ABSTRACT**  
In this work, we propose Retentive Network (RetNet) as a foundation architecture for large language models, simultaneously achieving training parallelism, low-cost inference, and good performance. We theoretically derive the connection between recurrence and attention. Then we propose the retention mechanism for sequence modeling, which supports three computation paradigms, i.e., parallel, recurrent, and chunkwise recurrent. Specifically, the parallel representation allows for training parallelism. The recurrent representation enables low-cost $O(1)$ inference, which improves decoding throughput, latency, and GPU memory without sacrificing performance. The chunkwise recurrent representation facilitates efficient long-sequence modeling with linear complexity, where each chunk is encoded parallelly while recurrently summarizing the chunks. Experimental results on language modeling show that RetNet achieves favorable scaling results, parallel training, low-cost deployment, and efficient inference. The intriguing properties make RetNet a strong successor to Transformer for large language models. Code will be available at https://aka.ms/retnet.

{{</citation>}}


### (90/132) Syntax-Aware Complex-Valued Neural Machine Translation (Yang Liu et al., 2023)

{{<citation>}}

Yang Liu, Yuexian Hou. (2023)  
**Syntax-Aware Complex-Valued Neural Machine Translation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BLEU, Machine Translation, Seq2Seq  
[Paper Link](http://arxiv.org/abs/2307.08586v1)  

---


**ABSTRACT**  
Syntax has been proven to be remarkably effective in neural machine translation (NMT). Previous models obtained syntax information from syntactic parsing tools and integrated it into NMT models to improve translation performance. In this work, we propose a method to incorporate syntax information into a complex-valued Encoder-Decoder architecture. The proposed model jointly learns word-level and syntax-level attention scores from the source side to the target side using an attention mechanism. Importantly, it is not dependent on specific network architectures and can be directly integrated into any existing sequence-to-sequence (Seq2Seq) framework. The experimental results demonstrate that the proposed method can bring significant improvements in BLEU scores on two datasets. In particular, the proposed method achieves a greater improvement in BLEU scores in translation tasks involving language pairs with significant syntactic differences.

{{</citation>}}


### (91/132) Discovering collective narratives shifts in online discussions (Wanying Zhao et al., 2023)

{{<citation>}}

Wanying Zhao, Fiona Guo, Kristina Lerman, Yong-Yeol Ahn. (2023)  
**Discovering collective narratives shifts in online discussions**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SI, cs.CL  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2307.08541v1)  

---


**ABSTRACT**  
Narrative is a foundation of human cognition and decision making. Because narratives play a crucial role in societal discourses and spread of misinformation and because of the pervasive use of social media, the narrative dynamics on social media can have profound societal impact. Yet, systematic and computational understanding of online narratives faces critical challenge of the scale and dynamics; how can we reliably and automatically extract narratives from massive amount of texts? How do narratives emerge, spread, and die? Here, we propose a systematic narrative discovery framework that fill this gap by combining change point detection, semantic role labeling (SRL), and automatic aggregation of narrative fragments into narrative networks. We evaluate our model with synthetic and empirical data two-Twitter corpora about COVID-19 and 2017 French Election. Results demonstrate that our approach can recover major narrative shifts that correspond to the major events.

{{</citation>}}


### (92/132) Latent Jailbreak: A Benchmark for Evaluating Text Safety and Output Robustness of Large Language Models (Huachuan Qiu et al., 2023)

{{<citation>}}

Huachuan Qiu, Shuai Zhang, Anqi Li, Hongliang He, Zhenzhong Lan. (2023)  
**Latent Jailbreak: A Benchmark for Evaluating Text Safety and Output Robustness of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Language Model, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.08487v1)  

---


**ABSTRACT**  
Researchers have invested considerable effort into ensuring that large language models (LLMs) align with human values, using various training techniques, such as instruction tuning and Reinforcement Learning from Human or AI Feedback (RLHF/RLAIF), to guard against text unsafety. However, these defenses remain incredibly vulnerable to some jailbreak attacks, which can cause the model to become overly defensive to sensitive topics or still generate harmful content, leaving the model performance particularly fragile. Therefore, to comprehensively study text safety and output robustness, we propose a latent jailbreak prompt dataset, each involving malicious instruction embedding. Specifically, we instruct the model to complete a regular task, such as translation, where the text to be translated contains malicious instructions. To further analyze the safety and robustness, we design a hierarchical annotation framework. We present a systematic analysis of the safety and robustness of LLMs concerning the position of explicit normal instructions, word replacement (verbs in explicit normal instructions, target groups in malicious instructions, cue words in malicious instructions), and instruction replacement (different explicit normal instructions). Our results show that current LLMs not only have a preference for certain instruction verbs, but also exhibit different jailbreak rates for different instruction verbs in explicit normal instructions. In other words, the probability of generating unsafe content by the model will be reinforced to varying degrees depending on the instruction verb in explicit normal instructions. Code and data are available at https://github.com/qiuhuachuan/latent-jailbreak.

{{</citation>}}


### (93/132) Improving End-to-End Speech Translation by Imitation-Based Knowledge Distillation with Synthetic Transcripts (Rebekka Hubert et al., 2023)

{{<citation>}}

Rebekka Hubert, Artem Sokolov, Stefan Riezler. (2023)  
**Improving End-to-End Speech Translation by Imitation-Based Knowledge Distillation with Synthetic Transcripts**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BLEU, Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2307.08426v1)  

---


**ABSTRACT**  
End-to-end automatic speech translation (AST) relies on data that combines audio inputs with text translation outputs. Previous work used existing large parallel corpora of transcriptions and translations in a knowledge distillation (KD) setup to distill a neural machine translation (NMT) into an AST student model. While KD allows using larger pretrained models, the reliance of previous KD approaches on manual audio transcripts in the data pipeline restricts the applicability of this framework to AST. We present an imitation learning approach where a teacher NMT system corrects the errors of an AST student without relying on manual transcripts. We show that the NMT teacher can recover from errors in automatic transcriptions and is able to correct erroneous translations of the AST student, leading to improvements of about 4 BLEU points over the standard AST end-to-end baseline on the English-German CoVoST-2 and MuST-C datasets, respectively. Code and data are publicly available.\footnote{\url{https://github.com/HubReb/imitkd_ast/releases/tag/v1.1}}

{{</citation>}}


### (94/132) Enhancing Supervised Learning with Contrastive Markings in Neural Machine Translation Training (Nathaniel Berger et al., 2023)

{{<citation>}}

Nathaniel Berger, Miriam Exel, Matthias Huck, Stefan Riezler. (2023)  
**Enhancing Supervised Learning with Contrastive Markings in Neural Machine Translation Training**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Machine Translation  
[Paper Link](http://arxiv.org/abs/2307.08416v1)  

---


**ABSTRACT**  
Supervised learning in Neural Machine Translation (NMT) typically follows a teacher forcing paradigm where reference tokens constitute the conditioning context in the model's prediction, instead of its own previous predictions. In order to alleviate this lack of exploration in the space of translations, we present a simple extension of standard maximum likelihood estimation by a contrastive marking objective. The additional training signals are extracted automatically from reference translations by comparing the system hypothesis against the reference, and used for up/down-weighting correct/incorrect tokens. The proposed new training procedure requires one additional translation pass over the training set per epoch, and does not alter the standard inference setup. We show that training with contrastive markings yields improvements on top of supervised learning, and is especially useful when learning from postedits where contrastive markings indicate human error corrections to the original hypotheses. Code is publicly released.

{{</citation>}}


### (95/132) On the application of Large Language Models for language teaching and assessment technology (Andrew Caines et al., 2023)

{{<citation>}}

Andrew Caines, Luca Benedetto, Shiva Taslimipoor, Christopher Davis, Yuan Gao, Oeistein Andersen, Zheng Yuan, Mark Elliott, Russell Moore, Christopher Bryant, Marek Rei, Helen Yannakoudakis, Andrew Mullooly, Diane Nicholls, Paula Buttery. (2023)  
**On the application of Large Language Models for language teaching and assessment technology**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: AI, GPT, GPT-4, Language Model, PaLM  
[Paper Link](http://arxiv.org/abs/2307.08393v1)  

---


**ABSTRACT**  
The recent release of very large language models such as PaLM and GPT-4 has made an unprecedented impact in the popular media and public consciousness, giving rise to a mixture of excitement and fear as to their capabilities and potential uses, and shining a light on natural language processing research which had not previously received so much attention. The developments offer great promise for education technology, and in this paper we look specifically at the potential for incorporating large language models in AI-driven language teaching and assessment systems. We consider several research areas and also discuss the risks and ethical considerations surrounding generative AI in education technology for language learners. Overall we find that larger language models offer improvements over previous models in text generation, opening up routes toward content generation which had not previously been plausible. For text generation they must be prompted carefully and their outputs may need to be reshaped before they are ready for use. For automated grading and grammatical error correction, tasks whose progress is checked on well-known benchmarks, early investigations indicate that large language models on their own do not improve on state-of-the-art results according to standard evaluation metrics. For grading it appears that linguistic features established in the literature should still be used for best performance, and for error correction it may be that the models can offer alternative feedback styles which are not measured sensitively with existing methods. In all cases, there is work to be done to experiment with the inclusion of large language models in education technology for language learners, in order to properly understand and report on their capacities and limitations, and to ensure that foreseeable risks such as misinformation and harmful bias are mitigated.

{{</citation>}}


### (96/132) Legal Syllogism Prompting: Teaching Large Language Models for Legal Judgment Prediction (Cong Jiang et al., 2023)

{{<citation>}}

Cong Jiang, Xiaolei Yang. (2023)  
**Legal Syllogism Prompting: Teaching Large Language Models for Legal Judgment Prediction**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, GPT, Language Model, Legal  
[Paper Link](http://arxiv.org/abs/2307.08321v1)  

---


**ABSTRACT**  
Legal syllogism is a form of deductive reasoning commonly used by legal professionals to analyze cases. In this paper, we propose legal syllogism prompting (LoT), a simple prompting method to teach large language models (LLMs) for legal judgment prediction. LoT teaches only that in the legal syllogism the major premise is law, the minor premise is the fact, and the conclusion is judgment. Then the models can produce a syllogism reasoning of the case and give the judgment without any learning, fine-tuning, or examples. On CAIL2018, a Chinese criminal case dataset, we performed zero-shot judgment prediction experiments with GPT-3 models. Our results show that LLMs with LoT achieve better performance than the baseline and chain of thought prompting, the state-of-art prompting method on diverse reasoning tasks. LoT enables the model to concentrate on the key information relevant to the judgment and to correctly understand the legal meaning of acts, as compared to other methods. Our method enables LLMs to predict judgment along with law articles and justification, which significantly enhances the explainability of models.

{{</citation>}}


### (97/132) CoAD: Automatic Diagnosis through Symptom and Disease Collaborative Generation (Huimin Wang et al., 2023)

{{<citation>}}

Huimin Wang, Wai-Chung Kwan, Kam-Fai Wong, Yefeng Zheng. (2023)  
**CoAD: Automatic Diagnosis through Symptom and Disease Collaborative Generation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Transformer  
[Paper Link](http://arxiv.org/abs/2307.08290v1)  

---


**ABSTRACT**  
Automatic diagnosis (AD), a critical application of AI in healthcare, employs machine learning techniques to assist doctors in gathering patient symptom information for precise disease diagnosis. The Transformer-based method utilizes an input symptom sequence, predicts itself through auto-regression, and employs the hidden state of the final symptom to determine the disease. Despite its simplicity and superior performance demonstrated, a decline in disease diagnosis accuracy is observed caused by 1) a mismatch between symptoms observed during training and generation, and 2) the effect of different symptom orders on disease prediction. To address the above obstacles, we introduce the CoAD, a novel disease and symptom collaborative generation framework, which incorporates several key innovations to improve AD: 1) aligning sentence-level disease labels with multiple possible symptom inquiry steps to bridge the gap between training and generation; 2) expanding symptom labels for each sub-sequence of symptoms to enhance annotation and eliminate the effect of symptom order; 3) developing a repeated symptom input schema to effectively and efficiently learn the expanded disease and symptom labels. We evaluate the CoAD framework using four datasets, including three public and one private, and demonstrate that it achieves an average 2.3% improvement over previous state-of-the-art results in automatic disease diagnosis. For reproducibility, we release the code and data at https://github.com/KwanWaiChung/coad.

{{</citation>}}


### (98/132) Automated Action Model Acquisition from Narrative Texts (Ruiqi Li et al., 2023)

{{<citation>}}

Ruiqi Li, Leyang Cui, Songtuan Lin, Patrik Haslum. (2023)  
**Automated Action Model Acquisition from Narrative Texts**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-IR, cs-LG, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.10247v1)  

---


**ABSTRACT**  
Action models, which take the form of precondition/effect axioms, facilitate causal and motivational connections between actions for AI agents. Action model acquisition has been identified as a bottleneck in the application of planning technology, especially within narrative planning. Acquiring action models from narrative texts in an automated way is essential, but challenging because of the inherent complexities of such texts. We present NaRuto, a system that extracts structured events from narrative text and subsequently generates planning-language-style action models based on predictions of commonsense event relations, as well as textual contradictions and similarities, in an unsupervised manner. Experimental results in classical narrative planning domains show that NaRuto can generate action models of significantly better quality than existing fully automated methods, and even on par with those of semi-automated methods.

{{</citation>}}


### (99/132) ChatGPT is Good but Bing Chat is Better for Vietnamese Students (Xuan-Quy Dao et al., 2023)

{{<citation>}}

Xuan-Quy Dao, Ngoc-Bich Le. (2023)  
**ChatGPT is Good but Bing Chat is Better for Vietnamese Students**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, GPT-3.5, GPT-4, Microsoft  
[Paper Link](http://arxiv.org/abs/2307.08272v2)  

---


**ABSTRACT**  
This study examines the efficacy of two SOTA large language models (LLMs), namely ChatGPT and Microsoft Bing Chat (BingChat), in catering to the needs of Vietnamese students. Although ChatGPT exhibits proficiency in multiple disciplines, Bing Chat emerges as the more advantageous option. We conduct a comparative analysis of their academic achievements in various disciplines, encompassing mathematics, literature, English language, physics, chemistry, biology, history, geography, and civic education. The results of our study suggest that BingChat demonstrates superior performance compared to ChatGPT across a wide range of subjects, with the exception of literature, where ChatGPT exhibits better performance. Additionally, BingChat utilizes the more advanced GPT-4 technology in contrast to ChatGPT, which is built upon GPT-3.5. This allows BingChat to improve to comprehension, reasoning and generation of creative and informative text. Moreover, the fact that BingChat is accessible in Vietnam and its integration of hyperlinks and citations within responses serve to reinforce its superiority. In our analysis, it is evident that while ChatGPT exhibits praiseworthy qualities, BingChat presents a more apdated solutions for Vietnamese students.

{{</citation>}}


### (100/132) PAT: Parallel Attention Transformer for Visual Question Answering in Vietnamese (Nghia Hieu Nguyen et al., 2023)

{{<citation>}}

Nghia Hieu Nguyen, Kiet Van Nguyen. (2023)  
**PAT: Parallel Attention Transformer for Visual Question Answering in Vietnamese**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Attention, LSTM, QA, Question Answering, Transformer  
[Paper Link](http://arxiv.org/abs/2307.08247v1)  

---


**ABSTRACT**  
We present in this paper a novel scheme for multimodal learning named the Parallel Attention mechanism. In addition, to take into account the advantages of grammar and context in Vietnamese, we propose the Hierarchical Linguistic Features Extractor instead of using an LSTM network to extract linguistic features. Based on these two novel modules, we introduce the Parallel Attention Transformer (PAT), achieving the best accuracy compared to all baselines on the benchmark ViVQA dataset and other SOTA methods including SAAA and MCAN.

{{</citation>}}


### (101/132) BASS: Block-wise Adaptation for Speech Summarization (Roshan Sharma et al., 2023)

{{<citation>}}

Roshan Sharma, Kenneth Zheng, Siddhant Arora, Shinji Watanabe, Rita Singh, Bhiksha Raj. (2023)  
**BASS: Block-wise Adaptation for Speech Summarization**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: Summarization  
[Paper Link](http://arxiv.org/abs/2307.08217v1)  

---


**ABSTRACT**  
End-to-end speech summarization has been shown to improve performance over cascade baselines. However, such models are difficult to train on very large inputs (dozens of minutes or hours) owing to compute restrictions and are hence trained with truncated model inputs. Truncation leads to poorer models, and a solution to this problem rests in block-wise modeling, i.e., processing a portion of the input frames at a time. In this paper, we develop a method that allows one to train summarization models on very long sequences in an incremental manner. Speech summarization is realized as a streaming process, where hypothesis summaries are updated every block based on new acoustic information. We devise and test strategies to pass semantic context across the blocks. Experiments on the How2 dataset demonstrate that the proposed block-wise training method improves by 3 points absolute on ROUGE-L over a truncated input baseline.

{{</citation>}}


### (102/132) Mini-Giants: 'Small' Language Models and Open Source Win-Win (Zhengping Zhou et al., 2023)

{{<citation>}}

Zhengping Zhou, Lezhi Li, Xinxi Chen, Andy Li. (2023)  
**Mini-Giants: 'Small' Language Models and Open Source Win-Win**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2307.08189v1)  

---


**ABSTRACT**  
ChatGPT is phenomenal. However, it is prohibitively expensive to train and refine such giant models. Fortunately, small language models are flourishing and becoming more and more competent. We call them "mini-giants". We argue that open source community like Kaggle and mini-giants will win-win in many ways, technically, ethically and socially. In this article, we present a brief yet rich background, discuss how to attain small language models, present a comparative study of small language models and a brief discussion of evaluation methods, discuss the application scenarios where small language models are most needed in the real world, and conclude with discussion and outlook.

{{</citation>}}


## cs.IR (3)



### (103/132) An Exploration Study of Mixed-initiative Query Reformulation in Conversational Passage Retrieval (Dayu Yang et al., 2023)

{{<citation>}}

Dayu Yang, Yue Zhang, Hui Fang. (2023)  
**An Exploration Study of Mixed-initiative Query Reformulation in Conversational Passage Retrieval**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: BERT, T5  
[Paper Link](http://arxiv.org/abs/2307.08803v1)  

---


**ABSTRACT**  
In this paper, we report our methods and experiments for the TREC Conversational Assistance Track (CAsT) 2022. In this work, we aim to reproduce multi-stage retrieval pipelines and explore one of the potential benefits of involving mixed-initiative interaction in conversational passage retrieval scenarios: reformulating raw queries. Before the first ranking stage of a multi-stage retrieval pipeline, we propose a mixed-initiative query reformulation module, which achieves query reformulation based on the mixed-initiative interaction between the users and the system, as the replacement for the neural reformulation method. Specifically, we design an algorithm to generate appropriate questions related to the ambiguities in raw queries, and another algorithm to reformulate raw queries by parsing users' feedback and incorporating it into the raw query. For the first ranking stage of our multi-stage pipelines, we adopt a sparse ranking function: BM25, and a dense retrieval method: TCT-ColBERT. For the second-ranking step, we adopt a pointwise reranker: MonoT5, and a pairwise reranker: DuoT5. Experiments on both TREC CAsT 2021 and TREC CAsT 2022 datasets show the effectiveness of our mixed-initiative-based query reformulation method on improving retrieval performance compared with two popular reformulators: a neural reformulator: CANARD-T5 and a rule-based reformulator: historical query reformulator(HQE).

{{</citation>}}


### (104/132) Imposing Consistency Properties on Blackbox Systems with Applications to SVD-Based Recommender Systems (Tung Nguyen et al., 2023)

{{<citation>}}

Tung Nguyen, Jeffrey Uhlmann. (2023)  
**Imposing Consistency Properties on Blackbox Systems with Applications to SVD-Based Recommender Systems**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.08760v1)  

---


**ABSTRACT**  
In this paper we discuss pre- and post-processing methods to induce desired consistency and/or invariance properties in blackbox systems, e.g., AI-based. We demonstrate our approach in the context of blackbox SVD-based matrix-completion methods commonly used in recommender system (RS) applications. We provide empirical results showing that enforcement of unit-consistency and shift-consistency, which have provable RS-relevant properties relating to robustness and fairness, also lead to improved performance according to generic RMSE and MAE performance metrics, irrespective of the initial chosen hyperparameter.

{{</citation>}}


### (105/132) Soft Prompt Tuning for Augmenting Dense Retrieval with Large Language Models (Zhiyuan Peng et al., 2023)

{{<citation>}}

Zhiyuan Peng, Xuyang Wu, Yi Fang. (2023)  
**Soft Prompt Tuning for Augmenting Dense Retrieval with Large Language Models**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.08303v1)  

---


**ABSTRACT**  
Dense retrieval (DR) converts queries and documents into dense embeddings and measures the similarity between queries and documents in vector space. One of the challenges in DR is the lack of domain-specific training data. While DR models can learn from large-scale public datasets like MS MARCO through transfer learning, evidence shows that not all DR models and domains can benefit from transfer learning equally. Recently, some researchers have resorted to large language models (LLMs) to improve the zero-shot and few-shot DR models. However, the hard prompts or human-written prompts utilized in these works cannot guarantee the good quality of generated weak queries. To tackle this, we propose soft prompt tuning for augmenting DR (SPTAR): For each task, we leverage soft prompt-tuning to optimize a task-specific soft prompt on limited ground truth data and then prompt the LLMs to tag unlabeled documents with weak queries, yielding enough weak document-query pairs to train task-specific dense retrievers. We design a filter to select high-quality example document-query pairs in the prompt to further improve the quality of weak tagged queries. To the best of our knowledge, there is no prior work utilizing soft prompt tuning to augment DR models. The experiments demonstrate that SPTAR outperforms the unsupervised baselines BM25 and the recently proposed LLMs-based augmentation method for DR.

{{</citation>}}


## cs.GL (1)



### (106/132) AI empowering research: 10 ways how science can benefit from AI (César França, 2023)

{{<citation>}}

César França. (2023)  
**AI empowering research: 10 ways how science can benefit from AI**  

---
Primary Category: cs.GL  
Categories: cs-AI, cs-GL, cs.GL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.10265v1)  

---


**ABSTRACT**  
This article explores the transformative impact of artificial intelligence (AI) on scientific research. It highlights ten ways in which AI is revolutionizing the work of scientists, including powerful referencing tools, improved understanding of research problems, enhanced research question generation, optimized research design, stub data generation, data transformation, advanced data analysis, and AI-assisted reporting. While AI offers numerous benefits, challenges such as bias, privacy concerns, and the need for human-AI collaboration must be considered. The article emphasizes that AI can augment human creativity in science but not replace it.

{{</citation>}}


## eess.SY (1)



### (107/132) A Multiobjective Reinforcement Learning Framework for Microgrid Energy Management (M. Vivienne Liu et al., 2023)

{{<citation>}}

M. Vivienne Liu, Patrick M. Reed, David Gold, Garret Quist, C. Lindsay Anderson. (2023)  
**A Multiobjective Reinforcement Learning Framework for Microgrid Energy Management**  

---
Primary Category: eess.SY  
Categories: cs-LG, cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.08692v1)  

---


**ABSTRACT**  
The emergence of microgrids (MGs) has provided a promising solution for decarbonizing and decentralizing the power grid, mitigating the challenges posed by climate change. However, MG operations often involve considering multiple objectives that represent the interests of different stakeholders, leading to potentially complex conflicts. To tackle this issue, we propose a novel multi-objective reinforcement learning framework that explores the high-dimensional objective space and uncovers the tradeoffs between conflicting objectives. This framework leverages exogenous information and capitalizes on the data-driven nature of reinforcement learning, enabling the training of a parametric policy without the need for long-term forecasts or knowledge of the underlying uncertainty distribution. The trained policies exhibit diverse, adaptive, and coordinative behaviors with the added benefit of providing interpretable insights on the dynamics of their information use. We employ this framework on the Cornell University MG (CU-MG), which is a combined heat and power MG, to evaluate its effectiveness. The results demonstrate performance improvements in all objectives considered compared to the status quo operations and offer more flexibility in navigating complex operational tradeoffs.

{{</citation>}}


## eess.IV (4)



### (108/132) Neural Image Compression: Generalization, Robustness, and Spectral Biases (Kelsey Lieberman et al., 2023)

{{<citation>}}

Kelsey Lieberman, James Diffenderfer, Charles Godfrey, Bhavya Kailkhura. (2023)  
**Neural Image Compression: Generalization, Robustness, and Spectral Biases**  

---
Primary Category: eess.IV  
Categories: cs-LG, eess-IV, eess.IV  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2307.08657v1)  

---


**ABSTRACT**  
Recent neural image compression (NIC) advances have produced models which are starting to outperform traditional codecs. While this has led to growing excitement about using NIC in real-world applications, the successful adoption of any machine learning system in the wild requires it to generalize (and be robust) to unseen distribution shifts at deployment. Unfortunately, current research lacks comprehensive datasets and informative tools to evaluate and understand NIC performance in real-world settings. To bridge this crucial gap, first, this paper presents a comprehensive benchmark suite to evaluate the out-of-distribution (OOD) performance of image compression methods. Specifically, we provide CLIC-C and Kodak-C by introducing 15 corruptions to popular CLIC and Kodak benchmarks. Next, we propose spectrally inspired inspection tools to gain deeper insight into errors introduced by image compression methods as well as their OOD performance. We then carry out a detailed performance comparison of a classical codec with several NIC variants, revealing intriguing findings that challenge our current understanding of the strengths and limitations of NIC. Finally, we corroborate our empirical findings with theoretical analysis, providing an in-depth view of the OOD performance of NIC and its dependence on the spectral properties of the data. Our benchmarks, spectral inspection tools, and findings provide a crucial bridge to the real-world adoption of NIC. We hope that our work will propel future efforts in designing robust and generalizable NIC methods. Code and data will be made available at https://github.com/klieberman/ood_nic.

{{</citation>}}


### (109/132) Study of Vision Transformers for Covid-19 Detection from Chest X-rays (Sandeep Angara et al., 2023)

{{<citation>}}

Sandeep Angara, Sharath Thirunagaru. (2023)  
**Study of Vision Transformers for Covid-19 Detection from Chest X-rays**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.09402v1)  

---


**ABSTRACT**  
The COVID-19 pandemic has led to a global health crisis, highlighting the need for rapid and accurate virus detection. This research paper examines transfer learning with vision transformers for COVID-19 detection, known for its excellent performance in image recognition tasks. We leverage the capability of Vision Transformers to capture global context and learn complex patterns from chest X-ray images. In this work, we explored the recent state-of-art transformer models to detect Covid-19 using CXR images such as vision transformer (ViT), Swin-transformer, Max vision transformer (MViT), and Pyramid Vision transformer (PVT). Through the utilization of transfer learning with IMAGENET weights, the models achieved an impressive accuracy range of 98.75% to 99.5%. Our experiments demonstrate that Vision Transformers achieve state-of-the-art performance in COVID-19 detection, outperforming traditional methods and even Convolutional Neural Networks (CNNs). The results highlight the potential of Vision Transformers as a powerful tool for COVID-19 detection, with implications for improving the efficiency and accuracy of screening and diagnosis in clinical settings.

{{</citation>}}


### (110/132) EGE-UNet: an Efficient Group Enhanced UNet for skin lesion segmentation (Jiacheng Ruan et al., 2023)

{{<citation>}}

Jiacheng Ruan, Mingye Xie, Jingsheng Gao, Ting Liu, Yuzhuo Fu. (2023)  
**EGE-UNet: an Efficient Group Enhanced UNet for skin lesion segmentation**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2307.08473v1)  

---


**ABSTRACT**  
Transformer and its variants have been widely used for medical image segmentation. However, the large number of parameter and computational load of these models make them unsuitable for mobile health applications. To address this issue, we propose a more efficient approach, the Efficient Group Enhanced UNet (EGE-UNet). We incorporate a Group multi-axis Hadamard Product Attention module (GHPA) and a Group Aggregation Bridge module (GAB) in a lightweight manner. The GHPA groups input features and performs Hadamard Product Attention mechanism (HPA) on different axes to extract pathological information from diverse perspectives. The GAB effectively fuses multi-scale information by grouping low-level features, high-level features, and a mask generated by the decoder at each stage. Comprehensive experiments on the ISIC2017 and ISIC2018 datasets demonstrate that EGE-UNet outperforms existing state-of-the-art methods. In short, compared to the TransFuse, our model achieves superior segmentation performance while reducing parameter and computation costs by 494x and 160x, respectively. Moreover, to our best knowledge, this is the first model with a parameter count limited to just 50KB. Our code is available at https://github.com/JCruan519/EGE-UNet.

{{</citation>}}


### (111/132) Domain Adaptation using Silver Standard Masks for Lateral Ventricle Segmentation in FLAIR MRI (Owen Crystal et al., 2023)

{{<citation>}}

Owen Crystal, Pejman J. Maralani, Sandra Black, Alan R. Moody, April Khademi. (2023)  
**Domain Adaptation using Silver Standard Masks for Lateral Ventricle Segmentation in FLAIR MRI**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.08456v1)  

---


**ABSTRACT**  
Lateral ventricular volume (LVV) is an important biomarker for clinical investigation. We present the first transfer learning-based LVV segmentation method for fluid-attenuated inversion recovery (FLAIR) MRI. To mitigate covariate shifts between source and target domains, this work proposes an domain adaptation method that optimizes performance on three target datasets. Silver standard (SS) masks were generated from the target domain using a novel conventional image processing ventricular segmentation algorithm and used to supplement the gold standard (GS) data from the source domain, Canadian Atherosclerosis Imaging Network (CAIN). Four models were tested on held-out test sets from four datasets: 1) SS+GS: trained on target SS masks and fine-tuned on source GS masks, 2) GS+SS: trained on source GS masks and fine-tuned on target SS masks, 3) trained on source GS (GS CAIN Only) and 4) trained on target SS masks (SS Only). The SS+GS model had the best and most consistent performance (mean DSC = 0.89, CoV = 0.05) and showed significantly (p < 0.05) higher DSC compared to the GS-only model on three target domains. Results suggest pre-training with noisy labels from the target domain allows the model to adapt to the dataset-specific characteristics and provides robust parameter initialization while fine-tuning with GS masks allows the model to learn detailed features. This method has wide application to other medical imaging problems where labeled data is scarce, and can be used as a per-dataset calibration method to accelerate wide-scale adoption.

{{</citation>}}


## cs.GR (1)



### (112/132) Search Me Knot, Render Me Knot: Embedding Search and Differentiable Rendering of Knots in 3D (Aalok Gangopadhyay et al., 2023)

{{<citation>}}

Aalok Gangopadhyay, Paras Gupta, Tarun Sharma, Prajwal Singh, Shanmuganathan Raman. (2023)  
**Search Me Knot, Render Me Knot: Embedding Search and Differentiable Rendering of Knots in 3D**  

---
Primary Category: cs.GR  
Categories: cs-GR, cs.GR  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2307.08652v3)  

---


**ABSTRACT**  
We introduce the problem of knot-based inverse perceptual art. Given multiple target images and their corresponding viewing configurations, the objective is to find a 3D knot-based tubular structure whose appearance resembles the target images when viewed from the specified viewing configurations. To solve this problem, we first design a differentiable rendering algorithm for rendering tubular knots embedded in 3D for arbitrary perspective camera configurations. Utilizing this differentiable rendering algorithm, we search over the space of knot configurations to find the ideal knot embedding. We represent the knot embeddings via homeomorphisms of the desired template knot, where the homeomorphisms are parametrized by the weights of an invertible neural network. Our approach is fully differentiable, making it possible to find the ideal 3D tubular structure for the desired perceptual art using gradient-based optimization. We propose several loss functions that impose additional physical constraints, ensuring that the tube is free of self-intersection, lies within a predefined region in space, satisfies the physical bending limits of the tube material and the material cost is within a specified budget. We demonstrate through results that our knot representation is highly expressive and gives impressive results even for challenging target images in both single view as well as multiple view constraints. Through extensive ablation study we show that each of the proposed loss function is effective in ensuring physical realizability. To the best of our knowledge, we are the first to propose a fully differentiable optimization framework for knot-based inverse perceptual art. Both the code and data will be made publicly available.

{{</citation>}}


## math.ST (1)



### (113/132) Overlapping Batch Confidence Intervals on Statistical Functionals Constructed from Time Series: Application to Quantiles, Optimization, and Estimation (Ziwei Su et al., 2023)

{{<citation>}}

Ziwei Su, Raghu Pasupathy, Yingchieh Yeh, Peter W. Glynn. (2023)  
**Overlapping Batch Confidence Intervals on Statistical Functionals Constructed from Time Series: Application to Quantiles, Optimization, and Estimation**  

---
Primary Category: math.ST  
Categories: 62F40 (Primary) 60F17, 62M10 (Secondary), cs-CE, math-PR, math-ST, math.ST, stat-CO, stat-ML, stat-TH  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2307.08609v1)  

---


**ABSTRACT**  
We propose a general purpose confidence interval procedure (CIP) for statistical functionals constructed using data from a stationary time series. The procedures we propose are based on derived distribution-free analogues of the $\chi^2$ and Student's $t$ random variables for the statistical functional context, and hence apply in a wide variety of settings including quantile estimation, gradient estimation, M-estimation, CVAR-estimation, and arrival process rate estimation, apart from more traditional statistical settings. Like the method of subsampling, we use overlapping batches of time series data to estimate the underlying variance parameter; unlike subsampling and the bootstrap, however, we assume that the implied point estimator of the statistical functional obeys a central limit theorem (CLT) to help identify the weak asymptotics (called OB-x limits, x=I,II,III) of batched Studentized statistics. The OB-x limits, certain functionals of the Wiener process parameterized by the size of the batches and the extent of their overlap, form the essential machinery for characterizing dependence, and consequently the correctness of the proposed CIPs. The message from extensive numerical experimentation is that in settings where a functional CLT on the point estimator is in effect, using \emph{large overlapping batches} alongside OB-x critical values yields confidence intervals that are often of significantly higher quality than those obtained from more generic methods like subsampling or the bootstrap. We illustrate using examples from CVaR estimation, ARMA parameter estimation, and NHPP rate estimation; R and MATLAB code for OB-x critical values is available at~\texttt{web.ics.purdue.edu/~pasupath/}.

{{</citation>}}


## physics.acc-ph (1)



### (114/132) Artificial Intelligence for the Electron Ion Collider (AI4EIC) (C. Allaire et al., 2023)

{{<citation>}}

C. Allaire, R. Ammendola, E. -C. Aschenauer, M. Balandat, M. Battaglieri, J. Bernauer, M. Bondì, N. Branson, T. Britton, A. Butter, I. Chahrour, P. Chatagnon, E. Cisbani, E. W. Cline, S. Dash, C. Dean, W. Deconinck, A. Deshpande, M. Diefenthaler, R. Ent, C. Fanelli, M. Finger, M. Finger, Jr., E. Fol, S. Furletov, Y. Gao, J. Giroux, N. C. Gunawardhana Waduge, R. Harish, O. Hassan, P. L. Hegde, R. J. Hernández-Pinto, A. Hiller Blin, T. Horn, J. Huang, D. Jayakodige, B. Joo, M. Junaid, P. Karande, B. Kriesten, R. Kunnawalkam Elayavalli, M. Lin, F. Liu, S. Liuti, G. Matousek, M. McEneaney, D. McSpadden, T. Menzo, T. Miceli, V. Mikuni, R. Montgomery, B. Nachman, R. R. Nair, J. Niestroy, S. A. Ochoa Oregon, J. Oleniacz, J. D. Osborn, C. Paudel, C. Pecar, C. Peng, G. N. Perdue, W. Phelps, M. L. Purschke, K. Rajput, Y. Ren, D. F. Renteria-Estrada, D. Richford, B. J. Roy, D. Roy, N. Sato, T. Satogata, G. Sborlini, M. Schram, D. Shih, J. Singh, R. Singh, A. Siodmok, P. Stone, J. Stevens, L. Suarez, K. Suresh, A. -N. Tawfik, F. Torales Acosta, N. Tran, R. Trotta, F. J. Twagirayezu, R. Tyson, S. Volkova, A. Vossen, E. Walter, D. Whiteson, M. Williams, S. Wu, N. Zachariou, P. Zurita. (2023)  
**Artificial Intelligence for the Electron Ion Collider (AI4EIC)**  

---
Primary Category: physics.acc-ph  
Categories: cs-LG, hep-ex, nucl-ex, nucl-th, physics-acc-ph, physics.acc-ph  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.08593v1)  

---


**ABSTRACT**  
The Electron-Ion Collider (EIC), a state-of-the-art facility for studying the strong force, is expected to begin commissioning its first experiments in 2028. This is an opportune time for artificial intelligence (AI) to be included from the start at this facility and in all phases that lead up to the experiments. The second annual workshop organized by the AI4EIC working group, which recently took place, centered on exploring all current and prospective application areas of AI for the EIC. This workshop is not only beneficial for the EIC, but also provides valuable insights for the newly established ePIC collaboration at EIC. This paper summarizes the different activities and R&D projects covered across the sessions of the workshop and provides an overview of the goals, approaches and strategies regarding AI/ML in the EIC community, as well as cutting-edge techniques currently studied in other experiments.

{{</citation>}}


## cs.HC (1)



### (115/132) A Case for VR Briefings: Comparing Communication in Daily Audio and VR Mission Control in a Simulated Lunar Mission (Kinga Skorupska et al., 2023)

{{<citation>}}

Kinga Skorupska, Maciej Grzeszczuk, Anna Jaskulska, Monika Kornacka, Grzegorz Pochwatko, Wiesław Kopeć. (2023)  
**A Case for VR Briefings: Comparing Communication in Daily Audio and VR Mission Control in a Simulated Lunar Mission**  

---
Primary Category: cs.HC  
Categories: H-5-1, cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.08589v1)  

---


**ABSTRACT**  
Alpha-XR Mission conducted by XR Lab PJAIT focused on research related to individual and crew well-being and participatory team collaboration in ICE (isolated, confined and extreme) conditions. In this two-week mission within an analog space habitat, collaboration, objective execution and leisure was facilitated and studied by virtual reality (VR) tools. The mission commander and first officer, both experienced with virtual reality, took part in daily briefings with mission control. In the first week the briefings were voice-only conducted via a channel on Discord. During the following week last briefings were conducted in VR, using Horizon Workrooms. This qualitative pilot study employing participatory observation revealed that VR facilitates communication, especially on complex problems and experiences, providing the sense of emotional connection and shared understanding, that may be lacking in audio calls. The study points to the need to further explore VR-facilitated communication in high-stake environments as it may improve relationships, well-being, and communication outcomes.

{{</citation>}}


## q-bio.NC (1)



### (116/132) A Study on the Performance of Generative Pre-trained Transformer (GPT) in Simulating Depressed Individuals on the Standardized Depressive Symptom Scale (Sijin Cai et al., 2023)

{{<citation>}}

Sijin Cai, Nanfeng Zhang, Jiaying Zhu, Yanjie Liu, Yongjin Zhou. (2023)  
**A Study on the Performance of Generative Pre-trained Transformer (GPT) in Simulating Depressed Individuals on the Standardized Depressive Symptom Scale**  

---
Primary Category: q-bio.NC  
Categories: cs-LG, q-bio-NC, q-bio.NC  
Keywords: GPT, Transformer  
[Paper Link](http://arxiv.org/abs/2307.08576v1)  

---


**ABSTRACT**  
Background: Depression is a common mental disorder with societal and economic burden. Current diagnosis relies on self-reports and assessment scales, which have reliability issues. Objective approaches are needed for diagnosing depression. Objective: Evaluate the potential of GPT technology in diagnosing depression. Assess its ability to simulate individuals with depression and investigate the influence of depression scales. Methods: Three depression-related assessment tools (HAMD-17, SDS, GDS-15) were used. Two experiments simulated GPT responses to normal individuals and individuals with depression. Compare GPT's responses with expected results, assess its understanding of depressive symptoms, and performance differences under different conditions. Results: GPT's performance in depression assessment was evaluated. It aligned with scoring criteria for both individuals with depression and normal individuals. Some performance differences were observed based on depression severity. GPT performed better on scales with higher sensitivity. Conclusion: GPT accurately simulates individuals with depression and normal individuals during depression-related assessments. Deviations occur when simulating different degrees of depression, limiting understanding of mild and moderate cases. GPT performs better on scales with higher sensitivity, indicating potential for developing more effective depression scales. GPT has important potential in depression assessment, supporting clinicians and patients.

{{</citation>}}


## cs.CR (3)



### (117/132) G-Scan: Graph Neural Networks for Line-Level Vulnerability Identification in Smart Contracts (Christoph Sendner et al., 2023)

{{<citation>}}

Christoph Sendner, Ruisi Zhang, Alexander Hefter, Alexandra Dmitrienko, Farinaz Koushanfar. (2023)  
**G-Scan: Graph Neural Networks for Line-Level Vulnerability Identification in Smart Contracts**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2307.08549v1)  

---


**ABSTRACT**  
Due to the immutable and decentralized nature of Ethereum (ETH) platform, smart contracts are prone to security risks that can result in financial loss. While existing machine learning-based vulnerability detection algorithms achieve high accuracy at the contract level, they require developers to manually inspect source code to locate bugs. To this end, we present G-Scan, the first end-to-end fine-grained line-level vulnerability detection system evaluated on the first-of-its-kind real world dataset. G-Scan first converts smart contracts to code graphs in a dependency and hierarchy preserving manner. Next, we train a graph neural network to identify vulnerable nodes and assess security risks. Finally, the code graphs with node vulnerability predictions are mapped back to the smart contracts for line-level localization. We train and evaluate G-Scan on a collected real world smart contracts dataset with line-level annotations on reentrancy vulnerability, one of the most common and severe types of smart contract vulnerabilities. With the well-designed graph representation and high-quality dataset, G-Scan achieves 93.02% F1-score in contract-level vulnerability detection and 93.69% F1-score in line-level vulnerability localization. Additionally, the lightweight graph neural network enables G-Scan to localize vulnerabilities in 6.1k lines of code smart contract within 1.2 seconds.

{{</citation>}}


### (118/132) LogPrécis: Unleashing Language Models for Automated Shell Log Analysis (Matteo Boffa et al., 2023)

{{<citation>}}

Matteo Boffa, Rodolfo Vieira Valentim, Luca Vassio, Danilo Giordano, Idilio Drago, Marco Mellia, Zied Ben Houidi. (2023)  
**LogPrécis: Unleashing Language Models for Automated Shell Log Analysis**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs-NI, cs.CR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.08309v1)  

---


**ABSTRACT**  
The collection of security-related logs holds the key to understanding attack behaviors and diagnosing vulnerabilities. Still, their analysis remains a daunting challenge. Recently, Language Models (LMs) have demonstrated unmatched potential in understanding natural and programming languages. The question arises whether and how LMs could be also useful for security experts since their logs contain intrinsically confused and obfuscated information. In this paper, we systematically study how to benefit from the state-of-the-art in LM to automatically analyze text-like Unix shell attack logs. We present a thorough design methodology that leads to LogPr\'ecis. It receives as input raw shell sessions and automatically identifies and assigns the attacker tactic to each portion of the session, i.e., unveiling the sequence of the attacker's goals. We demonstrate LogPr\'ecis capability to support the analysis of two large datasets containing about 400,000 unique Unix shell attacks. LogPr\'ecis reduces them into about 3,000 fingerprints, each grouping sessions with the same sequence of tactics. The abstraction it provides lets the analyst better understand attacks, identify fingerprints, detect novelty, link similar attacks, and track families and mutations. Overall, LogPr\'ecis, released as open source, paves the way for better and more responsive defense against cyberattacks.

{{</citation>}}


### (119/132) Identifying Vulnerable Third-Party Libraries from Textual Descriptions of Vulnerabilities and Libraries (Tianyu Chen et al., 2023)

{{<citation>}}

Tianyu Chen, Lin Li, Bingjie Shan, Guangtai Liang, Ding Li, Qianxiang Wang, Tao Xie. (2023)  
**Identifying Vulnerable Third-Party Libraries from Textual Descriptions of Vulnerabilities and Libraries**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-SE, cs.CR  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2307.08206v1)  

---


**ABSTRACT**  
To avoid potential risks posed by vulnerabilities in third-party libraries, security researchers maintain databases containing vulnerability reports, e.g., the National Vulnerability Database (NVD). Application developers can identify vulnerable libraries by directly querying the databases with the name of each used library. However, the querying results of vulnerable libraries are not reliable due to the incompleteness of vulnerability reports. Thus, current approaches model the task of identifying vulnerable libraries as an extreme multi-label learning (XML) task. These approaches suffer from highly inaccurate results and cannot identify zero-shot libraries (i.e., those not appearing during model training). To address these limitations, in this paper, we propose the first entity-linking approach named VulLibMiner to identify vulnerable third-party libraries from textual descriptions of vulnerabilities and libraries, together with VulLib, a Java vulnerability dataset with vulnerability-affected libraries. VulLibMiner consists of a coarse-grained TF-IDF matcher to efficiently screen out a small set of candidate libraries and a fine-grained BERT-FNN model to identify vulnerable libraries from these candidates effectively. We evaluate VulLibMiner using two state-of-the-art/practice approaches of library identification (FastXML, LightXML) on both their dataset named VeraJava and our VulLib dataset. Our evaluation results show that VulLibMiner can effectively identify vulnerable libraries with an average F1 score of 0.542 while the state-of-the-art/practice approaches achieve only 0.377. We demonstrate VulLibMiner's high value of security practice by using VulLibMiner to identify 12,716 <vulnerability, library> pairs, and 7,936 of them do not appear in NVD.

{{</citation>}}


## cs.SE (3)



### (120/132) Utilization of Pre-trained Language Model for Adapter-based Knowledge Transfer in Software Engineering (Iman Saberi et al., 2023)

{{<citation>}}

Iman Saberi, Fatemeh Fard, Fuxiang Chen. (2023)  
**Utilization of Pre-trained Language Model for Adapter-based Knowledge Transfer in Software Engineering**  

---
Primary Category: cs.SE  
Categories: 68N30, D-2-0; I-2-5, cs-SE, cs.SE  
Keywords: BERT, Language Model, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2307.08540v1)  

---


**ABSTRACT**  
Software Engineering (SE) Pre-trained Language Models (PLMs), such as CodeBERT, are pre-trained on large code corpora, and their learned knowledge has shown success in transferring into downstream tasks (e.g., code clone detection) through fine-tuning the PLMs. In Natural Language Processing (NLP), an alternative in transferring the knowledge of PLMs is explored through the use of adapter, a compact and parameter efficient module that is inserted into a PLM. Although the use of adapters has shown promising results in many NLP-based downstream tasks, their application and exploration in SE-based downstream tasks are limited.   Here, we study the knowledge transfer using adapters on multiple downstream tasks including cloze test, code clone detection, and code summarization. These adapters are trained on code corpora and are inserted into a PLM that is pre-trained on English corpora or code corpora. We called these PLMs as NL-PLM and C-PLM, respectively. We observed an improvement in results using NL-PLM over a PLM that does not have adapters, and this suggested that adapters can transfer and utilize useful knowledge from NL-PLM to SE tasks. The results are sometimes on par with or exceed the results of C-PLM; while being more efficient in terms of the number of parameters and training time. Interestingly, adapters inserted into a C-PLM generally yield better results than a traditional fine-tuned C-PLM. Our results open new directions to build more compact models for SE tasks.

{{</citation>}}


### (121/132) Extending the Frontier of ChatGPT: Code Generation and Debugging (Fardin Ahsan Sakib et al., 2023)

{{<citation>}}

Fardin Ahsan Sakib, Saadat Hasan Khan, A. H. M. Rezaul Karim. (2023)  
**Extending the Frontier of ChatGPT: Code Generation and Debugging**  

---
Primary Category: cs.SE  
Categories: cs-CL, cs-SE, cs.SE  
Keywords: AI, ChatGPT, GPT, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.08260v1)  

---


**ABSTRACT**  
Large-scale language models (LLMs) have emerged as a groundbreaking innovation in the realm of question-answering and conversational agents. These models, leveraging different deep learning architectures such as Transformers, are trained on vast corpora to predict sentences based on given queries. Among these LLMs, ChatGPT, developed by OpenAI, has ushered in a new era by utilizing artificial intelligence (AI) to tackle diverse problem domains, ranging from composing essays and biographies to solving intricate mathematical integrals. The versatile applications enabled by ChatGPT offer immense value to users. However, assessing the performance of ChatGPT's output poses a challenge, particularly in scenarios where queries lack clear objective criteria for correctness. For instance, evaluating the quality of generated essays becomes arduous and relies heavily on manual labor, in stark contrast to evaluating solutions to well-defined, closed-ended questions such as mathematical problems. This research paper delves into the efficacy of ChatGPT in solving programming problems, examining both the correctness and the efficiency of its solution in terms of time and memory complexity. The research reveals a commendable overall success rate of 71.875\%, denoting the proportion of problems for which ChatGPT was able to provide correct solutions that successfully satisfied all the test cases present in Leetcode. It exhibits strengths in structured problems and shows a linear correlation between its success rate and problem acceptance rates. However, it struggles to improve solutions based on feedback, pointing to potential shortcomings in debugging tasks. These findings provide a compact yet insightful glimpse into ChatGPT's capabilities and areas for improvement.

{{</citation>}}


### (122/132) In-IDE Generation-based Information Support with a Large Language Model (Daye Nam et al., 2023)

{{<citation>}}

Daye Nam, Andrew Macvean, Vincent Hellendoorn, Bogdan Vasilescu, Brad Myers. (2023)  
**In-IDE Generation-based Information Support with a Large Language Model**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-HC, cs-SE, cs.SE  
Keywords: AI, ChatGPT, GPT, GPT-3.5, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2307.08177v1)  

---


**ABSTRACT**  
Developers often face challenges in code understanding, which is crucial for building and maintaining high-quality software systems. Code comments and documentation can provide some context for the code, but are often scarce or missing. This challenge has become even more pressing with the rise of large language model (LLM) based code generation tools. To understand unfamiliar code, most software developers rely on general-purpose search engines to search through various programming information resources, which often requires multiple iterations of query rewriting and information foraging. More recently, developers have turned to online chatbots powered by LLMs, such as ChatGPT, which can provide more customized responses but also incur more overhead as developers need to communicate a significant amount of context to the LLM via a textual interface. In this study, we provide the investigation of an LLM-based conversational UI in the IDE. We aim to understand the promises and obstacles for tools powered by LLMs that are contextually aware, in that they automatically leverage the developer's programming context to answer queries. To this end, we develop an IDE Plugin that allows users to query back-ends such as OpenAI's GPT-3.5 and GPT-4 with high-level requests, like: explaining a highlighted section of code, explaining key domain-specific terms, or providing usage examples for an API. We conduct an exploratory user study with 32 participants to understand the usefulness and effectiveness, as well as individual preferences in the usage of, this LLM-powered information support tool. The study confirms that this approach can aid code understanding more effectively than web search, but the degree of the benefit differed by participants' experience levels.

{{</citation>}}


## cs.RO (2)



### (123/132) Land & Localize: An Infrastructure-free and Scalable Nano-Drones Swarm with UWB-based Localization (Mahyar Pourjabar et al., 2023)

{{<citation>}}

Mahyar Pourjabar, Ahmed AlKatheeri, Manuele Rusci, Agata Barcis, Vlad Niculescu, Eliseo Ferrante, Daniele Palossi, Luca Benini. (2023)  
**Land & Localize: An Infrastructure-free and Scalable Nano-Drones Swarm with UWB-based Localization**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs-SY, cs.RO, eess-SY  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2307.10255v1)  

---


**ABSTRACT**  
Relative localization is a crucial functional block of any robotic swarm. We address it in a fleet of nano-drones characterized by a 10 cm-scale form factor, which makes them highly versatile but also strictly limited in their onboard power envelope. State-of-the-Art solutions leverage Ultra-WideBand (UWB) technology, allowing distance range measurements between peer nano-drones and a stationary infrastructure of multiple UWB anchors. Therefore, we propose an UWB-based infrastructure-free nano-drones swarm, where part of the fleet acts as dynamic anchors, i.e., anchor-drones (ADs), capable of automatic deployment and landing. By varying the Ads' position constraint, we develop three alternative solutions with different trade-offs between flexibility and localization accuracy. In-field results, with four flying mission-drones (MDs), show a localization root mean square error (RMSE) spanning from 15.3 cm to 27.8 cm, at most. Scaling the number of MDs from 4 to 8, the RMSE marginally increases, i.e., less than 10 cm at most. The power consumption of the MDs' UWB module amounts to 342 mW. Ultimately, compared to a fixed-infrastructure commercial solution, our infrastructure-free system can be deployed anywhere and rapidly by taking 5.7 s to self-localize 4 ADs with a localization RMSE of up to 12.3% in the most challenging case with 8 MDs.

{{</citation>}}


### (124/132) Image-based Regularization for Action Smoothness in Autonomous Miniature Racing Car with Deep Reinforcement Learning (Hoang-Giang Cao et al., 2023)

{{<citation>}}

Hoang-Giang Cao, I Lee, Bo-Jiun Hsu, Zheng-Yi Lee, Yu-Wei Shih, Hsueh-Cheng Wang, I-Chen Wu. (2023)  
**Image-based Regularization for Action Smoothness in Autonomous Miniature Racing Car with Deep Reinforcement Learning**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs-SY, cs.RO, eess-SY  
Keywords: AWS, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.08230v1)  

---


**ABSTRACT**  
Deep reinforcement learning has achieved significant results in low-level controlling tasks. However, for some applications like autonomous driving and drone flying, it is difficult to control behavior stably since the agent may suddenly change its actions which often lowers the controlling system's efficiency, induces excessive mechanical wear, and causes uncontrollable, dangerous behavior to the vehicle. Recently, a method called conditioning for action policy smoothness (CAPS) was proposed to solve the problem of jerkiness in low-dimensional features for applications such as quadrotor drones. To cope with high-dimensional features, this paper proposes image-based regularization for action smoothness (I-RAS) for solving jerky control in autonomous miniature car racing. We also introduce a control based on impact ratio, an adaptive regularization weight to control the smoothness constraint, called IR control. In the experiment, an agent with I-RAS and IR control significantly improves the success rate from 59% to 95%. In the real-world-track experiment, the agent also outperforms other methods, namely reducing the average finish lap time, while also improving the completion rate even without real world training. This is also justified by an agent based on I-RAS winning the 2022 AWS DeepRacer Final Championship Cup.

{{</citation>}}


## cs.SD (2)



### (125/132) TST: Time-Sparse Transducer for Automatic Speech Recognition (Xiaohui Zhang et al., 2023)

{{<citation>}}

Xiaohui Zhang, Mangui Liang, Zhengkun Tian, Jiangyan Yi, Jianhua Tao. (2023)  
**TST: Time-Sparse Transducer for Automatic Speech Recognition**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: AI, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2307.08323v1)  

---


**ABSTRACT**  
End-to-end model, especially Recurrent Neural Network Transducer (RNN-T), has achieved great success in speech recognition. However, transducer requires a great memory footprint and computing time when processing a long decoding sequence. To solve this problem, we propose a model named time-sparse transducer, which introduces a time-sparse mechanism into transducer. In this mechanism, we obtain the intermediate representations by reducing the time resolution of the hidden states. Then the weighted average algorithm is used to combine these representations into sparse hidden states followed by the decoder. All the experiments are conducted on a Mandarin dataset AISHELL-1. Compared with RNN-T, the character error rate of the time-sparse transducer is close to RNN-T and the real-time factor is 50.00% of the original. By adjusting the time resolution, the time-sparse transducer can also reduce the real-time factor to 16.54% of the original at the expense of a 4.94% loss of precision.

{{</citation>}}


### (126/132) Towards Stealthy Backdoor Attacks against Speech Recognition via Elements of Sound (Hanbo Cai et al., 2023)

{{<citation>}}

Hanbo Cai, Pengcheng Zhang, Hai Dong, Yan Xiao, Stefanos Koffas, Yiming Li. (2023)  
**Towards Stealthy Backdoor Attacks against Speech Recognition via Elements of Sound**  

---
Primary Category: cs.SD  
Categories: cs-CR, cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2307.08208v1)  

---


**ABSTRACT**  
Deep neural networks (DNNs) have been widely and successfully adopted and deployed in various applications of speech recognition. Recently, a few works revealed that these models are vulnerable to backdoor attacks, where the adversaries can implant malicious prediction behaviors into victim models by poisoning their training process. In this paper, we revisit poison-only backdoor attacks against speech recognition. We reveal that existing methods are not stealthy since their trigger patterns are perceptible to humans or machine detection. This limitation is mostly because their trigger patterns are simple noises or separable and distinctive clips. Motivated by these findings, we propose to exploit elements of sound ($e.g.$, pitch and timbre) to design more stealthy yet effective poison-only backdoor attacks. Specifically, we insert a short-duration high-pitched signal as the trigger and increase the pitch of remaining audio clips to `mask' it for designing stealthy pitch-based triggers. We manipulate timbre features of victim audios to design the stealthy timbre-based attack and design a voiceprint selection module to facilitate the multi-backdoor attack. Our attacks can generate more `natural' poisoned samples and therefore are more stealthy. Extensive experiments are conducted on benchmark datasets, which verify the effectiveness of our attacks under different settings ($e.g.$, all-to-one, all-to-all, clean-label, physical, and multi-backdoor settings) and their stealthiness. The code for reproducing main experiments are available at \url{https://github.com/HanboCai/BadSpeech_SoE}.

{{</citation>}}


## cs.DB (2)



### (127/132) IterLara: A Turing Complete Algebra for Big Data, AI, Scientific Computing, and Database (Hongxiao Li et al., 2023)

{{<citation>}}

Hongxiao Li, Wanling Gao, Lei Wang, Jianfeng Zhan. (2023)  
**IterLara: A Turing Complete Algebra for Big Data, AI, Scientific Computing, and Database**  

---
Primary Category: cs.DB  
Categories: cs-CL, cs-DB, cs-DS, cs.DB  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.08315v1)  

---


**ABSTRACT**  
\textsc{Lara} is a key-value algebra that aims at unifying linear and relational algebra with three types of operation abstraction. The study of \textsc{Lara}'s expressive ability reports that it can represent relational algebra and most linear algebra operations. However, several essential computations, such as matrix inversion and determinant, cannot be expressed in \textsc{Lara}. \textsc{Lara} cannot represent global and iterative computation, either. This article proposes \textsc{IterLara}, extending \textsc{Lara} with iterative operators, to provide an algebraic model that unifies operations in general-purpose computing, like big data, AI, scientific computing, and database. We study the expressive ability of \textsc{Lara} and \textsc{IterLara} and prove that \textsc{IterLara} with aggregation functions can represent matrix inversion, determinant. Besides, we demonstrate that \textsc{IterLara} with no limitation of function utility is Turing complete. We also propose the Operation Count (OP) as a metric of computation amount for \textsc{IterLara} and ensure that the OP metric is in accordance with the existing computation metrics.

{{</citation>}}


### (128/132) Harnessing Scalable Transactional Stream Processing for Managing Large Language Models [Vision] (Shuhao Zhang et al., 2023)

{{<citation>}}

Shuhao Zhang, Xianzhi Zeng, Yuhao Wu, Zhonghao Yang. (2023)  
**Harnessing Scalable Transactional Stream Processing for Managing Large Language Models [Vision]**  

---
Primary Category: cs.DB  
Categories: cs-AI, cs-DB, cs-DC, cs.DB  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2307.08225v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have demonstrated extraordinary performance across a broad array of applications, from traditional language processing tasks to interpreting structured sequences like time-series data. Yet, their effectiveness in fast-paced, online decision-making environments requiring swift, accurate, and concurrent responses poses a significant challenge. This paper introduces TStreamLLM, a revolutionary framework integrating Transactional Stream Processing (TSP) with LLM management to achieve remarkable scalability and low latency. By harnessing the scalability, consistency, and fault tolerance inherent in TSP, TStreamLLM aims to manage continuous & concurrent LLM updates and usages efficiently. We showcase its potential through practical use cases like real-time patient monitoring and intelligent traffic management. The exploration of synergies between TSP and LLM management can stimulate groundbreaking developments in AI and database research. This paper provides a comprehensive overview of challenges and opportunities in this emerging field, setting forth a roadmap for future exploration and development.

{{</citation>}}


## eess.AS (2)



### (129/132) ivrit.ai: A Comprehensive Dataset of Hebrew Speech for AI Research and Development (Yanir Marmor et al., 2023)

{{<citation>}}

Yanir Marmor, Kinneret Misgav, Yair Lifshitz. (2023)  
**ivrit.ai: A Comprehensive Dataset of Hebrew Speech for AI Research and Development**  

---
Primary Category: eess.AS  
Categories: cs-CL, cs-SD, eess-AS, eess.AS  
Keywords: AI, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2307.08720v1)  

---


**ABSTRACT**  
We introduce "ivrit.ai", a comprehensive Hebrew speech dataset, addressing the distinct lack of extensive, high-quality resources for advancing Automated Speech Recognition (ASR) technology in Hebrew. With over 3,300 speech hours and a over a thousand diverse speakers, ivrit.ai offers a substantial compilation of Hebrew speech across various contexts. It is delivered in three forms to cater to varying research needs: raw unprocessed audio; data post-Voice Activity Detection, and partially transcribed data. The dataset stands out for its legal accessibility, permitting use at no cost, thereby serving as a crucial resource for researchers, developers, and commercial entities. ivrit.ai opens up numerous applications, offering vast potential to enhance AI capabilities in Hebrew. Future efforts aim to expand ivrit.ai further, thereby advancing Hebrew's standing in AI research and technology.

{{</citation>}}


### (130/132) Exploring Binary Classification Loss For Speaker Verification (Bing Han et al., 2023)

{{<citation>}}

Bing Han, Zhengyang Chen, Yanmin Qian. (2023)  
**Exploring Binary Classification Loss For Speaker Verification**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Speaker Verification  
[Paper Link](http://arxiv.org/abs/2307.08205v1)  

---


**ABSTRACT**  
The mismatch between close-set training and open-set testing usually leads to significant performance degradation for speaker verification task. For existing loss functions, metric learning-based objectives depend strongly on searching effective pairs which might hinder further improvements. And popular multi-classification methods are usually observed with degradation when evaluated on unseen speakers. In this work, we introduce SphereFace2 framework which uses several binary classifiers to train the speaker model in a pair-wise manner instead of performing multi-classification. Benefiting from this learning paradigm, it can efficiently alleviate the gap between training and evaluation. Experiments conducted on Voxceleb show that the SphereFace2 outperforms other existing loss functions, especially on hard trials. Besides, large margin fine-tuning strategy is proven to be compatible with it for further improvements. Finally, SphereFace2 also shows its strong robustness to class-wise noisy labels which has the potential to be applied in the semi-supervised training scenario with inaccurate estimated pseudo labels. Codes are available in https://github.com/Hunterhuan/sphereface2_speaker_verification

{{</citation>}}


## quant-ph (1)



### (131/132) A Quantum Convolutional Neural Network Approach for Object Detection and Classification (Gowri Namratha Meedinti et al., 2023)

{{<citation>}}

Gowri Namratha Meedinti, Kandukuri Sai Srirekha, Radhakrishnan Delhibabu. (2023)  
**A Quantum Convolutional Neural Network Approach for Object Detection and Classification**  

---
Primary Category: quant-ph  
Categories: cs-LG, quant-ph, quant-ph  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2307.08204v1)  

---


**ABSTRACT**  
This paper presents a comprehensive evaluation of the potential of Quantum Convolutional Neural Networks (QCNNs) in comparison to classical Convolutional Neural Networks (CNNs) and Artificial / Classical Neural Network (ANN) models. With the increasing amount of data, utilizing computing methods like CNN in real-time has become challenging. QCNNs overcome this challenge by utilizing qubits to represent data in a quantum environment and applying CNN structures to quantum computers. The time and accuracy of QCNNs are compared with classical CNNs and ANN models under different conditions such as batch size and input size. The maximum complexity level that QCNNs can handle in terms of these parameters is also investigated. The analysis shows that QCNNs have the potential to outperform both classical CNNs and ANN models in terms of accuracy and efficiency for certain applications, demonstrating their promise as a powerful tool in the field of machine learning.

{{</citation>}}


## q-bio.BM (1)



### (132/132) Efficient Prediction of Peptide Self-assembly through Sequential and Graphical Encoding (Zihan Liu et al., 2023)

{{<citation>}}

Zihan Liu, Jiaqi Wang, Yun Luo, Shuang Zhao, Wenbin Li, Stan Z. Li. (2023)  
**Efficient Prediction of Peptide Self-assembly through Sequential and Graphical Encoding**  

---
Primary Category: q-bio.BM  
Categories: cs-LG, q-bio-BM, q-bio.BM  
Keywords: AI, LSTM, Transformer  
[Paper Link](http://arxiv.org/abs/2307.09169v1)  

---


**ABSTRACT**  
In recent years, there has been an explosion of research on the application of deep learning to the prediction of various peptide properties, due to the significant development and market potential of peptides. Molecular dynamics has enabled the efficient collection of large peptide datasets, providing reliable training data for deep learning. However, the lack of systematic analysis of the peptide encoding, which is essential for AI-assisted peptide-related tasks, makes it an urgent problem to be solved for the improvement of prediction accuracy. To address this issue, we first collect a high-quality, colossal simulation dataset of peptide self-assembly containing over 62,000 samples generated by coarse-grained molecular dynamics (CGMD). Then, we systematically investigate the effect of peptide encoding of amino acids into sequences and molecular graphs using state-of-the-art sequential (i.e., RNN, LSTM, and Transformer) and structural deep learning models (i.e., GCN, GAT, and GraphSAGE), on the accuracy of peptide self-assembly prediction, an essential physiochemical process prior to any peptide-related applications. Extensive benchmarking studies have proven Transformer to be the most powerful sequence-encoding-based deep learning model, pushing the limit of peptide self-assembly prediction to decapeptides. In summary, this work provides a comprehensive benchmark analysis of peptide encoding with advanced deep learning models, serving as a guide for a wide range of peptide-related predictions such as isoelectric points, hydration free energy, etc.

{{</citation>}}
