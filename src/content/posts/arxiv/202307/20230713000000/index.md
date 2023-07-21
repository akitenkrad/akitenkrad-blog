---
draft: false
title: "arXiv @ 2023.07.13"
date: 2023-07-13
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.07.13"
    identifier: arxiv_20230713
    parent: 202307_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.LG (14)](#cslg-14)
- [cs.HC (2)](#cshc-2)
- [cs.RO (4)](#csro-4)
- [cs.CR (7)](#cscr-7)
- [cs.CV (16)](#cscv-16)
- [cs.CL (22)](#cscl-22)
- [eess.SP (2)](#eesssp-2)
- [cs.AI (9)](#csai-9)
- [cs.DB (1)](#csdb-1)
- [cs.SC (1)](#cssc-1)
- [eess.AS (1)](#eessas-1)
- [cs.SI (4)](#cssi-4)
- [eess.SY (1)](#eesssy-1)
- [cs.SD (1)](#cssd-1)
- [cs.IT (1)](#csit-1)
- [cs.NI (1)](#csni-1)
- [eess.IV (1)](#eessiv-1)
- [cs.CY (2)](#cscy-2)

## cs.LG (14)



### (1/90) MaxCorrMGNN: A Multi-Graph Neural Network Framework for Generalized Multimodal Fusion of Medical Data for Outcome Prediction (Niharika S. D'Souza et al., 2023)

{{<citation>}}

Niharika S. D'Souza, Hongzhi Wang, Andrea Giovannini, Antonio Foncubierta-Rodriguez, Kristen L. Beck, Orest Boyko, Tanveer Syeda-Mahmood. (2023)  
**MaxCorrMGNN: A Multi-Graph Neural Network Framework for Generalized Multimodal Fusion of Medical Data for Outcome Prediction**  

---
Primary Category: cs.LG
Categories: cs-LG, cs.LG, eess-SP  
Keywords: GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2307.07093v1)  

---


**ABSTRACT**  
With the emergence of multimodal electronic health records, the evidence for an outcome may be captured across multiple modalities ranging from clinical to imaging and genomic data. Predicting outcomes effectively requires fusion frameworks capable of modeling fine-grained and multi-faceted complex interactions between modality features within and across patients. We develop an innovative fusion approach called MaxCorr MGNN that models non-linear modality correlations within and across patients through Hirschfeld-Gebelein-Renyi maximal correlation (MaxCorr) embeddings, resulting in a multi-layered graph that preserves the identities of the modalities and patients. We then design, for the first time, a generalized multi-layered graph neural network (MGNN) for task-informed reasoning in multi-layered graphs, that learns the parameters defining patient-modality graph connectivity and message passing in an end-to-end fashion. We evaluate our model an outcome prediction task on a Tuberculosis (TB) dataset consistently outperforming several state-of-the-art neural, graph-based and traditional fusion techniques.

{{</citation>}}


### (2/90) Robotic Manipulation Datasets for Offline Compositional Reinforcement Learning (Marcel Hussing et al., 2023)

{{<citation>}}

Marcel Hussing, Jorge A. Mendez, Anisha Singrodia, Cassandra Kent, Eric Eaton. (2023)  
**Robotic Manipulation Datasets for Offline Compositional Reinforcement Learning**  

---
Primary Category: cs.LG
Categories: cs-AI, cs-LG, cs-RO, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.07091v1)  

---


**ABSTRACT**  
Offline reinforcement learning (RL) is a promising direction that allows RL agents to pre-train on large datasets, avoiding the recurrence of expensive data collection. To advance the field, it is crucial to generate large-scale datasets. Compositional RL is particularly appealing for generating such large datasets, since 1) it permits creating many tasks from few components, 2) the task structure may enable trained agents to solve new tasks by combining relevant learned components, and 3) the compositional dimensions provide a notion of task relatedness. This paper provides four offline RL datasets for simulated robotic manipulation created using the 256 tasks from CompoSuite [Mendez et al., 2022a]. Each dataset is collected from an agent with a different degree of performance, and consists of 256 million transitions. We provide training and evaluation settings for assessing an agent's ability to learn compositional task policies. Our benchmarking experiments on each setting show that current offline RL methods can learn the training tasks to some extent and that compositional methods significantly outperform non-compositional methods. However, current methods are still unable to extract the tasks' compositional structure to generalize to unseen tasks, showing a need for further research in offline compositional RL.

{{</citation>}}


### (3/90) Safe Reinforcement Learning as Wasserstein Variational Inference: Formal Methods for Interpretability (Yanran Wang et al., 2023)

{{<citation>}}

Yanran Wang, David Boyle. (2023)  
**Safe Reinforcement Learning as Wasserstein Variational Inference: Formal Methods for Interpretability**  

---
Primary Category: cs.LG
Categories: cs-AI, cs-LG, cs-RO, cs-SY, cs.LG, eess-SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.07084v1)  

---


**ABSTRACT**  
Reinforcement Learning or optimal control can provide effective reasoning for sequential decision-making problems with variable dynamics. Such reasoning in practical implementation, however, poses a persistent challenge in interpreting the reward function and corresponding optimal policy. Consequently, formalizing the sequential decision-making problems as inference has a considerable value, as probabilistic inference in principle offers diverse and powerful mathematical tools to infer the stochastic dynamics whilst suggesting a probabilistic interpretation of the reward design and policy convergence. In this study, we propose a novel Adaptive Wasserstein Variational Optimization (AWaVO) to tackle these challenges in sequential decision-making. Our approach utilizes formal methods to provide interpretations of reward design, transparency of training convergence, and probabilistic interpretation of sequential decisions. To demonstrate practicality, we show convergent training with guaranteed global convergence rates not only in simulation but also in real robot tasks, and empirically verify a reasonable tradeoff between high performance and conservative interpretability.

{{</citation>}}


### (4/90) Unsupervised Learning of Distributional Properties can Supplement Human Labeling and Increase Active Learning Efficiency in Anomaly Detection (Jaturong Kongmanee et al., 2023)

{{<citation>}}

Jaturong Kongmanee, Mark Chignell, Khilan Jerath, Abhay Raman. (2023)  
**Unsupervised Learning of Distributional Properties can Supplement Human Labeling and Increase Active Learning Efficiency in Anomaly Detection**  

---
Primary Category: cs.LG
Categories: cs-LG, cs.LG  
Keywords: Active Learning, Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2307.08782v1)  

---


**ABSTRACT**  
Exfiltration of data via email is a serious cybersecurity threat for many organizations. Detecting data exfiltration (anomaly) patterns typically requires labeling, most often done by a human annotator, to reduce the high number of false alarms. Active Learning (AL) is a promising approach for labeling data efficiently, but it needs to choose an efficient order in which cases are to be labeled, and there are uncertainties as to what scoring procedure should be used to prioritize cases for labeling, especially when detecting rare cases of interest is crucial. We propose an adaptive AL sampling strategy that leverages the underlying prior data distribution, as well as model uncertainty, to produce batches of cases to be labeled that contain instances of rare anomalies. We show that (1) the classifier benefits from a batch of representative and informative instances of both normal and anomalous examples, (2) unsupervised anomaly detection plays a useful role in building the classifier in the early stages of training when relatively little labeling has been done thus far. Our approach to AL for anomaly detection outperformed existing AL approaches on three highly unbalanced UCI benchmarks and on one real-world redacted email data set.

{{</citation>}}


### (5/90) Reward-Directed Conditional Diffusion: Provable Distribution Estimation and Reward Improvement (Hui Yuan et al., 2023)

{{<citation>}}

Hui Yuan, Kaixuan Huang, Chengzhuo Ni, Minshuo Chen, Mengdi Wang. (2023)  
**Reward-Directed Conditional Diffusion: Provable Distribution Estimation and Reward Improvement**  

---
Primary Category: cs.LG
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.07055v1)  

---


**ABSTRACT**  
We explore the methodology and theory of reward-directed generation via conditional diffusion models. Directed generation aims to generate samples with desired properties as measured by a reward function, which has broad applications in generative AI, reinforcement learning, and computational biology. We consider the common learning scenario where the data set consists of unlabeled data along with a smaller set of data with noisy reward labels. Our approach leverages a learned reward function on the smaller data set as a pseudolabeler. From a theoretical standpoint, we show that this directed generator can effectively learn and sample from the reward-conditioned data distribution. Additionally, our model is capable of recovering the latent subspace representation of data. Moreover, we establish that the model generates a new population that moves closer to a user-specified target reward value, where the optimality gap aligns with the off-policy bandit regret in the feature subspace. The improvement in rewards obtained is influenced by the interplay between the strength of the reward signal, the distribution shift, and the cost of off-support extrapolation. We provide empirical results to validate our theory and highlight the relationship between the strength of extrapolation and the quality of generated samples.

{{</citation>}}


### (6/90) Provable Multi-Task Representation Learning by Two-Layer ReLU Neural Networks (Liam Collins et al., 2023)

{{<citation>}}

Liam Collins, Hamed Hassani, Mahdi Soltanolkotabi, Aryan Mokhtari, Sanjay Shakkottai. (2023)  
**Provable Multi-Task Representation Learning by Two-Layer ReLU Neural Networks**  

---
Primary Category: cs.LG
Categories: cs-LG, cs.LG  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2307.06887v2)  

---


**ABSTRACT**  
Feature learning, i.e. extracting meaningful representations of data, is quintessential to the practical success of neural networks trained with gradient descent, yet it is notoriously difficult to explain how and why it occurs. Recent theoretical studies have shown that shallow neural networks optimized on a single task with gradient-based methods can learn meaningful features, extending our understanding beyond the neural tangent kernel or random feature regime in which negligible feature learning occurs. But in practice, neural networks are increasingly often trained on {\em many} tasks simultaneously with differing loss functions, and these prior analyses do not generalize to such settings. In the multi-task learning setting, a variety of studies have shown effective feature learning by simple linear models. However, multi-task learning via {\em nonlinear} models, arguably the most common learning paradigm in practice, remains largely mysterious. In this work, we present the first results proving feature learning occurs in a multi-task setting with a nonlinear model. We show that when the tasks are binary classification problems with labels depending on only $r$ directions within the ambient $d\gg r$-dimensional input space, executing a simple gradient-based multitask learning algorithm on a two-layer ReLU neural network learns the ground-truth $r$ directions. In particular, any downstream task on the $r$ ground-truth coordinates can be solved by learning a linear classifier with sample and neuron complexity independent of the ambient dimension $d$, while a random feature model requires exponential complexity in $d$ for such a guarantee.

{{</citation>}}


### (7/90) Sequential Monte Carlo Learning for Time Series Structure Discovery (Feras A. Saad et al., 2023)

{{<citation>}}

Feras A. Saad, Brian J. Patton, Matthew D. Hoffman, Rif A. Saurous, Vikash K. Mansinghka. (2023)  
**Sequential Monte Carlo Learning for Time Series Structure Discovery**  

---
Primary Category: cs.LG
Categories: cs-AI, cs-LG, cs.LG, stat-ME, stat-ML  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2307.09607v1)  

---


**ABSTRACT**  
This paper presents a new approach to automatically discovering accurate models of complex time series data. Working within a Bayesian nonparametric prior over a symbolic space of Gaussian process time series models, we present a novel structure learning algorithm that integrates sequential Monte Carlo (SMC) and involutive MCMC for highly effective posterior inference. Our method can be used both in "online" settings, where new data is incorporated sequentially in time, and in "offline" settings, by using nested subsets of historical data to anneal the posterior. Empirical measurements on real-world time series show that our method can deliver 10x--100x runtime speedups over previous MCMC and greedy-search structure learning algorithms targeting the same model family. We use our method to perform the first large-scale evaluation of Gaussian process time series structure learning on a prominent benchmark of 1,428 econometric datasets. The results show that our method discovers sensible models that deliver more accurate point forecasts and interval forecasts over multiple horizons as compared to widely used statistical and neural baselines that struggle on this challenging data.

{{</citation>}}


### (8/90) Identifying Early Help Referrals For Local Authorities With Machine Learning And Bias Analysis (Eufrásio de A. Lima Neto et al., 2023)

{{<citation>}}

Eufrásio de A. Lima Neto, Jonathan Bailiss, Axel Finke, Jo Miller, Georgina Cosma. (2023)  
**Identifying Early Help Referrals For Local Authorities With Machine Learning And Bias Analysis**  

---
Primary Category: cs.LG
Categories: cs-CY, cs-LG, cs.LG  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2307.06871v1)  

---


**ABSTRACT**  
Local authorities in England, such as Leicestershire County Council (LCC), provide Early Help services that can be offered at any point in a young person's life when they experience difficulties that cannot be supported by universal services alone, such as schools. This paper investigates the utilisation of machine learning (ML) to assist experts in identifying families that may need to be referred for Early Help assessment and support. LCC provided an anonymised dataset comprising 14360 records of young people under the age of 18. The dataset was pre-processed, machine learning models were build, and experiments were conducted to validate and test the performance of the models. Bias mitigation techniques were applied to improve the fairness of these models. During testing, while the models demonstrated the capability to identify young people requiring intervention or early help, they also produced a significant number of false positives, especially when constructed with imbalanced data, incorrectly identifying individuals who most likely did not need an Early Help referral. This paper empirically explores the suitability of data-driven ML models for identifying young people who may require Early Help services and discusses their appropriateness and limitations for this task.

{{</citation>}}


### (9/90) Neuro-symbolic Empowered Denoising Diffusion Probabilistic Models for Real-time Anomaly Detection in Industry 4.0 (Luigi Capogrosso et al., 2023)

{{<citation>}}

Luigi Capogrosso, Alessio Mascolini, Federico Girella, Geri Skenderi, Sebastiano Gaiardelli, Nicola Dall'Ora, Francesco Ponzio, Enrico Fraccaroli, Santa Di Cataldo, Sara Vinco, Enrico Macii, Franco Fummi, Marco Cristani. (2023)  
**Neuro-symbolic Empowered Denoising Diffusion Probabilistic Models for Real-time Anomaly Detection in Industry 4.0**  

---
Primary Category: cs.LG
Categories: cs-LG, cs.LG  
Keywords: AI, Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2307.06975v2)  

---


**ABSTRACT**  
Industry 4.0 involves the integration of digital technologies, such as IoT, Big Data, and AI, into manufacturing and industrial processes to increase efficiency and productivity. As these technologies become more interconnected and interdependent, Industry 4.0 systems become more complex, which brings the difficulty of identifying and stopping anomalies that may cause disturbances in the manufacturing process. This paper aims to propose a diffusion-based model for real-time anomaly prediction in Industry 4.0 processes. Using a neuro-symbolic approach, we integrate industrial ontologies in the model, thereby adding formal knowledge on smart manufacturing. Finally, we propose a simple yet effective way of distilling diffusion models through Random Fourier Features for deployment on an embedded system for direct integration into the manufacturing process. To the best of our knowledge, this approach has never been explored before.

{{</citation>}}


### (10/90) Implicit regularization in AI meets generalized hardness of approximation in optimization -- Sharp results for diagonal linear networks (Johan S. Wind et al., 2023)

{{<citation>}}

Johan S. Wind, Vegard Antun, Anders C. Hansen. (2023)  
**Implicit regularization in AI meets generalized hardness of approximation in optimization -- Sharp results for diagonal linear networks**  

---
Primary Category: cs.LG
Categories: 90C25, 68T07, 90C17 (Primary) 15A29, 94A08, 46N10 (Secondary), cs-LG, cs.LG, math-OC, stat-ML  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.07410v1)  

---


**ABSTRACT**  
Understanding the implicit regularization imposed by neural network architectures and gradient based optimization methods is a key challenge in deep learning and AI. In this work we provide sharp results for the implicit regularization imposed by the gradient flow of Diagonal Linear Networks (DLNs) in the over-parameterized regression setting and, potentially surprisingly, link this to the phenomenon of phase transitions in generalized hardness of approximation (GHA). GHA generalizes the phenomenon of hardness of approximation from computer science to, among others, continuous and robust optimization. It is well-known that the $\ell^1$-norm of the gradient flow of DLNs with tiny initialization converges to the objective function of basis pursuit. We improve upon these results by showing that the gradient flow of DLNs with tiny initialization approximates minimizers of the basis pursuit optimization problem (as opposed to just the objective function), and we obtain new and sharp convergence bounds w.r.t.\ the initialization size. Non-sharpness of our results would imply that the GHA phenomenon would not occur for the basis pursuit optimization problem -- which is a contradiction -- thus implying sharpness. Moreover, we characterize $\textit{which}$ $\ell_1$ minimizer of the basis pursuit problem is chosen by the gradient flow whenever the minimizer is not unique. Interestingly, this depends on the depth of the DLN.

{{</citation>}}


### (11/90) MPR-Net:Multi-Scale Pattern Reproduction Guided Universality Time Series Interpretable Forecasting (Tianlong Zhao et al., 2023)

{{<citation>}}

Tianlong Zhao, Xiang Ma, Xuemei Li, Caiming Zhang. (2023)  
**MPR-Net:Multi-Scale Pattern Reproduction Guided Universality Time Series Interpretable Forecasting**  

---
Primary Category: cs.LG
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Time Series, Transformer  
[Paper Link](http://arxiv.org/abs/2307.06736v1)  

---


**ABSTRACT**  
Time series forecasting has received wide interest from existing research due to its broad applications and inherent challenging. The research challenge lies in identifying effective patterns in historical series and applying them to future forecasting. Advanced models based on point-wise connected MLP and Transformer architectures have strong fitting power, but their secondary computational complexity limits practicality. Additionally, those structures inherently disrupt the temporal order, reducing the information utilization and making the forecasting process uninterpretable. To solve these problems, this paper proposes a forecasting model, MPR-Net. It first adaptively decomposes multi-scale historical series patterns using convolution operation, then constructs a pattern extension forecasting method based on the prior knowledge of pattern reproduction, and finally reconstructs future patterns into future series using deconvolution operation. By leveraging the temporal dependencies present in the time series, MPR-Net not only achieves linear time complexity, but also makes the forecasting process interpretable. By carrying out sufficient experiments on more than ten real data sets of both short and long term forecasting tasks, MPR-Net achieves the state of the art forecasting performance, as well as good generalization and robustness performance.

{{</citation>}}


### (12/90) Frameless Graph Knowledge Distillation (Dai Shi et al., 2023)

{{<citation>}}

Dai Shi, Zhiqi Shao, Yi Guo, Junbin Gao. (2023)  
**Frameless Graph Knowledge Distillation**  

---
Primary Category: cs.LG
Categories: cs-LG, cs.LG  
Keywords: GNN, Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2307.06631v1)  

---


**ABSTRACT**  
Knowledge distillation (KD) has shown great potential for transferring knowledge from a complex teacher model to a simple student model in which the heavy learning task can be accomplished efficiently and without losing too much prediction accuracy. Recently, many attempts have been made by applying the KD mechanism to the graph representation learning models such as graph neural networks (GNNs) to accelerate the model's inference speed via student models. However, many existing KD-based GNNs utilize MLP as a universal approximator in the student model to imitate the teacher model's process without considering the graph knowledge from the teacher model. In this work, we provide a KD-based framework on multi-scaled GNNs, known as graph framelet, and prove that by adequately utilizing the graph knowledge in a multi-scaled manner provided by graph framelet decomposition, the student model is capable of adapting both homophilic and heterophilic graphs and has the potential of alleviating the over-squashing issue with a simple yet effectively graph surgery. Furthermore, we show how the graph knowledge supplied by the teacher is learned and digested by the student model via both algebra and geometry. Comprehensive experiments show that our proposed model can generate learning accuracy identical to or even surpass the teacher model while maintaining the high speed of inference.

{{</citation>}}


### (13/90) Introducing Foundation Models as Surrogate Models: Advancing Towards More Practical Adversarial Attacks (Jiaming Zhang et al., 2023)

{{<citation>}}

Jiaming Zhang, Jitao Sang, Qi Yi, Changsheng Xu. (2023)  
**Introducing Foundation Models as Surrogate Models: Advancing Towards More Practical Adversarial Attacks**  

---
Primary Category: cs.LG
Categories: cs-AI, cs-CR, cs-LG, cs.LG  
Keywords: AI, Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2307.06608v2)  

---


**ABSTRACT**  
Recently, the no-box adversarial attack, in which the attacker lacks access to the model's architecture, weights, and training data, become the most practical and challenging attack setup. However, there is an unawareness of the potential and flexibility inherent in the surrogate model selection process on no-box setting. Inspired by the burgeoning interest in utilizing foundational models to address downstream tasks, this paper adopts an innovative idea that 1) recasting adversarial attack as a downstream task. Specifically, image noise generation to meet the emerging trend and 2) introducing foundational models as surrogate models. Harnessing the concept of non-robust features, we elaborate on two guiding principles for surrogate model selection to explain why the foundational model is an optimal choice for this role. However, paradoxically, we observe that these foundational models underperform. Analyzing this unexpected behavior within the feature space, we attribute the lackluster performance of foundational models (e.g., CLIP) to their significant representational capacity and, conversely, their lack of discriminative prowess. To mitigate this issue, we propose the use of a margin-based loss strategy for the fine-tuning of foundational models on target images. The experimental results verify that our approach, which employs the basic Fast Gradient Sign Method (FGSM) attack algorithm, outstrips the performance of other, more convoluted algorithms. We conclude by advocating for the research community to consider surrogate models as crucial determinants in the effectiveness of adversarial attacks in no-box settings. The implications of our work bear relevance for improving the efficacy of such adversarial attacks and the overall robustness of AI systems.

{{</citation>}}


### (14/90) On the Effective Horizon of Inverse Reinforcement Learning (Yiqing Xu et al., 2023)

{{<citation>}}

Yiqing Xu, Finale Doshi-Velez, David Hsu. (2023)  
**On the Effective Horizon of Inverse Reinforcement Learning**  

---
Primary Category: cs.LG
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.06541v1)  

---


**ABSTRACT**  
Inverse reinforcement learning (IRL) algorithms often rely on (forward) reinforcement learning or planning over a given time horizon to compute an approximately optimal policy for a hypothesized reward function and then match this policy with expert demonstrations. The time horizon plays a critical role in determining both the accuracy of reward estimate and the computational efficiency of IRL algorithms. Interestingly, an effective time horizon shorter than the ground-truth value often produces better results faster. This work formally analyzes this phenomenon and provides an explanation: the time horizon controls the complexity of an induced policy class and mitigates overfitting with limited data. This analysis leads to a principled choice of the effective horizon for IRL. It also prompts us to reexamine the classic IRL formulation: it is more natural to learn jointly the reward and the effective horizon together rather than the reward alone with a given horizon. Our experimental results confirm the theoretical analysis.

{{</citation>}}


## cs.HC (2)



### (15/90) An Analysis of Dialogue Repair in Virtual Voice Assistants (Matthew Carson Galbraith et al., 2023)

{{<citation>}}

Matthew Carson Galbraith, Mireia Gómez i Martínez. (2023)  
**An Analysis of Dialogue Repair in Virtual Voice Assistants**  

---
Primary Category: cs.HC
Categories: cs-CL, cs-HC, cs.HC  
Keywords: Dialog, Dialogue, Google  
[Paper Link](http://arxiv.org/abs/2307.07076v1)  

---


**ABSTRACT**  
Language speakers often use what are known as repair initiators to mend fundamental disconnects that occur between them during verbal communication. Previous research in this field has mainly focused on the human-to-human use of repair initiator. We proposed an examination of dialogue repair structure wherein the dialogue initiator is human and the party that initiates or responds to the repair is a virtual assistant. This study examined the use of repair initiators in both English and Spanish with two popular assistants, Google Assistant and Apple's Siri. Our aim was to codify the differences, if any, in responses by voice assistants to dialogues in need of repair as compared to human-human dialogues also in need of repair. Ultimately the data demonstrated that not only were there differences between human-assistant and human-human dialogue repair strategies, but that there were likewise differences among the assistants and the languages studied.

{{</citation>}}


### (16/90) Towards Ubiquitous Semantic Metaverse: Challenges, Approaches, and Opportunities (Kai Li et al., 2023)

{{<citation>}}

Kai Li, Billy Lau, Xin Yuan, Wei Ni, Mohsen Guizani, Chau Yuen. (2023)  
**Towards Ubiquitous Semantic Metaverse: Challenges, Approaches, and Opportunities**  

---
Primary Category: cs.HC
Categories: cs-AI, cs-HC, cs-NI, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.06687v1)  

---


**ABSTRACT**  
In recent years, ubiquitous semantic Metaverse has been studied to revolutionize immersive cyber-virtual experiences for augmented reality (AR) and virtual reality (VR) users, which leverages advanced semantic understanding and representation to enable seamless, context-aware interactions within mixed-reality environments. This survey focuses on the intelligence and spatio-temporal characteristics of four fundamental system components in ubiquitous semantic Metaverse, i.e., artificial intelligence (AI), spatio-temporal data representation (STDR), semantic Internet of Things (SIoT), and semantic-enhanced digital twin (SDT). We thoroughly survey the representative techniques of the four fundamental system components that enable intelligent, personalized, and context-aware interactions with typical use cases of the ubiquitous semantic Metaverse, such as remote education, work and collaboration, entertainment and socialization, healthcare, and e-commerce marketing. Furthermore, we outline the opportunities for constructing the future ubiquitous semantic Metaverse, including scalability and interoperability, privacy and security, performance measurement and standardization, as well as ethical considerations and responsible AI. Addressing those challenges is important for creating a robust, secure, and ethically sound system environment that offers engaging immersive experiences for the users and AR/VR applications.

{{</citation>}}


## cs.RO (4)



### (17/90) CART: Collision Avoidance and Robust Tracking Augmentation in Learning-based Motion Planning for Multi-Agent Systems (Hiroyasu Tsukamoto et al., 2023)

{{<citation>}}

Hiroyasu Tsukamoto, Benjamin Rivière, Changrak Choi, Amir Rahmani, Soon-Jo Chung. (2023)  
**CART: Collision Avoidance and Robust Tracking Augmentation in Learning-based Motion Planning for Multi-Agent Systems**  

---
Primary Category: cs.RO
Categories: cs-LG, cs-MA, cs-RO, cs-SY, cs.RO, eess-SY, math-OC  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2307.08602v1)  

---


**ABSTRACT**  
This paper presents CART, an analytical method to augment a learning-based, distributed motion planning policy of a nonlinear multi-agent system with real-time collision avoidance and robust tracking guarantees, independently of learning errors. We first derive an analytical form of an optimal safety filter for Lagrangian systems, which formally ensures a collision-free operation in a multi-agent setting in a disturbance-free environment, while allowing for its distributed implementation with minimal deviation from the learned policy. We then propose an analytical form of an optimal robust filter for Lagrangian systems to be used hierarchically with the learned collision-free target trajectory, which also enables distributed implementation and guarantees exponential boundedness of the trajectory tracking error for safety, even under the presence of deterministic and stochastic disturbance. These results are shown to extend further to general control-affine nonlinear systems using contraction theory. Our key contribution is to enhance the performance of the learned motion planning policy with collision avoidance and tracking-based robustness guarantees, independently of its original performance such as approximation errors and regret bounds in machine learning. We demonstrate the effectiveness of CART in motion planning and control of several examples of nonlinear systems, including spacecraft formation flying and rotor-failed UAV swarms.

{{</citation>}}


### (18/90) DRAGON: A Dialogue-Based Robot for Assistive Navigation with Visual Language Grounding (Shuijing Liu et al., 2023)

{{<citation>}}

Shuijing Liu, Aamir Hasan, Kaiwen Hong, Runxuan Wang, Peixin Chang, Zachary Mizrachi, Justin Lin, D. Livingston McPherson, Wendy A. Rogers, Katherine Driggs-Campbell. (2023)  
**DRAGON: A Dialogue-Based Robot for Assistive Navigation with Visual Language Grounding**  

---
Primary Category: cs.RO
Categories: cs-AI, cs-CL, cs-HC, cs-LG, cs-RO, cs.RO  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2307.06924v1)  

---


**ABSTRACT**  
Persons with visual impairments (PwVI) have difficulties understanding and navigating spaces around them. Current wayfinding technologies either focus solely on navigation or provide limited communication about the environment. Motivated by recent advances in visual-language grounding and semantic navigation, we propose DRAGON, a guiding robot powered by a dialogue system and the ability to associate the environment with natural language. By understanding the commands from the user, DRAGON is able to guide the user to the desired landmarks on the map, describe the environment, and answer questions from visual observations. Through effective utilization of dialogue, the robot can ground the user's free-form descriptions to landmarks in the environment, and give the user semantic information through spoken language. We conduct a user study with blindfolded participants in an everyday indoor environment. Our results demonstrate that DRAGON is able to communicate with the user smoothly, provide a good guiding experience, and connect users with their surrounding environment in an intuitive manner.

{{</citation>}}


### (19/90) Self-Supervised Learning for Interactive Perception of Surgical Thread for Autonomous Suture Tail-Shortening (Vincent Schorp et al., 2023)

{{<citation>}}

Vincent Schorp, Will Panitch, Kaushik Shivakumar, Vainavi Viswanath, Justin Kerr, Yahav Avigal, Danyal M Fer, Lionel Ott, Ken Goldberg. (2023)  
**Self-Supervised Learning for Interactive Perception of Surgical Thread for Autonomous Suture Tail-Shortening**  

---
Primary Category: cs.RO
Categories: cs-AI, cs-RO, cs.RO  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2307.06845v1)  

---


**ABSTRACT**  
Accurate 3D sensing of suturing thread is a challenging problem in automated surgical suturing because of the high state-space complexity, thinness and deformability of the thread, and possibility of occlusion by the grippers and tissue. In this work we present a method for tracking surgical thread in 3D which is robust to occlusions and complex thread configurations, and apply it to autonomously perform the surgical suture "tail-shortening" task: pulling thread through tissue until a desired "tail" length remains exposed. The method utilizes a learned 2D surgical thread detection network to segment suturing thread in RGB images. It then identifies the thread path in 2D and reconstructs the thread in 3D as a NURBS spline by triangulating the detections from two stereo cameras. Once a 3D thread model is initialized, the method tracks the thread across subsequent frames. Experiments suggest the method achieves a 1.33 pixel average reprojection error on challenging single-frame 3D thread reconstructions, and an 0.84 pixel average reprojection error on two tracking sequences. On the tail-shortening task, it accomplishes a 90% success rate across 20 trials. Supplemental materials are available at https://sites.google.com/berkeley.edu/autolab-surgical-thread/ .

{{</citation>}}


### (20/90) Aeolus Ocean -- A simulation environment for the autonomous COLREG-compliant navigation of Unmanned Surface Vehicles using Deep Reinforcement Learning and Maritime Object Detection (Andrew Alexander Vekinis et al., 2023)

{{<citation>}}

Andrew Alexander Vekinis, Stavros Perantonis. (2023)  
**Aeolus Ocean -- A simulation environment for the autonomous COLREG-compliant navigation of Unmanned Surface Vehicles using Deep Reinforcement Learning and Maritime Object Detection**  

---
Primary Category: cs.RO
Categories: cs-LG, cs-RO, cs.RO  
Keywords: Computer Vision, Object Detection, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.06688v1)  

---


**ABSTRACT**  
Heading towards navigational autonomy in unmanned surface vehicles (USVs) in the maritime sector can fundamentally lead towards safer waters as well as reduced operating costs, while also providing a range of exciting new capabilities for oceanic research, exploration and monitoring. However, achieving such a goal is challenging. USV control systems must, safely and reliably, be able to adhere to the international regulations for preventing collisions at sea (COLREGs) in encounters with other vessels as they navigate to a given waypoint while being affected by realistic weather conditions, either during the day or at night. To deal with the multitude of possible scenarios, it is critical to have a virtual environment that is able to replicate the realistic operating conditions USVs will encounter, before they can be implemented in the real world. Such "digital twins" form the foundations upon which Deep Reinforcement Learning (DRL) and Computer Vision (CV) algorithms can be used to develop and guide USV control systems. In this paper we describe the novel development of a COLREG-compliant DRL-based collision avoidant navigational system with CV-based awareness in a realistic ocean simulation environment. The performance of the trained autonomous Agents resulting from this approach is evaluated in several successful navigations to set waypoints in both open sea and coastal encounters with other vessels. A binary executable version of the simulator with trained agents is available at https://github.com/aavek/Aeolus-Ocean

{{</citation>}}


## cs.CR (7)



### (21/90) Proof of Training (PoT): Harnessing Crypto Mining Power for Distributed AI Training (Peihao Li, 2023)

{{<citation>}}

Peihao Li. (2023)  
**Proof of Training (PoT): Harnessing Crypto Mining Power for Distributed AI Training**  

---
Primary Category: cs.CR
Categories: cs-CE, cs-CR, cs-DC, cs-LG, cs.CR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.07066v1)  

---


**ABSTRACT**  
In the midst of the emerging trend of integrating artificial intelligence (AI) with crypto mining, we identify three major challenges that create a gap between these two fields. To bridge this gap, we introduce the proof-of-training (PoT) protocol, an approach that combines the strengths of both AI and blockchain technology. The PoT protocol utilizes the practical Byzantine fault tolerance (PBFT) consensus mechanism to synchronize global states. To evaluate the performance of the protocol design, we present an implementation of a decentralized training network (DTN) that adopts the PoT protocol. Our results indicate that the protocol exhibits considerable potential in terms of task throughput, system robustness, and network security.

{{</citation>}}


### (22/90) A Controlled Experiment on the Impact of Intrusion Detection False Alarm Rate on Analyst Performance (Lucas Layman et al., 2023)

{{<citation>}}

Lucas Layman, William Roden. (2023)  
**A Controlled Experiment on the Impact of Intrusion Detection False Alarm Rate on Analyst Performance**  

---
Primary Category: cs.CR
Categories: cs-CR, cs.CR  
Keywords: Intrusion Detection  
[Paper Link](http://arxiv.org/abs/2307.07023v1)  

---


**ABSTRACT**  
Organizations use intrusion detection systems (IDSes) to identify harmful activity among millions of computer network events. Cybersecurity analysts review IDS alarms to verify whether malicious activity occurred and to take remedial action. However, IDS systems exhibit high false alarm rates. This study examines the impact of IDS false alarm rate on human analyst sensitivity (probability of detection), precision (positive predictive value), and time on task when evaluating IDS alarms. A controlled experiment was conducted with participants divided into two treatment groups, 50% IDS false alarm rate and 86% false alarm rate, who classified whether simulated IDS alarms were true or false alarms. Results show statistically significant differences in precision and time on task. The median values for the 86% false alarm rate group were 47% lower precision and 40% slower time on task than the 50% false alarm rate group. No significant difference in analyst sensitivity was observed.

{{</citation>}}


### (23/90) Data Behind the Walls An Advanced Architecture for Data Privacy Management (Amen Faridoon et al., 2023)

{{<citation>}}

Amen Faridoon, M. Tahar Kechadi. (2023)  
**Data Behind the Walls An Advanced Architecture for Data Privacy Management**  

---
Primary Category: cs.CR
Categories: cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2307.06779v1)  

---


**ABSTRACT**  
In today's highly connected society, we are constantly asked to provide personal information to retailers, voter surveys, medical professionals, and other data collection efforts. The collected data is stored in large data warehouses. Organisations and statistical agencies share and use this data to facilitate research in public health, economics, sociology, etc. However, this data contains sensitive information about individuals, which can result in identity theft, financial loss, stress and depression, embarrassment, abuse, etc. Therefore, one must ensure rigorous management of individuals' privacy. We propose, an advanced data privacy management architecture composed of three layers. The data management layer consists of de-identification and anonymisation, the access management layer for re-enforcing data access based on the concepts of Role-Based Access Control and the Chinese Wall Security Policy, and the roles layer for regulating different users. The proposed system architecture is validated on healthcare datasets.

{{</citation>}}


### (24/90) A Comprehensive Analysis of Blockchain Applications for Securing Computer Vision Systems (Ramalingam M et al., 2023)

{{<citation>}}

Ramalingam M, Chemmalar Selvi, Nancy Victor, Rajeswari Chengoden, Sweta Bhattacharya, Praveen Kumar Reddy Maddikunta, Duehee Lee, Md. Jalil Piran, Neelu Khare, Gokul Yendri, Thippa Reddy Gadekallu. (2023)  
**A Comprehensive Analysis of Blockchain Applications for Securing Computer Vision Systems**  

---
Primary Category: cs.CR
Categories: cs-CR, cs-CV, cs-CY, cs.CR  
Keywords: Computer Vision  
[Paper Link](http://arxiv.org/abs/2307.06659v1)  

---


**ABSTRACT**  
Blockchain (BC) and Computer Vision (CV) are the two emerging fields with the potential to transform various sectors.The ability of BC can help in offering decentralized and secure data storage, while CV allows machines to learn and understand visual data. This integration of the two technologies holds massive promise for developing innovative applications that can provide solutions to the challenges in various sectors such as supply chain management, healthcare, smart cities, and defense. This review explores a comprehensive analysis of the integration of BC and CV by examining their combination and potential applications. It also provides a detailed analysis of the fundamental concepts of both technologies, highlighting their strengths and limitations. This paper also explores current research efforts that make use of the benefits offered by this combination. The effort includes how BC can be used as an added layer of security in CV systems and also ensure data integrity, enabling decentralized image and video analytics using BC. The challenges and open issues associated with this integration are also identified, and appropriate potential future directions are also proposed.

{{</citation>}}


### (25/90) SecureFalcon: The Next Cyber Reasoning System for Cyber Security (Mohamed Amine Ferrag et al., 2023)

{{<citation>}}

Mohamed Amine Ferrag, Ammar Battah, Norbert Tihanyi, Merouane Debbah, Thierry Lestable, Lucas C. Cordeiro. (2023)  
**SecureFalcon: The Next Cyber Reasoning System for Cyber Security**  

---
Primary Category: cs.CR
Categories: cs-AI, cs-CR, cs.CR  
Keywords: AI, Cyber Security, Falcon, Language Model, Reasoning, Security  
[Paper Link](http://arxiv.org/abs/2307.06616v1)  

---


**ABSTRACT**  
Software vulnerabilities leading to various detriments such as crashes, data loss, and security breaches, significantly hinder the quality, affecting the market adoption of software applications and systems. Although traditional methods such as automated software testing, fault localization, and repair have been intensively studied, static analysis tools are most commonly used and have an inherent false positives rate, posing a solid challenge to developer productivity. Large Language Models (LLMs) offer a promising solution to these persistent issues. Among these, FalconLLM has shown substantial potential in identifying intricate patterns and complex vulnerabilities, hence crucial in software vulnerability detection. In this paper, for the first time, FalconLLM is being fine-tuned for cybersecurity applications, thus introducing SecureFalcon, an innovative model architecture built upon FalconLLM. SecureFalcon is trained to differentiate between vulnerable and non-vulnerable C code samples. We build a new training dataset, FormAI, constructed thanks to Generative Artificial Intelligence (AI) and formal verification to evaluate its performance. SecureFalcon achieved an impressive 94% accuracy rate in detecting software vulnerabilities, emphasizing its significant potential to redefine software vulnerability detection methods in cybersecurity.

{{</citation>}}


### (26/90) TPU as Cryptographic Accelerator (Rabimba Karanjai et al., 2023)

{{<citation>}}

Rabimba Karanjai, Sangwon Shin, Xinxin Fan, Lin Chen, Tianwei Zhang, Taeweon Suh, Weidong Shi, Lei Xu. (2023)  
**TPU as Cryptographic Accelerator**  

---
Primary Category: cs.CR
Categories: cs-CR, cs.CR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.06554v1)  

---


**ABSTRACT**  
Polynomials defined on specific rings are heavily involved in various cryptographic schemes, and the corresponding operations are usually the computation bottleneck of the whole scheme.   We propose to utilize TPU, an emerging hardware designed for AI applications, to speed up polynomial operations and convert TPU to a cryptographic accelerator.   We also conduct preliminary evaluation and discuss the limitations of current work and future plan.

{{</citation>}}


### (27/90) Migrating to Post-Quantum Cryptography: a Framework Using Security Dependency Analysis (Khondokar Fida Hasan et al., 2023)

{{<citation>}}

Khondokar Fida Hasan, Leonie Simpson, Mir Ali Rezazadeh Baee, Chadni Islam, Ziaur Rahman, Warren Armstrong, Praveen Gauravaram, Matthew McKague. (2023)  
**Migrating to Post-Quantum Cryptography: a Framework Using Security Dependency Analysis**  

---
Primary Category: cs.CR
Categories: cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2307.06520v1)  

---


**ABSTRACT**  
Quantum computing is emerging as an unprecedented threat to the current state of widely used cryptographic systems. Cryptographic methods that have been considered secure for decades will likely be broken, with enormous impact on the security of sensitive data and communications in enterprises worldwide. A plan to migrate to quantum-resistant cryptographic systems is required. However, migrating an enterprise system to ensure a quantum-safe state is a complex process. Enterprises will require systematic guidance to perform this migration to remain resilient in a post-quantum era, as many organisations do not have staff with the expertise to manage this process unaided. This paper presents a comprehensive framework designed to aid enterprises in their migration. The framework articulates key steps and technical considerations in the cryptographic migration process. It makes use of existing organisational inventories and provides a roadmap for prioritising the replacement of cryptosystems in a post-quantum context. The framework enables the efficient identification of cryptographic objects, and can be integrated with other frameworks in enterprise settings to minimise operational disruption during migration. Practical case studies are included to demonstrate the utility and efficacy of the proposed framework using graph theoretic techniques to determine and evaluate cryptographic dependencies.

{{</citation>}}


## cs.CV (16)



### (28/90) Bootstrapping Vision-Language Learning with Decoupled Language Pre-training (Yiren Jian et al., 2023)

{{<citation>}}

Yiren Jian, Chongyang Gao, Soroush Vosoughi. (2023)  
**Bootstrapping Vision-Language Learning with Decoupled Language Pre-training**  

---
Primary Category: cs.CV
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.07063v1)  

---


**ABSTRACT**  
We present a novel methodology aimed at optimizing the application of frozen large language models (LLMs) for resource-intensive vision-language (VL) pre-training. The current paradigm uses visual features as prompts to guide language models, with a focus on determining the most relevant visual features for corresponding text. Our approach diverges by concentrating on the language component, specifically identifying the optimal prompts to align with visual features. We introduce the Prompt-Transformer (P-Former), a model that predicts these ideal prompts, which is trained exclusively on linguistic data, bypassing the need for image-text pairings. This strategy subtly bifurcates the end-to-end VL training process into an additional, separate stage. Our experiments reveal that our framework significantly enhances the performance of a robust image-to-text baseline (BLIP-2), and effectively narrows the performance gap between models trained with either 4M or 129M image-text pairs. Importantly, our framework is modality-agnostic and flexible in terms of architectural design, as validated by its successful application in a video learning task using varied base modules. The code is available at https://github.com/yiren-jian/BLIText

{{</citation>}}


### (29/90) A metric learning approach for endoscopic kidney stone identification (Jorge Gonzalez-Zapata et al., 2023)

{{<citation>}}

Jorge Gonzalez-Zapata, Francisco Lopez-Tiro, Elias Villalvazo-Avila, Daniel Flores-Araiza, Jacques Hubert, Andres Mendez-Vazquez, Gilberto Ochoa-Ruiz, Christian Daul. (2023)  
**A metric learning approach for endoscopic kidney stone identification**  

---
Primary Category: cs.CV
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Few-Shot, Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2307.07046v1)  

---


**ABSTRACT**  
Several Deep Learning (DL) methods have recently been proposed for an automated identification of kidney stones during an ureteroscopy to enable rapid therapeutic decisions. Even if these DL approaches led to promising results, they are mainly appropriate for kidney stone types for which numerous labelled data are available. However, only few labelled images are available for some rare kidney stone types. This contribution exploits Deep Metric Learning (DML) methods i) to handle such classes with few samples, ii) to generalize well to out of distribution samples, and iii) to cope better with new classes which are added to the database. The proposed Guided Deep Metric Learning approach is based on a novel architecture which was designed to learn data representations in an improved way. The solution was inspired by Few-Shot Learning (FSL) and makes use of a teacher-student approach. The teacher model (GEMINI) generates a reduced hypothesis space based on prior knowledge from the labeled data, and is used it as a guide to a student model (i.e., ResNet50) through a Knowledge Distillation scheme. Extensive tests were first performed on two datasets separately used for the recognition, namely a set of images acquired for the surfaces of the kidney stone fragments, and a set of images of the fragment sections. The proposed DML-approach improved the identification accuracy by 10% and 12% in comparison to DL-methods and other DML-approaches, respectively. Moreover, model embeddings from the two dataset types were merged in an organized way through a multi-view scheme to simultaneously exploit the information of surface and section fragments. Test with the resulting mixed model improves the identification accuracy by at least 3% and up to 30% with respect to DL-models and shallow machine learning methods, respectively.

{{</citation>}}


### (30/90) Deepfake Video Detection Using Generative Convolutional Vision Transformer (Deressa Wodajo et al., 2023)

{{<citation>}}

Deressa Wodajo, Solomon Atnafu, Zahid Akhtar. (2023)  
**Deepfake Video Detection Using Generative Convolutional Vision Transformer**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.07036v1)  

---


**ABSTRACT**  
Deepfakes have raised significant concerns due to their potential to spread false information and compromise digital media integrity. In this work, we propose a Generative Convolutional Vision Transformer (GenConViT) for deepfake video detection. Our model combines ConvNeXt and Swin Transformer models for feature extraction, and it utilizes Autoencoder and Variational Autoencoder to learn from the latent data distribution. By learning from the visual artifacts and latent data distribution, GenConViT achieves improved performance in detecting a wide range of deepfake videos. The model is trained and evaluated on DFDC, FF++, DeepfakeTIMIT, and Celeb-DF v2 datasets, achieving high classification accuracy, F1 scores, and AUC values. The proposed GenConViT model demonstrates robust performance in deepfake video detection, with an average accuracy of 95.8% and an AUC value of 99.3% across the tested datasets. Our proposed model addresses the challenge of generalizability in deepfake detection by leveraging visual and latent features and providing an effective solution for identifying a wide range of fake videos while preserving media integrity. The code for GenConViT is available at https://github.com/erprogs/GenConViT.

{{</citation>}}


### (31/90) Bridging the Gap: Heterogeneous Face Recognition with Conditional Adaptive Instance Modulation (Anjith George et al., 2023)

{{<citation>}}

Anjith George, Sebastien Marcel. (2023)  
**Bridging the Gap: Heterogeneous Face Recognition with Conditional Adaptive Instance Modulation**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.07032v1)  

---


**ABSTRACT**  
Heterogeneous Face Recognition (HFR) aims to match face images across different domains, such as thermal and visible spectra, expanding the applicability of Face Recognition (FR) systems to challenging scenarios. However, the domain gap and limited availability of large-scale datasets in the target domain make training robust and invariant HFR models from scratch difficult. In this work, we treat different modalities as distinct styles and propose a framework to adapt feature maps, bridging the domain gap. We introduce a novel Conditional Adaptive Instance Modulation (CAIM) module that can be integrated into pre-trained FR networks, transforming them into HFR networks. The CAIM block modulates intermediate feature maps, to adapt the style of the target modality effectively bridging the domain gap. Our proposed method allows for end-to-end training with a minimal number of paired samples. We extensively evaluate our approach on multiple challenging benchmarks, demonstrating superior performance compared to state-of-the-art methods. The source code and protocols for reproducing the findings will be made publicly available.

{{</citation>}}


### (32/90) HyperDreamBooth: HyperNetworks for Fast Personalization of Text-to-Image Models (Nataniel Ruiz et al., 2023)

{{<citation>}}

Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Wei Wei, Tingbo Hou, Yael Pritch, Neal Wadhwa, Michael Rubinstein, Kfir Aberman. (2023)  
**HyperDreamBooth: HyperNetworks for Fast Personalization of Text-to-Image Models**  

---
Primary Category: cs.CV
Categories: cs-AI, cs-CV, cs-GR, cs-LG, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.06949v1)  

---


**ABSTRACT**  
Personalization has emerged as a prominent aspect within the field of generative AI, enabling the synthesis of individuals in diverse contexts and styles, while retaining high-fidelity to their identities. However, the process of personalization presents inherent challenges in terms of time and memory requirements. Fine-tuning each personalized model needs considerable GPU time investment, and storing a personalized model per subject can be demanding in terms of storage capacity. To overcome these challenges, we propose HyperDreamBooth-a hypernetwork capable of efficiently generating a small set of personalized weights from a single image of a person. By composing these weights into the diffusion model, coupled with fast finetuning, HyperDreamBooth can generate a person's face in various contexts and styles, with high subject details while also preserving the model's crucial knowledge of diverse styles and semantic modifications. Our method achieves personalization on faces in roughly 20 seconds, 25x faster than DreamBooth and 125x faster than Textual Inversion, using as few as one reference image, with the same quality and style diversity as DreamBooth. Also our method yields a model that is 10000x smaller than a normal DreamBooth model. Project page: https://hyperdreambooth.github.io

{{</citation>}}


### (33/90) Video-FocalNets: Spatio-Temporal Focal Modulation for Video Action Recognition (Syed Talal Wasim et al., 2023)

{{<citation>}}

Syed Talal Wasim, Muhammad Uzair Khattak, Muzammal Naseer, Salman Khan, Mubarak Shah, Fahad Shahbaz Khan. (2023)  
**Video-FocalNets: Spatio-Temporal Focal Modulation for Video Action Recognition**  

---
Primary Category: cs.CV
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.06947v2)  

---


**ABSTRACT**  
Recent video recognition models utilize Transformer models for long-range spatio-temporal context modeling. Video transformer designs are based on self-attention that can model global context at a high computational cost. In comparison, convolutional designs for videos offer an efficient alternative but lack long-range dependency modeling. Towards achieving the best of both designs, this work proposes Video-FocalNet, an effective and efficient architecture for video recognition that models both local and global contexts. Video-FocalNet is based on a spatio-temporal focal modulation architecture that reverses the interaction and aggregation steps of self-attention for better efficiency. Further, the aggregation step and the interaction step are both implemented using efficient convolution and element-wise multiplication operations that are computationally less expensive than their self-attention counterparts on video representations. We extensively explore the design space of focal modulation-based spatio-temporal context modeling and demonstrate our parallel spatial and temporal encoding design to be the optimal choice. Video-FocalNets perform favorably well against the state-of-the-art transformer-based models for video recognition on three large-scale datasets (Kinetics-400, Kinetics-600, and SS-v2) at a lower computational cost. Our code/models are released at https://github.com/TalalWasim/Video-FocalNets.

{{</citation>}}


### (34/90) mBLIP: Efficient Bootstrapping of Multilingual Vision-LLMs (Gregor Geigle et al., 2023)

{{<citation>}}

Gregor Geigle, Abhay Jain, Radu Timofte, Goran Glavaš. (2023)  
**mBLIP: Efficient Bootstrapping of Multilingual Vision-LLMs**  

---
Primary Category: cs.CV
Categories: cs-CL, cs-CV, cs.CV  
Keywords: GLUE, Multilingual  
[Paper Link](http://arxiv.org/abs/2307.06930v1)  

---


**ABSTRACT**  
Modular vision-language models (Vision-LLMs) align pretrained image encoders with (pretrained) large language models (LLMs), representing a computationally much more efficient alternative to end-to-end training of large vision-language models from scratch, which is prohibitively expensive for most. Vision-LLMs instead post-hoc condition LLMs to `understand' the output of an image encoder. With the abundance of readily available high-quality English image-text data as well as monolingual English LLMs, the research focus has been on English-only Vision-LLMs. Multilingual vision-language models are still predominantly obtained via expensive end-to-end pretraining, resulting in comparatively smaller models, trained on limited multilingual image data supplemented with text-only multilingual corpora. In this work, we present mBLIP, the first multilingual Vision-LLM, which we obtain in a computationally efficient manner -- on consumer hardware using only a few million training examples -- by leveraging a pretrained multilingual LLM. To this end, we \textit{re-align} an image encoder previously tuned to an English LLM to a new, multilingual LLM -- for this, we leverage multilingual data from a mix of vision-and-language tasks, which we obtain by machine-translating high-quality English data to 95 languages. On the IGLUE benchmark, mBLIP yields results competitive with state-of-the-art models. Moreover, in image captioning on XM3600, mBLIP (zero-shot) even outperforms PaLI-X (a model with 55B parameters). Compared to these very large multilingual vision-language models trained from scratch, we obtain mBLIP by training orders of magnitude fewer parameters on magnitudes less data. We release our model and code at \url{https://github.com/gregor-ge/mBLIP}.

{{</citation>}}


### (35/90) Multimodal Object Detection in Remote Sensing (Abdelbadie Belmouhcine et al., 2023)

{{<citation>}}

Abdelbadie Belmouhcine, Jean-Christophe Burnel, Luc Courtrai, Minh-Tan Pham, Sébastien Lefèvre. (2023)  
**Multimodal Object Detection in Remote Sensing**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2307.06724v1)  

---


**ABSTRACT**  
Object detection in remote sensing is a crucial computer vision task that has seen significant advancements with deep learning techniques. However, most existing works in this area focus on the use of generic object detection and do not leverage the potential of multimodal data fusion. In this paper, we present a comparison of methods for multimodal object detection in remote sensing, survey available multimodal datasets suitable for evaluation, and discuss future directions.

{{</citation>}}


### (36/90) YOLIC: An Efficient Method for Object Localization and Classification on Edge Devices (Kai Su et al., 2023)

{{<citation>}}

Kai Su, Qiangfu Zhao, Yoichi Tomioka, Yong Liu. (2023)  
**YOLIC: An Efficient Method for Object Localization and Classification on Edge Devices**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.06689v2)  

---


**ABSTRACT**  
In the realm of Tiny AI, we introduce "You Only Look at Interested Cells" (YOLIC), an efficient method for object localization and classification on edge devices. Seamlessly blending the strengths of semantic segmentation and object detection, YOLIC offers superior computational efficiency and precision. By adopting Cells of Interest for classification instead of individual pixels, YOLIC encapsulates relevant information, reduces computational load, and enables rough object shape inference. Importantly, the need for bounding box regression is obviated, as YOLIC capitalizes on the predetermined cell configuration that provides information about potential object location, size, and shape. To tackle the issue of single-label classification limitations, a multi-label classification approach is applied to each cell, effectively recognizing overlapping or closely situated objects. This paper presents extensive experiments on multiple datasets, demonstrating that YOLIC achieves detection performance comparable to the state-of-the-art YOLO algorithms while surpassing in speed, exceeding 30fps on a Raspberry Pi 4B CPU. All resources related to this study, including datasets, cell designer, image annotation tool, and source code, have been made publicly available on our project website at https://kai3316.github.io/yolic.github.io

{{</citation>}}


### (37/90) DGCNet: An Efficient 3D-Densenet based on Dynamic Group Convolution for Hyperspectral Remote Sensing Image Classification (Guandong Li, 2023)

{{<citation>}}

Guandong Li. (2023)  
**DGCNet: An Efficient 3D-Densenet based on Dynamic Group Convolution for Hyperspectral Remote Sensing Image Classification**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2307.06667v1)  

---


**ABSTRACT**  
Deep neural networks face many problems in the field of hyperspectral image classification, lack of effective utilization of spatial spectral information, gradient disappearance and overfitting as the model depth increases. In order to accelerate the deployment of the model on edge devices with strict latency requirements and limited computing power, we introduce a lightweight model based on the improved 3D-Densenet model and designs DGCNet. It improves the disadvantage of group convolution. Referring to the idea of dynamic network, dynamic group convolution(DGC) is designed on 3d convolution kernel. DGC introduces small feature selectors for each grouping to dynamically decide which part of the input channel to connect based on the activations of all input channels. Multiple groups can capture different and complementary visual and semantic features of input images, allowing convolution neural network(CNN) to learn rich features. 3D convolution extracts high-dimensional and redundant hyperspectral data, and there is also a lot of redundant information between convolution kernels. DGC module allows 3D-Densenet to select channel information with richer semantic features and discard inactive regions. The 3D-CNN passing through the DGC module can be regarded as a pruned network. DGC not only allows 3D-CNN to complete sufficient feature extraction, but also takes into account the requirements of speed and calculation amount. The inference speed and accuracy have been improved, with outstanding performance on the IN, Pavia and KSC datasets, ahead of the mainstream hyperspectral image classification methods.

{{</citation>}}


### (38/90) Transformer-based end-to-end classification of variable-length volumetric data (Marzieh Oghbaie et al., 2023)

{{<citation>}}

Marzieh Oghbaie, Teresa Araujo, Taha Emre, Ursula Schmidt-Erfurth, Hrvoje Bogunovic. (2023)  
**Transformer-based end-to-end classification of variable-length volumetric data**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV, eess-IV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.06666v1)  

---


**ABSTRACT**  
The automatic classification of 3D medical data is memory-intensive. Also, variations in the number of slices between samples is common. Naive solutions such as subsampling can solve these problems, but at the cost of potentially eliminating relevant diagnosis information. Transformers have shown promising performance for sequential data analysis. However, their application for long-sequences is data, computationally, and memory demanding. In this paper, we propose an end-to-end Transformer-based framework that allows to classify volumetric data of variable length in an efficient fashion. Particularly, by randomizing the input slice-wise resolution during training, we enhance the capacity of the learnable positional embedding assigned to each volume slice. Consequently, the accumulated positional information in each positional embedding can be generalized to the neighbouring slices, even for high resolution volumes at the test time. By doing so, the model will be more robust to variable volume length and amenable to different computational budgets. We evaluated the proposed approach in retinal OCT volume classification and achieved 21.96% average improvement in balanced accuracy on a 9-class diagnostic task, compared to state-of-the-art video transformers. Our findings show that varying the slice-wise resolution of the input during training results in more informative volume representation as compared to training with fixed number of slices per volume. Our code is available at: https://github.com/marziehoghbaie/VLFAT.

{{</citation>}}


### (39/90) Image Transformation Sequence Retrieval with General Reinforcement Learning (Enrique Mas-Candela et al., 2023)

{{<citation>}}

Enrique Mas-Candela, Antonio Ríos-Vila, Jorge Calvo-Zaragoza. (2023)  
**Image Transformation Sequence Retrieval with General Reinforcement Learning**  

---
Primary Category: cs.CV
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.06630v1)  

---


**ABSTRACT**  
In this work, the novel Image Transformation Sequence Retrieval (ITSR) task is presented, in which a model must retrieve the sequence of transformations between two given images that act as source and target, respectively. Given certain characteristics of the challenge such as the multiplicity of a correct sequence or the correlation between consecutive steps of the process, we propose a solution to ITSR using a general model-based Reinforcement Learning such as Monte Carlo Tree Search (MCTS), which is combined with a deep neural network. Our experiments provide a benchmark in both synthetic and real domains, where the proposed approach is compared with supervised training. The results report that a model trained with MCTS is able to outperform its supervised counterpart in both the simplest and the most complex cases. Our work draws interesting conclusions about the nature of ITSR and its associated challenges.

{{</citation>}}


### (40/90) A Study on Differentiable Logic and LLMs for EPIC-KITCHENS-100 Unsupervised Domain Adaptation Challenge for Action Recognition 2023 (Yi Cheng et al., 2023)

{{<citation>}}

Yi Cheng, Ziwei Xu, Fen Fang, Dongyun Lin, Hehe Fan, Yongkang Wong, Ying Sun, Mohan Kankanhalli. (2023)  
**A Study on Differentiable Logic and LLMs for EPIC-KITCHENS-100 Unsupervised Domain Adaptation Challenge for Action Recognition 2023**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: GPT, GPT-3.5, Language Model  
[Paper Link](http://arxiv.org/abs/2307.06569v1)  

---


**ABSTRACT**  
In this technical report, we present our findings from a study conducted on the EPIC-KITCHENS-100 Unsupervised Domain Adaptation task for Action Recognition. Our research focuses on the innovative application of a differentiable logic loss in the training to leverage the co-occurrence relations between verb and noun, as well as the pre-trained Large Language Models (LLMs) to generate the logic rules for the adaptation to unseen action labels. Specifically, the model's predictions are treated as the truth assignment of a co-occurrence logic formula to compute the logic loss, which measures the consistency between the predictions and the logic constraints. By using the verb-noun co-occurrence matrix generated from the dataset, we observe a moderate improvement in model performance compared to our baseline framework. To further enhance the model's adaptability to novel action labels, we experiment with rules generated using GPT-3.5, which leads to a slight decrease in performance. These findings shed light on the potential and challenges of incorporating differentiable logic and LLMs for knowledge extraction in unsupervised domain adaptation for action recognition. Our final submission (entitled `NS-LLM') achieved the first place in terms of top-1 action recognition accuracy.

{{</citation>}}


### (41/90) Regression-Oriented Knowledge Distillation for Lightweight Ship Orientation Angle Prediction with Optical Remote Sensing Images (Zhan Shi et al., 2023)

{{<citation>}}

Zhan Shi, Xin Ding, Peng Ding, Chun Yang, Ru Huang, Xiaoxuan Song. (2023)  
**Regression-Oriented Knowledge Distillation for Lightweight Ship Orientation Angle Prediction with Optical Remote Sensing Images**  

---
Primary Category: cs.CV
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2307.06566v1)  

---


**ABSTRACT**  
Ship orientation angle prediction (SOAP) with optical remote sensing images is an important image processing task, which often relies on deep convolutional neural networks (CNNs) to make accurate predictions. This paper proposes a novel framework to reduce the model sizes and computational costs of SOAP models without harming prediction accuracy. First, a new SOAP model called Mobile-SOAP is designed based on MobileNetV2, achieving state-of-the-art prediction accuracy. Four tiny SOAP models are also created by replacing the convolutional blocks in Mobile-SOAP with four small-scale networks, respectively. Then, to transfer knowledge from Mobile-SOAP to four lightweight models, we propose a novel knowledge distillation (KD) framework termed SOAP-KD consisting of a novel feature-based guidance loss and an optimized synthetic samples-based knowledge transfer mechanism. Lastly, extensive experiments on the FGSC-23 dataset confirm the superiority of Mobile-SOAP over existing models and also demonstrate the effectiveness of SOAP-KD in improving the prediction performance of four specially designed tiny models. Notably, by using SOAP-KD, the test mean absolute error of the ShuffleNetV2x1.0-based model is only 8% higher than that of Mobile-SOAP, but its number of parameters and multiply-accumulate operations (MACs) are respectively 61.6% and 60.8% less.

{{</citation>}}


### (42/90) Multi-objective Evolutionary Search of Variable-length Composite Semantic Perturbations (Jialiang Sun et al., 2023)

{{<citation>}}

Jialiang Sun, Wen Yao, Tingsong Jiang, Xiaoqian Chen. (2023)  
**Multi-objective Evolutionary Search of Variable-length Composite Semantic Perturbations**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2307.06548v2)  

---


**ABSTRACT**  
Deep neural networks have proven to be vulnerable to adversarial attacks in the form of adding specific perturbations on images to make wrong outputs. Designing stronger adversarial attack methods can help more reliably evaluate the robustness of DNN models. To release the harbor burden and improve the attack performance, auto machine learning (AutoML) has recently emerged as one successful technique to help automatically find the near-optimal adversarial attack strategy. However, existing works about AutoML for adversarial attacks only focus on $L_{\infty}$-norm-based perturbations. In fact, semantic perturbations attract increasing attention due to their naturalnesses and physical realizability. To bridge the gap between AutoML and semantic adversarial attacks, we propose a novel method called multi-objective evolutionary search of variable-length composite semantic perturbations (MES-VCSP). Specifically, we construct the mathematical model of variable-length composite semantic perturbations, which provides five gradient-based semantic attack methods. The same type of perturbation in an attack sequence is allowed to be performed multiple times. Besides, we introduce the multi-objective evolutionary search consisting of NSGA-II and neighborhood search to find near-optimal variable-length attack sequences. Experimental results on CIFAR10 and ImageNet datasets show that compared with existing methods, MES-VCSP can obtain adversarial examples with a higher attack success rate, more naturalness, and less time cost.

{{</citation>}}


### (43/90) Microbial Genetic Algorithm-based Black-box Attack against Interpretable Deep Learning Systems (Eldor Abdukhamidov et al., 2023)

{{<citation>}}

Eldor Abdukhamidov, Mohammed Abuhamad, Simon S. Woo, Eric Chan-Tin, Tamer Abuhmed. (2023)  
**Microbial Genetic Algorithm-based Black-box Attack against Interpretable Deep Learning Systems**  

---
Primary Category: cs.CV
Categories: cs-AI, cs-CR, cs-CV, cs-LG, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2307.06496v1)  

---


**ABSTRACT**  
Deep learning models are susceptible to adversarial samples in white and black-box environments. Although previous studies have shown high attack success rates, coupling DNN models with interpretation models could offer a sense of security when a human expert is involved, who can identify whether a given sample is benign or malicious. However, in white-box environments, interpretable deep learning systems (IDLSes) have been shown to be vulnerable to malicious manipulations. In black-box settings, as access to the components of IDLSes is limited, it becomes more challenging for the adversary to fool the system. In this work, we propose a Query-efficient Score-based black-box attack against IDLSes, QuScore, which requires no knowledge of the target model and its coupled interpretation model. QuScore is based on transfer-based and score-based methods by employing an effective microbial genetic algorithm. Our method is designed to reduce the number of queries necessary to carry out successful attacks, resulting in a more efficient process. By continuously refining the adversarial samples created based on feedback scores from the IDLS, our approach effectively navigates the search space to identify perturbations that can fool the system. We evaluate the attack's effectiveness on four CNN models (Inception, ResNet, VGG, DenseNet) and two interpretation models (CAM, Grad), using both ImageNet and CIFAR datasets. Our results show that the proposed approach is query-efficient with a high attack success rate that can reach between 95% and 100% and transferability with an average success rate of 69% in the ImageNet and CIFAR datasets. Our attack method generates adversarial examples with attribution maps that resemble benign samples. We have also demonstrated that our attack is resilient against various preprocessing defense techniques and can easily be transferred to different DNN models.

{{</citation>}}


## cs.CL (22)



### (44/90) Leveraging Pretrained ASR Encoders for Effective and Efficient End-to-End Speech Intent Classification and Slot Filling (He Huang et al., 2023)

{{<citation>}}

He Huang, Jagadeesh Balam, Boris Ginsburg. (2023)  
**Leveraging Pretrained ASR Encoders for Effective and Efficient End-to-End Speech Intent Classification and Slot Filling**  

---
Primary Category: cs.CL
Categories: cs-CL, cs-CV, cs-SD, cs.CL, eess-AS  
Keywords: NLU, Transformer  
[Paper Link](http://arxiv.org/abs/2307.07057v1)  

---


**ABSTRACT**  
We study speech intent classification and slot filling (SICSF) by proposing to use an encoder pretrained on speech recognition (ASR) to initialize an end-to-end (E2E) Conformer-Transformer model, which achieves the new state-of-the-art results on the SLURP dataset, with 90.14% intent accuracy and 82.27% SLURP-F1. We compare our model with encoders pretrained on self-supervised learning (SSL), and show that ASR pretraining is much more effective than SSL for SICSF. To explore parameter efficiency, we freeze the encoder and add Adapter modules, and show that parameter efficiency is only achievable with an ASR-pretrained encoder, while the SSL encoder needs full finetuning to achieve comparable results. In addition, we provide an in-depth comparison on end-to-end models versus cascading models (ASR+NLU), and show that E2E models are better than cascaded models unless an oracle ASR model is provided. Last but not least, our model is the first E2E model that achieves the same performance as cascading models with oracle ASR. Code, checkpoints and configs are available.

{{</citation>}}


### (45/90) Making the Most Out of the Limited Context Length: Predictive Power Varies with Clinical Note Type and Note Section (Hongyi Zheng et al., 2023)

{{<citation>}}

Hongyi Zheng, Yixin Zhu, Lavender Yao Jiang, Kyunghyun Cho, Eric Karl Oermann. (2023)  
**Making the Most Out of the Limited Context Length: Predictive Power Varies with Clinical Note Type and Note Section**  

---
Primary Category: cs.CL
Categories: cs-CL, cs-IR, cs-LG, cs.CL  
Keywords: Clinical  
[Paper Link](http://arxiv.org/abs/2307.07051v1)  

---


**ABSTRACT**  
Recent advances in large language models have led to renewed interest in natural language processing in healthcare using the free text of clinical notes. One distinguishing characteristic of clinical notes is their long time span over multiple long documents. The unique structure of clinical notes creates a new design choice: when the context length for a language model predictor is limited, which part of clinical notes should we choose as the input? Existing studies either choose the inputs with domain knowledge or simply truncate them. We propose a framework to analyze the sections with high predictive power. Using MIMIC-III, we show that: 1) predictive power distribution is different between nursing notes and discharge notes and 2) combining different types of notes could improve performance when the context length is large. Our findings suggest that a carefully selected sampling function could enable more efficient information extraction from clinical notes.

{{</citation>}}


### (46/90) MegaWika: Millions of reports and their sources across 50 diverse languages (Samuel Barham et al., 2023)

{{<citation>}}

Samuel Barham, Orion Weller, Michelle Yuan, Kenton Murray, Mahsa Yarmohammadi, Zhengping Jiang, Siddharth Vashishtha, Alexander Martin, Anqi Liu, Aaron Steven White, Jordan Boyd-Graber, Benjamin Van Durme. (2023)  
**MegaWika: Millions of reports and their sources across 50 diverse languages**  

---
Primary Category: cs.CL
Categories: I-2-7, cs-CL, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.07049v1)  

---


**ABSTRACT**  
To foster the development of new models for collaborative AI-assisted report generation, we introduce MegaWika, consisting of 13 million Wikipedia articles in 50 diverse languages, along with their 71 million referenced source materials. We process this dataset for a myriad of applications, going beyond the initial Wikipedia citation extraction and web scraping of content, including translating non-English articles for cross-lingual applications and providing FrameNet parses for automated semantic analysis. MegaWika is the largest resource for sentence-level report generation and the only report generation dataset that is multilingual. We manually analyze the quality of this resource through a semantically stratified sample. Finally, we provide baseline results and trained models for crucial steps in automated report generation: cross-lingual question answering and citation retrieval.

{{</citation>}}


### (47/90) DIALGEN: Collaborative Human-LM Generated Dialogues for Improved Understanding of Human-Human Conversations (Bo-Ru Lu et al., 2023)

{{<citation>}}

Bo-Ru Lu, Nikita Haduong, Chia-Hsuan Lee, Zeqiu Wu, Hao Cheng, Paul Koester, Jean Utke, Tao Yu, Noah A. Smith, Mari Ostendorf. (2023)  
**DIALGEN: Collaborative Human-LM Generated Dialogues for Improved Understanding of Human-Human Conversations**  

---
Primary Category: cs.CL
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, Dialog, Dialogue, GPT  
[Paper Link](http://arxiv.org/abs/2307.07047v1)  

---


**ABSTRACT**  
Applications that could benefit from automatic understanding of human-human conversations often come with challenges associated with private information in real-world data such as call center or clinical conversations. Working with protected data also increases costs of annotation, which limits technology development. To address these challenges, we propose DIALGEN, a human-in-the-loop semi-automated dialogue generation framework. DIALGEN uses a language model (ChatGPT) that can follow schema and style specifications to produce fluent conversational text, generating a complex conversation through iteratively generating subdialogues and using human feedback to correct inconsistencies or redirect the flow. In experiments on structured summarization of agent-client information gathering calls, framed as dialogue state tracking, we show that DIALGEN data enables significant improvement in model performance.

{{</citation>}}


### (48/90) Data Augmentation for Machine Translation via Dependency Subtree Swapping (Attila Nagy et al., 2023)

{{<citation>}}

Attila Nagy, Dorina Petra Lakatos, Botond Barta, Patrick Nanys, Judit Ács. (2023)  
**Data Augmentation for Machine Translation via Dependency Subtree Swapping**  

---
Primary Category: cs.CL
Categories: cs-CL, cs.CL  
Keywords: Augmentation, BLEU, Machine Translation  
[Paper Link](http://arxiv.org/abs/2307.07025v1)  

---


**ABSTRACT**  
We present a generic framework for data augmentation via dependency subtree swapping that is applicable to machine translation. We extract corresponding subtrees from the dependency parse trees of the source and target sentences and swap these across bisentences to create augmented samples. We perform thorough filtering based on graphbased similarities of the dependency trees and additional heuristics to ensure that extracted subtrees correspond to the same meaning. We conduct resource-constrained experiments on 4 language pairs in both directions using the IWSLT text translation datasets and the Hunglish2 corpus. The results demonstrate consistent improvements in BLEU score over our baseline models in 3 out of 4 language pairs. Our code is available on GitHub.

{{</citation>}}


### (49/90) Electoral Agitation Data Set: The Use Case of the Polish Election (Mateusz Baran et al., 2023)

{{<citation>}}

Mateusz Baran, Mateusz Wójcik, Piotr Kolebski, Michał Bernaczyk, Krzysztof Rajda, Łukasz Augustyniak, Tomasz Kajdanowicz. (2023)  
**Electoral Agitation Data Set: The Use Case of the Polish Election**  

---
Primary Category: cs.CL
Categories: cs-CL, cs.CL  
Keywords: BERT, Language Model, Twitter  
[Paper Link](http://arxiv.org/abs/2307.07007v1)  

---


**ABSTRACT**  
The popularity of social media makes politicians use it for political advertisement. Therefore, social media is full of electoral agitation (electioneering), especially during the election campaigns. The election administration cannot track the spread and quantity of messages that count as agitation under the election code. It addresses a crucial problem, while also uncovering a niche that has not been effectively targeted so far. Hence, we present the first publicly open data set for detecting electoral agitation in the Polish language. It contains 6,112 human-annotated tweets tagged with four legally conditioned categories. We achieved a 0.66 inter-annotator agreement (Cohen's kappa score). An additional annotator resolved the mismatches between the first two improving the consistency and complexity of the annotation process. The newly created data set was used to fine-tune a Polish Language Model called HerBERT (achieving a 68% F1 score). We also present a number of potential use cases for such data sets and models, enriching the paper with an analysis of the Polish 2020 Presidential Election on Twitter.

{{</citation>}}


### (50/90) Classical Out-of-Distribution Detection Methods Benchmark in Text Classification Tasks (Mateusz Baran et al., 2023)

{{<citation>}}

Mateusz Baran, Joanna Baran, Mateusz Wójcik, Maciej Zięba, Adam Gonczarek. (2023)  
**Classical Out-of-Distribution Detection Methods Benchmark in Text Classification Tasks**  

---
Primary Category: cs.CL
Categories: cs-AI, cs-CL, cs.CL  
Keywords: NLP, Text Classification  
[Paper Link](http://arxiv.org/abs/2307.07002v1)  

---


**ABSTRACT**  
State-of-the-art models can perform well in controlled environments, but they often struggle when presented with out-of-distribution (OOD) examples, making OOD detection a critical component of NLP systems. In this paper, we focus on highlighting the limitations of existing approaches to OOD detection in NLP. Specifically, we evaluated eight OOD detection methods that are easily integrable into existing NLP systems and require no additional OOD data or model modifications. One of our contributions is providing a well-structured research environment that allows for full reproducibility of the results. Additionally, our analysis shows that existing OOD detection methods for NLP tasks are not yet sufficiently sensitive to capture all samples characterized by various types of distributional shifts. Particularly challenging testing scenarios arise in cases of background shift and randomly shuffled word order within in domain texts. This highlights the need for future work to develop more effective OOD detection approaches for the NLP problems, and our work provides a well-defined foundation for further research in this area.

{{</citation>}}


### (51/90) In-context Autoencoder for Context Compression in a Large Language Model (Tao Ge et al., 2023)

{{<citation>}}

Tao Ge, Jing Hu, Xun Wang, Si-Qing Chen, Furu Wei. (2023)  
**In-context Autoencoder for Context Compression in a Large Language Model**  

---
Primary Category: cs.CL
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.06945v1)  

---


**ABSTRACT**  
We propose the In-context Autoencoder (ICAE) for context compression in a large language model (LLM). The ICAE has two modules: a learnable encoder adapted with LoRA from an LLM for compressing a long context into a limited number of memory slots, and a fixed decoder which is the target LLM that can condition on the memory slots for various purposes. We first pretrain the ICAE using both autoencoding and language modeling objectives on massive text data, enabling it to generate memory slots that accurately and comprehensively represent the original context. Then, we fine-tune the pretrained ICAE on a small amount of instruct data to enhance its interaction with various prompts for producing desirable responses. Our experimental results demonstrate that the ICAE learned with our proposed pretraining and fine-tuning paradigm can effectively produce memory slots with $4\times$ context compression, which can be well conditioned on by the target LLM to respond to various prompts. The promising results demonstrate significant implications of the ICAE for its novel approach to the long context problem and its potential to reduce computation and memory overheads for LLM inference in practice, suggesting further research effort in context management for an LLM. Our code and data will be released shortly.

{{</citation>}}


### (52/90) Towards Populating Generalizable Engineering Design Knowledge (L Siddharth et al., 2023)

{{<citation>}}

L Siddharth, Jianxi Luo. (2023)  
**Towards Populating Generalizable Engineering Design Knowledge**  

---
Primary Category: cs.CL
Categories: cs-CL, cs-DB, cs-IR, cs.CL  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2307.06985v1)  

---


**ABSTRACT**  
Aiming to populate generalizable engineering design knowledge, we propose a method to extract facts of the form head entity :: relationship :: tail entity from sentences found in patent documents. These facts could be combined within and across patent documents to form knowledge graphs that serve as schemes for representing as well as storing design knowledge. Existing methods in engineering design literature often utilise a set of predefined relationships to populate triples that are statistical approximations rather than facts. In our method, we train a tagger to identify both entities and relationships from a sentence. Given a pair of entities thus identified, we train another tagger to identify the relationship tokens that specifically denote the relationship between the pair. For training these taggers, we manually construct a dataset of 44,227 sentences and corresponding facts. We also compare the performance of the method against typically recommended approaches, wherein, we predict the edges among tokens by pairing the tokens independently and as part of a graph. We apply our method to sentences found in patents related to fan systems and build a domain knowledge base. Upon providing an overview of the knowledge base, we search for solutions relevant to some key issues prevailing in fan systems. We organize the responses into knowledge graphs and hold a comparative discussion against the opinions from ChatGPT.

{{</citation>}}


### (53/90) Generating Benchmarks for Factuality Evaluation of Language Models (Dor Muhlgay et al., 2023)

{{<citation>}}

Dor Muhlgay, Ori Ram, Inbal Magar, Yoav Levine, Nir Ratner, Yonatan Belinkov, Omri Abend, Kevin Leyton-Brown, Amnon Shashua, Yoav Shoham. (2023)  
**Generating Benchmarks for Factuality Evaluation of Language Models**  

---
Primary Category: cs.CL
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2307.06908v1)  

---


**ABSTRACT**  
Before deploying a language model (LM) within a given domain, it is important to measure its tendency to generate factually incorrect information in that domain. Existing factual generation evaluation methods focus on facts sampled from the LM itself, and thus do not control the set of evaluated facts and might under-represent rare and unlikely facts. We propose FACTOR: Factual Assessment via Corpus TransfORmation, a scalable approach for evaluating LM factuality. FACTOR automatically transforms a factual corpus of interest into a benchmark evaluating an LM's propensity to generate true facts from the corpus vs. similar but incorrect statements. We use our framework to create two benchmarks: Wiki-FACTOR and News-FACTOR. We show that: (i) our benchmark scores increase with model size and improve when the LM is augmented with retrieval; (ii) benchmark score correlates with perplexity, but the two metrics do not always agree on model ranking; and (iii) when perplexity and benchmark score disagree, the latter better reflects factuality in open-ended generation, as measured by human annotators. We make our data and code publicly available in https://github.com/AI21Labs/factor.

{{</citation>}}


### (54/90) DecompEval: Evaluating Generated Texts as Unsupervised Decomposed Question Answering (Pei Ke et al., 2023)

{{<citation>}}

Pei Ke, Fei Huang, Fei Mi, Yasheng Wang, Qun Liu, Xiaoyan Zhu, Minlie Huang. (2023)  
**DecompEval: Evaluating Generated Texts as Unsupervised Decomposed Question Answering**  

---
Primary Category: cs.CL
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Question Answering  
[Paper Link](http://arxiv.org/abs/2307.06869v1)  

---


**ABSTRACT**  
Existing evaluation metrics for natural language generation (NLG) tasks face the challenges on generalization ability and interpretability. Specifically, most of the well-performed metrics are required to train on evaluation datasets of specific NLG tasks and evaluation dimensions, which may cause over-fitting to task-specific datasets. Furthermore, existing metrics only provide an evaluation score for each dimension without revealing the evidence to interpret how this score is obtained. To deal with these challenges, we propose a simple yet effective metric called DecompEval. This metric formulates NLG evaluation as an instruction-style question answering task and utilizes instruction-tuned pre-trained language models (PLMs) without training on evaluation datasets, aiming to enhance the generalization ability. To make the evaluation process more interpretable, we decompose our devised instruction-style question about the quality of generated texts into the subquestions that measure the quality of each sentence. The subquestions with their answers generated by PLMs are then recomposed as evidence to obtain the evaluation result. Experimental results show that DecompEval achieves state-of-the-art performance in untrained metrics for evaluating text summarization and dialogue generation, which also exhibits strong dimension-level / task-level generalization ability and interpretability.

{{</citation>}}


### (55/90) Negated Complementary Commonsense using Large Language Models (Navid Rezaei et al., 2023)

{{<citation>}}

Navid Rezaei, Marek Z. Reformat. (2023)  
**Negated Complementary Commonsense using Large Language Models**  

---
Primary Category: cs.CL
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2307.06794v1)  

---


**ABSTRACT**  
Larger language models, such as GPT-3, have shown to be excellent in many tasks. However, we demonstrate that out-of-ordinary questions can throw the model off guard. This work focuses on finding answers to negated complementary questions in commonsense scenarios. We illustrate how such questions adversely affect the model responses. We propose a model-agnostic methodology to improve the performance in negated complementary scenarios. Our method outperforms few-shot generation from GPT-3 (by more than 11 points) and, more importantly, highlights the significance of studying the response of large language models in negated complementary questions. The code, data, and experiments are available under: https://github.com/navidre/negated_complementary_commonsense.

{{</citation>}}


### (56/90) Tackling Fake News in Bengali: Unraveling the Impact of Summarization vs. Augmentation on Pre-trained Language Models (Arman Sakif Chowdhury et al., 2023)

{{<citation>}}

Arman Sakif Chowdhury, G. M. Shahariar, Ahammed Tarik Aziz, Syed Mohibul Alam, Md. Azad Sheikh, Tanveer Ahmed Belal. (2023)  
**Tackling Fake News in Bengali: Unraveling the Impact of Summarization vs. Augmentation on Pre-trained Language Models**  

---
Primary Category: cs.CL
Categories: cs-CL, cs.CL  
Keywords: Augmentation, BERT, Fake News, Language Model, Summarization  
[Paper Link](http://arxiv.org/abs/2307.06979v1)  

---


**ABSTRACT**  
With the rise of social media and online news sources, fake news has become a significant issue globally. However, the detection of fake news in low resource languages like Bengali has received limited attention in research. In this paper, we propose a methodology consisting of four distinct approaches to classify fake news articles in Bengali using summarization and augmentation techniques with five pre-trained language models. Our approach includes translating English news articles and using augmentation techniques to curb the deficit of fake news articles. Our research also focused on summarizing the news to tackle the token length limitation of BERT based models. Through extensive experimentation and rigorous evaluation, we show the effectiveness of summarization and augmentation in the case of Bengali fake news detection. We evaluated our models using three separate test datasets. The BanglaBERT Base model, when combined with augmentation techniques, achieved an impressive accuracy of 96% on the first test dataset. On the second test dataset, the BanglaBERT model, trained with summarized augmented news articles achieved 97% accuracy. Lastly, the mBERT Base model achieved an accuracy of 86% on the third test dataset which was reserved for generalization performance evaluation. The datasets and implementations are available at https://github.com/arman-sakif/Bengali-Fake-News-Detection

{{</citation>}}


### (57/90) Why Guided Dialog Policy Learning performs well? Understanding the role of adversarial learning and its alternative (Sho Shimoyama et al., 2023)

{{<citation>}}

Sho Shimoyama, Tetsuro Morimura, Kenshi Abe, Toda Takamichi, Yuta Tomomatsu, Masakazu Sugiyama, Asahi Hentona, Yuuki Azuma, Hirotaka Ninomiya. (2023)  
**Why Guided Dialog Policy Learning performs well? Understanding the role of adversarial learning and its alternative**  

---
Primary Category: cs.CL
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Dialog  
[Paper Link](http://arxiv.org/abs/2307.06721v1)  

---


**ABSTRACT**  
Dialog policies, which determine a system's action based on the current state at each dialog turn, are crucial to the success of the dialog. In recent years, reinforcement learning (RL) has emerged as a promising option for dialog policy learning (DPL). In RL-based DPL, dialog policies are updated according to rewards. The manual construction of fine-grained rewards, such as state-action-based ones, to effectively guide the dialog policy is challenging in multi-domain task-oriented dialog scenarios with numerous state-action pair combinations. One way to estimate rewards from collected data is to train the reward estimator and dialog policy simultaneously using adversarial learning (AL). Although this method has demonstrated superior performance experimentally, it is fraught with the inherent problems of AL, such as mode collapse. This paper first identifies the role of AL in DPL through detailed analyses of the objective functions of dialog policy and reward estimator. Next, based on these analyses, we propose a method that eliminates AL from reward estimation and DPL while retaining its advantages. We evaluate our method using MultiWOZ, a multi-domain task-oriented dialog corpus.

{{</citation>}}


### (58/90) Unsupervised Calibration through Prior Adaptation for Text Classification using Large Language Models (Lautaro Estienne, 2023)

{{<citation>}}

Lautaro Estienne. (2023)  
**Unsupervised Calibration through Prior Adaptation for Text Classification using Large Language Models**  

---
Primary Category: cs.CL
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Language Model, Text Classification  
[Paper Link](http://arxiv.org/abs/2307.06713v1)  

---


**ABSTRACT**  
A wide variety of natural language tasks are currently being addressed with large-scale language models (LLMs). These models are usually trained with a very large amount of unsupervised text data and adapted to perform a downstream natural language task using methods like fine-tuning, calibration or in-context learning. In this work, we propose an approach to adapt the prior class distribution to perform text classification tasks without the need for labelled samples and only few in-domain sample queries. The proposed approach treats the LLM as a black box, adding a stage where the model posteriors are calibrated to the task. Results show that these methods outperform the un-adapted model for different number of training shots in the prompt and a previous approach were calibration is performed without using any adaptation data.

{{</citation>}}


### (59/90) To share or not to share: What risks would laypeople accept to give sensitive data to differentially-private NLP systems? (Christopher Weiss et al., 2023)

{{<citation>}}

Christopher Weiss, Frauke Kreuter, Ivan Habernal. (2023)  
**To share or not to share: What risks would laypeople accept to give sensitive data to differentially-private NLP systems?**  

---
Primary Category: cs.CL
Categories: cs-CL, cs-CR, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2307.06708v1)  

---


**ABSTRACT**  
Although the NLP community has adopted central differential privacy as a go-to framework for privacy-preserving model training or data sharing, the choice and interpretation of the key parameter, privacy budget $\varepsilon$ that governs the strength of privacy protection, remains largely arbitrary. We argue that determining the $\varepsilon$ value should not be solely in the hands of researchers or system developers, but must also take into account the actual people who share their potentially sensitive data. In other words: Would you share your instant messages for $\varepsilon$ of 10? We address this research gap by designing, implementing, and conducting a behavioral experiment (311 lay participants) to study the behavior of people in uncertain decision-making situations with respect to privacy-threatening situations. Framing the risk perception in terms of two realistic NLP scenarios and using a vignette behavioral study help us determine what $\varepsilon$ thresholds would lead lay people to be willing to share sensitive textual data - to our knowledge, the first study of its kind.

{{</citation>}}


### (60/90) Intent-calibrated Self-training for Answer Selection in Open-domain Dialogues (Wentao Deng et al., 2023)

{{<citation>}}

Wentao Deng, Jiahuan Pei, Zhaochun Ren, Zhumin Chen, Pengjie Ren. (2023)  
**Intent-calibrated Self-training for Answer Selection in Open-domain Dialogues**  

---
Primary Category: cs.CL
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2307.06703v1)  

---


**ABSTRACT**  
Answer selection in open-domain dialogues aims to select an accurate answer from candidates. Recent success of answer selection models hinges on training with large amounts of labeled data. However, collecting large-scale labeled data is labor-intensive and time-consuming. In this paper, we introduce the predicted intent labels to calibrate answer labels in a self-training paradigm. Specifically, we propose the intent-calibrated self-training (ICAST) to improve the quality of pseudo answer labels through the intent-calibrated answer selection paradigm, in which we employ pseudo intent labels to help improve pseudo answer labels. We carry out extensive experiments on two benchmark datasets with open-domain dialogues. The experimental results show that ICAST outperforms baselines consistently with 1%, 5% and 10% labeled data. Specifically, it improves 2.06% and 1.00% of F1 score on the two datasets, compared with the strongest baseline with only 5% labeled data.

{{</citation>}}


### (61/90) Convolutional Neural Networks for Sentiment Analysis on Weibo Data: A Natural Language Processing Approach (Yufei Xie et al., 2023)

{{<citation>}}

Yufei Xie, Rodolfo C. Raga Jr. (2023)  
**Convolutional Neural Networks for Sentiment Analysis on Weibo Data: A Natural Language Processing Approach**  

---
Primary Category: cs.CL
Categories: I-2-7, cs-CL, cs-LG, cs.CL  
Keywords: AI, BERT, NLP, Natural Language Processing, Sentiment Analysis  
[Paper Link](http://arxiv.org/abs/2307.06540v1)  

---


**ABSTRACT**  
This study addressed the complex task of sentiment analysis on a dataset of 119,988 original tweets from Weibo using a Convolutional Neural Network (CNN), offering a new approach to Natural Language Processing (NLP). The data, sourced from Baidu's PaddlePaddle AI platform, were meticulously preprocessed, tokenized, and categorized based on sentiment labels. A CNN-based model was utilized, leveraging word embeddings for feature extraction, and trained to perform sentiment classification. The model achieved a macro-average F1-score of approximately 0.73 on the test set, showing balanced performance across positive, neutral, and negative sentiments. The findings underscore the effectiveness of CNNs for sentiment analysis tasks, with implications for practical applications in social media analysis, market research, and policy studies. The complete experimental content and code have been made publicly available on the Kaggle data platform for further research and development. Future work may involve exploring different architectures, such as Recurrent Neural Networks (RNN) or transformers, or using more complex pre-trained models like BERT, to further improve the model's ability to understand linguistic nuances and context.

{{</citation>}}


### (62/90) Exploring the Integration of Large Language Models into Automatic Speech Recognition Systems: An Empirical Study (Zeping Min et al., 2023)

{{<citation>}}

Zeping Min, Jinbo Wang. (2023)  
**Exploring the Integration of Large Language Models into Automatic Speech Recognition Systems: An Empirical Study**  

---
Primary Category: cs.CL
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: ChatGPT, GPT, GPT-4, Language Model, NLP, Natural Language Processing, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2307.06530v1)  

---


**ABSTRACT**  
This paper explores the integration of Large Language Models (LLMs) into Automatic Speech Recognition (ASR) systems to improve transcription accuracy. The increasing sophistication of LLMs, with their in-context learning capabilities and instruction-following behavior, has drawn significant attention in the field of Natural Language Processing (NLP). Our primary focus is to investigate the potential of using an LLM's in-context learning capabilities to enhance the performance of ASR systems, which currently face challenges such as ambient noise, speaker accents, and complex linguistic contexts. We designed a study using the Aishell-1 and LibriSpeech datasets, with ChatGPT and GPT-4 serving as benchmarks for LLM capabilities. Unfortunately, our initial experiments did not yield promising results, indicating the complexity of leveraging LLM's in-context learning for ASR applications. Despite further exploration with varied settings and models, the corrected sentences from the LLMs frequently resulted in higher Word Error Rates (WER), demonstrating the limitations of LLMs in speech applications. This paper provides a detailed overview of these experiments, their results, and implications, establishing that using LLMs' in-context learning capabilities to correct potential errors in speech recognition transcriptions is still a challenging task at the current stage.

{{</citation>}}


### (63/90) Agreement Tracking for Multi-Issue Negotiation Dialogues (Amogh Mannekote et al., 2023)

{{<citation>}}

Amogh Mannekote, Bonnie J. Dorr, Kristy Elizabeth Boyer. (2023)  
**Agreement Tracking for Multi-Issue Negotiation Dialogues**  

---
Primary Category: cs.CL
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue, GPT, T5  
[Paper Link](http://arxiv.org/abs/2307.06524v1)  

---


**ABSTRACT**  
Automated negotiation support systems aim to help human negotiators reach more favorable outcomes in multi-issue negotiations (e.g., an employer and a candidate negotiating over issues such as salary, hours, and promotions before a job offer). To be successful, these systems must accurately track agreements reached by participants in real-time. Existing approaches either focus on task-oriented dialogues or produce unstructured outputs, rendering them unsuitable for this objective. Our work introduces the novel task of agreement tracking for two-party multi-issue negotiations, which requires continuous monitoring of agreements within a structured state space. To address the scarcity of annotated corpora with realistic multi-issue negotiation dialogues, we use GPT-3 to build GPT-Negochat, a synthesized dataset that we make publicly available. We present a strong initial baseline for our task by transfer-learning a T5 model trained on the MultiWOZ 2.4 corpus. Pre-training T5-small and T5-base on MultiWOZ 2.4's DST task enhances results by 21% and 9% respectively over training solely on GPT-Negochat. We validate our method's sample-efficiency via smaller training subset experiments. By releasing GPT-Negochat and our baseline models, we aim to encourage further research in multi-issue negotiation dialogue agreement tracking.

{{</citation>}}


### (64/90) National Origin Discrimination in Deep-learning-powered Automated Resume Screening (Sihang Li et al., 2023)

{{<citation>}}

Sihang Li, Kuangzheng Li, Haibing Lu. (2023)  
**National Origin Discrimination in Deep-learning-powered Automated Resume Screening**  

---
Primary Category: cs.CL
Categories: cs-CL, cs.CL  
Keywords: AI, Google  
[Paper Link](http://arxiv.org/abs/2307.08624v1)  

---


**ABSTRACT**  
Many companies and organizations have started to use some form of AIenabled auto mated tools to assist in their hiring process, e.g. screening resumes, interviewing candi dates, performance evaluation. While those AI tools have greatly improved human re source operations efficiency and provided conveniences to job seekers as well, there are increasing concerns on unfair treatment to candidates, caused by underlying bias in AI systems. Laws around equal opportunity and fairness, like GDPR, CCPA, are introduced or under development, in attempt to regulate AI. However, it is difficult to implement AI regulations in practice, as technologies are constantly advancing and the risk perti nent to their applications can fail to be recognized. This study examined deep learning methods, a recent technology breakthrough, with focus on their application to automated resume screening. One impressive performance of deep learning methods is the represen tation of individual words as lowdimensional numerical vectors, called word embedding, which are learned from aggregated global wordword cooccurrence statistics from a cor pus, like Wikipedia or Google news. The resulting word representations possess interest ing linear substructures of the word vector space and have been widely used in down stream tasks, like resume screening. However, word embedding inherits and reinforces the stereotyping from the training corpus, as deep learning models essentially learn a probability distribution of words and their relations from history data. Our study finds out that if we rely on such deeplearningpowered automated resume screening tools, it may lead to decisions favoring or disfavoring certain demographic groups and raise eth ical, even legal, concerns. To address the issue, we developed bias mitigation method. Extensive experiments on real candidate resumes are conducted to validate our study

{{</citation>}}


### (65/90) AutoHint: Automatic Prompt Optimization with Hint Generation (Hong Sun et al., 2023)

{{<citation>}}

Hong Sun, Xue Li, Yinchuan Xu, Youkow Homma, Qi Cao, Min Wu, Jian Jiao, Denis Charles. (2023)  
**AutoHint: Automatic Prompt Optimization with Hint Generation**  

---
Primary Category: cs.CL
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.07415v1)  

---


**ABSTRACT**  
This paper presents AutoHint, a novel framework for automatic prompt engineering and optimization for Large Language Models (LLM). While LLMs have demonstrated remarkable ability in achieving high-quality annotation in various tasks, the key to applying this ability to specific tasks lies in developing high-quality prompts. Thus we propose a framework to inherit the merits of both in-context learning and zero-shot learning by incorporating enriched instructions derived from input-output demonstrations to optimize original prompt. We refer to the enrichment as the hint and propose a framework to automatically generate the hint from labeled data. More concretely, starting from an initial prompt, our method first instructs a LLM to deduce new hints for selected samples from incorrect predictions, and then summarizes from per-sample hints and adds the results back to the initial prompt to form a new, enriched instruction. The proposed method is evaluated on the BIG-Bench Instruction Induction dataset for both zero-shot and few-short prompts, where experiments demonstrate our method is able to significantly boost accuracy for multiple tasks.

{{</citation>}}


## eess.SP (2)



### (66/90) Corticomorphic Hybrid CNN-SNN Architecture for EEG-based Low-footprint Low-latency Auditory Attention Detection (Richard Gall et al., 2023)

{{<citation>}}

Richard Gall, Deniz Kocanaogullari, Murat Akcakaya, Deniz Erdogmus, Rajkumar Kubendran. (2023)  
**Corticomorphic Hybrid CNN-SNN Architecture for EEG-based Low-footprint Low-latency Auditory Attention Detection**  

---
Primary Category: eess.SP
Categories: cs-LG, cs-SD, eess-AS, eess-SP, eess.SP  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2307.08501v1)  

---


**ABSTRACT**  
In a multi-speaker "cocktail party" scenario, a listener can selectively attend to a speaker of interest. Studies into the human auditory attention network demonstrate cortical entrainment to speech envelopes resulting in highly correlated Electroencephalography (EEG) measurements. Current trends in EEG-based auditory attention detection (AAD) using artificial neural networks (ANN) are not practical for edge-computing platforms due to longer decision windows using several EEG channels, with higher power consumption and larger memory footprint requirements. Nor are ANNs capable of accurately modeling the brain's top-down attention network since the cortical organization is complex and layer. In this paper, we propose a hybrid convolutional neural network-spiking neural network (CNN-SNN) corticomorphic architecture, inspired by the auditory cortex, which uses EEG data along with multi-speaker speech envelopes to successfully decode auditory attention with low latency down to 1 second, using only 8 EEG electrodes strategically placed close to the auditory cortex, at a significantly higher accuracy of 91.03%, compared to the state-of-the-art. Simultaneously, when compared to a traditional CNN reference model, our model uses ~15% fewer parameters at a lower bit precision resulting in ~57% memory footprint reduction. The results show great promise for edge-computing in brain-embedded devices, like smart hearing aids.

{{</citation>}}


### (67/90) Defeating Proactive Jammers Using Deep Reinforcement Learning for Resource-Constrained IoT Networks (Abubakar Sani Ali et al., 2023)

{{<citation>}}

Abubakar Sani Ali, Shimaa Naser, Sami Muhaidat. (2023)  
**Defeating Proactive Jammers Using Deep Reinforcement Learning for Resource-Constrained IoT Networks**  

---
Primary Category: eess.SP
Categories: cs-SY, eess-SP, eess-SY, eess.SP  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.06796v1)  

---


**ABSTRACT**  
Traditional anti-jamming techniques like spread spectrum, adaptive power/rate control, and cognitive radio, have demonstrated effectiveness in mitigating jamming attacks. However, their robustness against the growing complexity of internet-of-thing (IoT) networks and diverse jamming attacks is still limited. To address these challenges, machine learning (ML)-based techniques have emerged as promising solutions. By offering adaptive and intelligent anti-jamming capabilities, ML-based approaches can effectively adapt to dynamic attack scenarios and overcome the limitations of traditional methods. In this paper, we propose a deep reinforcement learning (DRL)-based approach that utilizes state input from realistic wireless network interface cards. We train five different variants of deep Q-network (DQN) agents to mitigate the effects of jamming with the aim of identifying the most sample-efficient, lightweight, robust, and least complex agent that is tailored for power-constrained devices. The simulation results demonstrate the effectiveness of the proposed DRL-based anti-jamming approach against proactive jammers, regardless of their jamming strategy which eliminates the need for a pattern recognition or jamming strategy detection step. Our findings present a promising solution for securing IoT networks against jamming attacks and highlights substantial opportunities for continued investigation and advancement within this field.

{{</citation>}}


## cs.AI (9)



### (68/90) On the Connection between Game-Theoretic Feature Attributions and Counterfactual Explanations (Emanuele Albini et al., 2023)

{{<citation>}}

Emanuele Albini, Shubham Sharma, Saumitra Mishra, Danial Dervovic, Daniele Magazzeni. (2023)  
**On the Connection between Game-Theoretic Feature Attributions and Counterfactual Explanations**  

---
Primary Category: cs.AI
Categories: I-2; I-5; H-5; F-2, cs-AI, cs-CV, cs-GT, cs-HC, cs-LG, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.06941v1)  

---


**ABSTRACT**  
Explainable Artificial Intelligence (XAI) has received widespread interest in recent years, and two of the most popular types of explanations are feature attributions, and counterfactual explanations. These classes of approaches have been largely studied independently and the few attempts at reconciling them have been primarily empirical. This work establishes a clear theoretical connection between game-theoretic feature attributions, focusing on but not limited to SHAP, and counterfactuals explanations. After motivating operative changes to Shapley values based feature attributions and counterfactual explanations, we prove that, under conditions, they are in fact equivalent. We then extend the equivalency result to game-theoretic solution concepts beyond Shapley values. Moreover, through the analysis of the conditions of such equivalence, we shed light on the limitations of naively using counterfactual explanations to provide feature importances. Experiments on three datasets quantitatively show the difference in explanations at every stage of the connection between the two approaches and corroborate the theoretical findings.

{{</citation>}}


### (69/90) LLM-assisted Knowledge Graph Engineering: Experiments with ChatGPT (Lars-Peter Meyer et al., 2023)

{{<citation>}}

Lars-Peter Meyer, Claus Stadler, Johannes Frey, Norman Radtke, Kurt Junghanns, Roy Meissner, Gordian Dziwis, Kirill Bulert, Michael Martin. (2023)  
**LLM-assisted Knowledge Graph Engineering: Experiments with ChatGPT**  

---
Primary Category: cs.AI
Categories: cs-AI, cs-CL, cs-DB, cs.AI  
Keywords: ChatGPT, GPT, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2307.06917v1)  

---


**ABSTRACT**  
Knowledge Graphs (KG) provide us with a structured, flexible, transparent, cross-system, and collaborative way of organizing our knowledge and data across various domains in society and industrial as well as scientific disciplines. KGs surpass any other form of representation in terms of effectiveness. However, Knowledge Graph Engineering (KGE) requires in-depth experiences of graph structures, web technologies, existing models and vocabularies, rule sets, logic, as well as best practices. It also demands a significant amount of work. Considering the advancements in large language models (LLMs) and their interfaces and applications in recent years, we have conducted comprehensive experiments with ChatGPT to explore its potential in supporting KGE. In this paper, we present a selection of these experiments and their results to demonstrate how ChatGPT can assist us in the development and management of KGs.

{{</citation>}}


### (70/90) IntelliGraphs: Datasets for Benchmarking Knowledge Graph Generation (Thiviyan Thanapalasingam et al., 2023)

{{<citation>}}

Thiviyan Thanapalasingam, Emile van Krieken, Peter Bloem, Paul Groth. (2023)  
**IntelliGraphs: Datasets for Benchmarking Knowledge Graph Generation**  

---
Primary Category: cs.AI
Categories: cs-AI, cs-LG, cs.AI  
Keywords: Embedding, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2307.06698v2)  

---


**ABSTRACT**  
Knowledge Graph Embedding (KGE) models are used to learn continuous representations of entities and relations. A key task in the literature is predicting missing links between entities. However, Knowledge Graphs are not just sets of links but also have semantics underlying their structure. Semantics is crucial in several downstream tasks, such as query answering or reasoning. We introduce the subgraph inference task, where a model has to generate likely and semantically valid subgraphs. We propose IntelliGraphs, a set of five new Knowledge Graph datasets. The IntelliGraphs datasets contain subgraphs with semantics expressed in logical rules for evaluating subgraph inference. We also present the dataset generator that produced the synthetic datasets. We designed four novel baseline models, which include three models based on traditional KGEs. We evaluate their expressiveness and show that these models cannot capture the semantics. We believe this benchmark will encourage the development of machine learning models that emphasize semantic understanding.

{{</citation>}}


### (71/90) Reinforcement Learning for Syntax-Guided Synthesis (Julian Parsert et al., 2023)

{{<citation>}}

Julian Parsert, Elizabeth Polgreen. (2023)  
**Reinforcement Learning for Syntax-Guided Synthesis**  

---
Primary Category: cs.AI
Categories: cs-AI, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.09564v1)  

---


**ABSTRACT**  
Program synthesis is the task of automatically generating code based on a specification. In Syntax-Guided Synthesis(SyGuS) this specification is a combination of a syntactic template and a logical formula, and any generated code is proven to satisfy both. Techniques like SyGuS are critical to guaranteeing correct synthesis results. Despite the proliferation of machine learning in other types of program synthesis, state-of-the-art techniques in SyGuS are still driven by automated reasoning tools and simple enumeration. We hypothesize this is for two reasons: first the complexity of the search problem, and second the relatively small data sets available. In this work, we tackle these challenges by framing general SyGuS problems as a tree-search, and present a reinforcement learning guided synthesis algorithm for SyGuS based on Monte-Carlo Tree Search (MCTS). Our algorithm incorporates learned policy and value functions combined with the upper confidence bound for trees to balance exploration and exploitation. We incorporate this search procedure in a reinforcement learning setup in order to iteratively improve our policy and value estimators which are based on boosted tree models. To address the scarcity of training data, we present a method for automatically generating training data for SyGuS based on \emph{anti-unification} of existing first-order satisfiability problems, which we use to train our MCTS policy. We implement and evaluate this setup and demonstrate that learned policy and value improve the synthesis performance over a baseline enumerator by over $26$ percentage points in the training and testing sets. With these results our tool outperforms state-of-the-art-tools such as CVC5 on the training set and performs comparably on the testing set. We make our data set publicly available, enabling further application of machine learning methods to the SyGuS problem.

{{</citation>}}


### (72/90) Is Task-Agnostic Explainable AI a Myth? (Alicja Chaszczewicz, 2023)

{{<citation>}}

Alicja Chaszczewicz. (2023)  
**Is Task-Agnostic Explainable AI a Myth?**  

---
Primary Category: cs.AI
Categories: cs-AI, cs-LG, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.06963v1)  

---


**ABSTRACT**  
Our work serves as a framework for unifying the challenges of contemporary explainable AI (XAI). We demonstrate that while XAI methods provide supplementary and potentially useful output for machine learning models, researchers and decision-makers should be mindful of their conceptual and technical limitations, which frequently result in these methods themselves becoming black boxes. We examine three XAI research avenues spanning image, textual, and graph data, covering saliency, attention, and graph-type explainers. Despite the varying contexts and timeframes of the mentioned cases, the same persistent roadblocks emerge, highlighting the need for a conceptual breakthrough in the field to address the challenge of compatibility between XAI methods and application tasks.

{{</citation>}}


### (73/90) Prescriptive Process Monitoring Under Resource Constraints: A Reinforcement Learning Approach (Mahmoud Shoush et al., 2023)

{{<citation>}}

Mahmoud Shoush, Marlon Dumas. (2023)  
**Prescriptive Process Monitoring Under Resource Constraints: A Reinforcement Learning Approach**  

---
Primary Category: cs.AI
Categories: cs-AI, cs-LG, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.06564v1)  

---


**ABSTRACT**  
Prescriptive process monitoring methods seek to optimize the performance of business processes by triggering interventions at runtime, thereby increasing the probability of positive case outcomes. These interventions are triggered according to an intervention policy. Reinforcement learning has been put forward as an approach to learning intervention policies through trial and error. Existing approaches in this space assume that the number of resources available to perform interventions in a process is unlimited, an unrealistic assumption in practice. This paper argues that, in the presence of resource constraints, a key dilemma in the field of prescriptive process monitoring is to trigger interventions based not only on predictions of their necessity, timeliness, or effect but also on the uncertainty of these predictions and the level of resource utilization. Indeed, committing scarce resources to an intervention when the necessity or effects of this intervention are highly uncertain may intuitively lead to suboptimal intervention effects. Accordingly, the paper proposes a reinforcement learning approach for prescriptive process monitoring that leverages conformal prediction techniques to consider the uncertainty of the predictions upon which an intervention decision is based. An evaluation using real-life datasets demonstrates that explicitly modeling uncertainty using conformal predictions helps reinforcement learning agents converge towards policies with higher net intervention gain

{{</citation>}}


### (74/90) Artificial Intelligence for Drug Discovery: Are We There Yet? (Catrin Hasselgren et al., 2023)

{{<citation>}}

Catrin Hasselgren, Tudor I. Oprea. (2023)  
**Artificial Intelligence for Drug Discovery: Are We There Yet?**  

---
Primary Category: cs.AI
Categories: cs-AI, cs-LG, cs.AI, q-bio-QM  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.06521v1)  

---


**ABSTRACT**  
Drug discovery is adapting to novel technologies such as data science, informatics, and artificial intelligence (AI) to accelerate effective treatment development while reducing costs and animal experiments. AI is transforming drug discovery, as indicated by increasing interest from investors, industrial and academic scientists, and legislators. Successful drug discovery requires optimizing properties related to pharmacodynamics, pharmacokinetics, and clinical outcomes. This review discusses the use of AI in the three pillars of drug discovery: diseases, targets, and therapeutic modalities, with a focus on small molecule drugs. AI technologies, such as generative chemistry, machine learning, and multi-property optimization, have enabled several compounds to enter clinical trials. The scientific community must carefully vet known information to address the reproducibility crisis. The full potential of AI in drug discovery can only be realized with sufficient ground truth and appropriate human intervention at later pipeline stages.

{{</citation>}}


### (75/90) Leveraging Contextual Counterfactuals Toward Belief Calibration (Qiuyi et al., 2023)

{{<citation>}}

Qiuyi, Zhang, Michael S. Lee, Sherol Chen. (2023)  
**Leveraging Contextual Counterfactuals Toward Belief Calibration**  

---
Primary Category: cs.AI
Categories: cs-AI, cs-LG, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.06513v1)  

---


**ABSTRACT**  
Beliefs and values are increasingly being incorporated into our AI systems through alignment processes, such as carefully curating data collection principles or regularizing the loss function used for training. However, the meta-alignment problem is that these human beliefs are diverse and not aligned across populations; furthermore, the implicit strength of each belief may not be well calibrated even among humans, especially when trying to generalize across contexts. Specifically, in high regret situations, we observe that contextual counterfactuals and recourse costs are particularly important in updating a decision maker's beliefs and the strengths to which such beliefs are held. Therefore, we argue that including counterfactuals is key to an accurate calibration of beliefs during alignment. To do this, we first segment belief diversity into two categories: subjectivity (across individuals within a population) and epistemic uncertainty (within an individual across different contexts). By leveraging our notion of epistemic uncertainty, we introduce `the belief calibration cycle' framework to more holistically calibrate this diversity of beliefs with context-driven counterfactual reasoning by using a multi-objective optimization. We empirically apply our framework for finding a Pareto frontier of clustered optimal belief strengths that generalize across different contexts, demonstrating its efficacy on a toy dataset for credit decisions.

{{</citation>}}


### (76/90) Hybrid Control Policy for Artificial Pancreas via Ensemble Deep Reinforcement Learning (Wenzhou Lv et al., 2023)

{{<citation>}}

Wenzhou Lv, Tianyu Wu, Luolin Xiong, Liang Wu, Jian Zhou, Yang Tang, Feng Qian. (2023)  
**Hybrid Control Policy for Artificial Pancreas via Ensemble Deep Reinforcement Learning**  

---
Primary Category: cs.AI
Categories: cs-AI, cs-LG, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.06501v2)  

---


**ABSTRACT**  
Objective: The artificial pancreas (AP) has shown promising potential in achieving closed-loop glucose control for individuals with type 1 diabetes mellitus (T1DM). However, designing an effective control policy for the AP remains challenging due to the complex physiological processes, delayed insulin response, and inaccurate glucose measurements. While model predictive control (MPC) offers safety and stability through the dynamic model and safety constraints, it lacks individualization and is adversely affected by unannounced meals. Conversely, deep reinforcement learning (DRL) provides personalized and adaptive strategies but faces challenges with distribution shifts and substantial data requirements. Methods: We propose a hybrid control policy for the artificial pancreas (HyCPAP) to address the above challenges. HyCPAP combines an MPC policy with an ensemble DRL policy, leveraging the strengths of both policies while compensating for their respective limitations. To facilitate faster deployment of AP systems in real-world settings, we further incorporate meta-learning techniques into HyCPAP, leveraging previous experience and patient-shared knowledge to enable fast adaptation to new patients with limited available data. Results: We conduct extensive experiments using the FDA-accepted UVA/Padova T1DM simulator across three scenarios. Our approaches achieve the highest percentage of time spent in the desired euglycemic range and the lowest occurrences of hypoglycemia. Conclusion: The results clearly demonstrate the superiority of our methods for closed-loop glucose management in individuals with T1DM. Significance: The study presents novel control policies for AP systems, affirming the great potential of proposed methods for efficient closed-loop glucose control.

{{</citation>}}


## cs.DB (1)



### (77/90) Towards a Rosetta Stone for (meta)data: Learning from natural language to improve semantic and cognitive interoperability (Lars Vogt et al., 2023)

{{<citation>}}

Lars Vogt, Marcel Konrad, Manuel Prinz. (2023)  
**Towards a Rosetta Stone for (meta)data: Learning from natural language to improve semantic and cognitive interoperability**  

---
Primary Category: cs.DB
Categories: cs-DB, cs.DB  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.09605v1)  

---


**ABSTRACT**  
In order to effectively manage the overwhelming influx of data, it is crucial to ensure that data is findable, accessible, interoperable, and reusable (FAIR). While ontologies and knowledge graphs have been employed to enhance FAIRness, challenges remain regarding semantic and cognitive interoperability. We explore how English facilitates reliable communication of terms and statements, and transfer our findings to a framework of ontologies and knowledge graphs, while treating terms and statements as minimal information units. We categorize statement types based on their predicates, recognizing the limitations of modeling non-binary predicates with multiple triples, which negatively impacts interoperability. Terms are associated with different frames of reference, and different operations require different schemata. Term mappings and schema crosswalks are therefore vital for semantic interoperability. We propose a machine-actionable Rosetta Stone Framework for (meta)data, which uses reference terms and schemata as an interlingua to minimize mappings and crosswalks. Modeling statements rather than a human-independent reality ensures cognitive familiarity and thus better interoperability of data structures. We extend this Rosetta modeling paradigm to reference schemata, resulting in simple schemata with a consistent structure across statement types, empowering domain experts to create their own schemata using the Rosetta Editor, without requiring knowledge of semantics. The Editor also allows specifying textual and graphical display templates for each schema, delivering human-readable data representations alongside machine-actionable data structures. The Rosetta Query Builder derives queries based on completed input forms and the information from corresponding reference schemata. This work sets the conceptual ground for the Rosetta Stone Framework that we plan to develop in the future.

{{</citation>}}


## cs.SC (1)



### (78/90) Data Augmentation for Mathematical Objects (Tereso del Rio et al., 2023)

{{<citation>}}

Tereso del Rio, Matthew England. (2023)  
**Data Augmentation for Mathematical Objects**  

---
Primary Category: cs.SC
Categories: 68W30, 68T05, 03C10, I-2-6; I-1-1, cs-LG, cs-SC, cs.SC  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2307.06984v1)  

---


**ABSTRACT**  
This paper discusses and evaluates ideas of data balancing and data augmentation in the context of mathematical objects: an important topic for both the symbolic computation and satisfiability checking communities, when they are making use of machine learning techniques to optimise their tools. We consider a dataset of non-linear polynomial problems and the problem of selecting a variable ordering for cylindrical algebraic decomposition to tackle these with. By swapping the variable names in already labelled problems, we generate new problem instances that do not require any further labelling when viewing the selection as a classification problem. We find this augmentation increases the accuracy of ML models by 63% on average. We study what part of this improvement is due to the balancing of the dataset and what is achieved thanks to further increasing the size of the dataset, concluding that both have a very significant effect. We finish the paper by reflecting on how this idea could be applied in other uses of machine learning in mathematics.

{{</citation>}}


## eess.AS (1)



### (79/90) Personalization for BERT-based Discriminative Speech Recognition Rescoring (Jari Kolehmainen et al., 2023)

{{<citation>}}

Jari Kolehmainen, Yile Gu, Aditya Gourav, Prashanth Gurunath Shivakumar, Ankur Gandhe, Ariya Rastrow, Ivan Bulyko. (2023)  
**Personalization for BERT-based Discriminative Speech Recognition Rescoring**  

---
Primary Category: eess.AS
Categories: cs-CL, eess-AS, eess.AS  
Keywords: BERT, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2307.06832v1)  

---


**ABSTRACT**  
Recognition of personalized content remains a challenge in end-to-end speech recognition. We explore three novel approaches that use personalized content in a neural rescoring step to improve recognition: gazetteers, prompting, and a cross-attention based encoder-decoder model. We use internal de-identified en-US data from interactions with a virtual voice assistant supplemented with personalized named entities to compare these approaches. On a test set with personalized named entities, we show that each of these approaches improves word error rate by over 10%, against a neural rescoring baseline. We also show that on this test set, natural language prompts can improve word error rate by 7% without any training and with a marginal loss in generalization. Overall, gazetteers were found to perform the best with a 10% improvement in word error rate (WER), while also improving WER on a general test set by 1%.

{{</citation>}}


## cs.SI (4)



### (80/90) A Data-driven Understanding of Left-Wing Extremists on Social Media (Utkucan Balcı et al., 2023)

{{<citation>}}

Utkucan Balcı, Michael Sirivianos, Jeremy Blackburn. (2023)  
**A Data-driven Understanding of Left-Wing Extremists on Social Media**  

---
Primary Category: cs.SI
Categories: cs-CY, cs-SI, cs.SI  
Keywords: Social Media  
[Paper Link](http://arxiv.org/abs/2307.06981v1)  

---


**ABSTRACT**  
Social media's role in the spread and evolution of extremism is a focus of intense study. Online extremists have been involved in the spread of online hate, mis/disinformation, and real-world violence. However, the overwhelming majority of existing work has focused on right-wing extremism. In this paper, we perform a first of its kind large-scale, data-driven study exploring left-wing extremism. We focus on "tankies," a left-wing community that first arose in the 1950s in support of hardline actions of the USSR and has evolved to support what they call "actually existing socialist countries," e.g., CCP run China, the USSR, former soviet countries, and North Korea. We collect 1.3M posts from 53K authors from tankies subreddits, and explore the position of tankies within the broader far-left community on Reddit. Among other things, we find that tankies are clearly on the periphery of the larger far-left community. When examining the contents of posts, we find misalignments and conceptual homomorphisms that confirm the description of tankies in the theoretical work. We also discover that tankies focus more on state-level political events rather than social issues in comparison to other far-left communities. Finally, we show that tankies exhibit some of the same worrying behaviors as right-wing extremists, e.g., relatively high toxicity and an organized response to deplatforming events.

{{</citation>}}


### (81/90) Extended Graph Assessment Metrics for Graph Neural Networks (Tamara T. Mueller et al., 2023)

{{<citation>}}

Tamara T. Mueller, Sophie Starck, Leonhard F. Feiner, Kyriaki-Margarita Bintsi, Daniel Rueckert, Georgios Kaissis. (2023)  
**Extended Graph Assessment Metrics for Graph Neural Networks**  

---
Primary Category: cs.SI
Categories: cs-AI, cs-LG, cs-SI, cs.SI  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2307.10112v1)  

---


**ABSTRACT**  
When re-structuring patient cohorts into so-called population graphs, initially independent data points can be incorporated into one interconnected graph structure. This population graph can then be used for medical downstream tasks using graph neural networks (GNNs). The construction of a suitable graph structure is a challenging step in the learning pipeline that can have severe impact on model performance. To this end, different graph assessment metrics have been introduced to evaluate graph structures. However, these metrics are limited to classification tasks and discrete adjacency matrices, only covering a small subset of real-world applications. In this work, we introduce extended graph assessment metrics (GAMs) for regression tasks and continuous adjacency matrices. We focus on two GAMs in specific: \textit{homophily} and \textit{cross-class neighbourhood similarity} (CCNS). We extend the notion of GAMs to more than one hop, define homophily for regression tasks, as well as continuous adjacency matrices, and propose a light-weight CCNS distance for discrete and continuous adjacency matrices. We show the correlation of these metrics with model performance on different medical population graphs and under different learning settings.

{{</citation>}}


### (82/90) Unpacking polarization: Antagonism and Alignment in Signed Networks of Online Interaction (Emma Fraxanet et al., 2023)

{{<citation>}}

Emma Fraxanet, Max Pellert, Simon Schweighofer, Vicenç Gómez, David Garcia. (2023)  
**Unpacking polarization: Antagonism and Alignment in Signed Networks of Online Interaction**  

---
Primary Category: cs.SI
Categories: cs-CY, cs-SI, cs.SI  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2307.06571v1)  

---


**ABSTRACT**  
Online polarization research currently focuses on studying single-issue opinion distributions or computing distance metrics of interaction network structures. Limited data availability often restricts studies to positive interaction data, which can misrepresent the reality of a discussion. We introduce a novel framework that aims at combining these three aspects, content and interactions, as well as their nature (positive or negative), while challenging the prevailing notion of polarization as an umbrella term for all forms of online conflict or opposing opinions. In our approach, built on the concepts of cleavage structures and structural balance of signed social networks, we factorize polarization into two distinct metrics: Antagonism and Alignment. Antagonism quantifies hostility in online discussions, based on the reactions of users to content. Alignment uses signed structural information encoded in long-term user-user relations on the platform to describe how well user interactions fit the global and/or traditional sides of discussion. We can analyse the change of these metrics through time, localizing both relevant trends but also sudden changes that can be mapped to specific contexts or events. We apply our methods to two distinct platforms: Birdwatch, a US crowd-based fact-checking extension of Twitter, and DerStandard, an Austrian online newspaper with discussion forums. In these two use cases, we find that our framework is capable of describing the global status of the groups of users (identification of cleavages) while also providing relevant findings on specific issues or in specific time frames. Furthermore, we show that our four metrics describe distinct phenomena, emphasizing their independent consideration for unpacking polarization complexities.

{{</citation>}}


### (83/90) Causal Influences over Social Learning Networks (Mert Kayaalp et al., 2023)

{{<citation>}}

Mert Kayaalp, Ali H. Sayed. (2023)  
**Causal Influences over Social Learning Networks**  

---
Primary Category: cs.SI
Categories: cs-LG, cs-MA, cs-SI, cs.SI, eess-SP  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2307.09575v1)  

---


**ABSTRACT**  
This paper investigates causal influences between agents linked by a social graph and interacting over time. In particular, the work examines the dynamics of social learning models and distributed decision-making protocols, and derives expressions that reveal the causal relations between pairs of agents and explain the flow of influence over the network. The results turn out to be dependent on the graph topology and the level of information that each agent has about the inference problem they are trying to solve. Using these conclusions, the paper proposes an algorithm to rank the overall influence between agents to discover highly influential agents. It also provides a method to learn the necessary model parameters from raw observational data. The results and the proposed algorithm are illustrated by considering both synthetic data and real Twitter data.

{{</citation>}}


## eess.SY (1)



### (84/90) Vehicle Dispatching and Routing of On-Demand Intercity Ride-Pooling Services: A Multi-Agent Hierarchical Reinforcement Learning Approach (Jinhua Si et al., 2023)

{{<citation>}}

Jinhua Si, Fang He, Xi Lin, Xindi Tang. (2023)  
**Vehicle Dispatching and Routing of On-Demand Intercity Ride-Pooling Services: A Multi-Agent Hierarchical Reinforcement Learning Approach**  

---
Primary Category: eess.SY
Categories: cs-AI, cs-LG, cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.06742v1)  

---


**ABSTRACT**  
The integrated development of city clusters has given rise to an increasing demand for intercity travel. Intercity ride-pooling service exhibits considerable potential in upgrading traditional intercity bus services by implementing demand-responsive enhancements. Nevertheless, its online operations suffer the inherent complexities due to the coupling of vehicle resource allocation among cities and pooled-ride vehicle routing. To tackle these challenges, this study proposes a two-level framework designed to facilitate online fleet management. Specifically, a novel multi-agent feudal reinforcement learning model is proposed at the upper level of the framework to cooperatively assign idle vehicles to different intercity lines, while the lower level updates the routes of vehicles using an adaptive large neighborhood search heuristic. Numerical studies based on the realistic dataset of Xiamen and its surrounding cities in China show that the proposed framework effectively mitigates the supply and demand imbalances, and achieves significant improvement in both the average daily system profit and order fulfillment ratio.

{{</citation>}}


## cs.SD (1)



### (85/90) Real-time Percussive Technique Recognition and Embedding Learning for the Acoustic Guitar (Andrea Martelloni et al., 2023)

{{<citation>}}

Andrea Martelloni, Andrew P McPherson, Mathieu Barthet. (2023)  
**Real-time Percussive Technique Recognition and Embedding Learning for the Acoustic Guitar**  

---
Primary Category: cs.SD
Categories: cs-AI, cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2307.07426v1)  

---


**ABSTRACT**  
Real-time music information retrieval (RT-MIR) has much potential to augment the capabilities of traditional acoustic instruments. We develop RT-MIR techniques aimed at augmenting percussive fingerstyle, which blends acoustic guitar playing with guitar body percussion. We formulate several design objectives for RT-MIR systems for augmented instrument performance: (i) causal constraint, (ii) perceptually negligible action-to-sound latency, (iii) control intimacy support, (iv) synthesis control support. We present and evaluate real-time guitar body percussion recognition and embedding learning techniques based on convolutional neural networks (CNNs) and CNNs jointly trained with variational autoencoders (VAEs). We introduce a taxonomy of guitar body percussion based on hand part and location. We follow a cross-dataset evaluation approach by collecting three datasets labelled according to the taxonomy. The embedding quality of the models is assessed using KL-Divergence across distributions corresponding to different taxonomic classes. Results indicate that the networks are strong classifiers especially in a simplified 2-class recognition task, and the VAEs yield improved class separation compared to CNNs as evidenced by increased KL-Divergence across distributions. We argue that the VAE embedding quality could support control intimacy and rich interaction when the latent space's parameters are used to control an external synthesis engine. Further design challenges around generalisation to different datasets have been identified.

{{</citation>}}


## cs.IT (1)



### (86/90) Downlink Precoding for Cell-free FBMC/OQAM Systems With Asynchronous Reception (Yuhao Qi et al., 2023)

{{<citation>}}

Yuhao Qi, Jian Dang, Zaichen Zhang, Liang Wu, Yongpeng Wu. (2023)  
**Downlink Precoding for Cell-free FBMC/OQAM Systems With Asynchronous Reception**  

---
Primary Category: cs.IT
Categories: cs-IT, cs.IT, eess-SP, math-IT  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2307.06657v1)  

---


**ABSTRACT**  
In this work, an efficient precoding design scheme is proposed for downlink cell-free distributed massive multiple-input multiple-output (DM-MIMO) filter bank multi-carrier (FBMC) systems with asynchronous reception and highly frequency selectivity. The proposed scheme includes a multiple interpolation structure to eliminate the impact of response difference we recently discovered, which has better performance in highly frequency-selective channels. Besides, we also consider the phase shift in asynchronous reception and introduce a phase compensation in the design process. The phase compensation also benefits from the multiple interpolation structure and better adapts to asynchronous reception. Based on the proposed scheme, we theoretically analyze its ergodic achievable rate performance and derive a closed-form expression. Simulation results show that the derived expression can accurately characterize the rate performance, and FBMC with the proposed scheme outperforms orthogonal frequency-division multiplexing (OFDM) in the asynchronous scenario.

{{</citation>}}


## cs.NI (1)



### (87/90) Multivariate Time Series characterization and forecasting of VoIP traffic in real mobile networks (Mario Di Mauro et al., 2023)

{{<citation>}}

Mario Di Mauro, Giovanni Galatro, Fabio Postiglione, Wei Song, Antonio Liotta. (2023)  
**Multivariate Time Series characterization and forecasting of VoIP traffic in real mobile networks**  

---
Primary Category: cs.NI
Categories: cs-LG, cs-NI, cs.NI, stat-ML  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2307.06645v1)  

---


**ABSTRACT**  
Predicting the behavior of real-time traffic (e.g., VoIP) in mobility scenarios could help the operators to better plan their network infrastructures and to optimize the allocation of resources. Accordingly, in this work the authors propose a forecasting analysis of crucial QoS/QoE descriptors (some of which neglected in the technical literature) of VoIP traffic in a real mobile environment. The problem is formulated in terms of a multivariate time series analysis. Such a formalization allows to discover and model the temporal relationships among various descriptors and to forecast their behaviors for future periods. Techniques such as Vector Autoregressive models and machine learning (deep-based and tree-based) approaches are employed and compared in terms of performance and time complexity, by reframing the multivariate time series problem into a supervised learning one. Moreover, a series of auxiliary analyses (stationarity, orthogonal impulse responses, etc.) are performed to discover the analytical structure of the time series and to provide deep insights about their relationships. The whole theoretical analysis has an experimental counterpart since a set of trials across a real-world LTE-Advanced environment has been performed to collect, post-process and analyze about 600,000 voice packets, organized per flow and differentiated per codec.

{{</citation>}}


## eess.IV (1)



### (88/90) Explainable 2D Vision Models for 3D Medical Data (Alexander Ziller et al., 2023)

{{<citation>}}

Alexander Ziller, Alp Güvenir, Ayhan Can Erdur, Tamara T. Mueller, Philip Müller, Friederike Jungmann, Johannes Brandt, Jan Peeken, Rickmer Braren, Daniel Rueckert, Georgios Kaissis. (2023)  
**Explainable 2D Vision Models for 3D Medical Data**  

---
Primary Category: eess.IV
Categories: cs-CV, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.06614v1)  

---


**ABSTRACT**  
Training Artificial Intelligence (AI) models on three-dimensional image data presents unique challenges compared to the two-dimensional case: Firstly, the computational resources are significantly higher, and secondly, the availability of large pretraining datasets is often limited, impeding training success. In this study, we propose a simple approach of adapting 2D networks with an intermediate feature representation for processing 3D volumes. Our method involves sequentially applying these networks to slices of a 3D volume from all orientations. Subsequently, a feature reduction module combines the extracted slice features into a single representation, which is then used for classification. We evaluate our approach on medical classification benchmarks and a real-world clinical dataset, demonstrating comparable results to existing methods. Furthermore, by employing attention pooling as a feature reduction module we obtain weighted importance values for each slice during the forward pass. We show that slices deemed important by our approach allow the inspection of the basis of a model's prediction.

{{</citation>}}


## cs.CY (2)



### (89/90) On the Mechanics of NFT Valuation: AI Ethics and Social Media (Luyao Zhang et al., 2023)

{{<citation>}}

Luyao Zhang, Yutong Sun, Yutong Quan, Jiaxun Cao, Xin Tong. (2023)  
**On the Mechanics of NFT Valuation: AI Ethics and Social Media**  

---
Primary Category: cs.CY
Categories: cs-CY, cs-DC, cs.CY  
Keywords: AI, Social Media  
[Paper Link](http://arxiv.org/abs/2307.10201v1)  

---


**ABSTRACT**  
As CryptoPunks pioneers the innovation of non-fungible tokens (NFTs) in AI and art, the valuation mechanics of NFTs has become a trending topic. Earlier research identifies the impact of ethics and society on the price prediction of CryptoPunks. Since the booming year of the NFT market in 2021, the discussion of CryptoPunks has propagated on social media. Still, existing literature hasn't considered the social sentiment factors after the historical turning point on NFT valuation. In this paper, we study how sentiments in social media, together with gender and skin tone, contribute to NFT valuations by an empirical analysis of social media, blockchain, and crypto exchange data. We evidence social sentiments as a significant contributor to the price prediction of CryptoPunks. Furthermore, we document structure changes in the valuation mechanics before and after 2021. Although people's attitudes towards Cryptopunks are primarily positive, our findings reflect imbalances in transaction activities and pricing based on gender and skin tone. Our result is consistent and robust, controlling for the rarity of an NFT based on the set of human-readable attributes, including gender and skin tone. Our research contributes to the interdisciplinary study at the intersection of AI, Ethics, and Society, focusing on the ecosystem of decentralized AI or blockchain. We provide our data and code for replicability as open access on GitHub.

{{</citation>}}


### (90/90) Machine Learning practices and infrastructures (Glen Berman, 2023)

{{<citation>}}

Glen Berman. (2023)  
**Machine Learning practices and infrastructures**  

---
Primary Category: cs.CY
Categories: cs-CY, cs-LG, cs.CY  
Keywords: AI, Google  
[Paper Link](http://arxiv.org/abs/2307.06518v1)  

---


**ABSTRACT**  
Machine Learning (ML) systems, particularly when deployed in high-stakes domains, are deeply consequential. They can exacerbate existing inequities, create new modes of discrimination, and reify outdated social constructs. Accordingly, the social context (i.e. organisations, teams, cultures) in which ML systems are developed is a site of active research for the field of AI ethics, and intervention for policymakers. This paper focuses on one aspect of social context that is often overlooked: interactions between practitioners and the tools they rely on, and the role these interactions play in shaping ML practices and the development of ML systems. In particular, through an empirical study of questions asked on the Stack Exchange forums, the use of interactive computing platforms (e.g. Jupyter Notebook and Google Colab) in ML practices is explored. I find that interactive computing platforms are used in a host of learning and coordination practices, which constitutes an infrastructural relationship between interactive computing platforms and ML practitioners. I describe how ML practices are co-evolving alongside the development of interactive computing platforms, and highlight how this risks making invisible aspects of the ML life cycle that AI ethics researchers' have demonstrated to be particularly salient for the societal impact of deployed ML systems.

{{</citation>}}
